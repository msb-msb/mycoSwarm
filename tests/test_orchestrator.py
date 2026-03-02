"""Tests for mycoswarm.orchestrator — task routing and scoring."""

import asyncio
from unittest.mock import AsyncMock, patch

import pytest

from mycoswarm.api import TaskRequest, TaskResult, TaskStatus
from mycoswarm.discovery import Peer, PeerRegistry
from mycoswarm.orchestrator import Orchestrator, RoutingDecision


@pytest.mark.asyncio
async def test_executive_scored_low_for_web_search(make_peer, make_identity):
    """Executive peer scored below light peer for cpu_work (negative 500)."""
    registry = PeerRegistry()
    identity = make_identity()
    orch = Orchestrator(identity, registry)

    executive = make_peer(
        node_id="exec-1", hostname="exec-node",
        node_tier="executive", capabilities=["gpu_inference", "cpu_worker"],
        vram_total_mb=24576,
    )
    light = make_peer(
        node_id="light-1", hostname="light-node",
        node_tier="light", capabilities=["cpu_worker"],
    )

    exec_score = orch.router._score_peer_for_cpu_work(executive)
    light_score = orch.router._score_peer_for_cpu_work(light)

    # Executive gets -500, light gets +50 → light wins
    assert light_score > exec_score
    assert exec_score < 0  # 100 (cpu_worker) - 500 (executive penalty) = -400

    await orch.close()


@pytest.mark.asyncio
async def test_inflight_penalty_reduces_score(make_peer, make_identity):
    """record_dispatch → peer's cpu_work score drops by 100."""
    registry = PeerRegistry()
    identity = make_identity()
    orch = Orchestrator(identity, registry)

    peer = make_peer(
        node_id="busy-1", hostname="busy-node",
        node_tier="light", capabilities=["cpu_worker"],
    )

    score_before = orch.router._score_peer_for_cpu_work(peer)
    orch.record_dispatch("busy-1")
    score_after = orch.router._score_peer_for_cpu_work(peer)

    assert score_after == score_before - 100

    await orch.close()


@pytest.mark.asyncio
async def test_retry_selects_next_candidate(make_task, make_peer, make_identity):
    """dispatch_task retries on dispatch error, succeeds on second peer."""
    registry = PeerRegistry()
    identity = make_identity()
    orch = Orchestrator(identity, registry)

    peer1 = make_peer(
        node_id="peer-a", hostname="peer-a", ip="192.168.1.51",
        capabilities=["cpu_worker"],
    )
    peer2 = make_peer(
        node_id="peer-b", hostname="peer-b", ip="192.168.1.52",
        capabilities=["cpu_worker"],
    )
    await registry.add_or_update(peer1)
    await registry.add_or_update(peer2)

    task = make_task("web_search", {"query": "test"})

    call_count = 0

    async def mock_dispatch(peer, t):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            # First call: dispatch error (peer unreachable)
            return TaskResult(
                task_id=t.task_id, status=TaskStatus.FAILED,
                error="Can't reach peer", node_id=peer.node_id,
            ), True  # is_dispatch_error=True → retryable
        else:
            # Second call: success
            return TaskResult(
                task_id=t.task_id, status=TaskStatus.COMPLETED,
                result={"ok": True}, node_id=peer.node_id,
            ), False

    with patch.object(orch, "_dispatch_to_peer", side_effect=mock_dispatch):
        result = await orch.dispatch_task(task)

    assert call_count == 2
    assert result.status == TaskStatus.COMPLETED

    await orch.close()


@pytest.mark.asyncio
async def test_can_handle_locally_inference_needs_ollama(make_identity):
    """Identity without ollama → can_handle_locally('inference') is False."""
    registry = PeerRegistry()
    identity = make_identity(
        capabilities=["cpu_worker", "cpu_inference"],
        ollama_running=False,
    )
    orch = Orchestrator(identity, registry)

    assert orch.can_handle_locally("inference") is False
    # But cpu tasks should work
    assert orch.can_handle_locally("web_search") is True

    await orch.close()


@pytest.mark.asyncio
async def test_score_peer_for_inference_prefers_gpu(make_peer, make_identity):
    """gpu_inference peer scores > cpu_inference peer."""
    registry = PeerRegistry()
    identity = make_identity()
    orch = Orchestrator(identity, registry)

    gpu_peer = make_peer(
        node_id="gpu-1", hostname="gpu-node",
        node_tier="executive",
        capabilities=["gpu_inference", "cpu_worker"],
        vram_total_mb=24576,
    )
    cpu_peer = make_peer(
        node_id="cpu-1", hostname="cpu-node",
        node_tier="light",
        capabilities=["cpu_inference", "cpu_worker"],
        vram_total_mb=0,
    )

    gpu_score = orch.router._score_peer_for_inference(gpu_peer)
    cpu_score = orch.router._score_peer_for_inference(cpu_peer)

    assert gpu_score > cpu_score

    await orch.close()


# --- RoutingDecision tests ---


@pytest.mark.asyncio
async def test_route_task_distributable_to_peer(make_task, make_peer, make_identity):
    """Distributable task routes to peer when peer scores higher than local."""
    registry = PeerRegistry()
    identity = make_identity(
        node_tier="executive",
        capabilities=["gpu_inference", "cpu_worker"],
    )
    orch = Orchestrator(identity, registry)

    light_peer = make_peer(
        node_id="light-1", hostname="light-node",
        node_tier="light", capabilities=["cpu_worker"],
    )
    await registry.add_or_update(light_peer)

    task = make_task("web_search", {"query": "test"})
    decision = await orch.route_task(task)

    assert isinstance(decision, RoutingDecision)
    assert decision.target is not None
    assert decision.target.node_id == "light-1"
    assert decision.can_execute is True

    await orch.close()


@pytest.mark.asyncio
async def test_route_task_distributable_local_wins(make_task, make_peer, make_identity):
    """Distributable task stays local when local scores higher."""
    registry = PeerRegistry()
    identity = make_identity(
        node_tier="light",
        capabilities=["cpu_worker"],
    )
    orch = Orchestrator(identity, registry)

    # Executive peer penalized for cpu_work (-500)
    exec_peer = make_peer(
        node_id="exec-1", hostname="exec-node",
        node_tier="executive",
        capabilities=["gpu_inference", "cpu_worker"],
        vram_total_mb=24576,
    )
    await registry.add_or_update(exec_peer)

    task = make_task("web_search", {"query": "test"})
    decision = await orch.route_task(task)

    assert decision.target is None
    assert decision.can_execute is True
    assert "Local" in decision.reason

    await orch.close()


@pytest.mark.asyncio
async def test_route_task_inference_to_gpu_peer(make_task, make_peer, make_identity):
    """Inference task routes to GPU peer that has the requested model."""
    registry = PeerRegistry()
    identity = make_identity(
        node_tier="light",
        capabilities=["cpu_worker", "cpu_inference"],
        ollama_running=True,
        available_models=["gemma3:1b"],
    )
    orch = Orchestrator(identity, registry)

    gpu_peer = make_peer(
        node_id="gpu-1", hostname="gpu-node",
        node_tier="executive",
        capabilities=["gpu_inference", "cpu_worker"],
        vram_total_mb=24576,
        available_models=["gemma3:27b", "gemma3:1b"],
    )
    await registry.add_or_update(gpu_peer)

    task = make_task("inference", {"model": "gemma3:27b", "prompt": "hello"})
    decision = await orch.route_task(task)

    assert decision.target is not None
    assert decision.target.node_id == "gpu-1"
    assert "gemma3:27b" in decision.reason

    await orch.close()


@pytest.mark.asyncio
async def test_route_task_inference_specialist_preferred(make_task, make_peer, make_identity):
    """Specialist peer preferred over executive for inference (scoring flip)."""
    registry = PeerRegistry()
    identity = make_identity(
        node_tier="executive",
        capabilities=["gpu_inference", "cpu_worker"],
        vram_total_mb=24576,
        ollama_running=True,
        available_models=["gemma3:27b"],
    )
    orch = Orchestrator(identity, registry)

    # Specialist peer: less VRAM but higher tier bonus (+500 vs +200)
    # specialist: 1000 (gpu) + 122.88 (12288/100) + 500 (specialist) = 1622.88
    # local exec: 1000 (gpu) + 245.76 (24576/100) + 200 (executive) = 1445.76
    smaller_peer = make_peer(
        node_id="gpu-2", hostname="small-gpu",
        node_tier="specialist",
        capabilities=["gpu_inference", "cpu_worker"],
        vram_total_mb=12288,
        available_models=["gemma3:27b"],
    )
    await registry.add_or_update(smaller_peer)

    task = make_task("inference", {"model": "gemma3:27b", "prompt": "hello"})
    decision = await orch.route_task(task)

    # Specialist wins due to scoring flip
    assert decision.target is not None
    assert decision.target.node_id == "gpu-2"

    await orch.close()


@pytest.mark.asyncio
async def test_route_task_no_peer_has_model_local_fallback(
    make_task, make_peer, make_identity
):
    """When no peer has the requested model, fall back to local."""
    registry = PeerRegistry()
    identity = make_identity(
        node_tier="executive",
        capabilities=["gpu_inference", "cpu_worker"],
        ollama_running=True,
        available_models=["gemma3:27b"],
        vram_total_mb=24576,
    )
    orch = Orchestrator(identity, registry)

    peer = make_peer(
        node_id="peer-1", hostname="peer-node",
        node_tier="specialist",
        capabilities=["gpu_inference", "cpu_worker"],
        vram_total_mb=12288,
        available_models=["llama3:8b"],  # Different model
    )
    await registry.add_or_update(peer)

    task = make_task("inference", {"model": "gemma3:27b", "prompt": "hello"})
    decision = await orch.route_task(task)

    assert decision.target is None
    assert decision.can_execute is True
    assert "No peer has model" in decision.reason

    await orch.close()


@pytest.mark.asyncio
async def test_route_task_no_node_can_handle(make_task, make_identity):
    """When no peer and can't handle locally → can_execute=False."""
    registry = PeerRegistry()
    identity = make_identity(
        capabilities=["cpu_worker"],
        ollama_running=False,  # Can't do inference
    )
    orch = Orchestrator(identity, registry)

    task = make_task("inference", {"model": "gemma3:27b", "prompt": "hello"})
    decision = await orch.route_task(task)

    assert decision.target is None
    assert decision.can_execute is False

    await orch.close()


@pytest.mark.asyncio
async def test_route_task_cant_handle_locally_routes_to_peer(
    make_task, make_peer, make_identity
):
    """When local can't handle but peer can → routes to peer."""
    registry = PeerRegistry()
    identity = make_identity(
        capabilities=["cpu_worker"],
        ollama_running=False,
    )
    orch = Orchestrator(identity, registry)

    gpu_peer = make_peer(
        node_id="gpu-1", hostname="gpu-node",
        node_tier="executive",
        capabilities=["gpu_inference", "cpu_worker"],
        vram_total_mb=24576,
        available_models=["gemma3:27b"],
    )
    await registry.add_or_update(gpu_peer)

    task = make_task("inference", {"model": "gemma3:27b", "prompt": "hello"})
    decision = await orch.route_task(task)

    assert decision.target is not None
    assert decision.target.node_id == "gpu-1"
    assert decision.can_execute is True

    await orch.close()


@pytest.mark.asyncio
async def test_local_inference_score_gpu_node(make_identity):
    """Local GPU node gets high inference score."""
    registry = PeerRegistry()
    identity = make_identity(
        node_tier="executive",
        capabilities=["gpu_inference", "cpu_worker"],
        vram_total_mb=24576,
        ollama_running=True,
    )
    orch = Orchestrator(identity, registry)

    score = orch.router._local_inference_score()
    # 1000 (gpu) + 245.76 (vram) + 200 (executive) = ~1445
    assert score > 1400

    await orch.close()


@pytest.mark.asyncio
async def test_local_inference_score_no_ollama(make_identity):
    """Local node without Ollama gets very low inference score."""
    registry = PeerRegistry()
    identity = make_identity(
        node_tier="executive",
        capabilities=["gpu_inference", "cpu_worker"],
        vram_total_mb=24576,
        ollama_running=False,
    )
    orch = Orchestrator(identity, registry)

    score = orch.router._local_inference_score()
    assert score < 0  # -5000 penalty

    await orch.close()


@pytest.mark.asyncio
async def test_local_inference_score_inflight_penalty(make_identity):
    """Inflight tasks reduce local inference score."""
    registry = PeerRegistry()
    identity = make_identity(
        node_id="myco-local",
        node_tier="executive",
        capabilities=["gpu_inference", "cpu_worker"],
        vram_total_mb=24576,
        ollama_running=True,
    )
    orch = Orchestrator(identity, registry)

    score_idle = orch.router._local_inference_score()
    orch.record_dispatch("myco-local")
    orch.record_dispatch("myco-local")
    score_busy = orch.router._local_inference_score()

    assert score_busy == score_idle - 200  # 2 inflight × 100

    await orch.close()


# --- GPU role specialization (Phase 37a) ---


@pytest.mark.asyncio
async def test_support_score_prefers_specialist_over_executive(
    make_peer, make_identity
):
    """Specialist peer scores much higher than executive for support tasks."""
    registry = PeerRegistry()
    identity = make_identity()
    orch = Orchestrator(identity, registry)

    specialist = make_peer(
        node_id="spec-1", hostname="rushuna",
        node_tier="specialist",
        capabilities=["gpu_inference", "cpu_worker"],
        vram_total_mb=12288,
    )
    executive = make_peer(
        node_id="exec-1", hostname="miu",
        node_tier="executive",
        capabilities=["gpu_inference", "cpu_worker"],
        vram_total_mb=24576,
    )

    spec_score = orch.router._score_peer_for_support(specialist)
    exec_score = orch.router._score_peer_for_support(executive)

    # specialist: 1000 + 500 = 1500
    # executive: -2000 + 500 = -1500
    assert spec_score > exec_score
    assert spec_score > 0
    assert exec_score < 0

    await orch.close()


@pytest.mark.asyncio
async def test_support_score_prefers_light_over_executive(
    make_peer, make_identity
):
    """Even a light CPU node scores higher than executive for support."""
    registry = PeerRegistry()
    identity = make_identity()
    orch = Orchestrator(identity, registry)

    light = make_peer(
        node_id="light-1", hostname="naru",
        node_tier="light",
        capabilities=["cpu_worker"],
    )
    executive = make_peer(
        node_id="exec-1", hostname="miu",
        node_tier="executive",
        capabilities=["gpu_inference", "cpu_worker"],
        vram_total_mb=24576,
    )

    light_score = orch.router._score_peer_for_support(light)
    exec_score = orch.router._score_peer_for_support(executive)

    # light: 200, executive: -2000 + 500 = -1500
    assert light_score > exec_score

    await orch.close()


@pytest.mark.asyncio
async def test_embedding_routes_to_specialist_not_executive(
    make_task, make_peer, make_identity
):
    """Embedding task on executive node routes to specialist peer."""
    registry = PeerRegistry()
    identity = make_identity(
        node_tier="executive",
        capabilities=["gpu_inference", "cpu_worker"],
        vram_total_mb=24576,
        ollama_running=True,
        available_models=["gemma3:27b", "nomic-embed-text"],
    )
    orch = Orchestrator(identity, registry)

    specialist = make_peer(
        node_id="spec-1", hostname="rushuna",
        node_tier="specialist",
        capabilities=["gpu_inference", "cpu_worker"],
        vram_total_mb=12288,
        available_models=["gemma3:1b", "nomic-embed-text"],
    )
    await registry.add_or_update(specialist)

    task = make_task("embedding", {"model": "nomic-embed-text", "text": "hello"})
    decision = await orch.route_task(task)

    assert decision.target is not None
    assert decision.target.node_id == "spec-1"
    assert "protecting executive VRAM" in decision.reason

    await orch.close()


@pytest.mark.asyncio
async def test_embedding_falls_back_to_local_when_no_peers(
    make_task, make_identity
):
    """Embedding stays local when no peers have the model (last resort)."""
    registry = PeerRegistry()
    identity = make_identity(
        node_tier="executive",
        capabilities=["gpu_inference", "cpu_worker"],
        vram_total_mb=24576,
        ollama_running=True,
        available_models=["gemma3:27b", "nomic-embed-text"],
    )
    orch = Orchestrator(identity, registry)

    task = make_task("embedding", {"model": "nomic-embed-text", "text": "hello"})
    decision = await orch.route_task(task)

    assert decision.target is None
    assert decision.can_execute is True
    assert "no suitable peer" in decision.reason

    await orch.close()


@pytest.mark.asyncio
async def test_intent_classify_routes_to_specialist_over_executive(
    make_task, make_peer, make_identity
):
    """intent_classify avoids executive, routes to specialist."""
    registry = PeerRegistry()
    identity = make_identity(
        node_tier="executive",
        capabilities=["gpu_inference", "cpu_worker", "cpu_inference"],
        vram_total_mb=24576,
        ollama_running=True,
        available_models=["gemma3:27b"],
    )
    orch = Orchestrator(identity, registry)

    specialist = make_peer(
        node_id="spec-1", hostname="rushuna",
        node_tier="specialist",
        capabilities=["gpu_inference", "cpu_worker", "cpu_inference"],
        vram_total_mb=12288,
        available_models=["gemma3:1b"],
    )
    await registry.add_or_update(specialist)

    task = make_task("intent_classify", {"query": "what time is it"})
    decision = await orch.route_task(task)

    assert decision.target is not None
    assert decision.target.node_id == "spec-1"
    assert "protecting executive VRAM" in decision.reason

    await orch.close()


@pytest.mark.asyncio
async def test_intent_classify_routes_to_light_over_executive(
    make_task, make_peer, make_identity
):
    """intent_classify routes to light node when no specialist available."""
    registry = PeerRegistry()
    identity = make_identity(
        node_tier="executive",
        capabilities=["gpu_inference", "cpu_worker", "cpu_inference"],
        vram_total_mb=24576,
        ollama_running=True,
        available_models=["gemma3:27b"],
    )
    orch = Orchestrator(identity, registry)

    light = make_peer(
        node_id="light-1", hostname="naru",
        node_tier="light",
        capabilities=["cpu_worker", "cpu_inference"],
        available_models=[],
    )
    await registry.add_or_update(light)

    task = make_task("intent_classify", {"query": "hello"})
    decision = await orch.route_task(task)

    assert decision.target is not None
    assert decision.target.node_id == "light-1"

    await orch.close()


@pytest.mark.asyncio
async def test_embedding_on_specialist_stays_local(
    make_task, make_peer, make_identity
):
    """When the local node IS the specialist, embedding stays local."""
    registry = PeerRegistry()
    identity = make_identity(
        node_tier="specialist",
        capabilities=["gpu_inference", "cpu_worker"],
        vram_total_mb=12288,
        ollama_running=True,
        available_models=["gemma3:1b", "nomic-embed-text"],
    )
    orch = Orchestrator(identity, registry)

    # Executive peer exists but should NOT steal embedding work
    executive = make_peer(
        node_id="exec-1", hostname="miu",
        node_tier="executive",
        capabilities=["gpu_inference", "cpu_worker"],
        vram_total_mb=24576,
        available_models=["gemma3:27b", "nomic-embed-text"],
    )
    await registry.add_or_update(executive)

    task = make_task("embedding", {"model": "nomic-embed-text", "text": "hello"})
    decision = await orch.route_task(task)

    # Local specialist should handle it, not route to executive
    assert decision.target is None
    assert decision.can_execute is True

    await orch.close()


@pytest.mark.asyncio
async def test_local_support_score_executive_penalty(make_identity):
    """Executive local node gets heavy penalty for support tasks."""
    registry = PeerRegistry()
    identity = make_identity(
        node_tier="executive",
        capabilities=["gpu_inference", "cpu_worker"],
        vram_total_mb=24576,
        ollama_running=True,
    )
    orch = Orchestrator(identity, registry)

    score = orch.router._local_support_score()
    # -2000 (executive) + 500 (gpu) = -1500
    assert score < 0

    await orch.close()


@pytest.mark.asyncio
async def test_local_support_score_specialist_positive(make_identity):
    """Specialist local node gets positive support score."""
    registry = PeerRegistry()
    identity = make_identity(
        node_tier="specialist",
        capabilities=["gpu_inference", "cpu_worker"],
        vram_total_mb=12288,
        ollama_running=True,
    )
    orch = Orchestrator(identity, registry)

    score = orch.router._local_support_score()
    # 1000 (specialist) + 500 (gpu) = 1500
    assert score > 1000

    await orch.close()
