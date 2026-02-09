"""Tests for mycoswarm.orchestrator — task routing and scoring."""

import asyncio
from unittest.mock import AsyncMock, patch

import pytest

from mycoswarm.api import TaskRequest, TaskResult, TaskStatus
from mycoswarm.discovery import Peer, PeerRegistry
from mycoswarm.orchestrator import Orchestrator


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

    exec_score = orch._score_peer_for_cpu_work(executive)
    light_score = orch._score_peer_for_cpu_work(light)

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

    score_before = orch._score_peer_for_cpu_work(peer)
    orch.record_dispatch("busy-1")
    score_after = orch._score_peer_for_cpu_work(peer)

    assert score_after == score_before - 100

    await orch.close()


@pytest.mark.asyncio
async def test_retry_selects_next_candidate(make_task, make_peer, make_identity):
    """Mock _dispatch_to_peer to fail first, succeed second, verify two calls."""
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
        result = await orch.route_task(task)

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

    gpu_score = orch._score_peer_for_inference(gpu_peer)
    cpu_score = orch._score_peer_for_inference(cpu_peer)

    assert gpu_score > cpu_score

    await orch.close()
