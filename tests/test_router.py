"""Tests for mycoswarm.router — unified routing and scoring."""

import time
from unittest.mock import AsyncMock

import pytest

from mycoswarm.discovery import Peer, PeerRegistry
from mycoswarm.router import Router, RouteResult


@pytest.mark.asyncio
async def test_specialist_preferred_over_executive(make_peer, make_identity):
    """Specialist peer scores higher than executive for inference.

    specialist: 1000 (gpu) + 122.88 (12288/100) + 500 (specialist) = 1622.88
    executive:  1000 (gpu) + 245.76 (24576/100) + 200 (executive)  = 1445.76
    """
    identity = make_identity()
    router = Router(identity)

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

    spec_score = router._score_peer_for_inference(specialist)
    exec_score = router._score_peer_for_inference(executive)

    assert spec_score > exec_score
    assert spec_score == pytest.approx(1622.88)
    assert exec_score == pytest.approx(1445.76)


@pytest.mark.asyncio
async def test_executive_wins_when_sole_node(make_identity):
    """Executive node handles inference locally when no peers exist."""
    identity = make_identity(
        node_tier="executive",
        capabilities=["gpu_inference", "cpu_worker"],
        vram_total_mb=24576,
        ollama_running=True,
        available_models=["gemma3:27b"],
    )
    router = Router(identity, peers=[])

    result = router.resolve_sync("inference", model="gemma3:27b")

    assert result is not None
    assert result.is_local is True
    assert "No peer has model" in result.reason


@pytest.mark.asyncio
async def test_cpu_tasks_avoid_executive(make_peer, make_identity):
    """Light peer wins over executive for CPU work tasks."""
    identity = make_identity()
    router = Router(identity)

    executive = make_peer(
        node_id="exec-1", hostname="miu",
        node_tier="executive",
        capabilities=["gpu_inference", "cpu_worker"],
        vram_total_mb=24576,
    )
    light = make_peer(
        node_id="light-1", hostname="naru",
        node_tier="light",
        capabilities=["cpu_worker"],
    )

    exec_score = router._score_peer_for_cpu_work(executive)
    light_score = router._score_peer_for_cpu_work(light)

    assert light_score > exec_score
    assert exec_score < 0  # 100 - 500 = -400


def test_inflight_penalty(make_peer, make_identity):
    """record_dispatch reduces score by 100 per inflight task."""
    identity = make_identity()
    router = Router(identity)

    peer = make_peer(
        node_id="busy-1", hostname="busy-node",
        node_tier="light", capabilities=["cpu_worker"],
    )

    score_before = router._score_peer_for_cpu_work(peer)
    router.record_dispatch("busy-1")
    score_after = router._score_peer_for_cpu_work(peer)

    assert score_after == score_before - 100

    router.record_completion("busy-1")
    score_recovered = router._score_peer_for_cpu_work(peer)
    assert score_recovered == score_before


def test_stale_node_avoided(make_peer, make_identity):
    """Stale peers get massive negative score."""
    identity = make_identity()
    router = Router(identity)

    stale_peer = make_peer(
        node_id="stale-1", hostname="ghost",
        node_tier="specialist",
        capabilities=["gpu_inference", "cpu_worker"],
        vram_total_mb=12288,
        last_seen=time.time() - 600,  # 10 minutes ago
    )

    score = router._score_peer_for_inference(stale_peer)
    assert score < 0  # -2000 stale penalty dominates

    # Verify stale peer is filtered out by _select_candidates
    candidates = router._select_candidates("inference", [stale_peer])
    assert len(candidates) == 0


@pytest.mark.asyncio
async def test_resolve_async_with_registry(make_peer, make_identity):
    """Async resolve routes inference to GPU peer via registry."""
    registry = PeerRegistry()
    identity = make_identity(
        node_tier="light",
        capabilities=["cpu_worker", "cpu_inference"],
        ollama_running=True,
        available_models=["gemma3:1b"],
    )
    router = Router(identity, registry=registry)

    gpu_peer = make_peer(
        node_id="gpu-1", hostname="gpu-node",
        node_tier="specialist",
        capabilities=["gpu_inference", "cpu_worker"],
        vram_total_mb=12288,
        available_models=["gemma3:27b"],
    )
    await registry.add_or_update(gpu_peer)

    result = await router.resolve("inference", model="gemma3:27b")

    assert result is not None
    assert result.is_local is False
    assert result.node_hostname == "gpu-node"
    assert result._peer is not None
    assert result._peer.node_id == "gpu-1"


def test_build_routing_table(make_peer, make_identity):
    """build_routing_table resolves pipeline steps correctly."""
    identity = make_identity(
        node_tier="executive",
        capabilities=["gpu_inference", "cpu_worker"],
        vram_total_mb=24576,
        ollama_running=True,
        available_models=["gemma3:27b", "gemma3:12b", "nomic-embed-text"],
    )
    specialist = make_peer(
        node_id="spec-1", hostname="rushuna",
        node_tier="specialist",
        capabilities=["gpu_inference", "cpu_worker"],
        vram_total_mb=12288,
        available_models=["gemma3:12b", "nomic-embed-text"],
    )

    router = Router(identity, peers=[specialist])

    steps = [
        {"name": "researcher", "task_type": "research"},
        {"name": "writer", "task_type": "writing"},
        {"name": "editor", "task_type": "editing", "model": "gemma3:27b"},
    ]

    results = router.build_routing_table(steps)

    assert len(results) == 3

    # research prefers gemma3:12b — specialist has it and scores higher
    assert results[0].model == "gemma3:12b"

    # writing prefers gemma3:27b — both have it, specialist scores higher
    assert results[1].model == "gemma3:27b"

    # editor has explicit model override
    assert results[2].model == "gemma3:27b"

    # All results have proper fields
    for r in results:
        assert isinstance(r, RouteResult)
        assert r.model
        assert r.node_hostname
