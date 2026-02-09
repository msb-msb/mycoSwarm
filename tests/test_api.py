"""Tests for mycoswarm.api — FastAPI endpoints."""

import time

import pytest
import httpx

from mycoswarm.api import TaskQueue, TaskRequest, create_api
from mycoswarm.discovery import PeerRegistry


@pytest.mark.asyncio
async def test_health_returns_200(make_identity):
    """GET /health → 200, response has node_id, status='ok'."""
    identity = make_identity()
    registry = PeerRegistry()
    queue = TaskQueue()
    app = create_api(identity, registry, queue, time.time())

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/health")

    assert resp.status_code == 200
    data = resp.json()
    assert data["node_id"] == identity.node_id
    assert data["status"] == "ok"


@pytest.mark.asyncio
async def test_status_returns_node_info(make_identity):
    """GET /status → 200, response has hostname, capabilities, node_tier."""
    identity = make_identity(
        hostname="status-test",
        capabilities=["cpu_worker", "file_processing"],
        node_tier="light",
    )
    registry = PeerRegistry()
    queue = TaskQueue()
    app = create_api(identity, registry, queue, time.time())

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/status")

    assert resp.status_code == 200
    data = resp.json()
    assert data["hostname"] == "status-test"
    assert "cpu_worker" in data["capabilities"]
    assert data["node_tier"] == "light"


@pytest.mark.asyncio
async def test_task_accepts_and_queues_ping(make_identity):
    """POST /task with ping → 200, status=pending."""
    identity = make_identity()
    registry = PeerRegistry()
    queue = TaskQueue()
    app = create_api(identity, registry, queue, time.time())

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post("/task", json={
            "task_id": "test-ping-001",
            "task_type": "ping",
            "payload": {},
            "source_node": "myco-remote001",
        })

    assert resp.status_code == 200
    data = resp.json()
    assert data["task_id"] == "test-ping-001"
    assert data["status"] == "pending"
