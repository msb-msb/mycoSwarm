"""mycoSwarm node API service.

Each node runs a lightweight FastAPI server that exposes:
  GET  /health          → quick liveness check
  GET  /identity        → full node identity
  GET  /peers           → discovered peers
  POST /task            → submit a task to this node

All endpoints are LAN-only by default (bound to the LAN IP, not 0.0.0.0).
"""

from __future__ import annotations

import asyncio
import logging
import time
from enum import Enum
from typing import TYPE_CHECKING

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from mycoswarm.node import NodeIdentity, build_identity
from mycoswarm.discovery import PeerRegistry

if TYPE_CHECKING:
    from mycoswarm.orchestrator import Orchestrator

logger = logging.getLogger(__name__)


# --- Request/Response Models ---


class TaskStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class TaskRequest(BaseModel):
    """A task submitted to a node."""

    task_id: str
    task_type: str  # "inference", "web_fetch", "file_process", etc.
    payload: dict  # Task-specific data
    source_node: str  # node_id of the requester
    priority: int = Field(default=5, ge=1, le=10)  # 1=low, 10=urgent
    timeout_seconds: int = Field(default=300, ge=1, le=3600)


class TaskResponse(BaseModel):
    """Response after submitting a task."""

    task_id: str
    status: TaskStatus
    message: str


class TaskResult(BaseModel):
    """Result of a completed task."""

    task_id: str
    status: TaskStatus
    result: dict | None = None
    error: str | None = None
    duration_seconds: float = 0.0
    node_id: str = ""


class HealthResponse(BaseModel):
    node_id: str
    hostname: str
    status: str
    uptime_seconds: float
    peer_count: int


class PeerResponse(BaseModel):
    node_id: str
    hostname: str
    ip: str
    port: int
    node_tier: str
    capabilities: list[str]
    gpu_name: str | None
    vram_total_mb: int
    available_models: list[str]
    last_seen: float


# --- Task Queue ---


class TaskQueue:
    """Simple async task queue for a node."""

    def __init__(self, max_size: int = 100):
        self._queue: asyncio.Queue[TaskRequest] = asyncio.Queue(maxsize=max_size)
        self._results: dict[str, TaskResult] = {}
        self._active: dict[str, TaskRequest] = {}
        self._lock = asyncio.Lock()

    async def submit(self, task: TaskRequest) -> TaskResponse:
        """Add a task to the queue."""
        try:
            self._queue.put_nowait(task)
            return TaskResponse(
                task_id=task.task_id,
                status=TaskStatus.PENDING,
                message=f"Queued (position {self._queue.qsize()})",
            )
        except asyncio.QueueFull:
            raise HTTPException(
                status_code=503,
                detail="Task queue full — node is at capacity",
            )

    async def get_next(self) -> TaskRequest | None:
        """Get the next task from the queue (non-blocking)."""
        try:
            return self._queue.get_nowait()
        except asyncio.QueueEmpty:
            return None

    async def mark_active(self, task: TaskRequest):
        async with self._lock:
            self._active[task.task_id] = task

    async def store_result(self, result: TaskResult):
        async with self._lock:
            self._results[result.task_id] = result
            self._active.pop(result.task_id, None)

    async def get_result(self, task_id: str) -> TaskResult | None:
        async with self._lock:
            return self._results.get(task_id)

    @property
    def pending_count(self) -> int:
        return self._queue.qsize()

    @property
    def active_count(self) -> int:
        return len(self._active)


# --- API Factory ---


def create_api(
    identity: NodeIdentity,
    registry: PeerRegistry,
    task_queue: TaskQueue,
    start_time: float,
    orchestrator: Orchestrator | None = None,
) -> FastAPI:
    """Create the FastAPI application for this node."""

    app = FastAPI(
        title=f"mycoSwarm Node: {identity.hostname}",
        description=f"Node {identity.node_id} [{identity.node_tier}]",
        version="0.1.0",
    )

    @app.get("/health", response_model=HealthResponse)
    async def health():
        return HealthResponse(
            node_id=identity.node_id,
            hostname=identity.hostname,
            status="ok",
            uptime_seconds=round(time.time() - start_time, 1),
            peer_count=registry.count,
        )

    @app.get("/identity")
    async def get_identity():
        fresh = build_identity()
        return fresh.to_dict()

    @app.get("/peers", response_model=list[PeerResponse])
    async def get_peers():
        peers = await registry.get_all()
        return [
            PeerResponse(
                node_id=p.node_id,
                hostname=p.hostname,
                ip=p.ip,
                port=p.port,
                node_tier=p.node_tier,
                capabilities=p.capabilities,
                gpu_name=p.gpu_name,
                vram_total_mb=p.vram_total_mb,
                available_models=p.available_models,
                last_seen=p.last_seen,
            )
            for p in peers
        ]

    @app.post("/task", response_model=TaskResponse)
    async def submit_task(task: TaskRequest):
        logger.info(
            f"📥 Task received: {task.task_id} ({task.task_type}) "
            f"from {task.source_node}"
        )

        # Check if we can handle this locally
        can_local = (
            orchestrator is None
            or orchestrator.can_handle_locally(task.task_type)
        )

        if can_local:
            return await task_queue.submit(task)

        # Can't handle locally — route to best peer
        logger.info(
            f"🔀 Can't handle {task.task_type} locally, routing to peer..."
        )

        async def _route_remote():
            result = await orchestrator.route_task(task)
            if result is None:
                result = TaskResult(
                    task_id=task.task_id,
                    status=TaskStatus.FAILED,
                    error=f"No peer in swarm can handle: {task.task_type}",
                )
            await task_queue.store_result(result)

        asyncio.create_task(_route_remote())

        return TaskResponse(
            task_id=task.task_id,
            status=TaskStatus.PENDING,
            message="Routed to remote peer",
        )

    @app.get("/task/{task_id}")
    async def get_task_result(task_id: str):
        result = await task_queue.get_result(task_id)
        if result:
            return result
        return TaskResponse(
            task_id=task_id,
            status=TaskStatus.PENDING,
            message="Task is queued or in progress",
        )

    @app.get("/status")
    async def status():
        peers = await registry.get_all()
        return {
            "node_id": identity.node_id,
            "hostname": identity.hostname,
            "node_tier": identity.node_tier,
            "capabilities": identity.capabilities,
            "gpu": identity.gpu_name,
            "vram_total_mb": identity.vram_total_mb,
            "vram_free_mb": identity.vram_free_mb,
            "ollama_models": identity.available_models,
            "peers": len(peers),
            "tasks_pending": task_queue.pending_count,
            "tasks_active": task_queue.active_count,
            "uptime_seconds": round(time.time() - start_time, 1),
        }

    return app
