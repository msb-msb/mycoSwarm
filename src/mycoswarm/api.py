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
import json
import logging
import time
from enum import Enum
from typing import TYPE_CHECKING

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
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
    target_ip: str | None = None
    target_port: int | None = None


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
        self._streams: dict[str, asyncio.Queue] = {}
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

    def create_stream(self, task_id: str) -> asyncio.Queue:
        """Create a token stream queue for a task."""
        q: asyncio.Queue = asyncio.Queue()
        self._streams[task_id] = q
        return q

    def get_stream(self, task_id: str) -> asyncio.Queue | None:
        return self._streams.get(task_id)

    def remove_stream(self, task_id: str) -> None:
        self._streams.pop(task_id, None)

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

    # Task types that should be distributed across the swarm
    # even when the local node could handle them.
    DISTRIBUTABLE_TASKS = {"web_search", "web_fetch", "file_read", "code_run"}

    async def _route_to_peer(task: TaskRequest, target):
        """Forward task to a peer and poll for result in background."""
        # Fix model mismatch: if the task specifies a model the peer
        # doesn't have, swap to the peer's first available model.
        MODEL_TASKS = {"inference", "embedding", "translate", "file_summarize"}
        task_data = task.model_dump()
        if (
            task.task_type in MODEL_TASKS
            and target.available_models
            and task_data.get("payload", {}).get("model")
        ):
            requested_model = task_data["payload"]["model"]
            if requested_model not in target.available_models:
                new_model = target.available_models[0]
                logger.info(
                    f"🔄 Model swap: {requested_model} not on "
                    f"{target.hostname}, using {new_model}"
                )
                task_data["payload"]["model"] = new_model

        orchestrator.record_dispatch(target.node_id)
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.post(
                    f"http://{target.ip}:{target.port}/task",
                    json=task_data,
                )
                resp.raise_for_status()
        except (httpx.ConnectError, httpx.TimeoutException) as e:
            orchestrator.record_completion(target.node_id)
            registry.record_failure(target.node_id)
            raise HTTPException(
                status_code=502,
                detail=f"Can't reach {target.hostname}: {e}",
            )

        registry.record_success(target.node_id)
        logger.info(
            f"🎯 Routed {task.task_type} → {target.hostname} "
            f"({target.ip}:{target.port})"
        )

        # Background: poll target for result
        async def _poll_remote_result():
            base_url = f"http://{target.ip}:{target.port}"
            deadline = time.time() + task.timeout_seconds
            try:
                async with httpx.AsyncClient(timeout=10.0) as poll_client:
                    while time.time() < deadline:
                        await asyncio.sleep(1.0)
                        try:
                            r = await poll_client.get(
                                f"{base_url}/task/{task.task_id}"
                            )
                            data = r.json()
                            if data.get("status") in ("completed", "failed"):
                                result = TaskResult(
                                    task_id=task.task_id,
                                    status=TaskStatus(data["status"]),
                                    result=data.get("result"),
                                    error=data.get("error"),
                                    duration_seconds=data.get(
                                        "duration_seconds", 0
                                    ),
                                    node_id=data.get(
                                        "node_id", target.node_id
                                    ),
                                )
                                await task_queue.store_result(result)
                                return
                        except httpx.HTTPError:
                            pass
                await task_queue.store_result(TaskResult(
                    task_id=task.task_id,
                    status=TaskStatus.FAILED,
                    error=f"Timed out waiting for {target.hostname}",
                    node_id=target.node_id,
                ))
            except Exception as e:
                logger.error(f"💥 Poll error for {task.task_id}: {e}")
                await task_queue.store_result(TaskResult(
                    task_id=task.task_id,
                    status=TaskStatus.FAILED,
                    error=f"Poll error: {e}",
                    node_id=target.node_id,
                ))
            finally:
                orchestrator.record_completion(target.node_id)

        asyncio.create_task(_poll_remote_result())

        return TaskResponse(
            task_id=task.task_id,
            status=TaskStatus.PENDING,
            message=f"Routed to {target.hostname}",
            target_ip=target.ip,
            target_port=target.port,
        )

    @app.post("/task", response_model=TaskResponse)
    async def submit_task(task: TaskRequest):
        logger.info(
            f"📥 Task received: {task.task_id} ({task.task_type}) "
            f"from {task.source_node}"
        )

        can_local = (
            orchestrator is None
            or orchestrator.can_handle_locally(task.task_type)
        )

        # Distributable tasks: pick least-loaded node (peers + local)
        if (
            orchestrator is not None
            and task.task_type in DISTRIBUTABLE_TASKS
        ):
            candidates = await orchestrator._select_nodes(task.task_type)
            target = orchestrator.pick_for_distribution(candidates)
            if target is not None:
                return await _route_to_peer(task, target)
            # Local wins — track inflight and handle here
            if can_local:
                orchestrator.record_dispatch(identity.node_id)

                async def _track_local_completion(tid: str):
                    """Decrement local inflight when task completes."""
                    deadline = time.time() + task.timeout_seconds
                    while time.time() < deadline:
                        await asyncio.sleep(0.5)
                        result = await task_queue.get_result(tid)
                        if result:
                            orchestrator.record_completion(identity.node_id)
                            return
                    orchestrator.record_completion(identity.node_id)

                asyncio.create_task(
                    _track_local_completion(task.task_id)
                )
                return await task_queue.submit(task)

        # Inference/embedding: prefer GPU peers over local CPU inference
        INFERENCE_TASKS = {"inference", "embedding", "translate", "file_summarize"}
        if (
            orchestrator is not None
            and task.task_type in INFERENCE_TASKS
            and can_local
        ):
            candidates = await orchestrator._select_nodes(task.task_type)
            # Route to a GPU peer if one scores higher than local
            gpu_candidates = [
                p for p in candidates
                if "gpu_inference" in p.capabilities
            ]
            if gpu_candidates:
                logger.info(
                    f"🎯 GPU peer available — routing {task.task_type} "
                    f"to {gpu_candidates[0].hostname} instead of local CPU"
                )
                return await _route_to_peer(task, gpu_candidates[0])
            # No GPU peer — fall through to local

        if can_local:
            if task.task_type == "inference":
                task_queue.create_stream(task.task_id)
            return await task_queue.submit(task)

        # Can't handle locally — route to best peer
        logger.info(
            f"🔀 Can't handle {task.task_type} locally, routing to peer..."
        )

        if orchestrator is None:
            raise HTTPException(
                status_code=503,
                detail=f"No orchestrator and can't handle: {task.task_type}",
            )

        candidates = await orchestrator._select_nodes(task.task_type)
        if not candidates:
            raise HTTPException(
                status_code=503,
                detail=f"No peer in swarm can handle: {task.task_type}",
            )

        return await _route_to_peer(task, candidates[0])

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

    @app.get("/task/{task_id}/stream")
    async def stream_task(task_id: str):
        """SSE endpoint — streams tokens as they are generated."""

        async def _event_generator():
            # If result already exists, replay as single event
            result = await task_queue.get_result(task_id)
            if result is not None:
                if result.status == TaskStatus.COMPLETED and result.result:
                    response_text = result.result.get("response", "")
                    if response_text:
                        yield f"data: {json.dumps({'token': response_text, 'done': False})}\n\n"
                    yield f"data: {json.dumps({'done': True, 'model': result.result.get('model', ''), 'tokens_per_second': result.result.get('tokens_per_second', 0), 'duration_seconds': result.duration_seconds, 'node_id': result.node_id})}\n\n"
                elif result.error:
                    yield f"data: {json.dumps({'error': result.error, 'done': True})}\n\n"
                return

            stream_queue = task_queue.get_stream(task_id)
            if stream_queue is None:
                yield f"data: {json.dumps({'error': 'No stream for this task', 'done': True})}\n\n"
                return

            try:
                while True:
                    try:
                        event = await asyncio.wait_for(
                            stream_queue.get(), timeout=300
                        )
                    except asyncio.TimeoutError:
                        yield f"data: {json.dumps({'error': 'Stream timeout', 'done': True})}\n\n"
                        break

                    if event is None:
                        # Sentinel — stream complete
                        break

                    yield f"data: {json.dumps(event)}\n\n"
            finally:
                task_queue.remove_stream(task_id)

        return StreamingResponse(
            _event_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            },
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
