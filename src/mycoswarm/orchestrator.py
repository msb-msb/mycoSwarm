"""mycoSwarm orchestrator.

Routes tasks to the best available node based on task type,
node capabilities, and current load. Any node can be the
orchestrator — it's a role, not a fixed service.

Routing logic:
  - "inference" tasks → GPU nodes first, by VRAM (biggest wins)
  - "web_fetch", "file_process" → CPU workers, by lowest load
  - "storage" → nodes with storage capability
  - Falls back to local execution if no peers available
"""

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field

import httpx

from mycoswarm.discovery import Peer, PeerRegistry
from mycoswarm.api import TaskRequest, TaskResult, TaskStatus
from mycoswarm.node import NodeIdentity

logger = logging.getLogger(__name__)

# Task types and which capabilities they need
TASK_ROUTING = {
    "inference": ["gpu_inference", "cpu_inference"],
    "web_fetch": ["cpu_worker"],
    "web_search": ["cpu_worker"],
    "file_process": ["cpu_worker"],
    "file_read": ["storage", "cpu_worker"],
    "file_write": ["storage", "cpu_worker"],
    "embedding": ["gpu_inference", "cpu_inference"],
}

PEER_TIMEOUT = 10.0


@dataclass
class TaskRecord:
    """Tracks a dispatched task."""

    task_id: str
    task_type: str
    target_node: str
    target_ip: str
    target_port: int
    status: TaskStatus = TaskStatus.PENDING
    result: dict | None = None
    error: str | None = None
    dispatched_at: float = field(default_factory=time.time)
    completed_at: float | None = None

    @property
    def duration(self) -> float | None:
        if self.completed_at:
            return self.completed_at - self.dispatched_at
        return None


class Orchestrator:
    """Routes and dispatches tasks across the swarm."""

    def __init__(self, identity: NodeIdentity, registry: PeerRegistry):
        self.identity = identity
        self.registry = registry
        self._records: dict[str, TaskRecord] = {}
        self._client = httpx.AsyncClient(timeout=PEER_TIMEOUT)

    async def close(self):
        await self._client.aclose()

    def _score_peer_for_inference(self, peer: Peer) -> float:
        score = 0.0
        if "gpu_inference" in peer.capabilities:
            score += 1000
        score += peer.vram_total_mb / 100
        if peer.node_tier == "executive":
            score += 500
        elif peer.node_tier == "specialist":
            score += 200
        if peer.is_stale:
            score -= 2000
        return score

    def _score_peer_for_cpu_work(self, peer: Peer) -> float:
        score = 0.0
        if "cpu_worker" in peer.capabilities:
            score += 100
        if peer.node_tier == "light":
            score += 50
        elif peer.node_tier == "executive":
            score -= 20
        if peer.is_stale:
            score -= 2000
        return score

    async def _select_node(self, task_type: str) -> Peer | None:
        required_caps = TASK_ROUTING.get(task_type, ["cpu_worker"])
        peers = await self.registry.get_all()

        eligible = [
            p for p in peers
            if any(cap in p.capabilities for cap in required_caps)
            and not p.is_stale
        ]

        if not eligible:
            return None

        if task_type in ("inference", "embedding"):
            eligible.sort(key=self._score_peer_for_inference, reverse=True)
        else:
            eligible.sort(key=self._score_peer_for_cpu_work, reverse=True)

        selected = eligible[0]
        logger.info(
            f"🎯 Routing {task_type} → {selected.hostname} "
            f"({selected.ip}) [{selected.node_tier}]"
        )
        return selected

    async def _dispatch_to_peer(self, peer: Peer, task: TaskRequest) -> TaskResult:
        """Dispatch a task to a peer and poll until completion."""
        base_url = f"http://{peer.ip}:{peer.port}"

        try:
            # Submit the task
            response = await self._client.post(
                f"{base_url}/task",
                json=task.model_dump(),
                timeout=PEER_TIMEOUT,
            )
            response.raise_for_status()
            logger.info(
                f"📤 Dispatched {task.task_id} to {peer.hostname} "
                f"({peer.ip}:{peer.port})"
            )

            # Poll for the result
            deadline = time.time() + task.timeout_seconds
            while time.time() < deadline:
                await asyncio.sleep(0.5)
                try:
                    poll_resp = await self._client.get(
                        f"{base_url}/task/{task.task_id}",
                        timeout=PEER_TIMEOUT,
                    )
                    data = poll_resp.json()
                    status = data.get("status", "pending")

                    if status == "completed":
                        logger.info(
                            f"✅ Remote task {task.task_id} completed on "
                            f"{peer.hostname}"
                        )
                        return TaskResult(
                            task_id=task.task_id,
                            status=TaskStatus.COMPLETED,
                            result=data.get("result"),
                            duration_seconds=data.get("duration_seconds", 0),
                            node_id=data.get("node_id", peer.node_id),
                        )
                    elif status == "failed":
                        logger.error(
                            f"❌ Remote task {task.task_id} failed on "
                            f"{peer.hostname}: {data.get('error')}"
                        )
                        return TaskResult(
                            task_id=task.task_id,
                            status=TaskStatus.FAILED,
                            error=data.get("error", "Remote task failed"),
                            duration_seconds=data.get("duration_seconds", 0),
                            node_id=data.get("node_id", peer.node_id),
                        )
                except httpx.HTTPError:
                    pass  # Transient error, keep polling

            logger.error(f"⏱️ Remote task {task.task_id} timed out on {peer.hostname}")
            return TaskResult(
                task_id=task.task_id,
                status=TaskStatus.FAILED,
                error=f"Timed out waiting for {peer.hostname} after {task.timeout_seconds}s",
                node_id=peer.node_id,
            )

        except httpx.TimeoutException:
            logger.error(f"Timeout dispatching to {peer.hostname}")
            return TaskResult(
                task_id=task.task_id,
                status=TaskStatus.FAILED,
                error=f"Timeout contacting {peer.hostname}",
                node_id=peer.node_id,
            )
        except httpx.HTTPError as e:
            logger.error(f"HTTP error dispatching to {peer.hostname}: {e}")
            return TaskResult(
                task_id=task.task_id,
                status=TaskStatus.FAILED,
                error=str(e),
                node_id=peer.node_id,
            )

    def can_handle_locally(self, task_type: str) -> bool:
        """Check if this node has capabilities for the given task type."""
        required_caps = TASK_ROUTING.get(task_type, ["cpu_worker"])
        return any(cap in self.identity.capabilities for cap in required_caps)

    async def route_task(self, task: TaskRequest) -> TaskResult | None:
        """Route a task to the best peer. Returns None if no peer available.

        This is the main entry point used by the API when the local node
        can't handle a task. It dispatches to the best peer and polls
        until the result is ready.
        """
        target = await self._select_node(task.task_type)
        if not target:
            logger.warning(
                f"❌ No peer available for {task.task_type} "
                f"(task {task.task_id})"
            )
            return None

        record = TaskRecord(
            task_id=task.task_id,
            task_type=task.task_type,
            target_node=target.node_id,
            target_ip=target.ip,
            target_port=target.port,
        )
        self._records[task.task_id] = record

        result = await self._dispatch_to_peer(target, task)
        record.status = result.status
        record.result = result.result
        record.error = result.error
        record.completed_at = time.time()
        return result

    async def submit(
        self,
        task_type: str,
        payload: dict,
        prefer_remote: bool = True,
        priority: int = 5,
        timeout: int = 300,
    ) -> TaskResult:
        """Submit a task to the swarm."""
        task_id = f"task-{uuid.uuid4().hex[:8]}"

        task = TaskRequest(
            task_id=task_id,
            task_type=task_type,
            payload=payload,
            source_node=self.identity.node_id,
            priority=priority,
            timeout_seconds=timeout,
        )

        target = await self._select_node(task_type) if prefer_remote else None

        if target:
            record = TaskRecord(
                task_id=task_id,
                task_type=task_type,
                target_node=target.node_id,
                target_ip=target.ip,
                target_port=target.port,
            )
            self._records[task_id] = record
            result = await self._dispatch_to_peer(target, task)
            record.status = result.status
            record.result = result.result
            record.error = result.error
            record.completed_at = time.time()
            return result

        # No peer — handle locally
        if self.can_handle_locally(task_type):
            logger.info(f"📍 No peers for {task_type} — handling locally")
            return TaskResult(
                task_id=task_id,
                status=TaskStatus.PENDING,
                result={"routed_to": "local", "task_type": task_type},
                node_id=self.identity.node_id,
            )

        logger.warning(f"❌ No node available for task type: {task_type}")
        return TaskResult(
            task_id=task_id,
            status=TaskStatus.FAILED,
            error=f"No node in swarm can handle task type: {task_type}",
        )

    async def submit_parallel(
        self,
        tasks: list[tuple[str, dict]],
        timeout: int = 300,
    ) -> list[TaskResult]:
        """Submit multiple tasks in parallel across the swarm.

        This is the key advantage over serial agents.
        """
        coros = [
            self.submit(task_type, payload, timeout=timeout)
            for task_type, payload in tasks
        ]
        return await asyncio.gather(*coros)

    def get_record(self, task_id: str) -> TaskRecord | None:
        return self._records.get(task_id)

    async def swarm_status(self) -> dict:
        """Swarm overview from the orchestrator's perspective."""
        peers = await self.registry.get_all()
        active = [p for p in peers if not p.is_stale]

        gpu_nodes = [p for p in active if "gpu_inference" in p.capabilities]
        total_vram = sum(p.vram_total_mb for p in gpu_nodes) + self.identity.vram_total_mb

        all_models = set()
        for p in active:
            all_models.update(p.available_models)

        return {
            "orchestrator": self.identity.node_id,
            "total_nodes": len(active) + 1,
            "gpu_nodes": len(gpu_nodes) + (1 if self.identity.vram_total_mb > 0 else 0),
            "cpu_nodes": len(active) + 1 - len(gpu_nodes),
            "total_vram_mb": total_vram,
            "unique_models": sorted(all_models),
            "tasks_dispatched": len(self._records),
        }
