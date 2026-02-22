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
from mycoswarm.auth import get_auth_header
from mycoswarm.node import NodeIdentity

logger = logging.getLogger(__name__)

# Task types that should be load-balanced across the swarm
# even when the local node could handle them.
DISTRIBUTABLE_TASKS = {"web_search", "web_fetch", "file_read", "code_run", "intent_classify"}

# Task types that prefer GPU inference nodes
INFERENCE_TASKS = {"inference", "embedding", "translate", "file_summarize"}

# Task types and which capabilities they need
TASK_ROUTING = {
    "inference": ["gpu_inference", "cpu_inference"],
    "embedding": ["gpu_inference", "cpu_inference"],
    "translate": ["gpu_inference", "cpu_inference"],
    "file_summarize": ["gpu_inference", "cpu_inference"],
    "web_fetch": ["cpu_worker"],
    "web_search": ["cpu_worker"],
    "file_read": ["file_processing", "cpu_worker"],
    "code_run": ["code_execution", "cpu_worker"],
    "intent_classify": ["cpu_worker", "gpu_inference", "cpu_inference"],
    "file_process": ["cpu_worker"],
}

PEER_TIMEOUT = 10.0
MAX_DISPATCH_ATTEMPTS = 3  # first try + 2 retries


@dataclass
class RoutingDecision:
    """Where a task should be executed.

    Returned by Orchestrator.route_task() — the single routing authority.
    """

    target: Peer | None  # None = execute locally
    reason: str
    can_execute: bool = True  # False = no node in swarm can handle this


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

    def __init__(self, identity: NodeIdentity, registry: PeerRegistry, swarm_token: str | None = None):
        self.identity = identity
        self.registry = registry
        self._records: dict[str, TaskRecord] = {}
        self._inflight: dict[str, int] = {}  # node_id → active task count
        _headers = get_auth_header(swarm_token) if swarm_token else {}
        self._client = httpx.AsyncClient(timeout=PEER_TIMEOUT, headers=_headers)

    def record_dispatch(self, node_id: str) -> None:
        """Increment inflight count for a peer (call when dispatching)."""
        self._inflight[node_id] = self._inflight.get(node_id, 0) + 1

    def record_completion(self, node_id: str) -> None:
        """Decrement inflight count for a peer (call when result arrives)."""
        count = self._inflight.get(node_id, 0)
        if count > 0:
            self._inflight[node_id] = count - 1

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
            score -= 500  # Reserve GPU nodes for inference
        if peer.is_stale:
            score -= 2000
        # Penalise busy peers — distributes tasks round-robin style
        score -= self._inflight.get(peer.node_id, 0) * 100
        return score

    def _local_cpu_score(self) -> float:
        """Score the local node as a CPU work candidate."""
        score = 0.0
        if "cpu_worker" in self.identity.capabilities:
            score += 100
        if self.identity.node_tier == "light":
            score += 50
        elif self.identity.node_tier == "executive":
            score -= 500
        score -= self._inflight.get(self.identity.node_id, 0) * 100
        return score

    def _local_inference_score(self) -> float:
        """Score the local node as an inference candidate.

        Uses the same formula as _score_peer_for_inference so local
        and peer scores are directly comparable.
        """
        score = 0.0
        if "gpu_inference" in self.identity.capabilities:
            score += 1000
        score += self.identity.vram_total_mb / 100
        if self.identity.node_tier == "executive":
            score += 500
        elif self.identity.node_tier == "specialist":
            score += 200
        if not self.identity.ollama_running:
            score -= 5000  # Can't do inference without Ollama
        score -= self._inflight.get(self.identity.node_id, 0) * 100
        return score

    def pick_for_distribution(self, candidates: list[Peer]) -> Peer | None:
        """Pick the best node for a distributable task, including local.

        Returns a Peer to route to, or None if local should handle it.
        """
        local_score = self._local_cpu_score()

        if not candidates:
            return None  # Local only option

        best = candidates[0]  # Already sorted by score
        best_score = self._score_peer_for_cpu_work(best)

        if best_score > local_score:
            return best
        return None  # Local wins

    async def _select_nodes(self, task_type: str) -> list[Peer]:
        """Return eligible peers ranked by score, filtering stale/unhealthy."""
        required_caps = TASK_ROUTING.get(task_type, ["cpu_worker"])
        peers = await self.registry.get_all()

        eligible = []
        for p in peers:
            has_caps = any(cap in p.capabilities for cap in required_caps)
            stale = p.is_stale
            healthy = self.registry.is_healthy(p.node_id)

            if has_caps and not stale and healthy:
                eligible.append(p)
            else:
                logger.warning(
                    f"🔍 Filtered out {p.hostname}: "
                    f"caps={p.capabilities} (need {required_caps}, match={has_caps}) "
                    f"stale={stale} (age={p.age_seconds:.0f}s) "
                    f"healthy={healthy}"
                )

        if not eligible and peers:
            logger.warning(
                f"⚠️ {len(peers)} peer(s) known but none eligible for "
                f"{task_type} (need: {required_caps})"
            )

        if task_type in INFERENCE_TASKS:
            eligible.sort(key=self._score_peer_for_inference, reverse=True)
        else:
            eligible.sort(key=self._score_peer_for_cpu_work, reverse=True)

        return eligible

    async def _dispatch_to_peer(
        self, peer: Peer, task: TaskRequest
    ) -> tuple[TaskResult, bool]:
        """Dispatch a task to a peer and poll until completion.

        Returns (result, is_dispatch_error). Dispatch errors (can't reach
        the peer at all) are retryable; task-level failures are not.
        """
        base_url = f"http://{peer.ip}:{peer.port}"

        try:
            # Submit the task
            response = await self._client.post(
                f"{base_url}/task",
                json=task.model_dump(),
                timeout=PEER_TIMEOUT,
            )
            response.raise_for_status()
            self.registry.record_success(peer.node_id)
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
                        ), False
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
                        ), False  # Task failed, but peer was reachable
                except httpx.HTTPError:
                    pass  # Transient poll error, keep trying

            logger.error(f"⏱️ Remote task {task.task_id} timed out on {peer.hostname}")
            return TaskResult(
                task_id=task.task_id,
                status=TaskStatus.FAILED,
                error=f"Timed out waiting for {peer.hostname} after {task.timeout_seconds}s",
                node_id=peer.node_id,
            ), False  # Peer was reachable, just slow

        except (httpx.TimeoutException, httpx.ConnectError) as e:
            self.registry.record_failure(peer.node_id)
            logger.error(f"🔌 Can't reach {peer.hostname}: {e}")
            return TaskResult(
                task_id=task.task_id,
                status=TaskStatus.FAILED,
                error=f"Can't reach {peer.hostname}: {e}",
                node_id=peer.node_id,
            ), True  # Dispatch error — retryable
        except httpx.HTTPError as e:
            self.registry.record_failure(peer.node_id)
            logger.error(f"HTTP error dispatching to {peer.hostname}: {e}")
            return TaskResult(
                task_id=task.task_id,
                status=TaskStatus.FAILED,
                error=str(e),
                node_id=peer.node_id,
            ), True  # Dispatch error — retryable

    def can_handle_locally(self, task_type: str) -> bool:
        """Check if this node can actually handle the given task type.

        For inference/embedding, having the capability isn't enough —
        Ollama must be running too.
        """
        required_caps = TASK_ROUTING.get(task_type, ["cpu_worker"])
        has_caps = any(cap in self.identity.capabilities for cap in required_caps)
        if not has_caps:
            return False
        if task_type in INFERENCE_TASKS and not self.identity.ollama_running:
            return False
        return True

    async def route_task(self, task: TaskRequest) -> RoutingDecision:
        """Single authority for all routing decisions.

        Returns a RoutingDecision indicating whether to execute locally
        (target=None) or on a specific peer (target=Peer).  Does NOT
        dispatch — the caller is responsible for execution.
        """
        can_local = self.can_handle_locally(task.task_type)
        candidates = await self._select_nodes(task.task_type)

        # --- Distributable tasks: load-balance across peers + local ---
        if task.task_type in DISTRIBUTABLE_TASKS:
            target = self.pick_for_distribution(candidates)
            if target is not None:
                return RoutingDecision(
                    target=target,
                    reason=f"Distributed to {target.hostname}",
                )
            if can_local:
                return RoutingDecision(
                    target=None,
                    reason="Local wins distribution scoring",
                )
            # Can't handle locally and no good peer — fall through

        # --- Inference tasks: prefer GPU nodes with the requested model ---
        elif task.task_type in INFERENCE_TASKS:
            requested_model = (task.payload or {}).get("model", "")

            if requested_model:
                model_peers = [
                    p for p in candidates
                    if requested_model in getattr(p, "available_models", [])
                ]
                gpu_model_peers = [
                    p for p in model_peers
                    if "gpu_inference" in p.capabilities
                ]
                best_peers = gpu_model_peers or model_peers

                if best_peers:
                    best_peer = best_peers[0]  # Already sorted by score
                    best_score = self._score_peer_for_inference(best_peer)

                    # Compare against local if local also has this model
                    local_has_model = (
                        can_local
                        and requested_model in self.identity.available_models
                    )
                    if local_has_model:
                        local_score = self._local_inference_score()
                        if local_score >= best_score:
                            logger.info(
                                f"📍 Local inference wins "
                                f"({local_score:.0f} vs "
                                f"{best_score:.0f} on {best_peer.hostname})"
                            )
                            return RoutingDecision(
                                target=None,
                                reason="Local scores higher for inference",
                            )

                    logger.info(
                        f"🎯 Routing {task.task_type} → "
                        f"{best_peer.hostname} "
                        f"(has model {requested_model})"
                    )
                    return RoutingDecision(
                        target=best_peer,
                        reason=f"Peer {best_peer.hostname} has model "
                               f"{requested_model}",
                    )

                # No peer has the model — local fallback
                if can_local:
                    logger.warning(
                        f"⚠️ No peer has model '{requested_model}' "
                        f"— handling locally"
                    )
                    return RoutingDecision(
                        target=None,
                        reason=f"No peer has model '{requested_model}'",
                    )
            else:
                # No model specified — prefer GPU peers
                gpu_peers = [
                    p for p in candidates
                    if "gpu_inference" in p.capabilities
                ]
                if gpu_peers:
                    best_peer = gpu_peers[0]
                    best_score = self._score_peer_for_inference(best_peer)
                    if can_local:
                        local_score = self._local_inference_score()
                        if local_score >= best_score:
                            return RoutingDecision(
                                target=None,
                                reason="Local GPU wins",
                            )
                    return RoutingDecision(
                        target=best_peer,
                        reason=f"GPU peer {best_peer.hostname}",
                    )
                if can_local:
                    return RoutingDecision(
                        target=None,
                        reason="Local inference (no GPU peers)",
                    )

        # --- Generic fallback ---
        if can_local:
            return RoutingDecision(target=None, reason="Handle locally")

        if candidates:
            return RoutingDecision(
                target=candidates[0],
                reason=f"Can't handle locally, routing to "
                       f"{candidates[0].hostname}",
            )

        logger.warning(
            f"❌ No node available for {task.task_type} "
            f"(task {task.task_id})"
        )
        return RoutingDecision(
            target=None,
            reason=f"No node can handle: {task.task_type}",
            can_execute=False,
        )

    async def dispatch_task(self, task: TaskRequest) -> TaskResult | None:
        """Dispatch a task to peers with retry logic.

        Selects candidates, tries up to MAX_DISPATCH_ATTEMPTS peers in
        score order.  Only retries on dispatch errors (peer unreachable);
        task-level failures are returned immediately.

        Returns None if no peer is available (caller should try local).
        """
        candidates = await self._select_nodes(task.task_type)

        # For inference tasks, filter candidates to those that have the model
        if task.task_type in INFERENCE_TASKS:
            requested_model = (task.payload or {}).get("model", "")
            if requested_model:
                model_candidates = [
                    p for p in candidates
                    if requested_model in getattr(p, "available_models", [])
                ]
                if model_candidates:
                    candidates = model_candidates
                else:
                    return None  # No peer has the model

        if not candidates:
            return None

        last_result = None
        for i, target in enumerate(candidates[:MAX_DISPATCH_ATTEMPTS]):
            if i > 0:
                logger.info(
                    f"🔄 Retry {i}/{MAX_DISPATCH_ATTEMPTS - 1}: "
                    f"{task.task_id} → {target.hostname}"
                )
            else:
                logger.info(
                    f"🎯 Routing {task.task_type} → {target.hostname} "
                    f"({target.ip}) [{target.node_tier}]"
                )

            record = TaskRecord(
                task_id=task.task_id,
                task_type=task.task_type,
                target_node=target.node_id,
                target_ip=target.ip,
                target_port=target.port,
            )
            self._records[task.task_id] = record

            result, is_dispatch_error = await self._dispatch_to_peer(
                target, task
            )
            record.status = result.status
            record.result = result.result
            record.error = result.error
            record.completed_at = time.time()
            last_result = result

            if not is_dispatch_error:
                return result  # Task reached the peer — done

            # Dispatch error — try next candidate

        return last_result  # All attempts exhausted

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

        if prefer_remote:
            decision = await self.route_task(task)

            if decision.target is not None:
                # Dispatch to peer with retry
                result = await self.dispatch_task(task)
                if result:
                    return result
                # All dispatch attempts failed — fall through to local

            elif not decision.can_execute:
                return TaskResult(
                    task_id=task_id,
                    status=TaskStatus.FAILED,
                    error=decision.reason,
                )

        # Handle locally
        if self.can_handle_locally(task_type):
            logger.info(f"📍 Handling {task_type} locally")
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
