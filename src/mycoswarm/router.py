"""mycoSwarm unified router — single routing authority.

Scoring, candidate selection, and routing decisions used by both
the orchestrator (daemon async path) and the pipeline (sync HTTP path).

Scoring philosophy:
  - Specialist nodes get higher inference bonus (+500) than executive (+200).
    This naturally prefers specialist GPUs for small models that fit on them,
    while executive still wins on raw VRAM for large models.
  - Support tasks (gate, embedding) avoid executive to protect VRAM.
  - CPU tasks avoid executive entirely (-500 penalty).
"""

import logging
import re
import time
from dataclasses import dataclass, field

import httpx

from mycoswarm.discovery import Peer, PeerRegistry
from mycoswarm.node import NodeIdentity

logger = logging.getLogger(__name__)

# ── Constants (single source of truth for all routing) ──────────────

# Task types that should be load-balanced across the swarm
# even when the local node could handle them.
DISTRIBUTABLE_TASKS = {"web_search", "web_fetch", "file_read", "code_run", "intent_classify"}

# Task types that prefer GPU inference nodes
INFERENCE_TASKS = {"inference", "embedding", "translate", "file_summarize"}

# Task types that should avoid the primary GPU (executive) node.
# These use small models (gate, embedding) that would cause VRAM
# swapping with the main inference model on the executive node.
# Routing preference: specialist > light > executive (last resort).
INFERENCE_SUPPORT_TASKS = {"intent_classify", "embedding"}

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


# ── Data structures ─────────────────────────────────────────────────


@dataclass
class RouteResult:
    """Where a task should be routed.

    Used by both pipeline (for routing table display) and orchestrator
    (converted to RoutingDecision for dispatch).
    """

    model: str
    node_hostname: str
    node_ip: str
    node_port: int
    is_local: bool
    reason: str
    task_type: str = ""
    score: float = 0.0
    _peer: Peer | None = field(default=None, repr=False)


# ── Helpers ─────────────────────────────────────────────────────────


def _model_size(name: str) -> float:
    """Extract parameter count in billions from a model name."""
    m = re.search(r"(\d+(?:\.\d+)?)b", name.lower())
    return float(m.group(1)) if m else 0.0


# ── Router ──────────────────────────────────────────────────────────


class Router:
    """Unified routing: scoring, candidate selection, route resolution.

    Two usage modes:
      - Daemon (async): pass registry for live health-checked peers.
      - Pipeline (sync): pass peers list from HTTP snapshot.
    """

    def __init__(
        self,
        identity: NodeIdentity,
        registry: PeerRegistry | None = None,
        peers: list[Peer] | None = None,
        swarm_token: str | None = None,
    ):
        self.identity = identity
        self.registry = registry
        self._peers = peers or []
        self._inflight: dict[str, int] = {}  # node_id -> active task count
        self._swarm_token = swarm_token

    @classmethod
    def from_daemon(cls, daemon_url: str, swarm_token: str | None = None) -> "Router":
        """Build a Router from a running daemon's HTTP endpoints."""
        headers: dict[str, str] = {}
        if swarm_token:
            from mycoswarm.auth import get_auth_header
            headers = get_auth_header(swarm_token)

        with httpx.Client(headers=headers, timeout=5) as client:
            status = client.get(f"{daemon_url}/status").json()
            peers_data = client.get(f"{daemon_url}/peers").json()

        identity = NodeIdentity(
            node_id=status["node_id"],
            hostname=status["hostname"],
            lan_ip=status.get("ip"),
            node_tier=status["node_tier"],
            capabilities=status["capabilities"],
            gpu_tier="none",  # not exposed in /status
            max_model_params_b=0.0,
            gpu_name=status.get("gpu"),
            vram_total_mb=status.get("vram_total_mb", 0),
            vram_free_mb=status.get("vram_free_mb", 0),
            ram_total_mb=status.get("ram_total_mb", 0),
            ram_available_mb=0,
            cpu_model=status.get("cpu_model", ""),
            cpu_cores=status.get("cpu_cores", 0),
            disk_free_gb=0.0,
            ollama_running=len(status.get("ollama_models", [])) > 0,
            available_models=status.get("ollama_models", []),
            timestamp=time.time(),
            version=status.get("version", ""),
        )

        peers = []
        for p in peers_data:
            peers.append(Peer(
                node_id=p["node_id"],
                hostname=p["hostname"],
                ip=p["ip"],
                port=p["port"],
                node_tier=p["node_tier"],
                capabilities=p["capabilities"],
                gpu_tier=p.get("gpu_tier", "none"),
                gpu_name=p.get("gpu_name"),
                vram_total_mb=p.get("vram_total_mb", 0),
                available_models=p.get("available_models", []),
                version=p.get("version", ""),
                last_seen=p.get("last_seen", time.time()),
            ))

        return cls(identity, peers=peers, swarm_token=swarm_token)

    # ── Inflight tracking ───────────────────────────────────────────

    def record_dispatch(self, node_id: str) -> None:
        """Increment inflight count for a node (call when dispatching)."""
        self._inflight[node_id] = self._inflight.get(node_id, 0) + 1

    def record_completion(self, node_id: str) -> None:
        """Decrement inflight count for a node (call when result arrives)."""
        count = self._inflight.get(node_id, 0)
        if count > 0:
            self._inflight[node_id] = count - 1

    # ── Scoring functions ───────────────────────────────────────────

    def _score_peer_for_inference(self, peer: Peer) -> float:
        """Score a peer for inference tasks.

        Specialist nodes get +500, executive gets +200.  This naturally
        prefers specialist GPUs for small models while executive still
        wins on raw VRAM for large models.
        """
        score = 0.0
        if "gpu_inference" in peer.capabilities:
            score += 1000
        score += peer.vram_total_mb / 100
        if peer.node_tier == "specialist":
            score += 500
        elif peer.node_tier == "executive":
            score += 200
        if peer.is_stale:
            score -= 2000
        return score

    def _local_inference_score(self) -> float:
        """Score the local node as an inference candidate.

        Same formula as _score_peer_for_inference so scores are
        directly comparable.
        """
        score = 0.0
        if "gpu_inference" in self.identity.capabilities:
            score += 1000
        score += self.identity.vram_total_mb / 100
        if self.identity.node_tier == "specialist":
            score += 500
        elif self.identity.node_tier == "executive":
            score += 200
        if not self.identity.ollama_running:
            score -= 5000  # Can't do inference without Ollama
        score -= self._inflight.get(self.identity.node_id, 0) * 100
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

    def _score_peer_for_support(self, peer: Peer) -> float:
        """Score a peer for inference support tasks (gate, embedding).

        Prefers specialist > light > executive to protect the primary
        GPU node's VRAM from model swapping.
        """
        score = 0.0
        if peer.node_tier == "specialist":
            score += 1000  # Ideal: dedicated support GPU
        elif peer.node_tier == "light":
            score += 200   # Acceptable: CPU inference
        elif peer.node_tier == "executive":
            score -= 2000  # Avoid: protect primary GPU VRAM
        if "gpu_inference" in peer.capabilities:
            score += 500
        if peer.is_stale:
            score -= 5000
        score -= self._inflight.get(peer.node_id, 0) * 100
        return score

    def _local_support_score(self) -> float:
        """Score the local node for inference support tasks."""
        score = 0.0
        if self.identity.node_tier == "specialist":
            score += 1000
        elif self.identity.node_tier == "light":
            score += 200
        elif self.identity.node_tier == "executive":
            score -= 2000
        if "gpu_inference" in self.identity.capabilities:
            score += 500
        if not self.identity.ollama_running:
            score -= 5000
        score -= self._inflight.get(self.identity.node_id, 0) * 100
        return score

    # ── Candidate selection ─────────────────────────────────────────

    def _select_candidates(
        self,
        task_type: str,
        peers: list[Peer],
        use_health_check: bool = False,
    ) -> list[Peer]:
        """Return eligible peers ranked by score, filtering stale/unhealthy."""
        required_caps = TASK_ROUTING.get(task_type, ["cpu_worker"])

        eligible = []
        for p in peers:
            has_caps = any(cap in p.capabilities for cap in required_caps)
            stale = p.is_stale
            healthy = True
            if use_health_check and self.registry:
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

        if task_type in INFERENCE_SUPPORT_TASKS:
            eligible.sort(key=self._score_peer_for_support, reverse=True)
        elif task_type in INFERENCE_TASKS:
            eligible.sort(key=self._score_peer_for_inference, reverse=True)
        else:
            eligible.sort(key=self._score_peer_for_cpu_work, reverse=True)

        return eligible

    # ── Local capability check ──────────────────────────────────────

    def can_handle_locally(self, task_type: str) -> bool:
        """Check if this node can handle the given task type.

        For inference/embedding, having the capability isn't enough —
        Ollama must be running too.
        """
        required_caps = TASK_ROUTING.get(task_type, ["cpu_worker"])
        has_caps = any(cap in self.identity.capabilities for cap in required_caps)
        if not has_caps:
            return False
        if (task_type in INFERENCE_TASKS or task_type in INFERENCE_SUPPORT_TASKS) and not self.identity.ollama_running:
            return False
        return True

    # ── Core routing ────────────────────────────────────────────────

    def _resolve_internal(
        self,
        task_type: str,
        model: str | None,
        peers: list[Peer],
        use_health_check: bool,
    ) -> RouteResult | None:
        """Core routing logic shared by async and sync paths.

        Returns RouteResult, or None if no node can handle the task.
        """
        can_local = self.can_handle_locally(task_type)
        candidates = self._select_candidates(task_type, peers, use_health_check)

        local_ip = self.identity.lan_ip or "127.0.0.1"
        local_port = 7890

        def _local(reason: str, mdl: str = "", score: float = 0.0) -> RouteResult:
            return RouteResult(
                model=mdl or model or "",
                node_hostname=self.identity.hostname,
                node_ip=local_ip,
                node_port=local_port,
                is_local=True,
                reason=reason,
                task_type=task_type,
                score=score,
            )

        def _remote(peer: Peer, reason: str, mdl: str = "", score: float = 0.0) -> RouteResult:
            return RouteResult(
                model=mdl or model or "",
                node_hostname=peer.hostname,
                node_ip=peer.ip,
                node_port=peer.port,
                is_local=False,
                reason=reason,
                task_type=task_type,
                score=score,
                _peer=peer,
            )

        # --- Inference support tasks: protect executive VRAM ---
        if task_type in INFERENCE_SUPPORT_TASKS:
            requested_model = model or ""

            pool = candidates
            if requested_model:
                model_pool = [
                    p for p in pool
                    if requested_model in getattr(p, "available_models", [])
                ]
                pool = model_pool if model_pool else []

            if pool:
                best = pool[0]
                best_score = self._score_peer_for_support(best)
                local_score = self._local_support_score()

                local_eligible = can_local
                if requested_model and can_local:
                    local_eligible = (
                        requested_model in self.identity.available_models
                    )

                if local_eligible and local_score >= best_score:
                    return _local("Local handles support task", score=local_score)

                logger.info(
                    f"🛡️ Routing {task_type} → {best.hostname} "
                    f"(protecting executive VRAM)"
                )
                return _remote(
                    best,
                    f"Support task → {best.hostname} (protecting executive VRAM)",
                    score=best_score,
                )

            # No suitable peer — local as last resort
            if can_local:
                return _local("Support task locally (no suitable peer)")

        # --- Distributable tasks: load-balance across peers + local ---
        elif task_type in DISTRIBUTABLE_TASKS:
            local_score = self._local_cpu_score()

            if candidates:
                best = candidates[0]  # Already sorted by score
                best_score = self._score_peer_for_cpu_work(best)
                if best_score > local_score:
                    return _remote(best, f"Distributed to {best.hostname}", score=best_score)

            if can_local:
                return _local("Local wins distribution scoring", score=local_score)

        # --- Inference tasks: prefer GPU nodes with the requested model ---
        elif task_type in INFERENCE_TASKS:
            requested_model = model or ""

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
                            return _local(
                                "Local scores higher for inference",
                                score=local_score,
                            )

                    logger.info(
                        f"🎯 Routing {task_type} → "
                        f"{best_peer.hostname} "
                        f"(has model {requested_model})"
                    )
                    return _remote(
                        best_peer,
                        f"Peer {best_peer.hostname} has model {requested_model}",
                        score=best_score,
                    )

                # No peer has the model — local fallback
                if can_local:
                    logger.warning(
                        f"⚠️ No peer has model '{requested_model}' "
                        f"— handling locally"
                    )
                    return _local(f"No peer has model '{requested_model}'")
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
                            return _local("Local GPU wins", score=local_score)
                    return _remote(
                        best_peer,
                        f"GPU peer {best_peer.hostname}",
                        score=best_score,
                    )
                if can_local:
                    return _local("Local inference (no GPU peers)")

        # --- Generic fallback ---
        if can_local:
            return _local("Handle locally")

        if candidates:
            return _remote(
                candidates[0],
                f"Can't handle locally, routing to {candidates[0].hostname}",
            )

        logger.warning(f"❌ No node available for {task_type}")
        return None

    # ── Entry points ────────────────────────────────────────────────

    async def resolve(self, task_type: str, model: str | None = None) -> RouteResult | None:
        """Async resolve using live registry (daemon path)."""
        if self.registry is None:
            raise RuntimeError("Router has no registry — use resolve_sync for cached peers")
        peers = await self.registry.get_all()
        return self._resolve_internal(task_type, model, peers, use_health_check=True)

    def resolve_sync(self, task_type: str, model: str | None = None) -> RouteResult | None:
        """Sync resolve using cached peers (pipeline path)."""
        return self._resolve_internal(task_type, model, self._peers, use_health_check=False)

    def build_routing_table(self, steps: list[dict]) -> list[RouteResult]:
        """Resolve all pipeline steps via TASK_MODEL_MAP preference order.

        Returns a list[RouteResult] aligned 1:1 with the input steps.
        Uses the same scoring as the daemon's orchestrator, so the
        routing table printed by the pipeline matches reality.
        """
        from mycoswarm.capabilities import TASK_MODEL_MAP

        # Build set of all available models across the swarm
        all_models: set[str] = set(self.identity.available_models)
        for p in self._peers:
            all_models.update(p.available_models)

        results: list[RouteResult] = []
        for step in steps:
            # 1) Explicit model override
            if step.get("model"):
                mdl = step["model"]
                result = self.resolve_sync("inference", model=mdl)
                if result:
                    result.model = mdl
                    result.task_type = step.get("task_type", "general")
                    results.append(result)
                    continue
                # Model not found anywhere — local fallback
                results.append(RouteResult(
                    model=mdl,
                    node_hostname=self.identity.hostname,
                    node_ip=self.identity.lan_ip or "127.0.0.1",
                    node_port=7890,
                    is_local=True,
                    reason="Explicit model, local fallback",
                    task_type=step.get("task_type", "general"),
                ))
                continue

            # 2) Resolve via task_type + TASK_MODEL_MAP preference order
            task_type = step.get("task_type", "general")
            task_config = TASK_MODEL_MAP.get(task_type, TASK_MODEL_MAP["general"])

            resolved = False
            for preferred in task_config["prefer_models"]:
                if preferred not in all_models:
                    continue  # Skip models nobody has
                result = self.resolve_sync("inference", model=preferred)
                if result:
                    result.model = preferred
                    result.task_type = task_type
                    results.append(result)
                    resolved = True
                    break

            if resolved:
                continue

            # 3) Fallback: largest available model
            for m in sorted(all_models, key=_model_size, reverse=True):
                result = self.resolve_sync("inference", model=m)
                if result:
                    result.model = m
                    result.task_type = task_type
                    results.append(result)
                    resolved = True
                    break

            if resolved:
                continue

            # 4) Last resort: first available, local
            first_model = next(iter(all_models), "")
            results.append(RouteResult(
                model=first_model,
                node_hostname=self.identity.hostname,
                node_ip=self.identity.lan_ip or "127.0.0.1",
                node_port=7890,
                is_local=True,
                reason="No matching model, local fallback",
                task_type=task_type,
            ))

        return results
