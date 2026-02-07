"""Capability classification for mycoSwarm nodes.

Takes a HardwareProfile and determines what roles this node can fill
in the swarm. Capabilities are not mutually exclusive — a GPU node
can also be a coordinator and a storage node.
"""

from dataclasses import dataclass, field
from enum import Enum

from mycoswarm.hardware import HardwareProfile, GpuInfo


class Capability(str, Enum):
    """What a node can do."""

    GPU_INFERENCE = "gpu_inference"  # Can run LLM models on GPU
    CPU_INFERENCE = "cpu_inference"  # Can run tiny models on CPU
    CPU_WORKER = "cpu_worker"  # Web scraping, doc processing, parsing
    STORAGE = "storage"  # File serving, vector DB, artifacts
    COORDINATOR = "coordinator"  # Discovery, routing, orchestration
    EDGE = "edge"  # Audio, sensors, mobile (detected differently)


class GpuTier(str, Enum):
    """GPU classification by VRAM."""

    NONE = "none"
    ENTRY = "entry"  # 4-6 GB  — tiny models, image gen with limits
    MID = "mid"  # 8-12 GB — 7B-14B models
    HIGH = "high"  # 16-24 GB — 14B-32B models
    ULTRA = "ultra"  # 24+ GB — 32B+ models, multiple models loaded


class NodeTier(str, Enum):
    """Overall node classification."""

    EDGE = "edge"  # Minimal: RPi, Arduino, phone
    LIGHT = "light"  # CPU-only: ThinkCentre, old laptop
    WORKER = "worker"  # Entry GPU or strong CPU
    SPECIALIST = "specialist"  # Mid-high GPU: 3060, 3070, 4060
    EXECUTIVE = "executive"  # High-ultra GPU: 3090, 4090, A6000


@dataclass
class NodeCapabilities:
    """Complete capability assessment for a node."""

    capabilities: list[Capability] = field(default_factory=list)
    gpu_tier: GpuTier = GpuTier.NONE
    node_tier: NodeTier = NodeTier.LIGHT
    max_model_params_b: float = 0.0  # Largest model this node can run (billions)
    recommended_models: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def can_infer(self) -> bool:
        return (
            Capability.GPU_INFERENCE in self.capabilities
            or Capability.CPU_INFERENCE in self.capabilities
        )


def classify_gpu(gpu: GpuInfo) -> GpuTier:
    """Classify a GPU by its VRAM."""
    vram = gpu.vram_total_mb

    if vram >= 20_000:
        return GpuTier.ULTRA
    elif vram >= 14_000:
        return GpuTier.HIGH
    elif vram >= 7_000:
        return GpuTier.MID
    elif vram >= 3_500:
        return GpuTier.ENTRY
    else:
        return GpuTier.NONE


def estimate_max_model_params(gpu_tier: GpuTier, vram_mb: int) -> float:
    """Estimate the largest model (in billions of params) this GPU can run.

    Based on practical testing with GGUF quantized models (Q4_K_M):
    - ~0.5-0.7 GB per billion parameters at Q4
    - Need ~1-2 GB overhead for KV cache and runtime
    """
    if gpu_tier == GpuTier.NONE:
        return 0.0

    usable_vram = max(0, vram_mb - 1500)  # Reserve for overhead
    # Q4_K_M: roughly 0.6 GB per billion params
    return round(usable_vram / 600, 1)


def recommend_models(gpu_tier: GpuTier, vram_mb: int) -> list[str]:
    """Suggest models based on GPU tier.

    These are starting points — see InsiderLLM.com for detailed per-VRAM guides.
    """
    if gpu_tier == GpuTier.ULTRA:  # 24GB+
        return [
            "qwen2.5:32b-instruct-q4_K_M",
            "qwen2.5:14b-instruct-q8_0",
            "gemma3:27b-q4_K_M",
            "deepseek-r1:14b",
            "codestral:22b-q4_K_M",
        ]
    elif gpu_tier == GpuTier.HIGH:  # 16-24GB
        return [
            "qwen2.5:14b-instruct-q4_K_M",
            "gemma3:12b",
            "deepseek-r1:14b-q4_K_M",
            "codestral:22b-q3_K_M",
        ]
    elif gpu_tier == GpuTier.MID:  # 8-12GB
        return [
            "qwen2.5:7b-instruct-q8_0",
            "qwen2.5:14b-instruct-q4_K_M",
            "gemma3:4b",
            "deepseek-r1:7b",
        ]
    elif gpu_tier == GpuTier.ENTRY:  # 4-6GB
        return [
            "qwen2.5:3b",
            "gemma3:1b",
            "deepseek-r1:1.5b",
        ]
    else:
        return []


def classify_node(profile: HardwareProfile) -> NodeCapabilities:
    """Classify a node's capabilities based on its hardware profile."""
    caps = NodeCapabilities()

    # --- GPU classification ---
    if profile.gpus:
        best_gpu = max(profile.gpus, key=lambda g: g.vram_total_mb)
        caps.gpu_tier = classify_gpu(best_gpu)

        if caps.gpu_tier != GpuTier.NONE:
            caps.capabilities.append(Capability.GPU_INFERENCE)
            caps.max_model_params_b = estimate_max_model_params(
                caps.gpu_tier, best_gpu.vram_total_mb
            )
            caps.recommended_models = recommend_models(
                caps.gpu_tier, best_gpu.vram_total_mb
            )
            caps.notes.append(
                f"GPU: {best_gpu.name} ({best_gpu.vram_total_mb}MB VRAM) "
                f"→ up to ~{caps.max_model_params_b}B params (Q4)"
            )

    # --- CPU inference ---
    if profile.cpu:
        ram_mb = profile.memory.total_mb if profile.memory else 0
        cores = profile.cpu.cores_logical

        # CPU inference viable if enough RAM and cores
        if ram_mb >= 4096 and cores >= 2:
            caps.capabilities.append(Capability.CPU_INFERENCE)
            # CPU can handle tiny models (1.5B-3B)
            if not caps.can_infer:
                caps.max_model_params_b = min(3.0, ram_mb / 2000)
                caps.notes.append(
                    f"CPU inference: {cores} threads, "
                    f"~{caps.max_model_params_b:.1f}B max (slow)"
                )

    # --- CPU worker (almost every node) ---
    if profile.cpu and profile.cpu.cores_logical >= 2:
        caps.capabilities.append(Capability.CPU_WORKER)

    # --- Storage ---
    if profile.disks:
        total_free_gb = sum(d.free_gb for d in profile.disks)
        if total_free_gb >= 5.0:
            caps.capabilities.append(Capability.STORAGE)
            caps.notes.append(f"Storage: {total_free_gb:.0f}GB free")

    # --- Coordinator (any node with network + reasonable CPU) ---
    if profile.lan_ip and profile.cpu and profile.cpu.cores_logical >= 2:
        caps.capabilities.append(Capability.COORDINATOR)

    # --- Ollama status ---
    if profile.ollama_running:
        model_count = len(profile.ollama_models)
        caps.notes.append(f"Ollama: running, {model_count} model(s) available")
    elif profile.has_gpu:
        caps.notes.append("Ollama: not detected (install for GPU inference)")

    # --- Overall node tier ---
    if caps.gpu_tier == GpuTier.ULTRA:
        caps.node_tier = NodeTier.EXECUTIVE
    elif caps.gpu_tier in (GpuTier.HIGH, GpuTier.MID):
        caps.node_tier = NodeTier.SPECIALIST
    elif caps.gpu_tier == GpuTier.ENTRY:
        caps.node_tier = NodeTier.WORKER
    elif profile.cpu and profile.cpu.cores_logical >= 4:
        caps.node_tier = NodeTier.LIGHT
    else:
        caps.node_tier = NodeTier.EDGE

    return caps
