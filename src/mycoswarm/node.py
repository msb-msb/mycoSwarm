"""Node identity for mycoSwarm.

This is the data structure that represents a node in the swarm.
It combines hardware detection with capability classification
into a single announcement that other nodes can consume.
"""

import json
import time
import uuid
from dataclasses import dataclass, field, asdict

from mycoswarm import __version__
from mycoswarm.hardware import HardwareProfile, detect_all
from mycoswarm.capabilities import (
    Capability,
    GpuTier,
    NodeTier,
    NodeCapabilities,
    classify_node,
)


@dataclass
class NodeIdentity:
    """What this node tells the swarm about itself."""

    # Identity
    node_id: str  # Unique, persistent across restarts
    hostname: str
    lan_ip: str | None

    # Classification
    node_tier: str  # NodeTier value
    capabilities: list[str]  # List of Capability values
    gpu_tier: str  # GpuTier value

    # Capacity
    max_model_params_b: float
    gpu_name: str | None
    vram_total_mb: int
    vram_free_mb: int
    ram_total_mb: int
    ram_available_mb: int
    cpu_model: str
    cpu_cores: int
    disk_free_gb: float

    # Models
    ollama_running: bool
    available_models: list[str]

    # Status
    timestamp: float  # Unix timestamp of this announcement
    version: str  # mycoSwarm version
    notes: list[str] = field(default_factory=list)

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)

    def to_dict(self) -> dict:
        return asdict(self)


def _get_or_create_node_id(config_dir: str = "~/.config/mycoswarm") -> str:
    """Get persistent node ID, creating one if it doesn't exist."""
    from pathlib import Path

    config_path = Path(config_dir).expanduser()
    config_path.mkdir(parents=True, exist_ok=True)
    id_file = config_path / "node_id"

    if id_file.exists():
        return id_file.read_text().strip()

    node_id = f"myco-{uuid.uuid4().hex[:12]}"
    id_file.write_text(node_id)
    return node_id


def build_identity(
    profile: HardwareProfile | None = None,
    caps: NodeCapabilities | None = None,
) -> NodeIdentity:
    """Build a complete node identity from hardware detection."""
    if profile is None:
        profile = detect_all()
    if caps is None:
        caps = classify_node(profile)

    # Best GPU info
    best_gpu = max(profile.gpus, key=lambda g: g.vram_total_mb) if profile.gpus else None

    return NodeIdentity(
        node_id=_get_or_create_node_id(),
        hostname=profile.hostname,
        lan_ip=profile.lan_ip,
        node_tier=caps.node_tier.value,
        capabilities=[c.value for c in caps.capabilities],
        gpu_tier=caps.gpu_tier.value,
        max_model_params_b=caps.max_model_params_b,
        gpu_name=best_gpu.name if best_gpu else None,
        vram_total_mb=best_gpu.vram_total_mb if best_gpu else 0,
        vram_free_mb=best_gpu.vram_free_mb if best_gpu else 0,
        ram_total_mb=profile.memory.total_mb if profile.memory else 0,
        ram_available_mb=profile.memory.available_mb if profile.memory else 0,
        cpu_model=profile.cpu.model if profile.cpu else "Unknown",
        cpu_cores=profile.cpu.cores_logical if profile.cpu else 0,
        disk_free_gb=sum(d.free_gb for d in profile.disks),
        ollama_running=profile.ollama_running,
        available_models=[m.name for m in profile.ollama_models],
        timestamp=time.time(),
        version=__version__,
        notes=caps.notes,
    )
