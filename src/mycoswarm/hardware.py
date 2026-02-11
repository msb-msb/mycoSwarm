"""Hardware detection for mycoSwarm nodes.

Detects GPU, CPU, RAM, disk, network interfaces, and available Ollama models.
No external dependencies beyond psutil and subprocess calls to nvidia-smi/ollama.
Cross-platform: Linux and macOS (Apple Silicon + Intel).
"""

import json
import platform
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

import psutil

IS_MACOS = sys.platform == "darwin"
IS_LINUX = sys.platform.startswith("linux")


@dataclass
class GpuInfo:
    index: int
    name: str
    vram_total_mb: int
    vram_used_mb: int
    vram_free_mb: int
    temperature_c: int | None = None
    utilization_pct: int | None = None
    driver_version: str = ""
    cuda_version: str = ""


@dataclass
class CpuInfo:
    model: str
    cores_physical: int
    cores_logical: int
    frequency_mhz: float
    architecture: str


@dataclass
class MemoryInfo:
    total_mb: int
    available_mb: int
    used_mb: int
    percent_used: float


@dataclass
class DiskInfo:
    path: str
    total_gb: float
    used_gb: float
    free_gb: float
    percent_used: float


@dataclass
class NetworkInterface:
    name: str
    ipv4: str | None = None
    mac: str | None = None
    is_loopback: bool = False


@dataclass
class OllamaModel:
    name: str
    size_mb: int
    parameter_size: str = ""
    quantization: str = ""
    family: str = ""


@dataclass
class HardwareProfile:
    """Complete hardware profile for a node."""
    hostname: str
    gpus: list[GpuInfo] = field(default_factory=list)
    cpu: CpuInfo | None = None
    memory: MemoryInfo | None = None
    disks: list[DiskInfo] = field(default_factory=list)
    network: list[NetworkInterface] = field(default_factory=list)
    ollama_models: list[OllamaModel] = field(default_factory=list)
    ollama_running: bool = False

    @property
    def total_vram_mb(self) -> int:
        return sum(g.vram_total_mb for g in self.gpus)

    @property
    def has_gpu(self) -> bool:
        return len(self.gpus) > 0

    @property
    def lan_ip(self) -> str | None:
        """First non-loopback IPv4 address."""
        for iface in self.network:
            if not iface.is_loopback and iface.ipv4:
                return iface.ipv4
        return None


def _detect_nvidia_gpus() -> list[GpuInfo]:
    """Detect NVIDIA GPUs via nvidia-smi (Linux)."""
    if not shutil.which("nvidia-smi"):
        return []

    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.total,memory.used,memory.free,"
                "temperature.gpu,utilization.gpu,driver_version",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode != 0:
            return []

        gpus = []
        for line in result.stdout.strip().split("\n"):
            if not line.strip():
                continue
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 8:
                continue

            gpus.append(
                GpuInfo(
                    index=int(parts[0]),
                    name=parts[1],
                    vram_total_mb=int(parts[2]),
                    vram_used_mb=int(parts[3]),
                    vram_free_mb=int(parts[4]),
                    temperature_c=int(parts[5]) if parts[5] != "[N/A]" else None,
                    utilization_pct=int(parts[6]) if parts[6] != "[N/A]" else None,
                    driver_version=parts[7],
                )
            )

        # Get CUDA version separately
        header_result = subprocess.run(
            ["nvidia-smi"], capture_output=True, text=True, timeout=5
        )
        if header_result.returncode == 0:
            for line in header_result.stdout.split("\n"):
                if "CUDA Version" in line:
                    try:
                        idx = line.index("CUDA Version:")
                        version = line[idx:].split()[2].strip("|").strip()
                        for gpu in gpus:
                            gpu.cuda_version = version
                    except (ValueError, IndexError):
                        pass
                    break

        return gpus

    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        return []


def _detect_apple_silicon_gpu() -> list[GpuInfo]:
    """Detect Apple Silicon GPU via system_profiler.

    Apple Silicon uses unified memory — the full system RAM is available
    to the GPU (Metal) and to Ollama for model inference.
    """
    if not IS_MACOS:
        return []

    # Check for Apple Silicon (arm64)
    if platform.machine() != "arm64":
        return []

    try:
        # Get chip name from system_profiler
        result = subprocess.run(
            ["system_profiler", "SPDisplaysDataType", "-json"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode != 0:
            return []

        data = json.loads(result.stdout)
        displays = data.get("SPDisplaysDataType", [])
        if not displays:
            return []

        chip_name = displays[0].get("sppci_model", "Apple Silicon GPU")

        # Unified memory: total system RAM is available for GPU/inference
        mem = psutil.virtual_memory()
        total_mb = mem.total // (1024 * 1024)
        available_mb = mem.available // (1024 * 1024)
        used_mb = total_mb - available_mb

        return [
            GpuInfo(
                index=0,
                name=chip_name,
                vram_total_mb=total_mb,
                vram_used_mb=used_mb,
                vram_free_mb=available_mb,
                driver_version="Metal",
            )
        ]

    except (subprocess.TimeoutExpired, json.JSONDecodeError, OSError):
        return []


def detect_gpus() -> list[GpuInfo]:
    """Detect GPUs — NVIDIA on Linux, Apple Silicon on macOS."""
    gpus = _detect_nvidia_gpus()
    if not gpus:
        gpus = _detect_apple_silicon_gpu()
    return gpus


def detect_cpu() -> CpuInfo:
    """Detect CPU information. Works on Linux and macOS."""
    model = "Unknown"

    if IS_LINUX:
        cpuinfo_path = Path("/proc/cpuinfo")
        if cpuinfo_path.exists():
            for line in cpuinfo_path.read_text().split("\n"):
                if line.startswith("model name"):
                    model = line.split(":", 1)[1].strip()
                    break
    elif IS_MACOS:
        try:
            result = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0 and result.stdout.strip():
                model = result.stdout.strip()
        except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
            pass

    try:
        freq = psutil.cpu_freq()
    except AttributeError:
        freq = None  # psutil.cpu_freq() unavailable on macOS ARM

    return CpuInfo(
        model=model,
        cores_physical=psutil.cpu_count(logical=False) or 1,
        cores_logical=psutil.cpu_count(logical=True) or 1,
        frequency_mhz=freq.current if freq else 0.0,
        architecture=platform.machine(),
    )


def detect_memory() -> MemoryInfo:
    """Detect system memory."""
    mem = psutil.virtual_memory()
    return MemoryInfo(
        total_mb=mem.total // (1024 * 1024),
        available_mb=mem.available // (1024 * 1024),
        used_mb=mem.used // (1024 * 1024),
        percent_used=mem.percent,
    )


def detect_disks() -> list[DiskInfo]:
    """Detect mounted disks."""
    disks = []
    seen_devices = set()

    # Mountpoints to skip per platform
    _skip = ("/snap/", "/boot/")
    if IS_MACOS:
        _skip = ("/System/Volumes/", "/private/var/vm")

    for partition in psutil.disk_partitions(all=False):
        if partition.device in seen_devices:
            continue
        if any(s in partition.mountpoint for s in _skip):
            continue
        seen_devices.add(partition.device)

        try:
            usage = psutil.disk_usage(partition.mountpoint)
            disks.append(
                DiskInfo(
                    path=partition.mountpoint,
                    total_gb=usage.total / (1024**3),
                    used_gb=usage.used / (1024**3),
                    free_gb=usage.free / (1024**3),
                    percent_used=usage.percent,
                )
            )
        except PermissionError:
            continue

    return disks


def detect_network() -> list[NetworkInterface]:
    """Detect network interfaces with IPv4 addresses."""
    interfaces = []
    addrs = psutil.net_if_addrs()
    stats = psutil.net_if_stats()

    # AF_PACKET on Linux, AF_LINK on macOS — both carry MAC addresses
    _link_families = {"AF_PACKET", "AF_LINK"}

    for name, addr_list in addrs.items():
        # Skip interfaces that are down
        if name in stats and not stats[name].isup:
            continue

        iface = NetworkInterface(name=name)

        for addr in addr_list:
            if addr.family.name == "AF_INET":
                iface.ipv4 = addr.address
                iface.is_loopback = addr.address.startswith("127.")
            elif addr.family.name in _link_families:
                iface.mac = addr.address

        if iface.ipv4:  # Only include interfaces with an IPv4 address
            interfaces.append(iface)

    return interfaces


def detect_ollama() -> tuple[bool, list[OllamaModel]]:
    """Check if Ollama is running and list available models."""
    if not shutil.which("ollama"):
        return False, []

    # Check if Ollama is responding
    try:
        result = subprocess.run(
            ["curl", "-s", "--max-time", "3", "http://localhost:11434/api/tags"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode != 0:
            return False, []

        data = json.loads(result.stdout)
        models = []

        for m in data.get("models", []):
            details = m.get("details", {})
            size_bytes = m.get("size", 0)

            models.append(
                OllamaModel(
                    name=m.get("name", "unknown"),
                    size_mb=size_bytes // (1024 * 1024),
                    parameter_size=details.get("parameter_size", ""),
                    quantization=details.get("quantization_level", ""),
                    family=details.get("family", ""),
                )
            )

        return True, models

    except (subprocess.TimeoutExpired, json.JSONDecodeError, OSError):
        return False, []


def detect_all() -> HardwareProfile:
    """Run full hardware detection and return a complete profile."""
    import socket

    ollama_running, ollama_models = detect_ollama()

    return HardwareProfile(
        hostname=socket.gethostname(),
        gpus=detect_gpus(),
        cpu=detect_cpu(),
        memory=detect_memory(),
        disks=detect_disks(),
        network=detect_network(),
        ollama_models=ollama_models,
        ollama_running=ollama_running,
    )
