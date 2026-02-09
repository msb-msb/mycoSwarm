"""Tests for mycoswarm.hardware — hardware detection."""

from unittest.mock import patch, MagicMock
from collections import namedtuple

from mycoswarm.hardware import (
    HardwareProfile,
    GpuInfo,
    CpuInfo,
    MemoryInfo,
    NetworkInterface,
    detect_gpus,
    detect_all,
)
from mycoswarm.capabilities import classify_node, NodeTier


def test_detect_all_returns_hardware_profile():
    """Mock psutil/subprocess, verify detect_all returns HardwareProfile with fields populated."""
    VirtualMemory = namedtuple("svmem", ["total", "available", "used", "percent"])
    CpuFreq = namedtuple("scpufreq", ["current", "min", "max"])

    mock_mem = VirtualMemory(
        total=16 * 1024**3, available=8 * 1024**3, used=8 * 1024**3, percent=50.0
    )
    mock_freq = CpuFreq(current=3600.0, min=800.0, max=4200.0)

    with (
        patch("mycoswarm.hardware.psutil") as mock_psutil,
        patch("mycoswarm.hardware.shutil.which", return_value=None),
        patch("mycoswarm.hardware.subprocess.run"),
        patch("mycoswarm.hardware.detect_ollama", return_value=(False, [])),
        patch("socket.gethostname", return_value="test-host"),
    ):
        mock_psutil.virtual_memory.return_value = mock_mem
        mock_psutil.cpu_freq.return_value = mock_freq
        mock_psutil.cpu_count.side_effect = lambda logical=True: 8 if logical else 4
        mock_psutil.disk_partitions.return_value = []
        mock_psutil.net_if_addrs.return_value = {}
        mock_psutil.net_if_stats.return_value = {}

        profile = detect_all()

    assert isinstance(profile, HardwareProfile)
    assert profile.hostname == "test-host"
    assert profile.cpu is not None
    assert profile.memory is not None
    assert profile.memory.total_mb == 16 * 1024


def test_classify_node_executive(executive_profile):
    """RTX 3090 (24GB VRAM) + Ollama → EXECUTIVE tier."""
    caps = classify_node(executive_profile)
    assert caps.node_tier == NodeTier.EXECUTIVE


def test_classify_node_light(light_profile):
    """No GPU, 8GB RAM, 4 cores → LIGHT tier."""
    caps = classify_node(light_profile)
    assert caps.node_tier == NodeTier.LIGHT


def test_detect_gpus_no_nvidia_smi():
    """If nvidia-smi is not found, detect_gpus returns empty list."""
    with patch("mycoswarm.hardware.shutil.which", return_value=None):
        gpus = detect_gpus()
    assert gpus == []


def test_hardware_profile_properties():
    """Test has_gpu, total_vram_mb, and lan_ip properties."""
    gpu = GpuInfo(
        index=0, name="RTX 4090", vram_total_mb=24576,
        vram_used_mb=0, vram_free_mb=24576,
    )
    profile = HardwareProfile(
        hostname="test",
        gpus=[gpu],
        network=[
            NetworkInterface(name="lo", ipv4="127.0.0.1", is_loopback=True),
            NetworkInterface(name="eth0", ipv4="10.0.0.5"),
        ],
    )

    assert profile.has_gpu is True
    assert profile.total_vram_mb == 24576
    assert profile.lan_ip == "10.0.0.5"

    # No GPU profile
    empty = HardwareProfile(hostname="empty", gpus=[], network=[])
    assert empty.has_gpu is False
    assert empty.total_vram_mb == 0
    assert empty.lan_ip is None
