"""Tests for mycoswarm.capabilities — capability classification."""

import pytest

from mycoswarm.hardware import GpuInfo, CpuInfo, MemoryInfo, HardwareProfile, NetworkInterface, DiskInfo
from mycoswarm.capabilities import (
    Capability,
    GpuTier,
    NodeTier,
    classify_gpu,
    classify_node,
)


def test_executive_gets_gpu_inference(executive_profile):
    """24GB VRAM + Ollama running → gpu_inference in capabilities."""
    caps = classify_node(executive_profile)
    assert Capability.GPU_INFERENCE in caps.capabilities


def test_light_no_ollama_no_inference(light_profile):
    """No GPU, Ollama not running → cpu_worker yes, no inference caps."""
    caps = classify_node(light_profile)
    assert Capability.CPU_WORKER in caps.capabilities
    assert Capability.GPU_INFERENCE not in caps.capabilities
    assert Capability.CPU_INFERENCE not in caps.capabilities


def test_capabilities_include_file_processing(light_profile):
    """2+ core node → file_processing in capabilities."""
    caps = classify_node(light_profile)
    assert Capability.FILE_PROCESSING in caps.capabilities


def test_capabilities_include_code_execution(light_profile):
    """2+ core node → code_execution in capabilities."""
    caps = classify_node(light_profile)
    assert Capability.CODE_EXECUTION in caps.capabilities


@pytest.mark.parametrize(
    "vram_mb,expected_tier",
    [
        (2000, GpuTier.NONE),
        (4096, GpuTier.ENTRY),
        (8192, GpuTier.MID),
        (12288, GpuTier.MID),
        (16384, GpuTier.HIGH),
        (24576, GpuTier.ULTRA),
        (49152, GpuTier.ULTRA),
    ],
)
def test_classify_gpu_tiers(vram_mb, expected_tier):
    """Parametrized: VRAM → correct GpuTier."""
    gpu = GpuInfo(
        index=0, name="Test GPU", vram_total_mb=vram_mb,
        vram_used_mb=0, vram_free_mb=vram_mb,
    )
    assert classify_gpu(gpu) == expected_tier


def test_no_ollama_strips_inference():
    """Has GPU but no Ollama → inference capabilities removed."""
    profile = HardwareProfile(
        hostname="gpu-no-ollama",
        gpus=[
            GpuInfo(
                index=0, name="RTX 3090", vram_total_mb=24576,
                vram_used_mb=0, vram_free_mb=24576,
            )
        ],
        cpu=CpuInfo(
            model="AMD Ryzen 9", cores_physical=8, cores_logical=16,
            frequency_mhz=3400.0, architecture="x86_64",
        ),
        memory=MemoryInfo(total_mb=32768, available_mb=24000, used_mb=8768, percent_used=26.7),
        disks=[DiskInfo(path="/", total_gb=500, used_gb=200, free_gb=300, percent_used=40.0)],
        network=[NetworkInterface(name="eth0", ipv4="192.168.1.10")],
        ollama_running=False,
    )

    caps = classify_node(profile)
    assert Capability.GPU_INFERENCE not in caps.capabilities
    assert Capability.CPU_INFERENCE not in caps.capabilities
    # But cpu_worker should still be there
    assert Capability.CPU_WORKER in caps.capabilities
    # GPU tier classification still happens (it's about hardware, not software)
    assert caps.gpu_tier == GpuTier.ULTRA
