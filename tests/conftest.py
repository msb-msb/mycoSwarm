"""Shared fixtures for the mycoSwarm test suite."""

import time
import pytest

from mycoswarm.hardware import (
    CpuInfo,
    GpuInfo,
    HardwareProfile,
    MemoryInfo,
    DiskInfo,
    NetworkInterface,
)
from mycoswarm.api import TaskRequest, TaskResult, TaskStatus
from mycoswarm.discovery import Peer
from mycoswarm.node import NodeIdentity


# --- Factories ---


@pytest.fixture
def make_task():
    """Factory returning TaskRequest objects."""

    def _make(task_type: str = "ping", payload: dict | None = None, **kwargs):
        defaults = {
            "task_id": f"test-{task_type}-001",
            "task_type": task_type,
            "payload": payload or {},
            "source_node": "myco-testnode00",
        }
        defaults.update(kwargs)
        return TaskRequest(**defaults)

    return _make


@pytest.fixture
def make_peer():
    """Factory returning Peer objects with sensible defaults."""

    def _make(**kwargs):
        now = time.time()
        defaults = {
            "node_id": "myco-peer000001",
            "hostname": "test-peer",
            "ip": "192.168.1.50",
            "port": 7890,
            "node_tier": "light",
            "capabilities": ["cpu_worker", "file_processing", "code_execution"],
            "gpu_tier": "none",
            "gpu_name": None,
            "vram_total_mb": 0,
            "available_models": [],
            "version": "0.1.0",
            "first_seen": now,
            "last_seen": now,
        }
        defaults.update(kwargs)
        return Peer(**defaults)

    return _make


@pytest.fixture
def make_identity():
    """Factory returning NodeIdentity objects with sensible defaults."""

    def _make(**kwargs):
        defaults = {
            "node_id": "myco-testlocal0",
            "hostname": "test-local",
            "lan_ip": "192.168.1.10",
            "node_tier": "light",
            "capabilities": ["cpu_worker", "file_processing", "code_execution"],
            "gpu_tier": "none",
            "max_model_params_b": 0.0,
            "gpu_name": None,
            "vram_total_mb": 0,
            "vram_free_mb": 0,
            "ram_total_mb": 8192,
            "ram_available_mb": 4096,
            "cpu_model": "Intel i5-8500",
            "cpu_cores": 6,
            "disk_free_gb": 100.0,
            "ollama_running": False,
            "available_models": [],
            "timestamp": time.time(),
            "version": "0.1.0",
        }
        defaults.update(kwargs)
        return NodeIdentity(**defaults)

    return _make


# --- Pre-built hardware profiles ---


@pytest.fixture
def executive_profile():
    """HardwareProfile with RTX 3090, 64GB RAM, Ollama running."""
    return HardwareProfile(
        hostname="gpu-beast",
        gpus=[
            GpuInfo(
                index=0,
                name="NVIDIA GeForce RTX 3090",
                vram_total_mb=24576,
                vram_used_mb=1024,
                vram_free_mb=23552,
                temperature_c=45,
                utilization_pct=0,
                driver_version="535.183.01",
                cuda_version="12.2",
            )
        ],
        cpu=CpuInfo(
            model="AMD Ryzen 9 5950X",
            cores_physical=16,
            cores_logical=32,
            frequency_mhz=3400.0,
            architecture="x86_64",
        ),
        memory=MemoryInfo(
            total_mb=65536,
            available_mb=50000,
            used_mb=15536,
            percent_used=23.7,
        ),
        disks=[
            DiskInfo(
                path="/",
                total_gb=1000.0,
                used_gb=400.0,
                free_gb=600.0,
                percent_used=40.0,
            )
        ],
        network=[
            NetworkInterface(name="eth0", ipv4="192.168.1.100", mac="aa:bb:cc:dd:ee:ff"),
        ],
        ollama_running=True,
        ollama_models=[],
    )


@pytest.fixture
def light_profile():
    """HardwareProfile with no GPU, 8GB RAM, 4 cores, Ollama not running."""
    return HardwareProfile(
        hostname="thin-client",
        gpus=[],
        cpu=CpuInfo(
            model="Intel Core i5-8250U",
            cores_physical=4,
            cores_logical=4,
            frequency_mhz=1600.0,
            architecture="x86_64",
        ),
        memory=MemoryInfo(
            total_mb=8192,
            available_mb=4096,
            used_mb=4096,
            percent_used=50.0,
        ),
        disks=[
            DiskInfo(
                path="/",
                total_gb=256.0,
                used_gb=100.0,
                free_gb=156.0,
                percent_used=39.0,
            )
        ],
        network=[
            NetworkInterface(name="wlan0", ipv4="192.168.1.50", mac="11:22:33:44:55:66"),
        ],
        ollama_running=False,
        ollama_models=[],
    )
