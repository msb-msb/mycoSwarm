"""Hardware body reader — real-time physical state awareness.

Phase 31c Step 1: Reads GPU temperature, VRAM usage, and swarm node status
to give the system awareness of its own hardware body.
"""

import shutil
import subprocess

import httpx


def _read_gpu() -> dict:
    """Read GPU temp and VRAM from nvidia-smi. Returns partial dict."""
    if not shutil.which("nvidia-smi"):
        return {"gpu_temp": None, "vram_used_gb": None, "vram_total_gb": None, "vram_percent": None}

    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=temperature.gpu,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode != 0:
            return {"gpu_temp": None, "vram_used_gb": None, "vram_total_gb": None, "vram_percent": None}

        # Parse first GPU line
        line = result.stdout.strip().split("\n")[0]
        parts = [p.strip() for p in line.split(",")]
        temp = float(parts[0])
        used_mb = float(parts[1])
        total_mb = float(parts[2])
        used_gb = round(used_mb / 1024, 2)
        total_gb = round(total_mb / 1024, 2)
        percent = round((used_mb / total_mb) * 100, 1) if total_mb > 0 else 0.0

        return {
            "gpu_temp": temp,
            "vram_used_gb": used_gb,
            "vram_total_gb": total_gb,
            "vram_percent": percent,
        }
    except Exception:
        return {"gpu_temp": None, "vram_used_gb": None, "vram_total_gb": None, "vram_percent": None}


def _read_nodes(daemon_url: str | None) -> list[dict]:
    """Read swarm node status from daemon. Returns list of node dicts."""
    if not daemon_url:
        return []

    nodes = []

    # Get local node status
    try:
        resp = httpx.get(f"{daemon_url}/status", timeout=3.0)
        resp.raise_for_status()
        status = resp.json()
        nodes.append({
            "name": status.get("hostname", "local"),
            "online": True,
            "gpu": status.get("gpu", None),
            "tier": status.get("node_tier", "unknown"),
        })
    except Exception:
        return []

    # Get peer nodes
    try:
        resp = httpx.get(f"{daemon_url}/peers", timeout=3.0)
        resp.raise_for_status()
        peers = resp.json()
        for peer in peers:
            nodes.append({
                "name": peer.get("hostname", "unknown"),
                "online": True,
                "gpu": peer.get("gpu_name", None),
                "tier": peer.get("node_tier", "unknown"),
            })
    except Exception:
        pass  # local node already added, peers just unavailable

    return nodes


def get_body_state(daemon_url: str | None = None) -> dict:
    """Read current hardware state for body awareness.

    Returns:
        {
            "gpu_temp": float or None,
            "vram_used_gb": float or None,
            "vram_total_gb": float or None,
            "vram_percent": float or None,
            "nodes": [
                {"name": "Miu", "online": True, "gpu": "RTX 3090", "tier": "executive"},
                ...
            ]
        }

    Never raises — always returns partial data on failure.
    """
    state = _read_gpu()
    state["nodes"] = _read_nodes(daemon_url)
    return state
