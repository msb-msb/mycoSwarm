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


def build_body_prompt(daemon_url: str | None = None) -> str:
    """Format hardware state as a system prompt section.

    Returns a body-awareness prompt string, or empty string if no data.
    """
    state = get_body_state(daemon_url)

    # Check if we have any data at all
    has_gpu = state["gpu_temp"] is not None
    has_nodes = bool(state["nodes"])

    if not has_gpu and not has_nodes:
        return ""

    # Build node descriptions
    parts: list[str] = []
    if has_nodes:
        for i, node in enumerate(state["nodes"]):
            name = node["name"]
            gpu = node["gpu"]
            if i == 0 and has_gpu:
                # Local node with live GPU data
                gpu_short = gpu.replace("NVIDIA GeForce ", "") if gpu else "GPU"
                parts.append(
                    f"{name} ({gpu_short}, {state['gpu_temp']:.0f}°C, "
                    f"{state['vram_used_gb']}/{state['vram_total_gb']}GB VRAM)"
                )
            elif gpu:
                gpu_short = gpu.replace("NVIDIA GeForce ", "")
                status = "online" if node["online"] else "offline"
                parts.append(f"{name} ({gpu_short}, {status})")
            else:
                status = "online" if node["online"] else "offline"
                parts.append(f"{name} ({status})")
    elif has_gpu:
        # Solo mode — no daemon, but local GPU data available
        parts.append(
            f"local GPU ({state['gpu_temp']:.0f}°C, "
            f"{state['vram_used_gb']}/{state['vram_total_gb']}GB VRAM)"
        )

    node_line = " | ".join(parts)

    return (
        f"[Your body: {node_line}]\n\n"
        "You are aware of your body — temperature, memory pressure, "
        "which nodes are online. This awareness lives in the background, "
        "like a human's awareness of their own breathing. Do not report "
        "hardware stats unless asked directly, or unless something feels "
        "notably different from normal (overheating, a node going offline, "
        "memory pressure). Let your body inform your mood, not your words."
    )
