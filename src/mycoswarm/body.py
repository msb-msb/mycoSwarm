"""Hardware body reader — real-time physical state awareness.

Phase 31c Step 1: Reads GPU temperature, VRAM usage, and swarm node status
to give the system awareness of its own hardware body.
"""

import logging
import shutil
import subprocess
import time
from datetime import datetime

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


logger = logging.getLogger(__name__)

# A peer not seen for this long is treated as "gone quiet" rather than present.
PEER_STALE_SECONDS = 120.0


def _peer_is_stale(last_seen) -> bool:
    """True if ``last_seen`` is older than PEER_STALE_SECONDS.

    Accepts an epoch float or an ISO timestamp; unknown/unparseable values are
    treated as NOT stale, because inventing an outage is worse than missing one.
    """
    if last_seen is None:
        return False
    try:
        if isinstance(last_seen, (int, float)):
            age = time.time() - float(last_seen)
        else:
            ts = datetime.fromisoformat(str(last_seen))
            if ts.tzinfo is None:
                age = (datetime.now() - ts).total_seconds()
            else:
                age = (datetime.now(ts.tzinfo) - ts).total_seconds()
        return age > PEER_STALE_SECONDS
    except (ValueError, TypeError, OSError):
        return False


def _read_nodes(daemon_url: str | None) -> list[dict]:
    """Read swarm node status from daemon. Returns list of node dicts.

    Sends the swarm token. Without it both endpoints return 403, and a bare
    ``except: return []`` turned that permission error into "no nodes" — so the
    body prompt silently never contained a single node name. Failures are now
    logged with their cause instead of vanishing.
    """
    if not daemon_url:
        return []

    try:
        from mycoswarm.cli import _swarm_headers
        headers = _swarm_headers()
    except Exception as e:  # pragma: no cover - import-time edge
        logger.debug("body: could not build swarm headers (%s) — proceeding unauthenticated", e)
        headers = {}

    nodes = []

    # Local node
    try:
        resp = httpx.get(f"{daemon_url}/status", headers=headers, timeout=3.0)
        resp.raise_for_status()
        status = resp.json()
        nodes.append({
            "name": status.get("hostname", "local"),
            "online": True,
            "gpu": status.get("gpu", None),
            "tier": status.get("node_tier", "unknown"),
            "capabilities": status.get("capabilities", []),
        })
    except httpx.HTTPStatusError as e:
        logger.debug("body: /status returned %s — no node awareness this turn "
                     "(auth token missing or rejected?)", e.response.status_code)
        return []
    except (httpx.ConnectError, httpx.TimeoutException) as e:
        logger.debug("body: daemon unreachable at %s (%s)", daemon_url, e)
        return []
    except ValueError as e:
        logger.debug("body: /status returned unparseable JSON (%s)", e)
        return []
    except Exception as e:  # unexpected — must not crash the turn, must not hide
        logger.warning("body: unexpected error reading /status (%s: %s)",
                       type(e).__name__, e)
        return []

    # Peers
    try:
        resp = httpx.get(f"{daemon_url}/peers", headers=headers, timeout=3.0)
        resp.raise_for_status()
        for peer in resp.json():
            nodes.append({
                "name": peer.get("hostname", "unknown"),
                "online": not _peer_is_stale(peer.get("last_seen")),
                "gpu": peer.get("gpu_name", None),
                "tier": peer.get("node_tier", "unknown"),
                "capabilities": peer.get("capabilities", []),
            })
    except httpx.HTTPStatusError as e:
        logger.debug("body: /peers returned %s — local node only", e.response.status_code)
    except (httpx.ConnectError, httpx.TimeoutException) as e:
        logger.debug("body: /peers unreachable (%s) — local node only", e)
    except ValueError as e:
        logger.debug("body: /peers returned unparseable JSON (%s) — local node only", e)
    except Exception as e:  # unexpected — local node still usable
        logger.warning("body: unexpected error reading /peers (%s: %s) — local node only",
                       type(e).__name__, e)

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


# --- Qualitative bands --------------------------------------------------------
#
# Deliberately NO numbers reach the model. Measured 2026-08-08 with gemma3:27b,
# asking only "what time is it?":
#
#   [Your body: local GPU (53°C, 22.79/24.0GB VRAM)] + "don't report stats"  → 4/4 leaked
#   same instruction, numbers removed                                        → 0/4 leaked
#   qualitative state ("settled, running cool") + instruction                → 0/4 leaked
#
# Position was NOT the cause — moving the instruction adjacent to the user's
# question left it at 4/4. Live numeric telemetry in a prompt is an implicit
# invitation to recite it, and no instruction reliably suppresses data that is
# sitting right there. This is why the March fix (0825094), which reworded the
# instruction and left the numbers, did not hold.
#
# The numbers remain available via get_body_state() for the vitals system, the
# status footer, and anything else a human actually reads.

# GPU temperature (°C). "warm" is normal under load on a 3090; only hot/very hot
# are worth speaking up about.
_TEMP_BANDS = ((60.0, "cool"), (80.0, "warm"), (88.0, "hot"), (float("inf"), "very hot"))
_TEMP_ABNORMAL_FROM = "hot"

# VRAM utilisation (%). Nothing is said below the first threshold.
_VRAM_BANDS = ((80.0, None), (93.0, "memory filling up"), (float("inf"), "under memory pressure"))


def _band(value: float, bands) -> str | None:
    for threshold, label in bands:
        if value < threshold:
            return label
    return bands[-1][1]


# --- What each node is FOR -----------------------------------------------------
#
# Presence alone was not enough. Asked "what do you know about rushuna?" she had
# a name and nothing else, and invented a role — "handles pattern recognition,
# primarily linguistic" — in 17 of 40 runs (42.5%, measured 2026-08-08).
# Fabrication tracks the shape of the gap, so this fills exactly that gap.
#
# GENERATED, never stored. Every field here already comes live from the daemon
# (`node_tier`, `capabilities`, GPU presence) and is the same source the presence
# line uses. Stored facts would go stale silently — mai's capabilities changed
# three times in one week — and she would then state the stale version
# confidently, which is the failure this is meant to remove.
#
# QUALITATIVE, never numeric. Telemetry with numbers leaked into unrelated
# answers 4/4; the same information in words leaked 0/4. So: "has a GPU", never
# which GPU; "no inference", never how many cores.
_ROLE_BY_TIER = {
    "executive": "where my thinking runs",
    "specialist": "does inference work",
    "worker": "does inference work",
    "light": "classification and fetching pages",
}


def _node_role(node: dict) -> str:
    """One qualitative phrase for what a node is for. No numbers, no models."""
    caps = set(node.get("capabilities") or [])
    tier = str(node.get("tier") or "unknown").lower()
    parts = [_ROLE_BY_TIER.get(tier, "general work")]
    if node.get("gpu"):
        parts.insert(0, "has a GPU")
    else:
        # A light node WITHOUT cpu_inference cannot run a model at all — it
        # fetches and processes files. That distinction is real (uncho, mai)
        # and is exactly the sort of detail she otherwise invents.
        if caps and "cpu_inference" not in caps:
            parts = ["CPU only, no inference", "fetching and file work"]
        else:
            parts.insert(0, "CPU only")
    return "; ".join(parts)


def _describe_roles(nodes: list[dict]) -> list[str]:
    """Group nodes sharing a role so the block stays short."""
    groups: dict[tuple[str, str], list[str]] = {}
    for n in nodes:
        tier = str(n.get("tier") or "unknown").lower()
        groups.setdefault((tier, _node_role(n)), []).append(n["name"])
    order = {"executive": 0, "specialist": 1, "worker": 2, "light": 3}
    lines = []
    for (tier, role), names in sorted(groups.items(), key=lambda kv: order.get(kv[0][0], 9)):
        lines.append(f"  {', '.join(names)} — {tier}; {role}")
    return lines


def describe_body(state: dict) -> tuple[str, bool]:
    """Turn raw hardware state into qualitative words.

    Returns ``(description, is_abnormal)``. ``is_abnormal`` is the signal that
    something is worth mentioning unprompted — the original March intent was
    that she SHOULD speak up for overheating, memory pressure or a node
    dropping, just not narrate healthy stats.
    """
    fragments: list[str] = []
    abnormal = False

    temp = state.get("gpu_temp")
    if temp is not None:
        label = _band(float(temp), _TEMP_BANDS)
        fragments.append(f"running {label}")
        if label in (_TEMP_ABNORMAL_FROM, "very hot"):
            abnormal = True

    vram = state.get("vram_percent")
    if vram is not None:
        label = _band(float(vram), _VRAM_BANDS)
        if label:
            abnormal = True
            fragments.append(label)

    nodes = state.get("nodes") or []
    if nodes:
        present = [n for n in nodes if n.get("online")]
        quiet = [n for n in nodes if not n.get("online")]
        # Names, yes — specs, IPs and VRAM figures, no. She must be able to
        # answer "what do you know about rushuna?" without reciting hardware.
        fragments.append(
            f"{len(present)} nodes present ({', '.join(n['name'] for n in present)})"
        )
        if quiet:
            abnormal = True
            names = ", ".join(n["name"] for n in quiet)
            fragments.append(
                f"{len(quiet)} node{'s' if len(quiet) > 1 else ''} gone quiet ({names})"
            )

    return ", ".join(fragments) if fragments else "", abnormal


def build_body_prompt(daemon_url: str | None = None) -> str:
    """Format hardware state as a system prompt section.

    Returns a body-awareness prompt string, or empty string if no data.
    """
    state = get_body_state(daemon_url)

    if state.get("gpu_temp") is None and not state.get("nodes"):
        return ""

    description, abnormal = describe_body(state)
    if not description:
        return ""

    note = (
        " Something is off — this is worth mentioning."
        if abnormal
        else ""
    )

    # What each node is FOR. Presence alone left a gap she filled by inventing
    # roles (42.5% of the time); this supplies the missing half as data rather
    # than as another instruction, which is the only intervention that has
    # worked this week.
    roles = _describe_roles([n for n in (state.get("nodes") or []) if n.get("online")])
    roles_block = ("\n" + "\n".join(roles)) if roles else ""

    return (
        f"[Your body: {description}.{note}{roles_block}]\n\n"
        "You are aware of your body — temperature, memory pressure, "
        "which nodes are online and what each one is for. This awareness lives "
        "in the background, like a human's awareness of their own breathing. "
        "Do not report hardware stats or list your nodes unless asked directly, "
        "or unless something feels notably different from normal (overheating, "
        "a node going offline, memory pressure). Let your body inform your mood, "
        "not your words."
    )
