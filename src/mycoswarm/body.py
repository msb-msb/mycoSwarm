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
            "tasks": (status.get("tasks_pending", 0) or 0) + (status.get("tasks_active", 0) or 0),
            "cpu": status.get("cpu_usage_percent", 0.0) or 0.0,
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
                # needed by _attach_peer_activity; /peers is the only place
                # ip/port appear, and without them the fan-out KeyErrors into
                # the catch-all and every peer silently reads "unknown"
                "ip": peer.get("ip"),
                "port": peer.get("port"),
            })
        _attach_peer_activity(nodes[1:], headers)
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



def _attach_peer_activity(peers: list[dict], headers: dict) -> None:
    """Fill in tasks/cpu per peer. Mutates in place; never raises.

    /peers carries no activity data — task counts and CPU live only on each
    node's own /status — so this fans out. Concurrent with a tight per-request
    timeout: sequential would let one unreachable peer stall a chat turn for the
    full timeout, and this runs on EVERY turn. Measured 6/6 peers in 0.06s.

    A peer that does not answer keeps tasks/cpu None, which reads downstream as
    "unknown", not as "idle" — claiming a node is idle when we simply could not
    reach it is the same class of error this whole change exists to remove.
    """
    if not peers:
        return
    from concurrent.futures import ThreadPoolExecutor

    def one(p):
        if not p.get("ip") or not p.get("port"):
            logger.warning("body: peer %s has no ip/port — cannot read activity",
                           p.get("name"))
            return
        try:
            r = httpx.get(f"http://{p['ip']}:{p['port']}/status",
                          headers=headers, timeout=1.0)
            r.raise_for_status()
            d = r.json()
            p["tasks"] = (d.get("tasks_pending", 0) or 0) + (d.get("tasks_active", 0) or 0)
            p["cpu"] = d.get("cpu_usage_percent", 0.0) or 0.0
        except Exception as e:
            logger.debug("body: no activity from %s (%s) — reported as unknown",
                         p.get("hostname") or p.get("name"), type(e).__name__)

    with ThreadPoolExecutor(max_workers=8) as ex:
        list(ex.map(one, peers))


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


# --- Exertion: what each node is DOING right now --------------------------------
#
# The roles fix grounded IDENTITY (37.5% -> 2.5% invented roles) and thereby
# moved the gap rather than closing it. Asked "which node is doing the most work
# right now?" she reasoned from role to activity — "rushuna does inference,
# inference is happening, therefore rushuna is working hard" — while rushuna was
# idle. Plausible and wrong. There was no load data in context at all.
#
# This lives in the BODY layer, not the vitals. The 8 C's are IFS Self-energy
# qualities; exertion is no more psychological than temperature or memory
# pressure, which is why those FLOOR Calm rather than being vitals. Deliberately
# NOT wired into Calm yet: Calm is already a composite (response stability, tool
# complexity, GPU-temp floor) and a third input would make "I feel unsettled"
# untraceable to a cause.
#
# TWO SIGNALS, kept distinguishable, because they mean different things:
#   tasks_pending/active — work SHE dispatched. Answers "did I send anything
#       here?", which is what makes "rushuna is idle" derivable rather than
#       guessable, and is the direct fix for the observed failure.
#   cpu_usage_percent    — ambient exertion. A node at 40% because apt is
#       running is NOT Monica working, so it is never reported as her work.
#
# Neither is in /peers; both are on each node's own /status, so this fans out.
_EXERTION_WINDOW_S = 120.0     # samples older than this are dropped
_EXERTION_MAX_SAMPLES = 6
_exertion_history: dict[str, list[tuple[float, int, float]]] = {}


def _record_exertion(name: str, tasks: int, cpu: float) -> tuple[int, float]:
    """Add a sample and return the smoothed (tasks, cpu) for this node.

    Smoothing is required, not cosmetic: the body prompt is a snapshot taken at
    assembly time, and an instantaneous read is already stale by the time she
    answers. Window is 120s / 6 samples.

    MAX for tasks, MEAN for cpu — deliberately different. A task that starts and
    finishes between two turns would vanish from a mean, and "she dispatched work
    here recently" is the fact worth keeping; whereas CPU is spiky and a mean is
    what makes it meaningful.
    """
    now = time.time()
    hist = _exertion_history.setdefault(name, [])
    hist.append((now, tasks, cpu))
    fresh = [s for s in hist if now - s[0] <= _EXERTION_WINDOW_S][-_EXERTION_MAX_SAMPLES:]
    _exertion_history[name] = fresh
    return max(s[1] for s in fresh), sum(s[2] for s in fresh) / len(fresh)


# Dispatched-task bands. 1 task is real work; the queue only builds when she is
# sending faster than a node drains.
_TASK_BANDS = ((1, None), (3, "working"), (6, "working hard"), (10**9, "strained"))
# Ambient-CPU bands, used ONLY when no tasks are dispatched — otherwise her own
# work would get double-counted as ambient noise.
_CPU_BANDS = ((15.0, "idle"), (50.0, "ticking over"), (90.0, "busy with something else"),
              (10**9, "under heavy load of its own"))


def _exertion_phrase(tasks: int, cpu: float) -> tuple[str, bool]:
    """Qualitative exertion for one node. Returns (phrase, is_abnormal).

    No numbers, per the 2026-08-08 finding: numeric telemetry leaked into
    unrelated answers 4/4, the same information in words leaked 0/4.
    """
    if tasks >= 1:
        phrase = _band(float(tasks), _TASK_BANDS) or "working"
        return phrase, phrase == "strained"
    phrase = _band(float(cpu), _CPU_BANDS) or "idle"
    return phrase, phrase == "under heavy load of its own"


def describe_exertion(nodes: list[dict]) -> tuple[str, bool]:
    """One line naming who is doing what. Idle is stated, never implied.

    Silence about a node reads as absence of data, which is precisely what she
    fills in — so "everything else idle" is said out loud.
    """
    busy, idle, abnormal = [], [], False
    for n in nodes:
        if not n.get("online"):
            continue
        t, c = _record_exertion(n["name"], int(n.get("tasks") or 0),
                                float(n.get("cpu") or 0.0))
        phrase, abn = _exertion_phrase(t, c)
        abnormal = abnormal or abn
        (idle if phrase == "idle" else busy).append((n["name"], phrase))
    if not busy and not idle:
        return "", False
    if not busy:
        return "Right now: nothing dispatched anywhere; every node idle.", False
    parts = "; ".join(f"{n} {p}" for n, p in busy)
    tail = f"; everything else idle ({', '.join(n for n, _ in idle)})" if idle else ""
    return f"Right now: {parts}{tail}.", abnormal


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


# The exact state that went into the most recent prompt. The display layer reads
# THIS rather than calling get_body_state() again — a second call happens at a
# different moment in the turn and can disagree with what she was actually given,
# and a readout that can diverge from the prompt is worse than no readout. Same
# principle as exporting worker.estimate_tokens so the prompt-size line and the
# requested context window cannot contradict each other.
_LAST_RENDER: dict | None = None


def last_body_render() -> dict | None:
    """The state, bands and abnormal flag from the most recent prompt build.

    Returns None if no body prompt has been built this session.
    """
    return _LAST_RENDER


def format_body_status(numbers: bool = True, multiline: bool = False) -> str:
    """Human-facing view of the body state SHE WAS GIVEN.

    Numbers are shown here on purpose. The qualitative bands exist because
    numeric telemetry in her PROMPT leaked into unrelated answers 4/4; that
    constraint is about her context, not the operator's terminal, where the
    figure behind the band is what makes the band checkable.
    """
    r = _LAST_RENDER
    if not r:
        return "🫀 no body state yet"
    st = r["state"]
    warn = "⚠ " if r["abnormal"] else ""
    temp, vram = st.get("gpu_temp"), st.get("vram_percent")
    bits = []
    band = r["description"].split(",")[0].replace("running ", "")
    bits.append(f"{band} {temp:.0f}°C" if (numbers and temp is not None) else band)
    if vram is not None:
        mem = "mem " + (f"{vram:.0f}%" if numbers else "ok")
        bits.append(mem)
    online = [n for n in (st.get("nodes") or []) if n.get("online")]
    quiet = [n for n in (st.get("nodes") or []) if not n.get("online")]
    bits.append(f"{len(online)} nodes" + (f" (⚠{len(quiet)} quiet)" if quiet else ""))

    active = []
    for n in online:
        t, c = n.get("tasks"), n.get("cpu")
        if t is None and c is None:
            active.append(f"{n['name']} unknown")
            continue
        phrase, _ = _exertion_phrase(int(t or 0), float(c or 0.0))
        if phrase == "idle":
            continue
        detail = ""
        if numbers:
            detail = f" {c:.0f}%" if not t else f" {t}t"
        active.append(f"{n['name']} {phrase}{detail}")
    bits.append(", ".join(active) + (", rest idle" if active else "all idle"))

    if not multiline:
        return f"🫀 {warn}" + " | ".join(bits)

    lines = [f"🫀 {warn}Body — the state given to her this turn"]
    lines.append(f"   thermal   {band}" + (f"  ({temp:.0f}°C)" if numbers and temp is not None else ""))
    if vram is not None:
        lines.append(f"   memory    {'pressure' if vram >= 80 else 'ok'}" + (f"  ({vram:.0f}%)" if numbers else ""))
    lines.append(f"   nodes     {len(online)} present" + (f", {len(quiet)} gone quiet" if quiet else ""))
    for n in online:
        t, c = n.get("tasks"), n.get("cpu")
        if t is None and c is None:
            lines.append(f"     {n['name']:<9} unknown (unreachable)")
            continue
        phrase, _ = _exertion_phrase(int(t or 0), float(c or 0.0))
        num = f"  (cpu {c:.0f}%, {t} task{'s' if t != 1 else ''})" if numbers else ""
        lines.append(f"     {n['name']:<9} {phrase}{num}")
    lines.append(f"   prompt    {r['exertion'] or '(no exertion line)'}")
    return "\n".join(lines)


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

    # What each node is DOING. Roles grounded identity and moved the gap to
    # activity — she reasoned "rushuna does inference, so rushuna is busy" while
    # rushuna was idle. Idle is stated explicitly because silence about a node
    # reads as missing data, which is the thing she fills in.
    exertion, exertion_abnormal = describe_exertion(state.get("nodes") or [])
    exertion_block = ("\n  " + exertion) if exertion else ""
    if exertion_abnormal and not note:
        note = " Something is off — this is worth mentioning."

    global _LAST_RENDER
    _LAST_RENDER = {
        "state": state,
        "description": description,
        "abnormal": abnormal or exertion_abnormal,
        "exertion": exertion,
        "roles": roles,
    }

    return (
        f"[Your body: {description}.{note}{roles_block}{exertion_block}]\n\n"
        "You are aware of your body — temperature, memory pressure, "
        "which nodes are online and what each one is for. This awareness lives "
        "in the background, like a human's awareness of their own breathing. "
        "Do not report hardware stats or list your nodes unless asked directly, "
        "or unless something feels notably different from normal (overheating, "
        "a node going offline, memory pressure). Let your body inform your mood, "
        "not your words."
    )
