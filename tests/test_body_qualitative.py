"""Tests for the qualitative body prompt and authenticated node discovery.

Why qualitative: measured 2026-08-08 with gemma3:27b on "what time is it?",
numeric telemetry in the prompt leaked 4/4 despite an explicit instruction not
to report it; with the numbers removed, 0/4. Position was not the cause —
moving the instruction next to the user's question left it at 4/4.
"""

from unittest.mock import MagicMock, patch

import pytest

from mycoswarm.body import (
    PEER_STALE_SECONDS,
    _peer_is_stale,
    build_body_prompt,
    describe_body,
)


def _state(temp=None, vram=None, nodes=None):
    return {"gpu_temp": temp, "vram_percent": vram, "vram_used_gb": 1.0,
            "vram_total_gb": 24.0, "nodes": nodes or []}


class TestNoNumbersReachTheModel:
    """The regression this exists to prevent."""

    @pytest.mark.parametrize("temp,vram", [(53.0, 94.9), (81.0, 12.0), (57.0, 14.9)])
    def test_prompt_contains_no_digits_from_telemetry(self, temp, vram):
        desc, _ = describe_body(_state(temp, vram))
        for forbidden in ("°C", "GB", "%", str(int(temp)), str(int(vram))):
            assert forbidden not in desc, f"{forbidden!r} leaked into: {desc}"

    def test_node_names_are_kept_but_specs_are_not(self):
        """She must be able to answer "what do you know about rushuna?" — names
        are fine, hardware specs and addresses are not."""
        nodes = [{"name": "Miu", "online": True, "gpu": "NVIDIA GeForce RTX 3090",
                  "tier": "executive"},
                 {"name": "rushuna", "online": True, "gpu": "NVIDIA GeForce RTX 3060",
                  "tier": "specialist"}]
        desc, _ = describe_body(_state(57.0, 15.0, nodes))
        assert "rushuna" in desc and "Miu" in desc
        assert "3090" not in desc and "3060" not in desc
        assert "RTX" not in desc


class TestQualitativeBands:
    @pytest.mark.parametrize("temp,word", [
        (35.0, "cool"), (59.9, "cool"), (60.0, "warm"), (79.9, "warm"),
        (80.0, "hot"), (87.9, "hot"), (88.0, "very hot"), (110.0, "very hot"),
    ])
    def test_temperature_bands(self, temp, word):
        desc, _ = describe_body(_state(temp))
        assert f"running {word}" in desc

    def test_normal_temperature_is_not_abnormal(self):
        for t in (35.0, 59.0, 70.0, 79.0):
            _, abnormal = describe_body(_state(t))
            assert abnormal is False, t

    def test_hot_is_abnormal_so_she_speaks_up(self):
        """The March intent: stay quiet when healthy, DO say something when
        overheating. Dropping the numbers must not drop that signal."""
        for t in (80.0, 95.0):
            desc, abnormal = describe_body(_state(t))
            assert abnormal is True, t

    def test_vram_silent_until_it_matters(self):
        desc, abnormal = describe_body(_state(50.0, 40.0))
        assert "memory" not in desc
        assert abnormal is False

    @pytest.mark.parametrize("vram,phrase", [
        (85.0, "memory filling up"), (99.0, "under memory pressure")])
    def test_vram_pressure_is_flagged(self, vram, phrase):
        desc, abnormal = describe_body(_state(50.0, vram))
        assert phrase in desc
        assert abnormal is True


class TestNodePresence:
    def test_all_present(self):
        nodes = [{"name": n, "online": True} for n in ("Miu", "rushuna", "luvia")]
        desc, abnormal = describe_body(_state(50.0, 10.0, nodes))
        assert "3 nodes present" in desc
        assert abnormal is False

    def test_a_quiet_node_is_abnormal(self):
        nodes = [{"name": "Miu", "online": True}, {"name": "boa", "online": False}]
        desc, abnormal = describe_body(_state(50.0, 10.0, nodes))
        assert "gone quiet" in desc and "boa" in desc
        assert abnormal is True

    def test_abnormal_state_is_marked_in_the_prompt(self):
        with patch("mycoswarm.body.get_body_state",
                   return_value=_state(92.0, 10.0, [{"name": "Miu", "online": True}])):
            p = build_body_prompt("http://x")
        assert "worth mentioning" in p
        assert "92" not in p

    def test_healthy_state_has_no_alarm_text(self):
        with patch("mycoswarm.body.get_body_state",
                   return_value=_state(50.0, 10.0, [{"name": "Miu", "online": True}])):
            p = build_body_prompt("http://x")
        assert "worth mentioning" not in p
        assert "breathing" in p          # background-awareness rule retained


class TestStaleness:
    def test_recent_peer_is_present(self):
        import time
        assert _peer_is_stale(time.time() - 5) is False

    def test_old_peer_is_quiet(self):
        import time
        assert _peer_is_stale(time.time() - PEER_STALE_SECONDS - 60) is True

    @pytest.mark.parametrize("bad", [None, "", "not-a-date", object()])
    def test_unknown_last_seen_is_not_treated_as_an_outage(self, bad):
        """Inventing an outage is worse than missing one."""
        assert _peer_is_stale(bad) is False


class TestAuthenticatedNodeRead:
    """The 403: /status and /peers were called without the swarm token, and a
    bare `except: return []` turned the permission error into "no nodes"."""

    def test_swarm_headers_are_sent(self):
        from mycoswarm import body

        captured = {}

        def fake_get(url, headers=None, timeout=None):
            captured.setdefault("headers", headers)
            r = MagicMock()
            r.raise_for_status = MagicMock()
            r.json.return_value = ({"hostname": "Miu"} if url.endswith("/status") else [])
            return r

        with patch.object(body.httpx, "get", side_effect=fake_get), \
             patch("mycoswarm.cli._swarm_headers", return_value={"X-Swarm-Token": "t"}):
            body._read_nodes("http://localhost:7890")
        assert captured["headers"] == {"X-Swarm-Token": "t"}

    def test_403_is_logged_not_swallowed(self, caplog):
        import logging

        import httpx as _httpx

        from mycoswarm import body

        resp = MagicMock()
        resp.status_code = 403
        err = _httpx.HTTPStatusError("forbidden", request=MagicMock(), response=resp)

        def boom(*a, **kw):
            raise err

        with patch.object(body.httpx, "get", side_effect=boom), \
             patch("mycoswarm.cli._swarm_headers", return_value={}), \
             caplog.at_level(logging.DEBUG, logger="mycoswarm.body"):
            out = body._read_nodes("http://localhost:7890")
        assert out == []
        # the point: the cause is visible, not swallowed into an empty list
        assert any("403" in r.getMessage() for r in caplog.records), caplog.text


class TestNodeRoles:
    """Role data closed the role-fabrication gap: 37.5% -> 2.5% unsupported
    claims on the rushuna probe (n=40 each, Fisher p=0.00012), with 0/48
    recitation on unrelated questions. Data, not another instruction."""

    NODES = [
        {"name": "Miu", "online": True, "tier": "executive",
         "gpu": "NVIDIA GeForce RTX 3090",
         "capabilities": ["gpu_inference", "cpu_inference", "cpu_worker"]},
        {"name": "rushuna", "online": True, "tier": "specialist",
         "gpu": "NVIDIA GeForce RTX 3060",
         "capabilities": ["gpu_inference", "cpu_inference", "cpu_worker"]},
        {"name": "boa", "online": True, "tier": "light", "gpu": None,
         "capabilities": ["cpu_inference", "cpu_worker", "file_processing"]},
        {"name": "naru", "online": True, "tier": "light", "gpu": None,
         "capabilities": ["cpu_inference", "cpu_worker", "file_processing"]},
        {"name": "uncho", "online": True, "tier": "light", "gpu": None,
         "capabilities": ["cpu_worker", "file_processing"]},
    ]

    def _prompt(self, nodes=None):
        with patch("mycoswarm.body.get_body_state",
                   return_value=_state(50.0, 20.0, nodes if nodes is not None else self.NODES)):
            return build_body_prompt("http://x")

    def test_roles_are_present(self):
        p = self._prompt()
        assert "executive" in p and "specialist" in p and "light" in p

    def test_no_numbers_leak_into_roles(self):
        """The 4/4-vs-0/4 finding: numeric telemetry gets recited, words do not.
        GPU presence, never which GPU. No core counts, no VRAM, no IPs."""
        p = self._prompt()
        role_block = p.split("]")[0]
        for banned in ("3090", "3060", "RTX", "GeForce", "NVIDIA", "GB", "°C"):
            assert banned not in role_block, banned

    def test_gpu_nodes_say_has_a_gpu(self):
        p = self._prompt()
        assert "has a GPU" in p

    def test_light_node_without_cpu_inference_is_distinguished(self):
        """uncho/mai genuinely lack cpu_inference — a real distinction she
        would otherwise invent. Derived from live capability flags, not
        assumed."""
        p = self._prompt()
        assert "no inference" in p
        # role lines are the indented ones; the presence line also names every
        # node, so select on the indent or the assertion tests the wrong line
        roles = [l for l in p.split("\n") if l.startswith("  ")]
        unc = [l for l in roles if "uncho" in l][0]
        assert "no inference" in unc
        other = [l for l in roles if "boa" in l][0]
        assert "no inference" not in other

    def test_nodes_sharing_a_role_are_grouped(self):
        """Seven nodes must not cost seven lines."""
        p = self._prompt()
        line = [l for l in p.split("\n") if "boa" in l][0]
        assert "naru" in line, "boa and naru share a role and should share a line"

    def test_offline_nodes_are_not_given_roles(self):
        nodes = self.NODES + [{"name": "ghost", "online": False, "tier": "light",
                               "gpu": None, "capabilities": ["cpu_worker"]}]
        p = self._prompt(nodes)
        role_lines = [l for l in p.split("\n") if l.startswith("  ")]
        assert not any("ghost" in l for l in role_lines)

    def test_roles_are_generated_not_stored(self):
        """Regression guard: role text must come from live daemon fields, so a
        tier change is reflected immediately rather than going stale."""
        changed = [dict(self.NODES[2], tier="specialist",
                        gpu="NVIDIA GeForce RTX 4090")]
        p = self._prompt(changed)
        assert "specialist" in p
        assert "4090" not in p


class TestExertion:
    """Activity signal. Roles grounded IDENTITY and moved the gap to ACTIVITY:
    asked which node was busiest she reasoned role->activity ("rushuna does
    inference, so rushuna is working hard") while rushuna was idle. Measured
    after this change: correctly-says-idle 0/20 -> 19/20, and 12/12 correct
    attribution with real load on luvia."""

    def setup_method(self):
        from mycoswarm import body
        body._exertion_history.clear()

    def _nodes(self, spec):
        return [{"name": n, "online": True, "tasks": t, "cpu": c} for n, t, c in spec]

    def test_all_idle_is_stated_explicitly(self):
        """Silence about a node reads as missing data — which is what she
        fills in. Idle must be sayable."""
        from mycoswarm.body import describe_exertion
        line, abn = describe_exertion(self._nodes([("Miu", 0, 1.0), ("rushuna", 0, 0.5)]))
        assert "every node idle" in line
        assert abn is False

    def test_dispatched_work_is_named(self):
        from mycoswarm.body import describe_exertion
        line, _ = describe_exertion(self._nodes([("Miu", 0, 1.0), ("boa", 2, 5.0)]))
        assert "boa working" in line
        assert "everything else idle" in line and "Miu" in line

    def test_ambient_load_is_not_reported_as_her_work(self):
        """A node at 60% because apt is running is NOT Monica working."""
        from mycoswarm.body import describe_exertion
        line, _ = describe_exertion(self._nodes([("luvia", 0, 60.0)]))
        assert "busy with something else" in line
        assert "working" not in line

    def test_no_numbers_in_the_exertion_line(self):
        from mycoswarm.body import describe_exertion
        line, _ = describe_exertion(self._nodes([("Miu", 3, 47.5), ("boa", 0, 88.2)]))
        for banned in ("47", "88", "%", "3 task"):
            assert banned not in line, banned

    def test_strain_sets_the_abnormal_flag(self):
        """Sustained strain must reach the same 'something is off' path as
        thermal and memory pressure."""
        from mycoswarm.body import describe_exertion
        _, abn = describe_exertion(self._nodes([("boa", 9, 5.0)]))
        assert abn is True

    def test_offline_nodes_are_omitted(self):
        from mycoswarm.body import describe_exertion
        nodes = self._nodes([("Miu", 0, 1.0)]) + [
            {"name": "ghost", "online": False, "tasks": 0, "cpu": 0.0}]
        line, _ = describe_exertion(nodes)
        assert "ghost" not in line

    def test_task_burst_survives_smoothing(self):
        """MAX over the window, not mean: a task that starts and finishes
        between two turns must not vanish."""
        from mycoswarm.body import describe_exertion
        describe_exertion(self._nodes([("boa", 4, 2.0)]))   # burst
        line, _ = describe_exertion(self._nodes([("boa", 0, 2.0)]))  # now quiet
        assert "idle" not in line.split("everything else")[0]
        assert "boa" in line

    def test_cpu_is_averaged_not_maxed(self):
        """CPU is spiky; a single transient must not pin the band high."""
        from mycoswarm.body import describe_exertion
        describe_exertion(self._nodes([("boa", 0, 95.0)]))  # one spike
        for _ in range(4):
            describe_exertion(self._nodes([("boa", 0, 1.0)]))
        line, abn = describe_exertion(self._nodes([("boa", 0, 1.0)]))
        # mean is ~16%, so it reads as mild activity — NOT the heavy band the
        # spike alone would have produced, and not an abnormal-state alarm
        assert "under heavy load" not in line
        assert abn is False

    def test_unreachable_peer_is_not_called_idle(self):
        """A peer we could not reach has tasks/cpu absent. Reporting it as idle
        would be the same class of error this change removes."""
        from mycoswarm.body import _attach_peer_activity
        peers = [{"name": "boa", "ip": None, "port": None}]
        _attach_peer_activity(peers, {})
        assert "tasks" not in peers[0] and "cpu" not in peers[0]


class TestBodyDisplay:
    """Operator-facing readout. The week's method has been comparing what she
    says against what is true; the ground truth was never on screen."""

    def _build(self, nodes, temp=57.0, vram=16.0):
        from mycoswarm import body
        with patch.object(body, "get_body_state",
                          return_value=_state(temp, vram, nodes)):
            body.build_body_prompt("http://x")

    NODES = [{"name": "Miu", "online": True, "tier": "executive",
              "gpu": "RTX 3090", "capabilities": ["gpu_inference"],
              "tasks": 0, "cpu": 23.0},
             {"name": "luvia", "online": True, "tier": "light", "gpu": None,
              "capabilities": ["cpu_inference"], "tasks": 0, "cpu": 75.0}]

    def test_display_cannot_diverge_from_the_prompt(self):
        """THE requirement. The readout must show the state she was GIVEN, not a
        fresh sample — a second reading happens at a different moment in the turn
        and could disagree, which is worse than showing nothing."""
        from mycoswarm import body

        self._build(self.NODES)
        snap = body.last_body_render()
        assert snap is not None
        # the exertion text in the snapshot is the exact string in the prompt
        assert "luvia" in snap["exertion"]
        # now the underlying hardware changes — the display must NOT follow it
        with patch.object(body, "get_body_state",
                          return_value=_state(90.0, 99.0, [])):
            out = body.format_body_status()
        assert "90" not in out and "99" not in out
        assert "57" in out

    def test_numbers_are_shown_to_the_operator(self):
        """Bands exist because numbers in HER prompt leaked 4/4. That constraint
        is about her context, not the terminal, where the figure is what makes
        the band checkable."""
        self._build(self.NODES)
        from mycoswarm.body import format_body_status
        out = format_body_status(numbers=True)
        assert "57°C" in out and "75" in out

    def test_numbers_can_be_suppressed(self):
        self._build(self.NODES)
        from mycoswarm.body import format_body_status
        out = format_body_status(numbers=False)
        assert "°C" not in out and "%" not in out

    def test_abnormal_is_visually_obvious(self):
        self._build(self.NODES, temp=91.0, vram=96.0)
        from mycoswarm.body import format_body_status
        assert "⚠" in format_body_status()

    def test_healthy_has_no_warning_glyph(self):
        self._build(self.NODES)
        from mycoswarm.body import format_body_status
        assert "⚠" not in format_body_status()

    def test_unreachable_peer_shows_unknown_not_idle(self):
        nodes = self.NODES + [{"name": "boa", "online": True, "tier": "light",
                               "gpu": None, "capabilities": [],
                               "tasks": None, "cpu": None}]
        self._build(nodes)
        from mycoswarm.body import format_body_status
        out = format_body_status()
        assert "boa unknown" in out

    def test_no_state_yet_is_handled(self):
        from mycoswarm import body
        body._LAST_RENDER = None
        assert "no body state yet" in body.format_body_status()

    def test_multiline_lists_every_node(self):
        self._build(self.NODES)
        from mycoswarm.body import format_body_status
        out = format_body_status(multiline=True)
        assert "Miu" in out and "luvia" in out
        assert "prompt" in out          # shows the exact line she received
