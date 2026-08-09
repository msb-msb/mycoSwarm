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
