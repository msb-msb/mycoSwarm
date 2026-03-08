"""Tests for Phase 31d: 8 C's Vital Signs."""

import pytest

from mycoswarm.vitals import Vitals, compute_vitals


class TestComputeVitals:
    def test_simple_chat(self):
        """answer/chat/all → high calm, moderate everything."""
        v = compute_vitals(intent={"tool": "answer", "mode": "chat", "scope": "all"})
        assert v.calm == 0.9
        assert v.curiosity == 0.5
        assert v.compassion == 0.5

    def test_rag_grounded(self):
        """High grounding → high clarity and confidence."""
        v = compute_vitals(
            grounding_score=0.95,
            source_count=4,
            doc_hits=3,
            session_hits=1,
            intent={"tool": "rag", "mode": "recall", "scope": "all"},
        )
        assert v.clarity >= 0.9
        assert v.confidence >= 0.7

    def test_low_grounding(self):
        """Low grounding → alerts fire."""
        v = compute_vitals(
            grounding_score=0.2,
            source_count=0,
            intent={"tool": "rag", "mode": "explore", "scope": "all"},
        )
        assert v.clarity < 0.4
        assert v.confidence < 0.4
        alerts = v.alerts()
        assert any("grounding" in a.lower() for a in alerts)
        assert any("certain" in a.lower() for a in alerts)

    def test_explore_mode(self):
        """Explore mode → high curiosity."""
        v = compute_vitals(intent={"tool": "rag", "mode": "explore", "scope": "all"})
        assert v.curiosity == 0.9

    def test_rich_memory(self):
        """Many session/fact hits → high connectedness/compassion."""
        v = compute_vitals(
            session_hits=4,
            fact_hits=3,
            procedure_hits=2,
        )
        assert v.compassion >= 0.89
        assert v.connectedness >= 0.8

    def test_said_dont_know(self):
        """Courage score high when honest."""
        v = compute_vitals(said_dont_know=True)
        assert v.courage == 0.9

    def test_procedures_used(self):
        """Procedure hits → creativity boost."""
        v_without = compute_vitals()
        v_with = compute_vitals(procedure_hits=2)
        assert v_with.creativity > v_without.creativity


class TestStatusBar:
    def test_format(self):
        """Correct compact format string."""
        v = Vitals(
            calm=0.8, clarity=0.9, curiosity=0.7, compassion=0.6,
            courage=0.7, creativity=0.5, connectedness=0.8, confidence=0.7,
        )
        bar = v.status_bar()
        assert "Ca:0.8" in bar
        assert "Cl:0.9" in bar
        assert "\U0001f9ed" in bar  # compass emoji

    def test_warning_marker(self):
        """Low scores get warning marker."""
        v = Vitals(
            calm=0.3, clarity=0.9, curiosity=0.7, compassion=0.6,
            courage=0.7, creativity=0.5, connectedness=0.8, confidence=0.7,
        )
        bar = v.status_bar()
        assert "\u26a0Ca:0.3" in bar


class TestAlerts:
    def test_below_threshold(self):
        """Alerts fire at < 0.4."""
        v = Vitals(
            calm=0.3, clarity=0.3, curiosity=0.7, compassion=0.6,
            courage=0.3, creativity=0.5, connectedness=0.3, confidence=0.3,
        )
        alerts = v.alerts()
        assert len(alerts) >= 3

    def test_above_threshold(self):
        """No alerts when healthy."""
        v = Vitals(
            calm=0.8, clarity=0.9, curiosity=0.7, compassion=0.6,
            courage=0.7, creativity=0.5, connectedness=0.8, confidence=0.7,
        )
        assert v.alerts() == []


class TestVitalsDict:
    def test_to_dict(self):
        """Serialization roundtrip."""
        v = Vitals(
            calm=0.8, clarity=0.9, curiosity=0.7, compassion=0.6,
            courage=0.7, creativity=0.5, connectedness=0.8, confidence=0.7,
        )
        d = v.to_dict()
        assert d["calm"] == 0.8
        assert d["clarity"] == 0.9
        assert len(d) == 8

    def test_overall(self):
        """Overall is average of all 8."""
        v = Vitals(
            calm=0.8, clarity=0.8, curiosity=0.8, compassion=0.8,
            courage=0.8, creativity=0.8, connectedness=0.8, confidence=0.8,
        )
        assert v.overall() == 0.8


class TestDetailedDisplay:
    def test_has_all_labels(self):
        """Detailed display includes all 8 C labels."""
        v = Vitals(
            calm=0.8, clarity=0.9, curiosity=0.7, compassion=0.6,
            courage=0.7, creativity=0.5, connectedness=0.8, confidence=0.7,
        )
        display = v.detailed_display("Monica")
        assert "Monica" in display
        assert "Calm:" in display
        assert "Clarity:" in display
        assert "Curiosity:" in display
        assert "Compassion:" in display
        assert "Courage:" in display
        assert "Creativity:" in display
        assert "Connectedness:" in display
        assert "Confidence:" in display
        assert "Overall:" in display


# ── Hardware body floor modifier tests (Phase 31c) ───────────────────────


class TestBodyFloorModifiers:
    """Verify hardware state applies floor modifiers to vitals."""

    def _base_body(self, **overrides):
        body = {
            "gpu_temp": 50.0,
            "vram_used_gb": 5.0,
            "vram_total_gb": 24.0,
            "vram_percent": 20.0,
            "nodes": [
                {"name": "Miu", "online": True, "gpu": "RTX 3090", "tier": "executive"},
                {"name": "naru", "online": True, "gpu": None, "tier": "light"},
            ],
        }
        body.update(overrides)
        return body

    # --- GPU temp → Calm ---

    def test_gpu_temp_above_85_floors_calm(self):
        v = compute_vitals(body_state=self._base_body(gpu_temp=90.0))
        assert v.calm <= 0.3

    def test_gpu_temp_75_to_85_floors_calm(self):
        v = compute_vitals(body_state=self._base_body(gpu_temp=80.0))
        assert v.calm <= 0.6

    def test_gpu_temp_exactly_75_floors_calm(self):
        v = compute_vitals(body_state=self._base_body(gpu_temp=75.0))
        assert v.calm <= 0.6

    def test_gpu_temp_below_75_no_change(self):
        v_no_body = compute_vitals()
        v_with_body = compute_vitals(body_state=self._base_body(gpu_temp=60.0))
        assert v_with_body.calm == v_no_body.calm

    def test_gpu_temp_none_no_change(self):
        v_no_body = compute_vitals()
        v_with_body = compute_vitals(body_state=self._base_body(gpu_temp=None))
        assert v_with_body.calm == v_no_body.calm

    # --- VRAM usage → Clarity ---

    def test_vram_above_90_floors_clarity(self):
        v = compute_vitals(body_state=self._base_body(vram_percent=95.0))
        assert v.clarity <= 0.4

    def test_vram_70_to_90_floors_clarity(self):
        v = compute_vitals(body_state=self._base_body(vram_percent=80.0))
        assert v.clarity <= 0.7

    def test_vram_exactly_70_floors_clarity(self):
        v = compute_vitals(body_state=self._base_body(vram_percent=70.0))
        assert v.clarity <= 0.7

    def test_vram_below_70_no_change(self):
        v_no_body = compute_vitals()
        v_with_body = compute_vitals(body_state=self._base_body(vram_percent=50.0))
        assert v_with_body.clarity == v_no_body.clarity

    def test_vram_none_no_change(self):
        v_no_body = compute_vitals()
        v_with_body = compute_vitals(body_state=self._base_body(vram_percent=None))
        assert v_with_body.clarity == v_no_body.clarity

    # --- Node ratio → Connectedness ---

    def test_all_online_no_change(self):
        v_no_body = compute_vitals()
        body = self._base_body()
        v_with_body = compute_vitals(body_state=body)
        assert v_with_body.connectedness == v_no_body.connectedness

    def test_one_offline_floors_connectedness(self):
        body = self._base_body(nodes=[
            {"name": "Miu", "online": True},
            {"name": "naru", "online": False},
            {"name": "boa", "online": True},
        ])
        v = compute_vitals(body_state=body)
        assert v.connectedness <= 0.7

    def test_two_offline_floors_connectedness(self):
        body = self._base_body(nodes=[
            {"name": "Miu", "online": True},
            {"name": "naru", "online": False},
            {"name": "boa", "online": False},
        ])
        v = compute_vitals(body_state=body)
        assert v.connectedness <= 0.5

    def test_empty_nodes_no_change(self):
        v_no_body = compute_vitals()
        v_with_body = compute_vitals(body_state=self._base_body(nodes=[]))
        assert v_with_body.connectedness == v_no_body.connectedness

    # --- Floor behavior: never raises ---

    def test_floor_never_raises_calm(self):
        """If calm is already below the floor, it stays low."""
        # web_and_rag + long response → calm ~0.4
        v = compute_vitals(
            intent={"tool": "web_and_rag", "mode": "explore"},
            response_tokens=3000,
            body_state=self._base_body(gpu_temp=80.0),  # floor at 0.6
        )
        # Calm was already pulled below 0.6 by the pipeline signals
        assert v.calm <= 0.6  # floor doesn't raise it

    def test_floor_never_raises_clarity(self):
        """If clarity is already 0.3, VRAM floor at 0.7 doesn't raise it."""
        v = compute_vitals(
            grounding_score=0.2,
            body_state=self._base_body(vram_percent=80.0),  # floor at 0.7
        )
        assert v.clarity <= 0.2  # stays at grounding_score level

    # --- None body_state is no-op ---

    def test_none_body_state_is_noop(self):
        v1 = compute_vitals()
        v2 = compute_vitals(body_state=None)
        assert v1.calm == v2.calm
        assert v1.clarity == v2.clarity
        assert v1.connectedness == v2.connectedness
