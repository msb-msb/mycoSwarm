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
