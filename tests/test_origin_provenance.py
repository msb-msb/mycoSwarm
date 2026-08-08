"""Fact provenance: surfaced only when the question is about origin.

The bug, reproduced twice with identical mechanism:
  "do you remember coining the word 'weight'?"  → "You named it on February
      18th"  (wrong author, wrong date; date lifted from an unrelated session)
  "when did you first use 'allowance'?"         → "February 17th"  (the only
      date in context was an unrelated 2026-02-17 session hit)

Facts carry the DEFINITION but no provenance. Asked for the missing half she
welds on whatever date is nearest. Fabrication tracks the shape of the gap.
"""

from unittest.mock import patch

import pytest

from mycoswarm.intent_rules import extract_origin_term, is_origin_question


class TestOriginDetection:
    @pytest.mark.parametrize("q", [
        "when did you first use the word 'allowance'?",
        "do you remember coining the word 'weight'?",
        "who named 'echo'?",
        "where did the term 'overfunction' come from?",
        "what's the origin of 'readiness'?",
        "how long have you used the word 'resonance'?",
    ])
    def test_origin_questions_detected(self, q):
        assert is_origin_question(q) is True, q

    @pytest.mark.parametrize("q", [
        "what does allowance mean?",
        "what does 'echo' mean to you?",
        "tell me about weight",
        "hello Monica",
        "what time is it?",
        "what do you know about rushuna?",
    ])
    def test_meaning_questions_are_not_origin_questions(self, q):
        """The ordinary path must stay free — this is the whole point of
        detecting rather than always attaching provenance."""
        assert is_origin_question(q) is False, q

    @pytest.mark.parametrize("q,term", [
        ("when did you first use the word 'allowance'?", "allowance"),
        ('do you remember coining the word "weight"?', "weight"),
        ("where did the term overfunction come from?", "overfunction"),
    ])
    def test_term_extraction(self, q, term):
        assert extract_origin_term(q) == term


class TestProvenanceBlock:
    FACTS = [
        {"id": 1, "type": "identity", "added": "2026-03-08T14:01:17",
         "text": 'Monica\'s word for grief on important things: "weight" — loss sits heavier'},
        {"id": 2, "type": "identity", "added": "",
         "text": 'Monica\'s word for undated thing: "mystery" — no date recorded'},
    ]

    def _ctx(self, q, facts=None):
        from mycoswarm import memory
        with patch.object(memory, "load_facts", return_value=facts if facts is not None else self.FACTS):
            return memory.build_origin_context(q)

    def test_costs_nothing_on_ordinary_turns(self):
        """Zero standing cost is the design constraint — provenance for all 31
        facts on every turn is the opposite of this week's direction."""
        for q in ("what does weight mean?", "hello", "what do you know about rushuna?"):
            assert self._ctx(q) == "", q

    def test_includes_the_recorded_date(self):
        out = self._ctx("when did you first use the word 'weight'?")
        assert "2026-03-08" in out

    def test_attributes_authorship_to_her(self):
        """She credited the USER with her own coinage — the deference pattern.
        'Monica's word for X' means she authored it."""
        out = self._ctx("do you remember coining the word 'weight'?")
        assert "you (Monica) coined it" in out

    def test_instructs_against_borrowing_a_nearby_date(self):
        """The actual failure mode: a date from an unrelated retrieved session."""
        out = self._ctx("when did you first use the word 'weight'?")
        assert "Do not infer a date from any" in out

    def test_missing_date_says_so_rather_than_leaving_a_gap(self):
        """'Recorded at an unknown time' is something true she can say;
        silence is a hole she fills."""
        out = self._ctx("when did you first use the word 'mystery'?")
        assert "not recorded" in out.lower()

    def test_unknown_term_gets_an_explicit_no_record(self):
        out = self._ctx("when did you first use the word 'zarquon'?", facts=self.FACTS)
        assert "NO stored fact" in out
        assert "Do NOT supply a date" in out


class TestStoredDatesAreRealCoiningDates:
    """Provenance built on restore-date artifacts would be worse than none.
    Verified 2026-08-08: 'weight' added 2026-03-08 matches the grief session,
    'allowance' 2026-02-23 matches its coining. Dates span Feb-Jul on real
    session days, so they are not a single restore stamp."""

    def test_dates_are_not_all_identical(self):
        from mycoswarm.memory import load_facts
        facts = [f for f in load_facts() if f.get("type") == "identity"]
        if len(facts) < 5:
            pytest.skip("real fact store not present")
        days = {str(f.get("added"))[:10] for f in facts}
        assert len(days) > 1, "all identity facts share one date — restore artifact?"
