"""Regression tests for the context-window overflow that silently truncated
generation and then persisted the fragment as a complete answer.

Observed live: retrieval returned three fetched pages, tool_context reached
19,970 chars, the prompt filled 4,091 of a 4,096-token window, Ollama emitted
ONE token and stopped with done_reason="length". The user saw the single word
"Based" presented as the answer, and the next turn's history contained
[4] assistant: 'Based'.
"""

import pytest

from mycoswarm.worker import _build_ollama_request, _fit_num_ctx, _metrics_from_ollama


class TestFitNumCtx:
    def test_typical_turn_stays_small(self):
        """Don't pay for a huge window on ordinary turns."""
        assert _fit_num_ctx(1_400, 2048) == 4096

    def test_the_failing_turn_gets_room(self):
        """19,970 chars of tool_context must not land in a 4k window."""
        assert _fit_num_ctx(19_970, 2048) >= 16384

    def test_reserves_space_for_generation(self):
        """A window that merely fits the PROMPT still leaves nothing to answer
        with — that is precisely how one token got emitted."""
        chars = 10_000
        without = int(chars / 2.5)
        assert _fit_num_ctx(chars, 2048) >= without + 2048

    def test_estimate_is_pessimistic_not_optimistic(self):
        """Retrieved web text tokenizes ~2.5 chars/token, not 4. A 4:1 estimate
        picked 8k for the failing turn and it truncated anyway."""
        assert _fit_num_ctx(20_470, 2048) > 8192

    def test_caps_out_rather_than_growing_unbounded(self):
        assert _fit_num_ctx(10_000_000, 2048) == 32768

    def test_explicit_num_ctx_is_respected(self):
        msgs = [{"role": "user", "content": "hi"}]
        _ep, payload, _ = _build_ollama_request(
            {"model": "m", "messages": msgs, "num_ctx": 65536})
        assert payload["options"]["num_ctx"] == 65536

    def test_long_messages_raise_the_window(self):
        msgs = [{"role": "system", "content": "s" * 2_000},
                {"role": "user", "content": "u" * 30_000}]
        _ep, payload, _ = _build_ollama_request({"model": "m", "messages": msgs})
        assert payload["options"]["num_ctx"] >= 16384


class TestTruncationIsReported:
    def test_length_stop_is_flagged(self):
        m = _metrics_from_ollama(
            {"done_reason": "length", "eval_count": 1, "eval_duration": 1}, "gemma3:27b")
        assert m["truncated"] is True
        assert m["context_exhausted"] is True

    def test_normal_stop_is_not_flagged(self):
        m = _metrics_from_ollama(
            {"done_reason": "stop", "eval_count": 400, "eval_duration": 1}, "gemma3:27b")
        assert m["truncated"] is False
        assert m["context_exhausted"] is False

    def test_length_stop_after_a_long_answer_is_truncated_but_not_exhausted(self):
        """Hitting num_predict is a different thing from running out of context."""
        m = _metrics_from_ollama(
            {"done_reason": "length", "eval_count": 2048, "eval_duration": 1}, "x")
        assert m["truncated"] is True
        assert m["context_exhausted"] is False


class TestFragmentIsNotPersisted:
    """The data-loss half: a fragment must never enter session history."""

    def _guard(self):
        from mycoswarm.cli import _reject_truncated
        return _reject_truncated

    def test_truncated_generation_is_rejected(self, capsys):
        msgs = [{"role": "user", "content": "who won the Giants game last night?"}]
        rejected = self._guard()("Based", {"truncated": True, "context_exhausted": True}, msgs)
        assert rejected is True
        out = capsys.readouterr().out
        assert "cut off" in out.lower() or "incomplete" in out.lower()
        assert "not been saved" in out.lower()

    def test_normal_generation_is_kept(self):
        msgs = [{"role": "user", "content": "hello"}]
        assert self._guard()("A full answer.", {"truncated": False}, msgs) is False

    def test_missing_metrics_does_not_reject(self):
        """Absent metrics must not be read as truncation — that would discard
        good answers from any path that reports nothing."""
        msgs = [{"role": "user", "content": "hello"}]
        assert self._guard()("A full answer.", {}, msgs) is False
        assert self._guard()("A full answer.", None, msgs) is False
