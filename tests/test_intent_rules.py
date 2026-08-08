"""Tests for the deterministic intent short-circuits (intent_rules.py).

Bias throughout: a false positive costs an unnecessary web search; a false
NEGATIVE just defers to the model, which is the status quo. So the negative
cases below matter more than the positive ones — they are what stops the rules
from hijacking queries the model should decide.
"""

import pytest

from mycoswarm.intent_rules import classify_fast


def _tool(q):
    r = classify_fast(q)
    return None if r is None else r[0]["tool"]


def _rule(q):
    r = classify_fast(q)
    return None if r is None else r[1]


class TestAcceptanceCase:
    def test_the_hardware_store_query(self):
        """The reported failure. Explicit search verb + a place — this must
        never reach a model, let alone a remote node that can time out."""
        q = ("can you search for a hardware store near college and ashby "
             "in Berkeley, ca. please...")
        result, rule = classify_fast(q)
        assert result == {"tool": "web_search", "mode": "explore", "scope": "all"}
        assert rule == "web_search"


class TestWebSearch:
    @pytest.mark.parametrize("q", [
        "search for the best noise cancelling headphones",
        "search the web for rtx 5090 availability",
        "can you google that for me",
        "what's the weather in Berkeley tomorrow",
        "latest news on the port strike",
        "current price of copper",
        "who is the current mayor of Oakland",
        "find a pharmacy near me",
        "where is the nearest hardware store",
    ])
    def test_unambiguous_web_phrasing(self, q):
        assert _tool(q) == "web_search", q


class TestWebSearchIsNotTriggerHappy:
    """These must all DEFER — the model decides. A wrong short-circuit cannot
    be corrected downstream, so these are the important assertions."""

    @pytest.mark.parametrize("q", [
        "search my notes for the bee discussion",
        "look up what we discussed about trading",
        "search our past conversations for the GPU decision",
        "what does PLAN.md say about Phase 37",
        "can you find my notes on tai chi",
        "what do you think of Google as a company",
        "how does a search algorithm work",
        "explain the nearest neighbour algorithm",
    ])
    def test_personal_or_conceptual_defers_to_model(self, q):
        assert _tool(q) != "web_search", f"should not have short-circuited: {q}"


class TestSmallTalk:
    @pytest.mark.parametrize("q", [
        "hi", "hello Monica", "hey there", "good morning",
        "thanks!", "thank you Monica", "cheers",
        "ok", "cool", "got it", "makes sense", "that's interesting",
        "how are you?", "hi Monica! how are you?", "see you tomorrow",
    ])
    def test_greetings_and_acks(self, q):
        r = classify_fast(q)
        assert r is not None, q
        assert r[0] == {"tool": "answer", "mode": "chat", "scope": "all"}
        assert r[1] == "small_talk"

    @pytest.mark.parametrize("q", [
        "ok so what did we decide about the subnet drop-in last week",
        "thanks — now search for a hardware store near me",
        "hi, what does PLAN.md say about Phase 37",
        "good morning, can you summarise our last conversation",
        "yes, and what did you say about resonance earlier",
    ])
    def test_greeting_prefix_does_not_hijack_a_real_request(self, q):
        """Starting with 'ok' or 'hi' must not classify the whole message as
        small talk — the word cap plus whole-message anchoring guards this."""
        r = classify_fast(q)
        assert r is None or r[1] != "small_talk", f"wrongly matched small talk: {q}"


class TestDatetime:
    @pytest.mark.parametrize("q", [
        "what time is it?",
        "what's the date?",
        "what day is it today",
        "Monica, is it still morning?",
    ])
    def test_datetime_is_answer_chat_facts(self, q):
        r = classify_fast(q)
        assert r is not None, q
        assert r[0] == {"tool": "answer", "mode": "chat", "scope": "facts"}
        assert r[1] == "datetime"

    def test_datetime_beats_weather(self):
        """'what day is it' and 'what's the weather today' share vocabulary.
        The prompt is explicit that date/time must NEVER become a web search,
        so the datetime rule is checked first."""
        assert _rule("what day is it today") == "datetime"
        assert _rule("what's the weather today") == "web_search"


class TestDefersOnAmbiguity:
    @pytest.mark.parametrize("q", [
        "Monica, what does 'readiness' mean to you?",
        "explain how photosynthesis works",
        "what can't you do?",
        "I just spent a lot of time working on your parts.",
        "",
        "   ",
    ])
    def test_returns_none(self, q):
        assert classify_fast(q) is None, q


class TestBypassCorrectnessOnRealTraffic:
    """Every bypass on the real 89-example eval set must be CORRECT. A bad
    short-circuit is worse than a bad model: nothing downstream can fix it."""

    def test_no_incorrect_bypass_on_eval_set(self):
        import json
        import pathlib

        p = pathlib.Path("/media/minotaur/Storage_Disk_1/LLM_repo/intent_gold.json")
        if not p.exists():
            pytest.skip("eval set not present on this machine")
        gold = json.loads(p.read_text())
        wrong = []
        for g in gold:
            r = classify_fast(g["msg"])
            if r is not None and r[0] != g["gold"]:
                wrong.append((g["msg"][:60], r[0], g["gold"]))
        assert not wrong, f"incorrect bypasses: {wrong}"


class TestDatetimeInjectionReachesThePrompt:
    """FIX 1. Classifying a datetime query correctly is worthless if the real
    clock never reaches the model — it just invents a plausible time. Observed
    live: answered "9:48 PM" when the clock read 21:32."""

    def test_datetime_string_matches_system_clock(self):
        import datetime as dt

        from mycoswarm.solo import _datetime_string

        now = dt.datetime.now().astimezone()
        s = _datetime_string()
        assert now.strftime("%A") in s
        assert now.strftime("%Y") in s
        assert f"{now.minute:02d}" in s
        hour12 = now.hour % 12 or 12
        assert f"{hour12}:{now.minute:02d}" in s

    def test_worker_prepends_real_datetime_to_system_message(self):
        """The daemon inference path must carry the clock into the prompt."""
        import datetime as dt

        from mycoswarm.worker import _build_ollama_request

        msgs = [{"role": "system", "content": "You are Monica."},
                {"role": "user", "content": "what is the date and time?"}]
        _ep, payload, _is_chat = _build_ollama_request(
            {"model": "gemma3:27b", "messages": msgs})
        sysmsg = payload["messages"][0]["content"]
        assert "Current date and time" in sysmsg
        assert f"{dt.datetime.now().minute:02d}" in sysmsg
        assert "You are Monica." in sysmsg  # original content preserved

    def test_solo_chat_stream_prepends_datetime(self, monkeypatch):
        """The solo path must do the same."""
        import datetime as dt

        import mycoswarm.solo as solo

        captured = {}

        class _Resp:
            def raise_for_status(self): pass
            def iter_lines(self):
                return iter(['{"message":{"content":"ok"},"done":true}'])

        class _Stream:
            def __enter__(self): return _Resp()
            def __exit__(self, *a): return False

        class _Client:
            def __init__(self, *a, **kw): pass
            def __enter__(self): return self
            def __exit__(self, *a): return False
            def stream(self, method, url, json=None):
                captured["payload"] = json
                return _Stream()

        monkeypatch.setattr(solo.httpx, "Client", _Client)
        solo.chat_stream([{"role": "user", "content": "what time is it?"}], "gemma3:4b")
        sysmsg = captured["payload"]["messages"][0]["content"]
        assert "Current date and time" in sysmsg
        assert f"{dt.datetime.now().minute:02d}" in sysmsg
