"""Deterministic short-circuits for intent classification.

An explicit search request does not need an LLM's opinion, and neither does
"thanks". These rules run BEFORE the model — and, importantly, before the
daemon round-trip — so unambiguous traffic is classified in microseconds and
cannot be lost to a remote node timing out.

Precedent: ``solo._DATETIME_QUERY_RE`` already did exactly this for date/time,
but only on the solo path — the daemon path never checked it, so date questions
were still being shipped to a peer. These rules are shared by solo.py, worker.py
and cli.py so all three agree.

DESIGN BIAS: conservative. A false positive sends a query to the web that did
not need it (mildly wasteful, still answerable). A false negative just falls
through to the model, which is the status quo. So every pattern here is written
to match only phrasing that is unambiguous, and anything carrying a
personal-context signal is explicitly handed back to the model.

Evidence for the small-talk rule: intent-eval-2026-08-07 found that NO model
beat the 47.2% majority-class baseline of always answering answer/chat/all.
For the traffic where that constant is provably right, the model is pure cost.
"""

import re

__all__ = ["classify_fast", "FAST_RULES"]

# --- Signals that a query is about the USER'S OWN material, not the web -------
# If any of these appear, we never short-circuit to web_search — "search my
# notes for X" and "look up what we discussed" are RAG, not web.
_PERSONAL_CONTEXT_RE = re.compile(
    r"(?i)\b(?:"
    r"my |our |your |we discussed|we talked|you said|you told|remember when"
    r"|past (?:conversation|discussion)|earlier (?:you|we)"
    r"|\.md\b|PLAN\.md|README|my notes|my documents|my files|my library"
    r"|the library|stored fact|identity\.json"
    r")"
)

# --- Unambiguous web-search phrasing ------------------------------------------
_WEB_SEARCH_RE = re.compile(
    r"(?i)(?:"
    # explicit search verbs
    r"\bsearch (?:for|up)\b"
    r"|\bsearch (?:the )?(?:web|internet|online)\b"
    r"|\b(?:web|internet|online) search\b"
    r"|\bgoogle (?:it|that|for)\b|\bsearch google\b"
    r"|\blook (?:it |that |this |them )?up (?:online|on the web)\b"
    # real-time categories the prompt already names as web_search
    r"|\bweather (?:in|at|for|near|today|tomorrow|this week)\b"
    r"|\bwhat.s the weather\b"
    r"|\b(?:latest|recent|breaking) news\b|\bnews (?:on|about) \b"
    r"|\b(?:current|latest) (?:price|prices) (?:of|for)\b|\bstock price\b"
    r"|\bwho is the current \b"
    # "near me" is inherently a real-world lookup and needs no other signal.
    # Bare "nearest X" deliberately does NOT live here: it matched "explain the
    # nearest neighbour algorithm", a pure knowledge question. It now requires
    # a locational verb via _NEAR_PLACE_RE below.
    r"|\bnear me\b"
    r")"
)

# "find/where is X near <place>" — the failing hardware-store case. Needs two
# parts to co-occur, which makes it far less trigger-happy than "near" alone.
_NEAR_PLACE_RE = re.compile(
    r"(?i)\b(?:find|where(?:'s| is| are)?|locate|any|is there a?n?)\b.{0,60}"
    r"\b(?:near|nearest)\b"
)

# --- Small talk ---------------------------------------------------------------
# Anchored to the WHOLE message so "hi, can you search for X" does not match.
# A trailing name, punctuation or emoji is tolerated.
_GREETING_BODY = (
    r"(?:hi|hello|hey|yo|greetings"
    r"|good (?:morning|afternoon|evening|night)|morning|evening"
    r"|thanks(?: so much| a lot| again)?|thank you|thx|ty|cheers|much appreciated"
    r"|bye|goodbye|see you(?: later| tomorrow| soon)?|talk (?:to you )?later"
    r"|ok|okay|kk|cool|nice|great|awesome|lovely|excellent|perfect"
    r"|got it|i see|understood|makes sense|fair enough|no worries"
    r"|sure|right|indeed|true|yep|yeah|yes|nope|nah|no"
    r"|that.s (?:interesting|great|cool|nice|fair|true|good|helpful)"
    r"|how are you(?: doing| today)?|how.s it going|what.s up"
    r")"
)
_SMALL_TALK_RE = re.compile(
    r"(?i)^\W*(?:monica[,!\s]*)?"
    rf"{_GREETING_BODY}"
    rf"(?:[,!\.\s]+(?:monica|there|again))?"
    rf"(?:[,!\.\s]+{_GREETING_BODY})*"
    r"[\s\.,!?…🙂🙏👍]*$"
)
# Hard cap: anything longer than this is doing more than saying hello, even if
# it starts with one. Guards against "ok so what did we decide about X".
_SMALL_TALK_MAX_WORDS = 8

# --- Date/time (moved here from solo.py so BOTH paths get it) -----------------
_DATETIME_QUERY_RE = re.compile(
    r"(?i)\b(?:"
    r"what time|what date|what day"
    r"|what is the (?:date|time|day)"
    r"|what.s the (?:date|time|day)"
    r"|tell me the (?:date|time|day)"
    r"|current (?:date|time|day)"
    r"|today.s date"
    r"|date and time|time and date"
    r"|is it still (?:morning|afternoon|evening|night)"
    r")\b"
)

FAST_RULES = ("datetime", "small_talk", "web_search")


def classify_fast(query: str) -> tuple[dict, str] | None:
    """Classify ``query`` without a model, or return None to defer to the model.

    Returns ``({"tool":…, "mode":…, "scope":…}, rule_name)`` on a match.

    Order matters: date/time is checked first because "what's the weather
    today" and "what day is it" share vocabulary, and the prompt is explicit
    that date/time must never become a web search.
    """
    if not query or not query.strip():
        return None
    q = query.strip()

    # 1. date/time — the prompt states these are ALWAYS answer/chat/facts and
    #    must NEVER become a web search. Datetime is already in the system prompt.
    if _DATETIME_QUERY_RE.search(q):
        return {"tool": "answer", "mode": "chat", "scope": "facts"}, "datetime"

    # 2. small talk — the prompt states these are ALWAYS answer/chat/all, and
    #    the eval showed that constant beats every model on this traffic.
    if len(q.split()) <= _SMALL_TALK_MAX_WORDS and _SMALL_TALK_RE.match(q):
        return {"tool": "answer", "mode": "chat", "scope": "all"}, "small_talk"

    # 3. web search — but never when the query is about the user's own material.
    #    "search my notes" is RAG; only the model can weigh the ambiguous middle.
    if not _PERSONAL_CONTEXT_RE.search(q):
        if _WEB_SEARCH_RE.search(q) or _NEAR_PLACE_RE.search(q):
            return {"tool": "web_search", "mode": "explore", "scope": "all"}, "web_search"

    return None
