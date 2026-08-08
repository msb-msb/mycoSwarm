"""mycoSwarm single-node mode — direct Ollama inference without the daemon.

Provides instant access to local Ollama models with zero setup.
No mDNS, no orchestrator, no API server — just detect hardware and talk to Ollama.
"""

import json
import logging
import re
import sys
import time

import httpx

from mycoswarm.hardware import detect_all, HardwareProfile
from mycoswarm.intent_rules import classify_fast

OLLAMA_BASE = "http://localhost:11434"
OLLAMA_TIMEOUT = 300.0

logger = logging.getLogger(__name__)


def _datetime_string() -> str:
    """Current date/time formatted for prompt injection."""
    from datetime import datetime

    now = datetime.now().astimezone()
    day = now.day
    hour = now.hour % 12 or 12
    return now.strftime(f"Current date and time: %A, %B {day}, %Y at {hour}:%M %p %Z")


def check_daemon(port: int = 7890) -> bool:
    """Return True if the mycoSwarm daemon is reachable on the given port."""
    profile = detect_all()
    ip = profile.lan_ip or "localhost"
    try:
        with httpx.Client(timeout=2) as client:
            resp = client.get(f"http://{ip}:{port}/health")
            return resp.status_code == 200
    except (httpx.ConnectError, httpx.TimeoutException):
        return False


def check_ollama() -> tuple[bool, list[str]]:
    """Check if Ollama is running and return (running, model_names)."""
    try:
        with httpx.Client(timeout=3) as client:
            resp = client.get(f"{OLLAMA_BASE}/api/tags")
            resp.raise_for_status()
            data = resp.json()
            models = [m["name"] for m in data.get("models", [])]
            return True, models
    except (httpx.ConnectError, httpx.TimeoutException, httpx.HTTPError):
        return False, []


def pick_model(models: list[str], prefer: str | None = None) -> str:
    """Pick the model to run.

    Precedence: explicit ``prefer`` > the ``monica_chat`` role binding (if
    installed) > the role's named fallback. No substring matching — the model is
    a declared binding, not an emergent property of Ollama's tag order.
    """
    if not models and not prefer:
        print("❌ No Ollama models found. Install one with: ollama pull gemma3:27b")
        sys.exit(1)
    from mycoswarm.bindings import resolve_model, unavailable_message
    model, how = resolve_model("monica_chat", models, override=prefer)
    if how == "unavailable":
        if prefer:
            print(f"❌ Model '{prefer}' is not installed on this node.")
            print(f"   Pull it with: ollama pull {prefer}")
        else:
            print(unavailable_message("monica_chat"))
        sys.exit(1)
    return model


def ask_direct(prompt: str, model: str) -> None:
    """Send a prompt directly to Ollama /api/generate and print the result."""
    datetime_line = _datetime_string()
    payload = {
        "model": model,
        "prompt": f"{datetime_line}\nAlways respond in English unless the user explicitly asks for another language.\n\n{prompt}",
        "options": {"temperature": 0.7, "num_predict": 2048},
        "stream": True,
    }

    start = time.time()
    tokens: list[str] = []
    eval_count = 0
    eval_duration = 0

    try:
        with httpx.Client(timeout=httpx.Timeout(5.0, read=OLLAMA_TIMEOUT)) as client:
            with client.stream("POST", f"{OLLAMA_BASE}/api/generate", json=payload) as resp:
                resp.raise_for_status()
                for line in resp.iter_lines():
                    if not line:
                        continue
                    try:
                        chunk = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    token = chunk.get("response", "")
                    if token:
                        print(token, end="", flush=True)
                        tokens.append(token)

                    if chunk.get("done"):
                        eval_count = chunk.get("eval_count", 0)
                        eval_duration = chunk.get("eval_duration", 0)

    except httpx.ConnectError:
        print("❌ Cannot connect to Ollama. Is it running? Start with: ollama serve")
        sys.exit(1)
    except httpx.TimeoutException:
        print("\n❌ Ollama timed out.")
        sys.exit(1)

    duration = time.time() - start
    tps = eval_count / (eval_duration / 1e9) if eval_duration else 0

    print(f"\n{'─' * 50}")
    print(f"  ⏱  {duration:.1f}s | {tps:.1f} tok/s | model: {model}")


_PAST_REFERENCE_RE = re.compile(
    r"(?i)\b(?:"
    r"we discussed|we talked about|you said|you told me"
    r"|we mentioned|remember when|what did we|did we discuss"
    r"|last time|earlier conversation|before.{0,20}we"
    r"|our conversation|our past|past discussions|what we.ve|our discussions|you suggested|you recommended|we decided"
    r")\b"
)

# Moved to intent_rules (both the solo AND daemon paths need it; previously only
# solo checked it, so date questions were still being shipped to a peer).
# Re-exported for backwards compatibility with existing importers.
from mycoswarm.intent_rules import _DATETIME_QUERY_RE  # noqa: E402


def detect_past_reference(query: str) -> bool:
    """Return True if the query references past conversations."""
    return bool(_PAST_REFERENCE_RE.search(query))


_EMBEDDING_ONLY = ("nomic-embed-text", "mxbai-embed", "all-minilm", "snowflake-arctic-embed")


def _is_embedding_model(name: str) -> bool:
    """Return True if the model name looks like an embedding-only model."""
    lower = name.lower()
    return any(pat in lower for pat in _EMBEDDING_ONLY)


def _pick_gate_model() -> str | None:
    """Pick the best available small model for gate tasks (classification, etc.).

    Order comes from bindings.GATE_MODEL_PREFERENCE — the single shared list, so
    this path and worker.py cannot drift apart again (they had: solo preferred
    gemma3:1b while worker preferred gemma3:4b, so the CLI and the daemon were
    classifying with different models).

    Falls back to first available non-embedding model, or None.
    """
    from mycoswarm.bindings import GATE_MODEL_PREFERENCE

    try:
        with httpx.Client(timeout=5) as client:
            resp = client.get(f"{OLLAMA_BASE}/api/tags")
            resp.raise_for_status()
            models = [m["name"] for m in resp.json().get("models", [])]
    except (httpx.ConnectError, httpx.TimeoutException, httpx.HTTPError):
        return None

    for pattern in GATE_MODEL_PREFERENCE:
        for m in models:
            if pattern in m and not _is_embedding_model(m):
                return m

    # Fall back to first non-embedding model
    for m in models:
        if not _is_embedding_model(m):
            return m
    return None


_INTENT_SYSTEM_PROMPT = (
    "You are an intent classifier. Analyze the user's message and respond "
    "with ONLY a JSON object, no other text.\n\n"
    '{"tool": "", "mode": "", "scope": ""}\n\n'
    "tool — what tools are needed:\n"
    "  answer: general knowledge, math, coding, creative, conversation, date/time questions (datetime is always available)\n"
    "  web_search: current/real-time info (news, prices, weather, sports) — NOT date/time\n"
    "  rag: user's documents, notes, library, or past conversations\n"
    "  web_and_rag: needs both web and user's documents\n\n"
    "mode — what kind of thinking:\n"
    '  recall: remembering something specific ("what did we...", "where is...", "what does X say...")\n'
    '  explore: open-ended research or brainstorming ("what are some...", "how might...")\n'
    '  execute: precise action ("fix this code", "write the function")\n'
    "  chat: casual conversation, greetings, small talk\n\n"
    "scope — where to search (only matters when tool is rag or web_and_rag):\n"
    "  session: past conversations, things we discussed\n"
    "  docs: user's document library, files, notes, stored documents. Use docs when the user references a specific file by name (e.g. PLAN.md, README.md)\n"
    "  facts: stored user preferences and facts\n"
    "  all: search everything\n\n"
    "IMPORTANT RULES:\n"
    "- Date/time/day questions are ALWAYS answer/chat/facts. The current date and time are already in the system prompt. NEVER use web_search for date or time.\n"
    "- Greetings and small talk are ALWAYS answer/chat/all.\n\n"
    "Choose web_and_rag when the user wants BOTH their personal context "
    "(past discussions, documents) AND current web information.\n\n"
    'Examples:\n'
    '{"tool": "answer", "mode": "chat", "scope": "facts"} — "what time is it?", "what\'s the date?", "what day is it?"\n'
    '{"tool": "answer", "mode": "chat", "scope": "all"} — "hello", "what is photosynthesis?"\n'
    '{"tool": "web_search", "mode": "explore", "scope": "all"} — "what are the latest nvidia announcements?"\n'
    '{"tool": "rag", "mode": "recall", "scope": "session"} — "what did we discuss about bees?"\n'
    '{"tool": "rag", "mode": "recall", "scope": "docs"} — "what does PLAN.md say about Phase 20?"\n'
    '{"tool": "web_and_rag", "mode": "explore", "scope": "all"} — "based on our past discussions and current web info, what GPU should I buy?"\n'
    '{"tool": "web_and_rag", "mode": "recall", "scope": "session"} — "we talked about trading strategies — search the web for the latest BTC news too"'
)

_INTENT_DEFAULT = {"tool": "answer", "mode": "chat", "scope": "all"}
_VALID_TOOLS = {"answer", "web_search", "rag", "web_and_rag"}
_VALID_MODES = {"recall", "explore", "execute", "chat"}
_VALID_SCOPES = {"session", "docs", "facts", "all"}


def _log_sanitize(field: str, bad, fallback: str, model: str, query: str) -> None:
    """Record that the intent sanitiser had to repair a field.

    DEBUG only — this fires on every malformed classification and would be very
    noisy at normal verbosity. But it must exist: without it a model emitting an
    illegal enum on half its inputs is indistinguishable from a healthy one,
    because the fallback often lands on the right answer anyway.
    """
    logger.debug(
        "🤔 intent sanitiser: %s=%r invalid (model=%s) → using %r | query=%r",
        field, bad, model, fallback, query[:60],
    )


def intent_classify(query: str, model: str | None = None) -> dict:
    """Classify user intent with structured output.

    Returns: {
        "tool": "answer" | "web_search" | "rag" | "web_and_rag",
        "mode": "recall" | "explore" | "execute" | "chat",
        "scope": "session" | "docs" | "facts" | "all"
    }
    Falls back to {"tool": "answer", "mode": "chat", "scope": "all"} on error.
    """
    # Deterministic rules first — an explicit search request or a "thanks" does
    # not need an LLM's opinion, and this runs before any model is even chosen.
    # See intent_rules for why each pattern is deliberately narrow.
    fast = classify_fast(query)
    if fast is not None:
        result, rule = fast
        return {**result, "_via": rule, "_model": None}

    if model is None:
        model = _pick_gate_model()
    if model is None:
        return {**_INTENT_DEFAULT, "_via": "no_model", "_model": None}

    try:
        with httpx.Client(timeout=httpx.Timeout(5.0, read=15.0)) as client:
            resp = client.post(
                f"{OLLAMA_BASE}/api/chat",
                json={
                    "model": model,
                    "messages": [
                        {"role": "system", "content": _INTENT_SYSTEM_PROMPT},
                        {"role": "user", "content": query},
                    ],
                    "options": {"temperature": 0.0, "num_predict": 100},
                    "stream": False,
                },
            )
            resp.raise_for_status()
            raw = resp.json().get("message", {}).get("content", "").strip()
    except (httpx.ConnectError, httpx.TimeoutException, httpx.HTTPError):
        return {**_INTENT_DEFAULT, "_via": "error", "_model": model}

    # Parse JSON from response (may contain markdown fences)
    raw = raw.strip("`").strip()
    if raw.startswith("json"):
        raw = raw[4:].strip()

    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        # Fallback: try to extract tool from raw text like the old classifier
        result = dict(_INTENT_DEFAULT)
        lower = raw.lower()
        for category in ("web_and_rag", "web_search", "rag", "answer"):
            if category in lower:
                result["tool"] = category
                break
        return {**result, "_via": "unparseable", "_model": model}

    # Validate and sanitize. Each rewrite is logged at DEBUG: the repair is
    # silent by design (an invalid tool becomes "answer", which for greetings is
    # correct by luck), so a model emitting garbage looks healthy in production.
    # gemma3:1b did exactly that on 51% of inputs and nothing surfaced it.
    result = dict(_INTENT_DEFAULT)
    tool = data.get("tool", "answer")
    if tool in _VALID_TOOLS:
        result["tool"] = tool
    else:
        _log_sanitize("tool", tool, result["tool"], model, query)
    mode = data.get("mode", "chat")
    if mode in _VALID_MODES:
        result["mode"] = mode
    else:
        _log_sanitize("mode", mode, result["mode"], model, query)
    scope = data.get("scope", "all")
    if scope in _VALID_SCOPES:
        result["scope"] = scope
    else:
        _log_sanitize("scope", scope, result["scope"], model, query)

    # Override: regex-detected past reference forces session scope
    if detect_past_reference(query):
        result["scope"] = "session"
    # Override: docs scope never needs web search
    if result["scope"] == "docs" and result["tool"] == "web_and_rag":
        result["tool"] = "rag"

    result["_via"] = "model"
    result["_model"] = model
    return result


def classify_query(query: str, model: str) -> str:
    """Classify a user query to determine what tools are needed.

    Returns one of: "answer", "web_search", "rag", "web_and_rag".
    Falls back to "answer" on any error.

    This is a backward-compatible wrapper around intent_classify().
    """
    result = intent_classify(query, model=model)
    return result["tool"]


def web_search_solo(query: str, max_results: int = 5) -> list[dict]:
    """Run a web search locally via DuckDuckGo. Returns list of result dicts."""
    try:
        from ddgs import DDGS
        raw = DDGS().text(query, max_results=max_results)
        return [
            {
                "title": r.get("title", ""),
                "url": r.get("href", ""),
                "snippet": r.get("body", ""),
            }
            for r in raw
        ]
    except Exception:
        return []


def generate_search_variants(query: str, max_variants: int = 3) -> list[str]:
    """Generate search query variants for fan-out web search.

    Returns up to max_variants distinct queries:
    1. Original query (always)
    2. Keyword-only version (strip filler words)
    3. Recency-focused version (append current year)

    These produce different DuckDuckGo result sets when dispatched
    to separate nodes in parallel.
    """
    variants = [query]

    # --- Variant 2: keyword extraction ---
    _FILLER = {
        "what", "is", "the", "a", "an", "of", "in", "for", "to", "and",
        "or", "how", "does", "do", "can", "will", "are", "was", "were",
        "been", "being", "have", "has", "had", "about", "with", "from",
        "this", "that", "these", "those", "it", "its", "my", "your",
        "our", "their", "me", "you", "we", "they", "he", "she",
        "tell", "show", "give", "find", "get", "let", "please",
        "could", "would", "should", "might", "may", "shall",
    }
    keywords = [w for w in query.split() if w.lower().strip("?.,!") not in _FILLER]
    if len(keywords) >= 2 and keywords != query.split():
        variants.append(" ".join(keywords))

    # --- Variant 3: recency-focused ---
    from datetime import datetime
    year = str(datetime.now().year)
    if year not in query:
        variants.append(f"{query} {year}")

    return variants[:max_variants]


def chat_stream(
    messages: list[dict], model: str
) -> tuple[str, dict]:
    """Stream a chat completion from Ollama. Returns (full_text, metrics)."""
    datetime_line = _datetime_string()

    # Inject datetime into messages
    msgs = list(messages)
    if msgs and msgs[0].get("role") == "system":
        msgs[0] = {
            **msgs[0],
            "content": f"{datetime_line}\nAlways respond in English unless the user explicitly asks for another language.\n\n{msgs[0]['content']}",
        }
    else:
        msgs.insert(0, {"role": "system", "content": f"{datetime_line}\nAlways respond in English unless the user explicitly asks for another language."})

    payload = {
        "model": model,
        "messages": msgs,
        "options": {"temperature": 0.7, "num_predict": 2048},
        "stream": True,
        "think": False,
    }

    start = time.time()
    tokens: list[str] = []
    eval_count = 0
    eval_duration = 0

    try:
        with httpx.Client(timeout=httpx.Timeout(5.0, read=OLLAMA_TIMEOUT)) as client:
            with client.stream("POST", f"{OLLAMA_BASE}/api/chat", json=payload) as resp:
                resp.raise_for_status()
                for line in resp.iter_lines():
                    if not line:
                        continue
                    try:
                        chunk = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    token = chunk.get("message", {}).get("content", "")
                    if token:
                        print(token, end="", flush=True)
                        tokens.append(token)

                    if chunk.get("done"):
                        eval_count = chunk.get("eval_count", 0)
                        eval_duration = chunk.get("eval_duration", 0)

    except httpx.ConnectError:
        print("❌ Cannot connect to Ollama. Is it running? Start with: ollama serve")
        return "", {}
    except httpx.TimeoutException:
        print("\n❌ Ollama timed out.")
        return "".join(tokens), {}
    except httpx.HTTPStatusError as e:
        # Belt-and-braces. resolve_model should never hand us an uninstalled model
        # now, but any other path that does must not surface as a raw traceback.
        code = e.response.status_code
        if code == 404:
            print(f"\n❌ Ollama does not have model '{model}'.")
            print(f"   Pull it with: ollama pull {model}")
        else:
            print(f"\n❌ Ollama returned HTTP {code} for model '{model}'.")
        return "".join(tokens), {}

    duration = time.time() - start
    tps = eval_count / (eval_duration / 1e9) if eval_duration else 0
    full_text = "".join(tokens)
    metrics = {
        "duration_seconds": round(duration, 2),
        "tokens_per_second": round(tps, 1),
        "model": model,
    }
    return full_text, metrics
