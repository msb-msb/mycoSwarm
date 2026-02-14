"""mycoSwarm single-node mode — direct Ollama inference without the daemon.

Provides instant access to local Ollama models with zero setup.
No mDNS, no orchestrator, no API server — just detect hardware and talk to Ollama.
"""

import json
import re
import sys
import time

import httpx

from mycoswarm.hardware import detect_all, HardwareProfile

OLLAMA_BASE = "http://localhost:11434"
OLLAMA_TIMEOUT = 300.0


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
    """Pick the best model from available Ollama models."""
    if prefer:
        return prefer
    if not models:
        print("❌ No Ollama models found. Install one with: ollama pull llama3.2")
        sys.exit(1)
    # Prefer a 14b+ model, fall back to first
    for m in models:
        if "14b" in m or "32b" in m or "27b" in m:
            return m
    return models[0]


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
    r"|our conversation|you suggested|you recommended|we decided"
    r")\b"
)


def detect_past_reference(query: str) -> bool:
    """Return True if the query references past conversations."""
    return bool(_PAST_REFERENCE_RE.search(query))


_EMBEDDING_ONLY = ("nomic-embed-text", "mxbai-embed", "all-minilm", "snowflake-arctic-embed")


def _is_embedding_model(name: str) -> bool:
    """Return True if the model name looks like an embedding-only model."""
    lower = name.lower()
    return any(pat in lower for pat in _EMBEDDING_ONLY)


def _pick_gate_model() -> str | None:
    """Pick the smallest available model for gate tasks (classification, etc.).

    Preference order: gemma3:1b > llama3.2:1b > gemma3:4b > llama3.2:3b.
    Falls back to first available non-embedding model, or None.
    """
    try:
        with httpx.Client(timeout=5) as client:
            resp = client.get(f"{OLLAMA_BASE}/api/tags")
            resp.raise_for_status()
            models = [m["name"] for m in resp.json().get("models", [])]
    except (httpx.ConnectError, httpx.TimeoutException, httpx.HTTPError):
        return None

    for pattern in ("gemma3:1b", "llama3.2:1b", "gemma3:4b", "llama3.2:3b"):
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
    "  answer: general knowledge, math, coding, creative, conversation\n"
    "  web_search: current/real-time info (news, prices, weather)\n"
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
    'Examples:\n{"tool": "rag", "mode": "recall", "scope": "session"} — "what did we discuss about bees?"\n{"tool": "rag", "mode": "recall", "scope": "docs"} — "what does PLAN.md say about Phase 20?"'
)

_INTENT_DEFAULT = {"tool": "answer", "mode": "chat", "scope": "all"}
_VALID_TOOLS = {"answer", "web_search", "rag", "web_and_rag"}
_VALID_MODES = {"recall", "explore", "execute", "chat"}
_VALID_SCOPES = {"session", "docs", "facts", "all"}


def intent_classify(query: str, model: str | None = None) -> dict:
    """Classify user intent with structured output.

    Returns: {
        "tool": "answer" | "web_search" | "rag" | "web_and_rag",
        "mode": "recall" | "explore" | "execute" | "chat",
        "scope": "session" | "docs" | "facts" | "all"
    }
    Falls back to {"tool": "answer", "mode": "chat", "scope": "all"} on error.
    """
    if model is None:
        model = _pick_gate_model()
    if model is None:
        return dict(_INTENT_DEFAULT)

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
        return dict(_INTENT_DEFAULT)

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
        return result

    # Validate and sanitize
    result = dict(_INTENT_DEFAULT)
    tool = data.get("tool", "answer")
    if tool in _VALID_TOOLS:
        result["tool"] = tool
    mode = data.get("mode", "chat")
    if mode in _VALID_MODES:
        result["mode"] = mode
    scope = data.get("scope", "all")
    if scope in _VALID_SCOPES:
        result["scope"] = scope

    # Override: regex-detected past reference forces session scope
    if detect_past_reference(query):
        result["scope"] = "session"
    # Override: docs scope never needs web search
    if result["scope"] == "docs" and result["tool"] == "web_and_rag":
        result["tool"] = "rag"

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

    duration = time.time() - start
    tps = eval_count / (eval_duration / 1e9) if eval_duration else 0
    full_text = "".join(tokens)
    metrics = {
        "duration_seconds": round(duration, 2),
        "tokens_per_second": round(tps, 1),
        "model": model,
    }
    return full_text, metrics
