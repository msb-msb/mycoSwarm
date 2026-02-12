"""mycoSwarm single-node mode — direct Ollama inference without the daemon.

Provides instant access to local Ollama models with zero setup.
No mDNS, no orchestrator, no API server — just detect hardware and talk to Ollama.
"""

import json
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


def classify_query(query: str, model: str) -> str:
    """Classify a user query to determine what tools are needed.

    Returns one of: "answer", "web_search", "rag", "web_and_rag".
    Falls back to "answer" on any error.
    """
    try:
        with httpx.Client(timeout=httpx.Timeout(5.0, read=15.0)) as client:
            resp = client.post(
                f"{OLLAMA_BASE}/api/chat",
                json={
                    "model": model,
                    "messages": [
                        {
                            "role": "system",
                            "content": (
                                "Classify the user's message into exactly one category. "
                                "Respond with ONLY the category word, nothing else.\n\n"
                                "Categories:\n"
                                "- answer: can be answered from general knowledge, math, coding, creative writing, or conversation\n"
                                "- web_search: needs current/real-time info (news, weather, prices, sports, recent events, current status of things)\n"
                                "- rag: asks about the user's own documents, files, notes, stored library content, or past conversations\n"
                                "- web_and_rag: needs both web info and the user's documents"
                            ),
                        },
                        {"role": "user", "content": query},
                    ],
                    "options": {"temperature": 0.0, "num_predict": 20},
                    "stream": False,
                },
            )
            resp.raise_for_status()
            raw = resp.json().get("message", {}).get("content", "").strip().lower()
            # Extract the classification word from the response
            for category in ("web_and_rag", "web_search", "rag", "answer"):
                if category in raw:
                    return category
            return "answer"
    except (httpx.ConnectError, httpx.TimeoutException, httpx.HTTPError):
        return "answer"


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
