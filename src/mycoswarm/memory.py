"""mycoSwarm persistent memory — facts + session summaries.

Provides cross-session memory by injecting structured context into the
system prompt. Two layers:

  Layer 1 — Session summaries: auto-generated after each chat session
  Layer 2 — User facts: explicit /remember commands

Data stored in ~/.config/mycoswarm/memory/:
  facts.json       — versioned fact store
  sessions.jsonl   — append-only session summaries
"""

import json
import logging
from datetime import datetime
from pathlib import Path

import httpx

logger = logging.getLogger(__name__)

MEMORY_DIR = Path("~/.config/mycoswarm/memory").expanduser()
FACTS_PATH = MEMORY_DIR / "facts.json"
SESSIONS_PATH = MEMORY_DIR / "sessions.jsonl"

OLLAMA_BASE = "http://localhost:11434"


def _ensure_dir() -> None:
    MEMORY_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Layer 2: Fact Store
# ---------------------------------------------------------------------------

def load_facts() -> list[dict]:
    """Load facts from disk. Returns list of {id, text, added}."""
    if not FACTS_PATH.exists():
        return []
    try:
        data = json.loads(FACTS_PATH.read_text())
        return data.get("facts", [])
    except (json.JSONDecodeError, KeyError):
        return []


def save_facts(facts: list[dict]) -> None:
    """Write the full facts list to disk."""
    _ensure_dir()
    data = {"version": 1, "facts": facts}
    FACTS_PATH.write_text(json.dumps(data, indent=2))


def add_fact(text: str) -> dict:
    """Add a new fact. Returns the new fact dict."""
    facts = load_facts()
    next_id = max((f["id"] for f in facts), default=0) + 1
    fact = {
        "id": next_id,
        "text": text,
        "added": datetime.now().isoformat(),
    }
    facts.append(fact)
    save_facts(facts)
    return fact


def remove_fact(fact_id: int) -> bool:
    """Remove a fact by ID. Returns True if found and removed."""
    facts = load_facts()
    before = len(facts)
    facts = [f for f in facts if f["id"] != fact_id]
    if len(facts) == before:
        return False
    save_facts(facts)
    return True


def format_facts_for_prompt(facts: list[dict]) -> str:
    """Format facts for system prompt injection."""
    if not facts:
        return ""
    lines = ["Known facts about the user:"]
    for f in facts:
        lines.append(f"- {f['text']}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Layer 1: Session Summaries
# ---------------------------------------------------------------------------

def load_session_summaries(limit: int = 10) -> list[dict]:
    """Load the last N session summaries from the JSONL file."""
    if not SESSIONS_PATH.exists():
        return []
    summaries = []
    try:
        for line in SESSIONS_PATH.read_text().splitlines():
            line = line.strip()
            if line:
                try:
                    summaries.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    except OSError:
        return []
    return summaries[-limit:]


def save_session_summary(
    name: str, model: str, summary: str, count: int,
) -> None:
    """Append one session summary to the JSONL file."""
    _ensure_dir()
    entry = {
        "session_name": name,
        "model": model,
        "timestamp": datetime.now().isoformat(),
        "summary": summary,
        "message_count": count,
    }
    with open(SESSIONS_PATH, "a") as f:
        f.write(json.dumps(entry) + "\n")


def summarize_session(messages: list[dict], model: str) -> str | None:
    """Ask Ollama to summarize a chat session. Returns summary or None.

    Best-effort: returns None on any failure. Truncates to last 30
    messages, 500 chars each to keep the request small.
    """
    if len(messages) < 2:
        return None

    # Truncate for summarization
    recent = messages[-30:]
    trimmed = []
    for m in recent:
        content = m.get("content", "")
        if len(content) > 500:
            content = content[:500] + "..."
        trimmed.append(f"{m['role']}: {content}")
    transcript = "\n".join(trimmed)

    try:
        with httpx.Client(timeout=httpx.Timeout(5.0, read=30.0)) as client:
            resp = client.post(
                f"{OLLAMA_BASE}/api/chat",
                json={
                    "model": model,
                    "messages": [
                        {
                            "role": "system",
                            "content": (
                                "Summarize this conversation in 1-2 sentences. "
                                "Focus on the main topics discussed and any "
                                "decisions or outcomes. Be concise."
                            ),
                        },
                        {
                            "role": "user",
                            "content": transcript,
                        },
                    ],
                    "options": {"temperature": 0.3, "num_predict": 150},
                    "stream": False,
                },
            )
            resp.raise_for_status()
            data = resp.json()
            return data.get("message", {}).get("content", "").strip() or None
    except Exception as e:
        logger.debug("Session summarization failed: %s", e)
        return None


def format_summaries_for_prompt(summaries: list[dict]) -> str:
    """Format session summaries for system prompt injection."""
    if not summaries:
        return ""
    lines = ["Previous conversations:"]
    for s in summaries:
        ts = s.get("timestamp", "")[:10]  # date only
        summary = s.get("summary", "")
        lines.append(f"- [{ts}] {summary}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Prompt Builder
# ---------------------------------------------------------------------------

def build_memory_system_prompt() -> str | None:
    """Build a combined memory prompt from facts + summaries.

    Returns a string to inject as a system message, or None if empty.
    """
    parts = [
        "You have persistent memory across conversations. When the user "
        "asks what you know about them, state the facts naturally without "
        "disclaimers about memory limitations."
    ]

    facts = load_facts()
    facts_text = format_facts_for_prompt(facts)
    if facts_text:
        parts.append(facts_text)

    summaries = load_session_summaries()
    summaries_text = format_summaries_for_prompt(summaries)
    if summaries_text:
        parts.append(summaries_text)

    if len(parts) == 1:
        return None  # only the instruction line, no actual memory content

    return "\n\n".join(parts)
