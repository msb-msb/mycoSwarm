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


def compute_grounding_score(
    summary: str,
    user_messages: list[str],
    rag_context: list[str],
) -> float:
    """Check what fraction of summary claims are grounded in context.

    Extracts key terms (capitalized words, quoted phrases) from the summary
    and checks how many appear in the user messages or RAG context.
    Returns 0.0-1.0 (fraction of grounded terms).  Returns 1.0 if no
    extractable terms (nothing to check → assume grounded).
    """
    import re as _re

    terms: set[str] = set()
    # Capitalized words (proper nouns, technical terms, >2 chars)
    for word in summary.split():
        cleaned = word.strip(".,;:!?\"'()-")
        if len(cleaned) > 2 and cleaned[0:1].isupper():
            terms.add(cleaned.lower())
    # Quoted phrases
    for match in _re.findall(r'"([^"]+)"', summary):
        terms.add(match.lower())
    for match in _re.findall(r"'([^']+)'", summary):
        if len(match) > 2:
            terms.add(match.lower())

    if not terms:
        return 1.0

    corpus = " ".join(user_messages + rag_context).lower()
    grounded = sum(1 for t in terms if t in corpus)
    return round(grounded / len(terms), 4)


def save_session_summary(
    name: str, model: str, summary: str, count: int,
    grounding_score: float | None = None,
) -> None:
    """Append one session summary to the JSONL file and index into ChromaDB."""
    _ensure_dir()
    timestamp = datetime.now().isoformat()
    entry = {
        "session_name": name,
        "model": model,
        "timestamp": timestamp,
        "summary": summary,
        "message_count": count,
        "source_type": "model_generated",
        "grounding_score": grounding_score if grounding_score is not None else 1.0,
    }
    with open(SESSIONS_PATH, "a") as f:
        f.write(json.dumps(entry) + "\n")

    # Split into per-topic chunks and index each separately
    try:
        from mycoswarm.library import index_session_summary
        topics = split_session_topics(summary, model)
        for i, t in enumerate(topics):
            index_session_summary(
                session_id=f"{name}::topic_{i}",
                summary=t["summary"],
                date=timestamp[:10],
                topic=t["topic"],
            )
    except Exception as e:
        logger.debug("Failed to index session summary: %s", e)


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


def split_session_topics(
    summary: str, model: str,
) -> list[dict]:
    """Split a session summary into per-topic chunks via Ollama.

    Returns [{"topic": "short label", "summary": "paragraph"}, ...].
    Falls back to the full summary as a single topic on any failure.
    """
    fallback = [{"topic": "general", "summary": summary}]

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
                                "Split the following session summary into distinct "
                                "topics. Respond with ONLY JSON, no explanation:\n"
                                '{"topics": [{"topic": "short label", '
                                '"summary": "paragraph about that topic"}]}\n'
                                "Each topic should be a separate subject discussed "
                                "in the session. If there is only one topic, return "
                                "a single-element list."
                            ),
                        },
                        {"role": "user", "content": summary},
                    ],
                    "options": {"temperature": 0.3, "num_predict": 500},
                    "stream": False,
                },
            )
            resp.raise_for_status()
            data = resp.json()
            raw = data.get("message", {}).get("content", "").strip()
            if not raw:
                return fallback
    except Exception as e:
        logger.debug("Topic splitting failed: %s", e)
        return fallback

    # Extract JSON (model might wrap in markdown fences)
    json_str = raw
    if "```" in json_str:
        for block in json_str.split("```"):
            block = block.strip()
            if block.startswith("json"):
                block = block[4:].strip()
            if block.startswith("{"):
                json_str = block
                break

    try:
        parsed = json.loads(json_str)
        topics = parsed.get("topics", [])
        if not isinstance(topics, list) or not topics:
            return fallback
        # Validate each entry has the required keys
        result = []
        for t in topics:
            if isinstance(t, dict) and "topic" in t and "summary" in t:
                result.append({"topic": str(t["topic"]), "summary": str(t["summary"])})
        return result if result else fallback
    except (json.JSONDecodeError, KeyError, TypeError):
        return fallback


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

def build_memory_system_prompt(query: str | None = None) -> str:
    """Build a combined memory prompt from facts + session summaries.

    When *query* is provided, uses semantic search to find the most
    relevant past sessions instead of just the last 10 chronological ones.

    Always returns a system prompt string — includes capability boundaries
    to prevent hallucination of real-time data, plus any memory content.
    """
    parts = [
        "You are a local AI assistant with persistent memory and a "
        "document library. When document excerpts [D1], [D2] etc. or "
        "session memories [S1], [S2] etc. appear in your context, "
        "these are REAL retrieved results from the user's indexed "
        "files and past conversations — use them confidently to "
        "answer questions.\n"
        "Without web search tools active, you cannot look up current "
        "weather, news, stock prices, sports scores, or real-time "
        "information. If asked and no web results are provided, say: "
        "'I don't have access to real-time information. You can try: "
        "mycoswarm research <your question> for web-sourced answers.' "
        "Never fabricate current data like weather, prices, or news.",

        "You have persistent memory across conversations. Your memory has "
        "two distinct sources:\n"
        "  1. FACTS — things the user explicitly told you to remember via "
        "/remember. State these naturally as known preferences or details.\n"
        "  2. SESSION HISTORY — summaries of past conversations with dates. "
        "When referencing these, always cite the date naturally: "
        "\"On [date], we discussed...\" or \"Back on [date], you asked about...\"\n"
        "If the session summaries below don't contain anything relevant to "
        "the user's current question, say something like: \"I don't recall "
        "us discussing that — could you remind me?\" Never fabricate or "
        "guess at past conversations that aren't in your session history.",
    ]

    facts = load_facts()
    facts_text = format_facts_for_prompt(facts)
    if facts_text:
        parts.append(facts_text)

    # Try semantic session search first, fall back to chronological
    session_text = ""
    if query:
        try:
            from mycoswarm.library import search_sessions
            hits = search_sessions(query, n_results=3)
            if hits:
                lines = ["Relevant past conversations:"]
                for h in hits:
                    lines.append(f"- [{h['date']}] {h['summary']}")
                session_text = "\n".join(lines)
        except Exception:
            pass  # fall through to chronological

    if not session_text:
        summaries = load_session_summaries()
        session_text = format_summaries_for_prompt(summaries)

    if session_text:
        parts.append(session_text)

    return "\n\n".join(parts)
