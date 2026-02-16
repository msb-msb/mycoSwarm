"""mycoSwarm persistent memory — facts + session summaries + procedures.

Provides cross-session memory by injecting structured context into the
system prompt. Three layers:

  Layer 1 — Session summaries: auto-generated after each chat session
  Layer 2 — User facts: explicit /remember commands
  Layer 3 — Procedural memory: reusable problem/solution patterns

Data stored in ~/.config/mycoswarm/memory/:
  facts.json       — versioned fact store
  sessions.jsonl   — append-only session summaries
  procedures.jsonl — problem/solution patterns
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
PROCEDURES_PATH = MEMORY_DIR / "procedures.jsonl"

OLLAMA_BASE = "http://localhost:11434"

# Valid emotional tones for rich episodic summaries
VALID_TONES = {
    "frustration", "discovery", "confusion", "resolution",
    "flow", "stuck", "exploratory", "routine", "neutral",
}

# Fact types — different retention and prompt behavior
FACT_TYPE_PREFERENCE = "preference"  # User likes/dislikes, style choices
FACT_TYPE_FACT = "fact"              # Objective info (name, location, etc.)
FACT_TYPE_PROJECT = "project"        # Active project context
FACT_TYPE_EPHEMERAL = "ephemeral"    # Temporary, auto-expires
VALID_FACT_TYPES = {
    FACT_TYPE_PREFERENCE,
    FACT_TYPE_FACT,
    FACT_TYPE_PROJECT,
    FACT_TYPE_EPHEMERAL,
}
DEFAULT_FACT_TYPE = FACT_TYPE_FACT


def _ensure_dir() -> None:
    MEMORY_DIR.mkdir(parents=True, exist_ok=True)


def _migrate_fact(fact: dict) -> dict:
    """Add missing fields to facts from older schema versions."""
    if "type" not in fact:
        fact["type"] = DEFAULT_FACT_TYPE
    if "last_referenced" not in fact:
        fact["last_referenced"] = fact.get("added", datetime.now().isoformat())
    if "reference_count" not in fact:
        fact["reference_count"] = 0
    return fact


# ---------------------------------------------------------------------------
# Layer 2: Fact Store
# ---------------------------------------------------------------------------

def load_facts() -> list[dict]:
    """Load facts from disk with migration for older schemas.

    Returns list of {id, text, added, type, last_referenced, reference_count}.
    """
    if not FACTS_PATH.exists():
        return []
    try:
        data = json.loads(FACTS_PATH.read_text())
        facts = data.get("facts", [])
        return [_migrate_fact(f) for f in facts]
    except (json.JSONDecodeError, KeyError):
        return []


def save_facts(facts: list[dict]) -> None:
    """Write the full facts list to disk."""
    _ensure_dir()
    data = {"version": 2, "facts": facts}
    FACTS_PATH.write_text(json.dumps(data, indent=2))


def add_fact(text: str, fact_type: str = DEFAULT_FACT_TYPE) -> dict:
    """Add a new fact. Returns the new fact dict.

    fact_type: preference | fact | project | ephemeral
    """
    if fact_type not in VALID_FACT_TYPES:
        fact_type = DEFAULT_FACT_TYPE
    facts = load_facts()
    next_id = max((f["id"] for f in facts), default=0) + 1
    now = datetime.now().isoformat()
    fact = {
        "id": next_id,
        "text": text,
        "type": fact_type,
        "added": now,
        "last_referenced": now,
        "reference_count": 0,
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


def reference_fact(fact_id: int) -> bool:
    """Mark a fact as referenced (updates timestamp + counter).

    Call when a fact is retrieved for use in a prompt or response.
    Returns True if found and updated.
    """
    facts = load_facts()
    for f in facts:
        if f["id"] == fact_id:
            f["last_referenced"] = datetime.now().isoformat()
            f["reference_count"] = f.get("reference_count", 0) + 1
            save_facts(facts)
            return True
    return False


def get_stale_facts(days: int = 30) -> list[dict]:
    """Find facts unreferenced in the last N days.

    Ephemeral facts use a shorter window (7 days).
    Returns list of stale fact dicts.
    """
    facts = load_facts()
    now = datetime.now()
    stale = []
    for f in facts:
        threshold = 7 if f.get("type") == FACT_TYPE_EPHEMERAL else days
        try:
            last_ref = datetime.fromisoformat(f.get("last_referenced", f["added"]))
            age_days = (now - last_ref).days
            if age_days >= threshold:
                stale.append(f)
        except (ValueError, TypeError):
            continue
    return stale


def format_facts_for_prompt(facts: list[dict]) -> str:
    """Format facts for system prompt injection, grouped by type."""
    if not facts:
        return ""

    type_labels = {
        FACT_TYPE_PREFERENCE: "User preferences",
        FACT_TYPE_FACT: "Known facts about the user",
        FACT_TYPE_PROJECT: "Active projects",
        FACT_TYPE_EPHEMERAL: "Temporary notes",
    }

    # Group by type
    grouped: dict[str, list[str]] = {}
    for f in facts:
        ft = f.get("type", DEFAULT_FACT_TYPE)
        grouped.setdefault(ft, []).append(f["text"])

    lines = []
    for ft in [FACT_TYPE_FACT, FACT_TYPE_PREFERENCE, FACT_TYPE_PROJECT, FACT_TYPE_EPHEMERAL]:
        items = grouped.get(ft, [])
        if items:
            label = type_labels.get(ft, ft.title())
            lines.append(f"{label}:")
            for item in items:
                lines.append(f"- {item}")
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
    decisions: list[str] | None = None,
    lessons: list[str] | None = None,
    surprises: list[str] | None = None,
    emotional_tone: str | None = None,
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
    if decisions:
        entry["decisions"] = decisions
    if lessons:
        entry["lessons"] = lessons
    if surprises:
        entry["surprises"] = surprises
    if emotional_tone:
        entry["emotional_tone"] = emotional_tone

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
        # Index lessons as separate searchable chunks
        if lessons:
            for j, lesson in enumerate(lessons):
                index_session_summary(
                    session_id=f"{name}::lesson_{j}",
                    summary=lesson,
                    date=timestamp[:10],
                    topic="lesson_learned",
                )
    except Exception as e:
        logger.debug("Failed to index session summary: %s", e)

    # --- Procedure candidate extraction (Phase 21d) ---
    expire_old_candidates()
    if lessons:
        candidates_added = 0
        for lesson in lessons:
            if candidates_added >= 3:
                logger.debug("Procedure candidate cap (3) reached for session")
                break
            try:
                extracted = extract_procedure_from_lesson(
                    lesson,
                    model=model,
                    session_context=summary[:200] if summary else "",
                )
                if extracted:
                    if _is_duplicate_procedure(extracted["problem"]):
                        logger.debug("Skipping duplicate procedure candidate: %s", lesson[:60])
                        continue
                    candidate = add_procedure_candidate(
                        lesson,
                        extracted=extracted,
                        session_name=name,
                    )
                    candidates_added += 1
                    logger.debug(
                        "Procedure candidate %s from lesson: %s",
                        candidate["id"],
                        lesson[:60],
                    )
            except Exception as e:
                logger.debug("Procedure candidate extraction failed: %s", e)


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


def _sanitize_str_list(items: object) -> list[str]:
    """Validate and clean a list-of-strings from model output."""
    if not isinstance(items, list):
        return []
    result: list[str] = []
    for x in items:
        if isinstance(x, (str, int, float)):
            s = str(x).strip()
            if s:
                result.append(s)
    return result


def _parse_rich_summary(
    raw: str, messages: list[dict], model: str,
) -> dict | None:
    """Parse JSON from a rich summarization response.

    Returns validated dict with: summary, decisions, lessons, surprises,
    emotional_tone.  Falls back to ``_rich_fallback`` if JSON is invalid
    or the summary field is empty.
    """
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
    except (json.JSONDecodeError, TypeError):
        return _rich_fallback(messages, model)

    if not isinstance(parsed, dict):
        return _rich_fallback(messages, model)

    summary = parsed.get("summary", "")
    if not isinstance(summary, str) or not summary.strip():
        return _rich_fallback(messages, model)

    tone = parsed.get("emotional_tone", "neutral")
    if not isinstance(tone, str) or tone not in VALID_TONES:
        tone = "neutral"

    return {
        "summary": summary.strip(),
        "decisions": _sanitize_str_list(parsed.get("decisions")),
        "lessons": _sanitize_str_list(parsed.get("lessons")),
        "surprises": _sanitize_str_list(parsed.get("surprises")),
        "emotional_tone": tone,
    }


def _rich_fallback(messages: list[dict], model: str) -> dict | None:
    """Fall back to plain summarize_session and wrap in rich format."""
    summary = summarize_session(messages, model)
    if not summary:
        return None
    return {
        "summary": summary,
        "decisions": [],
        "lessons": [],
        "surprises": [],
        "emotional_tone": "neutral",
    }


def summarize_session_rich(messages: list[dict], model: str) -> dict | None:
    """Structured session reflection — extracts summary, decisions, lessons,
    surprises, and emotional tone.

    Returns dict with keys: summary, decisions, lessons, surprises,
    emotional_tone.  Falls back to plain summary on failure.
    Returns None if the session is too short.
    """
    if len(messages) < 2:
        return None

    recent = messages[-30:]
    trimmed = []
    for m in recent:
        content = m.get("content", "")
        if len(content) > 500:
            content = content[:500] + "..."
        trimmed.append(f"{m['role']}: {content}")
    transcript = "\n".join(trimmed)

    try:
        with httpx.Client(timeout=httpx.Timeout(5.0, read=45.0)) as client:
            resp = client.post(
                f"{OLLAMA_BASE}/api/chat",
                json={
                    "model": model,
                    "messages": [
                        {
                            "role": "system",
                            "content": (
                                "Reflect on this conversation and produce a JSON "
                                "object with these fields:\n"
                                '  "summary": "1-2 sentence overview of what was discussed",\n'
                                '  "decisions": ["choices made and their reasoning"],\n'
                                '  "lessons": ["reusable SUBJECT-MATTER insights with the PRINCIPLE behind them. '
                                "Good: 'Inject RAG context into user message because models treat system prompts "
                                "as behavioral guidelines.' Bad: 'The assistant provided helpful information.'\"],\n"
                                '  "surprises": ["unexpected findings or counter-intuitive results"],\n'
                                '  "emotional_tone": one of: neutral, frustration, discovery, '
                                "confusion, resolution, flow, stuck, exploratory, "
                                "routine\n\n"
                                "Rules:\n"
                                "- lessons must be about the SUBJECT MATTER, not about how "
                                "the conversation went. Bad: 'The assistant provided helpful answers.' "
                                "Good: 'RAG context injected as a system message gets ignored by gemma3.'\n"
                                "- decisions: only explicit choices with reasoning. Empty list if none.\n"
                                "- surprises: counter-intuitive results only. Empty list if nothing surprising.\n"
                                "- Keep each item to one sentence.\n"
                                "Respond with ONLY the JSON object, no explanation."
                            ),
                        },
                        {"role": "user", "content": transcript},
                    ],
                    "options": {"temperature": 0.3, "num_predict": 400},
                    "stream": False,
                },
            )
            resp.raise_for_status()
            data = resp.json()
            raw = data.get("message", {}).get("content", "").strip()
            if not raw:
                return _rich_fallback(messages, model)
    except Exception as e:
        logger.debug("Rich summarization failed, trying fallback: %s", e)
        return _rich_fallback(messages, model)

    result = _parse_rich_summary(raw, messages, model)
    return result


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
    """Format session summaries for system prompt injection.

    Includes tone tags, lessons (max 3), and decisions (max 2) when
    available.  Backward compatible with old entries that lack these fields.
    """
    if not summaries:
        return ""
    lines = ["Previous conversations:"]
    for s in summaries:
        ts = s.get("timestamp", "")[:10]  # date only
        summary = s.get("summary", "")
        tone = s.get("emotional_tone", "")
        tone_tag = f" ({tone})" if tone and tone != "neutral" else ""
        lines.append(f"- [{ts}]{tone_tag} {summary}")
        for d in s.get("decisions", [])[:2]:
            lines.append(f"  Decision: {d}")
        for le in s.get("lessons", [])[:3]:
            lines.append(f"  Lesson: {le}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Layer 3: Procedural Memory
# ---------------------------------------------------------------------------

def load_procedures() -> list[dict]:
    """Load all procedures from the JSONL file."""
    if not PROCEDURES_PATH.exists():
        return []
    procedures = []
    for line in PROCEDURES_PATH.read_text().splitlines():
        line = line.strip()
        if line:
            try:
                procedures.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return procedures


def _next_procedure_id() -> str:
    """Generate next procedure ID."""
    procs = load_procedures()
    if not procs:
        return "proc_001"
    max_num = 0
    for p in procs:
        pid = p.get("id", "")
        if pid.startswith("proc_"):
            try:
                num = int(pid.split("_")[1])
                max_num = max(max_num, num)
            except (ValueError, IndexError):
                pass
    return f"proc_{max_num + 1:03d}"


def add_procedure(
    problem: str,
    solution: str,
    *,
    reasoning: str = "",
    anti_patterns: list[str] | None = None,
    outcome: str = "success",
    tags: list[str] | None = None,
    source_session: str = "",
) -> dict:
    """Add a new procedure. Returns the procedure dict.

    Also indexes into ChromaDB for semantic search.
    """
    _ensure_dir()
    now = datetime.now().isoformat()
    proc = {
        "id": _next_procedure_id(),
        "status": "active",
        "problem": problem,
        "solution": solution,
        "reasoning": reasoning,
        "anti_patterns": anti_patterns or [],
        "outcome": outcome,
        "tags": tags or [],
        "source_session": source_session,
        "created": now,
        "last_used": now,
        "use_count": 0,
    }

    with open(PROCEDURES_PATH, "a") as f:
        f.write(json.dumps(proc) + "\n")

    # Index into ChromaDB
    try:
        from mycoswarm.library import index_procedure
        index_procedure(proc)
    except Exception as e:
        logger.debug("Failed to index procedure: %s", e)

    return proc


def remove_procedure(proc_id: str) -> bool:
    """Remove a procedure by ID. Returns True if found and removed."""
    procs = load_procedures()
    filtered = [p for p in procs if p.get("id") != proc_id]
    if len(filtered) == len(procs):
        return False
    # Rewrite file
    _ensure_dir()
    with open(PROCEDURES_PATH, "w") as f:
        for p in filtered:
            f.write(json.dumps(p) + "\n")
    return True


def reference_procedure(proc_id: str) -> bool:
    """Mark a procedure as used (updates last_used + use_count).

    Call when a procedure is retrieved and injected into context.
    Returns True if found and updated.
    """
    procs = load_procedures()
    found = False
    for p in procs:
        if p.get("id") == proc_id:
            p["last_used"] = datetime.now().isoformat()
            p["use_count"] = p.get("use_count", 0) + 1
            found = True
            break
    if not found:
        return False
    _ensure_dir()
    with open(PROCEDURES_PATH, "w") as f:
        for p in procs:
            f.write(json.dumps(p) + "\n")
    return True


def format_procedures_for_prompt(procedures: list[dict]) -> str:
    """Format procedures for injection into chat context.

    Uses [P1], [P2] tags matching the [D1]/[S1] pattern.
    Only formats active procedures (candidates are excluded).
    """
    # Filter to active only
    procedures = [p for p in procedures if p.get("status", "active") == "active"]
    if not procedures:
        return ""
    lines = []
    for i, p in enumerate(procedures):
        tag = f"[P{i + 1}]"
        outcome_marker = "\u2713" if p.get("outcome") == "success" else "\u2717"
        lines.append(f"{tag} ({outcome_marker}) Problem: {p['problem']}")
        lines.append(f"    Solution: {p['solution']}")
        if p.get("reasoning"):
            lines.append(f"    Why: {p['reasoning']}")
        for ap in p.get("anti_patterns", []):
            lines.append(f"    Avoid: {ap}")
    return "\n".join(lines)


def promote_lesson_to_procedure(
    lesson: str,
    *,
    session_name: str = "",
    tags: list[str] | None = None,
) -> dict | None:
    """Promote an episodic lesson to a procedure.

    Attempts to split the lesson into problem/solution format.
    Returns the new procedure dict, or None if the lesson doesn't
    have a clear problem/solution structure.
    """
    # Simple heuristic: if lesson contains action language, it's promotable
    action_signals = [
        "should", "must", "don't", "avoid", "use", "instead",
        "works better", "causes", "prevents", "requires",
    ]
    has_action = any(s in lesson.lower() for s in action_signals)
    if not has_action:
        return None

    return add_procedure(
        problem=lesson,
        solution=lesson,
        reasoning="Extracted from session lesson \u2014 refine with /procedure edit",
        outcome="success",
        tags=tags or [],
        source_session=session_name,
    )


def _tokenize_problem(text: str) -> set[str]:
    """Tokenize a problem string into a set of lowercase words (>=3 chars)."""
    import re as _re
    cleaned = _re.sub(r'[^\w\s]', '', text.lower())
    return {w for w in cleaned.split() if len(w) >= 3}


def _is_duplicate_procedure(problem: str, threshold: float = 0.6) -> bool:
    """Check if a problem is too similar to any existing procedure.

    Uses Jaccard similarity on word tokens. Returns True if any
    existing procedure (active or candidate) exceeds threshold.
    """
    new_tokens = _tokenize_problem(problem)
    if not new_tokens:
        return False
    for p in load_procedures():
        existing_tokens = _tokenize_problem(p.get("problem", ""))
        if not existing_tokens:
            continue
        intersection = len(new_tokens & existing_tokens)
        union = len(new_tokens | existing_tokens)
        if union > 0 and intersection / union >= threshold:
            return True
    return False


def expire_old_candidates(days: int = 14) -> int:
    """Remove candidate procedures older than N days.

    Only affects candidates — active procedures are never expired.
    Returns count of expired candidates.
    """
    from datetime import timedelta
    procs = load_procedures()
    cutoff = datetime.now() - timedelta(days=days)
    keep = []
    expired = 0
    for p in procs:
        if p.get("status") == "candidate":
            try:
                created = datetime.fromisoformat(p.get("created", ""))
                if created < cutoff:
                    expired += 1
                    continue
            except (ValueError, TypeError):
                pass
        keep.append(p)
    if expired:
        _ensure_dir()
        with open(PROCEDURES_PATH, "w") as f:
            for p in keep:
                f.write(json.dumps(p) + "\n")
        logger.info("Expired %d unreviewed procedure candidates (>%d days old)", expired, days)
    return expired


# ---------------------------------------------------------------------------
# Layer 3b: Procedure Growth from Experience
# ---------------------------------------------------------------------------

def extract_procedure_from_lesson(
    lesson: str,
    *,
    model: str = "",
    session_context: str = "",
) -> dict | None:
    """Use LLM to structure a lesson into procedure fields.

    Returns dict with problem/solution/reasoning/anti_patterns/tags,
    or None if the lesson isn't procedural.
    """
    prompt = f"""Analyze this lesson and determine if it contains a REUSABLE DEBUGGING PATTERN or TECHNICAL DECISION — something that would help solve a similar problem in the future.

Lesson: "{lesson}"
{f'Session context: {session_context}' if session_context else ''}

Return is_procedural: true ONLY if this describes:
- A specific technical problem and its solution
- A debugging strategy that transfers to other situations
- A design decision with clear reasoning

Return is_procedural: false if this is:
- A general observation or fact
- Domain knowledge without a problem/solution pattern
- An opinion or preference
- Something obvious that any developer would know

If procedural, respond with ONLY this JSON (no markdown fences):
{{
    "is_procedural": true,
    "problem": "What problem or situation triggers this procedure (1-2 sentences)",
    "solution": "What to do — the specific action or approach (1-2 sentences)",
    "reasoning": "Why this works — the principle behind it (1 sentence)",
    "anti_patterns": ["What NOT to do (0-2 items)"],
    "tags": ["3-5 relevant topic tags"]
}}

If not procedural, respond with:
{{"is_procedural": false}}"""

    try:
        resp = httpx.post(
            f"{OLLAMA_BASE}/api/generate",
            json={
                "model": model or _pick_extraction_model(),
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.1, "num_predict": 300},
            },
            timeout=30,
        )
        if resp.status_code != 200:
            return None

        raw = resp.json().get("response", "").strip()
        # Strip markdown fences if present
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1] if "\n" in raw else raw[3:]
        if raw.endswith("```"):
            raw = raw[:-3]
        raw = raw.strip()

        data = json.loads(raw)
        if not data.get("is_procedural"):
            return None

        return {
            "problem": data.get("problem", lesson),
            "solution": data.get("solution", lesson),
            "reasoning": data.get("reasoning", ""),
            "anti_patterns": _sanitize_str_list(data.get("anti_patterns", [])),
            "tags": _sanitize_str_list(data.get("tags", [])),
        }
    except Exception as e:
        logger.debug("Procedure extraction failed: %s", e)
        return None


def _pick_extraction_model() -> str:
    """Pick a model for procedure extraction. Prefer small-but-capable."""
    try:
        from mycoswarm.solo import _pick_gate_model
        return _pick_gate_model()
    except Exception:
        return "gemma3:4b"


def add_procedure_candidate(
    lesson: str,
    *,
    extracted: dict,
    session_name: str = "",
) -> dict:
    """Store an LLM-structured procedure as a candidate (not indexed).

    Candidates require human review via /procedure review before
    being promoted to active and indexed in ChromaDB.
    """
    _ensure_dir()
    now = datetime.now().isoformat()
    proc = {
        "id": _next_procedure_id(),
        "status": "candidate",
        "problem": extracted["problem"],
        "solution": extracted["solution"],
        "reasoning": extracted.get("reasoning", ""),
        "anti_patterns": extracted.get("anti_patterns", []),
        "outcome": "success",
        "tags": extracted.get("tags", []),
        "source_session": session_name,
        "source_lesson": lesson,
        "created": now,
        "last_used": now,
        "use_count": 0,
    }

    with open(PROCEDURES_PATH, "a") as f:
        f.write(json.dumps(proc) + "\n")

    # Do NOT index into ChromaDB — candidates wait for review
    return proc


def load_procedure_candidates() -> list[dict]:
    """Load only candidate (unreviewed) procedures.

    Automatically expires old candidates (>14 days) before returning.
    """
    expire_old_candidates()
    return [p for p in load_procedures() if p.get("status") == "candidate"]


def load_active_procedures() -> list[dict]:
    """Load only active (reviewed) procedures."""
    return [p for p in load_procedures() if p.get("status", "active") == "active"]


def approve_procedure(proc_id: str) -> bool:
    """Promote a candidate to active and index into ChromaDB.

    Returns True if found and promoted.
    """
    procs = load_procedures()
    found = False
    target = None
    for p in procs:
        if p.get("id") == proc_id and p.get("status") == "candidate":
            p["status"] = "active"
            found = True
            target = p
            break
    if not found:
        return False

    _ensure_dir()
    with open(PROCEDURES_PATH, "w") as f:
        for p in procs:
            f.write(json.dumps(p) + "\n")

    # Now index into ChromaDB
    if target:
        try:
            from mycoswarm.library import index_procedure
            index_procedure(target)
        except Exception as e:
            logger.debug("Failed to index approved procedure: %s", e)

    return True


def reject_procedure(proc_id: str) -> bool:
    """Remove a candidate procedure. Returns True if found and removed."""
    procs = load_procedures()
    target = None
    for p in procs:
        if p.get("id") == proc_id and p.get("status") == "candidate":
            target = p
            break
    if not target:
        return False
    return remove_procedure(proc_id)


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
        parts.append(
            "The following are facts about THE USER that they asked you to "
            "remember. Refer to these using 'you' (second person), not 'I'.\n"
            + facts_text
        )

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

    # Procedural memory context (active only, not candidates)
    procs = load_active_procedures()
    if procs:
        parts.append(
            "You also have procedural memory \u2014 past solutions to problems. "
            "When [P1], [P2] etc. appear in your context, these are proven "
            "approaches from previous problem-solving sessions. Apply them "
            "when the current problem matches the pattern. If a procedure "
            "is marked as a failure, avoid that approach."
        )

    parts.append(
        "Be concise \u2014 short sentences, minimal padding. Apply procedures "
        "naturally without quoting them verbatim. Don't reference unrelated "
        "session memories. Only cite sources that directly answer the query.\n"
        "NEVER fabricate citations. If no [S], [D], or [P] tags appear in "
        "your context, do not invent them. Do not claim 'we discussed this' "
        "unless a specific session memory is present in your context."
    )

    return "\n\n".join(parts)
