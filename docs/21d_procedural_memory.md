# Phase 21d: Procedural Memory — Integration Guide

## What This Is

Procedural memory stores **how we solved problems** — not just what happened (episodic)
or what's true (semantic), but reusable patterns: "when X happens, do Y because Z."

Currently, lessons from rich episodic memory (29a) are indexed as `topic=lesson_learned`
in the session_memory collection. They compete with all other session hits. 21d gives
them a dedicated home with problem signatures, outcomes, and intent-triggered retrieval.

## Schema: procedures.jsonl

Location: `~/.config/mycoswarm/memory/procedures.jsonl`

Each line is a JSON object:

```json
{
    "id": "proc_001",
    "problem": "RAG context injected as system message gets ignored by the model",
    "solution": "Inject RAG context into the user message instead of system message",
    "reasoning": "Models treat system prompts as behavioral guidelines, not factual context. User messages get higher attention weight.",
    "anti_patterns": ["Don't use a separate system message for RAG results"],
    "outcome": "success",
    "tags": ["rag", "prompt-engineering", "grounding"],
    "source_session": "debug-rag-2026-02-14",
    "created": "2026-02-14T19:30:00",
    "last_used": "2026-02-15T14:00:00",
    "use_count": 3
}
```

### Fields

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| id | str | yes | `proc_{NNN}` auto-incremented |
| problem | str | yes | Problem signature — what triggers this procedure |
| solution | str | yes | What to do |
| reasoning | str | no | Why this works — the principle behind it |
| anti_patterns | list[str] | no | What NOT to do |
| outcome | str | yes | "success" or "failure" (failures are anti-exemplars) |
| tags | list[str] | no | Searchable topic tags |
| source_session | str | no | Session that generated this procedure |
| created | str | yes | ISO timestamp |
| last_used | str | yes | Updated on retrieval |
| use_count | int | yes | Incremented on retrieval |

## Implementation

### 1. Add to memory.py

Add these constants and functions after the session summary section:

```python
# ---------------------------------------------------------------------------
# Layer 3: Procedural Memory
# ---------------------------------------------------------------------------

PROCEDURES_PATH = MEMORY_DIR / "procedures.jsonl"


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
    """
    if not procedures:
        return ""
    lines = []
    for i, p in enumerate(procedures):
        tag = f"[P{i + 1}]"
        outcome_marker = "✓" if p.get("outcome") == "success" else "✗"
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
        reasoning="Extracted from session lesson — refine with /procedure edit",
        outcome="success",
        tags=tags or [],
        source_session=session_name,
    )
```

### 2. Add to library.py

Add the procedural memory ChromaDB collection and search functions.

#### 2a. Collection getter (near _get_session_collection)

```python
def _get_procedural_collection():
    """Get or create the procedural_memory ChromaDB collection."""
    import chromadb
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    return client.get_or_create_collection(
        name="procedural_memory",
        metadata={"hnsw:space": "cosine"},
    )


# BM25 index for procedural memory
_bm25_procedures = BM25Index("procedural_memory")
```

#### 2b. Index function

```python
def index_procedure(proc: dict, model: str | None = None) -> bool:
    """Index a procedure into the procedural_memory ChromaDB collection.

    Creates a searchable document combining problem + solution + reasoning.
    Returns True on success.
    """
    model = _get_embedding_model(model)

    # Combine fields into a searchable document
    doc_parts = [proc["problem"], proc["solution"]]
    if proc.get("reasoning"):
        doc_parts.append(proc["reasoning"])
    for ap in proc.get("anti_patterns", []):
        doc_parts.append(f"Avoid: {ap}")
    document = " ".join(doc_parts)

    embedding = embed_text(document, model)
    if embedding is None:
        return False

    metadata = {
        "proc_id": proc["id"],
        "outcome": proc.get("outcome", "success"),
        "tags": ",".join(proc.get("tags", [])),
        "created": proc.get("created", ""),
        "embedding_model": model,
    }

    collection = _get_procedural_collection()
    collection.upsert(
        ids=[proc["id"]],
        documents=[document],
        embeddings=[embedding],
        metadatas=[metadata],
    )
    _bm25_procedures.invalidate()
    return True


def reindex_procedures(model: str | None = None) -> dict:
    """Drop and rebuild the procedural_memory collection from procedures.jsonl.

    Returns {"procedures": count, "indexed": count, "failed": count}.
    """
    import chromadb
    from mycoswarm.memory import load_procedures

    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    try:
        client.delete_collection("procedural_memory")
    except (ValueError, Exception):
        pass
    _bm25_procedures.invalidate()

    model = _get_embedding_model(model)
    procs = load_procedures()
    stats = {"procedures": len(procs), "indexed": 0, "failed": 0}

    for proc in procs:
        if index_procedure(proc, model):
            stats["indexed"] += 1
        else:
            stats["failed"] += 1

    return stats
```

#### 2c. Search function

```python
def search_procedures(
    query: str,
    n_results: int = 3,
    model: str | None = None,
) -> list[dict]:
    """Search procedural memory using hybrid search (vector + BM25).

    Returns list of procedure dicts with rrf_score added.
    Only returns 'success' outcomes by default (anti-patterns are in the doc text).
    """
    from mycoswarm.memory import load_procedures

    model = _get_embedding_model(model)
    query_embedding = embed_text(query, model)
    if query_embedding is None:
        return []

    try:
        col = _get_procedural_collection()
        if col.count() == 0:
            return []

        n_fetch = min(n_results * 2, col.count())

        # Vector search
        vec_results = col.query(
            query_embeddings=[query_embedding],
            n_results=n_fetch,
        )

        # BM25 search
        bm25_results = _bm25_procedures.search(query, n_results=n_fetch)

        # Build lookup
        proc_data: dict[str, dict] = {}
        vec_ids: list[str] = []

        if vec_results and vec_results["ids"]:
            for i, pid in enumerate(vec_results["ids"][0]):
                vec_ids.append(pid)
                meta = vec_results["metadatas"][0][i] if vec_results["metadatas"] else {}
                proc_data[pid] = {
                    "proc_id": pid,
                    "document": vec_results["documents"][0][i],
                    "outcome": meta.get("outcome", "success"),
                    "tags": meta.get("tags", ""),
                }

        bm25_ids: list[str] = []
        for hit in bm25_results:
            bm25_ids.append(hit["id"])
            if hit["id"] not in proc_data:
                meta = hit["metadata"]
                proc_data[hit["id"]] = {
                    "proc_id": hit["id"],
                    "document": hit["document"],
                    "outcome": meta.get("outcome", "success"),
                    "tags": meta.get("tags", ""),
                }

        # RRF fusion
        rrf_scores = _rrf_fuse(vec_ids, bm25_ids)
        sorted_ids = sorted(rrf_scores, key=lambda x: rrf_scores[x], reverse=True)
        sorted_ids = sorted_ids[:n_results]

        # Hydrate with full procedure data from JSONL
        all_procs = {p["id"]: p for p in load_procedures()}
        results = []
        for pid in sorted_ids:
            full = all_procs.get(pid)
            if full:
                full["rrf_score"] = round(rrf_scores[pid], 6)
                results.append(full)

        return results

    except Exception:
        return []
```

#### 2d. Wire into search_all()

Add procedural retrieval AFTER the session memory section and BEFORE re-ranking.
The trigger: `intent mode == "execute"` OR procedures collection is non-empty and
query looks like a problem (contains error/bug/fix/issue/how to/why does).

In `search_all()`, add a third return value. Update signature:

```python
def search_all(
    query: str,
    n_results: int = 5,
    model: str | None = None,
    rerank_model: str | None = None,
    do_rerank: bool = False,
    session_boost: bool = False,
    intent: dict | None = None,
) -> tuple[list[dict], list[dict], list[dict]]:
    """Search documents, sessions, AND procedural memory.

    Returns (doc_hits, session_hits, procedure_hits).
    """
```

Add this block after the session memory section (after poison loop detection,
before re-ranking):

```python
    # --- Procedural memory: triggered by execute mode or problem-like queries ---
    procedure_hits: list[dict] = []
    _mode = intent.get("mode", "explore") if intent else "explore"
    _PROBLEM_RE = re.compile(
        r'\b(error|bug|fix|issue|fail|broke|wrong|how\s+to|why\s+does|'
        r'doesn.t\s+work|not\s+working|problem|debug|solve)\b',
        re.IGNORECASE,
    )
    if _mode == "execute" or _PROBLEM_RE.search(query):
        try:
            procedure_hits = search_procedures(query, n_results=3, model=model)
        except Exception:
            pass

    # Also search procedures when mode is recall and scope is all
    if not procedure_hits and _mode == "recall" and _scope == "all":
        try:
            procedure_hits = search_procedures(query, n_results=2, model=model)
        except Exception:
            pass
```

Update the return:
```python
    return doc_hits, session_hits, procedure_hits
```

**IMPORTANT**: Every caller of `search_all()` must be updated to unpack 3 values
instead of 2. Search for `search_all(` in cli.py — there are 3 call sites.
Update each from:
```python
doc_hits, session_hits = search_all(...)
```
to:
```python
doc_hits, session_hits, procedure_hits = search_all(...)
```

### 3. Add to cli.py

#### 3a. Format procedure hits in chat context

Where doc_hits and session_hits get formatted (search for the `[D1]` / `[S1]` formatting
sections in `cmd_chat` and `cmd_rag`), add procedure formatting:

```python
from mycoswarm.memory import format_procedures_for_prompt, reference_procedure

# After session hits formatting:
if procedure_hits:
    proc_text = format_procedures_for_prompt(procedure_hits)
    rag_parts.append(f"\nRelevant procedures (past solutions):\n{proc_text}")
    # Track usage
    for p in procedure_hits:
        reference_procedure(p.get("id", ""))
```

Update the cite_parts hint to include procedures:
```python
if procedure_hits:
    cite_parts.append("[P1], [P2] for known procedures")
```

#### 3b. Add /procedure slash command

In the slash command section of `cmd_chat`, add:

```python
elif user_input.startswith("/procedure"):
    parts = user_input.split(maxsplit=1)
    subcmd = parts[1].strip() if len(parts) > 1 else "list"

    if subcmd == "list":
        from mycoswarm.memory import load_procedures
        procs = load_procedures()
        if not procs:
            print("  No procedures stored yet.")
            print("  Add one: /procedure add <problem> | <solution>")
        else:
            for p in procs:
                outcome = "ok" if p["outcome"] == "success" else "FAIL"
                uses = p.get("use_count", 0)
                print(f"  [{p['id']}] ({outcome}, used {uses}x) {p['problem'][:60]}")
        continue

    elif subcmd.startswith("add "):
        text = subcmd[4:].strip()
        if "|" in text:
            problem, solution = text.split("|", 1)
            from mycoswarm.memory import add_procedure
            proc = add_procedure(
                problem=problem.strip(),
                solution=solution.strip(),
            )
            print(f"  Stored: {proc['id']}")
        else:
            print("  Format: /procedure add <problem> | <solution>")
        continue

    elif subcmd.startswith("remove "):
        proc_id = subcmd[7:].strip()
        from mycoswarm.memory import remove_procedure
        if remove_procedure(proc_id):
            print(f"  Removed: {proc_id}")
        else:
            print(f"  Not found: {proc_id}")
        continue

    elif subcmd.startswith("promote"):
        # Promote recent lessons to procedures
        from mycoswarm.memory import load_session_summaries, promote_lesson_to_procedure
        sessions = load_session_summaries(limit=5)
        promoted = 0
        for s in sessions:
            for lesson in s.get("lessons", []):
                result = promote_lesson_to_procedure(
                    lesson,
                    session_name=s.get("session_name", ""),
                )
                if result:
                    print(f"  Promoted: {result['id']} — {lesson[:60]}")
                    promoted += 1
        if promoted == 0:
            print("  No promotable lessons found in recent sessions.")
        else:
            print(f"  Promoted {promoted} lessons to procedures.")
        continue

    else:
        print("  /procedure list | add <problem>|<solution> | remove <id> | promote")
        continue
```

#### 3c. Add library subcommand

In `cmd_library`, add a `reindex-procedures` subcommand alongside `reindex-sessions`:

```python
elif action == "reindex-procedures":
    from mycoswarm.library import reindex_procedures
    stats = reindex_procedures()
    print(f"  Reindexed: {stats['indexed']}/{stats['procedures']} procedures")
    if stats["failed"]:
        print(f"  Failed: {stats['failed']}")
```

### 4. Update build_memory_system_prompt() in memory.py

Add procedure awareness to the system prompt. After the session history section:

```python
    # Procedural memory context
    procs = load_procedures()
    if procs:
        parts.append(
            "You also have procedural memory — past solutions to problems. "
            "When [P1], [P2] etc. appear in your context, these are proven "
            "approaches from previous problem-solving sessions. Apply them "
            "when the current problem matches the pattern. If a procedure "
            "is marked as a failure, avoid that approach."
        )
```

### 5. Seed Procedures from Existing Lessons

After implementing, run this to promote existing lessons:

```bash
mycoswarm chat --session seed-procedures
> /procedure promote
```

This scans the last 5 sessions for lessons with action language and promotes them.

You can also manually add key procedures from the Phase 20 debugging arc:

```bash
mycoswarm chat --session seed-procedures
> /procedure add RAG context ignored when injected as system message | Inject RAG context into the user message instead — models treat system prompts as behavioral guidelines, not factual context
> /procedure add Hallucinated session summary gets indexed and poisons future queries | Score summaries with compute_grounding_score() and exclude < 0.3 from ChromaDB index
> /procedure add Model cites sessions that contradict ingested documents | Apply 2x RRF boost to user_document sources over model_generated; run _detect_contradictions() to drop conflicting session hits
```

### 6. Tests

Add `tests/test_procedures.py`:

```python
"""Tests for Phase 21d: Procedural Memory."""

import json
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from mycoswarm.memory import (
    add_procedure,
    load_procedures,
    remove_procedure,
    reference_procedure,
    format_procedures_for_prompt,
    promote_lesson_to_procedure,
    PROCEDURES_PATH,
)


@pytest.fixture(autouse=True)
def tmp_procedures(tmp_path, monkeypatch):
    """Use temp dir for procedures."""
    proc_path = tmp_path / "procedures.jsonl"
    monkeypatch.setattr("mycoswarm.memory.PROCEDURES_PATH", proc_path)
    monkeypatch.setattr("mycoswarm.memory.MEMORY_DIR", tmp_path)
    return proc_path


class TestAddProcedure:
    @patch("mycoswarm.memory.index_procedure", return_value=True)
    def test_basic_add(self, mock_idx):
        proc = add_procedure("test problem", "test solution")
        assert proc["id"] == "proc_001"
        assert proc["problem"] == "test problem"
        assert proc["solution"] == "test solution"
        assert proc["outcome"] == "success"
        assert proc["use_count"] == 0

    @patch("mycoswarm.memory.index_procedure", return_value=True)
    def test_auto_increment_id(self, mock_idx):
        add_procedure("p1", "s1")
        proc2 = add_procedure("p2", "s2")
        assert proc2["id"] == "proc_002"

    @patch("mycoswarm.memory.index_procedure", return_value=True)
    def test_with_all_fields(self, mock_idx):
        proc = add_procedure(
            "problem", "solution",
            reasoning="because reasons",
            anti_patterns=["don't do this"],
            outcome="failure",
            tags=["rag", "debug"],
            source_session="test-session",
        )
        assert proc["reasoning"] == "because reasons"
        assert proc["anti_patterns"] == ["don't do this"]
        assert proc["outcome"] == "failure"
        assert proc["tags"] == ["rag", "debug"]


class TestLoadProcedures:
    @patch("mycoswarm.memory.index_procedure", return_value=True)
    def test_load_empty(self, mock_idx):
        assert load_procedures() == []

    @patch("mycoswarm.memory.index_procedure", return_value=True)
    def test_load_after_add(self, mock_idx):
        add_procedure("p1", "s1")
        add_procedure("p2", "s2")
        procs = load_procedures()
        assert len(procs) == 2
        assert procs[0]["problem"] == "p1"
        assert procs[1]["problem"] == "p2"


class TestRemoveProcedure:
    @patch("mycoswarm.memory.index_procedure", return_value=True)
    def test_remove_existing(self, mock_idx):
        add_procedure("p1", "s1")
        assert remove_procedure("proc_001") is True
        assert load_procedures() == []

    def test_remove_nonexistent(self):
        assert remove_procedure("proc_999") is False


class TestReferenceProcedure:
    @patch("mycoswarm.memory.index_procedure", return_value=True)
    def test_reference_increments(self, mock_idx):
        add_procedure("p1", "s1")
        assert reference_procedure("proc_001") is True
        procs = load_procedures()
        assert procs[0]["use_count"] == 1

    @patch("mycoswarm.memory.index_procedure", return_value=True)
    def test_reference_updates_timestamp(self, mock_idx):
        add_procedure("p1", "s1")
        original = load_procedures()[0]["last_used"]
        reference_procedure("proc_001")
        updated = load_procedures()[0]["last_used"]
        assert updated >= original

    def test_reference_nonexistent(self):
        assert reference_procedure("proc_999") is False


class TestFormatProcedures:
    def test_empty(self):
        assert format_procedures_for_prompt([]) == ""

    def test_success_procedure(self):
        proc = {
            "id": "proc_001",
            "problem": "RAG ignored",
            "solution": "Inject into user message",
            "reasoning": "Models prefer user messages",
            "anti_patterns": ["Don't use system message"],
            "outcome": "success",
        }
        result = format_procedures_for_prompt([proc])
        assert "[P1]" in result
        assert "RAG ignored" in result
        assert "Inject into user message" in result
        assert "Models prefer user messages" in result
        assert "Don't use system message" in result

    def test_failure_procedure(self):
        proc = {
            "id": "proc_001",
            "problem": "Bad approach",
            "solution": "This failed",
            "outcome": "failure",
        }
        result = format_procedures_for_prompt([proc])
        assert "[P1]" in result


class TestPromoteLesson:
    @patch("mycoswarm.memory.index_procedure", return_value=True)
    def test_promotes_actionable_lesson(self, mock_idx):
        result = promote_lesson_to_procedure(
            "RAG context should be injected into user message instead of system message"
        )
        assert result is not None
        assert result["id"] == "proc_001"

    @patch("mycoswarm.memory.index_procedure", return_value=True)
    def test_skips_non_actionable(self, mock_idx):
        result = promote_lesson_to_procedure(
            "Taoist philosophy emphasizes individual spiritual growth"
        )
        assert result is None

    @patch("mycoswarm.memory.index_procedure", return_value=True)
    def test_promotes_with_avoid(self, mock_idx):
        result = promote_lesson_to_procedure(
            "Avoid injecting ungrounded summaries into the index"
        )
        assert result is not None
```

### 7. Smoke Test

Add `tests/smoke/smoke_procedure.sh`:

```bash
#!/usr/bin/env bash
set -e
echo "=== Procedure Memory Smoke Test ==="

# Test 1: Add a procedure via CLI
echo "Test 1: Add procedure..."
echo "/procedure add Model hallucination from bad summary | Score with grounding check, exclude < 0.3
/quit" | mycoswarm chat --session proc-smoke-1 2>/dev/null

# Test 2: List procedures
echo "Test 2: List procedures..."
OUTPUT=$(echo "/procedure list
/quit" | mycoswarm chat --session proc-smoke-2 2>/dev/null)
echo "$OUTPUT" | grep -q "proc_001" && echo "  PASS: Procedure found" || echo "  FAIL: Procedure not found"

# Test 3: Check ChromaDB indexing
echo "Test 3: Check index..."
python3 -c "
from mycoswarm.library import _get_procedural_collection
col = _get_procedural_collection()
print(f'  Procedures indexed: {col.count()}')
assert col.count() >= 1, 'No procedures in ChromaDB'
print('  PASS')
"

# Test 4: Search retrieval
echo "Test 4: Search procedures..."
python3 -c "
from mycoswarm.library import search_procedures
hits = search_procedures('hallucination grounding problem')
print(f'  Hits: {len(hits)}')
assert len(hits) >= 1, 'No procedure hits'
print(f'  Top hit: {hits[0][\"problem\"][:60]}')
print('  PASS')
"

echo "=== All procedure smoke tests passed ==="
```

### 8. Update PLAN.md

Mark Phase 21d items as done:

```markdown
#### 21d: Procedural Memory & Wisdom Layer
- [x] New memory type: exemplar store (procedures.jsonl + ChromaDB procedural_memory collection)
- [x] "How we solved X before" — success/fail patterns with problem signature matching
- [x] Anti-patterns: Explicitly store what *not* to do and why
- [x] Stored separately from episodic and factual memory with dedicated retrieval path
- [x] Procedural retrieval trigger: When intent mode=execute or problem signature matches known pattern
- [ ] Value-informed procedures: Store not just the solution but the reasoning (schema supports it, needs prompt refinement)
- [ ] Procedure growth from experience: End-of-session extraction identifies reusable patterns → auto-creates procedure candidates
- [ ] Ethical reasoning domain: Wisdom from Taoism, IFS, martial arts philosophy indexed as procedural knowledge
```

## Summary of Changes

| File | Changes |
|------|---------|
| `memory.py` | +120 lines: procedures JSONL CRUD, format, promote |
| `library.py` | +120 lines: ChromaDB collection, index, search, reindex |
| `library.py` | ~15 lines: wire procedures into search_all() |
| `cli.py` | ~60 lines: /procedure slash command |
| `cli.py` | ~10 lines: format procedure hits in context |
| `cli.py` | ~5 lines: update all search_all() unpack sites (3 locations) |
| `memory.py` | ~5 lines: system prompt awareness |
| `tests/test_procedures.py` | ~120 lines: 12 unit tests |
| `tests/smoke/smoke_procedure.sh` | ~40 lines: 4 smoke tests |

## CC Command

```
Read docs/21d_procedural_memory.md and implement all changes.
Run pytest to verify, then run the smoke test.
Update PLAN.md with completion dates.
```
