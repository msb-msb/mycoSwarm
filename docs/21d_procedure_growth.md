# Phase 21d Continued: Procedure Growth from Experience

## What This Is

Close the loop: sessions generate lessons (29a) → LLM evaluates lessons for procedural potential → structures as problem/solution/reasoning → stores as **candidates** → human reviews with `/procedure review`.

Currently `promote_lesson_to_procedure()` is manual and dumb — it copies the lesson text to both `problem` and `solution` fields. This upgrade makes the system actually learn from experience, with human oversight before anything goes live.

## Design Principles

- **Wu Wei:** Candidates wait for human review. No auto-promotion.
- **Candidates ≠ Active:** Candidates are stored but NOT indexed into ChromaDB. They don't affect retrieval until approved.
- **LLM does the structuring:** An Ollama call splits a raw lesson into proper problem/solution/reasoning/tags fields.
- **Low friction review:** `/procedure review` shows one candidate at a time — approve, reject, edit, or skip.

## Schema Change

Add `status` field to procedures. Backward compatible — existing procedures default to `"active"`.

```python
# Existing procedure (unchanged, defaults to active)
{
    "id": "proc_001",
    "status": "active",      # NEW — "active" or "candidate"
    ...
}

# New candidate (not indexed, waiting for review)
{
    "id": "proc_016",
    "status": "candidate",
    "problem": "LLM-structured problem description",
    "solution": "LLM-structured solution",
    "reasoning": "LLM-extracted reasoning",
    "anti_patterns": ["LLM-extracted anti-patterns"],
    "outcome": "success",
    "tags": ["auto-extracted", "tags"],
    "source_session": "session-name-2026-02-16",
    "source_lesson": "Original lesson text from rich episodic memory",
    "created": "2026-02-16T10:00:00",
    "last_used": "2026-02-16T10:00:00",
    "use_count": 0
}
```

## Implementation

### 1. LLM Procedure Extraction — memory.py

Add `extract_procedure_from_lesson()` after `promote_lesson_to_procedure()`:

```python
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
    prompt = f"""Analyze this lesson from a development session and determine if it contains a reusable procedure — a pattern that could help solve similar problems in the future.

Lesson: "{lesson}"
{f'Session context: {session_context}' if session_context else ''}

If this lesson contains a reusable procedure, respond with ONLY this JSON (no markdown fences):
{{
    "is_procedural": true,
    "problem": "What problem or situation triggers this procedure (1-2 sentences)",
    "solution": "What to do — the specific action or approach (1-2 sentences)",
    "reasoning": "Why this works — the principle behind it (1 sentence)",
    "anti_patterns": ["What NOT to do (0-2 items)"],
    "tags": ["3-5 relevant topic tags"]
}}

If this lesson is just an observation, opinion, or fact without an actionable pattern, respond with:
{{"is_procedural": false}}"""

    try:
        import httpx
        resp = httpx.post(
            "http://localhost:11434/api/generate",
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
    # Reuse the gate model picker — it finds the smallest available model
    try:
        from mycoswarm.solo import _pick_gate_model
        return _pick_gate_model()
    except Exception:
        return "gemma3:4b"
```

### 2. Add Candidate to procedures.jsonl — memory.py

Add `add_procedure_candidate()`:

```python
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
```

### 3. Auto-Extract After Session — memory.py

In `summarize_session_rich()`, after lessons are extracted, evaluate each for procedural potential. Add this at the end of the function, after the rich summary is saved:

```python
    # --- Procedure candidate extraction (Phase 21d) ---
    if result and result.get("lessons"):
        session_summary = result.get("summary", "")
        for lesson in result["lessons"]:
            try:
                extracted = extract_procedure_from_lesson(
                    lesson,
                    model=model,
                    session_context=session_summary[:200],
                )
                if extracted:
                    candidate = add_procedure_candidate(
                        lesson,
                        extracted=extracted,
                        session_name=name if 'name' in dir() else "",
                    )
                    logger.debug(
                        "Procedure candidate %s from lesson: %s",
                        candidate["id"],
                        lesson[:60],
                    )
            except Exception as e:
                logger.debug("Procedure candidate extraction failed: %s", e)
```

Wait — `summarize_session_rich()` doesn't have `name` in scope. The session name is passed to `save_session_summary()` separately. Better approach: do the extraction in `save_session_summary()` where we already have `name`, `lessons`, and `model`:

**Instead**, add it at the end of `save_session_summary()`, after the lesson indexing loop:

```python
    # --- Procedure candidate extraction (Phase 21d) ---
    if lessons:
        for lesson in lessons:
            try:
                extracted = extract_procedure_from_lesson(
                    lesson,
                    model=model,
                    session_context=summary[:200] if summary else "",
                )
                if extracted:
                    candidate = add_procedure_candidate(
                        lesson,
                        extracted=extracted,
                        session_name=name,
                    )
                    logger.debug(
                        "Procedure candidate %s from lesson: %s",
                        candidate["id"],
                        lesson[:60],
                    )
            except Exception as e:
                logger.debug("Procedure candidate extraction failed: %s", e)
```

### 4. Load Helpers — memory.py

Add helpers for candidates vs active:

```python
def load_procedure_candidates() -> list[dict]:
    """Load only candidate (unreviewed) procedures."""
    return [p for p in load_procedures() if p.get("status") == "candidate"]


def load_active_procedures() -> list[dict]:
    """Load only active (reviewed) procedures."""
    return [p for p in load_procedures() if p.get("status", "active") == "active"]
```

### 5. Approve/Reject — memory.py

```python
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
```

### 6. Update Existing Functions — memory.py

**`format_procedures_for_prompt()`** — only format active procedures:
```python
# At the top of the function, filter to active only:
procedures = [p for p in procedures if p.get("status", "active") == "active"]
```

**`load_procedures()` in `build_memory_system_prompt()`** — use `load_active_procedures()`:
```python
    # Replace: procs = load_procedures()
    # With:
    procs = load_active_procedures()
```

**`search_procedures()` in library.py** — already only searches ChromaDB which only has active procedures, so no change needed.

### 7. /procedure review CLI — cli.py

Add the `review` subcommand to the `/procedure` handler:

```python
    elif subcmd == "review":
        from mycoswarm.memory import (
            load_procedure_candidates,
            approve_procedure,
            reject_procedure,
        )
        candidates = load_procedure_candidates()
        if not candidates:
            print("  No procedure candidates to review.")
            continue

        print(f"  {len(candidates)} candidate(s) to review:\n")
        for c in candidates:
            print(f"  ┌─ {c['id']} ─────────────────────────")
            print(f"  │ Problem:  {c['problem']}")
            print(f"  │ Solution: {c['solution']}")
            if c.get("reasoning"):
                print(f"  │ Why:      {c['reasoning']}")
            for ap in c.get("anti_patterns", []):
                print(f"  │ Avoid:    {ap}")
            if c.get("tags"):
                print(f"  │ Tags:     {', '.join(c['tags'])}")
            if c.get("source_lesson"):
                print(f"  │ Lesson:   {c['source_lesson'][:80]}")
            print(f"  └────────────────────────────────────")

            while True:
                choice = input("  [a]pprove / [r]eject / [s]kip / [q]uit review? ").strip().lower()
                if choice in ("a", "approve"):
                    if approve_procedure(c["id"]):
                        print(f"  ✓ Approved and indexed: {c['id']}")
                    break
                elif choice in ("r", "reject"):
                    if reject_procedure(c["id"]):
                        print(f"  ✗ Rejected: {c['id']}")
                    break
                elif choice in ("s", "skip"):
                    print(f"  — Skipped: {c['id']}")
                    break
                elif choice in ("q", "quit"):
                    print("  Review ended.")
                    break
                else:
                    print("  Type a/r/s/q")
            if choice in ("q", "quit"):
                break
        continue
```

Also update the help text:
```python
    else:
        print("  /procedure list | add <problem>|<solution> | remove <id> | promote | review")
        continue
```

### 8. Update /procedure list — cli.py

Show candidate count in the list output:

```python
    if subcmd == "list":
        from mycoswarm.memory import load_procedures
        procs = load_procedures()
        if not procs:
            print("  No procedures stored yet.")
            print("  Add one: /procedure add <problem> | <solution>")
        else:
            active = [p for p in procs if p.get("status", "active") == "active"]
            candidates = [p for p in procs if p.get("status") == "candidate"]
            for p in active:
                uses = p.get("use_count", 0)
                print(f"  [{p['id']}] (used {uses}x) {p['problem'][:60]}")
            if candidates:
                print(f"\n  📋 {len(candidates)} candidate(s) pending review — /procedure review")
        continue
```

### 9. Value-Informed Reasoning Prompt Refinement

The extraction prompt in step 1 already asks for `reasoning` — the "why this works" field. But the existing `summarize_session_rich()` prompt should also emphasize extracting the *principle* behind lessons, not just the action.

In `summarize_session_rich()`, find the lessons instruction in the JSON prompt and update:

```
Find the current text:
"lessons": ["What was learned — focus on SUBJECT-MATTER insights, not observations about the assistant"]

Replace with:
"lessons": ["What was learned — focus on SUBJECT-MATTER insights with the PRINCIPLE behind them. Good: 'Inject RAG context into user message because models treat system prompts as behavioral guidelines.' Bad: 'The assistant provided helpful information.'"]
```

This way the raw lessons already contain reasoning, making procedure extraction higher quality.

### 10. Tests

Add to `tests/test_procedures.py`:

```python
class TestProcedureCandidates:
    @patch("mycoswarm.memory.index_procedure", return_value=True)
    def test_add_candidate_not_indexed(self, mock_idx):
        """Candidates should NOT be indexed into ChromaDB."""
        candidate = add_procedure_candidate(
            "Test lesson",
            extracted={
                "problem": "Test problem",
                "solution": "Test solution",
                "reasoning": "Test reasoning",
                "anti_patterns": [],
                "tags": ["test"],
            },
            session_name="test-session",
        )
        assert candidate["status"] == "candidate"
        assert candidate["source_lesson"] == "Test lesson"
        # index_procedure should NOT have been called
        mock_idx.assert_not_called()

    @patch("mycoswarm.memory.index_procedure", return_value=True)
    def test_load_candidates_only(self, mock_idx):
        add_procedure("active problem", "active solution")
        add_procedure_candidate(
            "lesson",
            extracted={"problem": "cand problem", "solution": "cand solution"},
        )
        candidates = load_procedure_candidates()
        assert len(candidates) == 1
        assert candidates[0]["status"] == "candidate"

    @patch("mycoswarm.memory.index_procedure", return_value=True)
    def test_load_active_only(self, mock_idx):
        add_procedure("active problem", "active solution")
        add_procedure_candidate(
            "lesson",
            extracted={"problem": "cand problem", "solution": "cand solution"},
        )
        active = load_active_procedures()
        assert len(active) == 1
        assert active[0]["status"] == "active"

    @patch("mycoswarm.memory.index_procedure", return_value=True)
    def test_approve_candidate(self, mock_idx):
        candidate = add_procedure_candidate(
            "lesson",
            extracted={"problem": "p", "solution": "s"},
        )
        mock_idx.reset_mock()
        result = approve_procedure(candidate["id"])
        assert result is True
        # Should now be active
        procs = load_procedures()
        approved = [p for p in procs if p["id"] == candidate["id"]]
        assert approved[0]["status"] == "active"
        # Should have been indexed
        mock_idx.assert_called_once()

    @patch("mycoswarm.memory.index_procedure", return_value=True)
    def test_reject_candidate(self, mock_idx):
        candidate = add_procedure_candidate(
            "lesson",
            extracted={"problem": "p", "solution": "s"},
        )
        result = reject_procedure(candidate["id"])
        assert result is True
        assert load_procedure_candidates() == []

    def test_approve_nonexistent(self):
        assert approve_procedure("proc_999") is False

    def test_reject_nonexistent(self):
        assert reject_procedure("proc_999") is False

    @patch("mycoswarm.memory.index_procedure", return_value=True)
    def test_approve_active_fails(self, mock_idx):
        """Can't approve an already-active procedure."""
        proc = add_procedure("p", "s")
        assert approve_procedure(proc["id"]) is False

    @patch("mycoswarm.memory.index_procedure", return_value=True)
    def test_candidates_excluded_from_prompt(self, mock_idx):
        add_procedure("active problem", "active solution")
        add_procedure_candidate(
            "lesson",
            extracted={"problem": "cand problem", "solution": "cand solution"},
        )
        result = format_procedures_for_prompt(load_procedures())
        assert "active problem" in result
        assert "cand problem" not in result


class TestExtractProcedure:
    @patch("httpx.post")
    def test_extracts_procedural_lesson(self, mock_post):
        mock_post.return_value = MagicMock(
            status_code=200,
            json=lambda: {"response": json.dumps({
                "is_procedural": True,
                "problem": "RAG context ignored",
                "solution": "Inject into user message",
                "reasoning": "Models prioritize user messages",
                "anti_patterns": ["Don't use system message"],
                "tags": ["rag", "prompt"],
            })},
        )
        result = extract_procedure_from_lesson("RAG should be injected into user message")
        assert result is not None
        assert result["problem"] == "RAG context ignored"
        assert result["solution"] == "Inject into user message"
        assert len(result["tags"]) == 2

    @patch("httpx.post")
    def test_rejects_non_procedural(self, mock_post):
        mock_post.return_value = MagicMock(
            status_code=200,
            json=lambda: {"response": '{"is_procedural": false}'},
        )
        result = extract_procedure_from_lesson("Taoist philosophy is interesting")
        assert result is None

    @patch("httpx.post")
    def test_handles_ollama_failure(self, mock_post):
        mock_post.side_effect = Exception("Connection refused")
        result = extract_procedure_from_lesson("Some lesson")
        assert result is None
```

### 11. Update PLAN.md

Mark the completed items:

```markdown
- [x] **Value-informed procedures:** Extraction prompt asks for reasoning/principle behind each procedure. Session reflection prompt updated to capture principles in lessons. (2026-02-16)
- [x] **Procedure growth from experience:** End-of-session extraction evaluates lessons via LLM, structures as problem/solution/reasoning, stores as candidates for human review via `/procedure review`. (2026-02-16)
```

## Summary of Changes

| File | Changes |
|------|---------|
| `memory.py` | +80 lines: `extract_procedure_from_lesson()`, `add_procedure_candidate()`, `load_procedure_candidates()`, `load_active_procedures()`, `approve_procedure()`, `reject_procedure()` |
| `memory.py` | ~10 lines: auto-extraction in `save_session_summary()`, filter active in `format_procedures_for_prompt()` and `build_memory_system_prompt()` |
| `memory.py` | ~3 lines: improve lesson prompt in `summarize_session_rich()` |
| `cli.py` | ~40 lines: `/procedure review` interactive flow |
| `cli.py` | ~5 lines: update `/procedure list` to show candidate count |
| `tests/test_procedures.py` | +100 lines: 11 new tests for candidates, approve, reject, extract |

## CC Command

```
Read docs/21d_procedure_growth.md and implement all changes.
Run pytest to verify all tests pass.
Then start a test session, have a short conversation about a technical topic, /quit, and check if procedure candidates were generated.
Show me the output of /procedure review.
```
