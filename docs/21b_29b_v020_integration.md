# Phase 21b + 29b + v0.2.0 — Integration Guide for CC

## Phase 21b: Decay Scoring

### Concept
Session memories should decay in retrieval priority over time. Recently referenced
sessions rank higher. Old unreferenced sessions fade. "Forgetting as technology."

### Where to add: library.py — search_all() session scoring section

Currently (around line 1280-1286), session RRF scores are multiplied by grounding_score:
```python
base_rrf = rrf_scores[doc_id]
gs = hit.get("grounding_score", 1.0)
hit["rrf_score"] = round(base_rrf * gs, 6)
```

Add a **recency decay multiplier** after the grounding score multiplier. The decay
is based on the session's date field relative to today.

```python
from datetime import datetime, timedelta

def _recency_decay(date_str: str, half_life_days: int = 30) -> float:
    """Compute a recency multiplier between 0.1 and 1.0.
    
    Uses exponential decay with configurable half-life.
    A session from today scores 1.0.
    A session from half_life_days ago scores 0.5.
    Never goes below 0.1 (old sessions still retrievable, just deprioritized).
    """
    try:
        session_date = datetime.fromisoformat(date_str)
    except (ValueError, TypeError):
        return 0.5  # unknown date gets neutral score
    
    age_days = (datetime.now() - session_date).days
    if age_days <= 0:
        return 1.0
    
    import math
    decay = math.pow(0.5, age_days / half_life_days)
    return max(0.1, round(decay, 4))
```

Place this function near the top of library.py (after imports, before constants).

### Apply in search_all() session scoring:

Change the session scoring block from:
```python
base_rrf = rrf_scores[doc_id]
gs = hit.get("grounding_score", 1.0)
hit["rrf_score"] = round(base_rrf * gs, 6)
```

To:
```python
base_rrf = rrf_scores[doc_id]
gs = hit.get("grounding_score", 1.0)
decay = _recency_decay(hit.get("date", ""))
hit["rrf_score"] = round(base_rrf * gs * decay, 6)
hit["recency_decay"] = decay
```

### Also apply to format_summaries_for_prompt in memory.py

No changes needed — the retrieval scoring handles it. Older sessions will
naturally rank lower and be less likely to appear in the prompt.

### Exception: lessons should decay slower

Lessons (topic="lesson_learned") are more durable than regular session memories.
In the session scoring section, after computing decay:

```python
# Lessons decay at half the rate (double half-life)
if hit.get("topic") == "lesson_learned":
    decay = _recency_decay(hit.get("date", ""), half_life_days=60)
```

### Tests to add (in tests/test_library.py or new tests/test_decay.py):

1. `test_recency_decay_today` — date=today → decay=1.0
2. `test_recency_decay_30_days` — date=30 days ago → decay≈0.5
3. `test_recency_decay_90_days` — date=90 days ago → decay≈0.125
4. `test_recency_decay_floor` — very old date → decay=0.1 (never zero)
5. `test_recency_decay_bad_date` — invalid date string → decay=0.5
6. `test_recency_decay_future` — future date → decay=1.0
7. `test_lesson_decays_slower` — lesson at 30 days has higher decay than regular session at 30 days
8. `test_search_all_applies_decay` — mock search_all, verify rrf_score includes decay factor

---

## Phase 29b: Reflection Prompt Refinement

### Problem
The current rich summarization produces self-referential lessons like:
"The assistant effectively utilizes both conversational memory and document
retrieval to provide nuanced and connected answers."

These are about the *assistant's performance*, not about the *topics discussed*.
Not useful for future retrieval.

### Fix: Update the summarize_session_rich() prompt in memory.py

Find the system prompt in summarize_session_rich() that starts with:
"Reflect on this conversation and produce a JSON..."

Replace/update the rules section. The current rules are:
```
"Respond with ONLY the JSON object, no explanation."
```

Change the full system prompt content to:
```python
"Reflect on this conversation and produce a JSON "
"object with these fields:\n"
'  "summary": "1-2 sentence overview of what was discussed",\n'
'  "decisions": ["choices made and their reasoning"],\n'
'  "lessons": ["reusable insights about the TOPICS discussed, '
'not about the assistant\'s performance"],\n'
'  "surprises": ["unexpected findings or counter-intuitive results"],\n'
'  "emotional_tone": one of: neutral, frustration, discovery, '
"confusion, resolution, flow, stuck, exploratory, routine\n\n"
"Rules:\n"
"- lessons must be about the SUBJECT MATTER, not about how "
"the conversation went. Bad: 'The assistant provided helpful answers.' "
"Good: 'RAG context injected as a system message gets ignored by gemma3.'\n"
"- decisions: only explicit choices with reasoning. Empty list if none.\n"
"- surprises: counter-intuitive results only. Empty list if nothing surprising.\n"
"- Keep each item to one sentence.\n"
"Respond with ONLY the JSON object, no explanation."
```

### Test: update test_summarize_session_rich_parses_response

The existing mock test is fine — it tests parsing, not prompt quality.
Add one new test:

```python
def test_rich_prompt_asks_for_subject_matter_lessons():
    """Verify the reflection prompt explicitly requests subject-matter lessons."""
    import inspect
    source = inspect.getsource(summarize_session_rich)
    assert "SUBJECT MATTER" in source or "TOPICS discussed" in source
```

---

## Version Bump: v0.2.0

This is a milestone release — cognitive architecture foundations.

### Changes for this release:

1. **Bump version** in pyproject.toml: "0.1.9" → "0.2.0"

2. **Update PLAN.md** — mark 21a, 21b, 29a, 29b as done with today's date

3. **Run full test suite**: `pytest tests/ -v`

4. **Run smoke tests**: `bash tests/smoke/run_all.sh`

5. **Build and upload**:
```bash
python -m build
twine upload dist/mycoswarm-0.2.0*
```

6. **GitHub release** — tag v0.2.0 with notes:
```
## v0.2.0 — Cognitive Architecture Foundations

### New: Rich Episodic Memory (Phase 29a)
- Sessions now capture decisions, lessons, surprises, and emotional tone
- Lessons indexed separately in ChromaDB for procedural retrieval
- "Reflecting on session..." replaces "Summarizing session..."
- Tone and lessons displayed on session exit

### New: Fact Lifecycle Tags (Phase 21a)  
- Facts now have types: preference, fact, project, ephemeral
- `/remember pref:` / `project:` / `temp:` prefixes
- `/stale` command shows unreferenced facts
- reference_count and last_referenced tracking
- Ephemeral facts auto-stale after 7 days

### New: Decay Scoring (Phase 21b)
- Session memories decay with exponential half-life (30 days)
- Lessons decay slower (60-day half-life)  
- Old sessions still retrievable, just deprioritized
- "Forgetting as technology"

### Improved: Reflection Prompt (Phase 29b)
- Lessons now capture subject-matter insights, not self-referential observations
- Explicit prompt guidance against "the assistant effectively..." patterns

### Architecture
- docs/ARCHITECTURE-COGNITIVE.md — IFS + CoALA framework
- Phase 29/30 added to PLAN.md
- 8 C's of Healthy AI as design principles
```

7. **Update swarm nodes**:
```bash
for host in naru boa uncho; do
  ssh $host "cd ~/mycoSwarm && source .venv/bin/activate && pip install --upgrade mycoswarm"
  ssh -t $host "sudo systemctl restart mycoswarm"
done
```

---

## Summary: Tell CC

"Implement these three changes in order:
1. Read docs/21b_29b_v020_integration.md  
2. Implement 21b (decay scoring in library.py)
3. Implement 29b (fix reflection prompt in memory.py)
4. Run full test suite + smoke tests
5. Bump version to 0.2.0, build, upload to PyPI, create GitHub release with tag + notes
6. Update all swarm nodes"
