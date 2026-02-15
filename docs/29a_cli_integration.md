# Phase 29a: Rich Episodic Memory — CLI Integration Guide

## Changes to memory.py (done)

### New functions
- `summarize_session_rich(messages, model)` → returns dict with summary, decisions, lessons, surprises, emotional_tone
- `_parse_rich_summary(raw, messages, model)` → JSON parsing with validation
- `_rich_fallback(messages, model)` → wraps plain summary in rich format
- `_sanitize_str_list(items)` → validates list-of-strings from model output

### Modified functions
- `save_session_summary()` — accepts optional: decisions, lessons, surprises, emotional_tone. Indexes lessons as separate ChromaDB chunks (topic="lesson_learned")
- `format_summaries_for_prompt()` — shows tone tags, lessons (max 3), decisions (max 2) when available. Backward compatible with old entries.

### Constants
- `VALID_TONES` — set of valid emotional tone values

## Changes needed in cli.py (or solo.py chat loop)

### 1. Replace summarize_session with summarize_session_rich at exit

Find the session exit code that currently does:
```python
summary = summarize_session(messages, model)
if summary:
    score = compute_grounding_score(summary, user_msgs, rag_ctx)
    save_session_summary(name, model, summary, len(messages), grounding_score=score)
```

Replace with:
```python
from mycoswarm.memory import summarize_session_rich

rich = summarize_session_rich(messages, model)
if rich:
    score = compute_grounding_score(rich["summary"], user_msgs, rag_ctx)
    save_session_summary(
        name, model, rich["summary"], len(messages),
        grounding_score=score,
        decisions=rich.get("decisions"),
        lessons=rich.get("lessons"),
        surprises=rich.get("surprises"),
        emotional_tone=rich.get("emotional_tone"),
    )
```

The progress indicator should change from:
```
Summarizing session... done.
```
to:
```
Reflecting on session... done.
```

### 2. Keep plain summarize_session() as fallback

summarize_session_rich already calls summarize_session() internally as
its fallback path, so no separate fallback code needed in CLI.

### 3. Optional: Show rich summary on exit

After summarization, optionally display what was captured:
```python
if rich:
    tone = rich.get("emotional_tone", "neutral")
    lessons = rich.get("lessons", [])
    if tone != "neutral":
        print(f"   Tone: {tone}")
    if lessons:
        for l in lessons[:2]:
            print(f"   💡 {l}")
```

This gives the user feedback that the system is learning, not just logging.

## Test commands
```bash
# Unit tests
pytest tests/test_rich_episodic.py -v

# Full suite (should be ~305+ tests)
pytest tests/ -v

# Manual test — have a real conversation, then /quit and check:
cat ~/.config/mycoswarm/memory/sessions.jsonl | python3 -c "
import sys, json
for line in sys.stdin:
    e = json.loads(line)
    if 'lessons' in e:
        print(f'{e[\"session_name\"]}: {e[\"emotional_tone\"]}')
        for l in e.get('lessons', []):
            print(f'  💡 {l}')
"
```

## Backward compatibility
- Old sessions.jsonl entries work fine — missing fields treated as empty
- format_summaries_for_prompt handles both old and new format
- summarize_session() still exists and works unchanged
- Grounding score computation uses rich["summary"] (same string as before)
- Topic splitting uses rich["summary"] (same string as before)

## What this enables next
- **21d Procedural Memory:** Lessons indexed as topic="lesson_learned" are already searchable. Phase 21d adds a dedicated retrieval path.
- **29b End-of-Session Reflection:** The rich prompt is the foundation. 29b adds a review step where the user confirms/edits the extracted record.
- **29c Interaction Quality Tracking:** emotional_tone is now stored per-session. 29c adds turn-level tracking.
- **29d 8 C's Dashboard:** Tone distribution becomes the Compassion metric.
