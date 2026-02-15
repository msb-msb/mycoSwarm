# Phase 21c-fix: Temporal Recency Boost for Session Retrieval

## Problem
Query "what were we talking about last time?" returns older sessions (Layens hives, ADHD, Bitcoin)
instead of the most recent session (Lao-tzu philosophy). The semantic search has no topical keywords
to match on — it's a pure recency query.

## Root Cause
- search_all() ranks sessions by RRF (semantic + BM25 fusion)
- "last time" has no semantic overlap with any session content
- 21b decay helps slightly but 30-day half-life doesn't differentiate same-day sessions
- No mechanism to detect and handle temporal/recency queries differently

## Fix: Temporal Recency Boost in library.py

### Step 1: Add temporal keyword detector (near top of library.py)

```python
import re

_TEMPORAL_RECENCY_RE = re.compile(
    r'\b(?:last\s+time|earlier\s+today|yesterday|just\s+now|before|recently|'
    r'previous(?:ly)?|last\s+session|last\s+chat|most\s+recent|what\s+did\s+we)\b',
    re.IGNORECASE,
)

def _is_temporal_recency_query(query: str) -> bool:
    """Detect if query is asking about recent conversations by time, not topic."""
    return bool(_TEMPORAL_RECENCY_RE.search(query))
```

### Step 2: Apply heavy recency boost in search_all() session scoring

Find the session scoring block (around line 1280-1286) where we compute:
```python
base_rrf = rrf_scores[doc_id]
gs = hit.get("grounding_score", 1.0)
decay = _recency_decay(hit.get("date", ""))
```

After that block, add a temporal recency boost:

```python
# Temporal recency boost: when query is about "last time" / "recently",
# heavily boost the most recent sessions by adding a date-sorted bonus
is_temporal = _is_temporal_recency_query(query)
if is_temporal and session_hits:
    # Sort by date descending, assign bonus: most recent gets +0.1, next +0.05, etc.
    dated_hits = sorted(
        session_hits,
        key=lambda h: h.get("date", ""),
        reverse=True,
    )
    for i, hit in enumerate(dated_hits):
        bonus = max(0.1 - (i * 0.02), 0.0)  # 0.1, 0.08, 0.06, 0.04, 0.02, 0...
        hit["rrf_score"] = round(hit.get("rrf_score", 0) + bonus, 6)
```

Place this AFTER the decay scoring and BEFORE the contradiction detection.

### Step 3: Tests (in tests/test_library.py or new tests/test_temporal.py)

```python
from mycoswarm.library import _is_temporal_recency_query

def test_temporal_detects_last_time():
    assert _is_temporal_recency_query("what were we talking about last time?")

def test_temporal_detects_recently():
    assert _is_temporal_recency_query("what did we discuss recently?")

def test_temporal_detects_yesterday():
    assert _is_temporal_recency_query("what did we talk about yesterday?")

def test_temporal_detects_previous():
    assert _is_temporal_recency_query("continue our previous conversation")

def test_temporal_detects_what_did_we():
    assert _is_temporal_recency_query("what did we decide about the architecture?")

def test_temporal_ignores_topical():
    assert not _is_temporal_recency_query("what does the Wu Wei book say about effortless action?")

def test_temporal_ignores_casual():
    assert not _is_temporal_recency_query("hey how are you doing?")

def test_temporal_ignores_factual():
    assert not _is_temporal_recency_query("what is the capital of France?")
```

Also add an integration-style test verifying the boost is applied:

```python
def test_temporal_boost_applied_in_search_all():
    """When temporal recency detected, most recent session should rank highest."""
    # This test would need mocked ChromaDB — use same pattern as existing search_all tests
    pass  # CC can implement based on existing test patterns
```

## Expected Result
- "what were we talking about last time?" → returns most recent session first
- "what does PLAN.md say about Phase 20?" → unaffected (no temporal keywords)
- Topical queries still use semantic ranking as before
- Temporal boost stacks with existing RRF + decay scoring

## PLAN.md Update
Mark 21c as partially done:
```
#### 21c: Mode-Aware Retrieval
- [x] Connect intent gates (Phase 20) to memory retrieval (2026-02-13)
- [x] Temporal recency boost: "last time" / "recently" queries prioritize newest sessions (2026-02-15)
- [ ] Brainstorm/planning → broad retrieval, more results
- [ ] Execution → narrow retrieval, precise constraints
```
