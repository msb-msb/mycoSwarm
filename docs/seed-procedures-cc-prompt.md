# Seed Procedural Memory & Test Full Loop

## Step 1: Verify 21d is Working

```bash
cd ~/Desktop/mycoSwarm
pip install -e . 2>/dev/null

# Quick check — should show no procedures yet
python3 -c "from mycoswarm.memory import load_procedures; print(f'Procedures: {len(load_procedures())}')"

# Verify ChromaDB collection exists
python3 -c "from mycoswarm.library import _get_procedural_collection; print(f'Collection count: {_get_procedural_collection().count()}')"
```

If either errors, check that Phase 21d was merged properly.

## Step 2: Seed Phase 20 Debugging Procedures

These are the core lessons from the hallucination feedback loop debugging arc. Add them via Python (not CLI, to include all fields):

```python
import sys
sys.path.insert(0, 'src')
from mycoswarm.memory import add_procedure

# --- RAG Pipeline Fixes ---

add_procedure(
    problem="RAG context injected as system message gets ignored by local models (gemma3, phi4)",
    solution="Inject RAG context directly into the user message: augmented_query = rag_context + '\\n\\nUSER QUESTION: ' + user_input",
    reasoning="Local models treat system prompts as behavioral guidelines, not factual context. User messages get higher attention weight in the chat template.",
    anti_patterns=["Don't put RAG results in a separate system message", "Don't assume system message injection works the same across models"],
    outcome="success",
    tags=["rag", "prompt-engineering", "grounding", "local-models"],
    source_session="debug-rag-phase20-2026-02-14",
)

add_procedure(
    problem="Hallucinated session summary gets indexed into ChromaDB and poisons future queries on the same topic",
    solution="Score every summary with compute_grounding_score() before indexing. Exclude summaries scoring below 0.3 from the ChromaDB index.",
    reasoning="A summary that mentions terms not present in the actual conversation or RAG context is likely hallucinated. The grounding score measures what fraction of key terms are grounded in source material.",
    anti_patterns=["Don't auto-index every session summary without quality gating", "Don't trust grounding scores above 0.5 blindly — cross-reference with contradiction detection"],
    outcome="success",
    tags=["hallucination", "grounding", "memory", "self-correction"],
    source_session="debug-rag-phase20-2026-02-14",
)

add_procedure(
    problem="Model-generated session summaries compete equally with user-ingested documents in retrieval, allowing hallucinations to outrank real content",
    solution="Apply 2x RRF boost to user_document sources over model_generated sources in search_all(). Tag every indexed item with source_type.",
    reasoning="User-provided documents are primary sources — they represent ground truth. Model-generated summaries are secondary and should never outrank the real thing.",
    anti_patterns=["Don't give equal retrieval weight to all source types"],
    outcome="success",
    tags=["rag", "source-priority", "retrieval", "trust"],
    source_session="debug-rag-phase20-2026-02-14",
)

add_procedure(
    problem="Session summaries that contradict ingested documents persist in retrieval and confuse the model",
    solution="Run _detect_contradictions() in the retrieval pipeline: cross-reference session hits against document hits, drop sessions with <0.5 grounding that conflict with docs.",
    reasoning="When a session claim and a document claim disagree, the document wins. Sessions are derived; documents are source-of-truth.",
    anti_patterns=["Don't return contradicting session and document hits to the model without filtering"],
    outcome="success",
    tags=["contradiction", "retrieval", "self-correction", "memory"],
    source_session="debug-rag-phase20-2026-02-14",
)

add_procedure(
    problem="Same hallucinated claim appears across multiple session summaries, creating a self-reinforcing poison loop",
    solution="Detect poison loops: when >50% of key terms in multiple low-grounding sessions are the same ungrounded terms, quarantine those sessions.",
    reasoning="Feedback loops amplify errors exponentially. A single hallucination becomes 'evidence' for future hallucinations. The immune system must detect the pattern, not just individual instances.",
    anti_patterns=["Don't just filter individual sessions — check for cross-session amplification patterns"],
    outcome="success",
    tags=["poison-loop", "hallucination", "self-correction", "immune-system"],
    source_session="debug-rag-phase20-2026-02-14",
)

# --- Debugging & Development Patterns ---

add_procedure(
    problem="Testing RAG changes contaminates production session memory with test data",
    solution="Use --debug flag which skips session save entirely. Test queries never get summarized or indexed.",
    reasoning="Test pollution is invisible and cumulative. One bad test session can seed a hallucination loop that takes hours to diagnose.",
    anti_patterns=["Don't test RAG changes without --debug mode", "Don't manually delete ChromaDB entries to fix test pollution — use debug isolation to prevent it"],
    outcome="success",
    tags=["testing", "debug", "isolation", "development"],
    source_session="debug-rag-phase20-2026-02-14",
)

add_procedure(
    problem="Intent classifier (1B model) misclassifies query, sending it through wrong retrieval path",
    solution="Check intent classification output with --verbose flag. If misclassifying, the gate model may need the query reformulated, or the intent prompt needs adjustment.",
    reasoning="Small gate models are fast but imprecise. When retrieval fails, check intent classification first — it's the most common upstream cause.",
    anti_patterns=["Don't assume retrieval is broken when intent classification is wrong — fix upstream first"],
    outcome="success",
    tags=["intent", "classification", "debugging", "gate-model"],
    source_session="debug-rag-phase20-2026-02-14",
)

# --- Architecture Principles ---

add_procedure(
    problem="Choosing between upgrading model size vs fixing retrieval pipeline for hallucination reduction",
    solution="Fix retrieval first. Source priority, grounding scores, and context injection method have more impact than model size on hallucination rates.",
    reasoning="A 27B model hallucinated not because it's weak, but because the right context was buried at position 8 in retrieval. A 70B model would have done the same thing. Retrieval quality > model quality for grounding.",
    anti_patterns=["Don't throw a bigger model at a retrieval problem"],
    outcome="success",
    tags=["architecture", "retrieval", "model-selection", "principle"],
    source_session="debug-rag-phase20-2026-02-14",
)

add_procedure(
    problem="System needs manual intervention to fix corrupted memory (purge ChromaDB, delete sessions)",
    solution="Build self-correcting defenses: grounding gates prevent bad data from entering, contradiction detection removes it if it slips through, poison loop detection catches amplification patterns.",
    reasoning="Wu Wei principle — the system should naturally resist corruption rather than requiring manual purges. Self-correcting flow.",
    anti_patterns=["Don't rely on manual curation to keep memory clean", "Don't design a system that requires a human to periodically purge bad data"],
    outcome="success",
    tags=["architecture", "self-correction", "wu-wei", "principle", "immune-system"],
    source_session="debug-rag-phase20-2026-02-14",
)

print("Seeded 9 procedures from Phase 20 debugging arc.")
```

## Step 3: Verify Seed Data

```bash
# List all procedures
python3 -c "
from mycoswarm.memory import load_procedures
procs = load_procedures()
print(f'Total procedures: {len(procs)}')
for p in procs:
    print(f'  [{p[\"id\"]}] {p[\"outcome\"]:7s} | {p[\"problem\"][:70]}')
"

# Verify ChromaDB indexing
python3 -c "
from mycoswarm.library import _get_procedural_collection
col = _get_procedural_collection()
print(f'Procedures in ChromaDB: {col.count()}')
assert col.count() == 9, f'Expected 9, got {col.count()}'
print('PASS: All 9 indexed')
"
```

## Step 4: Test Retrieval

```python
from mycoswarm.library import search_procedures

# Test 1: Should find RAG injection procedure
hits = search_procedures("model ignores my RAG context")
print(f"\nTest 1 - RAG context ignored:")
for h in hits:
    print(f"  [{h['id']}] score={h['rrf_score']:.4f} | {h['problem'][:60]}")
assert any("system message" in h["problem"].lower() for h in hits), "FAIL: RAG injection not found"
print("PASS")

# Test 2: Should find hallucination/grounding procedures
hits = search_procedures("hallucination keeps coming back")
print(f"\nTest 2 - Recurring hallucination:")
for h in hits:
    print(f"  [{h['id']}] score={h['rrf_score']:.4f} | {h['problem'][:60]}")
assert len(hits) >= 1, "FAIL: No hallucination procedures found"
print("PASS")

# Test 3: Should find architecture principle
hits = search_procedures("should I use a bigger model to reduce errors")
print(f"\nTest 3 - Model size vs retrieval:")
for h in hits:
    print(f"  [{h['id']}] score={h['rrf_score']:.4f} | {h['problem'][:60]}")
print("PASS")

# Test 4: Should find debug isolation
hits = search_procedures("how to test without polluting memory")
print(f"\nTest 4 - Test isolation:")
for h in hits:
    print(f"  [{h['id']}] score={h['rrf_score']:.4f} | {h['problem'][:60]}")
print("PASS")
```

## Step 5: Test Full Loop (search_all integration)

```python
from mycoswarm.library import search_all

# search_all should return 3-tuple now
result = search_all("how do I fix RAG hallucination", intent={"mode": "execute"})
assert len(result) == 3, f"Expected 3-tuple, got {len(result)}"
doc_hits, session_hits, procedure_hits = result

print(f"Doc hits: {len(doc_hits)}")
print(f"Session hits: {len(session_hits)}")
print(f"Procedure hits: {len(procedure_hits)}")

if procedure_hits:
    print("\nProcedure hits retrieved in search_all:")
    for p in procedure_hits:
        print(f"  [{p['id']}] {p['problem'][:60]}")
    print("PASS: Procedures surfacing in search_all")
else:
    print("NOTE: No procedure hits — may need intent=execute or problem-like query")

# Test problem-pattern trigger (should fire without explicit intent)
result2 = search_all("why does the model keep hallucinating wrong answers")
_, _, proc_hits2 = result2
print(f"\nProblem-pattern trigger: {len(proc_hits2)} procedure hits")
if proc_hits2:
    print("PASS: Problem regex triggered procedural retrieval")
```

## Step 6: Promote Existing Lessons

If there are rich episodic sessions with lessons, try promoting them:

```bash
# Check for existing lessons
python3 -c "
from mycoswarm.memory import load_session_summaries
sessions = load_session_summaries(limit=10)
lesson_count = sum(len(s.get('lessons', [])) for s in sessions)
print(f'Sessions with lessons: {sum(1 for s in sessions if s.get(\"lessons\"))}')
print(f'Total lessons: {lesson_count}')
for s in sessions:
    for l in s.get('lessons', []):
        print(f'  💡 {l[:80]}')
"
```

If lessons exist, promote actionable ones:
```bash
python3 -c "
from mycoswarm.memory import load_session_summaries, promote_lesson_to_procedure
sessions = load_session_summaries(limit=10)
promoted = 0
for s in sessions:
    for lesson in s.get('lessons', []):
        result = promote_lesson_to_procedure(lesson, session_name=s.get('session_name', ''))
        if result:
            print(f'Promoted: {result[\"id\"]} — {lesson[:60]}')
            promoted += 1
print(f'\nPromoted {promoted} lessons to procedures')
"
```

## Step 7: Run Full Test Suite

```bash
pytest -x --tb=short
```

All 337+ tests should pass. If any procedure-related tests fail, fix before proceeding.

## Step 8: Live Chat Test

Start a chat session and ask a problem-like question:

```bash
mycoswarm chat --session test-procedures
```

Then ask: "My RAG context keeps getting ignored by the model, what should I do?"

The response should include `[P1]` citations from the seeded procedures. Check the RAG context panel (--verbose) to confirm procedure hits are being injected.

## Expected Final State

- 9+ procedures in `~/.config/mycoswarm/memory/procedures.jsonl`
- 9+ entries in the `procedural_memory` ChromaDB collection
- `search_all()` returns 3-tuple with procedure hits on problem-like queries
- `/procedure list` shows all seeded procedures
- Chat responses cite `[P1]`, `[P2]` when relevant procedures match

## After Verification

```bash
git add -A
git commit -m "Seed procedural memory with Phase 20 debugging exemplars (9 procedures)"
git push
```
