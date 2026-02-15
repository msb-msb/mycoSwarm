#!/bin/bash
# smoke_memory.sh — Validate source priority, contradiction detection,
# and grounding score mechanics in the retrieval pipeline.

set -uo pipefail

PASS=0
FAIL=0

echo "  Testing memory priority & contradiction detection..."

# Test 1: Source priority — doc hits should have source_type=user_document
python3 -c "
from mycoswarm.library import search_all
doc_hits, session_hits = search_all(
    'what does PLAN.md say about Phase 20?',
    n_results=5,
    intent={'tool': 'rag', 'mode': 'recall', 'scope': 'docs'}
)
for h in doc_hits:
    if h.get('source_type') != 'user_document':
        print('  ❌ Doc hit missing source_type=user_document')
        exit(1)
print('  ✅ All doc hits tagged as user_document')
exit(0)
" && PASS=$((PASS+1)) || FAIL=$((FAIL+1))

# Test 2: Session hits should have source_type=model_generated
python3 -c "
from mycoswarm.library import search_all
doc_hits, session_hits = search_all(
    'what did we discuss about bees?',
    n_results=5,
    intent={'tool': 'rag', 'mode': 'recall', 'scope': 'session'}
)
if not session_hits:
    print('  ⚠️  No session hits — skipping source_type check')
    exit(0)
for h in session_hits:
    if h.get('source_type') != 'model_generated':
        print('  ❌ Session hit missing source_type=model_generated')
        exit(1)
print('  ✅ All session hits tagged as model_generated')
exit(0)
" && PASS=$((PASS+1)) || FAIL=$((FAIL+1))

# Test 3: Source filter — PLAN.md query returns only PLAN.md chunks
python3 -c "
from mycoswarm.library import search_all
doc_hits, _ = search_all(
    'what does PLAN.md say about Phase 20?',
    n_results=5,
    intent={'tool': 'rag', 'mode': 'recall', 'scope': 'docs'}
)
sources = set(h.get('source', '') for h in doc_hits)
if sources == {'PLAN.md'}:
    print('  ✅ Source filter: only PLAN.md chunks returned')
    exit(0)
else:
    print(f'  ❌ Source filter leaked: got {sources}')
    exit(1)
" && PASS=$((PASS+1)) || FAIL=$((FAIL+1))

# Test 4: Section header boost — Phase 20 chunk should rank #1
python3 -c "
from mycoswarm.library import search_all
doc_hits, _ = search_all(
    'what does PLAN.md say about Phase 20?',
    n_results=5,
    intent={'tool': 'rag', 'mode': 'recall', 'scope': 'docs'}
)
if not doc_hits:
    print('  ❌ No doc hits returned')
    exit(1)
top = doc_hits[0]
section = top.get('section', '')
if 'Phase 20' in section:
    print(f'  ✅ Section boost: Phase 20 chunk ranked #1 ({section})')
    exit(0)
else:
    print(f'  ❌ Section boost failed: top chunk is \"{section}\"')
    exit(1)
" && PASS=$((PASS+1)) || FAIL=$((FAIL+1))

# Test 5: Grounding score computation
python3 -c "
from mycoswarm.memory import compute_grounding_score
# Fully grounded summary
score = compute_grounding_score(
    'We discussed Phase 20 Intent Classification Gate.',
    ['what does PLAN.md say about Phase 20?'],
    ['Phase 20: Intent Classification Gate — distributed task']
)
if score >= 0.5:
    print(f'  ✅ Grounding score for grounded summary: {score:.2f}')
    exit(0)
else:
    print(f'  ❌ Grounding score too low for grounded summary: {score:.2f}')
    exit(1)
" && PASS=$((PASS+1)) || FAIL=$((FAIL+1))

# Test 6: Ungrounded summary gets low score
python3 -c "
from mycoswarm.memory import compute_grounding_score
score = compute_grounding_score(
    'Phase 20 is about building a Spaceship to Mars with Rocket Fuel.',
    ['what does PLAN.md say about Phase 20?'],
    ['Phase 20: Intent Classification Gate — distributed task']
)
if score < 0.5:
    print(f'  ✅ Grounding score for ungrounded summary: {score:.2f}')
    exit(0)
else:
    print(f'  ❌ Grounding score too high for hallucinated summary: {score:.2f}')
    exit(1)
" && PASS=$((PASS+1)) || FAIL=$((FAIL+1))

echo ""
echo "  Memory Results: $PASS passed, $FAIL failed"
[ "$FAIL" -eq 0 ]
