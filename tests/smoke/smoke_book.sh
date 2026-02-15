#!/bin/bash
# smoke_book.sh — Large RAG stress test with book-length PDFs
#
# Tests chunking, retrieval, and grounding at scale (100+ chunks).
# Requires a book PDF to be ingested first.
#
# Usage:
#   1. Place a PDF in the library:
#      mycoswarm library ingest /path/to/book.pdf
#   2. Run this test:
#      bash tests/smoke/smoke_book.sh "book_filename.pdf"
#
# Recommended test books:
#   - Jason Gregory "Wu Wei: Effortless Living" (~200 pages, philosophy)
#   - Any CC-licensed technical book on AI/ML

set -uo pipefail

BOOK="${1:-}"
if [ -z "$BOOK" ]; then
    echo "Usage: bash smoke_book.sh <book_filename.pdf>"
    echo "  Ingest the book first: mycoswarm library ingest /path/to/book.pdf"
    exit 1
fi

PASS=0
FAIL=0

echo "  Testing large RAG with book: $BOOK"
echo ""

# ─── Pre-checks ───────────────────────────────────────────────

# Verify book is indexed
CHUNK_COUNT=$(python3 -c "
from mycoswarm.library import _get_collection
col = _get_collection()
data = col.get(include=['metadatas'])
count = sum(1 for m in data['metadatas'] if m.get('source','') == '$BOOK')
print(count)
")

echo "  Chunks indexed for $BOOK: $CHUNK_COUNT"

if [ "$CHUNK_COUNT" -eq 0 ]; then
    echo "  ❌ No chunks found — did you run: mycoswarm library ingest /path/to/$BOOK?"
    exit 1
fi

if [ "$CHUNK_COUNT" -lt 10 ]; then
    echo "  ⚠️  Only $CHUNK_COUNT chunks — book may not be fully indexed"
fi

# ─── Test 1: Source filter at scale ──────────────────────────

echo ""
echo "  Test 1: Source filter at scale..."
python3 -c "
from mycoswarm.library import search_all
doc_hits, _, _ = search_all(
    'what does $BOOK say about the main topic?',
    n_results=5,
    intent={'tool': 'rag', 'mode': 'recall', 'scope': 'docs'}
)
sources = set(h.get('source', '') for h in doc_hits)
if '$BOOK' in sources and len(sources) == 1:
    print('  ✅ Source filter: only $BOOK chunks returned')
    exit(0)
elif '$BOOK' in sources:
    print(f'  ⚠️  Source filter partial: got {sources}')
    exit(0)
else:
    print(f'  ❌ Source filter failed: got {sources}')
    exit(1)
" && PASS=$((PASS+1)) || FAIL=$((FAIL+1))

# ─── Test 2: Section header detection ────────────────────────

echo ""
echo "  Test 2: Section headers detected..."
python3 -c "
from mycoswarm.library import _get_collection
col = _get_collection()
data = col.get(include=['metadatas'])
sections = set()
for m in data['metadatas']:
    if m.get('source','') == '$BOOK':
        s = m.get('section', 'untitled')
        sections.add(s)
unique = len(sections)
untitled = sum(1 for s in sections if s == 'untitled')
print(f'  Unique sections: {unique}, untitled: {untitled}')
if unique > 1 and untitled < unique:
    print('  ✅ Section headers detected in chunks')
    exit(0)
else:
    print('  ⚠️  Poor section detection — most chunks are untitled')
    exit(0)  # Warning, not failure
" && PASS=$((PASS+1)) || FAIL=$((FAIL+1))

# ─── Test 3: Retrieval relevance ─────────────────────────────

echo ""
echo "  Test 3: Retrieval relevance at scale..."
python3 -c "
from mycoswarm.library import search_all

# Generic query that should match book content
doc_hits, _, _ = search_all(
    'what are the key concepts in $BOOK?',
    n_results=5,
    intent={'tool': 'rag', 'mode': 'explore', 'scope': 'docs'}
)

if not doc_hits:
    print('  ❌ No hits returned for broad query')
    exit(1)

# Check that RRF scores are differentiated (not all identical)
scores = [h.get('rrf_score', 0) for h in doc_hits]
score_range = max(scores) - min(scores)
print(f'  RRF score range: {score_range:.6f} (min={min(scores):.4f}, max={max(scores):.4f})')

if score_range > 0.001:
    print('  ✅ Scores are differentiated — ranking is meaningful')
    exit(0)
else:
    print('  ⚠️  Scores are flat — ranking may be arbitrary')
    exit(0)
" && PASS=$((PASS+1)) || FAIL=$((FAIL+1))

# ─── Test 4: Context window pressure ─────────────────────────

echo ""
echo "  Test 4: Context window size..."
python3 -c "
from mycoswarm.library import search_all

doc_hits, _, _ = search_all(
    'explain the main argument of $BOOK',
    n_results=5,
    intent={'tool': 'rag', 'mode': 'recall', 'scope': 'docs'}
)

total_chars = sum(len(h.get('text', '')) for h in doc_hits)
total_tokens_est = total_chars // 4  # rough estimate

print(f'  5 chunks = {total_chars} chars ≈ {total_tokens_est} tokens')

if total_tokens_est > 8000:
    print('  ⚠️  Context may exceed model window — consider reducing chunk size')
elif total_tokens_est > 4000:
    print('  ✅ Context is substantial but within limits')
else:
    print('  ✅ Context fits comfortably in model window')
exit(0)
" && PASS=$((PASS+1)) || FAIL=$((FAIL+1))

# ─── Test 5: Full chat grounding ─────────────────────────────

echo ""
echo "  Test 5: Full chat grounding with book content..."
session="smoke-book-$(date +%s)"
output=$(echo "what does $BOOK say about its central theme?" | mycoswarm chat --session "$session" 2>&1)

# Check that response mentions something (not empty/error)
response_lines=$(echo "$output" | grep -v '🍄\|──\|/model\|Session\|Resumed\|Running\|Bye\|intent\|📚\|💭' | wc -l)
if [ "$response_lines" -gt 1 ]; then
    echo "  ✅ Model generated a response ($response_lines lines)"
    PASS=$((PASS+1))
else
    echo "  ❌ Model response too short or empty"
    FAIL=$((FAIL+1))
fi

# Check for citation markers
if echo "$output" | grep -q '\[D[0-9]\]'; then
    echo "  ✅ Response includes document citations"
    PASS=$((PASS+1))
else
    echo "  ⚠️  No citations found — model may not be grounding"
    FAIL=$((FAIL+1))
fi

# ─── Test 6: Chunk boundary quality ──────────────────────────

echo ""
echo "  Test 6: Chunk boundary quality..."
python3 -c "
from mycoswarm.library import _get_collection
col = _get_collection()
data = col.get(include=['documents', 'metadatas'])

short_chunks = 0
mid_sentence = 0
total = 0

for doc, meta in zip(data['documents'], data['metadatas']):
    if meta.get('source','') != '$BOOK':
        continue
    total += 1
    words = len(doc.split())
    if words < 50:
        short_chunks += 1
    # Check if chunk starts mid-sentence (lowercase first word, no heading marker)
    first_word = doc.strip().split()[0] if doc.strip() else ''
    if first_word and first_word[0].islower() and not first_word.startswith('#'):
        mid_sentence += 1

print(f'  Total chunks: {total}')
print(f'  Short chunks (<50 words): {short_chunks} ({100*short_chunks//max(total,1)}%)')
print(f'  Mid-sentence starts: {mid_sentence} ({100*mid_sentence//max(total,1)}%)')

if short_chunks / max(total,1) > 0.3:
    print('  ⚠️  Many short chunks — chunker may be splitting too aggressively')
elif mid_sentence / max(total,1) > 0.5:
    print('  ⚠️  Many mid-sentence starts — chunker not respecting boundaries')
else:
    print('  ✅ Chunk boundaries look reasonable')
exit(0)
" && PASS=$((PASS+1)) || FAIL=$((FAIL+1))

# ─── Summary ─────────────────────────────────────────────────

echo ""
echo "  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Book Test Summary ($BOOK)"
echo "  Chunks: $CHUNK_COUNT"
echo "  Results: $PASS passed, $FAIL failed"
echo "  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
[ "$FAIL" -eq 0 ]
