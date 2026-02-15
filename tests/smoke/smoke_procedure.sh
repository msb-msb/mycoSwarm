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
