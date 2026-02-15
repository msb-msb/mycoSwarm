#!/bin/bash
# smoke_intent.sh — Validate intent classification accuracy
#
# Tests that the gate model correctly classifies different query types
# and that deterministic overrides work.

set -uo pipefail

PASS=0
FAIL=0

check_intent() {
    local label="$1"
    local query="$2"
    local expect_tool="$3"
    local expect_scope="$4"  # use "*" to accept any scope

    local result
    result=$(python3 -c "
from mycoswarm.solo import intent_classify
r = intent_classify('$query')
print(f\"{r['tool']}|{r.get('scope','all')}\")
")
    local got_tool got_scope
    got_tool=$(echo "$result" | cut -d'|' -f1)
    got_scope=$(echo "$result" | cut -d'|' -f2)

    local ok=true
    if [ "$got_tool" != "$expect_tool" ]; then
        ok=false
    fi
    if [ "$expect_scope" != "*" ] && [ "$got_scope" != "$expect_scope" ]; then
        ok=false
    fi

    if $ok; then
        echo "  ✅ $label — $got_tool/$got_scope"
        PASS=$((PASS+1))
    else
        echo "  ❌ $label — expected $expect_tool/${expect_scope}, got $got_tool/$got_scope"
        FAIL=$((FAIL+1))
    fi
}

echo "  Testing intent classification..."

# Named file → rag/docs (deterministic override)
check_intent "PLAN.md query" \
    "what does PLAN.md say about Phase 20?" \
    "rag" "docs"

# Session recall → rag/session
check_intent "Bees session recall" \
    "what did we discuss about bees?" \
    "rag" "session"

# Casual chat → answer (scope varies by model)
check_intent "Casual greeting" \
    "hey how are you?" \
    "answer" "*"

# README reference → rag (scope varies by model)
check_intent "README.md query" \
    "summarize README.md" \
    "rag" "*"

# Deterministic override: docs scope forces rag (never web_and_rag)
echo "  Testing deterministic override (5 runs)..."
override_pass=true
for i in $(seq 1 5); do
    result=$(python3 -c "
from mycoswarm.solo import intent_classify
r = intent_classify('what does PLAN.md say about Phase 20?')
print(r['tool'])
")
    if [ "$result" != "rag" ]; then
        override_pass=false
        break
    fi
done

if $override_pass; then
    echo "  ✅ Deterministic override — 5/5 returned rag"
    PASS=$((PASS+1))
else
    echo "  ❌ Deterministic override — got $result instead of rag"
    FAIL=$((FAIL+1))
fi

echo ""
echo "  Intent Results: $PASS passed, $FAIL failed"
[ "$FAIL" -eq 0 ]
