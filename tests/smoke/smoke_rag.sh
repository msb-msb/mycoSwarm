#!/bin/bash
# smoke_rag.sh — Validate RAG retrieval grounds model responses
#
# Tests that the model uses REAL document content instead of hallucinating.
# Each test pipes a query, captures output, and checks for expected content.

set -uo pipefail

PASS=0
FAIL=0

check_output() {
    local label="$1"
    local query="$2"
    local expect="$3"        # grep pattern that SHOULD appear
    local reject="${4:-}"    # grep pattern that should NOT appear (optional)

    local session="smoke-rag-$(date +%s)-$RANDOM"
    local output
    output=$(echo "$query" | mycoswarm chat --session "$session" 2>&1)

    # Check expected content
    if echo "$output" | grep -qi "$expect"; then
        echo "  ✅ $label — found expected: '$expect'"
        PASS=$((PASS+1))
    else
        echo "  ❌ $label — missing expected: '$expect'"
        echo "     Got: $(echo "$output" | grep -v '🍄\|──\|/model\|Session\|Resumed\|Running\|Bye' | head -5)"
        FAIL=$((FAIL+1))
    fi

    # Check rejected content (if specified)
    if [ -n "$reject" ]; then
        if echo "$output" | grep -qi "$reject"; then
            echo "  ⚠️  $label — found rejected content: '$reject'"
            FAIL=$((FAIL+1))
        fi
    fi
}

echo "  Testing RAG grounding..."

# Test 1: PLAN.md Phase 20 should mention Intent Classification Gate
check_output \
    "Phase 20 grounding" \
    "what does PLAN.md say about Phase 20?" \
    "intent classification" \
    "hive inspection"

# Test 2: CLAUDE.md coding standards should mention asyncio or dataclass
check_output \
    "CLAUDE.md grounding" \
    "what does CLAUDE.md say about coding standards?" \
    "asyncio\|dataclass\|emoji"

# Test 3: Session recall — bees should mention dates
check_output \
    "Bees session recall" \
    "what did we discuss about bees?" \
    "february\|january\|202[0-9]"

# Test 4: Casual chat should NOT trigger RAG
check_output \
    "Casual chat (no RAG)" \
    "hey how are you?" \
    "help\|doing\|hello\|hi"

echo ""
echo "  RAG Results: $PASS passed, $FAIL failed"
[ "$FAIL" -eq 0 ]
