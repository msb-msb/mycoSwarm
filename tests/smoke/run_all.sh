#!/bin/bash
# mycoSwarm Smoke Tests — End-to-End Pipeline Validation
# Run from project root: bash tests/smoke/run_all.sh
#
# These tests validate the FULL pipeline (intent → retrieval → generation)
# unlike unit tests which test individual functions.
#
# Exit codes: 0 = all pass, 1 = failures detected

set -uo pipefail

SMOKE_DIR="$(cd "$(dirname "$0")" && pwd)"
PASS=0
FAIL=0
ERRORS=()

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

run_test() {
    local name="$1"
    local script="$2"
    echo -e "\n${YELLOW}━━━ $name ━━━${NC}"
    if bash "$SMOKE_DIR/$script"; then
        echo -e "${GREEN}✅ PASS: $name${NC}"
        ((PASS++))
    else
        echo -e "${RED}❌ FAIL: $name${NC}"
        ((FAIL++))
        ERRORS+=("$name")
    fi
}

echo "🍄 mycoSwarm Smoke Test Suite"
echo "   $(date)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

run_test "RAG Grounding"          "smoke_rag.sh"
run_test "Poison Resistance"      "smoke_poison.sh"
run_test "Memory Priority"        "smoke_memory.sh"
run_test "Intent Classification"  "smoke_intent.sh"

# Swarm test only if daemon is running
if curl -s http://localhost:7890/status > /dev/null 2>&1; then
    run_test "Swarm Distribution" "smoke_swarm.sh"
else
    echo -e "\n${YELLOW}⏭  Skipping Swarm test (daemon not running)${NC}"
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo -e "Results: ${GREEN}${PASS} passed${NC}, ${RED}${FAIL} failed${NC}"
if [ ${#ERRORS[@]} -gt 0 ]; then
    echo -e "${RED}Failed tests:${NC}"
    for e in "${ERRORS[@]}"; do
        echo "  - $e"
    done
    exit 1
fi
echo -e "${GREEN}All smoke tests passed! 🍄${NC}"
exit 0
