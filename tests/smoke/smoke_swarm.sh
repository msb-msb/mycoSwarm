#!/bin/bash
# smoke_swarm.sh — Validate distributed task execution across swarm
#
# Tests that intent classification dispatches to peers and returns
# correct results. Requires daemon running with active peers.

set -uo pipefail

PASS=0
FAIL=0

echo "  Testing swarm distribution..."

# Check daemon is running
if ! curl -s http://localhost:7890/status > /dev/null 2>&1; then
    echo "  ❌ Daemon not running — skipping swarm tests"
    exit 1
fi

# Get peer count
PEERS=$(curl -s http://localhost:7890/status | python3 -c "import sys,json; print(json.load(sys.stdin).get('peers',0))")
echo "  Peers detected: $PEERS"

if [ "$PEERS" -eq 0 ]; then
    echo "  ⚠️  No peers — skipping distributed tests"
    exit 0
fi

# Test 1: Distributed intent classification via chat
session="smoke-swarm-$(date +%s)"
output=$(echo "what does PLAN.md say about Phase 20?" | mycoswarm chat --debug --session "$session" 2>&1)

intent_line=$(echo "$output" | grep "🤔 intent:")
if echo "$intent_line" | grep -q "rag/recall/docs"; then
    echo "  ✅ Distributed intent — rag/recall/docs"
    PASS=$((PASS+1))
else
    echo "  ❌ Distributed intent — got: $intent_line"
    FAIL=$((FAIL+1))
fi

# Check response quality
if echo "$output" | grep -qi "intent classification"; then
    echo "  ✅ Distributed response grounded — mentions intent classification"
    PASS=$((PASS+1))
else
    echo "  ⚠️  Distributed response may not be grounded"
    echo "     $(echo "$output" | grep -v '🍄\|──\|/model\|Session\|Resumed\|DEBUG\|Bye' | head -3)"
    FAIL=$((FAIL+1))
fi

# Test 2: Direct peer task submission
PEER_IP="192.168.50.12"  # boa
TASK_ID="smoke-$(date +%s)"

echo "  Submitting direct task to $PEER_IP..."
submit_result=$(curl -s -X POST "http://$PEER_IP:7890/task" \
    -H "Content-Type: application/json" \
    -d "{\"task_id\": \"$TASK_ID\", \"task_type\": \"intent_classify\", \"payload\": {\"query\": \"what does PLAN.md say about Phase 20?\"}, \"source_node\": \"smoke-test\"}" 2>&1)

if echo "$submit_result" | grep -q "pending"; then
    echo "  ✅ Task submitted to peer"
    PASS=$((PASS+1))
else
    echo "  ❌ Task submission failed: $submit_result"
    FAIL=$((FAIL+1))
fi

# Wait for completion
sleep 10
task_result=$(curl -s "http://$PEER_IP:7890/task/$TASK_ID" 2>&1)
status=$(echo "$task_result" | python3 -c "import sys,json; print(json.load(sys.stdin).get('status','unknown'))" 2>/dev/null)
tool=$(echo "$task_result" | python3 -c "import sys,json; print(json.load(sys.stdin).get('result',{}).get('tool','none'))" 2>/dev/null)

if [ "$status" = "completed" ] && [ "$tool" = "rag" ]; then
    echo "  ✅ Peer returned rag (scope override working)"
    PASS=$((PASS+1))
elif [ "$status" = "completed" ]; then
    echo "  ⚠️  Peer completed but tool=$tool (expected rag)"
    FAIL=$((FAIL+1))
else
    echo "  ❌ Peer task status: $status"
    FAIL=$((FAIL+1))
fi

echo ""
echo "  Swarm Results: $PASS passed, $FAIL failed"
[ "$FAIL" -eq 0 ]
