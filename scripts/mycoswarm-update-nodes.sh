#!/usr/bin/env bash
# mycoswarm-update-nodes.sh — Update all swarm nodes to latest PyPI version
# Usage: ./scripts/mycoswarm-update-nodes.sh [version]
#   version: optional, e.g. "0.3.0" — defaults to latest on PyPI
#
# Node inventory:
#   Miu      — local dev install (pip install -e .), skip PyPI update
#   rushuna  — RTX 3060 specialist, venv at ~/mycoSwarm
#   boa      — light node, system-wide pip
#   naru     — light node, system-wide pip
#   uncho    — light node, system-wide pip
#   pi       — Raspberry Pi edge node, user=pi, venv at ~/mycoSwarm

set -euo pipefail

VERSION="${1:-}"
if [ -n "$VERSION" ]; then
    PKG="mycoswarm==$VERSION"
else
    PKG="mycoswarm --upgrade"
fi

GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

ok()   { echo -e "  ${GREEN}✅ $1${NC}"; }
fail() { echo -e "  ${RED}❌ $1${NC}"; }
warn() { echo -e "  ${YELLOW}⚠️  $1${NC}"; }

FAILED_NODES=()

# --- Helper: update a node via SSH ---
update_node() {
    local name="$1"
    local ssh_target="$2"
    local pip_cmd="$3"
    local restart_cmd="$4"

    echo ""
    echo "━━━ $name ━━━"

    # Check connectivity
    if ! ssh -o ConnectTimeout=5 -o BatchMode=yes "$ssh_target" "echo ok" &>/dev/null; then
        fail "$name: SSH connection failed"
        FAILED_NODES+=("$name")
        return
    fi

    # Install
    if ssh "$ssh_target" "$pip_cmd" 2>&1; then
        ok "$name: package updated"
    else
        fail "$name: pip install failed"
        FAILED_NODES+=("$name")
        return
    fi

    # Restart daemon
    if ssh -t "$ssh_target" "$restart_cmd" 2>&1; then
        ok "$name: daemon restarted"
    else
        warn "$name: restart failed (may need manual sudo)"
        FAILED_NODES+=("$name")
        return
    fi

    # Verify version
    local remote_ver
    remote_ver=$(ssh "$ssh_target" "python3 -c 'import mycoswarm; print(mycoswarm.__version__)'" 2>/dev/null || echo "unknown")
    echo "  📦 $name: v$remote_ver"
}

echo "🍄 mycoSwarm Node Updater"
echo "========================="
echo "  Package: $PKG"
echo ""

# --- Miu (local) ---
echo ""
echo "━━━ Miu (local) ━━━"
echo "  Miu uses dev install (pip install -e .) — skipping PyPI update."
echo "  To update Miu: git pull && pip install -e ."
LOCAL_VER=$(python3 -c "import mycoswarm; print(mycoswarm.__version__)" 2>/dev/null || echo "unknown")
echo "  📦 Miu: v$LOCAL_VER"

# --- rushuna (specialist, venv) ---
update_node "rushuna" "minotaur@rushuna" \
    "cd ~/mycoSwarm && source .venv/bin/activate && pip install $PKG" \
    "sudo systemctl restart mycoswarm"

# --- boa (light, system-wide) ---
update_node "boa" "minotaur@boa" \
    "pip install $PKG --break-system-packages" \
    "sudo systemctl restart mycoswarm"

# --- naru (light, system-wide) ---
update_node "naru" "minotaur@naru" \
    "pip install $PKG --break-system-packages" \
    "sudo systemctl restart mycoswarm"

# --- uncho (light, system-wide) ---
update_node "uncho" "minotaur@uncho" \
    "pip install $PKG --break-system-packages" \
    "sudo systemctl restart mycoswarm"

# --- pi (edge, user=pi, venv) ---
update_node "pi" "pi@pi" \
    "cd ~/mycoSwarm && source .venv/bin/activate && pip install $PKG" \
    "sudo systemctl restart mycoswarm"

# --- Restart Miu daemon ---
echo ""
echo "━━━ Miu daemon restart ━━━"
if sudo systemctl restart mycoswarm 2>&1; then
    ok "Miu: daemon restarted"
else
    warn "Miu: restart failed"
    FAILED_NODES+=("Miu")
fi

# --- Summary ---
echo ""
echo "========================="
if [ ${#FAILED_NODES[@]} -eq 0 ]; then
    echo -e "${GREEN}🍄 All nodes updated successfully${NC}"
else
    echo -e "${RED}⚠️  Failed nodes: ${FAILED_NODES[*]}${NC}"
    echo "  Check connectivity and retry manually."
fi

# --- Verify swarm ---
echo ""
echo "Run 'mycoswarm swarm' to verify all nodes are online."
