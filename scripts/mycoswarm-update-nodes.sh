#!/usr/bin/env bash
# mycoswarm-update-nodes.sh — Update all swarm nodes to latest PyPI version
# Usage: ./scripts/mycoswarm-update-nodes.sh [version]
#   version: optional, e.g. "0.3.0" — defaults to latest on PyPI
#
# ALL nodes use venv at ~/mycoSwarm/.venv
# The systemd service runs: /home/{user}/mycoSwarm/.venv/bin/mycoswarm daemon
#
# Node inventory (edit if nodes change):
#   Miu      — local dev install (pip install -e .), skip PyPI update
#   rushuna  — RTX 3060 specialist, user=minotaur
#   boa      — light node, user=minotaur
#   naru     — light node, user=minotaur
#   uncho    — light node, user=minotaur
#   pi       — Raspberry Pi edge, user=pi

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

echo "🍄 mycoSwarm Node Updater"
echo "========================="
echo "  Package: $PKG"
echo ""

read -s -p "  Enter sudo password for minotaur nodes: " SUDO_PASS
echo ""
read -s -p "  Enter password for pi@pi (Enter to skip pi): " PI_PASS
echo ""
echo ""

# --- Helper: update a venv node ---
update_node() {
    local name="$1"
    local ssh_target="$2"
    local venv_dir="$3"
    local sudo_pass="$4"

    echo ""
    echo "━━━ $name ━━━"

    # Check connectivity
    if ! ssh -o ConnectTimeout=5 -o BatchMode=yes "$ssh_target" "echo ok" &>/dev/null; then
        fail "$name: SSH connection failed (run: ssh-copy-id $ssh_target)"
        FAILED_NODES+=("$name")
        return
    fi

    # Install into venv
    if ssh "$ssh_target" "cd $venv_dir && source .venv/bin/activate && pip install $PKG" 2>&1; then
        ok "$name: package updated"
    else
        fail "$name: pip install failed"
        FAILED_NODES+=("$name")
        return
    fi

    # Restart daemon
    if ssh "$ssh_target" "echo '$sudo_pass' | sudo -S systemctl restart mycoswarm" 2>/dev/null; then
        ok "$name: daemon restarted"
    else
        fail "$name: restart failed"
        FAILED_NODES+=("$name")
        return
    fi

    # Verify version
    local remote_ver
    remote_ver=$(ssh "$ssh_target" "cd $venv_dir && source .venv/bin/activate && python3 -c 'import mycoswarm; print(mycoswarm.__version__)'" 2>/dev/null || echo "unknown")
    echo "  📦 $name: v$remote_ver"
}

# ===================================================================
# UPDATE SEQUENCE
# ===================================================================

# --- Miu (local dev install) ---
echo "━━━ Miu (local) ━━━"
echo "  Miu uses dev install (pip install -e .) — skipping PyPI update."
echo "  To update Miu: git pull && pip install -e ."
LOCAL_VER=$(python3 -c "import mycoswarm; print(mycoswarm.__version__)" 2>/dev/null || echo "unknown")
echo "  📦 Miu: v$LOCAL_VER"

# --- All minotaur nodes ---
update_node "rushuna" "minotaur@rushuna" "~/mycoSwarm" "$SUDO_PASS"
update_node "boa"     "minotaur@boa"     "~/mycoSwarm" "$SUDO_PASS"
update_node "naru"    "minotaur@naru"    "~/mycoSwarm" "$SUDO_PASS"
update_node "uncho"   "minotaur@uncho"   "~/mycoSwarm" "$SUDO_PASS"

# --- pi (different user) ---
echo ""
echo "━━━ pi ━━━"
if [ -z "$PI_PASS" ]; then
    warn "pi: skipped (no password provided)"
else
    if command -v sshpass &>/dev/null; then
        PI_SSH="sshpass -p $PI_PASS ssh -o StrictHostKeyChecking=no pi@pi"

        if ! eval $PI_SSH "echo ok" &>/dev/null; then
            fail "pi: SSH connection failed"
            FAILED_NODES+=("pi")
        else
            if eval $PI_SSH "cd ~/mycoSwarm && source .venv/bin/activate && pip install $PKG" 2>&1; then
                ok "pi: package updated"
            else
                fail "pi: pip install failed"
                FAILED_NODES+=("pi")
            fi

            if [[ ! " ${FAILED_NODES[*]:-} " =~ " pi " ]]; then
                if eval $PI_SSH "echo '$PI_PASS' | sudo -S systemctl restart mycoswarm" 2>/dev/null; then
                    ok "pi: daemon restarted"
                else
                    fail "pi: restart failed"
                    FAILED_NODES+=("pi")
                fi

                pi_ver=$(eval $PI_SSH "cd ~/mycoSwarm && source .venv/bin/activate && python3 -c 'import mycoswarm; print(mycoswarm.__version__)'" 2>/dev/null || echo "unknown")
                echo "  📦 pi: v$pi_ver"
            fi
        fi
    else
        # No sshpass — try BatchMode (key auth)
        if ssh -o ConnectTimeout=5 -o BatchMode=yes "pi@pi" "echo ok" &>/dev/null; then
            update_node "pi" "pi@pi" "~/mycoSwarm" "$PI_PASS"
        else
            warn "pi: no key auth and sshpass not installed"
            warn "  Fix with: ssh-copy-id pi@pi  OR  sudo apt install sshpass"
            FAILED_NODES+=("pi")
        fi
    fi
fi

# --- Restart Miu daemon ---
echo ""
echo "━━━ Miu daemon restart ━━━"
if echo "$SUDO_PASS" | sudo -S systemctl restart mycoswarm 2>/dev/null; then
    ok "Miu: daemon restarted"
else
    fail "Miu: restart failed"
    FAILED_NODES+=("Miu")
fi

# --- Swarm check ---
echo ""
echo "━━━ Swarm verification (waiting 5s for nodes to rejoin) ━━━"
sleep 5
mycoswarm swarm 2>/dev/null || warn "Could not run swarm status check"

# --- Summary ---
echo ""
echo "========================="
if [ ${#FAILED_NODES[@]} -eq 0 ]; then
    echo -e "${GREEN}🍄 All nodes updated successfully${NC}"
else
    echo -e "${RED}⚠️  Failed nodes: ${FAILED_NODES[*]}${NC}"
    echo "  Check connectivity and retry manually."
fi
