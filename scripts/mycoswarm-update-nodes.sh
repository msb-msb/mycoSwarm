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
#   luvia    — light node, user=minotaur
#   pi       — RETIRED: Raspberry Pi 2 (1GB RAM, 32-bit ARM), manifesto demo, not connected

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
echo ""
# pi is retired (Raspberry Pi 2, not connected) — see the commented-out block
# near the end of this script. No pi@pi password prompt needed.
echo ""

# --- Helper: disable Ubuntu unattended-upgrades on a node (idempotent) ---
# ROOT CAUSE we traced on mai: apt-daily-upgrade.service triggers a systemd
# `daemon-reexec`; on Ubuntu 24 PID 1 can hang mid-re-exec — userspace dies while
# the kernel stays alive (node still pings, port 22 accepts on a stale listen
# backlog with no live sshd behind it, ACPI power button dead because logind is
# gone). Only a hard power cut recovers it. These nodes are LAN-only and patched
# manually via this script, so we disable + mask the auto-upgrade machinery.
# The password is fed on stdin (never argv/disk/logs). Safe to run every time —
# no-ops when already hardened, so it self-heals if apt ever pulls the timers back.
harden_node() {
    local name="$1"
    local ssh_target="$2"
    local sudo_pass="$3"

    local script result
    script=$(cat <<'HS'
systemctl disable --now apt-daily.timer apt-daily-upgrade.timer unattended-upgrades.service >/dev/null 2>&1
systemctl mask apt-daily.service apt-daily-upgrade.service >/dev/null 2>&1
printf '%s\n' 'APT::Periodic::Update-Package-Lists "0";' 'APT::Periodic::Unattended-Upgrade "0";' > /etc/apt/apt.conf.d/20auto-upgrades
if systemctl list-timers --all 2>/dev/null | grep -qi apt; then echo STILL_ARMED; else echo CLEAR; fi
HS
)
    result=$(printf '%s\n%s\n' "$sudo_pass" "$script" \
        | ssh -o ConnectTimeout=10 -o ServerAliveInterval=15 "$ssh_target" "sudo -S -p '' bash -s" 2>/dev/null)

    if echo "$result" | grep -q CLEAR; then
        ok "$name: auto-upgrades disabled (apt timers masked)"
    elif echo "$result" | grep -q STILL_ARMED; then
        warn "$name: hardening ran but an apt timer is still present"
    else
        warn "$name: could not disable auto-upgrades (ssh/sudo failed) — non-fatal"
    fi
}

# --- Helper: update a venv node ---
update_node() {
    local name="$1"
    local ssh_target="$2"
    local venv_dir="$3"
    local sudo_pass="$4"

    echo ""
    echo "━━━ $name ━━━"

    # Check connectivity
    if ! ssh -o ConnectTimeout=5 -o ServerAliveInterval=15 -o BatchMode=yes "$ssh_target" "echo ok" &>/dev/null; then
        fail "$name: SSH connection failed (run: ssh-copy-id $ssh_target)"
        FAILED_NODES+=("$name")
        return
    fi

    # Harden first — keep auto-upgrades disabled so this node can't silently
    # wedge itself in the apt-daily window (see harden_node above). Idempotent.
    harden_node "$name" "$ssh_target" "$sudo_pass"

    # Verify venv exists
    if ! ssh -o ConnectTimeout=10 "$ssh_target" "test -f $venv_dir/.venv/bin/activate" &>/dev/null; then
        warn "$name: .venv not found — creating"
        if ! ssh -o ConnectTimeout=10 -o ServerAliveInterval=15 "$ssh_target" "cd $venv_dir && python3 -m venv .venv" 2>&1; then
            fail "$name: venv creation failed"
            FAILED_NODES+=("$name")
            return
        fi
        ok "$name: .venv created"
    fi

    # Install into venv (quiet + timeout to prevent hangs)
    if ssh -o ConnectTimeout=10 -o ServerAliveInterval=15 -o ServerAliveCountMax=4 "$ssh_target" "cd $venv_dir && source .venv/bin/activate && pip install -q $PKG" 2>&1; then
        ok "$name: package updated"
    else
        fail "$name: pip install failed"
        FAILED_NODES+=("$name")
        return
    fi

    # Restart daemon
    if ssh -o ConnectTimeout=10 -o ServerAliveInterval=15 "$ssh_target" "echo '$sudo_pass' | sudo -S systemctl restart mycoswarm" 2>/dev/null; then
        ok "$name: daemon restarted"
    else
        fail "$name: restart failed"
        FAILED_NODES+=("$name")
        return
    fi

    # Verify version
    local remote_ver
    remote_ver=$(ssh -o ConnectTimeout=10 "$ssh_target" "cd $venv_dir && source .venv/bin/activate && python3 -c 'import mycoswarm; print(mycoswarm.__version__)'" 2>/dev/null || echo "unknown")
    echo "  📦 $name: v$remote_ver"
}

# ===================================================================
# UPDATE SEQUENCE
# ===================================================================

# --- Miu (local — now updates from PyPI like other nodes) ---
echo "━━━ Miu (local) ━━━"
cd ~/Desktop/mycoSwarm
if [ ! -f .venv/bin/activate ]; then
    warn "Miu: .venv not found — creating"
    python3 -m venv .venv
    ok "Miu: .venv created"
fi
source .venv/bin/activate
if pip install -q $PKG 2>&1; then
    ok "Miu: package updated"
else
    fail "Miu: pip install failed"
    FAILED_NODES+=("Miu")
fi
LOCAL_VER=$(python3 -c "import mycoswarm; print(mycoswarm.__version__)" 2>/dev/null || echo "unknown")
echo "  📦 Miu: v$LOCAL_VER"

# --- All minotaur nodes ---
update_node "rushuna" "minotaur@rushuna" "~/mycoSwarm" "$SUDO_PASS"
update_node "boa"     "minotaur@boa"     "~/mycoSwarm" "$SUDO_PASS"
update_node "naru"    "minotaur@naru"    "~/mycoSwarm" "$SUDO_PASS"
update_node "uncho"   "minotaur@uncho"   "~/mycoSwarm" "$SUDO_PASS"
update_node "luvia"   "minotaur@luvia"   "~/mycoSwarm" "$SUDO_PASS"

# --- pi (RETIRED — commented out, not deleted) ---
# pi was a Raspberry Pi 2 (192.168.50.16, user=pi): 1GB RAM, 32-bit ARM. It was a
# MANIFESTO demo ("a student with two old laptops can participate"), never a
# workhorse, and is not physically connected and won't be. The loop below would
# just try it and fail every run, so it's disabled. Kept as a record; re-enable by
# uncommenting this block and restoring the PI_PASS prompt near the top.
#
# echo ""
# echo "━━━ pi ━━━"
# if [ -z "$PI_PASS" ]; then
#     warn "pi: skipped (no password provided)"
# else
#     if command -v sshpass &>/dev/null; then
#         PI_SSH="sshpass -p $PI_PASS ssh -o StrictHostKeyChecking=no pi@pi"
#
#         if ! eval $PI_SSH "echo ok" &>/dev/null; then
#             fail "pi: SSH connection failed"
#             FAILED_NODES+=("pi")
#         else
#             if eval $PI_SSH "cd ~/mycoSwarm && source .venv/bin/activate && pip install $PKG" 2>&1; then
#                 ok "pi: package updated"
#             else
#                 fail "pi: pip install failed"
#                 FAILED_NODES+=("pi")
#             fi
#
#             if [[ ! " ${FAILED_NODES[*]:-} " =~ " pi " ]]; then
#                 if eval $PI_SSH "echo '$PI_PASS' | sudo -S systemctl restart mycoswarm" 2>/dev/null; then
#                     ok "pi: daemon restarted"
#                 else
#                     fail "pi: restart failed"
#                     FAILED_NODES+=("pi")
#                 fi
#
#                 pi_ver=$(eval $PI_SSH "cd ~/mycoSwarm && source .venv/bin/activate && python3 -c 'import mycoswarm; print(mycoswarm.__version__)'" 2>/dev/null || echo "unknown")
#                 echo "  📦 pi: v$pi_ver"
#             fi
#         fi
#     else
#         # No sshpass — try BatchMode (key auth)
#         if ssh -o ConnectTimeout=5 -o BatchMode=yes "pi@pi" "echo ok" &>/dev/null; then
#             update_node "pi" "pi@pi" "~/mycoSwarm" "$PI_PASS"
#         else
#             warn "pi: no key auth and sshpass not installed"
#             warn "  Fix with: ssh-copy-id pi@pi  OR  sudo apt install sshpass"
#             FAILED_NODES+=("pi")
#         fi
#     fi
# fi

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
