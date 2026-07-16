#!/bin/bash
# swarm-update.sh — Upgrade mycoswarm on all swarm nodes
# Usage: ./scripts/swarm-update.sh [version]
# Example: ./scripts/swarm-update.sh 0.2.14
#          ./scripts/swarm-update.sh          (upgrades to latest)

set -e

VERSION="${1:-}"

if [ -n "$VERSION" ]; then
    PIP_SPEC="mycoswarm==$VERSION"
    echo "🍄 Upgrading swarm to mycoswarm $VERSION"
else
    PIP_SPEC="mycoswarm --upgrade"
    echo "🍄 Upgrading swarm to latest mycoswarm"
fi

echo ""

# Node definitions: name|ip|user|pip_path
NODES=(
    "Miu|localhost|minotaur|/home/minotaur/Desktop/mycoSwarm/.venv/bin/pip"
    "naru|192.168.50.15|minotaur|/home/minotaur/mycoSwarm/.venv/bin/pip"
    "boa|192.168.50.12|minotaur|/home/minotaur/mycoSwarm/.venv/bin/pip"
    "uncho|192.168.50.13|minotaur|/home/minotaur/mycoSwarm/.venv/bin/pip"
    "luvia|192.168.50.11|minotaur|/home/minotaur/mycoSwarm/.venv/bin/pip"
    "mai|192.168.50.14|minotaur|/home/minotaur/mycoSwarm/.venv/bin/pip"
    # pi RETIRED — Raspberry Pi 2 (1GB RAM, 32-bit ARM), manifesto demo, not connected:
    # "pi|192.168.50.16|pi|/home/pi/mycoSwarm/.venv/bin/pip"
)

# Prompt for sudo password once (same password for minotaur nodes)
echo -n "🔑 sudo password for minotaur nodes: "
read -s SUDO_PASS
echo ""
echo -n "🔑 sudo password for pi node (enter to use 'mycoswarm'): "
read -s PI_PASS
echo ""
PI_PASS="${PI_PASS:-mycoswarm}"

RESULTS=()

for node_def in "${NODES[@]}"; do
    IFS='|' read -r NAME IP USER PIP_PATH <<< "$node_def"

    echo ""
    echo "━━━ $NAME ($IP) ━━━"

    PASS="$SUDO_PASS"
    if [ "$USER" = "pi" ]; then
        PASS="$PI_PASS"
    fi

    if [ "$IP" = "localhost" ]; then
        # Local upgrade
        echo "  📦 Installing..."
        if [ -n "$VERSION" ]; then
            $PIP_PATH install "mycoswarm==$VERSION" --no-cache-dir 2>&1 | tail -1
        else
            $PIP_PATH install --upgrade mycoswarm --no-cache-dir 2>&1 | tail -1
        fi

        echo "  🔄 Restarting daemon..."
        echo "$PASS" | sudo -S systemctl restart mycoswarm 2>/dev/null
        sleep 2

        # Verify
        INSTALLED=$($PIP_PATH show mycoswarm 2>/dev/null | grep "^Version:" | awk '{print $2}')
        echo "  ✅ $NAME: v$INSTALLED"
        RESULTS+=("$NAME: v$INSTALLED")
    else
        # Remote upgrade via SSH
        echo "  📦 Installing..."
        if [ -n "$VERSION" ]; then
            ssh -o ConnectTimeout=5 "$USER@$IP" "$PIP_PATH install mycoswarm==$VERSION --no-cache-dir" 2>&1 | tail -1
        else
            ssh -o ConnectTimeout=5 "$USER@$IP" "$PIP_PATH install --upgrade mycoswarm --no-cache-dir" 2>&1 | tail -1
        fi

        if [ $? -ne 0 ]; then
            echo "  ❌ $NAME: install failed"
            RESULTS+=("$NAME: FAILED")
            continue
        fi

        echo "  🔄 Restarting daemon..."
        ssh -o ConnectTimeout=5 "$USER@$IP" "echo '$PASS' | sudo -S systemctl restart mycoswarm" 2>/dev/null
        sleep 2

        # Verify
        INSTALLED=$(ssh -o ConnectTimeout=5 "$USER@$IP" "$PIP_PATH show mycoswarm 2>/dev/null | grep '^Version:' | awk '{print \$2}'" 2>/dev/null)
        if [ -n "$INSTALLED" ]; then
            echo "  ✅ $NAME: v$INSTALLED"
            RESULTS+=("$NAME: v$INSTALLED")
        else
            echo "  ⚠️  $NAME: version check failed"
            RESULTS+=("$NAME: UNKNOWN")
        fi
    fi
done

echo ""
echo "━━━━━━━━━━━━━━━━━━━━"
echo "🍄 Swarm Update Summary"
echo "━━━━━━━━━━━━━━━━━━━━"
for r in "${RESULTS[@]}"; do
    echo "  $r"
done
echo ""
echo ""
echo "━━━━━━━━━━━━━━━━━━━━"
echo "📦 Don't forget:"
echo "  python -m build && twine upload dist/*"
echo "  git tag v\$(python -c 'import mycoswarm; print(mycoswarm.__version__)')"
echo "  git push --tags"
echo "━━━━━━━━━━━━━━━━━━━━"
