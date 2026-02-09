#!/usr/bin/env bash
# mycoSwarm installer
# Usage: curl -fsSL https://mycoswarm.org/install.sh | bash
#
# Installs Python 3, Ollama, mycoswarm, and pulls a default model
# sized for your hardware. Works on Linux (apt/dnf) and macOS (brew).

set -euo pipefail

# --- Helpers ---

info()  { printf '\033[1;34m==>\033[0m %s\n' "$*"; }
ok()    { printf '\033[1;32m  ✓\033[0m %s\n' "$*"; }
warn()  { printf '\033[1;33m  !\033[0m %s\n' "$*"; }
fail()  { printf '\033[1;31m  ✗\033[0m %s\n' "$*"; exit 1; }

need_cmd() {
    command -v "$1" >/dev/null 2>&1
}

# --- Detect OS ---

info "Detecting operating system..."

OS="$(uname -s)"
case "$OS" in
    Linux)  PLATFORM="linux" ;;
    Darwin) PLATFORM="macos" ;;
    *)      fail "Unsupported OS: $OS (need Linux or macOS)" ;;
esac

ARCH="$(uname -m)"
ok "$OS $ARCH"

# --- Detect package manager ---

if [ "$PLATFORM" = "linux" ]; then
    if need_cmd apt-get; then
        PKG="apt"
    elif need_cmd dnf; then
        PKG="dnf"
    elif need_cmd pacman; then
        PKG="pacman"
    else
        PKG="none"
    fi
elif [ "$PLATFORM" = "macos" ]; then
    if need_cmd brew; then
        PKG="brew"
    else
        fail "Homebrew not found. Install it first: https://brew.sh"
    fi
fi

# --- Python 3.10+ ---

info "Checking Python..."

PYTHON=""
for candidate in python3.13 python3.12 python3.11 python3.10 python3; do
    if need_cmd "$candidate"; then
        PY_VERSION=$("$candidate" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")' 2>/dev/null || true)
        PY_MAJOR=$(echo "$PY_VERSION" | cut -d. -f1)
        PY_MINOR=$(echo "$PY_VERSION" | cut -d. -f2)
        if [ "${PY_MAJOR:-0}" -ge 3 ] && [ "${PY_MINOR:-0}" -ge 10 ]; then
            PYTHON="$candidate"
            break
        fi
    fi
done

if [ -z "$PYTHON" ]; then
    info "Installing Python..."
    case "$PKG" in
        apt)
            sudo apt-get update -qq
            sudo apt-get install -y -qq python3 python3-pip python3-venv
            ;;
        dnf)
            sudo dnf install -y -q python3 python3-pip
            ;;
        pacman)
            sudo pacman -Sy --noconfirm python python-pip
            ;;
        brew)
            brew install python@3.12
            ;;
        *)
            fail "No supported package manager found. Install Python 3.10+ manually."
            ;;
    esac

    # Re-detect after install
    for candidate in python3.13 python3.12 python3.11 python3.10 python3; do
        if need_cmd "$candidate"; then
            PYTHON="$candidate"
            break
        fi
    done
    [ -z "$PYTHON" ] && fail "Python installation failed."
fi

PY_VERSION=$("$PYTHON" --version 2>&1 | awk '{print $2}')
ok "Python $PY_VERSION ($PYTHON)"

# --- pip ---

info "Checking pip..."

if ! "$PYTHON" -m pip --version >/dev/null 2>&1; then
    info "Installing pip..."
    case "$PKG" in
        apt)    sudo apt-get install -y -qq python3-pip ;;
        dnf)    sudo dnf install -y -q python3-pip ;;
        pacman) sudo pacman -Sy --noconfirm python-pip ;;
        brew)   : ;; # brew python includes pip
        *)      "$PYTHON" -m ensurepip --default-pip 2>/dev/null || fail "Cannot install pip." ;;
    esac
fi

ok "pip available"

# --- Ollama ---

info "Checking Ollama..."

if need_cmd ollama; then
    ok "Ollama already installed"
else
    info "Installing Ollama..."
    if [ "$PLATFORM" = "macos" ]; then
        brew install ollama
    else
        curl -fsSL https://ollama.com/install.sh | sh
    fi
    need_cmd ollama || fail "Ollama installation failed."
    ok "Ollama installed"
fi

# --- Start Ollama if not running ---

if curl -sf --max-time 2 http://localhost:11434/api/tags >/dev/null 2>&1; then
    ok "Ollama is running"
else
    info "Starting Ollama..."
    if [ "$PLATFORM" = "linux" ] && need_cmd systemctl; then
        # systemd-managed Ollama (standard Linux install)
        sudo systemctl start ollama 2>/dev/null || true
        sleep 2
    fi

    # If systemd didn't work or on macOS, start in background
    if ! curl -sf --max-time 2 http://localhost:11434/api/tags >/dev/null 2>&1; then
        ollama serve >/dev/null 2>&1 &
        OLLAMA_PID=$!
        # Wait up to 10 seconds for Ollama to be ready
        for i in $(seq 1 20); do
            if curl -sf --max-time 1 http://localhost:11434/api/tags >/dev/null 2>&1; then
                break
            fi
            sleep 0.5
        done
    fi

    if curl -sf --max-time 2 http://localhost:11434/api/tags >/dev/null 2>&1; then
        ok "Ollama started"
    else
        warn "Could not start Ollama automatically. Run 'ollama serve' manually."
    fi
fi

# --- Install mycoswarm ---

info "Installing mycoswarm..."

"$PYTHON" -m pip install --quiet --user mycoswarm 2>/dev/null \
    || "$PYTHON" -m pip install --quiet mycoswarm 2>/dev/null \
    || "$PYTHON" -m pip install --quiet --break-system-packages mycoswarm \
    || fail "pip install failed. Try: $PYTHON -m pip install mycoswarm"

# Ensure ~/.local/bin is in PATH for --user installs
if ! need_cmd mycoswarm; then
    LOCAL_BIN="$HOME/.local/bin"
    if [ -f "$LOCAL_BIN/mycoswarm" ]; then
        export PATH="$LOCAL_BIN:$PATH"
    fi
fi

need_cmd mycoswarm || fail "mycoswarm not found in PATH after install."
ok "mycoswarm installed"

# --- Detect RAM and pull appropriate model ---

info "Detecting hardware..."

RAM_MB=0
if [ "$PLATFORM" = "linux" ]; then
    RAM_MB=$(awk '/MemTotal/ {print int($2/1024)}' /proc/meminfo 2>/dev/null || echo 0)
elif [ "$PLATFORM" = "macos" ]; then
    RAM_BYTES=$(sysctl -n hw.memsize 2>/dev/null || echo 0)
    RAM_MB=$((RAM_BYTES / 1024 / 1024))
fi

# Pick model based on available RAM
if [ "$RAM_MB" -ge 16000 ]; then
    MODEL="gemma3:27b"
    info "16GB+ RAM detected — pulling gemma3:27b (best quality)"
elif [ "$RAM_MB" -ge 8000 ]; then
    MODEL="gemma3:4b"
    info "8-16GB RAM detected — pulling gemma3:4b (good balance)"
else
    MODEL="gemma3:1b"
    info "<8GB RAM detected — pulling gemma3:1b (lightweight)"
fi

# Check if model is already available
EXISTING=$(curl -sf http://localhost:11434/api/tags 2>/dev/null | grep -o "\"$MODEL\"" || true)
if [ -n "$EXISTING" ]; then
    ok "$MODEL already available"
else
    info "Pulling $MODEL (this may take a few minutes)..."
    ollama pull "$MODEL"
    ok "$MODEL ready"
fi

# --- Show what was detected ---

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
mycoswarm detect
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# --- Done ---

echo ""
info "Ready! Run:"
echo ""
echo "    mycoswarm chat"
echo ""
