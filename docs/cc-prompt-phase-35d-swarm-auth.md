# Phase 35d: Swarm Authentication — Join Token

## What This Is

Right now any device on your LAN can POST to `/task` and submit work to
Monica. There's no authentication. A rogue device, a misconfigured service,
or anyone on your WiFi can make her do things.

This adds a shared secret (join token) that every node must present on
every API request. No token = no access.

## How It Works

1. First time `mycoswarm daemon` runs, if no token exists, generate one
   and save it to `~/.config/mycoswarm/swarm-token`
2. Every outbound request to a peer includes the token in a header
3. Every inbound API request is validated against the local token
4. All nodes in the swarm share the same token (you copy it once during setup)

## Create/modify: src/mycoswarm/auth.py

```python
"""
Swarm Authentication — shared join token (Phase 35d)

Every peer API request must include a valid token. No token = 403.
Token is generated on first run and stored locally. All nodes in
the swarm must share the same token.
"""

import os
import secrets
import logging

logger = logging.getLogger(__name__)

TOKEN_PATH = os.path.expanduser("~/.config/mycoswarm/swarm-token")
_HEADER_NAME = "X-Swarm-Token"


def _ensure_config_dir():
    os.makedirs(os.path.dirname(TOKEN_PATH), exist_ok=True)


def generate_token() -> str:
    """Generate a new 32-byte hex token."""
    return secrets.token_hex(32)


def load_token() -> str | None:
    """Load token from disk. Returns None if not found."""
    try:
        with open(TOKEN_PATH, "r") as f:
            token = f.read().strip()
            return token if token else None
    except FileNotFoundError:
        return None


def save_token(token: str) -> None:
    """Save token to disk with restrictive permissions."""
    _ensure_config_dir()
    with open(TOKEN_PATH, "w") as f:
        f.write(token)
    os.chmod(TOKEN_PATH, 0o600)  # owner read/write only
    logger.info("Swarm token saved to %s", TOKEN_PATH)


def ensure_token() -> str:
    """Load existing token or generate and save a new one."""
    token = load_token()
    if token is None:
        token = generate_token()
        save_token(token)
        logger.info("Generated new swarm token")
    return token


def get_auth_header(token: str) -> dict:
    """Return header dict for outbound requests."""
    return {_HEADER_NAME: token}


def validate_request(request_token: str | None, local_token: str) -> bool:
    """Check if request token matches local token. Constant-time comparison."""
    if request_token is None:
        return False
    return secrets.compare_digest(request_token, local_token)


# Header name exposed for FastAPI dependency
HEADER_NAME = _HEADER_NAME
```

## Modify: src/mycoswarm/api.py

Add a FastAPI dependency that validates the token on every request:

```python
from fastapi import Depends, HTTPException, Request
from mycoswarm.auth import validate_request, HEADER_NAME

# Store token reference at app startup (passed from daemon)
_swarm_token: str | None = None

def set_swarm_token(token: str):
    global _swarm_token
    _swarm_token = token

async def verify_token(request: Request):
    """FastAPI dependency — validates swarm token on every request."""
    if _swarm_token is None:
        return  # auth not configured, allow (backward compat)
    request_token = request.headers.get(HEADER_NAME)
    if not validate_request(request_token, _swarm_token):
        raise HTTPException(status_code=403, detail="Invalid or missing swarm token")
```

Then add the dependency to the router or individual endpoints. The cleanest
approach is a router-level dependency so ALL endpoints are protected:

```python
from fastapi import APIRouter

router = APIRouter(dependencies=[Depends(verify_token)])
```

**IMPORTANT:** Exempt the `/health` endpoint from auth so basic connectivity
checks still work without a token:

```python
@app.get("/health")  # NOT on the authenticated router
async def health():
    return {"status": "ok"}
```

## Modify: src/mycoswarm/daemon.py

At startup, load/generate the token and pass it to the API:

```python
from mycoswarm.auth import ensure_token
from mycoswarm.api import set_swarm_token

# During daemon startup, before uvicorn:
swarm_token = ensure_token()
set_swarm_token(swarm_token)
logger.info("Swarm authentication enabled")
```

## Modify: outbound peer requests

Everywhere the orchestrator or CLI makes HTTP requests to peers, include
the auth header. Search for `httpx` calls that hit peer URLs and add:

```python
from mycoswarm.auth import get_auth_header

# When making requests to peers:
headers = get_auth_header(swarm_token)
response = httpx.post(peer_url + "/task", json=payload, headers=headers)
```

Key places to check:
- `orchestrator.py` — `_route_remote()` or wherever it dispatches to peers
- `cli.py` — `_stream_response()` or wherever it connects to the API
- Any other peer-to-peer HTTP calls

The CLI talks to the local daemon, so it also needs the token. Load it
at CLI startup:

```python
from mycoswarm.auth import load_token, get_auth_header

# At top of CLI commands that hit the API:
_swarm_token = load_token()
_auth_headers = get_auth_header(_swarm_token) if _swarm_token else {}
```

## Add: /token slash command

```python
elif user_input.strip().lower() == "/token":
    from mycoswarm.auth import load_token, TOKEN_PATH
    token = load_token()
    if token:
        # Show first and last 4 chars only
        masked = token[:4] + "..." + token[-4:]
        print(f"\n🔑 Swarm token: {masked}")
        print(f"   Path: {TOKEN_PATH}")
        print(f"   To add a node: copy this file to the new node's")
        print(f"   ~/.config/mycoswarm/swarm-token")
    else:
        print("\n🔑 No swarm token configured.")
    print()
    continue
```

## Node setup workflow

After this lands, adding a new node to the swarm:

```bash
# On Miu (or any existing node):
cat ~/.config/mycoswarm/swarm-token

# On new node:
mkdir -p ~/.config/mycoswarm
echo "PASTE_TOKEN_HERE" > ~/.config/mycoswarm/swarm-token
chmod 600 ~/.config/mycoswarm/swarm-token
sudo systemctl restart mycoswarm
```

Document this in README or a setup guide.

## Backward Compatibility

- If no token file exists on a node, `ensure_token()` generates one on
  first daemon start
- If `_swarm_token` is None in the API (old node without auth), requests
  pass through — this prevents breaking existing deployments
- Nodes with mismatched tokens will get 403 errors and show as unreachable
- `/health` stays unauthenticated for basic connectivity checks

The migration path: after upgrading all nodes, copy the token from Miu to
every other node and restart. Until you do, nodes with no token still work
(backward compat mode). Once all nodes have tokens, auth is enforced.

## Tests: tests/test_auth.py

```python
"""Tests for swarm authentication (Phase 35d)."""

import os
import pytest
import tempfile
from unittest.mock import patch
from mycoswarm.auth import (
    generate_token,
    load_token,
    save_token,
    ensure_token,
    get_auth_header,
    validate_request,
    HEADER_NAME,
)


class TestTokenGeneration:
    def test_generates_64_char_hex(self):
        token = generate_token()
        assert len(token) == 64
        assert all(c in "0123456789abcdef" for c in token)

    def test_generates_unique_tokens(self):
        t1 = generate_token()
        t2 = generate_token()
        assert t1 != t2


class TestTokenStorage:
    def test_save_and_load(self, tmp_path):
        path = str(tmp_path / "swarm-token")
        with patch("mycoswarm.auth.TOKEN_PATH", path):
            save_token("test_token_123")
            assert load_token() == "test_token_123"

    def test_load_missing_returns_none(self, tmp_path):
        path = str(tmp_path / "nonexistent")
        with patch("mycoswarm.auth.TOKEN_PATH", path):
            assert load_token() is None

    def test_save_sets_permissions(self, tmp_path):
        path = str(tmp_path / "swarm-token")
        with patch("mycoswarm.auth.TOKEN_PATH", path):
            save_token("test_token")
            mode = os.stat(path).st_mode & 0o777
            assert mode == 0o600

    def test_ensure_generates_if_missing(self, tmp_path):
        path = str(tmp_path / "swarm-token")
        with patch("mycoswarm.auth.TOKEN_PATH", path):
            token = ensure_token()
            assert len(token) == 64
            # Second call returns same token
            assert ensure_token() == token

    def test_ensure_loads_existing(self, tmp_path):
        path = str(tmp_path / "swarm-token")
        with patch("mycoswarm.auth.TOKEN_PATH", path):
            save_token("existing_token")
            assert ensure_token() == "existing_token"


class TestValidation:
    def test_valid_token(self):
        assert validate_request("abc123", "abc123") is True

    def test_invalid_token(self):
        assert validate_request("wrong", "abc123") is False

    def test_none_token(self):
        assert validate_request(None, "abc123") is False

    def test_empty_token(self):
        assert validate_request("", "abc123") is False

    def test_constant_time(self):
        """validate_request uses secrets.compare_digest (constant-time)."""
        # This is a design test — just verify it doesn't short-circuit
        # on first character mismatch
        assert validate_request("a" * 64, "b" * 64) is False


class TestAuthHeader:
    def test_header_format(self):
        headers = get_auth_header("my_token")
        assert headers == {HEADER_NAME: "my_token"}

    def test_header_name(self):
        assert HEADER_NAME == "X-Swarm-Token"
```

## Smoke Tests (run after implementation)

### Test 1: Token auto-generated on first run
```bash
# Delete existing token if any:
rm ~/.config/mycoswarm/swarm-token
mycoswarm daemon &
cat ~/.config/mycoswarm/swarm-token
# Should exist and be 64 hex chars
kill %1
```

### Test 2: /token command
```
mycoswarm chat
you> /token
```
Should show masked token and file path.

### Test 3: Peer rejected without token
```bash
# From any machine, try hitting the API without a token:
curl -s http://miu:7890/status
# Should get 403

# With token:
curl -s -H "X-Swarm-Token: $(cat ~/.config/mycoswarm/swarm-token)" http://miu:7890/status
# Should get 200
```

### Test 4: /health still works without token
```bash
curl -s http://miu:7890/health
# Should get 200 {"status": "ok"} — no auth required
```

### Test 5: Swarm still works after token sync
Copy token to all nodes, restart daemons, then:
```bash
mycoswarm swarm
```
All nodes should show online.

## PLAN.md Update

Mark Phase 35d items as done after implementation:
- [x] Join token generated on first daemon run
- [x] Token validated on all peer API requests (except /health)
- [x] Outbound peer requests include token header
- [x] /token slash command
- [x] Tests in tests/test_auth.py
- [x] Backward compatible — nodes without token still work until migration
