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
