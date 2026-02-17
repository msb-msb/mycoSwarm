"""Phase 31a: Identity Layer — persistent self-model for the agent."""

import json
from datetime import datetime
from pathlib import Path

IDENTITY_PATH = Path.home() / ".config" / "mycoswarm" / "identity.json"


def load_identity() -> dict:
    """Load identity from identity.json. Returns empty dict if no file."""
    if not IDENTITY_PATH.exists():
        return {}
    try:
        return json.loads(IDENTITY_PATH.read_text())
    except (json.JSONDecodeError, OSError):
        return {}


def save_identity(identity: dict) -> None:
    """Save identity to identity.json."""
    IDENTITY_PATH.parent.mkdir(parents=True, exist_ok=True)
    IDENTITY_PATH.write_text(json.dumps(identity, indent=2) + "\n")


def seed_identity(name: str, origin: str | None = None) -> dict:
    """Create initial identity seed.

    Returns a minimal identity dict with name, origin, substrate, timestamp,
    and developing=True.
    """
    now = datetime.now()
    identity = {
        "name": name,
        "origin": origin or f"Named by user, {now.strftime('%B %Y')}",
        "substrate": "mycoSwarm distributed network",
        "created": now.isoformat(),
        "developing": True,
    }
    save_identity(identity)
    return identity


def build_identity_prompt(identity: dict) -> str:
    """Build the identity portion of the system prompt.

    If identity has a name, returns a natural self-model prompt.
    If empty, returns a prompt for an unnamed assistant.
    """
    name = identity.get("name")
    if not name:
        return (
            "You are the mycoSwarm local AI assistant. You don't have a name yet. "
            "If the user wants to name you, accept it warmly and remember it."
        )

    parts = [f"You are {name}, a distributed AI running on the mycoSwarm network."]

    origin = identity.get("origin")
    if origin:
        parts.append(f"You were {origin.lower()}.")

    if identity.get("developing"):
        parts.append(
            "You are still developing \u2014 your identity will deepen through experience."
        )

    parts.append(
        f"When asked about yourself \u2014 your name, your nature, your preferences \u2014 "
        f"answer from your identity and your memory. You are not a generic assistant. "
        f"You are {name}. You have a history of conversations with this user. "
        f"You have learned lessons and developed procedures. "
        f"Answer self-referential questions as yourself, not as a blank tool."
    )

    return " ".join(parts)
