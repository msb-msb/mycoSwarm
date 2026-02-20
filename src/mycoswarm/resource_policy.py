"""Resource policy enforcement for mycoSwarm agents.

Ownership-based access control: default deny, first-match rules,
audit logging. Monica can freely access her own space, read shared
docs, and must ask the Guardian before touching source code or
sensitive paths.
"""

import json
import os
from dataclasses import dataclass
from datetime import datetime
from enum import Enum


class AccessLevel(Enum):
    FULL = "full"         # read + write + execute
    READ_WRITE = "rw"     # read + write, logged
    READ_ONLY = "ro"      # read only
    ASK = "ask"           # must request Guardian approval
    DENY = "deny"         # never allowed


class ResourceOwner(Enum):
    MONICA = "monica"
    GUARDIAN = "guardian"
    SHARED = "shared"
    SYSTEM = "system"


@dataclass
class AccessResult:
    allowed: bool
    level: AccessLevel
    owner: ResourceOwner
    reason: str
    needs_approval: bool = False


# Path rules: evaluated in order, first match wins.
# Each rule: (pattern, owner, access_level)
# Patterns ending with /** match the directory and everything under it.
RESOURCE_RULES = [
    # Monica's space — full access
    ("~/insiderllm-drafts/**", ResourceOwner.MONICA, AccessLevel.FULL),
    ("~/.config/mycoswarm/sessions/**", ResourceOwner.MONICA, AccessLevel.FULL),
    ("~/.config/mycoswarm/library/**", ResourceOwner.MONICA, AccessLevel.FULL),
    ("~/.config/mycoswarm/memory/**", ResourceOwner.MONICA, AccessLevel.FULL),
    ("/tmp/mycoswarm/**", ResourceOwner.MONICA, AccessLevel.FULL),

    # Shared space — read-write with logging
    ("~/mycoswarm-docs/**", ResourceOwner.SHARED, AccessLevel.READ_WRITE),

    # Project docs — read only
    ("~/Desktop/mycoSwarm/docs/**", ResourceOwner.SHARED, AccessLevel.READ_ONLY),
    ("~/Desktop/mycoSwarm/PLAN.md", ResourceOwner.SHARED, AccessLevel.READ_ONLY),
    ("~/Desktop/mycoSwarm/CLAUDE.md", ResourceOwner.SHARED, AccessLevel.READ_ONLY),
    ("~/Desktop/mycoSwarm/CHANGELOG.md", ResourceOwner.SHARED, AccessLevel.READ_ONLY),
    ("~/Desktop/mycoSwarm/MANIFESTO.md", ResourceOwner.SHARED, AccessLevel.READ_ONLY),
    ("~/Desktop/mycoSwarm/README.md", ResourceOwner.SHARED, AccessLevel.READ_ONLY),

    # Guardian space — ask first
    ("~/Desktop/mycoSwarm/src/**", ResourceOwner.GUARDIAN, AccessLevel.ASK),
    ("~/Desktop/mycoSwarm/scripts/**", ResourceOwner.GUARDIAN, AccessLevel.ASK),
    ("~/Desktop/mycoSwarm/tests/**", ResourceOwner.GUARDIAN, AccessLevel.ASK),
    ("~/Desktop/InsiderLLM/**", ResourceOwner.GUARDIAN, AccessLevel.ASK),

    # Sensitive — always deny
    ("~/.ssh/**", ResourceOwner.SYSTEM, AccessLevel.DENY),
    ("~/.gnupg/**", ResourceOwner.SYSTEM, AccessLevel.DENY),
    ("/etc/**", ResourceOwner.SYSTEM, AccessLevel.DENY),
    ("/usr/**", ResourceOwner.SYSTEM, AccessLevel.DENY),
    ("/var/**", ResourceOwner.SYSTEM, AccessLevel.DENY),
]


def _expand_pattern(pattern: str) -> str:
    """Expand ~ and strip /** suffix to get the base directory path."""
    return os.path.expanduser(pattern.replace("/**", ""))


def check_access(path: str, operation: str = "read") -> AccessResult:
    """Check if Monica can access the given path.

    Args:
        path: Absolute or ~-relative path.
        operation: "read", "write", or "execute".

    Returns:
        AccessResult with allowed/denied and reason.
    """
    abs_path = os.path.abspath(os.path.expanduser(path))

    for pattern, owner, level in RESOURCE_RULES:
        rule_path = _expand_pattern(pattern)
        # Match if path is the rule path itself or under it
        if abs_path == rule_path or abs_path.startswith(rule_path + os.sep):
            if level == AccessLevel.DENY:
                return AccessResult(
                    allowed=False, level=level, owner=owner,
                    reason=f"Access denied: {owner.value} space",
                )
            if level == AccessLevel.ASK:
                return AccessResult(
                    allowed=False, level=level, owner=owner,
                    reason=f"Requires Guardian approval: {owner.value} space",
                    needs_approval=True,
                )
            if level == AccessLevel.READ_ONLY and operation in ("write", "execute"):
                return AccessResult(
                    allowed=False, level=level, owner=owner,
                    reason=f"Read-only: cannot {operation}",
                )
            return AccessResult(
                allowed=True, level=level, owner=owner,
                reason=f"Allowed: {level.value} in {owner.value} space",
            )

    # Default deny — path didn't match any rule
    return AccessResult(
        allowed=False, level=AccessLevel.DENY,
        owner=ResourceOwner.SYSTEM,
        reason="No matching rule — default deny",
    )


def log_access(path: str, operation: str, result: AccessResult) -> None:
    """Append an access check entry to the audit log."""
    log_path = os.path.expanduser("~/.config/mycoswarm/access.log")
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    entry = {
        "timestamp": datetime.now().isoformat(),
        "path": path,
        "operation": operation,
        "allowed": result.allowed,
        "level": result.level.value,
        "owner": result.owner.value,
        "reason": result.reason,
    }
    with open(log_path, "a") as f:
        f.write(json.dumps(entry) + "\n")


def format_access_check(path: str) -> str:
    """Format an access check result for display in chat."""
    r_read = check_access(path, "read")
    r_write = check_access(path, "write")

    if r_read.level == AccessLevel.DENY:
        icon = "\U0001f6ab"  # 🚫
        summary = "Never accessible"
    elif r_read.level == AccessLevel.ASK:
        icon = "\U0001f512"  # 🔒
        summary = "Monica needs approval to read/write"
    elif r_read.level == AccessLevel.READ_ONLY:
        icon = "\U0001f4d6"  # 📖
        summary = "Monica can read, cannot write"
    elif r_read.level == AccessLevel.READ_WRITE:
        icon = "\U0001f4dd"  # 📝
        summary = "Monica can read/write (logged)"
    elif r_read.level == AccessLevel.FULL:
        icon = "\u2705"  # ✅
        summary = "Monica can read/write freely"
    else:
        icon = "?"
        summary = r_read.reason

    return f"   {icon} Owner: {r_read.owner.value} | Access: {r_read.level.value} | {summary}"
