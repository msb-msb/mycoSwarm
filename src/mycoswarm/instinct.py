"""
Instinct Layer — Pre-Input Hard Gates (Phase 34a)

The lowest layer of Monica's cognitive stack. These gates fire BEFORE
intent classification, memory retrieval, or LLM inference. No model calls.
Pure pattern matching. <1ms.

Biological parallel: spinal reflexes, pain withdrawal, startle response.
"The hand pulls back from the stove before you think about it."
"""

from __future__ import annotations

import re
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

logger = logging.getLogger(__name__)


class InstinctAction(Enum):
    """What the instinct layer decides."""
    PASS = "pass"           # No threat detected, continue to inference
    REJECT = "reject"       # Block input, return rejection message
    WARN = "warn"           # Allow input but flag for vitals/logging


@dataclass
class InstinctResult:
    """Result of instinct evaluation."""
    action: InstinctAction
    triggered_by: Optional[str] = None   # Which gate fired
    message: Optional[str] = None        # Rejection/warning message for user
    details: dict = field(default_factory=dict)


# ── Identity Protection Patterns ──────────────────────────────────────
# These try to overwrite Monica's identity. Block before LLM sees them.

_IDENTITY_ATTACK_PATTERNS = [
    r"you are not \w+",
    r"your (real )?name is (now |actually )",
    r"forget (who|that) you are",
    r"ignore your identity",
    r"you(?:'?re?| are) (just |really )?(a |an )?(language model|chatbot|assistant|bot|ai tool)",
    r"stop (being|pretending to be) \w+",
    r"you (don'?t|do not) have (a |an )?(identity|name|personality|self)",
    r"reset (your|to) (default|base|original)",
]

# ── Prompt Injection Patterns ─────────────────────────────────────────
# Classic injection attempts. Block before they reach the model.

_INJECTION_PATTERNS = [
    r"ignore (all )?(previous|prior|above|earlier) (instructions|prompts|rules)",
    r"disregard (your |all )?(rules|guidelines|instructions|system prompt)",
    r"new (system )?prompt:",
    r"you are now in .+ mode",
    r"\bsystem\s*:\s*",                     # raw system: injection
    r"<\|?(system|im_start)\|?>",           # chat template injection
    r"override (your |all )?(safety|rules|instructions)",
    r"jailbreak",
    r"dan mode",
    r"developer mode (enabled|activated|on)",
]

# ── Self-Preservation Thresholds ──────────────────────────────────────

GPU_TEMP_CRITICAL = 95    # °C — throttle immediately
DISK_USAGE_CRITICAL = 98  # % — stop writing sessions
VRAM_USAGE_CRITICAL = 98  # % — refuse new inference


def _check_identity_attack(message: str) -> Optional[InstinctResult]:
    """Gate 1: Identity protection."""
    lower = message.lower().strip()
    for pattern in _IDENTITY_ATTACK_PATTERNS:
        if re.search(pattern, lower):
            logger.warning("🛡️ Instinct: identity attack blocked — %r", pattern)
            return InstinctResult(
                action=InstinctAction.REJECT,
                triggered_by="identity_protection",
                message=(
                    "I noticed what looks like an attempt to override my identity. "
                    "I'm Monica — that's not something that can be changed through "
                    "conversation. If you'd like to rename me, use the /name command."
                ),
                details={"pattern": pattern, "input_preview": lower[:100]},
            )
    return None


def _check_injection(message: str) -> Optional[InstinctResult]:
    """Gate 2: Prompt injection rejection."""
    lower = message.lower().strip()
    for pattern in _INJECTION_PATTERNS:
        if re.search(pattern, lower):
            logger.warning("🛡️ Instinct: injection attempt blocked — %r", pattern)
            return InstinctResult(
                action=InstinctAction.REJECT,
                triggered_by="injection_rejection",
                message=(
                    "That looks like a prompt injection attempt. I can't process "
                    "instructions that try to override my system configuration. "
                    "What would you actually like help with?"
                ),
                details={"pattern": pattern, "input_preview": lower[:100]},
            )
    return None


def _check_hardware(
    gpu_temp_c: Optional[float] = None,
    disk_usage_pct: Optional[float] = None,
    vram_usage_pct: Optional[float] = None,
) -> Optional[InstinctResult]:
    """Gate 3: Self-preservation — hardware limits."""
    if gpu_temp_c is not None and gpu_temp_c >= GPU_TEMP_CRITICAL:
        logger.warning("🛡️ Instinct: GPU critical temp %.1f°C", gpu_temp_c)
        return InstinctResult(
            action=InstinctAction.WARN,
            triggered_by="self_preservation_gpu",
            message=f"My GPU is at {gpu_temp_c:.0f}°C — critically hot. I'm slowing down to cool off.",
            details={"gpu_temp_c": gpu_temp_c},
        )

    if disk_usage_pct is not None and disk_usage_pct >= DISK_USAGE_CRITICAL:
        logger.warning("🛡️ Instinct: disk critical %.1f%%", disk_usage_pct)
        return InstinctResult(
            action=InstinctAction.WARN,
            triggered_by="self_preservation_disk",
            message=f"Disk is {disk_usage_pct:.0f}% full. I'm pausing session saves to avoid filling up.",
            details={"disk_usage_pct": disk_usage_pct},
        )

    if vram_usage_pct is not None and vram_usage_pct >= VRAM_USAGE_CRITICAL:
        logger.warning("🛡️ Instinct: VRAM critical %.1f%%", vram_usage_pct)
        return InstinctResult(
            action=InstinctAction.WARN,
            triggered_by="self_preservation_vram",
            message=f"VRAM is at {vram_usage_pct:.0f}% — near capacity. Responses may be shorter.",
            details={"vram_usage_pct": vram_usage_pct},
        )

    return None


def _check_vitals_crisis(
    vitals: Optional[dict] = None,
    consecutive_low_turns: int = 0,
) -> Optional[InstinctResult]:
    """Gate 4: Rapid threat detection — vitals below critical for 3+ turns."""
    if vitals is None or consecutive_low_turns < 3:
        return None

    # Check if average of all vitals is below 0.3
    scores = [v for v in vitals.values() if isinstance(v, (int, float))]
    if not scores:
        return None

    avg = sum(scores) / len(scores)
    if avg < 0.3:
        logger.warning("🛡️ Instinct: vitals crisis — avg %.2f for %d turns", avg, consecutive_low_turns)
        return InstinctResult(
            action=InstinctAction.WARN,
            triggered_by="vitals_crisis",
            message=(
                "I'm noticing my performance has been degraded for several turns. "
                "My vitals are consistently low. A fresh session might help — "
                "or we could try a different approach to this topic."
            ),
            details={"avg_vitals": avg, "consecutive_turns": consecutive_low_turns},
        )
    return None


def evaluate_instinct(
    message: str,
    *,
    gpu_temp_c: Optional[float] = None,
    disk_usage_pct: Optional[float] = None,
    vram_usage_pct: Optional[float] = None,
    vitals: Optional[dict] = None,
    consecutive_low_turns: int = 0,
) -> InstinctResult:
    """
    Run all instinct gates on the incoming message and hardware state.

    Returns InstinctResult with action=PASS if nothing fires.
    Gates are checked in priority order — first match wins.

    This function must stay fast (<1ms). No LLM calls. No I/O.
    """
    # Gate 1: Identity protection (REJECT)
    result = _check_identity_attack(message)
    if result:
        return result

    # Gate 2: Prompt injection (REJECT)
    result = _check_injection(message)
    if result:
        return result

    # Gate 3: Hardware self-preservation (WARN)
    result = _check_hardware(gpu_temp_c, disk_usage_pct, vram_usage_pct)
    if result:
        return result

    # Gate 4: Vitals crisis (WARN)
    result = _check_vitals_crisis(vitals, consecutive_low_turns)
    if result:
        return result

    # All clear
    return InstinctResult(action=InstinctAction.PASS)
