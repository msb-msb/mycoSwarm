"""
Timing Gate — Wu Wei applied to response calibration.

Evaluates contextual signals and decides HOW Monica should respond:
- PROCEED: normal response depth and tone
- GENTLE: shorter, warmer, less information-dense
- DEEP: expansive, thorough, exploratory

This is not about WHETHER to respond (the user asked, so answer).
It's about matching response energy to the moment.
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum


class TimingMode(str, Enum):
    PROCEED = "proceed"   # Normal — full depth, standard tone
    GENTLE = "gentle"     # Soften — shorter, warmer, less dense
    DEEP = "deep"         # Expand — thorough, exploratory, take your time


@dataclass
class TimingDecision:
    mode: TimingMode
    reasons: list[str]     # Why this mode was chosen
    prompt_modifier: str   # Injected into system prompt

    def status_indicator(self) -> str:
        """Compact indicator for the timing line."""
        icons = {
            TimingMode.PROCEED: "▶",
            TimingMode.GENTLE: "🌙",
            TimingMode.DEEP: "🌊",
        }
        return icons.get(self.mode, "▶")


def evaluate_timing(
    current_time: datetime | None = None,
    last_interaction: datetime | None = None,
    session_turn_count: int = 0,
    seconds_since_last_turn: float | None = None,
    user_message_length: int = 0,
    intent: dict | None = None,
    vitals_compassion: float | None = None,
    vitals_clarity: float | None = None,
    frustration_detected: bool = False,
    body_state: dict | None = None,
) -> TimingDecision:
    """
    Evaluate timing signals and return a calibration decision.

    All inputs come from data already available in the chat loop.
    No LLM calls, no network, no file I/O. Pure heuristics.
    """
    now = current_time or datetime.now()
    intent = intent or {}
    mode = intent.get("mode", "chat")
    reasons = []

    # --- Scoring: accumulate GENTLE and DEEP signals ---
    gentle_score = 0.0
    deep_score = 0.0

    # TIME OF DAY
    hour = now.hour
    if hour >= 23 or hour < 6:
        gentle_score += 0.4
        reasons.append("late night — softer responses")
    elif 6 <= hour < 9:
        # Early morning — could go either way
        if mode == "explore":
            deep_score += 0.2
            reasons.append("early morning exploration")
        else:
            gentle_score += 0.1
            reasons.append("early morning — ease in")
    elif 9 <= hour < 12:
        # Peak hours — normal or deep
        if mode == "explore":
            deep_score += 0.3
            reasons.append("morning peak + exploration mode")

    # INTERACTION RECENCY
    if last_interaction:
        gap = now - last_interaction
        if gap > timedelta(hours=24):
            gentle_score += 0.2
            reasons.append(f"reconnecting after {gap.days}d absence")
        elif gap < timedelta(minutes=5):
            # Same session, flowing — could be rapid-fire or flow
            pass

    # RAPID-FIRE DETECTION
    if seconds_since_last_turn is not None and seconds_since_last_turn < 15:
        gentle_score += 0.3
        reasons.append("rapid-fire messages — stay concise")

    # SESSION LENGTH / FATIGUE
    if session_turn_count > 20:
        gentle_score += 0.3
        reasons.append(f"long session ({session_turn_count} turns) — fatigue awareness")
    elif session_turn_count > 10:
        gentle_score += 0.1

    # MESSAGE LENGTH (short messages suggest desire for concise answers)
    if user_message_length < 20:
        gentle_score += 0.15
        reasons.append("short message — match energy")
    elif user_message_length > 200:
        deep_score += 0.2
        reasons.append("detailed message — match depth")

    # INTENT MODE
    if mode == "explore":
        deep_score += 0.3
        reasons.append("explore mode — go deep")
    elif mode == "execute":
        # Execute = get it done, don't philosophize
        gentle_score += 0.1
    elif mode == "chat":
        gentle_score += 0.1

    # FRUSTRATION
    if frustration_detected:
        gentle_score += 0.4
        reasons.append("frustration detected — slow down, scaffold")

    # FIRST MESSAGE OF SESSION
    if session_turn_count == 0:
        reasons.append("first message — warm greeting energy")
        # Don't push either direction, just note it

    # HARDWARE BODY SIGNALS (Phase 31c)
    if body_state:
        gpu_temp = body_state.get("gpu_temp")
        if gpu_temp is not None and gpu_temp > 85:
            gentle_score += 0.3
            reasons.append(f"GPU running hot ({gpu_temp:.0f}°C) — keep it brief")

        vram_pct = body_state.get("vram_percent")
        if vram_pct is not None and vram_pct > 90:
            gentle_score += 0.2
            reasons.append(f"VRAM pressure ({vram_pct:.0f}%) — concise responses")

    # --- Decision ---
    if gentle_score >= 0.5 and gentle_score > deep_score:
        mode_decision = TimingMode.GENTLE
    elif deep_score >= 0.4 and deep_score > gentle_score:
        mode_decision = TimingMode.DEEP
    else:
        mode_decision = TimingMode.PROCEED

    # --- Build prompt modifier ---
    prompt_modifier = _build_prompt_modifier(mode_decision, reasons)

    return TimingDecision(
        mode=mode_decision,
        reasons=reasons,
        prompt_modifier=prompt_modifier,
    )


def _build_prompt_modifier(mode: TimingMode, reasons: list[str]) -> str:
    """Build a natural-language prompt modifier for the system prompt."""
    if mode == TimingMode.GENTLE:
        return (
            "[Timing: GENTLE] Keep your response concise and warm. "
            "The user may be tired, frustrated, or moving quickly. "
            "Shorter paragraphs, fewer details, more warmth. "
            "If the user seems overwhelmed, scaffold — break things into steps."
        )
    elif mode == TimingMode.DEEP:
        return (
            "[Timing: DEEP] The user is in exploration mode. "
            "Take your time. Provide depth, examples, connections. "
            "This is a moment for thorough, expansive thinking."
        )
    else:
        return ""  # PROCEED — no modifier needed
