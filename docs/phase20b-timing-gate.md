# Phase 20b: Timing Gate (Wu Wei Gate) — CC Implementation Guide

## Part 1: Add to PLAN.md

Add under Phase 20 section:

```markdown
### Phase 20b: Timing Gate (Wu Wei Gate)
Reference: docs/ARCHITECTURE-INTENT.md — Gap #2: Timing
Influences: Wu Wei (action through non-action), Tai Chi (yield before push)

The thesis: agents fire on every input with equal urgency. A healthy mind
knows when to act, when to wait, and when to yield. The Timing Gate runs
before the main response and outputs a TimingDecision that shapes HOW
Monica responds — not just WHAT she says.

This is not about suppressing responses in chat (the user asked, so answer).
It's about calibrating depth, tone, and energy based on context signals.

#### 20b-1: TimingGate Core
- [ ] `timing.py`: TimingGate class with evaluate() method
- [ ] Six input signals: time_of_day, interaction_recency, emotional_trajectory,
      action_urgency, reversibility, session_length
- [ ] Three outputs: PROCEED (normal), GENTLE (soften/shorten), DEEP (expand/explore)
- [ ] No LLM call — pure heuristic computation, <1ms
- [ ] Timing modifier injected into system prompt as behavioral guidance

#### 20b-2: Response Calibration
- [ ] Late night (after 11pm) + short messages → GENTLE: shorter, warmer responses
- [ ] Early morning + exploratory intent → DEEP: thorough, expansive responses
- [ ] Rapid-fire messages (<30s between turns) → GENTLE: concise, don't overwhelm
- [ ] Long session (>20 turns) → GENTLE: fatigue awareness, suggest break
- [ ] Frustration detected (from vitals) → GENTLE: scaffold, slow down
- [ ] First message of day → warm greeting energy
- [ ] After long absence (>24h) → reconnection tone

#### 20b-3: Agentic Action Gate (future)
- [ ] For proactive actions (not chat): SUPPRESS / DEFER / PROCEED
- [ ] Applies to: automated suggestions, scheduled tasks, procedure extraction
- [ ] Irreversible actions require PROCEED + confirmation
- [ ] Low-urgency actions auto-DEFER to next natural interaction
```

---

## Part 2: CC Prompt — Implement Phase 20b-1 and 20b-2

```
Implement Phase 20b: Timing Gate (Wu Wei Gate).

This gate evaluates contextual signals and outputs a timing decision
that shapes Monica's response style. No LLM calls — pure heuristics.
The gate adds <1ms of latency.

### 1. Create src/mycoswarm/timing.py

```python
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
```

### 2. Integrate into chat loop

In cli.py, in the chat loop AFTER intent classification but BEFORE 
the main LLM call:

```python
from mycoswarm.timing import evaluate_timing

# Track timing between turns
last_turn_time = None  # set after each response

# Before each LLM call:
seconds_since_last = None
if last_turn_time:
    seconds_since_last = (datetime.now() - last_turn_time).total_seconds()

timing = evaluate_timing(
    current_time=datetime.now(),
    last_interaction=session_start_time,  # or last session end time
    session_turn_count=turn_count,
    seconds_since_last_turn=seconds_since_last,
    user_message_length=len(user_input),
    intent=intent_result,
    frustration_detected=(vitals.clarity < 0.4 if hasattr(vitals, 'clarity') else False),
)

# Inject timing modifier into system prompt if not PROCEED
if timing.prompt_modifier:
    system_prompt = identity_prompt + "\n\n" + timing.prompt_modifier + "\n\n" + rest_of_prompt
```

### 3. Display timing indicator

Add the timing mode to the existing status line, between the 
timing bar and the vitals:

```
  ⏱  5.8s | 32.8 tok/s | gemma3:27b | node: Miu
  🌙 gentle: late night — softer responses
  🧭 Ca:0.9 Cl:0.7 Cu:0.5 Cp:0.7 Co:0.6 Cr:0.5 Cn:0.5 Cf:0.6
```

Only show the timing line when mode is NOT PROCEED (proceed is 
the default, no need to announce it):

```python
if timing.mode != TimingMode.PROCEED:
    indicator = timing.status_indicator()
    reason_text = "; ".join(timing.reasons[:2])  # max 2 reasons shown
    print(f"  {indicator} {timing.mode.value}: {reason_text}")
```

### 4. /timing slash command

Show current timing state and all active signals:

```
/timing output:

🕐 Timing Gate
─────────────────────────
  Mode:     GENTLE 🌙
  Time:     11:45 PM (late night)
  Session:  turn 23 of current session
  Gap:      12s since last message
  Reasons:
    • late night — softer responses
    • long session (23 turns) — fatigue awareness
    • rapid-fire messages — stay concise
─────────────────────────
```

### 5. Tests

Add tests/test_timing.py:

1. test_late_night_gentle — hour=23 → GENTLE with "late night" reason
2. test_early_morning_explore_deep — hour=8 + explore mode → DEEP
3. test_peak_hours_proceed — hour=10 + chat mode → PROCEED
4. test_rapid_fire_gentle — seconds_since_last=10 → GENTLE
5. test_long_session_gentle — turn_count=25 → GENTLE with fatigue reason
6. test_short_message_gentle — message_length=10 → contributes to GENTLE
7. test_long_message_deep — message_length=300 → contributes to DEEP
8. test_frustration_gentle — frustration_detected=True → GENTLE
9. test_explore_mode_deep — mode=explore → DEEP signal
10. test_first_message_noted — turn_count=0 → "first message" in reasons
11. test_reconnection_after_absence — gap > 24h → noted in reasons
12. test_proceed_no_prompt_modifier — PROCEED → empty prompt_modifier
13. test_gentle_prompt_modifier — GENTLE → contains "concise and warm"
14. test_deep_prompt_modifier — DEEP → contains "exploration mode"
15. test_status_indicator_icons — correct emoji per mode

### 6. Update PLAN.md

Mark 20b-1 and 20b-2 items as done with today's date.

### 7. Commit

git add -A
git commit -m "Phase 20b: Timing Gate (Wu Wei) — response calibration from contextual signals, GENTLE/DEEP/PROCEED modes, /timing command"
git push

Do NOT bump version — bundle with next release.
```
