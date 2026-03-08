"""Tests for the Timing Gate (Wu Wei Gate) — Phase 20b."""

from datetime import datetime, timedelta

from mycoswarm.timing import TimingMode, TimingDecision, evaluate_timing


# 1. Late night → GENTLE
def test_late_night_gentle():
    result = evaluate_timing(
        current_time=datetime(2026, 2, 17, 23, 30),
        user_message_length=15,
    )
    assert result.mode == TimingMode.GENTLE
    assert any("late night" in r for r in result.reasons)


# 2. Early morning + explore → DEEP
def test_early_morning_explore_deep():
    result = evaluate_timing(
        current_time=datetime(2026, 2, 17, 7, 0),
        intent={"tool": "answer", "mode": "explore", "scope": "all"},
        user_message_length=250,
    )
    assert result.mode == TimingMode.DEEP
    assert any("early morning exploration" in r for r in result.reasons)


# 3. Peak hours + chat → PROCEED
def test_peak_hours_proceed():
    result = evaluate_timing(
        current_time=datetime(2026, 2, 17, 10, 0),
        intent={"tool": "answer", "mode": "chat", "scope": "all"},
        user_message_length=50,
    )
    assert result.mode == TimingMode.PROCEED


# 4. Rapid-fire → GENTLE
def test_rapid_fire_gentle():
    result = evaluate_timing(
        current_time=datetime(2026, 2, 17, 14, 0),
        seconds_since_last_turn=10,
        user_message_length=15,
    )
    assert result.mode == TimingMode.GENTLE
    assert any("rapid-fire" in r for r in result.reasons)


# 5. Long session → GENTLE
def test_long_session_gentle():
    result = evaluate_timing(
        current_time=datetime(2026, 2, 17, 14, 0),
        session_turn_count=25,
        user_message_length=15,
    )
    assert result.mode == TimingMode.GENTLE
    assert any("fatigue" in r for r in result.reasons)


# 6. Short message contributes to GENTLE
def test_short_message_gentle():
    result = evaluate_timing(
        current_time=datetime(2026, 2, 17, 14, 0),
        user_message_length=10,
    )
    # Short message alone (0.15) + chat mode (0.1) = 0.25, not enough for GENTLE
    # But the signal is recorded
    assert any("short message" in r for r in result.reasons)


# 7. Long message contributes to DEEP
def test_long_message_deep():
    result = evaluate_timing(
        current_time=datetime(2026, 2, 17, 10, 0),
        intent={"tool": "answer", "mode": "explore", "scope": "all"},
        user_message_length=300,
    )
    assert result.mode == TimingMode.DEEP
    assert any("detailed message" in r for r in result.reasons)


# 8. Frustration → GENTLE
def test_frustration_gentle():
    result = evaluate_timing(
        current_time=datetime(2026, 2, 17, 14, 0),
        frustration_detected=True,
        user_message_length=15,
    )
    assert result.mode == TimingMode.GENTLE
    assert any("frustration" in r for r in result.reasons)


# 9. Explore mode → DEEP signal
def test_explore_mode_deep():
    result = evaluate_timing(
        current_time=datetime(2026, 2, 17, 10, 0),
        intent={"tool": "answer", "mode": "explore", "scope": "all"},
        user_message_length=250,
    )
    assert result.mode == TimingMode.DEEP
    assert any("explore mode" in r for r in result.reasons)


# 10. First message noted
def test_first_message_noted():
    result = evaluate_timing(
        current_time=datetime(2026, 2, 17, 14, 0),
        session_turn_count=0,
        user_message_length=50,
    )
    assert any("first message" in r for r in result.reasons)


# 11. Reconnection after absence
def test_reconnection_after_absence():
    result = evaluate_timing(
        current_time=datetime(2026, 2, 17, 14, 0),
        last_interaction=datetime(2026, 2, 15, 14, 0),
        user_message_length=50,
    )
    assert any("reconnecting" in r for r in result.reasons)


# 12. PROCEED → empty prompt_modifier
def test_proceed_no_prompt_modifier():
    result = evaluate_timing(
        current_time=datetime(2026, 2, 17, 10, 0),
        intent={"tool": "answer", "mode": "chat", "scope": "all"},
        user_message_length=50,
    )
    assert result.mode == TimingMode.PROCEED
    assert result.prompt_modifier == ""


# 13. GENTLE → prompt modifier contains "concise and warm"
def test_gentle_prompt_modifier():
    result = evaluate_timing(
        current_time=datetime(2026, 2, 17, 23, 30),
        user_message_length=15,
    )
    assert result.mode == TimingMode.GENTLE
    assert "concise and warm" in result.prompt_modifier


# 14. DEEP → prompt modifier contains "exploration mode"
def test_deep_prompt_modifier():
    result = evaluate_timing(
        current_time=datetime(2026, 2, 17, 10, 0),
        intent={"tool": "answer", "mode": "explore", "scope": "all"},
        user_message_length=300,
    )
    assert result.mode == TimingMode.DEEP
    assert "exploration mode" in result.prompt_modifier


# 15. Status indicator icons
def test_status_indicator_icons():
    proceed = TimingDecision(mode=TimingMode.PROCEED, reasons=[], prompt_modifier="")
    gentle = TimingDecision(mode=TimingMode.GENTLE, reasons=[], prompt_modifier="")
    deep = TimingDecision(mode=TimingMode.DEEP, reasons=[], prompt_modifier="")

    assert proceed.status_indicator() == "▶"
    assert gentle.status_indicator() == "🌙"
    assert deep.status_indicator() == "🌊"


# ── Hardware body signals (Phase 31c) ────────────────────────────────────


# 16. GPU temp > 85°C biases GENTLE
def test_hot_gpu_biases_gentle():
    result = evaluate_timing(
        current_time=datetime(2026, 2, 17, 10, 0),
        user_message_length=50,
        session_turn_count=5,
        body_state={"gpu_temp": 90.0, "vram_percent": 40.0, "nodes": []},
    )
    assert any("GPU running hot" in r for r in result.reasons)
    # With gentle_score: chat(0.1) + hot_gpu(0.3) + mid_session = 0.4-0.5
    # May or may not tip to GENTLE alone, but the signal is present


# 17. GPU temp exactly 85 does NOT trigger
def test_gpu_temp_85_no_trigger():
    result = evaluate_timing(
        current_time=datetime(2026, 2, 17, 10, 0),
        user_message_length=50,
        body_state={"gpu_temp": 85.0, "vram_percent": 40.0, "nodes": []},
    )
    assert not any("GPU running hot" in r for r in result.reasons)


# 18. VRAM > 90% biases GENTLE
def test_high_vram_biases_gentle():
    result = evaluate_timing(
        current_time=datetime(2026, 2, 17, 10, 0),
        user_message_length=50,
        body_state={"gpu_temp": 50.0, "vram_percent": 95.0, "nodes": []},
    )
    assert any("VRAM pressure" in r for r in result.reasons)


# 19. VRAM exactly 90 does NOT trigger
def test_vram_90_no_trigger():
    result = evaluate_timing(
        current_time=datetime(2026, 2, 17, 10, 0),
        user_message_length=50,
        body_state={"gpu_temp": 50.0, "vram_percent": 90.0, "nodes": []},
    )
    assert not any("VRAM pressure" in r for r in result.reasons)


# 20. Both hot GPU + high VRAM → definitely GENTLE
def test_hot_gpu_and_high_vram_gentle():
    result = evaluate_timing(
        current_time=datetime(2026, 2, 17, 10, 0),
        user_message_length=50,
        body_state={"gpu_temp": 92.0, "vram_percent": 95.0, "nodes": []},
    )
    assert result.mode == TimingMode.GENTLE
    assert any("GPU running hot" in r for r in result.reasons)
    assert any("VRAM pressure" in r for r in result.reasons)


# 21. None body_state is no-op
def test_none_body_state_no_change():
    r1 = evaluate_timing(
        current_time=datetime(2026, 2, 17, 10, 0),
        user_message_length=50,
    )
    r2 = evaluate_timing(
        current_time=datetime(2026, 2, 17, 10, 0),
        user_message_length=50,
        body_state=None,
    )
    assert r1.mode == r2.mode
    assert len(r1.reasons) == len(r2.reasons)


# 22. Normal body state doesn't add signals
def test_normal_body_no_signals():
    result = evaluate_timing(
        current_time=datetime(2026, 2, 17, 10, 0),
        user_message_length=50,
        body_state={"gpu_temp": 55.0, "vram_percent": 40.0, "nodes": []},
    )
    assert not any("GPU" in r for r in result.reasons)
    assert not any("VRAM" in r for r in result.reasons)
