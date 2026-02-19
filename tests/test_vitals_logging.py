"""Tests for vitals per-turn logging (Phase 31d)."""

import json


def test_vitals_attached_to_assistant_message():
    """Vitals should be stored alongside assistant responses."""
    msg = {"role": "assistant", "content": "Hello!"}
    vitals = {
        "calm": 0.8, "clarity": 0.9, "curiosity": 0.7, "compassion": 0.9,
        "courage": 0.6, "creativity": 0.5, "connectedness": 0.7, "confidence": 0.8,
    }
    msg["vitals"] = vitals
    assert msg["vitals"]["calm"] == 0.8
    assert "content" in msg  # content still there


def test_message_without_vitals_still_works():
    """Old-format messages (no vitals) should not break anything."""
    msg = {"role": "assistant", "content": "Hello!"}
    assert "vitals" not in msg  # no crash
    assert msg.get("vitals") is None


def test_session_json_roundtrip():
    """Messages with vitals should survive JSON serialization."""
    msg = {
        "role": "assistant",
        "content": "Test",
        "vitals": {"calm": 0.8, "clarity": 0.9},
    }
    serialized = json.dumps(msg)
    deserialized = json.loads(serialized)
    assert deserialized["vitals"]["calm"] == 0.8


def test_instinct_attached_to_user_message():
    """Instinct results should be stored on user messages when triggered."""
    msg = {"role": "user", "content": "test"}
    msg["instinct"] = {"action": "warn", "triggered_by": "self_preservation_gpu"}
    assert msg["instinct"]["action"] == "warn"
    assert "content" in msg


def test_user_message_without_instinct():
    """Normal user messages should not have instinct key."""
    msg = {"role": "user", "content": "hello"}
    assert "instinct" not in msg
    assert msg.get("instinct") is None


def test_vitals_history_extraction():
    """Simulates /history command logic — extract vitals from message list."""
    messages = [
        {"role": "system", "content": "system prompt"},
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi!", "vitals": {"calm": 0.8, "clarity": 0.9}},
        {"role": "user", "content": "how are you?"},
        {"role": "assistant", "content": "good!", "vitals": {"calm": 0.7, "clarity": 0.85}},
        {"role": "user", "content": "bye"},
        {"role": "assistant", "content": "goodbye!"},  # no vitals (old format)
    ]
    turns_with_vitals = [
        m["vitals"] for m in messages
        if m.get("role") == "assistant" and "vitals" in m
    ]
    assert len(turns_with_vitals) == 2
    assert turns_with_vitals[0]["calm"] == 0.8
    assert turns_with_vitals[1]["calm"] == 0.7


def test_vitals_dict_keys_match_expected():
    """Vitals dict from to_dict() should have all 8 C's."""
    from mycoswarm.vitals import Vitals
    v = Vitals(
        calm=0.8, clarity=0.9, curiosity=0.7, compassion=0.6,
        courage=0.5, creativity=0.4, connectedness=0.3, confidence=0.2,
    )
    d = v.to_dict()
    expected_keys = {"calm", "clarity", "curiosity", "compassion",
                     "courage", "creativity", "connectedness", "confidence"}
    assert set(d.keys()) == expected_keys
    assert d["calm"] == 0.8
    assert d["confidence"] == 0.2
