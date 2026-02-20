# Vitals Per-Turn Logging

## What This Is

Currently `_last_vitals` is computed each turn and displayed in the status bar,
but it's not saved anywhere. When the session ends, all vitals history is lost.

This task adds vitals to the session message history so we can track Monica's
8 C's longitudinally across turns and sessions.

## The Change

When appending the assistant's response to the messages list in `cmd_chat()`,
include the vitals dict:

### Before (current):
```python
messages.append({"role": "assistant", "content": full_response})
```

### After:
```python
msg = {"role": "assistant", "content": full_response}
if _last_vitals is not None:
    msg["vitals"] = _last_vitals
messages.append(msg)
```

That's it for storage. The vitals ride along in the session JSON when
`/quit` or `/save` serializes messages to disk.

## Also log on user messages (optional but useful)

For user messages, log the instinct result if the instinct layer is integrated:

```python
user_msg = {"role": "user", "content": user_input}
if instinct_result and instinct_result.action != InstinctAction.PASS:
    user_msg["instinct"] = {
        "action": instinct_result.action.value,
        "triggered_by": instinct_result.triggered_by,
    }
messages.append(user_msg)
```

If instinct layer isn't merged yet, skip this part — just do the assistant vitals.

## Add /history command

Show vitals trend for the current session:

```python
elif user_input.strip().lower() == "/history":
    print("\n📊 Vitals History (this session)")
    print(f"{'Turn':<6} {'Ca':<6} {'Cl':<6} {'Cu':<6} {'Co':<6} {'Cr':<6} {'Cf':<6} {'Cn':<6} {'Cp':<6}")
    print("─" * 54)
    turn = 0
    for msg in messages:
        if msg["role"] == "assistant" and "vitals" in msg:
            turn += 1
            v = msg["vitals"]
            print(f"{turn:<6} {v.get('Ca', '—'):<6} {v.get('Cl', '—'):<6} "
                  f"{v.get('Cu', '—'):<6} {v.get('Co', '—'):<6} "
                  f"{v.get('Cr', '—'):<6} {v.get('Cf', '—'):<6} "
                  f"{v.get('Cn', '—'):<6} {v.get('Cp', '—'):<6}")
    if turn == 0:
        print("  No vitals recorded yet.")
    print()
    continue
```

Note: check what the actual vitals dict keys are in `compute_vitals()` — they
might be full names like `"Calm"` instead of `"Ca"`. Adjust the key lookups
to match whatever `vitals.py` returns. The status bar already abbreviates them
so look at how it does it there.

## Backward Compatibility

Old sessions won't have `vitals` in their messages. That's fine — the
`/history` command just checks `if "vitals" in msg` and skips messages
without it. No migration needed.

When loading a resumed session, old messages work as before. New messages
get vitals appended going forward.

## Tests

Add to `tests/test_cli.py` or create `tests/test_vitals_logging.py`:

```python
def test_vitals_attached_to_assistant_message():
    """Vitals should be stored alongside assistant responses."""
    msg = {"role": "assistant", "content": "Hello!"}
    vitals = {"Ca": 0.8, "Cl": 0.9, "Cu": 0.7, "Co": 0.6, "Cr": 0.5, "Cf": 0.8, "Cn": 0.7, "Cp": 0.9}
    msg["vitals"] = vitals
    assert msg["vitals"]["Ca"] == 0.8
    assert "content" in msg  # content still there

def test_message_without_vitals_still_works():
    """Old-format messages (no vitals) should not break anything."""
    msg = {"role": "assistant", "content": "Hello!"}
    assert "vitals" not in msg  # no crash
    assert msg.get("vitals") is None

def test_session_json_roundtrip():
    """Messages with vitals should survive JSON serialization."""
    import json
    msg = {
        "role": "assistant",
        "content": "Test",
        "vitals": {"Ca": 0.8, "Cl": 0.9}
    }
    serialized = json.dumps(msg)
    deserialized = json.loads(serialized)
    assert deserialized["vitals"]["Ca"] == 0.8
```

## PLAN.md Update

Under Phase 31d, mark done:
- [x] Vitals logged per-turn in session for longitudinal tracking (date)
- [x] `/history` slash command: vitals trend for current session (date)
