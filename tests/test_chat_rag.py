"""Tests for RAG context handling in multi-turn chat.

Verifies that RAG/web context injected per-turn does NOT accumulate
in the persistent message history across turns.
"""


def _simulate_turn(messages: list[dict], user_input: str, tool_context: str, response: str):
    """Simulate one turn of the chat loop's message-building logic.

    Mirrors the logic in cmd_chat (cli.py): tool_context is injected only
    into the send copy, not the persistent messages list.
    """
    # User message goes into persistent history
    messages.append({"role": "user", "content": user_input})

    # Build send copy
    send_msgs = list(messages)

    # Inject tool context into send copy only (not persistent history)
    if tool_context:
        send_msgs.insert(-1, {"role": "system", "content": tool_context})

    # Simulate assistant response stored in persistent history
    messages.append({"role": "assistant", "content": response})

    return send_msgs


def test_rag_context_not_accumulated_across_turns():
    """RAG context from turn 1 must NOT appear in the messages sent for turn 2."""
    messages: list[dict] = [
        {"role": "system", "content": "You are a helpful assistant."},
    ]

    # --- Turn 1: RAG query ---
    rag_context_1 = (
        "DOCUMENT EXCERPTS:\n"
        "[D1] (notes.txt > chapter1) The speed of light is 299,792 km/s."
    )
    send_1 = _simulate_turn(messages, "What is the speed of light?", rag_context_1, "299,792 km/s")

    # send_1 should contain the RAG context
    system_msgs_1 = [m for m in send_1 if m["role"] == "system"]
    assert any("speed of light" in m["content"] for m in system_msgs_1)

    # Persistent messages should NOT contain any RAG context system messages
    assert not any(
        m["role"] == "system" and "DOCUMENT EXCERPTS" in m["content"]
        for m in messages
    ), "RAG context leaked into persistent message history after turn 1"

    # --- Turn 2: different RAG query ---
    rag_context_2 = (
        "DOCUMENT EXCERPTS:\n"
        "[D1] (history.txt > intro) The French Revolution began in 1789."
    )
    send_2 = _simulate_turn(messages, "When did the French Revolution start?", rag_context_2, "1789")

    # send_2 should contain turn 2's RAG context
    system_msgs_2 = [m for m in send_2 if m["role"] == "system"]
    assert any("French Revolution" in m["content"] for m in system_msgs_2)

    # send_2 must NOT contain turn 1's RAG context
    assert not any(
        "speed of light" in m["content"]
        for m in send_2
        if m["role"] == "system"
    ), "Turn 1 RAG context leaked into turn 2 send messages"

    # Persistent messages should still have NO RAG context
    assert not any(
        m["role"] == "system" and "DOCUMENT EXCERPTS" in m["content"]
        for m in messages
    ), "RAG context leaked into persistent message history after turn 2"

    # --- Turn 3: no RAG (plain question) ---
    send_3 = _simulate_turn(messages, "Hello!", "", "Hi there!")

    # send_3 must NOT contain any RAG context from previous turns
    assert not any(
        "DOCUMENT EXCERPTS" in m["content"]
        for m in send_3
        if m["role"] == "system"
    ), "Old RAG context appeared in a non-RAG turn"


def test_conversation_history_persists_across_turns():
    """User messages and assistant responses must persist across turns."""
    messages: list[dict] = [
        {"role": "system", "content": "You are a helpful assistant."},
    ]

    _simulate_turn(messages, "Question 1", "DOCUMENT EXCERPTS:\n[D1] chunk1", "Answer 1")
    _simulate_turn(messages, "Question 2", "DOCUMENT EXCERPTS:\n[D1] chunk2", "Answer 2")
    _simulate_turn(messages, "Question 3", "", "Answer 3")

    # All user messages should be in persistent history
    user_msgs = [m["content"] for m in messages if m["role"] == "user"]
    assert user_msgs == ["Question 1", "Question 2", "Question 3"]

    # All assistant messages should be in persistent history
    assistant_msgs = [m["content"] for m in messages if m["role"] == "assistant"]
    assert assistant_msgs == ["Answer 1", "Answer 2", "Answer 3"]

    # Only the initial system message should be in persistent history
    system_msgs = [m for m in messages if m["role"] == "system"]
    assert len(system_msgs) == 1
    assert system_msgs[0]["content"] == "You are a helpful assistant."


def test_send_messages_include_full_history_plus_current_rag():
    """The send copy for each turn should include all history + current RAG only."""
    messages: list[dict] = [
        {"role": "system", "content": "System prompt."},
    ]

    # Turn 1
    send_1 = _simulate_turn(messages, "Q1", "DOCUMENT EXCERPTS:\nrag1", "A1")
    # send_1: system, rag_context, user
    assert len(send_1) == 3
    assert send_1[0]["role"] == "system"
    assert send_1[1]["role"] == "system"  # RAG context
    assert "rag1" in send_1[1]["content"]
    assert send_1[2]["role"] == "user"

    # Turn 2
    send_2 = _simulate_turn(messages, "Q2", "DOCUMENT EXCERPTS:\nrag2", "A2")
    # send_2: system, user(Q1), assistant(A1), rag_context, user(Q2)
    assert len(send_2) == 5
    assert send_2[0]["role"] == "system"
    assert send_2[1] == {"role": "user", "content": "Q1"}
    assert send_2[2] == {"role": "assistant", "content": "A1"}
    assert send_2[3]["role"] == "system"  # RAG context for turn 2
    assert "rag2" in send_2[3]["content"]
    assert "rag1" not in send_2[3]["content"]
    assert send_2[4] == {"role": "user", "content": "Q2"}
