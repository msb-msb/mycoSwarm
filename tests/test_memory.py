"""Tests for mycoswarm.memory — persistent fact store + session summaries."""

import json
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from mycoswarm import memory


@pytest.fixture(autouse=True)
def temp_memory_dir(tmp_path, monkeypatch):
    """Redirect all memory I/O to a temp directory."""
    mem_dir = tmp_path / "memory"
    mem_dir.mkdir()
    monkeypatch.setattr(memory, "MEMORY_DIR", mem_dir)
    monkeypatch.setattr(memory, "FACTS_PATH", mem_dir / "facts.json")
    monkeypatch.setattr(memory, "SESSIONS_PATH", mem_dir / "sessions.jsonl")
    return mem_dir


# ---------------------------------------------------------------------------
# Fact Store
# ---------------------------------------------------------------------------

class TestFactStore:
    def test_load_empty(self):
        assert memory.load_facts() == []

    def test_add_and_load(self):
        fact = memory.add_fact("User teaches Tai Chi")
        assert fact["id"] == 1
        assert fact["text"] == "User teaches Tai Chi"
        assert "added" in fact

        facts = memory.load_facts()
        assert len(facts) == 1
        assert facts[0]["text"] == "User teaches Tai Chi"

    def test_add_multiple(self):
        memory.add_fact("Fact one")
        memory.add_fact("Fact two")
        memory.add_fact("Fact three")

        facts = memory.load_facts()
        assert len(facts) == 3
        assert [f["id"] for f in facts] == [1, 2, 3]

    def test_remove_fact(self):
        memory.add_fact("Keep this")
        memory.add_fact("Remove this")
        memory.add_fact("Keep this too")

        assert memory.remove_fact(2) is True
        facts = memory.load_facts()
        assert len(facts) == 2
        assert [f["text"] for f in facts] == ["Keep this", "Keep this too"]

    def test_remove_nonexistent(self):
        memory.add_fact("Only fact")
        assert memory.remove_fact(999) is False
        assert len(memory.load_facts()) == 1

    def test_id_auto_increment_after_remove(self):
        memory.add_fact("First")
        memory.add_fact("Second")
        memory.remove_fact(1)
        fact = memory.add_fact("Third")
        assert fact["id"] == 3  # max existing is 2, so next is 3

    def test_save_and_load_roundtrip(self):
        facts = [
            {"id": 10, "text": "Custom fact", "added": "2026-01-01T00:00:00"},
        ]
        memory.save_facts(facts)
        loaded = memory.load_facts()
        assert loaded == facts

    def test_load_corrupted_file(self, temp_memory_dir):
        (temp_memory_dir / "facts.json").write_text("not json")
        assert memory.load_facts() == []

    def test_format_facts_empty(self):
        assert memory.format_facts_for_prompt([]) == ""

    def test_format_facts(self):
        facts = [
            {"id": 1, "text": "Likes Python", "added": "2026-01-01"},
            {"id": 2, "text": "Teaches Tai Chi", "added": "2026-01-02"},
        ]
        result = memory.format_facts_for_prompt(facts)
        assert "Known facts about the user:" in result
        assert "- Likes Python" in result
        assert "- Teaches Tai Chi" in result


# ---------------------------------------------------------------------------
# Session Summaries
# ---------------------------------------------------------------------------

class TestSessionSummaries:
    def test_load_empty(self):
        assert memory.load_session_summaries() == []

    def test_save_and_load(self):
        memory.save_session_summary("chat-1", "gemma3:27b", "Discussed Python", 10)
        memory.save_session_summary("chat-2", "gemma3:27b", "Talked about GPUs", 5)

        summaries = memory.load_session_summaries()
        assert len(summaries) == 2
        assert summaries[0]["session_name"] == "chat-1"
        assert summaries[1]["summary"] == "Talked about GPUs"

    def test_load_with_limit(self):
        for i in range(20):
            memory.save_session_summary(f"chat-{i}", "m", f"Summary {i}", 3)

        summaries = memory.load_session_summaries(limit=5)
        assert len(summaries) == 5
        assert summaries[0]["session_name"] == "chat-15"

    def test_format_summaries_empty(self):
        assert memory.format_summaries_for_prompt([]) == ""

    def test_format_summaries(self):
        summaries = [
            {"timestamp": "2026-02-09T14:30:00", "summary": "Discussed AI swarms"},
            {"timestamp": "2026-02-08T10:00:00", "summary": "Debugged mDNS"},
        ]
        result = memory.format_summaries_for_prompt(summaries)
        assert "Previous conversations:" in result
        assert "[2026-02-09] Discussed AI swarms" in result
        assert "[2026-02-08] Debugged mDNS" in result

    def test_corrupted_jsonl(self, temp_memory_dir):
        (temp_memory_dir / "sessions.jsonl").write_text(
            '{"valid": true}\nnot json\n{"also": "valid"}\n'
        )
        summaries = memory.load_session_summaries()
        assert len(summaries) == 2


# ---------------------------------------------------------------------------
# Summarize Session (mocked Ollama)
# ---------------------------------------------------------------------------

class TestSummarizeSession:
    def test_too_short(self):
        assert memory.summarize_session([{"role": "user", "content": "hi"}], "m") is None

    def test_success(self):
        messages = [
            {"role": "user", "content": "What is Python?"},
            {"role": "assistant", "content": "Python is a programming language."},
        ]
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "message": {"content": "User asked about Python basics."}
        }
        mock_resp.raise_for_status = MagicMock()

        with patch("mycoswarm.memory.httpx.Client") as mock_client:
            mock_client.return_value.__enter__ = MagicMock(return_value=mock_client.return_value)
            mock_client.return_value.__exit__ = MagicMock(return_value=False)
            mock_client.return_value.post.return_value = mock_resp

            result = memory.summarize_session(messages, "gemma3:27b")
            assert result == "User asked about Python basics."

    def test_failure_returns_none(self):
        messages = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ]
        with patch("mycoswarm.memory.httpx.Client") as mock_client:
            mock_client.return_value.__enter__ = MagicMock(return_value=mock_client.return_value)
            mock_client.return_value.__exit__ = MagicMock(return_value=False)
            mock_client.return_value.post.side_effect = Exception("connection refused")

            result = memory.summarize_session(messages, "gemma3:27b")
            assert result is None

    def test_truncation(self):
        """Long messages should be truncated before sending."""
        messages = [
            {"role": "user", "content": "x" * 1000},
            {"role": "assistant", "content": "y" * 1000},
        ]
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"message": {"content": "Summary."}}
        mock_resp.raise_for_status = MagicMock()

        with patch("mycoswarm.memory.httpx.Client") as mock_client:
            mock_client.return_value.__enter__ = MagicMock(return_value=mock_client.return_value)
            mock_client.return_value.__exit__ = MagicMock(return_value=False)
            mock_client.return_value.post.return_value = mock_resp

            memory.summarize_session(messages, "m")

            # Check the transcript sent to Ollama was truncated
            call_args = mock_client.return_value.post.call_args
            sent_msgs = call_args[1]["json"]["messages"]
            user_content = sent_msgs[1]["content"]
            # Each message should be truncated to 500 chars + "..."
            assert "x" * 500 in user_content
            assert len(user_content) < 2000  # not full 2000 chars


# ---------------------------------------------------------------------------
# Prompt Builder
# ---------------------------------------------------------------------------

class TestPromptBuilder:
    def test_empty_returns_none(self):
        assert memory.build_memory_system_prompt() is None

    def test_facts_only(self):
        memory.add_fact("Likes coffee")
        prompt = memory.build_memory_system_prompt()
        assert prompt is not None
        assert "persistent memory across conversations" in prompt
        assert "Known facts about the user:" in prompt
        assert "Likes coffee" in prompt
        assert "Previous conversations:" not in prompt

    def test_summaries_only(self):
        memory.save_session_summary("s1", "m", "Talked about coffee", 5)
        prompt = memory.build_memory_system_prompt()
        assert prompt is not None
        assert "persistent memory across conversations" in prompt
        assert "Previous conversations:" in prompt
        assert "Talked about coffee" in prompt
        assert "Known facts" not in prompt

    def test_both(self):
        memory.add_fact("Likes coffee")
        memory.save_session_summary("s1", "m", "Discussed brewing methods", 5)
        prompt = memory.build_memory_system_prompt()
        assert prompt is not None
        assert "persistent memory across conversations" in prompt
        assert "Known facts about the user:" in prompt
        assert "Previous conversations:" in prompt
