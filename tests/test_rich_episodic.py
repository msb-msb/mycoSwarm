"""Tests for Phase 29a: Rich Episodic Memory."""

import json
import pytest
from datetime import datetime
from pathlib import Path
from unittest.mock import patch, MagicMock

from mycoswarm.memory import (
    VALID_TONES,
    summarize_session_rich,
    save_session_summary,
    format_summaries_for_prompt,
    _parse_rich_summary,
    _rich_fallback,
    _sanitize_str_list,
)


# ---------------------------------------------------------------------------
# _sanitize_str_list
# ---------------------------------------------------------------------------

def test_sanitize_str_list_normal():
    assert _sanitize_str_list(["a", "b", "c"]) == ["a", "b", "c"]


def test_sanitize_str_list_strips_empty():
    assert _sanitize_str_list(["a", "", "  ", "b"]) == ["a", "b"]


def test_sanitize_str_list_coerces_numbers():
    assert _sanitize_str_list([1, 2.5, "three"]) == ["1", "2.5", "three"]


def test_sanitize_str_list_not_a_list():
    assert _sanitize_str_list("not a list") == []
    assert _sanitize_str_list(None) == []
    assert _sanitize_str_list(42) == []


def test_sanitize_str_list_nested_dicts_skipped():
    assert _sanitize_str_list(["good", {"bad": "dict"}, "also good"]) == ["good", "also good"]


# ---------------------------------------------------------------------------
# _parse_rich_summary
# ---------------------------------------------------------------------------

def test_parse_rich_summary_valid_json():
    raw = json.dumps({
        "summary": "Debugged RAG pipeline",
        "decisions": ["Use user-message injection instead of system message"],
        "lessons": ["When local models ignore context, check injection method"],
        "surprises": ["gemma3:27b completely ignores separate system messages"],
        "emotional_tone": "discovery",
    })
    result = _parse_rich_summary(raw, [], "test")
    assert result["summary"] == "Debugged RAG pipeline"
    assert len(result["decisions"]) == 1
    assert len(result["lessons"]) == 1
    assert len(result["surprises"]) == 1
    assert result["emotional_tone"] == "discovery"


def test_parse_rich_summary_markdown_fences():
    raw = '```json\n{"summary": "test", "decisions": [], "lessons": [], "surprises": [], "emotional_tone": "neutral"}\n```'
    result = _parse_rich_summary(raw, [], "test")
    assert result["summary"] == "test"


def test_parse_rich_summary_invalid_tone_defaults():
    raw = json.dumps({
        "summary": "test",
        "decisions": [],
        "lessons": [],
        "surprises": [],
        "emotional_tone": "EXTREMELY_ANGRY",
    })
    result = _parse_rich_summary(raw, [], "test")
    assert result["emotional_tone"] == "neutral"


def test_parse_rich_summary_missing_fields_filled():
    raw = json.dumps({"summary": "minimal"})
    result = _parse_rich_summary(raw, [], "test")
    assert result["summary"] == "minimal"
    assert result["decisions"] == []
    assert result["lessons"] == []
    assert result["surprises"] == []
    assert result["emotional_tone"] == "neutral"


def test_parse_rich_summary_bad_json_falls_back():
    messages = [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi there"},
    ]
    with patch("mycoswarm.memory.summarize_session", return_value="plain fallback"):
        result = _parse_rich_summary("not json at all {{{", messages, "test")
        assert result["summary"] == "plain fallback"
        assert result["decisions"] == []


def test_parse_rich_summary_empty_summary_falls_back():
    messages = [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi"},
    ]
    raw = json.dumps({"summary": "", "decisions": []})
    with patch("mycoswarm.memory.summarize_session", return_value="fallback"):
        result = _parse_rich_summary(raw, messages, "test")
        assert result["summary"] == "fallback"


# ---------------------------------------------------------------------------
# _rich_fallback
# ---------------------------------------------------------------------------

def test_rich_fallback_wraps_plain_summary():
    messages = [
        {"role": "user", "content": "test"},
        {"role": "assistant", "content": "response"},
    ]
    with patch("mycoswarm.memory.summarize_session", return_value="plain summary"):
        result = _rich_fallback(messages, "test")
        assert result["summary"] == "plain summary"
        assert result["decisions"] == []
        assert result["lessons"] == []
        assert result["emotional_tone"] == "neutral"


def test_rich_fallback_none_on_failure():
    with patch("mycoswarm.memory.summarize_session", return_value=None):
        result = _rich_fallback([], "test")
        assert result is None


# ---------------------------------------------------------------------------
# save_session_summary with rich fields
# ---------------------------------------------------------------------------

def test_save_session_summary_rich_fields(tmp_path):
    sessions_path = tmp_path / "sessions.jsonl"
    with patch("mycoswarm.memory.SESSIONS_PATH", sessions_path), \
         patch("mycoswarm.memory.MEMORY_DIR", tmp_path), \
         patch("mycoswarm.memory.split_session_topics", return_value=[{"topic": "test", "summary": "test"}]), \
         patch("mycoswarm.library.index_session_summary"):
        save_session_summary(
            name="test-session",
            model="gemma3:27b",
            summary="Debugged the RAG pipeline",
            count=10,
            grounding_score=0.85,
            decisions=["Switched to user-message injection"],
            lessons=["Local models ignore system message RAG context"],
            surprises=["gemma3:27b hallucinates bees when no context found"],
            emotional_tone="discovery",
        )

    line = sessions_path.read_text().strip()
    entry = json.loads(line)
    assert entry["summary"] == "Debugged the RAG pipeline"
    assert entry["decisions"] == ["Switched to user-message injection"]
    assert entry["lessons"] == ["Local models ignore system message RAG context"]
    assert entry["surprises"] == ["gemma3:27b hallucinates bees when no context found"]
    assert entry["emotional_tone"] == "discovery"
    assert entry["grounding_score"] == 0.85


def test_save_session_summary_no_rich_fields_backward_compat(tmp_path):
    sessions_path = tmp_path / "sessions.jsonl"
    with patch("mycoswarm.memory.SESSIONS_PATH", sessions_path), \
         patch("mycoswarm.memory.MEMORY_DIR", tmp_path), \
         patch("mycoswarm.memory.split_session_topics", return_value=[{"topic": "test", "summary": "test"}]), \
         patch("mycoswarm.library.index_session_summary"):
        save_session_summary(
            name="old-style",
            model="gemma3:27b",
            summary="Plain old summary",
            count=5,
        )

    line = sessions_path.read_text().strip()
    entry = json.loads(line)
    assert entry["summary"] == "Plain old summary"
    assert "decisions" not in entry
    assert "lessons" not in entry
    assert "emotional_tone" not in entry


def test_save_session_summary_indexes_lessons(tmp_path):
    sessions_path = tmp_path / "sessions.jsonl"
    indexed = []

    def mock_index(session_id, summary, date, topic):
        indexed.append({"session_id": session_id, "summary": summary, "topic": topic})

    with patch("mycoswarm.memory.SESSIONS_PATH", sessions_path), \
         patch("mycoswarm.memory.MEMORY_DIR", tmp_path), \
         patch("mycoswarm.memory.split_session_topics", return_value=[{"topic": "debugging", "summary": "Fixed RAG"}]), \
         patch("mycoswarm.library.index_session_summary", side_effect=mock_index):
        save_session_summary(
            name="lesson-test",
            model="gemma3:27b",
            summary="Fixed RAG",
            count=5,
            lessons=["Inject context into user message", "Always check source_filter"],
        )

    # Should have: 1 topic chunk + 2 lesson chunks
    assert len(indexed) == 3
    lesson_chunks = [x for x in indexed if x["topic"] == "lesson_learned"]
    assert len(lesson_chunks) == 2
    assert lesson_chunks[0]["summary"] == "Inject context into user message"


# ---------------------------------------------------------------------------
# format_summaries_for_prompt with rich fields
# ---------------------------------------------------------------------------

def test_format_summaries_rich():
    summaries = [
        {
            "timestamp": "2026-02-15T10:00:00",
            "summary": "Debugged RAG hallucination pipeline",
            "emotional_tone": "discovery",
            "lessons": ["When model ignores context, check injection method"],
            "decisions": ["Switched from system message to user message injection"],
        },
    ]
    result = format_summaries_for_prompt(summaries)
    assert "(discovery)" in result
    assert "Lesson:" in result
    assert "Decision:" in result
    assert "injection method" in result


def test_format_summaries_neutral_tone_hidden():
    summaries = [
        {
            "timestamp": "2026-02-15T10:00:00",
            "summary": "Casual chat about weather",
            "emotional_tone": "neutral",
        },
    ]
    result = format_summaries_for_prompt(summaries)
    assert "(neutral)" not in result
    assert "Casual chat" in result


def test_format_summaries_old_format_no_crash():
    summaries = [
        {"timestamp": "2026-02-14T10:00:00", "summary": "Old style summary"},
    ]
    result = format_summaries_for_prompt(summaries)
    assert "Old style summary" in result
    assert "Lesson:" not in result


def test_format_summaries_caps_lessons_and_decisions():
    summaries = [
        {
            "timestamp": "2026-02-15T10:00:00",
            "summary": "Big session",
            "lessons": ["L1", "L2", "L3", "L4", "L5"],
            "decisions": ["D1", "D2", "D3", "D4"],
        },
    ]
    result = format_summaries_for_prompt(summaries)
    # Max 3 lessons, 2 decisions
    assert result.count("Lesson:") == 3
    assert result.count("Decision:") == 2


# ---------------------------------------------------------------------------
# Valid tones
# ---------------------------------------------------------------------------

def test_valid_tones_complete():
    expected = {
        "frustration", "discovery", "confusion", "resolution",
        "flow", "stuck", "exploratory", "routine", "neutral",
    }
    assert VALID_TONES == expected


# ---------------------------------------------------------------------------
# summarize_session_rich (integration-ish, mocked Ollama)
# ---------------------------------------------------------------------------

def test_summarize_session_rich_too_short():
    result = summarize_session_rich([{"role": "user", "content": "hi"}], "test")
    assert result is None


def test_summarize_session_rich_parses_response():
    messages = [
        {"role": "user", "content": "Fix the RAG bug"},
        {"role": "assistant", "content": "Found the issue - context was injected wrong"},
    ]
    mock_response = json.dumps({
        "summary": "Fixed RAG context injection bug",
        "decisions": ["Use user message instead of system message"],
        "lessons": ["gemma3:27b ignores system message RAG context"],
        "surprises": [],
        "emotional_tone": "resolution",
    })

    mock_resp = MagicMock()
    mock_resp.json.return_value = {"message": {"content": mock_response}}
    mock_resp.raise_for_status = MagicMock()

    mock_client = MagicMock()
    mock_client.__enter__ = MagicMock(return_value=mock_client)
    mock_client.__exit__ = MagicMock(return_value=False)
    mock_client.post.return_value = mock_resp

    with patch("httpx.Client", return_value=mock_client):
        result = summarize_session_rich(messages, "gemma3:27b")

    assert result["summary"] == "Fixed RAG context injection bug"
    assert result["emotional_tone"] == "resolution"
    assert len(result["decisions"]) == 1
    assert len(result["lessons"]) == 1


def test_summarize_session_rich_falls_back_on_error():
    messages = [
        {"role": "user", "content": "test"},
        {"role": "assistant", "content": "response"},
    ]

    mock_client = MagicMock()
    mock_client.__enter__ = MagicMock(return_value=mock_client)
    mock_client.__exit__ = MagicMock(return_value=False)
    mock_client.post.side_effect = Exception("connection refused")

    with patch("httpx.Client", return_value=mock_client), \
         patch("mycoswarm.memory.summarize_session", return_value="fallback text"):
        result = summarize_session_rich(messages, "test")

    assert result["summary"] == "fallback text"
    assert result["emotional_tone"] == "neutral"
