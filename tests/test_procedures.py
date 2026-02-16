"""Tests for Phase 21d: Procedural Memory."""

import json
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from mycoswarm.memory import (
    add_procedure,
    load_procedures,
    remove_procedure,
    reference_procedure,
    format_procedures_for_prompt,
    promote_lesson_to_procedure,
    add_procedure_candidate,
    load_procedure_candidates,
    load_active_procedures,
    approve_procedure,
    reject_procedure,
    extract_procedure_from_lesson,
    PROCEDURES_PATH,
)


@pytest.fixture(autouse=True)
def tmp_procedures(tmp_path, monkeypatch):
    """Use temp dir for procedures."""
    proc_path = tmp_path / "procedures.jsonl"
    monkeypatch.setattr("mycoswarm.memory.PROCEDURES_PATH", proc_path)
    monkeypatch.setattr("mycoswarm.memory.MEMORY_DIR", tmp_path)
    return proc_path


class TestAddProcedure:
    @patch("mycoswarm.library.index_procedure", return_value=True)
    def test_basic_add(self, mock_idx):
        proc = add_procedure("test problem", "test solution")
        assert proc["id"] == "proc_001"
        assert proc["problem"] == "test problem"
        assert proc["solution"] == "test solution"
        assert proc["outcome"] == "success"
        assert proc["use_count"] == 0

    @patch("mycoswarm.library.index_procedure", return_value=True)
    def test_auto_increment_id(self, mock_idx):
        add_procedure("p1", "s1")
        proc2 = add_procedure("p2", "s2")
        assert proc2["id"] == "proc_002"

    @patch("mycoswarm.library.index_procedure", return_value=True)
    def test_with_all_fields(self, mock_idx):
        proc = add_procedure(
            "problem", "solution",
            reasoning="because reasons",
            anti_patterns=["don't do this"],
            outcome="failure",
            tags=["rag", "debug"],
            source_session="test-session",
        )
        assert proc["reasoning"] == "because reasons"
        assert proc["anti_patterns"] == ["don't do this"]
        assert proc["outcome"] == "failure"
        assert proc["tags"] == ["rag", "debug"]


class TestLoadProcedures:
    @patch("mycoswarm.library.index_procedure", return_value=True)
    def test_load_empty(self, mock_idx):
        assert load_procedures() == []

    @patch("mycoswarm.library.index_procedure", return_value=True)
    def test_load_after_add(self, mock_idx):
        add_procedure("p1", "s1")
        add_procedure("p2", "s2")
        procs = load_procedures()
        assert len(procs) == 2
        assert procs[0]["problem"] == "p1"
        assert procs[1]["problem"] == "p2"


class TestRemoveProcedure:
    @patch("mycoswarm.library.index_procedure", return_value=True)
    def test_remove_existing(self, mock_idx):
        add_procedure("p1", "s1")
        assert remove_procedure("proc_001") is True
        assert load_procedures() == []

    def test_remove_nonexistent(self):
        assert remove_procedure("proc_999") is False


class TestReferenceProcedure:
    @patch("mycoswarm.library.index_procedure", return_value=True)
    def test_reference_increments(self, mock_idx):
        add_procedure("p1", "s1")
        assert reference_procedure("proc_001") is True
        procs = load_procedures()
        assert procs[0]["use_count"] == 1

    @patch("mycoswarm.library.index_procedure", return_value=True)
    def test_reference_updates_timestamp(self, mock_idx):
        add_procedure("p1", "s1")
        original = load_procedures()[0]["last_used"]
        reference_procedure("proc_001")
        updated = load_procedures()[0]["last_used"]
        assert updated >= original

    def test_reference_nonexistent(self):
        assert reference_procedure("proc_999") is False


class TestFormatProcedures:
    def test_empty(self):
        assert format_procedures_for_prompt([]) == ""

    def test_success_procedure(self):
        proc = {
            "id": "proc_001",
            "problem": "RAG ignored",
            "solution": "Inject into user message",
            "reasoning": "Models prefer user messages",
            "anti_patterns": ["Don't use system message"],
            "outcome": "success",
        }
        result = format_procedures_for_prompt([proc])
        assert "[P1]" in result
        assert "RAG ignored" in result
        assert "Inject into user message" in result
        assert "Models prefer user messages" in result
        assert "Don't use system message" in result

    def test_failure_procedure(self):
        proc = {
            "id": "proc_001",
            "problem": "Bad approach",
            "solution": "This failed",
            "outcome": "failure",
        }
        result = format_procedures_for_prompt([proc])
        assert "[P1]" in result


class TestPromoteLesson:
    @patch("mycoswarm.library.index_procedure", return_value=True)
    def test_promotes_actionable_lesson(self, mock_idx):
        result = promote_lesson_to_procedure(
            "RAG context should be injected into user message instead of system message"
        )
        assert result is not None
        assert result["id"] == "proc_001"

    @patch("mycoswarm.library.index_procedure", return_value=True)
    def test_skips_non_actionable(self, mock_idx):
        result = promote_lesson_to_procedure(
            "Taoist philosophy emphasizes individual spiritual growth"
        )
        assert result is None

    @patch("mycoswarm.library.index_procedure", return_value=True)
    def test_promotes_with_avoid(self, mock_idx):
        result = promote_lesson_to_procedure(
            "Avoid injecting ungrounded summaries into the index"
        )
        assert result is not None


class TestProcedureCandidates:
    @patch("mycoswarm.library.index_procedure", return_value=True)
    def test_add_candidate_not_indexed(self, mock_idx):
        """Candidates should NOT be indexed into ChromaDB."""
        candidate = add_procedure_candidate(
            "Test lesson",
            extracted={
                "problem": "Test problem",
                "solution": "Test solution",
                "reasoning": "Test reasoning",
                "anti_patterns": [],
                "tags": ["test"],
            },
            session_name="test-session",
        )
        assert candidate["status"] == "candidate"
        assert candidate["source_lesson"] == "Test lesson"
        # index_procedure should NOT have been called
        mock_idx.assert_not_called()

    @patch("mycoswarm.library.index_procedure", return_value=True)
    def test_load_candidates_only(self, mock_idx):
        add_procedure("active problem", "active solution")
        add_procedure_candidate(
            "lesson",
            extracted={"problem": "cand problem", "solution": "cand solution"},
        )
        candidates = load_procedure_candidates()
        assert len(candidates) == 1
        assert candidates[0]["status"] == "candidate"

    @patch("mycoswarm.library.index_procedure", return_value=True)
    def test_load_active_only(self, mock_idx):
        add_procedure("active problem", "active solution")
        add_procedure_candidate(
            "lesson",
            extracted={"problem": "cand problem", "solution": "cand solution"},
        )
        active = load_active_procedures()
        assert len(active) == 1
        assert active[0]["status"] == "active"

    @patch("mycoswarm.library.index_procedure", return_value=True)
    def test_approve_candidate(self, mock_idx):
        candidate = add_procedure_candidate(
            "lesson",
            extracted={"problem": "p", "solution": "s"},
        )
        mock_idx.reset_mock()
        result = approve_procedure(candidate["id"])
        assert result is True
        # Should now be active
        procs = load_procedures()
        approved = [p for p in procs if p["id"] == candidate["id"]]
        assert approved[0]["status"] == "active"
        # Should have been indexed
        mock_idx.assert_called_once()

    @patch("mycoswarm.library.index_procedure", return_value=True)
    def test_reject_candidate(self, mock_idx):
        candidate = add_procedure_candidate(
            "lesson",
            extracted={"problem": "p", "solution": "s"},
        )
        result = reject_procedure(candidate["id"])
        assert result is True
        assert load_procedure_candidates() == []

    def test_approve_nonexistent(self):
        assert approve_procedure("proc_999") is False

    def test_reject_nonexistent(self):
        assert reject_procedure("proc_999") is False

    @patch("mycoswarm.library.index_procedure", return_value=True)
    def test_approve_active_fails(self, mock_idx):
        """Can't approve an already-active procedure."""
        proc = add_procedure("p", "s")
        assert approve_procedure(proc["id"]) is False

    @patch("mycoswarm.library.index_procedure", return_value=True)
    def test_candidates_excluded_from_prompt(self, mock_idx):
        add_procedure("active problem", "active solution")
        add_procedure_candidate(
            "lesson",
            extracted={"problem": "cand problem", "solution": "cand solution"},
        )
        result = format_procedures_for_prompt(load_procedures())
        assert "active problem" in result
        assert "cand problem" not in result


class TestExtractProcedure:
    @patch("httpx.post")
    def test_extracts_procedural_lesson(self, mock_post):
        mock_post.return_value = MagicMock(
            status_code=200,
            json=lambda: {"response": json.dumps({
                "is_procedural": True,
                "problem": "RAG context ignored",
                "solution": "Inject into user message",
                "reasoning": "Models prioritize user messages",
                "anti_patterns": ["Don't use system message"],
                "tags": ["rag", "prompt"],
            })},
        )
        result = extract_procedure_from_lesson("RAG should be injected into user message")
        assert result is not None
        assert result["problem"] == "RAG context ignored"
        assert result["solution"] == "Inject into user message"
        assert len(result["tags"]) == 2

    @patch("httpx.post")
    def test_rejects_non_procedural(self, mock_post):
        mock_post.return_value = MagicMock(
            status_code=200,
            json=lambda: {"response": '{"is_procedural": false}'},
        )
        result = extract_procedure_from_lesson("Taoist philosophy is interesting")
        assert result is None

    @patch("httpx.post")
    def test_handles_ollama_failure(self, mock_post):
        mock_post.side_effect = Exception("Connection refused")
        result = extract_procedure_from_lesson("Some lesson")
        assert result is None
