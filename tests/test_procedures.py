"""Tests for Phase 21d: Procedural Memory."""

import json
import pytest
from datetime import datetime, timedelta
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
    expire_old_candidates,
    _is_duplicate_procedure,
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


class TestDuplicateDetection:
    @patch("mycoswarm.library.index_procedure", return_value=True)
    def test_exact_duplicate_detected(self, mock_idx):
        """Exact same problem text should be detected as duplicate."""
        add_procedure("RAG context ignored by local models", "Inject into user message")
        assert _is_duplicate_procedure("RAG context ignored by local models") is True

    @patch("mycoswarm.library.index_procedure", return_value=True)
    def test_paraphrase_detected(self, mock_idx):
        """Paraphrased problem with high word overlap should be detected."""
        add_procedure(
            "RAG context in system message gets ignored by local models",
            "Inject into user message",
        )
        assert _is_duplicate_procedure(
            "Local models ignore system message RAG context"
        ) is True

    @patch("mycoswarm.library.index_procedure", return_value=True)
    def test_different_passes(self, mock_idx):
        """Unrelated problem should not be flagged as duplicate."""
        add_procedure("RAG context ignored by local models", "Inject into user message")
        assert _is_duplicate_procedure(
            "Python asyncio error handling"
        ) is False

    def test_empty_procedures_no_duplicate(self):
        """No existing procedures means nothing is a duplicate."""
        assert _is_duplicate_procedure("Any problem at all") is False


class TestSessionCap:
    @patch("mycoswarm.library.index_procedure", return_value=True)
    @patch("mycoswarm.memory.extract_procedure_from_lesson")
    @patch("mycoswarm.memory.split_session_topics", return_value=[{"topic": "general", "summary": "test"}])
    @patch("mycoswarm.library.index_session_summary")
    def test_max_3_candidates_per_session(self, mock_idx_sess, mock_split, mock_extract, mock_idx):
        """Only 3 candidates should be created per session even with 6 lessons."""
        from mycoswarm.memory import save_session_summary
        mock_extract.return_value = {
            "problem": "unique problem",
            "solution": "unique solution",
            "reasoning": "reason",
            "anti_patterns": [],
            "tags": [],
        }
        # Make each call return a truly different problem so dedup doesn't trigger
        different_problems = [
            "RAG context gets ignored by local models",
            "asyncio create_task swallows exceptions silently",
            "ChromaDB collection count mismatch after reindex",
            "zeroconf mDNS announcements stop after network change",
            "BM25 index returns stale results after document deletion",
            "FastAPI endpoint timeout when Ollama is overloaded",
        ]
        call_count = [0]
        def unique_extract(*args, **kwargs):
            idx = call_count[0]
            call_count[0] += 1
            return {
                "problem": different_problems[idx % len(different_problems)],
                "solution": "fix it",
                "reasoning": "reason",
                "anti_patterns": [],
                "tags": [],
            }
        mock_extract.side_effect = unique_extract

        save_session_summary(
            name="test-cap",
            model="test",
            summary="test summary",
            count=10,
            lessons=["lesson1", "lesson2", "lesson3", "lesson4", "lesson5", "lesson6"],
        )
        candidates = load_procedure_candidates()
        assert len(candidates) == 3


class TestAutoExpire:
    def test_expires_old_candidates(self, tmp_procedures):
        """Candidates older than 14 days should be expired."""
        old_date = (datetime.now() - timedelta(days=15)).isoformat()
        proc = {
            "id": "proc_001", "status": "candidate",
            "problem": "old problem", "solution": "old solution",
            "created": old_date, "last_used": old_date, "use_count": 0,
        }
        tmp_procedures.write_text(json.dumps(proc) + "\n")
        count = expire_old_candidates()
        assert count == 1
        assert load_procedure_candidates() == []

    def test_keeps_recent_candidates(self, tmp_procedures):
        """Candidates less than 14 days old should survive."""
        recent_date = (datetime.now() - timedelta(days=5)).isoformat()
        proc = {
            "id": "proc_001", "status": "candidate",
            "problem": "recent problem", "solution": "recent solution",
            "created": recent_date, "last_used": recent_date, "use_count": 0,
        }
        tmp_procedures.write_text(json.dumps(proc) + "\n")
        count = expire_old_candidates()
        assert count == 0
        assert len(load_procedure_candidates()) == 1

    def test_keeps_active_procedures(self, tmp_procedures):
        """Active procedures should never be expired regardless of age."""
        old_date = (datetime.now() - timedelta(days=30)).isoformat()
        proc = {
            "id": "proc_001", "status": "active",
            "problem": "old active", "solution": "still valid",
            "created": old_date, "last_used": old_date, "use_count": 5,
        }
        tmp_procedures.write_text(json.dumps(proc) + "\n")
        count = expire_old_candidates()
        assert count == 0
        assert len(load_active_procedures()) == 1
