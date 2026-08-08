"""Tests for intent classification — solo.py, worker.py, routing registration."""

import json
from unittest.mock import patch, MagicMock, AsyncMock

import pytest

from mycoswarm.api import TaskRequest, TaskResult, TaskStatus


# --- Helpers ---


def _mock_ollama_tags(models: list[str]):
    """Return a mock httpx response for /api/tags."""
    resp = MagicMock()
    resp.json.return_value = {"models": [{"name": m} for m in models]}
    resp.raise_for_status = MagicMock()
    return resp


def _mock_ollama_chat(content: str):
    """Return a mock httpx response for /api/chat."""
    resp = MagicMock()
    resp.json.return_value = {"message": {"content": content}}
    resp.raise_for_status = MagicMock()
    return resp


# ============================================================
# TestPickGateModel — solo.py _pick_gate_model()
# ============================================================


class TestPickGateModel:
    """Test _pick_gate_model preference order and embedding exclusion."""

    @patch("mycoswarm.solo.httpx.Client")
    def test_prefers_gemma3_4b_over_1b(self, mock_client_cls):
        """4b leads the preference list on measured evidence, not size.

        intent-eval-2026-08-07: accuracy is statistically identical to 1b
        (McNemar p=1.000) but 1b emitted a schema-invalid enum on 51% of inputs
        vs 4b's 0%, and 1b is 1.7x SLOWER on the real ~650-token prompt.
        """
        from mycoswarm.solo import _pick_gate_model

        models = ["qwen2.5:14b", "gemma3:1b", "gemma3:4b", "llama3.2:3b"]
        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.get.return_value = _mock_ollama_tags(models)
        mock_client_cls.return_value = mock_client

        result = _pick_gate_model()
        assert result == "gemma3:4b"

    @patch("mycoswarm.solo.httpx.Client")
    def test_falls_back_to_gemma3_1b_when_4b_absent(self, mock_client_cls):
        """Nodes too small for 4b must still get a gate model."""
        from mycoswarm.solo import _pick_gate_model

        models = ["qwen2.5:14b", "gemma3:1b"]
        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.get.return_value = _mock_ollama_tags(models)
        mock_client_cls.return_value = mock_client

        result = _pick_gate_model()
        assert result == "gemma3:1b"

    @patch("mycoswarm.solo.httpx.Client")
    def test_prefers_llama32_3b_over_1b_models(self, mock_client_cls):
        """With no gemma3:4b, the 3B-class llama beats either 1b."""
        from mycoswarm.solo import _pick_gate_model

        models = ["qwen2.5:14b", "llama3.2:1b", "llama3.2:3b", "gemma3:1b"]
        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.get.return_value = _mock_ollama_tags(models)
        mock_client_cls.return_value = mock_client

        result = _pick_gate_model()
        assert result == "llama3.2:3b"

    def test_solo_and_worker_share_one_preference_list(self):
        """The regression this change exists to prevent: the two paths had
        drifted (solo preferred gemma3:1b, worker preferred gemma3:4b), so the
        CLI and the daemon classified the same query with different models."""
        from mycoswarm.bindings import GATE_MODEL_PREFERENCE
        from mycoswarm.worker import _GATE_MODEL_PREFERENCE

        assert _GATE_MODEL_PREFERENCE is GATE_MODEL_PREFERENCE
        assert GATE_MODEL_PREFERENCE[0] == "gemma3:4b"

    def test_agrees_with_task_model_map_classification(self):
        """bindings.GATE_MODEL_PREFERENCE is deliberately NOT read from
        TASK_MODEL_MAP (different consumers/semantics — see the comment there),
        so this asserts they at least still agree on the head model."""
        from mycoswarm.bindings import GATE_MODEL_PREFERENCE
        from mycoswarm.capabilities import TASK_MODEL_MAP

        declared = TASK_MODEL_MAP["classification"]["prefer_models"]
        assert declared[0] == GATE_MODEL_PREFERENCE[0] == "gemma3:4b"

    @patch("mycoswarm.solo.httpx.Client")
    def test_falls_back_to_first_non_embedding_model(self, mock_client_cls):
        from mycoswarm.solo import _pick_gate_model

        models = ["nomic-embed-text:latest", "phi3:mini", "mistral:7b"]
        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.get.return_value = _mock_ollama_tags(models)
        mock_client_cls.return_value = mock_client

        result = _pick_gate_model()
        assert result == "phi3:mini"

    @patch("mycoswarm.solo.httpx.Client")
    def test_skips_embedding_only_models(self, mock_client_cls):
        from mycoswarm.solo import _pick_gate_model

        models = ["nomic-embed-text:latest", "mxbai-embed-large", "all-minilm:v2"]
        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.get.return_value = _mock_ollama_tags(models)
        mock_client_cls.return_value = mock_client

        result = _pick_gate_model()
        assert result is None

    @patch("mycoswarm.solo.httpx.Client")
    def test_returns_none_on_connection_error(self, mock_client_cls):
        import httpx
        from mycoswarm.solo import _pick_gate_model

        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.get.side_effect = httpx.ConnectError("refused")
        mock_client_cls.return_value = mock_client

        result = _pick_gate_model()
        assert result is None

    @patch("mycoswarm.solo.httpx.Client")
    def test_returns_none_on_empty_models(self, mock_client_cls):
        from mycoswarm.solo import _pick_gate_model

        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.get.return_value = _mock_ollama_tags([])
        mock_client_cls.return_value = mock_client

        result = _pick_gate_model()
        assert result is None


# ============================================================
# TestIntentClassify — solo.py intent_classify()
# ============================================================


class TestIntentClassify:
    """Test intent_classify() with mocked Ollama."""

    @patch("mycoswarm.solo.httpx.Client")
    def test_valid_json_response(self, mock_client_cls):
        from mycoswarm.solo import intent_classify

        ollama_reply = json.dumps({
            "tool": "web_search",
            "mode": "explore",
            "scope": "all",
        })

        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.post.return_value = _mock_ollama_chat(ollama_reply)
        mock_client_cls.return_value = mock_client

        result = intent_classify("What's the weather today?", model="gemma3:1b")
        assert result["tool"] == "web_search"
        assert result["mode"] == "explore"
        assert result["scope"] == "all"

    @patch("mycoswarm.solo.httpx.Client")
    def test_fallback_on_connection_error(self, mock_client_cls):
        import httpx
        from mycoswarm.solo import intent_classify

        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.post.side_effect = httpx.ConnectError("refused")
        mock_client_cls.return_value = mock_client

        result = intent_classify("hello", model="gemma3:1b")
        assert result["tool"] == "answer"
        assert result["mode"] == "chat"
        assert result["scope"] == "all"

    @patch("mycoswarm.solo.httpx.Client")
    def test_fallback_on_malformed_json(self, mock_client_cls):
        from mycoswarm.solo import intent_classify

        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.post.return_value = _mock_ollama_chat("not valid json {{{")
        mock_client_cls.return_value = mock_client

        result = intent_classify("hello", model="gemma3:1b")
        assert result["tool"] == "answer"
        assert result["mode"] == "chat"

    @patch("mycoswarm.solo.httpx.Client")
    def test_extracts_tool_from_plaintext_fallback(self, mock_client_cls):
        from mycoswarm.solo import intent_classify

        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        # Model returns raw text instead of JSON
        mock_client.post.return_value = _mock_ollama_chat("I think this is web_search.")
        mock_client_cls.return_value = mock_client

        result = intent_classify("latest news today", model="gemma3:1b")
        assert result["tool"] == "web_search"

    @patch("mycoswarm.solo.httpx.Client")
    def test_past_reference_overrides_scope_to_session(self, mock_client_cls):
        from mycoswarm.solo import intent_classify

        ollama_reply = json.dumps({
            "tool": "rag",
            "mode": "recall",
            "scope": "docs",
        })

        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.post.return_value = _mock_ollama_chat(ollama_reply)
        mock_client_cls.return_value = mock_client

        # "we discussed" triggers past-reference regex → scope becomes "session"
        result = intent_classify("What did we discussed yesterday?", model="gemma3:1b")
        assert result["scope"] == "session"

    def test_returns_default_when_no_model(self):
        from mycoswarm.solo import intent_classify

        # No model arg, and _pick_gate_model returns None
        with patch("mycoswarm.solo._pick_gate_model", return_value=None):
            result = intent_classify("test query")
        assert {k: result[k] for k in ("tool", "mode", "scope")} == {
            "tool": "answer", "mode": "chat", "scope": "all"}
        # provenance says WHY it is the default, so a silent fallback is visible
        assert result["_via"] == "no_model"

    @patch("mycoswarm.solo.httpx.Client")
    def test_validates_mode_field(self, mock_client_cls):
        from mycoswarm.solo import intent_classify

        ollama_reply = json.dumps({
            "tool": "rag",
            "mode": "recall",
            "scope": "session",
        })

        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.post.return_value = _mock_ollama_chat(ollama_reply)
        mock_client_cls.return_value = mock_client

        result = intent_classify("what did we say about X?", model="gemma3:1b")
        assert result["mode"] == "recall"

    @patch("mycoswarm.solo.httpx.Client")
    def test_invalid_mode_uses_default(self, mock_client_cls):
        from mycoswarm.solo import intent_classify

        ollama_reply = json.dumps({
            "tool": "answer",
            "mode": "flying",
            "scope": "all",
        })

        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.post.return_value = _mock_ollama_chat(ollama_reply)
        mock_client_cls.return_value = mock_client

        result = intent_classify("hello", model="gemma3:1b")
        assert result["mode"] == "chat"

    @patch("mycoswarm.solo.httpx.Client")
    def test_invalid_tool_uses_default(self, mock_client_cls):
        from mycoswarm.solo import intent_classify

        ollama_reply = json.dumps({
            "tool": "magic_tool",
            "mode": "explore",
            "scope": "all",
        })

        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.post.return_value = _mock_ollama_chat(ollama_reply)
        mock_client_cls.return_value = mock_client

        result = intent_classify("hello", model="gemma3:1b")
        assert result["tool"] == "answer"


# ============================================================
# TestHandleIntentClassify — worker.py handler
# ============================================================


class TestHandleIntentClassify:
    """Test handle_intent_classify async handler."""

    @pytest.fixture
    def make_task(self):
        def _make(payload: dict):
            return TaskRequest(
                task_id="test-intent-001",
                task_type="intent_classify",
                payload=payload,
                source_node="myco-testnode00",
            )
        return _make

    @pytest.mark.asyncio
    async def test_successful_classification(self, make_task):
        from mycoswarm.worker import handle_intent_classify

        task = make_task({"query": "What's the weather?", "model": "gemma3:1b"})

        ollama_reply = json.dumps({
            "tool": "web_search",
            "mode": "explore",
            "scope": "all",
        })

        mock_resp = MagicMock()
        mock_resp.json.return_value = {"message": {"content": ollama_reply}}
        mock_resp.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_resp)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)

        with patch("mycoswarm.worker.httpx.AsyncClient", return_value=mock_client):
            result = await handle_intent_classify(task)

        assert result.status == TaskStatus.COMPLETED
        assert result.result["tool"] == "web_search"
        assert result.result["mode"] == "explore"
        assert result.result["scope"] == "all"

    @pytest.mark.asyncio
    async def test_missing_query_fails(self, make_task):
        from mycoswarm.worker import handle_intent_classify

        task = make_task({"model": "gemma3:1b"})  # no query
        result = await handle_intent_classify(task)

        assert result.status == TaskStatus.FAILED
        assert "query" in result.error

    @pytest.mark.asyncio
    async def test_connection_error_fails(self, make_task):
        import httpx
        from mycoswarm.worker import handle_intent_classify

        task = make_task({"query": "test", "model": "gemma3:1b"})

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(side_effect=httpx.ConnectError("refused"))
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)

        with patch("mycoswarm.worker.httpx.AsyncClient", return_value=mock_client):
            result = await handle_intent_classify(task)

        assert result.status == TaskStatus.FAILED
        assert "connect" in result.error.lower() or "Ollama" in result.error

    @pytest.mark.asyncio
    async def test_timeout_fails(self, make_task):
        import httpx
        from mycoswarm.worker import handle_intent_classify

        task = make_task({"query": "test", "model": "gemma3:1b"})

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(side_effect=httpx.TimeoutException("slow"))
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)

        with patch("mycoswarm.worker.httpx.AsyncClient", return_value=mock_client):
            result = await handle_intent_classify(task)

        assert result.status == TaskStatus.FAILED
        assert "timed out" in result.error.lower()

    @pytest.mark.asyncio
    async def test_result_has_required_fields(self, make_task):
        from mycoswarm.worker import handle_intent_classify

        task = make_task({"query": "Tell me about Python", "model": "gemma3:1b"})

        ollama_reply = json.dumps({
            "tool": "answer",
            "mode": "explore",
            "scope": "all",
        })

        mock_resp = MagicMock()
        mock_resp.json.return_value = {"message": {"content": ollama_reply}}
        mock_resp.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_resp)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)

        with patch("mycoswarm.worker.httpx.AsyncClient", return_value=mock_client):
            result = await handle_intent_classify(task)

        assert result.status == TaskStatus.COMPLETED
        assert "tool" in result.result
        assert "mode" in result.result
        assert "scope" in result.result

    @pytest.mark.asyncio
    async def test_no_model_picks_gate_model(self, make_task):
        from mycoswarm.worker import handle_intent_classify

        task = make_task({"query": "hello world"})  # no model

        ollama_reply = json.dumps({
            "tool": "answer", "mode": "chat", "scope": "all",
        })

        mock_resp = MagicMock()
        mock_resp.json.return_value = {"message": {"content": ollama_reply}}
        mock_resp.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_resp)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)

        # Mock _pick_gate_model_async as an async function returning a model
        async def _fake_pick():
            return "gemma3:1b"

        with patch("mycoswarm.worker._pick_gate_model_async", side_effect=_fake_pick), \
             patch("mycoswarm.worker.httpx.AsyncClient", return_value=mock_client):
            result = await handle_intent_classify(task)

        assert result.status == TaskStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_no_model_no_ollama_fails(self, make_task):
        from mycoswarm.worker import handle_intent_classify

        # NB: must be a query no fast rule intercepts — "hello" now short-
        # circuits to small_talk and never reaches model selection at all.
        task = make_task({"query": "what does readiness mean to you"})  # no model

        with patch("mycoswarm.worker._pick_gate_model_async", return_value=None):
            result = await handle_intent_classify(task)

        assert result.status == TaskStatus.FAILED
        assert "No model" in result.error


# ============================================================
# TestClassifyQueryBackcompat — solo.py backward compat
# ============================================================


class TestClassifyQueryBackcompat:
    """Test classify_query() still returns a string."""

    @patch("mycoswarm.solo.intent_classify")
    def test_returns_string(self, mock_intent):
        from mycoswarm.solo import classify_query

        mock_intent.return_value = {
            "tool": "web_search",
            "mode": "explore",
            "scope": "all",
        }
        result = classify_query("weather today", "gemma3:1b")
        assert isinstance(result, str)
        assert result == "web_search"

    @patch("mycoswarm.solo.intent_classify")
    def test_returns_valid_categories(self, mock_intent):
        from mycoswarm.solo import classify_query

        for tool in ("answer", "web_search", "rag", "web_and_rag"):
            mock_intent.return_value = {"tool": tool, "mode": "chat", "scope": "all"}
            result = classify_query("test", "model")
            assert result == tool
            assert result in {"answer", "web_search", "rag", "web_and_rag"}


# ============================================================
# TestIntentRouting — registration in orchestrator/api/worker
# ============================================================


class TestIntentRouting:
    """Test intent_classify is registered in routing tables."""

    def test_task_routing_includes_intent_classify(self):
        from mycoswarm.router import TASK_ROUTING

        assert "intent_classify" in TASK_ROUTING
        assert "cpu_worker" in TASK_ROUTING["intent_classify"]

    def test_handlers_includes_intent_classify(self):
        from mycoswarm.worker import HANDLERS

        assert "intent_classify" in HANDLERS
        assert callable(HANDLERS["intent_classify"])

    def test_distributable_tasks_includes_intent_classify(self):
        """intent_classify is in DISTRIBUTABLE_TASKS (now in router.py)."""
        from mycoswarm.router import DISTRIBUTABLE_TASKS

        assert "intent_classify" in DISTRIBUTABLE_TASKS


class TestSanitiserLogging:
    """The sanitiser repairs invalid enums silently, which is why a 51%
    malformation rate from gemma3:1b was invisible in production. It must now
    leave a DEBUG trace naming the offending value."""

    @patch("mycoswarm.solo.httpx.Client")
    def test_invalid_tool_is_logged_at_debug(self, mock_client_cls, caplog):
        import logging as _logging

        from mycoswarm.solo import intent_classify

        # the exact real-world failure: "chat" is a MODE, not a tool
        resp = MagicMock()
        resp.raise_for_status = MagicMock()
        resp.json.return_value = {
            "message": {"content": '{"tool": "chat", "mode": "chat", "scope": "all"}'}
        }
        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.post.return_value = resp
        mock_client_cls.return_value = mock_client

        # deliberately NOT a greeting: small talk short-circuits before the
        # model, so it could never exercise the sanitiser
        with caplog.at_level(_logging.DEBUG, logger="mycoswarm.solo"):
            result = intent_classify("what does readiness mean to you",
                                     model="gemma3:1b")

        # repair still happens — behaviour unchanged
        assert result["tool"] == "answer"
        # ...but it is no longer silent, and names the bad value
        assert any("sanitiser" in r.message and "chat" in r.message
                   for r in caplog.records), caplog.text
        assert all(r.levelno == _logging.DEBUG
                   for r in caplog.records if "sanitiser" in r.message)

    @patch("mycoswarm.solo.httpx.Client")
    def test_valid_output_logs_nothing(self, mock_client_cls, caplog):
        import logging as _logging

        from mycoswarm.solo import intent_classify

        resp = MagicMock()
        resp.raise_for_status = MagicMock()
        resp.json.return_value = {
            "message": {"content": '{"tool": "rag", "mode": "recall", "scope": "docs"}'}
        }
        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.post.return_value = resp
        mock_client_cls.return_value = mock_client

        with caplog.at_level(_logging.DEBUG, logger="mycoswarm.solo"):
            result = intent_classify("what does PLAN.md say about Phase 37?",
                                     model="gemma3:4b")

        assert {k: result[k] for k in ("tool", "mode", "scope")} == {
            "tool": "rag", "mode": "recall", "scope": "docs"}
        assert result["_via"] == "model" and result["_model"] == "gemma3:4b"
        assert not any("sanitiser" in r.message for r in caplog.records)
