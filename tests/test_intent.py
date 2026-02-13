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
    def test_prefers_gemma3_1b(self, mock_client_cls):
        from mycoswarm.solo import _pick_gate_model

        models = ["qwen2.5:14b", "gemma3:1b", "llama3.2:3b"]
        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.get.return_value = _mock_ollama_tags(models)
        mock_client_cls.return_value = mock_client

        result = _pick_gate_model()
        assert result == "gemma3:1b"

    @patch("mycoswarm.solo.httpx.Client")
    def test_prefers_llama32_1b_over_4b(self, mock_client_cls):
        from mycoswarm.solo import _pick_gate_model

        models = ["qwen2.5:14b", "llama3.2:1b", "gemma3:4b"]
        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.get.return_value = _mock_ollama_tags(models)
        mock_client_cls.return_value = mock_client

        result = _pick_gate_model()
        assert result == "llama3.2:1b"

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
        assert result == {"tool": "answer", "mode": "chat", "scope": "all"}

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

        task = make_task({"query": "hello"})  # no model

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
        from mycoswarm.orchestrator import TASK_ROUTING

        assert "intent_classify" in TASK_ROUTING
        assert "cpu_worker" in TASK_ROUTING["intent_classify"]

    def test_handlers_includes_intent_classify(self):
        from mycoswarm.worker import HANDLERS

        assert "intent_classify" in HANDLERS
        assert callable(HANDLERS["intent_classify"])

    def test_distributable_tasks_includes_intent_classify(self):
        """DISTRIBUTABLE_TASKS is defined inside create_api — verify via source."""
        import inspect
        from mycoswarm.api import create_api

        source = inspect.getsource(create_api)
        assert "intent_classify" in source
