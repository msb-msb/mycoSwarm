"""Tests for agentic chat classification and tool routing."""

from unittest.mock import patch, MagicMock

from mycoswarm.solo import classify_query, web_search_solo


class TestClassifyQuery:
    @patch("mycoswarm.solo.httpx.Client")
    def test_classifies_web_search(self, mock_client_cls):
        mock_client = MagicMock()
        mock_client_cls.return_value.__enter__ = MagicMock(return_value=mock_client)
        mock_client_cls.return_value.__exit__ = MagicMock(return_value=False)

        mock_resp = MagicMock()
        mock_resp.json.return_value = {"message": {"content": "web_search"}}
        mock_resp.raise_for_status = MagicMock()
        mock_client.post.return_value = mock_resp

        result = classify_query("What's the weather in Tokyo today?", "gemma3:27b")
        assert result == "web_search"

    @patch("mycoswarm.solo.httpx.Client")
    def test_classifies_answer(self, mock_client_cls):
        mock_client = MagicMock()
        mock_client_cls.return_value.__enter__ = MagicMock(return_value=mock_client)
        mock_client_cls.return_value.__exit__ = MagicMock(return_value=False)

        mock_resp = MagicMock()
        mock_resp.json.return_value = {"message": {"content": "answer"}}
        mock_resp.raise_for_status = MagicMock()
        mock_client.post.return_value = mock_resp

        result = classify_query("What is the capital of France?", "gemma3:27b")
        assert result == "answer"

    @patch("mycoswarm.solo.httpx.Client")
    def test_classifies_rag(self, mock_client_cls):
        mock_client = MagicMock()
        mock_client_cls.return_value.__enter__ = MagicMock(return_value=mock_client)
        mock_client_cls.return_value.__exit__ = MagicMock(return_value=False)

        mock_resp = MagicMock()
        mock_resp.json.return_value = {"message": {"content": "rag"}}
        mock_resp.raise_for_status = MagicMock()
        mock_client.post.return_value = mock_resp

        result = classify_query("What does my notes file say about the project?", "gemma3:27b")
        assert result == "rag"

    @patch("mycoswarm.solo.httpx.Client")
    def test_classifies_web_and_rag(self, mock_client_cls):
        mock_client = MagicMock()
        mock_client_cls.return_value.__enter__ = MagicMock(return_value=mock_client)
        mock_client_cls.return_value.__exit__ = MagicMock(return_value=False)

        mock_resp = MagicMock()
        mock_resp.json.return_value = {"message": {"content": "web_and_rag"}}
        mock_resp.raise_for_status = MagicMock()
        mock_client.post.return_value = mock_resp

        result = classify_query("Compare my notes with latest research", "gemma3:27b")
        assert result == "web_and_rag"

    @patch("mycoswarm.solo.httpx.Client")
    def test_extracts_category_from_verbose_response(self, mock_client_cls):
        mock_client = MagicMock()
        mock_client_cls.return_value.__enter__ = MagicMock(return_value=mock_client)
        mock_client_cls.return_value.__exit__ = MagicMock(return_value=False)

        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "message": {"content": "I would classify this as web_search because it needs current data."}
        }
        mock_resp.raise_for_status = MagicMock()
        mock_client.post.return_value = mock_resp

        result = classify_query("What is bitcoin price right now?", "gemma3:27b")
        assert result == "web_search"

    @patch("mycoswarm.solo.httpx.Client")
    def test_fallback_on_connection_error(self, mock_client_cls):
        import httpx
        mock_client = MagicMock()
        mock_client_cls.return_value.__enter__ = MagicMock(return_value=mock_client)
        mock_client_cls.return_value.__exit__ = MagicMock(return_value=False)
        mock_client.post.side_effect = httpx.ConnectError("refused")

        result = classify_query("anything", "model")
        assert result == "answer"

    @patch("mycoswarm.solo.httpx.Client")
    def test_fallback_on_garbage_response(self, mock_client_cls):
        mock_client = MagicMock()
        mock_client_cls.return_value.__enter__ = MagicMock(return_value=mock_client)
        mock_client_cls.return_value.__exit__ = MagicMock(return_value=False)

        mock_resp = MagicMock()
        mock_resp.json.return_value = {"message": {"content": "I'm not sure what you mean"}}
        mock_resp.raise_for_status = MagicMock()
        mock_client.post.return_value = mock_resp

        result = classify_query("test", "model")
        assert result == "answer"


class TestWebSearchSolo:
    @patch("ddgs.DDGS")
    def test_returns_results(self, mock_ddgs_cls):
        mock_ddgs = MagicMock()
        mock_ddgs_cls.return_value = mock_ddgs
        mock_ddgs.text.return_value = [
            {"title": "Result 1", "href": "https://example.com/1", "body": "Snippet 1"},
            {"title": "Result 2", "href": "https://example.com/2", "body": "Snippet 2"},
        ]

        results = web_search_solo("test query", max_results=2)
        assert len(results) == 2
        assert results[0]["title"] == "Result 1"
        assert results[0]["url"] == "https://example.com/1"
        assert results[0]["snippet"] == "Snippet 1"

    @patch("ddgs.DDGS")
    def test_returns_empty_on_error(self, mock_ddgs_cls):
        mock_ddgs_cls.side_effect = Exception("network error")

        results = web_search_solo("test query")
        assert results == []
