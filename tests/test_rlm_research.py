"""Tests for the RLM research agent — decomposition + execution + synthesis."""

import json
from unittest.mock import MagicMock, patch

import pytest

from mycoswarm.agents.rlm_research import (
    RLMResearchAgent,
    _MAX_QUERIES_PER_SUBTOPIC,
    _MAX_SUBTOPICS,
    _execute_search,
    _extract_urls,
    decompose_topic,
)


def _mock_ollama_response(content: str):
    """Create a mock httpx response with the given content."""
    resp = MagicMock()
    resp.status_code = 200
    resp.raise_for_status = MagicMock()
    resp.json.return_value = {"message": {"content": content}}
    return resp


VALID_JSON = """[
    {"subtopic": "pricing", "queries": ["RTX 5090 price", "RTX 5090 MSRP"], "depth_hint": 3},
    {"subtopic": "specs", "queries": ["RTX 5090 specifications"], "depth_hint": 1},
    {"subtopic": "benchmarks", "queries": ["RTX 5090 benchmark results"], "depth_hint": 2}
]"""

VALID_PYTHON_LIST = """[
    {'subtopic': 'pricing', 'queries': ['RTX 5090 price'], 'depth_hint': 3},
    {'subtopic': 'specs', 'queries': ['RTX 5090 specs'], 'depth_hint': 1}
]"""

MALFORMED_JSON = """Here is my analysis:
{"subtopic": "pricing", not valid json at all"""

MARKDOWN_WRAPPED = """```json
[
    {"subtopic": "pricing", "queries": ["RTX 5090 price"], "depth_hint": 3}
]
```"""

SAMPLE_SEARCH_RESULTS = [
    {"title": "RTX 5090 Review", "url": "https://example.com/review", "snippet": "Great GPU"},
    {"title": "RTX 5090 Price", "url": "https://example.com/price", "snippet": "$1999 MSRP"},
    {"title": "RTX 5090 Bench", "url": "https://example.com/bench", "snippet": "Fast"},
]


# ── decompose_topic tests ────────────────────────────────────────────────


class TestDecomposeTopic:
    @patch("mycoswarm.agents.rlm_research.httpx.post")
    def test_valid_json(self, mock_post):
        mock_post.return_value = _mock_ollama_response(VALID_JSON)
        result = decompose_topic("RTX 5090 review", "http://localhost:11434")
        assert len(result) == 3
        assert result[0]["subtopic"] == "pricing"
        assert result[0]["depth_hint"] == 3
        assert "RTX 5090 price" in result[0]["queries"]

    @patch("mycoswarm.agents.rlm_research.httpx.post")
    def test_malformed_json_returns_empty(self, mock_post):
        mock_post.return_value = _mock_ollama_response(MALFORMED_JSON)
        result = decompose_topic("RTX 5090 review", "http://localhost:11434")
        assert result == []

    @patch("mycoswarm.agents.rlm_research.httpx.post")
    def test_python_list_fallback(self, mock_post):
        mock_post.return_value = _mock_ollama_response(VALID_PYTHON_LIST)
        result = decompose_topic("RTX 5090 review", "http://localhost:11434")
        assert len(result) == 2
        assert result[0]["subtopic"] == "pricing"

    @patch("mycoswarm.agents.rlm_research.httpx.post")
    def test_markdown_fences_stripped(self, mock_post):
        mock_post.return_value = _mock_ollama_response(MARKDOWN_WRAPPED)
        result = decompose_topic("RTX 5090 review", "http://localhost:11434")
        assert len(result) == 1
        assert result[0]["subtopic"] == "pricing"

    @patch("mycoswarm.agents.rlm_research.httpx.post")
    def test_cap_subtopics(self, mock_post):
        many = [
            {"subtopic": f"topic_{i}", "queries": [f"query_{i}"], "depth_hint": 1}
            for i in range(12)
        ]
        mock_post.return_value = _mock_ollama_response(json.dumps(many))
        result = decompose_topic("big topic", "http://localhost:11434")
        assert len(result) == _MAX_SUBTOPICS

    @patch("mycoswarm.agents.rlm_research.httpx.post")
    def test_cap_queries_per_subtopic(self, mock_post):
        data = [
            {
                "subtopic": "specs",
                "queries": ["q1", "q2", "q3", "q4", "q5"],
                "depth_hint": 1,
            }
        ]
        mock_post.return_value = _mock_ollama_response(json.dumps(data))
        result = decompose_topic("test", "http://localhost:11434")
        assert len(result[0]["queries"]) == _MAX_QUERIES_PER_SUBTOPIC

    @patch("mycoswarm.agents.rlm_research.httpx.post")
    def test_dedup_queries_across_subtopics(self, mock_post):
        data = [
            {"subtopic": "a", "queries": ["shared query", "unique_a"], "depth_hint": 1},
            {"subtopic": "b", "queries": ["shared query", "unique_b"], "depth_hint": 2},
        ]
        mock_post.return_value = _mock_ollama_response(json.dumps(data))
        result = decompose_topic("test", "http://localhost:11434")
        all_queries = [q for s in result for q in s["queries"]]
        assert len(all_queries) == len(set(all_queries))
        assert "shared query" in result[0]["queries"]
        assert "shared query" not in result[1]["queries"]

    @patch("mycoswarm.agents.rlm_research.httpx.post")
    def test_http_failure_returns_empty(self, mock_post):
        mock_post.side_effect = Exception("connection refused")
        result = decompose_topic("test", "http://localhost:11434")
        assert result == []

    @patch("mycoswarm.agents.rlm_research.httpx.post")
    def test_missing_fields_skipped(self, mock_post):
        data = [
            {"subtopic": "valid", "queries": ["q1"], "depth_hint": 1},
            {"queries": ["q2"]},  # missing subtopic
            {"subtopic": "no_queries"},  # missing queries
            "not a dict",
        ]
        mock_post.return_value = _mock_ollama_response(json.dumps(data))
        result = decompose_topic("test", "http://localhost:11434")
        assert len(result) == 1
        assert result[0]["subtopic"] == "valid"

    @patch("mycoswarm.agents.rlm_research.httpx.post")
    def test_depth_hint_defaults_and_clamps(self, mock_post):
        data = [
            {"subtopic": "no_hint", "queries": ["q1"]},
            {"subtopic": "too_high", "queries": ["q2"], "depth_hint": 10},
            {"subtopic": "too_low", "queries": ["q3"], "depth_hint": -1},
            {"subtopic": "not_int", "queries": ["q4"], "depth_hint": "high"},
        ]
        mock_post.return_value = _mock_ollama_response(json.dumps(data))
        result = decompose_topic("test", "http://localhost:11434")
        assert result[0]["depth_hint"] == 2  # default
        assert result[1]["depth_hint"] == 3  # clamped
        assert result[2]["depth_hint"] == 1  # clamped
        assert result[3]["depth_hint"] == 2  # replaced non-int


# ── Helper function tests ────────────────────────────────────────────────


class TestHelpers:
    def test_extract_urls_basic(self):
        results = [
            {"title": "A", "url": "https://a.com", "snippet": "..."},
            {"title": "B", "url": "https://b.com", "snippet": "..."},
            {"title": "C", "url": "https://c.com", "snippet": "..."},
        ]
        urls = _extract_urls(results, max_urls=2)
        assert urls == ["https://a.com", "https://b.com"]

    def test_extract_urls_deduplicates(self):
        results = [
            {"title": "A", "url": "https://a.com", "snippet": "..."},
            {"title": "A2", "url": "https://a.com", "snippet": "..."},
            {"title": "B", "url": "https://b.com", "snippet": "..."},
        ]
        urls = _extract_urls(results, max_urls=3)
        assert urls == ["https://a.com", "https://b.com"]

    def test_extract_urls_skips_non_http(self):
        results = [
            {"title": "A", "url": "", "snippet": "..."},
            {"title": "B", "url": "ftp://b.com", "snippet": "..."},
            {"title": "C", "url": "https://c.com", "snippet": "..."},
        ]
        urls = _extract_urls(results, max_urls=2)
        assert urls == ["https://c.com"]


# ── _research_subtopic tests ─────────────────────────────────────────────


class TestResearchSubtopic:
    @patch("mycoswarm.agents.rlm_research._execute_fetch")
    @patch("mycoswarm.agents.rlm_research._execute_search")
    def test_returns_findings_dict(self, mock_search, mock_fetch):
        mock_search.return_value = SAMPLE_SEARCH_RESULTS
        mock_fetch.return_value = "[FULL PAGE: https://example.com/review]\nDetailed GPU review content here"

        agent = RLMResearchAgent(
            ollama_url="http://localhost:11434", model="qwen3.5:9b"
        )
        subtopic = {
            "subtopic": "pricing",
            "queries": ["RTX 5090 price"],
            "depth_hint": 3,
        }
        result = agent._research_subtopic(subtopic)

        assert result["subtopic"] == "pricing"
        assert result["depth_hint"] == 3
        assert result["queries_run"] == ["RTX 5090 price"]
        assert result["searches"] == 1
        assert result["pages_fetched"] == 2  # top 2 URLs fetched
        assert len(result["sources"]) == 2
        assert "raw_content" in result
        assert len(result["raw_content"]) > 0

    @patch("mycoswarm.agents.rlm_research._execute_fetch")
    @patch("mycoswarm.agents.rlm_research._execute_search")
    def test_handles_empty_search(self, mock_search, mock_fetch):
        mock_search.return_value = []
        mock_fetch.return_value = ""

        agent = RLMResearchAgent(
            ollama_url="http://localhost:11434", model="qwen3.5:9b"
        )
        subtopic = {
            "subtopic": "pricing",
            "queries": ["q1"],
            "depth_hint": 3,
        }
        result = agent._research_subtopic(subtopic)

        assert result["subtopic"] == "pricing"
        assert result["searches"] == 1
        assert result["pages_fetched"] == 0

    @patch("mycoswarm.agents.rlm_research._execute_fetch")
    @patch("mycoswarm.agents.rlm_research._execute_search")
    def test_multiple_queries(self, mock_search, mock_fetch):
        mock_search.return_value = SAMPLE_SEARCH_RESULTS
        mock_fetch.return_value = "page content"

        agent = RLMResearchAgent(
            ollama_url="http://localhost:11434", model="qwen3.5:9b"
        )
        subtopic = {
            "subtopic": "benchmarks",
            "queries": ["q1", "q2", "q3"],
            "depth_hint": 2,
        }
        result = agent._research_subtopic(subtopic)

        assert result["searches"] == 3
        # 2 URLs fetched per query * 3 queries = 6
        assert result["pages_fetched"] == 6


# ── Full run() tests ─────────────────────────────────────────────────────


class TestRLMRun:
    @patch("mycoswarm.agents.rlm_research.RLMResearchAgent._synthesize")
    @patch("mycoswarm.agents.rlm_research.RLMResearchAgent._research_subtopic")
    @patch("mycoswarm.agents.rlm_research.decompose_topic")
    def test_run_full_pipeline(self, mock_decompose, mock_research, mock_synth):
        mock_decompose.return_value = [
            {"subtopic": "pricing", "queries": ["price q"], "depth_hint": 3},
            {"subtopic": "specs", "queries": ["spec q"], "depth_hint": 1},
        ]
        mock_research.return_value = {
            "subtopic": "test",
            "depth_hint": 2,
            "queries_run": ["q1"],
            "searches": 1,
            "pages_fetched": 2,
            "sources": ["https://example.com"],
            "raw_content": "Some research findings " * 50,
        }
        mock_synth.return_value = (
            "## Key Facts & Data Points\n- Fact 1\n\n"
            "## Price Data\n- $299\n\n"
            "## Benchmark Data\n- 100 tok/s\n\n"
            "## Key Specs\n- 16GB VRAM\n\n"
            "## Expert Opinions & Analysis\n- Good GPU\n\n"
            "## Gaps\n- Missing power data\n\n"
            "## Suggested Angle\n- Budget focus"
        )

        agent = RLMResearchAgent(
            ollama_url="http://localhost:11434",
            model="qwen3.5:9b",
            root_model="gemma3:27b",
            synthesis_url="http://192.168.50.10:11434",
        )
        result = agent.run(topic="Best budget GPU")

        assert "Key Facts" in result
        assert "Price Data" in result
        assert result != ""
        # Verify decompose was called
        mock_decompose.assert_called_once()
        # Verify research was called for each subtopic
        assert mock_research.call_count == 2
        # Verify synthesis was called
        mock_synth.assert_called_once()

    @patch("mycoswarm.agents.rlm_research.decompose_topic")
    def test_run_returns_empty_on_decompose_failure(self, mock_decompose):
        mock_decompose.return_value = []
        agent = RLMResearchAgent(
            ollama_url="http://localhost:11434", model="qwen3.5:9b"
        )
        result = agent.run(topic="RTX 5090")
        assert result == ""

    @patch("mycoswarm.agents.rlm_research.RLMResearchAgent._synthesize")
    @patch("mycoswarm.agents.rlm_research.RLMResearchAgent._research_subtopic")
    @patch("mycoswarm.agents.rlm_research.decompose_topic")
    def test_run_returns_empty_on_synthesis_failure(
        self, mock_decompose, mock_research, mock_synth
    ):
        mock_decompose.return_value = [
            {"subtopic": "test", "queries": ["q1"], "depth_hint": 1},
        ]
        mock_research.return_value = {
            "subtopic": "test",
            "depth_hint": 1,
            "queries_run": ["q1"],
            "searches": 1,
            "pages_fetched": 1,
            "sources": ["https://example.com"],
            "raw_content": "content",
        }
        mock_synth.return_value = ""

        agent = RLMResearchAgent(
            ollama_url="http://localhost:11434", model="qwen3.5:9b"
        )
        result = agent.run(topic="test")
        assert result == ""

    @patch("mycoswarm.agents.rlm_research.RLMResearchAgent._synthesize")
    @patch("mycoswarm.agents.rlm_research.RLMResearchAgent._research_subtopic")
    @patch("mycoswarm.agents.rlm_research.decompose_topic")
    def test_debug_log_tracks_subtopics(
        self, mock_decompose, mock_research, mock_synth
    ):
        mock_decompose.return_value = [
            {"subtopic": "a", "queries": ["q1"], "depth_hint": 1},
            {"subtopic": "b", "queries": ["q2"], "depth_hint": 2},
        ]
        mock_research.return_value = {
            "subtopic": "test",
            "depth_hint": 1,
            "queries_run": ["q1"],
            "searches": 1,
            "pages_fetched": 1,
            "sources": ["https://example.com"],
            "raw_content": "content " * 100,
        }
        mock_synth.return_value = "bundle output " * 50

        agent = RLMResearchAgent(
            ollama_url="http://localhost:11434", model="qwen3.5:9b"
        )
        agent.run(topic="test")

        log_text = "\n".join(agent.debug_log)
        assert "decomposed" in log_text
        assert "subtopic 1/2" in log_text
        assert "subtopic 2/2" in log_text
        assert "compiling final bundle" in log_text
        assert "done:" in log_text

    @patch("mycoswarm.agents.rlm_research.RLMResearchAgent._synthesize")
    @patch("mycoswarm.agents.rlm_research.RLMResearchAgent._research_subtopic")
    @patch("mycoswarm.agents.rlm_research.decompose_topic")
    def test_synthesis_uses_root_model(
        self, mock_decompose, mock_research, mock_synth
    ):
        mock_decompose.return_value = [
            {"subtopic": "a", "queries": ["q1"], "depth_hint": 1},
        ]
        mock_research.return_value = {
            "subtopic": "a",
            "depth_hint": 1,
            "queries_run": ["q1"],
            "searches": 1,
            "pages_fetched": 1,
            "sources": [],
            "raw_content": "data " * 100,
        }
        mock_synth.return_value = "bundle " * 100

        agent = RLMResearchAgent(
            ollama_url="http://localhost:11434",
            model="qwen3.5:9b",
            root_model="gemma3:27b",
            synthesis_url="http://miu:11434",
        )
        agent.run(topic="test")

        log_text = "\n".join(agent.debug_log)
        assert "gemma3:27b" in log_text
