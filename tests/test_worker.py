"""Tests for mycoswarm.worker — task handlers."""

import asyncio
from unittest.mock import patch, AsyncMock, MagicMock

import pytest

from mycoswarm.api import TaskRequest, TaskResult, TaskStatus
from mycoswarm.worker import (
    handle_ping,
    handle_web_search,
    handle_web_fetch,
    handle_code_run,
    handle_translate,
    _build_ollama_request,
)


@pytest.mark.asyncio
async def test_handle_ping_returns_pong(make_task):
    """handle_ping returns COMPLETED with 'pong' in result."""
    task = make_task("ping", {"echo": "hello"})
    result = await handle_ping(task)

    assert result.status == TaskStatus.COMPLETED
    assert result.result["message"] == "pong"
    assert result.result["echo"] == {"echo": "hello"}


@pytest.mark.asyncio
async def test_handle_web_search_returns_results(make_task):
    """Mock DDGS, verify results list."""
    task = make_task("web_search", {"query": "python testing", "max_results": 3})

    fake_results = [
        {"title": "Result 1", "href": "https://example.com/1", "body": "Snippet 1"},
        {"title": "Result 2", "href": "https://example.com/2", "body": "Snippet 2"},
    ]

    mock_ddgs_instance = MagicMock()
    mock_ddgs_instance.text.return_value = fake_results

    with patch("ddgs.DDGS", return_value=mock_ddgs_instance):
        result = await handle_web_search(task)

    assert result.status == TaskStatus.COMPLETED
    assert result.result["query"] == "python testing"
    assert result.result["count"] == 2
    assert result.result["results"][0]["title"] == "Result 1"
    assert result.result["results"][0]["url"] == "https://example.com/1"


@pytest.mark.asyncio
async def test_handle_web_fetch_returns_text(make_task):
    """Mock httpx.AsyncClient, verify url/text/status_code in result."""
    task = make_task("web_fetch", {"url": "https://example.com"})

    mock_response = MagicMock()
    mock_response.text = "<html><body><p>Hello World</p></body></html>"
    mock_response.status_code = 200
    mock_response.url = "https://example.com"
    mock_response.raise_for_status = MagicMock()

    mock_client = AsyncMock()
    mock_client.get = AsyncMock(return_value=mock_response)
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)

    with patch("mycoswarm.worker.httpx.AsyncClient", return_value=mock_client):
        result = await handle_web_fetch(task)

    assert result.status == TaskStatus.COMPLETED
    assert result.result["status_code"] == 200
    assert "Hello World" in result.result["text"]
    assert str(result.result["url"]) == "https://example.com"


@pytest.mark.asyncio
async def test_handle_code_run_simple_python(make_task):
    """Run print('hello'), verify stdout."""
    task = make_task("code_run", {"code": "print('hello')", "timeout": 10})
    result = await handle_code_run(task)

    assert result.status == TaskStatus.COMPLETED
    assert result.result["stdout"].strip() == "hello"
    assert result.result["return_code"] == 0


@pytest.mark.asyncio
async def test_handle_code_run_timeout(make_task):
    """Run time.sleep(10) with timeout=1, verify FAILED."""
    task = make_task("code_run", {"code": "import time; time.sleep(10)", "timeout": 1})
    result = await handle_code_run(task)

    assert result.status == TaskStatus.FAILED
    assert "timed out" in result.error.lower()


@pytest.mark.asyncio
async def test_handle_translate_builds_correct_prompt(make_task):
    """Mock handle_inference, verify messages contain target_language."""
    task = make_task("translate", {
        "text": "Hello world",
        "target_language": "Spanish",
        "model": "qwen2.5:7b",
    })

    captured_task = None

    async def mock_handle_inference(t, **kwargs):
        nonlocal captured_task
        captured_task = t
        return TaskResult(
            task_id=t.task_id,
            status=TaskStatus.COMPLETED,
            result={"response": "Hola mundo", "tokens_per_second": 42.0},
        )

    with patch("mycoswarm.worker.handle_inference", side_effect=mock_handle_inference):
        result = await handle_translate(task)

    assert result.status == TaskStatus.COMPLETED
    assert captured_task is not None
    messages = captured_task.payload["messages"]
    system_content = messages[0]["content"]
    assert "Spanish" in system_content


def test_datetime_injected_into_generate_prompt():
    """_build_ollama_request with prompt payload injects datetime string."""
    payload = {"model": "test-model", "prompt": "What time is it?"}
    endpoint, ollama_payload, is_chat = _build_ollama_request(payload)

    assert endpoint.endswith("/api/generate")
    assert is_chat is False
    assert "Current date and time:" in ollama_payload["prompt"]
    assert "What time is it?" in ollama_payload["prompt"]


def test_datetime_injected_into_chat_messages():
    """_build_ollama_request with messages injects datetime into system message."""
    payload = {
        "model": "test-model",
        "messages": [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hi"},
        ],
    }
    endpoint, ollama_payload, is_chat = _build_ollama_request(payload)

    assert endpoint.endswith("/api/chat")
    assert is_chat is True
    sys_msg = ollama_payload["messages"][0]
    assert sys_msg["role"] == "system"
    assert "Current date and time:" in sys_msg["content"]
    assert "You are helpful." in sys_msg["content"]
