"""mycoSwarm task worker.

Pulls tasks from the node's queue and executes them.
Each task type has a handler function. Currently supports:

  - inference: Send prompt to Ollama and return response
  - web_search: Search the web via DuckDuckGo (no API key)
  - web_fetch: Fetch a URL and extract readable text
  - ping: Simple health check task (for testing)

Future handlers:
  - file_process: Read/transform files
  - embedding: Generate embeddings via Ollama
"""

import asyncio
import json
import logging
import re
import time
from typing import Callable, Awaitable

import httpx

from mycoswarm.api import TaskQueue, TaskRequest, TaskResult, TaskStatus

logger = logging.getLogger(__name__)

# Ollama API
OLLAMA_BASE = "http://localhost:11434"
OLLAMA_TIMEOUT = 300.0  # 5 min max for inference


# --- Task Handlers ---


def _push_safe(queue: asyncio.Queue, event: dict | None) -> None:
    """Non-blocking push to stream queue; silently drops if consumer gone."""
    try:
        queue.put_nowait(event)
    except asyncio.QueueFull:
        pass


def _build_ollama_request(
    payload: dict,
) -> tuple[str, dict, bool]:
    """Build Ollama endpoint + payload from task payload.

    Returns (endpoint, ollama_payload, is_chat).
    """
    model = payload["model"]
    messages = payload.get("messages")
    prompt = payload.get("prompt")

    if messages:
        endpoint = f"{OLLAMA_BASE}/api/chat"
        ollama_payload: dict = {
            "model": model,
            "messages": messages,
            "options": {
                "temperature": payload.get("temperature", 0.7),
                "num_predict": payload.get("max_tokens", 2048),
            },
        }
        is_chat = True
    else:
        endpoint = f"{OLLAMA_BASE}/api/generate"
        ollama_payload = {
            "model": model,
            "prompt": prompt,
            "options": {
                "temperature": payload.get("temperature", 0.7),
                "num_predict": payload.get("max_tokens", 2048),
            },
        }
        is_chat = False

    if payload.get("system"):
        ollama_payload["system"] = payload["system"]

    return endpoint, ollama_payload, is_chat


def _metrics_from_ollama(data: dict, model: str) -> dict:
    """Extract standard metrics from Ollama response JSON."""
    return {
        "model": model,
        "total_duration_ms": data.get("total_duration", 0) / 1_000_000,
        "eval_count": data.get("eval_count", 0),
        "eval_duration_ms": data.get("eval_duration", 0) / 1_000_000,
        "tokens_per_second": (
            data.get("eval_count", 0)
            / (data.get("eval_duration", 1) / 1_000_000_000)
            if data.get("eval_duration")
            else 0
        ),
    }


async def _inference_batch(
    task: TaskRequest, endpoint: str, ollama_payload: dict, is_chat: bool
) -> TaskResult:
    """Non-streaming (batch) Ollama inference."""
    model = task.payload["model"]
    ollama_payload["stream"] = False
    start = time.time()

    try:
        async with httpx.AsyncClient(timeout=OLLAMA_TIMEOUT) as client:
            response = await client.post(endpoint, json=ollama_payload)
            response.raise_for_status()
            data = response.json()

        duration = time.time() - start

        if is_chat:
            response_text = data.get("message", {}).get("content", "")
        else:
            response_text = data.get("response", "")

        metrics = _metrics_from_ollama(data, model)
        metrics["response"] = response_text

        return TaskResult(
            task_id=task.task_id,
            status=TaskStatus.COMPLETED,
            result=metrics,
            duration_seconds=round(duration, 2),
        )

    except httpx.TimeoutException:
        return TaskResult(
            task_id=task.task_id,
            status=TaskStatus.FAILED,
            error=f"Ollama inference timed out after {OLLAMA_TIMEOUT}s",
            duration_seconds=round(time.time() - start, 2),
        )
    except httpx.ConnectError:
        return TaskResult(
            task_id=task.task_id,
            status=TaskStatus.FAILED,
            error="Cannot connect to Ollama — is it running?",
        )
    except httpx.HTTPError as e:
        return TaskResult(
            task_id=task.task_id,
            status=TaskStatus.FAILED,
            error=f"Ollama error: {e}",
            duration_seconds=round(time.time() - start, 2),
        )


async def _inference_stream(
    task: TaskRequest,
    endpoint: str,
    ollama_payload: dict,
    is_chat: bool,
    stream_queue: asyncio.Queue,
) -> TaskResult:
    """Streaming Ollama inference — pushes tokens to queue as they arrive."""
    model = task.payload["model"]
    ollama_payload["stream"] = True
    start = time.time()
    collected_tokens: list[str] = []
    final_data: dict = {}

    try:
        async with httpx.AsyncClient(timeout=OLLAMA_TIMEOUT) as client:
            async with client.stream(
                "POST", endpoint, json=ollama_payload
            ) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if not line:
                        continue
                    try:
                        chunk = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    # Extract token from chunk
                    if is_chat:
                        token = chunk.get("message", {}).get("content", "")
                    else:
                        token = chunk.get("response", "")

                    if token:
                        collected_tokens.append(token)
                        _push_safe(
                            stream_queue,
                            {"token": token, "done": False},
                        )

                    if chunk.get("done"):
                        final_data = chunk
                        break

        duration = time.time() - start
        full_response = "".join(collected_tokens)
        metrics = _metrics_from_ollama(final_data, model)
        metrics["response"] = full_response

        # Push final done event with metrics
        _push_safe(
            stream_queue,
            {
                "done": True,
                "model": model,
                "tokens_per_second": metrics["tokens_per_second"],
                "duration_seconds": round(duration, 2),
            },
        )
        # Sentinel — tells SSE generator to stop
        _push_safe(stream_queue, None)

        return TaskResult(
            task_id=task.task_id,
            status=TaskStatus.COMPLETED,
            result=metrics,
            duration_seconds=round(duration, 2),
        )

    except httpx.TimeoutException:
        _push_safe(
            stream_queue,
            {"error": f"Ollama timed out after {OLLAMA_TIMEOUT}s", "done": True},
        )
        _push_safe(stream_queue, None)
        return TaskResult(
            task_id=task.task_id,
            status=TaskStatus.FAILED,
            error=f"Ollama inference timed out after {OLLAMA_TIMEOUT}s",
            duration_seconds=round(time.time() - start, 2),
        )
    except httpx.ConnectError:
        _push_safe(
            stream_queue,
            {"error": "Cannot connect to Ollama — is it running?", "done": True},
        )
        _push_safe(stream_queue, None)
        return TaskResult(
            task_id=task.task_id,
            status=TaskStatus.FAILED,
            error="Cannot connect to Ollama — is it running?",
        )
    except httpx.HTTPError as e:
        _push_safe(
            stream_queue,
            {"error": f"Ollama error: {e}", "done": True},
        )
        _push_safe(stream_queue, None)
        return TaskResult(
            task_id=task.task_id,
            status=TaskStatus.FAILED,
            error=f"Ollama error: {e}",
            duration_seconds=round(time.time() - start, 2),
        )


async def handle_inference(
    task: TaskRequest, stream_queue: asyncio.Queue | None = None
) -> TaskResult:
    """Run inference via local Ollama.

    Supports two modes:
      - prompt mode: payload has "prompt" → uses /api/generate
      - chat mode:   payload has "messages" → uses /api/chat

    When stream_queue is provided, tokens are pushed in real-time.

    Expected payload:
        model: str          — Ollama model name (e.g. "qwen2.5:14b-instruct-q4_K_M")
        prompt: str         — The prompt text (generate mode)
        messages: list      — Chat messages [{role, content}] (chat mode)
        system: str         — Optional system prompt
        temperature: float  — Optional, default 0.7
        max_tokens: int     — Optional, default 2048
    """
    payload = task.payload
    model = payload.get("model")
    messages = payload.get("messages")
    prompt = payload.get("prompt")

    if not model or (not prompt and not messages):
        if stream_queue:
            _push_safe(
                stream_queue,
                {"error": "Missing 'model' and ('prompt' or 'messages')", "done": True},
            )
            _push_safe(stream_queue, None)
        return TaskResult(
            task_id=task.task_id,
            status=TaskStatus.FAILED,
            error="Missing required fields: 'model' and ('prompt' or 'messages')",
        )

    endpoint, ollama_payload, is_chat = _build_ollama_request(payload)

    if stream_queue is not None:
        return await _inference_stream(
            task, endpoint, ollama_payload, is_chat, stream_queue
        )
    else:
        return await _inference_batch(task, endpoint, ollama_payload, is_chat)


async def handle_ping(task: TaskRequest) -> TaskResult:
    """Simple test handler — echoes back the payload."""
    return TaskResult(
        task_id=task.task_id,
        status=TaskStatus.COMPLETED,
        result={"echo": task.payload, "message": "pong"},
        duration_seconds=0.0,
    )


async def handle_web_search(task: TaskRequest) -> TaskResult:
    """Search the web via DuckDuckGo. No API key needed.

    Expected payload:
        query: str         — search terms
        max_results: int   — number of results (default 5, max 20)
    """
    from duckduckgo_search import DDGS

    payload = task.payload
    query = payload.get("query")
    if not query:
        return TaskResult(
            task_id=task.task_id,
            status=TaskStatus.FAILED,
            error="Missing required field: 'query'",
        )

    max_results = min(payload.get("max_results", 5), 20)
    start = time.time()

    def _search():
        with DDGS() as ddgs:
            return list(ddgs.text(query, max_results=max_results))

    try:
        raw = await asyncio.to_thread(_search)

        results = [
            {
                "title": r.get("title", ""),
                "url": r.get("href", ""),
                "snippet": r.get("body", ""),
            }
            for r in raw
        ]

        return TaskResult(
            task_id=task.task_id,
            status=TaskStatus.COMPLETED,
            result={"query": query, "results": results, "count": len(results)},
            duration_seconds=round(time.time() - start, 2),
        )
    except Exception as e:
        return TaskResult(
            task_id=task.task_id,
            status=TaskStatus.FAILED,
            error=f"Web search failed: {e}",
            duration_seconds=round(time.time() - start, 2),
        )


def _strip_html(html: str) -> str:
    """Strip HTML tags and collapse whitespace into readable text."""
    # Remove script/style blocks
    text = re.sub(r"<(script|style)[^>]*>.*?</\1>", "", html, flags=re.DOTALL | re.IGNORECASE)
    # Remove tags
    text = re.sub(r"<[^>]+>", " ", text)
    # Decode common entities
    for entity, char in [("&amp;", "&"), ("&lt;", "<"), ("&gt;", ">"),
                         ("&quot;", '"'), ("&#39;", "'"), ("&nbsp;", " ")]:
        text = text.replace(entity, char)
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


async def handle_web_fetch(task: TaskRequest) -> TaskResult:
    """Fetch a URL and return its content.

    Expected payload:
        url: str             — the URL to fetch
        extract_text: bool   — strip HTML tags (default true)
        max_length: int      — max chars to return (default 50000)
    """
    payload = task.payload
    url = payload.get("url")
    if not url:
        return TaskResult(
            task_id=task.task_id,
            status=TaskStatus.FAILED,
            error="Missing required field: 'url'",
        )

    extract_text = payload.get("extract_text", True)
    max_length = payload.get("max_length", 50_000)
    start = time.time()

    try:
        async with httpx.AsyncClient(
            timeout=30.0, follow_redirects=True
        ) as client:
            resp = await client.get(url, headers={
                "User-Agent": "mycoSwarm/0.1 (web_fetch task handler)",
            })
            resp.raise_for_status()

        content = resp.text
        if extract_text:
            content = _strip_html(content)
        if len(content) > max_length:
            content = content[:max_length] + "... [truncated]"

        return TaskResult(
            task_id=task.task_id,
            status=TaskStatus.COMPLETED,
            result={
                "url": str(resp.url),
                "status_code": resp.status_code,
                "text": content,
                "length": len(content),
            },
            duration_seconds=round(time.time() - start, 2),
        )
    except httpx.TimeoutException:
        return TaskResult(
            task_id=task.task_id,
            status=TaskStatus.FAILED,
            error=f"Fetch timed out: {url}",
            duration_seconds=round(time.time() - start, 2),
        )
    except httpx.HTTPError as e:
        return TaskResult(
            task_id=task.task_id,
            status=TaskStatus.FAILED,
            error=f"Fetch failed: {e}",
            duration_seconds=round(time.time() - start, 2),
        )


# --- Handler Registry ---

HANDLERS: dict[str, Callable[[TaskRequest], Awaitable[TaskResult]]] = {
    "inference": handle_inference,
    "web_search": handle_web_search,
    "web_fetch": handle_web_fetch,
    "ping": handle_ping,
}


# --- Worker Loop ---


class TaskWorker:
    """Continuously pulls tasks from the queue and executes them.

    Runs as an asyncio task alongside the daemon.
    """

    def __init__(
        self,
        task_queue: TaskQueue,
        node_id: str,
        max_concurrent: int = 2,
    ):
        self.task_queue = task_queue
        self.node_id = node_id
        self.max_concurrent = max_concurrent
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._running = False
        self._tasks_completed = 0
        self._tasks_failed = 0

    async def _execute_task(self, task: TaskRequest):
        """Execute a single task with concurrency control."""
        async with self._semaphore:
            handler = HANDLERS.get(task.task_type)
            stream_queue = self.task_queue.get_stream(task.task_id)

            if not handler:
                if stream_queue:
                    _push_safe(stream_queue, {"error": f"Unknown task type: {task.task_type}", "done": True})
                    _push_safe(stream_queue, None)
                result = TaskResult(
                    task_id=task.task_id,
                    status=TaskStatus.FAILED,
                    error=f"Unknown task type: {task.task_type}",
                    node_id=self.node_id,
                )
                await self.task_queue.store_result(result)
                self._tasks_failed += 1
                logger.warning(f"❓ Unknown task type: {task.task_type}")
                return

            logger.info(f"⚙️  Executing: {task.task_id} ({task.task_type})")
            await self.task_queue.mark_active(task)
            start = time.time()

            try:
                if stream_queue is not None and task.task_type == "inference":
                    result = await asyncio.wait_for(
                        handler(task, stream_queue=stream_queue),
                        timeout=task.timeout_seconds,
                    )
                else:
                    result = await asyncio.wait_for(
                        handler(task), timeout=task.timeout_seconds
                    )
                result.node_id = self.node_id
                await self.task_queue.store_result(result)

                if result.status == TaskStatus.COMPLETED:
                    self._tasks_completed += 1
                    logger.info(
                        f"✅ Completed: {task.task_id} "
                        f"({result.duration_seconds:.1f}s)"
                    )
                else:
                    self._tasks_failed += 1
                    logger.error(
                        f"❌ Failed: {task.task_id} — {result.error}"
                    )

            except asyncio.TimeoutError:
                elapsed = round(time.time() - start, 2)
                if stream_queue:
                    _push_safe(stream_queue, {"error": f"Task timed out after {task.timeout_seconds}s", "done": True})
                    _push_safe(stream_queue, None)
                result = TaskResult(
                    task_id=task.task_id,
                    status=TaskStatus.FAILED,
                    error=f"Task timed out after {task.timeout_seconds}s",
                    duration_seconds=elapsed,
                    node_id=self.node_id,
                )
                await self.task_queue.store_result(result)
                self._tasks_failed += 1
                logger.error(
                    f"⏱️ Timeout: {task.task_id} after {elapsed:.1f}s"
                )

            except Exception as e:
                if stream_queue:
                    _push_safe(stream_queue, {"error": f"Unhandled error: {e}", "done": True})
                    _push_safe(stream_queue, None)
                result = TaskResult(
                    task_id=task.task_id,
                    status=TaskStatus.FAILED,
                    error=f"Unhandled error: {e}",
                    node_id=self.node_id,
                )
                await self.task_queue.store_result(result)
                self._tasks_failed += 1
                logger.exception(f"💥 Unhandled error in task {task.task_id}")

    async def run(self):
        """Main worker loop — polls the queue and dispatches tasks."""
        self._running = True
        logger.info(
            f"👷 Worker started (max_concurrent={self.max_concurrent})"
        )

        while self._running:
            task = await self.task_queue.get_next()

            if task:
                # Fire and forget — semaphore controls concurrency
                asyncio.create_task(self._execute_task(task))
            else:
                # No tasks — sleep briefly before checking again
                await asyncio.sleep(0.1)

    def stop(self):
        """Signal the worker to stop."""
        self._running = False
        logger.info(
            f"👷 Worker stopped. "
            f"Completed: {self._tasks_completed}, "
            f"Failed: {self._tasks_failed}"
        )

    @property
    def stats(self) -> dict:
        return {
            "running": self._running,
            "max_concurrent": self.max_concurrent,
            "tasks_completed": self._tasks_completed,
            "tasks_failed": self._tasks_failed,
        }
