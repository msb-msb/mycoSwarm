"""mycoSwarm task worker.

Pulls tasks from the node's queue and executes them.
Each task type has a handler function. Supports:

  - inference: Send prompt to Ollama and return response
  - embedding: Generate embeddings via Ollama /api/embeddings
  - web_search: Search the web via DuckDuckGo (no API key)
  - web_fetch: Fetch a URL and extract readable text
  - file_read: Extract text from PDF, markdown, txt, html, csv, json
  - file_summarize: Read a file then summarize via inference
  - translate: Translate text using inference with a translation prompt
  - code_run: Run Python code in a sandboxed subprocess
  - ping: Simple health check task (for testing)
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


def _datetime_string() -> str:
    """Current date/time formatted for system prompt injection."""
    from datetime import datetime, timezone

    now = datetime.now().astimezone()
    day = now.day
    hour = now.hour % 12 or 12
    return now.strftime(f"Current date and time: %A, %B {day}, %Y at {hour}:%M %p %Z")


def _build_ollama_request(
    payload: dict,
) -> tuple[str, dict, bool]:
    """Build Ollama endpoint + payload from task payload.

    Returns (endpoint, ollama_payload, is_chat).
    Automatically injects current date/time into the system context.
    """
    model = payload["model"]
    messages = payload.get("messages")
    prompt = payload.get("prompt")
    datetime_line = _datetime_string()

    if messages:
        endpoint = f"{OLLAMA_BASE}/api/chat"
        # Inject datetime into existing system message or prepend one
        msgs = list(messages)
        if msgs and msgs[0].get("role") == "system":
            msgs[0] = {
                **msgs[0],
                "content": f"{datetime_line}\n\n{msgs[0]['content']}",
            }
        else:
            msgs.insert(0, {"role": "system", "content": datetime_line})
        ollama_payload: dict = {
            "model": model,
            "messages": msgs,
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
            "prompt": f"{datetime_line}\n\n{prompt}",
            "options": {
                "temperature": payload.get("temperature", 0.7),
                "num_predict": payload.get("max_tokens", 2048),
            },
        }
        is_chat = False

    # System prompt: prepend datetime to explicit system prompts
    existing_system = payload.get("system", "")
    if existing_system:
        ollama_payload["system"] = f"{datetime_line}\n\n{existing_system}"

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
    from ddgs import DDGS

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
    logger.info(f"🔍 Web search: {query!r} (max_results={max_results})")

    def _search():
        return DDGS().text(query, max_results=max_results)

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


# --- File Processing Helpers ---


def _extract_pdf(raw: bytes) -> str:
    """Extract text from PDF bytes using pymupdf."""
    import pymupdf

    doc = pymupdf.open(stream=raw, filetype="pdf")
    pages = []
    for page in doc:
        pages.append(page.get_text())
    doc.close()
    return "\n\n".join(pages)


def _extract_text(raw: bytes, filetype: str) -> str:
    """Extract readable text from raw file bytes based on filetype."""
    if filetype == "pdf":
        return _extract_pdf(raw)
    elif filetype in ("html", "htm"):
        return _strip_html(raw.decode("utf-8", errors="replace"))
    elif filetype == "json":
        try:
            return json.dumps(json.loads(raw), indent=2)
        except json.JSONDecodeError:
            return raw.decode("utf-8", errors="replace")
    else:
        # md, txt, csv, and anything else — just decode
        return raw.decode("utf-8", errors="replace")


async def _get_default_model() -> str | None:
    """Query Ollama for the first available model."""
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            resp = await client.get(f"{OLLAMA_BASE}/api/tags")
            resp.raise_for_status()
            models = resp.json().get("models", [])
            if models:
                return models[0]["name"]
    except (httpx.ConnectError, httpx.HTTPError):
        pass
    return None


# --- Embedding Handler ---


async def handle_embedding(task: TaskRequest) -> TaskResult:
    """Generate embeddings via Ollama /api/embeddings.

    Expected payload:
        model: str   — Ollama model name (e.g. "nomic-embed-text")
        text: str    — Text to embed
    """
    payload = task.payload
    model = payload.get("model")
    text = payload.get("text")

    if not model or not text:
        return TaskResult(
            task_id=task.task_id,
            status=TaskStatus.FAILED,
            error="Missing required fields: 'model' and 'text'",
        )

    start = time.time()

    try:
        async with httpx.AsyncClient(timeout=OLLAMA_TIMEOUT) as client:
            resp = await client.post(
                f"{OLLAMA_BASE}/api/embeddings",
                json={"model": model, "prompt": text},
            )
            resp.raise_for_status()
            data = resp.json()

        embedding = data.get("embedding", [])

        return TaskResult(
            task_id=task.task_id,
            status=TaskStatus.COMPLETED,
            result={
                "embedding": embedding,
                "model": model,
                "dimensions": len(embedding),
            },
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
            error=f"Ollama embedding error: {e}",
            duration_seconds=round(time.time() - start, 2),
        )


# --- File Processing Handlers ---


async def handle_file_read(task: TaskRequest) -> TaskResult:
    """Read text from various file formats.

    Expected payload:
        path: str       — File path on this node (optional if content given)
        content: str    — Base64-encoded file content (optional if path given)
        filetype: str   — File type hint: pdf, md, txt, html, csv, json
    """
    import base64
    from pathlib import Path

    payload = task.payload
    content_b64 = payload.get("content")
    file_path = payload.get("path")
    filetype = payload.get("filetype", "").lower()

    if not content_b64 and not file_path:
        return TaskResult(
            task_id=task.task_id,
            status=TaskStatus.FAILED,
            error="Missing required field: 'path' or 'content'",
        )

    start = time.time()

    try:
        if content_b64:
            raw = base64.b64decode(content_b64)
        else:
            p = Path(file_path)
            if not p.exists():
                return TaskResult(
                    task_id=task.task_id,
                    status=TaskStatus.FAILED,
                    error=f"File not found: {file_path}",
                )
            raw = p.read_bytes()
            if not filetype:
                filetype = p.suffix.lstrip(".").lower()

        if not filetype:
            filetype = "txt"

        text = _extract_text(raw, filetype)

        if len(text) > 500_000:
            text = text[:500_000] + "\n\n... [truncated at 500KB]"

        return TaskResult(
            task_id=task.task_id,
            status=TaskStatus.COMPLETED,
            result={
                "text": text,
                "length": len(text),
                "filetype": filetype,
            },
            duration_seconds=round(time.time() - start, 2),
        )
    except Exception as e:
        return TaskResult(
            task_id=task.task_id,
            status=TaskStatus.FAILED,
            error=f"File read failed: {e}",
            duration_seconds=round(time.time() - start, 2),
        )


async def handle_file_summarize(task: TaskRequest) -> TaskResult:
    """Read a file and summarize it via inference.

    Extracts text locally then runs Ollama inference to produce a summary.
    Routes to inference-capable nodes so both steps happen on one node.

    Expected payload:
        path: str        — File path (optional if content given)
        content: str     — Base64-encoded content (optional if path given)
        filetype: str    — File type hint
        model: str       — Ollama model (optional, auto-detected)
        max_length: int  — Max chars to send to inference (default 32000)
    """
    import base64
    from pathlib import Path

    payload = task.payload
    content_b64 = payload.get("content")
    file_path = payload.get("path")
    filetype = payload.get("filetype", "").lower()
    model = payload.get("model")
    max_length = payload.get("max_length", 32_000)

    if not content_b64 and not file_path:
        return TaskResult(
            task_id=task.task_id,
            status=TaskStatus.FAILED,
            error="Missing required field: 'path' or 'content'",
        )

    start = time.time()

    # Step 1: Extract text
    try:
        if content_b64:
            raw = base64.b64decode(content_b64)
        else:
            p = Path(file_path)
            if not p.exists():
                return TaskResult(
                    task_id=task.task_id,
                    status=TaskStatus.FAILED,
                    error=f"File not found: {file_path}",
                )
            raw = p.read_bytes()
            if not filetype:
                filetype = p.suffix.lstrip(".").lower()

        if not filetype:
            filetype = "txt"

        text = _extract_text(raw, filetype)
    except Exception as e:
        return TaskResult(
            task_id=task.task_id,
            status=TaskStatus.FAILED,
            error=f"File read failed: {e}",
            duration_seconds=round(time.time() - start, 2),
        )

    if len(text) > max_length:
        text = text[:max_length] + "\n\n[truncated]"

    # Step 2: Auto-detect model if not specified
    if not model:
        model = await _get_default_model()
    if not model:
        return TaskResult(
            task_id=task.task_id,
            status=TaskStatus.FAILED,
            error="No model specified and no Ollama models available",
        )

    # Step 3: Summarize via inference
    summary_task = TaskRequest(
        task_id=task.task_id,
        task_type="inference",
        payload={
            "model": model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a document summarizer. Read the provided document "
                        "and produce a clear, concise summary. Include key points, "
                        "main arguments, and important details. Be thorough but brief."
                    ),
                },
                {
                    "role": "user",
                    "content": f"Summarize this document:\n\n{text}",
                },
            ],
            "temperature": 0.3,
            "max_tokens": 2048,
        },
        source_node=task.source_node,
        priority=task.priority,
        timeout_seconds=task.timeout_seconds,
    )

    result = await handle_inference(summary_task)

    if result.status == TaskStatus.COMPLETED and result.result:
        result.result = {
            "summary": result.result.get("response", ""),
            "source_length": len(text),
            "filetype": filetype,
            "model": model,
            "tokens_per_second": result.result.get("tokens_per_second", 0),
        }
    result.duration_seconds = round(time.time() - start, 2)
    return result


# --- Translation Handler ---


async def handle_translate(task: TaskRequest) -> TaskResult:
    """Translate text using inference with a translation system prompt.

    Expected payload:
        text: str             — Text to translate
        target_language: str  — Target language (e.g. "Spanish", "Japanese")
        model: str            — Ollama model (optional, auto-detected)
    """
    payload = task.payload
    text = payload.get("text")
    target_language = payload.get("target_language")
    model = payload.get("model")

    if not text or not target_language:
        return TaskResult(
            task_id=task.task_id,
            status=TaskStatus.FAILED,
            error="Missing required fields: 'text' and 'target_language'",
        )

    if not model:
        model = await _get_default_model()
    if not model:
        return TaskResult(
            task_id=task.task_id,
            status=TaskStatus.FAILED,
            error="No model specified and no Ollama models available",
        )

    start = time.time()

    translate_task = TaskRequest(
        task_id=task.task_id,
        task_type="inference",
        payload={
            "model": model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        f"You are a translator. Translate the following text to "
                        f"{target_language}. Output ONLY the translation, nothing "
                        f"else. Preserve formatting, paragraph breaks, and tone."
                    ),
                },
                {"role": "user", "content": text},
            ],
            "temperature": 0.3,
            "max_tokens": 4096,
        },
        source_node=task.source_node,
        priority=task.priority,
        timeout_seconds=task.timeout_seconds,
    )

    result = await handle_inference(translate_task)

    if result.status == TaskStatus.COMPLETED and result.result:
        result.result = {
            "translation": result.result.get("response", ""),
            "source_text": text,
            "target_language": target_language,
            "model": model,
            "tokens_per_second": result.result.get("tokens_per_second", 0),
        }
    result.duration_seconds = round(time.time() - start, 2)
    return result


# --- Code Execution Handler ---


_UNSHARE_AVAILABLE: bool | None = None


async def _check_unshare() -> bool:
    """Check if Linux unshare -rn is available for network isolation."""
    global _UNSHARE_AVAILABLE
    if _UNSHARE_AVAILABLE is not None:
        return _UNSHARE_AVAILABLE
    try:
        proc = await asyncio.create_subprocess_exec(
            "unshare", "-rn", "true",
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
        )
        await asyncio.wait_for(proc.wait(), timeout=3)
        _UNSHARE_AVAILABLE = proc.returncode == 0
    except (FileNotFoundError, asyncio.TimeoutError):
        _UNSHARE_AVAILABLE = False
    return _UNSHARE_AVAILABLE


async def handle_code_run(task: TaskRequest) -> TaskResult:
    """Run Python code in a sandboxed subprocess.

    Sandbox: runs in a temp directory under /tmp, minimal env variables,
    network isolated via Linux unshare when available. Code is written to
    a file and executed directly (no shell injection).

    Expected payload:
        code: str      — Python code to execute
        timeout: int   — Max execution time in seconds (default 30, max 60)
    """
    import sys
    import tempfile
    from pathlib import Path

    payload = task.payload
    code = payload.get("code")
    timeout = min(payload.get("timeout", 30), 60)

    if not code:
        return TaskResult(
            task_id=task.task_id,
            status=TaskStatus.FAILED,
            error="Missing required field: 'code'",
        )

    start = time.time()

    try:
        with tempfile.TemporaryDirectory(prefix="myco_sandbox_") as tmpdir:
            code_path = Path(tmpdir) / "run.py"
            code_path.write_text(code)

            # Minimal environment — no credentials, restricted paths
            sandbox_path = "/usr/bin:/bin:/usr/local/bin"
            if sys.platform == "darwin":
                sandbox_path = "/opt/homebrew/bin:" + sandbox_path
            env = {
                "PATH": sandbox_path,
                "HOME": tmpdir,
                "TMPDIR": tmpdir,
                "PYTHONDONTWRITEBYTECODE": "1",
            }

            # Try network-isolated execution via Linux user namespaces
            # (not available on macOS — falls back to unsandboxed subprocess)
            net_isolated = await _check_unshare()
            if net_isolated:
                cmd = ["unshare", "-rn", sys.executable, str(code_path)]
            else:
                cmd = [sys.executable, str(code_path)]

            logger.info(
                f"🔒 Running code (timeout={timeout}s, "
                f"net_isolated={net_isolated})"
            )

            proc = await asyncio.create_subprocess_exec(
                *cmd,
                cwd=tmpdir,
                env=env,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            try:
                stdout_raw, stderr_raw = await asyncio.wait_for(
                    proc.communicate(), timeout=timeout
                )
            except asyncio.TimeoutError:
                proc.kill()
                await proc.wait()
                return TaskResult(
                    task_id=task.task_id,
                    status=TaskStatus.FAILED,
                    error=f"Code execution timed out after {timeout}s",
                    duration_seconds=round(time.time() - start, 2),
                )

            stdout = stdout_raw.decode(errors="replace")[:50_000]
            stderr = stderr_raw.decode(errors="replace")[:50_000]

            return TaskResult(
                task_id=task.task_id,
                status=TaskStatus.COMPLETED,
                result={
                    "stdout": stdout,
                    "stderr": stderr,
                    "return_code": proc.returncode,
                    "network_isolated": net_isolated,
                    "timeout": timeout,
                },
                duration_seconds=round(time.time() - start, 2),
            )
    except Exception as e:
        return TaskResult(
            task_id=task.task_id,
            status=TaskStatus.FAILED,
            error=f"Code execution failed: {e}",
            duration_seconds=round(time.time() - start, 2),
        )


# --- Intent Classification Handler ---


_GATE_MODEL_PREFERENCE = ("gemma3:1b", "llama3.2:1b", "gemma3:4b", "llama3.2:3b")
_EMBEDDING_ONLY = ("nomic-embed-text", "mxbai-embed", "all-minilm", "snowflake-arctic-embed")


def _is_embedding_model(name: str) -> bool:
    """Return True if the model name looks like an embedding-only model."""
    lower = name.lower()
    return any(pat in lower for pat in _EMBEDDING_ONLY)


async def _pick_gate_model_async() -> str | None:
    """Pick the smallest available model for gate tasks (async version)."""
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            resp = await client.get(f"{OLLAMA_BASE}/api/tags")
            resp.raise_for_status()
            models = [m["name"] for m in resp.json().get("models", [])]
    except (httpx.ConnectError, httpx.TimeoutException, httpx.HTTPError):
        return None

    for pattern in _GATE_MODEL_PREFERENCE:
        for m in models:
            if pattern in m and not _is_embedding_model(m):
                return m

    # Fall back to first non-embedding model
    for m in models:
        if not _is_embedding_model(m):
            return m
    return None


_INTENT_SYSTEM_PROMPT = (
    "You are an intent classifier. Analyze the user's message and respond "
    "with ONLY a JSON object, no other text.\n\n"
    '{"tool": "", "mode": "", "scope": ""}\n\n'
    "tool — what tools are needed:\n"
    "  answer: general knowledge, math, coding, creative, conversation\n"
    "  web_search: current/real-time info (news, prices, weather)\n"
    "  rag: user's documents, notes, library, or past conversations\n"
    "  web_and_rag: needs both web and user's documents\n\n"
    "mode — what kind of thinking:\n"
    '  recall: remembering something specific ("what did we...", "where is...", "what does X say...")\n'
    '  explore: open-ended research or brainstorming ("what are some...", "how might...")\n'
    '  execute: precise action ("fix this code", "write the function")\n'
    "  chat: casual conversation, greetings, small talk\n\n"
    "scope — where to search (only matters when tool is rag or web_and_rag):\n"
    "  session: past conversations, things we discussed\n"
    "  docs: user's document library, files, notes, stored documents\n"
    "  facts: stored user preferences and facts\n"
    "  all: search everything\n\n"
    'Example: {"tool": "rag", "mode": "recall", "scope": "session"}'
)

_INTENT_DEFAULT = {"tool": "answer", "mode": "chat", "scope": "all"}
_VALID_TOOLS = {"answer", "web_search", "rag", "web_and_rag"}
_VALID_MODES = {"recall", "explore", "execute", "chat"}
_VALID_SCOPES = {"session", "docs", "facts", "all"}


async def handle_intent_classify(task: TaskRequest) -> TaskResult:
    """Classify user intent on a CPU worker node.

    Expected payload:
        query: str      — the user's message
        model: str      — gate model override (optional)

    Returns result dict with: tool, scope, confidence
    """
    payload = task.payload
    query = payload.get("query")

    if not query:
        return TaskResult(
            task_id=task.task_id,
            status=TaskStatus.FAILED,
            error="Missing required field: 'query'",
        )

    model = payload.get("model")
    if not model:
        model = await _pick_gate_model_async()
    if not model:
        return TaskResult(
            task_id=task.task_id,
            status=TaskStatus.FAILED,
            error="No model available for intent classification",
        )

    start = time.time()
    logger.info(f"🧠 Classifying intent: {query[:60]!r} (model={model})")

    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(5.0, read=15.0)) as client:
            resp = await client.post(
                f"{OLLAMA_BASE}/api/chat",
                json={
                    "model": model,
                    "messages": [
                        {"role": "system", "content": _INTENT_SYSTEM_PROMPT},
                        {"role": "user", "content": query},
                    ],
                    "options": {"temperature": 0.0, "num_predict": 100},
                    "stream": False,
                },
            )
            resp.raise_for_status()
            raw = resp.json().get("message", {}).get("content", "").strip()
    except httpx.TimeoutException:
        return TaskResult(
            task_id=task.task_id,
            status=TaskStatus.FAILED,
            error="Ollama timed out during intent classification",
            duration_seconds=round(time.time() - start, 2),
        )
    except httpx.ConnectError:
        return TaskResult(
            task_id=task.task_id,
            status=TaskStatus.FAILED,
            error="Cannot connect to Ollama — is it running?",
        )

    # Parse JSON from response
    raw = raw.strip("`").strip()
    if raw.startswith("json"):
        raw = raw[4:].strip()

    result = dict(_INTENT_DEFAULT)
    try:
        data = json.loads(raw)
        tool = data.get("tool", "answer")
        if tool in _VALID_TOOLS:
            result["tool"] = tool
        mode = data.get("mode", "chat")
        if mode in _VALID_MODES:
            result["mode"] = mode
        scope = data.get("scope", "all")
        if scope in _VALID_SCOPES:
            result["scope"] = scope
    except (json.JSONDecodeError, ValueError):
        lower = raw.lower()
        for category in ("web_and_rag", "web_search", "rag", "answer"):
            if category in lower:
                result["tool"] = category
                break

    # Check for past-reference patterns
    from mycoswarm.solo import detect_past_reference
    if detect_past_reference(query):
        result["scope"] = "session"

    duration = round(time.time() - start, 2)
    logger.info(f"🧠 Intent: {result['tool']}/{result['mode']}/{result['scope']} ({duration}s)")

    return TaskResult(
        task_id=task.task_id,
        status=TaskStatus.COMPLETED,
        result=result,
        duration_seconds=duration,
    )


# --- Handler Registry ---

HANDLERS: dict[str, Callable[[TaskRequest], Awaitable[TaskResult]]] = {
    "inference": handle_inference,
    "embedding": handle_embedding,
    "web_search": handle_web_search,
    "web_fetch": handle_web_fetch,
    "file_read": handle_file_read,
    "file_summarize": handle_file_summarize,
    "translate": handle_translate,
    "code_run": handle_code_run,
    "intent_classify": handle_intent_classify,
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
