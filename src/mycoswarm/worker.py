"""mycoSwarm task worker.

Pulls tasks from the node's queue and executes them.
Each task type has a handler function. Currently supports:

  - inference: Send prompt to Ollama and return response
  - ping: Simple health check task (for testing)

Future handlers:
  - web_fetch: Download and parse web pages
  - file_process: Read/transform files
  - embedding: Generate embeddings via Ollama
"""

import asyncio
import logging
import time
from typing import Callable, Awaitable

import httpx

from mycoswarm.api import TaskQueue, TaskRequest, TaskResult, TaskStatus

logger = logging.getLogger(__name__)

# Ollama API
OLLAMA_BASE = "http://localhost:11434"
OLLAMA_TIMEOUT = 300.0  # 5 min max for inference


# --- Task Handlers ---


async def handle_inference(task: TaskRequest) -> TaskResult:
    """Run inference via local Ollama.

    Expected payload:
        model: str          — Ollama model name (e.g. "qwen2.5:14b-instruct-q4_K_M")
        prompt: str         — The prompt text
        system: str         — Optional system prompt
        temperature: float  — Optional, default 0.7
        max_tokens: int     — Optional, default 2048
    """
    payload = task.payload
    model = payload.get("model")
    prompt = payload.get("prompt")

    if not model or not prompt:
        return TaskResult(
            task_id=task.task_id,
            status=TaskStatus.FAILED,
            error="Missing required fields: 'model' and 'prompt'",
        )

    # Build Ollama request
    ollama_payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": payload.get("temperature", 0.7),
            "num_predict": payload.get("max_tokens", 2048),
        },
    }

    if payload.get("system"):
        ollama_payload["system"] = payload["system"]

    start = time.time()

    try:
        async with httpx.AsyncClient(timeout=OLLAMA_TIMEOUT) as client:
            response = await client.post(
                f"{OLLAMA_BASE}/api/generate",
                json=ollama_payload,
            )
            response.raise_for_status()
            data = response.json()

        duration = time.time() - start

        return TaskResult(
            task_id=task.task_id,
            status=TaskStatus.COMPLETED,
            result={
                "response": data.get("response", ""),
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
            },
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


async def handle_ping(task: TaskRequest) -> TaskResult:
    """Simple test handler — echoes back the payload."""
    return TaskResult(
        task_id=task.task_id,
        status=TaskStatus.COMPLETED,
        result={"echo": task.payload, "message": "pong"},
        duration_seconds=0.0,
    )


# --- Handler Registry ---

HANDLERS: dict[str, Callable[[TaskRequest], Awaitable[TaskResult]]] = {
    "inference": handle_inference,
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

            if not handler:
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

            try:
                result = await handler(task)
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

            except Exception as e:
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
