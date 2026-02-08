"""Example plugin handler — summarize text.

Every plugin handler must export:
    async def handle(task: TaskRequest) -> TaskResult

The task.payload dict contains task-specific data.
"""

import re
import time

from mycoswarm.api import TaskRequest, TaskResult, TaskStatus


async def handle(task: TaskRequest) -> TaskResult:
    """Extract the first N sentences from the input text.

    Expected payload:
        text: str           — text to summarize
        sentences: int      — number of sentences to keep (default 3)
    """
    payload = task.payload
    text = payload.get("text", "")
    if not text:
        return TaskResult(
            task_id=task.task_id,
            status=TaskStatus.FAILED,
            error="Missing required field: 'text'",
        )

    num_sentences = payload.get("sentences", 3)
    start = time.time()

    # Split on sentence boundaries
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    summary = " ".join(parts[:num_sentences])

    return TaskResult(
        task_id=task.task_id,
        status=TaskStatus.COMPLETED,
        result={
            "summary": summary,
            "original_length": len(text),
            "summary_length": len(summary),
            "sentences": min(num_sentences, len(parts)),
        },
        duration_seconds=round(time.time() - start, 4),
    )
