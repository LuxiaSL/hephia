"""
services/worker_queue.py

In-memory async task queue with TTL-based cleanup.
Wraps WorkerModel.execute_task() for tracked async execution.
"""

from __future__ import annotations

import asyncio
import time
from typing import Dict, Optional, TYPE_CHECKING

from shared_models.api_models import WorkerTaskStatus
from loggers import SystemLogger

if TYPE_CHECKING:
    from mind.worker_model import WorkerModel


class WorkerTaskQueue:
    """
    In-memory async task queue.

    - submit() creates an asyncio task and returns a task_id immediately
    - get_status() returns current state of any tracked task
    - cleanup() prunes completed tasks older than TTL
    """

    TTL_SECONDS: int = 3600  # 1 hour
    MAX_TASKS: int = 100

    def __init__(self, worker_model: WorkerModel) -> None:
        self.worker_model = worker_model
        self._tasks: Dict[str, WorkerTaskStatus] = {}
        self._lock = asyncio.Lock()

    def submit(self, task_spec: str, context: Optional[str] = None) -> str:
        """Submit a task for async execution. Returns task_id."""
        import uuid
        task_id = str(uuid.uuid4())

        status = WorkerTaskStatus(
            task_id=task_id,
            status="pending",
            created_at=time.time(),
        )
        self._tasks[task_id] = status

        asyncio.create_task(self._execute(task_id, task_spec, context))
        return task_id

    def get_status(self, task_id: str) -> Optional[WorkerTaskStatus]:
        """Get current status of a task."""
        return self._tasks.get(task_id)

    async def cleanup(self) -> None:
        """Remove completed/failed tasks older than TTL, enforce max count."""
        async with self._lock:
            now = time.time()
            expired = [
                tid
                for tid, t in self._tasks.items()
                if t.status in ("completed", "failed")
                and t.completed_at is not None
                and (now - t.completed_at) > self.TTL_SECONDS
            ]
            for tid in expired:
                del self._tasks[tid]

            # If still over max, prune oldest completed first
            if len(self._tasks) > self.MAX_TASKS:
                completed = sorted(
                    [
                        (tid, t)
                        for tid, t in self._tasks.items()
                        if t.status in ("completed", "failed")
                    ],
                    key=lambda x: x[1].completed_at or 0,
                )
                to_remove = len(self._tasks) - self.MAX_TASKS
                for tid, _ in completed[:to_remove]:
                    del self._tasks[tid]

    async def _execute(self, task_id: str, task_spec: str, context: Optional[str]) -> None:
        """Run the worker model and update task status."""
        task = self._tasks.get(task_id)
        if task is None:
            return

        task.status = "running"
        try:
            result = await self.worker_model.execute_task(
                task_spec=task_spec,
                context=context,
            )
            task.status = "completed"
            task.result = result
            task.completed_at = time.time()
        except Exception as e:
            SystemLogger.error(f"Worker task {task_id} failed: {e}")
            task.status = "failed"
            task.error = str(e)
            task.completed_at = time.time()
