import asyncio
import logging
from typing import Optional
from contextlib import asynccontextmanager

logger = logging.getLogger(__name__)

class ReentrantAsyncLock:
    """A reentrant asyncio lock for nested acquisitions by the same task."""
    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._owner: Optional[asyncio.Task] = None
        self._count: int = 0

    async def acquire(self):
        current_task = asyncio.current_task()
        if current_task is None:
            raise RuntimeError("No current task found. Must be within asyncio.")

        if self._owner == current_task:
            self._count += 1
            logger.debug("ReentrantAsyncLock: re-acquired by task %r, count=%d", current_task, self._count)
        else:
            logger.debug("ReentrantAsyncLock: task %r waiting for lock...", current_task)
            await self._lock.acquire()
            self._owner = current_task
            self._count = 1
            logger.debug("ReentrantAsyncLock: acquired by task %r, count=1", current_task)

    def release(self):
        current_task = asyncio.current_task()
        if self._owner != current_task:
            raise RuntimeError("Lock release attempted by non-owner task.")

        self._count -= 1
        logger.debug("ReentrantAsyncLock: released by task %r, new count=%d", current_task, self._count)
        if self._count == 0:
            self._owner = None
            self._lock.release()
            logger.debug("ReentrantAsyncLock: fully released by task %r", current_task)

    async def __aenter__(self):
        await self.acquire()
        return self

    async def __aexit__(self, exc_type, exc, tb):
        self.release()

shared_reentrant_lock = ReentrantAsyncLock()
