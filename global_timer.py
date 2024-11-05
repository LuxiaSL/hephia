# global_timer.py

import time
import asyncio
from asyncio import create_task, sleep
from typing import Callable, List, Optional

class TimedTask:
    def __init__(self, interval: float, callback: Callable):
        self.interval = interval
        self.callback = callback
        self.last_run = time.time()

class GlobalTimer:
    def __init__(self):
        self.tasks: List[TimedTask] = []
        self.is_running = False

    def add_task(self, interval: float, callback: Callable) -> None:
        """Add a new task with a specified interval."""
        self.tasks.append(TimedTask(interval, callback))

    async def run(self):
        """Run the timer, executing tasks at their specified intervals."""
        self.is_running = True
        while self.is_running:
            current_time = time.time()
            for task in self.tasks:
                if current_time - task.last_run >= task.interval:
                    create_task(task.callback()) if asyncio.iscoroutinefunction(task.callback) else task.callback()
                    task.last_run = current_time
            await sleep(0.1)  # Small delay to prevent CPU hogging

    def stop(self):
        """Stop the timer."""
        self.is_running = False
