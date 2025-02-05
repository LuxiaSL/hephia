"""
Enhanced timer system for coordinating updates across Hephia's subsystems.

This timer system manages different update frequencies for various components
while maintaining non-blocking async operation. It extends the original
GlobalTimer with additional features for component coordination.
"""

import asyncio
import time
from dataclasses import dataclass
from typing import Callable, Optional, Dict, List
from event_dispatcher import global_event_dispatcher, Event

@dataclass
class TimedTask:
    """
    Represents a task that should be executed at specific intervals.
    
    Attributes:
        name: Unique identifier for the task
        interval: Time between executions in seconds
        callback: Async function to execute
        condition: Optional function that must return True for execution
        last_run: Timestamp of last execution
        priority: Lower numbers run first
    """
    name: str
    interval: float
    callback: Callable
    condition: Optional[Callable] = None
    last_run: float = 0
    priority: int = 0

class TimerCoordinator:
    """
    Coordinates timed updates across all system components.
    
    This class extends the original GlobalTimer concept to handle
    the complexity of managing updates for the integrated system.
    """
    
    def __init__(self):
        """Initialize the timer coordinator."""
        self.tasks: Dict[str, TimedTask] = {}
        self.is_running: bool = False
        self.task_groups: Dict[str, List[str]] = {
            'high_frequency': [],    # Needs, behaviors (1-5s)
            'medium_frequency': [],  # Emotions, mood (30-60s)
            'low_frequency': []      # Thoughts, memory (300s+)
        }
    
    def add_task(self, 
                 name: str, 
                 interval: float, 
                 callback: Callable, 
                 condition: Optional[Callable] = None,
                 priority: int = 0) -> None:
        """
        Add a new task using parameters.
        
        Args:
            name: Unique identifier for the task
            interval: Time between executions in seconds
            callback: Async function to execute
            condition: Optional function that must return True for execution
            priority: Lower numbers run first
        """
        task = TimedTask(
            name=name,
            interval=interval,
            callback=callback,
            condition=condition,
            priority=priority
        )
        
        self.tasks[task.name] = task
        
        # Categorize task by frequency
        if task.interval < 10:
            self.task_groups['high_frequency'].append(task.name)
        elif task.interval < 60:
            self.task_groups['medium_frequency'].append(task.name)
        else:
            self.task_groups['low_frequency'].append(task.name)
            
        global_event_dispatcher.dispatch_event(
            Event("timer:task_added", {
                "task_name": task.name,
                "interval": task.interval
            })
        )
    
    def remove_task(self, task_name: str) -> None:
        """
        Remove a task from the coordinator.
        
        Args:
            task_name: Name of task to remove
        """
        if task_name in self.tasks:
            
            # Remove from task groups
            for group in self.task_groups.values():
                if task_name in group:
                    group.remove(task_name)
            
            # Notify system of task removal
            global_event_dispatcher.dispatch_event(
                Event("timer:task_removed", {
                    "task_name": task_name
                })
            )
    
    async def run(self) -> None:
        """
        Main loop for executing tasks at their specified intervals.
        
        This method runs continuously while is_running is True,
        executing tasks when their interval has elapsed and their
        conditions are met.
        """
        self.is_running = True
        
        while self.is_running:
            current_time = time.time()
            
            # Process tasks in priority order
            sorted_tasks = sorted(
                self.tasks.values(),
                key=lambda x: (x.priority, x.interval)
            )
            
            for task in sorted_tasks:
                if current_time - task.last_run >= task.interval:
                    if task.condition is None or task.condition():
                        try:
                            # Create task for execution
                            asyncio.create_task(self._execute_task(task))
                        except Exception as e:
                            # Log error but continue running
                            global_event_dispatcher.dispatch_event(
                                Event("timer:task_error", {
                                    "task_name": task.name,
                                    "error": str(e)
                                })
                            )
            
            # Small sleep to prevent CPU hogging
            await asyncio.sleep(0.1)
    
    async def _execute_task(self, task: TimedTask) -> None:
        """
        Execute a single task and update its last_run time.
        
        Args:
            task: TimedTask to execute
        """
        try:
            await task.callback()
            task.last_run = time.time()
            
            # Notify system of successful execution
            global_event_dispatcher.dispatch_event(
                Event("timer:task_executed", {
                    "task_name": task.name,
                    "execution_time": task.last_run
                })
            )
        except Exception as e:
            # Notify system of execution error
            global_event_dispatcher.dispatch_event(
                Event("timer:task_execution_error", {
                    "task_name": task.name,
                    "error": str(e)
                })
            )
    
    def stop(self) -> None:
        """Stop the timer coordinator."""
        self.is_running = False
        global_event_dispatcher.dispatch_event(
            Event("timer:stopped", None)
        )

    def get_task_status(self) -> Dict:
        """
        Get current status of all tasks.
        
        Returns:
            Dict containing task statuses grouped by frequency
        """
        current_time = time.time()
        status = {
            'high_frequency': {},
            'medium_frequency': {},
            'low_frequency': {}
        }
        
        for group_name, task_names in self.task_groups.items():
            for task_name in task_names:
                task = self.tasks[task_name]
                status[group_name][task_name] = {
                    'next_run': task.last_run + task.interval - current_time,
                    'last_run': task.last_run,
                    'interval': task.interval
                }
                
        return status