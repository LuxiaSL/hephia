"""
services/worker_queue.py

In-memory async task queue with TTL-based cleanup.
Uses ClaudeSDKClient for persistent interactive agent sessions with tool access.
Falls back to WorkerModel.execute_task() if the SDK is not installed.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any, Dict, Optional, TYPE_CHECKING

from config import Config
from shared_models.api_models import (
    AgentProgressEntry,
    AgentQuestion,
    AgentQuestionOption,
    WorkerTaskStatus,
)
from loggers import SystemLogger

if TYPE_CHECKING:
    from mind.worker_model import WorkerModel

# Lazy import flag — set on first _execute() call
_agent_sdk_available: Optional[bool] = None

# Timeout for waiting on user replies to agent questions
_REPLY_TIMEOUT_SECONDS = 300  # 5 minutes


def _check_agent_sdk() -> bool:
    """Check if claude_agent_sdk is importable. Cached after first call."""
    global _agent_sdk_available
    if _agent_sdk_available is None:
        try:
            import claude_agent_sdk  # noqa: F401
            _agent_sdk_available = True
        except ImportError:
            _agent_sdk_available = False
            SystemLogger.warning(
                "claude-agent-sdk not installed — worker tasks will use "
                "legacy single-shot WorkerModel. Install with: "
                "pip install claude-agent-sdk"
            )
    return _agent_sdk_available


def _resolve_agent_model() -> Optional[str]:
    """
    Map the configured WORKER_MODEL to an Anthropic model ID the SDK understands.
    Returns None for non-Anthropic providers (lets the SDK use its default).
    """
    from config import ProviderType

    model_name = Config.get_worker_model()
    model_config = Config.AVAILABLE_MODELS.get(model_name)
    if model_config is None:
        return None
    if model_config.provider != ProviderType.ANTHROPIC:
        return None
    return model_config.model_id


def _summarize_tool_input(tool_name: str, tool_input: Dict[str, Any]) -> str:
    """Extract the most useful field from a tool's input for progress display."""
    key_map = {
        "Bash": "command",
        "Read": "file_path",
        "Write": "file_path",
        "Edit": "file_path",
        "Grep": "pattern",
        "Glob": "pattern",
        "WebSearch": "query",
        "WebFetch": "url",
    }
    key = key_map.get(tool_name)
    if key and key in tool_input:
        value = str(tool_input[key])
        if len(value) > 120:
            value = value[:117] + "..."
        return value
    # Fallback: show first key's value
    for v in tool_input.values():
        s = str(v)
        if len(s) > 120:
            s = s[:117] + "..."
        return s
    return ""


class WorkerTaskQueue:
    """
    In-memory async task queue with persistent agent sessions.

    - submit() creates an asyncio task and returns a task_id immediately
    - get_status() returns current state of any tracked task
    - reply() sends a user reply to an active session (question answer or follow-up)
    - cleanup() prunes completed tasks older than TTL
    """

    TTL_SECONDS: int = 3600  # 1 hour
    MAX_TASKS: int = 100

    def __init__(self, worker_model: WorkerModel) -> None:
        self.worker_model = worker_model
        self._tasks: Dict[str, WorkerTaskStatus] = {}
        self._running: set[asyncio.Task[None]] = set()  # prevent GC of asyncio tasks
        self._lock = asyncio.Lock()

        # Per-task ClaudeSDKClient instances (for reply routing)
        self._clients: Dict[str, Any] = {}  # task_id → ClaudeSDKClient

        # Reply coordination for AskUserQuestion interception
        self._reply_events: Dict[str, asyncio.Event] = {}
        self._reply_answers: Dict[str, str] = {}

    def submit(self, task_spec: str, context: Optional[str] = None) -> str:
        """Submit a task for async execution. Returns task_id."""
        import uuid

        task_id = str(uuid.uuid4())
        SystemLogger.info(f"Worker task submitted: {task_id} — {task_spec[:80]}")

        status = WorkerTaskStatus(
            task_id=task_id,
            status="pending",
            created_at=time.time(),
        )
        self._tasks[task_id] = status

        # Must hold a strong reference — asyncio only keeps weak refs (Python 3.12+)
        bg = asyncio.create_task(self._execute(task_id, task_spec, context))
        self._running.add(bg)
        bg.add_done_callback(self._running.discard)
        return task_id

    def get_status(self, task_id: str) -> Optional[WorkerTaskStatus]:
        """Get current status of a task."""
        return self._tasks.get(task_id)

    async def reply(self, task_id: str, message: str) -> WorkerTaskStatus:
        """
        Send a user reply to an active worker session.

        - If awaiting_input: delivers the answer to the blocked can_use_tool callback.
        - If running: interrupts the current execution, then sends as follow-up.
        - If completed/failed: sends a follow-up query on the existing session.

        Returns the updated task status.
        Raises ValueError if the task is not found or not in a replyable state.
        """
        task = self._tasks.get(task_id)
        if task is None:
            raise ValueError(f"Task {task_id} not found")

        if task.status == "awaiting_input":
            # Deliver answer to the waiting can_use_tool callback
            event = self._reply_events.get(task_id)
            if event is None:
                raise ValueError(f"Task {task_id} has no pending reply event")

            self._reply_answers[task_id] = message
            task.pending_questions = []
            task.status = "running"
            event.set()
            SystemLogger.info(
                f"Worker task {task_id} — reply delivered to awaiting question"
            )

        elif task.status == "running":
            # Interrupt current execution and send as correction/follow-up
            client = self._clients.get(task_id)
            if client is None:
                raise ValueError(
                    f"Task {task_id} has no active client session"
                )

            SystemLogger.info(
                f"Worker task {task_id} — interrupting and sending reply: {message[:80]}"
            )
            await client.interrupt()

            # Wait briefly for the interrupt to settle (ResultMessage from interrupted run)
            await asyncio.sleep(0.5)

            # Send the follow-up and process the new response
            task.status = "running"
            await client.query(message)
            bg = asyncio.create_task(
                self._process_response_loop(task_id)
            )
            self._running.add(bg)
            bg.add_done_callback(self._running.discard)

        elif task.status in ("completed", "failed"):
            # Follow-up on a finished session — resume the client
            client = self._clients.get(task_id)
            if client is None:
                raise ValueError(
                    f"Task {task_id} session expired — no client available for follow-up"
                )

            SystemLogger.info(f"Worker task {task_id} — follow-up reply: {message[:80]}")
            task.status = "running"
            task.completed_at = None
            task.result = None
            task.error = None

            # Send the follow-up query and spawn a new response processor
            await client.query(message)
            bg = asyncio.create_task(
                self._process_response_loop(task_id)
            )
            self._running.add(bg)
            bg.add_done_callback(self._running.discard)

        elif task.status == "pending":
            raise ValueError(
                f"Task {task_id} hasn't started yet — cannot reply"
            )

        return task

    async def stop(self, task_id: str) -> WorkerTaskStatus:
        """
        Interrupt a running or awaiting task.
        The agent's current execution is cancelled and the task is marked completed.
        """
        task = self._tasks.get(task_id)
        if task is None:
            raise ValueError(f"Task {task_id} not found")

        if task.status not in ("running", "awaiting_input", "pending"):
            raise ValueError(
                f"Task {task_id} is '{task.status}' — nothing to stop"
            )

        client = self._clients.get(task_id)

        # If awaiting input, unblock the event so the callback doesn't hang
        event = self._reply_events.pop(task_id, None)
        if event is not None:
            self._reply_answers[task_id] = ""
            event.set()

        if client is not None:
            try:
                await client.interrupt()
            except Exception as e:
                SystemLogger.debug(f"Interrupt error for {task_id}: {e}")

        task.pending_questions = []
        task.status = "completed"
        task.result = task.result or "(Stopped by user)"
        task.completed_at = time.time()

        SystemLogger.info(f"Worker task {task_id} — stopped by user")
        return task

    async def delete_task(self, task_id: str) -> bool:
        """Delete a single task and clean up its resources. Returns True if found."""
        task = self._tasks.get(task_id)
        if task is None:
            return False
        # Don't allow deleting a running task
        if task.status in ("running", "pending"):
            raise ValueError(f"Cannot delete task {task_id} while it is {task.status}")
        await self._cleanup_task(task_id)
        return True

    async def clear_tasks(self, completed_only: bool = True) -> int:
        """
        Bulk remove tasks. Returns count of removed tasks.
        If completed_only is True, only removes completed/failed tasks.
        If False, removes all non-running tasks.
        """
        to_remove: list[str] = []
        for tid, t in self._tasks.items():
            if completed_only:
                if t.status in ("completed", "failed"):
                    to_remove.append(tid)
            else:
                if t.status not in ("running", "pending"):
                    to_remove.append(tid)
        for tid in to_remove:
            await self._cleanup_task(tid)
        return len(to_remove)

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
                await self._cleanup_task(tid)

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
                    await self._cleanup_task(tid)

    async def _cleanup_task(self, task_id: str) -> None:
        """Clean up all resources associated with a task."""
        client = self._clients.pop(task_id, None)
        if client is not None:
            try:
                await client.disconnect()
            except Exception as e:
                SystemLogger.debug(f"Error disconnecting client for {task_id}: {e}")

        self._reply_events.pop(task_id, None)
        self._reply_answers.pop(task_id, None)
        self._tasks.pop(task_id, None)

    async def _execute(
        self, task_id: str, task_spec: str, context: Optional[str]
    ) -> None:
        """Run the agent (or legacy worker model) and update task status."""
        task = self._tasks.get(task_id)
        if task is None:
            return

        task.status = "running"
        sdk_available = _check_agent_sdk()
        SystemLogger.info(
            f"Worker task {task_id} executing — "
            f"agent_sdk={'available' if sdk_available else 'not found, using legacy'}"
        )

        if sdk_available:
            await self._execute_agent(task, task_spec, context)
        else:
            await self._execute_legacy(task, task_spec, context)

    async def _execute_legacy(
        self,
        task: WorkerTaskStatus,
        task_spec: str,
        context: Optional[str],
    ) -> None:
        """Fallback: single-shot WorkerModel execution."""
        try:
            result = await self.worker_model.execute_task(
                task_spec=task_spec,
                context=context,
            )
            task.status = "completed"
            task.result = result
            task.completed_at = time.time()
        except Exception as e:
            SystemLogger.error(f"Worker task {task.task_id} failed: {e}")
            task.status = "failed"
            task.error = str(e)
            task.completed_at = time.time()

    async def _execute_agent(
        self,
        task: WorkerTaskStatus,
        task_spec: str,
        context: Optional[str],
    ) -> None:
        """Run a persistent ClaudeSDKClient session with AskUserQuestion interception."""
        from claude_agent_sdk import ClaudeSDKClient, ClaudeAgentOptions
        from claude_agent_sdk.types import (
            PermissionResultAllow,
            ToolPermissionContext,
        )
        from mind.prompts import AGENT_SYSTEM_PROMPT

        task_id = task.task_id

        # Build prompt
        prompt = task_spec
        if context:
            prompt = f"{context}\n\nTask: {task_spec}"

        # Resolve model
        model_id = _resolve_agent_model()
        agent_cwd = Config.get_agent_cwd()
        SystemLogger.info(
            f"Agent task {task_id} — model={model_id or 'SDK default'}, "
            f"cwd={agent_cwd}, max_turns={Config.get_agent_max_turns()}, "
            f"budget=${Config.get_agent_max_budget()}"
        )

        # AskUserQuestion interception callback
        async def can_use_tool(
            tool_name: str,
            input_data: Dict[str, Any],
            _context: ToolPermissionContext,
        ) -> PermissionResultAllow:
            """Intercept AskUserQuestion to route questions to the dashboard UI."""
            if tool_name != "AskUserQuestion":
                # Auto-approve all other tools
                return PermissionResultAllow(updated_input=input_data)

            # Extract questions from the tool input
            raw_questions = input_data.get("questions", [])
            agent_questions: list[AgentQuestion] = []
            for q in raw_questions:
                options = [
                    AgentQuestionOption(
                        label=opt.get("label", ""),
                        description=opt.get("description", ""),
                    )
                    for opt in q.get("options", [])
                ]
                agent_questions.append(
                    AgentQuestion(
                        question=q.get("question", ""),
                        options=options,
                        multi_select=q.get("multiSelect", False),
                    )
                )

            # Store questions on the task and set awaiting_input
            task.pending_questions = agent_questions
            task.status = "awaiting_input"

            # Add a progress entry so the user sees it in the feed
            task.progress.append(
                AgentProgressEntry(
                    type="text",
                    content=f"Asking: {agent_questions[0].question}" if agent_questions else "Asking a question...",
                    timestamp=time.time(),
                )
            )

            SystemLogger.info(
                f"Agent task {task_id} — awaiting user input "
                f"({len(agent_questions)} question(s))"
            )

            # Create event and wait for the user's reply
            event = asyncio.Event()
            self._reply_events[task_id] = event

            try:
                await asyncio.wait_for(event.wait(), timeout=_REPLY_TIMEOUT_SECONDS)
            except asyncio.TimeoutError:
                SystemLogger.warning(
                    f"Agent task {task_id} — reply timeout after "
                    f"{_REPLY_TIMEOUT_SECONDS}s, auto-answering"
                )
                # Auto-answer with a default so the agent can continue
                self._reply_answers[task_id] = (
                    "The user did not respond in time. "
                    "Please proceed with your best judgment."
                )
                task.pending_questions = []
                task.status = "running"

            # Build the answer dict keyed by question text
            answer_text = self._reply_answers.pop(task_id, "")
            self._reply_events.pop(task_id, None)

            answers: Dict[str, str] = {}
            if agent_questions:
                # Map the single reply to all questions (most common: 1 question)
                for q in agent_questions:
                    answers[q.question] = answer_text

            return PermissionResultAllow(
                updated_input={
                    "questions": raw_questions,
                    "answers": answers,
                }
            )

        # Build allowed tools — include AskUserQuestion for interception
        allowed_tools = Config.get_agent_allowed_tools()
        if "AskUserQuestion" not in allowed_tools:
            allowed_tools = [*allowed_tools, "AskUserQuestion"]

        # Build options
        options = ClaudeAgentOptions(
            system_prompt=AGENT_SYSTEM_PROMPT,
            allowed_tools=allowed_tools,
            permission_mode="bypassPermissions",
            max_turns=Config.get_agent_max_turns(),
            max_budget_usd=Config.get_agent_max_budget(),
            cwd=agent_cwd,
            can_use_tool=can_use_tool,
        )
        if model_id:
            options.model = model_id

        extra_dirs = Config.get_agent_allowed_dirs()
        if extra_dirs:
            options.add_dirs = extra_dirs

        try:
            SystemLogger.info(f"Agent task {task_id} — connecting ClaudeSDKClient")
            client = ClaudeSDKClient(options)
            await client.connect()
            self._clients[task_id] = client

            # Send initial query
            await client.query(prompt)

            # Process the response stream
            await self._process_response_loop(task_id)

        except Exception as e:
            SystemLogger.error(f"Agent task {task_id} failed: {e}")
            task.status = "failed"
            task.error = str(e)
            task.progress.append(
                AgentProgressEntry(
                    type="error",
                    content=str(e)[:500],
                    timestamp=time.time(),
                )
            )
            task.completed_at = time.time()
            # Don't disconnect on error — client may still be usable for follow-ups
            # It will be cleaned up by TTL cleanup

    async def _process_response_loop(self, task_id: str) -> None:
        """
        Consume messages from receive_response() and update task progress.
        Used for both initial execution and follow-up replies.
        """
        from claude_agent_sdk import (
            AssistantMessage,
            ResultMessage,
            TextBlock,
            ToolUseBlock,
            ThinkingBlock,
        )

        task = self._tasks.get(task_id)
        client = self._clients.get(task_id)
        if task is None or client is None:
            return

        accumulated_text: list[str] = []

        try:
            async for message in client.receive_response():
                SystemLogger.debug(
                    f"Agent task {task_id} — received: {type(message).__name__}"
                )
                if isinstance(message, AssistantMessage):
                    for block in message.content:
                        if isinstance(block, TextBlock):
                            accumulated_text.append(block.text)
                            task.progress.append(
                                AgentProgressEntry(
                                    type="text",
                                    content=block.text[:500],
                                    timestamp=time.time(),
                                )
                            )
                        elif isinstance(block, ToolUseBlock):
                            summary = _summarize_tool_input(block.name, block.input)
                            task.progress.append(
                                AgentProgressEntry(
                                    type="tool_use",
                                    content=f"{block.name}: {summary}" if summary else block.name,
                                    timestamp=time.time(),
                                    tool_name=block.name,
                                    tool_input_summary=summary or None,
                                )
                            )
                        elif isinstance(block, ThinkingBlock):
                            task.progress.append(
                                AgentProgressEntry(
                                    type="thinking",
                                    content="Thinking...",
                                    timestamp=time.time(),
                                )
                            )

                elif isinstance(message, ResultMessage):
                    SystemLogger.info(
                        f"Agent task {task_id} — result: "
                        f"error={message.is_error}, turns={message.num_turns}, "
                        f"cost=${message.total_cost_usd}"
                    )
                    task.cost_usd = message.total_cost_usd
                    task.num_turns = message.num_turns

                    if message.is_error:
                        task.status = "failed"
                        task.error = (
                            message.result
                            or "\n".join(accumulated_text)
                            or "Agent encountered an error"
                        )
                    else:
                        task.status = "completed"
                        task.result = (
                            message.result
                            or "\n".join(accumulated_text)
                            or "Task completed (no output)"
                        )
                    task.completed_at = time.time()
                    return

            # If we exit the loop without a ResultMessage, mark completed
            if task.status == "running":
                task.status = "completed"
                task.result = "\n".join(accumulated_text) or "Task completed (no output)"
                task.completed_at = time.time()

        except Exception as e:
            SystemLogger.error(f"Agent task {task_id} response loop failed: {e}")
            task.status = "failed"
            task.error = str(e)
            task.progress.append(
                AgentProgressEntry(
                    type="error",
                    content=str(e)[:500],
                    timestamp=time.time(),
                )
            )
            task.completed_at = time.time()
