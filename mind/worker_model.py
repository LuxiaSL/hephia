"""
mind/worker_model.py

Worker helper layer — capable model for task execution.
Stateless: receives a task spec, returns structured results.
"""

from __future__ import annotations

from typing import List, Dict, Any, Optional, TYPE_CHECKING

from config import Config
from loggers.loggers import BrainLogger
from .prompts import WORKER_SYSTEM_PROMPT

if TYPE_CHECKING:
    from api_clients import APIManager


class WorkerModel:
    """
    Interface for capable task execution.
    Uses a strong model (default: opus 4.6) for complex tasks.
    Only fires when the user explicitly requests a capability.
    """

    def __init__(self, api_manager: APIManager) -> None:
        self.api_manager = api_manager
        self.logger = BrainLogger

    async def execute_task(
        self,
        task_spec: str,
        context: Optional[str] = None,
    ) -> str:
        """
        Execute a task and return the result as text.

        Args:
            task_spec: Description of what to do.
            context: Optional additional context for the task.

        Returns:
            Result text, or error description on failure.
        """
        try:
            provider, model_id, temperature, max_tokens = self._resolve_model()

            user_content = task_spec
            if context:
                user_content = f"{context}\n\nTask: {task_spec}"

            messages: List[Dict[str, str]] = [
                {"role": "system", "content": WORKER_SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ]

            response = await self.api_manager.create_completion(
                provider=provider,
                messages=messages,
                model=model_id,
                temperature=temperature,
                max_tokens=max_tokens,
                return_content_only=True,
            )

            if isinstance(response, str):
                return response

            if isinstance(response, dict):
                return (
                    response.get("content", "")
                    or response.get("text", "")
                    or str(response)
                )

            return str(response)

        except Exception as e:
            self.logger.error(f"Worker task execution failed: {e}")
            return f"I tried to help with that but ran into an issue: {e}"

    def _resolve_model(self) -> tuple:
        """
        Resolve worker model config from Config.
        Returns (provider_key, model_id, temperature, max_tokens).
        """
        model_name = Config.get_worker_model()
        model_config = Config.AVAILABLE_MODELS.get(model_name)

        if model_config is None:
            self.logger.warning(
                f"Worker model '{model_name}' not found in AVAILABLE_MODELS, "
                f"falling back to cognitive model"
            )
            model_name = Config.get_cognitive_model()
            model_config = Config.AVAILABLE_MODELS[model_name]

        return (
            model_config.provider.value,
            model_config.model_id,
            model_config.temperature,
            model_config.max_tokens,
        )
