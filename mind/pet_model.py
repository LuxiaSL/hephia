"""
mind/pet_model.py

Pet friend layer — generates personality-rich responses using a small/cheap model.
State and memory context color the response naturally.
"""

from __future__ import annotations

from typing import List, Dict, Any, Optional, TYPE_CHECKING

from config import Config
from loggers.loggers import BrainLogger
from .models import MindResponse
from .prompts import build_pet_system_prompt, TASK_CLASSIFICATION_TEMPLATE

if TYPE_CHECKING:
    from api_clients import APIManager
    from .models import RetrievedMemory


class PetModel:
    """
    Interface for personality-rich pet responses.
    Uses a small/cheap model (default: haiku 4.5) for low-latency conversation.
    """

    def __init__(self, api_manager: APIManager) -> None:
        self.api_manager = api_manager
        self.logger = BrainLogger

    async def generate_response(
        self,
        messages: List[Dict[str, str]],
    ) -> str:
        """
        Generate a personality-rich response from formatted messages.
        Messages should already include the system prompt with state/memory context.

        Returns the response text, or a fallback on error.
        """
        try:
            provider, model_id, temperature, max_tokens = self._resolve_model()

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

            # Some providers may return dict even with return_content_only
            if isinstance(response, dict):
                return (
                    response.get("content", "")
                    or response.get("text", "")
                    or str(response)
                )

            return str(response)

        except Exception as e:
            self.logger.error(f"Pet model generation failed: {e}")
            return "..."

    async def classify_intent(self, message: str) -> str:
        """
        Quick intent classification using the pet model.
        Returns 'CHAT' or 'TASK'.
        """
        try:
            provider, model_id, _, _ = self._resolve_model()

            prompt = TASK_CLASSIFICATION_TEMPLATE.format(message=message)
            messages = [
                {"role": "system", "content": "Respond with exactly one word: CHAT or TASK."},
                {"role": "user", "content": prompt},
            ]

            response = await self.api_manager.create_completion(
                provider=provider,
                messages=messages,
                model=model_id,
                temperature=0.0,
                max_tokens=10,
                return_content_only=True,
            )

            result = str(response).strip().upper()
            return "TASK" if "TASK" in result else "CHAT"

        except Exception as e:
            self.logger.error(f"Intent classification failed: {e}")
            return "CHAT"

    def _resolve_model(self) -> tuple:
        """
        Resolve pet model config from Config.
        Returns (provider_key, model_id, temperature, max_tokens).
        """
        model_name = Config.get_pet_model()
        model_config = Config.AVAILABLE_MODELS.get(model_name)

        if model_config is None:
            self.logger.warning(
                f"Pet model '{model_name}' not found in AVAILABLE_MODELS, "
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
