"""
mind/memory_formation.py

LLM-based memory prose generation and significance-gated storage.
Listens for conversation turn events and forms cognitive memories
when content is significant enough to remember.
"""

from __future__ import annotations

import asyncio
import time
from typing import Dict, Any, Optional, TYPE_CHECKING

from config import Config
from event_dispatcher import global_event_dispatcher, Event
from loggers.loggers import BrainLogger, MemoryLogger
from .prompts import build_memory_formation_prompt

if TYPE_CHECKING:
    from internal.modules.cognition.cognitive_bridge import PetCognitiveBridge
    from api_clients import APIManager


class MemoryFormationPipeline:
    """
    Converts conversation into cognitive memories.

    Flow:
    1. Listen for 'mind:conversation_turn' events
    2. Rate-limit and check conversation length
    3. Generate memory prose via LLM
    4. Evaluate significance via bridge
    5. Store if significant
    """

    MIN_FORMATION_INTERVAL = 30.0  # seconds between formation attempts
    MIN_TURNS_FOR_FORMATION = 2    # minimum assistant turns before attempting

    def __init__(
        self,
        bridge: PetCognitiveBridge,
        api_manager: APIManager,
    ) -> None:
        self.bridge = bridge
        self.api_manager = api_manager
        self.logger = BrainLogger
        self._last_formation_time: float = 0.0
        self._formation_lock = asyncio.Lock()
        self.setup_event_listeners()

    def setup_event_listeners(self) -> None:
        """Listen for conversation completion events."""
        global_event_dispatcher.add_listener(
            "mind:conversation_turn",
            lambda event: asyncio.create_task(self._on_conversation_turn(event)),
        )

    async def _on_conversation_turn(self, event: Event) -> None:
        """Process a conversation turn for potential memory formation."""
        try:
            data = event.data or {}
            conversation_excerpt = data.get("excerpt", "")
            turns_since_formation = data.get("turns_since_formation", 0)
            state_context = data.get("state_context", {})

            # Rate-limit
            now = time.time()
            if now - self._last_formation_time < self.MIN_FORMATION_INTERVAL:
                return

            # Need enough conversation to form a meaningful memory
            if turns_since_formation < self.MIN_TURNS_FOR_FORMATION:
                return

            if not conversation_excerpt.strip():
                return

            async with self._formation_lock:
                await self._attempt_formation(
                    conversation_excerpt=conversation_excerpt,
                    state_context=state_context,
                )

        except Exception as e:
            self.logger.error(f"Memory formation event handler failed: {e}")

    async def _attempt_formation(
        self,
        conversation_excerpt: str,
        state_context: Dict[str, Any],
    ) -> Optional[str]:
        """
        Attempt to form a cognitive memory from conversation.
        Returns node_id if a memory was formed, None otherwise.
        """
        try:
            # Generate memory prose
            prose = await self._generate_memory_prose(
                conversation_excerpt=conversation_excerpt,
                state_context=state_context,
            )

            if not prose or len(prose.strip()) < 10:
                self.logger.debug("Memory prose generation returned nothing meaningful")
                return None

            # Evaluate significance
            significance = await self.bridge.evaluate_significance(
                content=prose,
                source="conversation",
            )

            self.logger.info(
                f"Memory significance: {significance:.3f} "
                f"(threshold: {self.bridge.SIGNIFICANCE_THRESHOLD})"
            )

            # Form memory if significant
            node_id = await self.bridge.form_memory(
                content=prose,
                significance=significance,
                source="conversation",
            )

            self._last_formation_time = time.time()

            if node_id:
                MemoryLogger.info(f"Formed memory from conversation: {node_id}")
                global_event_dispatcher.dispatch_event(Event(
                    "mind:memory_formed",
                    {"node_id": node_id, "content": prose, "significance": significance},
                ))

            return node_id

        except Exception as e:
            self.logger.error(f"Memory formation attempt failed: {e}")
            return None

    async def _generate_memory_prose(
        self,
        conversation_excerpt: str,
        state_context: Dict[str, Any],
    ) -> Optional[str]:
        """
        Use the pet model to generate first-person memory prose from conversation.
        """
        try:
            prompt = build_memory_formation_prompt(
                conversation_excerpt=conversation_excerpt,
                state_context=state_context,
            )

            model_name = Config.get_pet_model()
            model_config = Config.AVAILABLE_MODELS.get(model_name)

            if model_config is None:
                model_name = Config.get_cognitive_model()
                model_config = Config.AVAILABLE_MODELS[model_name]

            messages = [
                {"role": "system", "content": "Generate a brief first-person memory. 1-3 sentences only."},
                {"role": "user", "content": prompt},
            ]

            response = await self.api_manager.create_completion(
                provider=model_config.provider.value,
                messages=messages,
                model=model_config.model_id,
                temperature=0.7,
                max_tokens=150,
                return_content_only=True,
            )

            return str(response).strip() if response else None

        except Exception as e:
            self.logger.error(f"Memory prose generation failed: {e}")
            return None

    async def process_explicit_memory(
        self,
        content: str,
        source: str = "explicit",
    ) -> Optional[str]:
        """
        Form a memory from explicit content (e.g., 'remember this').
        Skips LLM generation, goes straight to significance evaluation + formation.
        """
        try:
            significance = await self.bridge.evaluate_significance(
                content=content,
                source=source,
            )

            return await self.bridge.form_memory(
                content=content,
                significance=max(significance, 0.7),  # Boost explicit memories
                source=source,
            )

        except Exception as e:
            self.logger.error(f"Explicit memory formation failed: {e}")
            return None
