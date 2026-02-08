"""
mind/mind.py

Top-level mind layer orchestrator.
Wires pet model, worker model, task router, conversation, memory formation,
and cognitive bridge into a single entry point for processing messages.
"""

from __future__ import annotations

from typing import List, Dict, Any, Optional, TYPE_CHECKING

from config import Config
from event_dispatcher import global_event_dispatcher, Event
from loggers.loggers import BrainLogger
from .models import MindResponse, IntentType
from .pet_model import PetModel
from .worker_model import WorkerModel
from .task_router import TaskRouter
from .conversation import ConversationManager
from .memory_formation import MemoryFormationPipeline
from .prompts import build_pet_system_prompt

if TYPE_CHECKING:
    from internal.modules.cognition.cognitive_bridge import PetCognitiveBridge
    from api_clients import APIManager


class Mind:
    """
    Top-level mind layer orchestrator.
    Entry point for all cognitive functions — chat, task execution,
    memory formation, and introspection.
    """

    def __init__(
        self,
        bridge: PetCognitiveBridge,
        api_manager: APIManager,
    ) -> None:
        self.bridge = bridge
        self.api_manager = api_manager
        self.logger = BrainLogger

        # Sub-components
        self.pet_model = PetModel(api_manager)
        self.worker_model = WorkerModel(api_manager)
        self.task_router = TaskRouter(self.pet_model)
        self.conversation = ConversationManager(
            max_messages=Config.get_exo_max_turns(),
        )
        self.memory_formation = MemoryFormationPipeline(bridge, api_manager)

    async def process_message(self, user_message: str) -> MindResponse:
        """
        Process a user message through the full mind pipeline.

        1. Add user message to conversation
        2. Get state context + retrieve relevant memories
        3. Classify intent (chat vs task)
        4. Generate response (pet model, optionally with worker)
        5. Dispatch events for memory formation + persistence
        6. Return MindResponse
        """
        try:
            # 1. Record user message
            self.conversation.add_message("user", user_message)

            # 2. Get context and memories in parallel
            state_context = await self.bridge.get_state_context()
            memories = await self.bridge.retrieve_memories(
                query=user_message,
                limit=5,
            )

            # 3. Classify intent
            routed = await self.task_router.classify(user_message)

            # 4. Generate response
            memories_used: List[str] = [m.id for m in memories]
            task_result: Optional[str] = None

            if routed.intent == IntentType.TASK and routed.task_spec:
                # Task path: worker executes, pet presents
                task_result = await self.worker_model.execute_task(
                    task_spec=routed.task_spec,
                )
                # Build pet prompt that includes the worker result
                system_prompt = build_pet_system_prompt(state_context, memories)
                messages = self.conversation.get_formatted_messages(
                    system_prompt=system_prompt,
                    limit=10,
                )
                # Append the worker result as context for the pet to present
                messages.append({
                    "role": "user",
                    "content": f"[A helper found this information for you to share naturally: {task_result}]",
                })
                response_text = await self.pet_model.generate_response(messages)
            else:
                # Chat path: direct pet response
                system_prompt = build_pet_system_prompt(state_context, memories)
                messages = self.conversation.get_formatted_messages(
                    system_prompt=system_prompt,
                    limit=20,
                )
                response_text = await self.pet_model.generate_response(messages)

            # 5. Record assistant response
            self.conversation.add_message("assistant", response_text)

            # 6. Dispatch events
            self._dispatch_conversation_events(state_context)

            return MindResponse(
                content=response_text,
                memories_used=memories_used,
                was_task=(routed.intent == IntentType.TASK),
                task_result=task_result,
            )

        except Exception as e:
            self.logger.error(f"Mind.process_message failed: {e}")
            # Return a graceful fallback
            fallback = "I'm having trouble thinking right now. Can you try again?"
            self.conversation.add_message("assistant", fallback)
            return MindResponse(content=fallback)

    def _dispatch_conversation_events(self, state_context: Dict[str, Any]) -> None:
        """Dispatch events for memory formation and state persistence."""
        # Memory formation event
        global_event_dispatcher.dispatch_event(Event(
            "mind:conversation_turn",
            {
                "excerpt": self.conversation.get_recent_excerpt(turns=4),
                "turns_since_formation": self.conversation.turns_since_formation,
                "state_context": state_context,
            },
        ))

        # Conversation persistence event (consumed by StateBridge)
        global_event_dispatcher.dispatch_event(Event(
            "cognitive:context_update",
            {
                "source": "mind",
                "raw_state": self.conversation.to_state(),
                "processed_state": self.conversation.get_recent_excerpt(turns=2),
            },
        ))

    # ---- State Persistence ----

    def get_conversation_state(self) -> List[Dict[str, str]]:
        """Get serializable conversation state for persistence."""
        return self.conversation.to_state()

    def restore_conversation_state(self, state: List[Dict[str, str]]) -> None:
        """Restore conversation from persisted state."""
        self.conversation.from_state(state)
