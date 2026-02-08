"""
cognitive_bridge.py

Thin interface between the mind layer and internal state + memory systems.
Replaces the old 758-line CognitiveBridge with direct method delegation.
"""

from __future__ import annotations

import asyncio
import time
from typing import Dict, List, Any, Optional, TYPE_CHECKING

from event_dispatcher import Event, global_event_dispatcher
from mind.models import RetrievedMemory, IntrospectionResult
from loggers.loggers import BrainLogger

if TYPE_CHECKING:
    from internal.internal_context import InternalContext
    from internal.modules.memory.memory_system import MemorySystemOrchestrator


class PetCognitiveBridge:
    """
    Thin bridge between the mind layer and internal state + memory.
    Delegates to MemorySystemOrchestrator and InternalContext without
    owning event loops or significance evaluation state.
    """

    SIGNIFICANCE_THRESHOLD = 0.625

    def __init__(
        self,
        internal_context: InternalContext,
        memory_system: MemorySystemOrchestrator,
    ) -> None:
        self.internal_context = internal_context
        self.memory_system = memory_system
        self.logger = BrainLogger

    # ---- Memory Retrieval ----

    async def retrieve_memories(
        self,
        query: str,
        limit: int = 5,
        threshold: float = 0.0,
    ) -> List[RetrievedMemory]:
        """
        Retrieve relevant cognitive memories, triggering echo effects.

        Returns a list of RetrievedMemory models sorted by relevance.
        """
        try:
            context = await self.internal_context.get_memory_context(is_cognitive=False)
            comparison_state = context.get("raw_state", {})

            nodes, details = await self.memory_system.retrieve_cognitive_memories(
                query=query,
                comparison_state=comparison_state,
                top_k=limit,
                threshold=threshold,
                return_details=True,
                dispatch_echo=True,
            )

            results: List[RetrievedMemory] = []
            for i, node in enumerate(nodes):
                relevance = 0.0
                if i < len(details):
                    detail = details[i]
                    relevance = detail.get("final_score", 0.0) if isinstance(detail, dict) else float(detail)

                results.append(RetrievedMemory(
                    id=str(node.node_id),
                    content=node.text_content,
                    timestamp=node.timestamp,
                    strength=node.strength,
                    relevance=relevance,
                ))

            return results

        except Exception as e:
            self.logger.error(f"Memory retrieval failed: {e}")
            return []

    # ---- Memory Formation ----

    async def form_memory(
        self,
        content: str,
        significance: float,
        source: str = "conversation",
    ) -> Optional[str]:
        """
        Store a cognitive memory if significance meets threshold.

        Returns the node_id if stored, None otherwise.
        """
        try:
            if significance < self.SIGNIFICANCE_THRESHOLD:
                self.logger.debug(
                    f"Memory below threshold ({significance:.3f} < {self.SIGNIFICANCE_THRESHOLD}), skipping"
                )
                return None

            context = await self.internal_context.get_memory_context(is_cognitive=False)

            node_id = await self.memory_system.form_cognitive_memory(
                text_content=content,
                current_state=context,
                formation_source=source,
            )

            if node_id:
                self.logger.info(f"Formed cognitive memory: {node_id}")

            return node_id

        except Exception as e:
            self.logger.error(f"Memory formation failed: {e}")
            return None

    async def evaluate_significance(
        self,
        content: str,
        source: str = "conversation",
    ) -> float:
        """
        Evaluate how significant content is for memory formation.
        Returns a score between 0.0 and 1.0.
        """
        try:
            context = await self.internal_context.get_memory_context(is_cognitive=False)

            score = await self.memory_system.evaluate_memory_significance(
                generated_content=content,
                context=context,
                source_type=source,
            )

            return float(score)

        except Exception as e:
            self.logger.error(f"Significance evaluation failed: {e}")
            return 0.0

    # ---- State Context ----

    async def get_state_context(self) -> Dict[str, Any]:
        """
        Get current internal state for LLM consumption.
        Returns the raw API context dict (mood, needs, behavior, emotional_state).
        """
        try:
            return await self.internal_context.get_api_context(use_memory_emotions=True)
        except Exception as e:
            self.logger.error(f"Failed to get state context: {e}")
            return {}

    # ---- Recent Memories ----

    async def get_recent_memories(self, limit: int = 3) -> List[RetrievedMemory]:
        """Get most recent cognitive memories (no query, no echo)."""
        try:
            nodes = await self.memory_system.get_recent_memories(
                count=limit,
                network_type="cognitive",
                include_ghosted=False,
            )

            return [
                RetrievedMemory(
                    id=str(node.node_id),
                    content=node.text_content,
                    timestamp=node.timestamp,
                    strength=node.strength,
                    relevance=0.0,
                )
                for node in nodes
            ]

        except Exception as e:
            self.logger.error(f"Failed to get recent memories: {e}")
            return []

    # ---- Introspection ----

    async def introspect(self, topic: str) -> Optional[IntrospectionResult]:
        """
        Reflect on memories related to a topic, affecting emotional state.
        Retrieves memories (triggering echoes) and dispatches emotional influence.
        """
        try:
            memories = await self.retrieve_memories(query=topic, limit=5)
            if not memories:
                return None

            # Extract emotional patterns from retrieved memories and
            # dispatch influence event so EmotionalProcessor can absorb it
            emotional_impact: Dict[str, float] = {"valence": 0.0, "arousal": 0.0}
            influence_count = 0

            for mem in memories:
                # The echo dispatch in retrieve_memories already triggers
                # emotional influence via the echo manager. We aggregate
                # the emotional direction for the result.
                emotional_impact["valence"] += mem.relevance * 0.1
                emotional_impact["arousal"] += mem.relevance * 0.05
                influence_count += 1

            if influence_count > 0:
                emotional_impact["valence"] /= influence_count
                emotional_impact["arousal"] /= influence_count

            return IntrospectionResult(
                topic=topic,
                memories=memories,
                emotional_impact=emotional_impact,
            )

        except Exception as e:
            self.logger.error(f"Introspection failed: {e}")
            return None
