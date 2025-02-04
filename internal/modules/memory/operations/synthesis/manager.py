"""
manager.py

Implements a concrete SynthesisManager (ISynthesisHandler) that handles conflict-based merges
and the creation of new 'synthesis' nodes for cognitive memory. Integrates with your
CognitiveDBManager or CognitiveMemoryNetwork to persist changes.
"""

import time
from typing import Dict, Any, Optional, List, Union

from .base import ISynthesisHandler

from ...nodes.cognitive_node import CognitiveMemoryNode
from ...networks.cognitive_network import CognitiveMemoryNetwork
from ...db.managers import CognitiveDBManager, SynthesisRelationManager
from ...db.schema import SYNTHESIS_TYPES

from loggers.loggers import MemoryLogger


class SynthesisManager(ISynthesisHandler):
    """
    Handles all synthesis operations including conflict resolution,
    resurrection, and merges.
    """

    def __init__(
        self,
        cognitive_network: CognitiveMemoryNetwork,
        db_manager: CognitiveDBManager,
        relation_manager: SynthesisRelationManager,
        metrics_orchestrator: Any  # Type properly based on metrics
    ):
        self.network = cognitive_network
        self.db = db_manager
        self.relations = relation_manager
        self.metrics = metrics_orchestrator
        self.logger = MemoryLogger()

    async def handle_conflict_synthesis(
        self,
        conflict_data: Dict[str, Any],
        child: CognitiveMemoryNode,
        parent: CognitiveMemoryNode,
        synthesis_content: str,  # From LLM resolution
        synthesis_embedding: List[float],  # Pre-calculated
        additional_strength: float = 0.0
    ) -> str:
        """
        Creates a new synthesis node from conflicting nodes.
        Uses LLM-provided synthesis content and pre-calculated embedding.
        Returns new node ID.
        """
        # Create new node with blended state
        new_node = await self._create_synthesis_node(
            child,
            parent,
            conflict_data,
            synthesis_content,
            synthesis_embedding,
            additional_strength
        )

        # Record synthesis relations for the child
        await self.relations.add_relation(
            synthesis_node_id=int(new_node.node_id),
            constituent_node_id=int(child.node_id),
            relationship_type=SYNTHESIS_TYPES.get('CONFLICT', 'conflict'),
            metadata={
                "timestamp": time.time(),
                "conflict_data": conflict_data,
                "synthesis_type": "conflict_resolution"
            }
        )

        # Also relate to parent
        await self.relations.add_relation(
            synthesis_node_id=int(new_node.node_id),
            constituent_node_id=int(parent.node_id),
            relationship_type=SYNTHESIS_TYPES.get('CONFLICT', 'conflict'),
            metadata={
                "timestamp": time.time(),
                "conflict_data": conflict_data,
                "synthesis_type": "conflict_resolution"
            }
        )

        return new_node.node_id

    async def handle_resurrection(
        self,
        node: CognitiveMemoryNode,
        parent_id: int
    ) -> None:
        """Handle node resurrection and record relation."""
        await self.relations.add_relation(
            synthesis_node_id=int(node.node_id),
            constituent_node_id=parent_id,
            relationship_type=SYNTHESIS_TYPES.get('RESURRECTION', 'resurrection'),
            metadata={"timestamp": time.time()}
        )

    async def _create_synthesis_node(
        self,
        nodeA: CognitiveMemoryNode,
        nodeB: CognitiveMemoryNode,
        conflict_data: Dict[str, Any],
        synthesis_content: str,  # From LLM
        synthesis_embedding: List[float],  # Pre-calculated embedding
        additional_strength: float
    ) -> CognitiveMemoryNode:
        """
        Create new node from synthesis of others.
        Uses provided synthesis content and embedding from conflict resolution.
        """
        # Create a new node with blended raw and processed states.
        new_node = CognitiveMemoryNode(
            timestamp=time.time(),
            text_content=synthesis_content,
            embedding=synthesis_embedding,
            raw_state=self._blend_states(nodeA.raw_state, nodeB.raw_state, is_raw=True),
            processed_state=self._blend_states(nodeA.processed_state, nodeB.processed_state, is_raw=False),
            strength=min(1.0, 0.3 + additional_strength),
            formation_source="conflict_synthesis"
        )

        # Persist the new synthesis node.
        new_id = await self.db.create_node(new_node)
        new_node.node_id = str(new_id)
        return new_node
    
    async def handle_synthesis_complete(
        self,
        synthesis_node_id: str,
        constituents: list
    ) -> None:
        """
        Optional post-synthesis step.
        In this implementation, no further action is required.
        """
        # Optionally log or perform a simple confirmation action.
        self.logger.info(f"Synthesis complete for node {synthesis_node_id} with constituents {constituents}.")
        # No further action required.

    # -------------------------------------------------------------------------
    # Internal Utility Methods
    # -------------------------------------------------------------------------
    def _blend_states(self, sA: Dict[str, Any], sB: Dict[str, Any], is_raw: bool) -> Dict[str, Any]:
        from ...nodes.node_utils import blend_states
        return blend_states(sA, sB, weights=None, is_raw=is_raw)

    def _combine_embeddings(self, embA: List[float], embB: List[float]) -> List[float]:
        """
        Compute a weighted average or partial combination of two embeddings.
        Assumes both embeddings have the same dimension.
        """
        if not embA:
            return embB
        if not embB:
            return embA
        return [(a + b) / 2.0 for a, b in zip(embA, embB)]
