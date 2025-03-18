"""
merge_manager.py
"""

from typing import List, Optional, Dict, Any, Union

from loggers.loggers import MemoryLogger

from .synthesis.manager import SynthesisManager  
from .synthesis.conflict import detect_cognitive_conflict
from ..nodes.body_node import BodyMemoryNode
from ..nodes.cognitive_node import CognitiveMemoryNode
from ..networks.body_network import BodyMemoryNetwork
from ..networks.cognitive_network import CognitiveMemoryNetwork
from ..networks.connections.base_manager import BaseConnectionManager, MergeCandidate
from ..metrics.orchestrator import RetrievalMetricsOrchestrator


class MergeManager:
    """
    Coordinates merges for both body & cognitive memory nodes. If a direct merge 
    is straightforward (esp. for body memory), handles them directly.
    For cognitive merges, checks conflicts and optionally routes 
    to SynthesisManager for conflict resolution or advanced merges.
    """

    def __init__(
        self,
        body_network: BodyMemoryNetwork,
        cognitive_network: CognitiveMemoryNetwork,
        metrics_orchestrator: RetrievalMetricsOrchestrator,
        synthesis_manager: Optional[SynthesisManager] = None
    ):
        """
        Initialize MergeManager with references to both memory networks and 
        an optional SynthesisManager for advanced merges or conflict resolution.
        
        Args:
            body_network: BodyMemoryNetwork instance.
            cognitive_network: CognitiveMemoryNetwork instance.
            synthesis_manager: Optional manager for conflict-based merges / synthesis.
            metrics_orchestrator: RetrievalMetricsOrchestrator for computing merge metrics.
        """
        self.body_network = body_network
        self.cognitive_network = cognitive_network
        self.metrics_orchestrator = metrics_orchestrator
        self.synthesis_manager = synthesis_manager

        self.logger = MemoryLogger

    async def handle_weak_node(self, node: Union[BodyMemoryNode, CognitiveMemoryNode]) -> bool:
        """
        Handle a weak node during consolidation by attempting merges or ghost conversion.
        Returns True if node was successfully merged.

        The process:
          1. Try to find viable merge candidates based on connections and metrics.
          2. If good candidates are found, perform merge (with conflict detection for cognitive).
          3. If no merge is possible, convert to ghost state.
          4. Update network state accordingly.
        """
        if not node or not node.node_id:
            raise ValueError("Invalid node provided to handle_weak_node")

        try:
            # Skip if already ghosted
            if node.ghosted:
                return False

            # Get potential merge candidates using the appropriate connection manager.
            connection_manager = self._get_connection_manager(node)
            candidates = await self.find_merge_candidates(node, connection_manager)
            if candidates:
                target_candidate = candidates[0]  # Take the best candidate
                target_node = self._get_node_by_id(target_candidate.node_id)
                if target_node is None or target_node.ghosted:
                    self.logger.warning(f"Target node {target_candidate.node_id} not found or ghosted, skipping merge")
                    return False
                
                if isinstance(node, CognitiveMemoryNode):
                    await self.merge_cognitive_nodes(node, target_node)
                else:
                    # For body nodes, perform a direct merge.
                    await self.merge_body_nodes(node, target_node)
                return True
            else:
                return False

        except Exception as e:
            self.logger.error(f"Failed to handle weak node {node.node_id}: {e}")
            raise

    async def find_merge_candidates(
        self,
        node: Union[BodyMemoryNode, CognitiveMemoryNode],
        connection_manager: BaseConnectionManager
    ) -> List[MergeCandidate]:
        """
        Find potential merge candidates based on connection weights.
        """
        candidates = []
        threshold = 0.7  # High connection weight threshold
        health_data = connection_manager.connection_health.get(node.node_id, {})

        for other_id, weight in node.connections.items():
            if weight < threshold:
                continue

            health = health_data.get(other_id)
            if not health:
                continue

            # Check for stable strong connection
            if (len(health.strength_history) >= 3 and
                all(s >= threshold for s in health.strength_history[-3:])):
                candidate_node = self._get_node_by_id(other_id)
                if candidate_node:
                    candidates.append(MergeCandidate(
                        node_id=other_id,
                        weight=weight,
                        health=health,
                    ))
        return sorted(candidates, key=lambda x: x.weight, reverse=True)

    def _get_connection_manager(self, node: Union[BodyMemoryNode, CognitiveMemoryNode]) -> BaseConnectionManager:
        """
        Helper to retrieve the appropriate connection manager based on node type.
        """
        if isinstance(node, BodyMemoryNode):
            return self.body_network.connection_manager
        else:
            return self.cognitive_network.connection_manager

    def _get_node_by_id(self, node_id: str) -> Union[BodyMemoryNode, CognitiveMemoryNode, None]:
        """
        Retrieve a node by ID from the appropriate network.
        """
        node = self.cognitive_network.nodes.get(node_id)
        if node is not None:
            return node
        return self.body_network.nodes.get(node_id)

    # -------------------------------------------------------------------------
    # Body Memory Merges
    # -------------------------------------------------------------------------
    async def merge_body_nodes(
        self,
        child: BodyMemoryNode,
        parent: BodyMemoryNode
    ) -> None:
        """
        Merge one BodyMemoryNode into another using the network's existing merge_nodes method.
        
        Args:
            child: Node to be merged.
            parent: Node that absorbs the child node.
        """
        if not child.node_id or not parent.node_id:
            self.logger.error("[MergeManager] merge_body_nodes called with invalid node IDs.")
            return

        if child.node_id == parent.node_id:
            self.logger.warning("[MergeManager] Attempted to merge a BodyNode with itself.")
            return

        self.logger.info(f"[MergeManager] Merging BodyMemoryNode {child.node_id} into {parent.node_id}")
        await self._handle_body_merge(child, parent)

    async def _handle_body_merge(self, child: BodyMemoryNode, parent: BodyMemoryNode) -> None:
        """
        Handle the actual merging of two body memory nodes.
        Ensures proper connection transfer, state preservation, and network updates.
        
        Args:
            child: Node to be merged.
            parent: Node that absorbs the child node.
        """
        if not child or not parent or not child.node_id or not parent.node_id:
            self.logger.error("Invalid nodes provided for body merge")
            return
        try:
            # Update node state.
            child.merge_into_parent(parent)

            # Transfer connections using the body network's connection manager.
            await self.body_network.connection_manager.transfer_connections(
                from_node=child,
                to_node=parent,
                transfer_weight=0.9  # Preserve 90% of connection strength.
            )

            # Mark the child as ghosted and record its parent.
            child.ghosted = True
            child.parent_node_id = parent.node_id

            # Update both nodes in the network.
            await self.body_network.update_node(parent)
            await self.body_network.update_node(child)

            self.logger.info(f"[MergeManager] Successfully merged BodyNode {child.node_id} into {parent.node_id}")
        except Exception as e:
            self.logger.error(f"[MergeManager] Body merge failed: {e}")
            raise

    # -------------------------------------------------------------------------
    # Cognitive Memory Merges
    # -------------------------------------------------------------------------
    async def merge_cognitive_nodes(self, child: CognitiveMemoryNode, parent: CognitiveMemoryNode) -> bool:
        """
        Merge one CognitiveMemoryNode into another. If a conflict is detected,
        a synthesis node may be created instead.
        
        Args:
            child: Node to be merged.
            parent: Node that absorbs the child node.
        
        Returns:
            True if merge was successful; False if conflict resolution was triggered.
        """
        if not child.node_id or not parent.node_id:
            self.logger.error("[MergeManager] merge_cognitive_nodes called with invalid node IDs.")
            return False

        if child.node_id == parent.node_id:
            self.logger.warning("[MergeManager] Attempted to merge a CognitiveNode with itself.")
            return False

        self.logger.info(f"[MergeManager] Attempting to merge CognitiveNode {child.node_id} into {parent.node_id}")

        # Use the pairwise comparison to compute dissonance between the two nodes.
        dissonance_metrics = await self.metrics_orchestrator.compare_nodes(child, parent)

        # Optionally, you can log or inspect dissonance_metrics for debugging.
        self.logger.info(f"[MergeManager] Pairwise dissonance metrics: {dissonance_metrics}")

        # Based on the dissonance, decide if there's a conflict.
        conflict_data = detect_cognitive_conflict(child, parent, {'component_metrics': dissonance_metrics}, self.metrics_orchestrator)

        if conflict_data.get('has_conflicts') and self.synthesis_manager:
            return await self._handle_conflict_merge(child, parent, conflict_data)
        else:
            await self._handle_reg_cognitive_merge(child, parent)
            return True

    async def _handle_reg_cognitive_merge(self, child: CognitiveMemoryNode, parent: CognitiveMemoryNode) -> None:
        """
        Handle a regular cognitive memory merge without conflicts.
        
        Args:
            child: Node to be merged.
            parent: Node that absorbs the child node.
        """
        self.logger.info(f"[MergeManager] Merging CognitiveMemoryNode {child.node_id} into {parent.node_id}")
        parent.strength += child.strength * 0.5
        child.ghosted = True
        child.parent_node_id = parent.node_id
        await self.cognitive_network.update_node(parent)
        await self.cognitive_network.update_node(child)

    # -------------------------------------------------------------------------
    # Conflict / Synthesis Handling
    # -------------------------------------------------------------------------
    async def _handle_conflict_merge(
        self,
        child: CognitiveMemoryNode, 
        parent: CognitiveMemoryNode,
        conflict_data: Dict[str, Any]
    ) -> bool:
        """
        Handle cognitive memory conflicts by creating a synthesis node.
        
        Args:
            child: Node to be merged.
            parent: Node that would absorb the child.
            conflict_data: Conflict analysis details.
        
        Returns:
            True if conflict was successfully resolved.
        """
        self.logger.info(f"[MergeManager] Conflict merging {child.node_id} -> {parent.node_id}")
        try:
            # Adjust strengths proportionally.
            child_contrib = child.strength * 0.3
            parent_contrib = parent.strength * 0.3
            child.strength -= child_contrib
            parent.strength -= parent_contrib

            # Update nodes with new strengths.
            await self.cognitive_network.update_node(child)
            await self.cognitive_network.update_node(parent)

            # Create synthesis node via the synthesis manager.
            synthesis_node_id = await self.synthesis_manager.handle_conflict_synthesis(
                conflict_data=conflict_data,
                child=child,
                parent=parent,
                synthesis_content="",  # Insert LLM-provided synthesis content as needed.
                synthesis_embedding=[],  # Insert pre-calculated embedding as needed.
                additional_strength=(child_contrib + parent_contrib)
            )

            self.logger.info(f"[MergeManager] Created synthesis node {synthesis_node_id}")
            return True
        except Exception as e:
            self.logger.error(f"[MergeManager] Conflict merge failed: {e}")
            return False
