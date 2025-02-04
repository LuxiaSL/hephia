"""
ghost_manager.py

Centralized manager for handling ghost transitions, revivals, and final pruning
across both BodyMemoryNetwork and CognitiveMemoryNetwork.

Key Responsibilities:
- Periodic "ghost cycle" to evaluate nodes in both body & cognitive networks:
  1. Detect weak nodes (strength below ghost_threshold) and ghost them if not already.
  2. Merge a weak node into a connected target if conditions allow (body or cognitive).
  3. Prune any ghosted node with strength below final_prune_threshold.
  4. Optionally revive a ghosted node if its strength rises above revive_threshold.
- Consolidate or preserve state as needed (e.g., storing ghost states in a parent node).
"""

import time
from typing import Optional, Union, Dict

from loggers.loggers import MemoryLogger

from ..nodes.body_node import BodyMemoryNode
from ..nodes.cognitive_node import CognitiveMemoryNode
from ..networks.body_network import BodyMemoryNetwork
from ..networks.cognitive_network import CognitiveMemoryNetwork


class GhostManager:
    def __init__(
        self,
        body_network: BodyMemoryNetwork,
        cognitive_network: CognitiveMemoryNetwork,
        ghost_threshold: float = 0.1,
        final_prune_threshold: float = 0.05,
        revive_threshold: float = 0.2
    ):
        """
        Initialize GhostManager.

        Args:
            body_network: Instance of BodyMemoryNetwork.
            cognitive_network: Instance of CognitiveMemoryNetwork.
            ghost_threshold: Strength below which an active node is ghosted.
            final_prune_threshold: Strength below which a ghosted node is fully removed.
            revive_threshold: Strength above which a ghosted node is reactivated.
        """
        self.body_network = body_network
        self.cognitive_network = cognitive_network

        self.ghost_threshold = ghost_threshold
        self.final_prune_threshold = final_prune_threshold
        self.revive_threshold = revive_threshold

        self.logger = MemoryLogger()

    async def run_ghost_cycle(self) -> None:
        """
        Evaluate every node in body and cognitive networks, applying ghost/merge/prune/revive logic.
        This can be called periodically by the top-level memory system.
        """
        # Process body nodes
        body_ids = list(self.body_network.nodes.keys())
        for node_id in body_ids:
            node = self.body_network.nodes.get(node_id)
            if node is not None:
                await self._evaluate_body_node(node)

        # Process cognitive nodes
        cog_ids = list(self.cognitive_network.nodes.keys())
        for node_id in cog_ids:
            node = self.cognitive_network.nodes.get(node_id)
            if node is not None:
                await self._evaluate_cognitive_node(node)
    
    async def consider_ghosting(self, node: Union[BodyMemoryNode, CognitiveMemoryNode]) -> bool:
        """
        Check if a node should be ghosted based on its strength.
        Returns True if node was ghosted, False otherwise.
        """
        if node.ghosted:
            return False
        if node.strength >= self.ghost_threshold:
            return False
        if isinstance(node, BodyMemoryNode):
            await self._ghost_body_node(node)
        else:
            await self._ghost_cognitive_node(node)
        return True

    # -------------------------------------------------------------------------
    # Body Node Evaluation
    # -------------------------------------------------------------------------
    async def _evaluate_body_node(self, node: BodyMemoryNode) -> None:
        """
        Evaluate a single BodyMemoryNode for ghost transitions, merges, final prune, or revival.
        """
        if not node.node_id:
            return

        if node.ghosted:
            await self._maybe_prune_node(node, network="body")
            await self._maybe_revive_node(node, network="body")
            return

        if node.strength < self.ghost_threshold:
            await self._ghost_body_node(node)

    async def _ghost_body_node(self, node: BodyMemoryNode) -> None:
        """
        Mark a body node as ghosted.
        """
        if node.ghosted:
            return
        node.ghosted = True
        self.logger.info(
            f"[GhostManager] BodyNode {node.node_id} ghosted (strength={node.strength:.2f})."
        )
        await self.body_network.update_node(node)

    # -------------------------------------------------------------------------
    # Cognitive Node Evaluation
    # -------------------------------------------------------------------------
    async def _evaluate_cognitive_node(self, node: CognitiveMemoryNode) -> None:
        """
        Evaluate a single CognitiveMemoryNode for ghost transitions, merges, final prune, or revival.
        """
        if not node.node_id:
            return

        if node.ghosted:
            await self._maybe_prune_node(node, network="cognitive")
            await self._maybe_revive_node(node, network="cognitive")
            return

        if node.strength < self.ghost_threshold:
            # thoughts on implementing a merge step here over a ghost but it's hard to inject and need to think about intent
            await self._ghost_cognitive_node(node)

    async def _ghost_cognitive_node(self, node: CognitiveMemoryNode) -> None:
        """
        Mark a cognitive node as ghosted.
        """
        if node.ghosted:
            return
        node.ghosted = True
        self.logger.info(
            f"[GhostManager] CognitiveNode {node.node_id} ghosted (strength={node.strength:.2f})."
        )
        await self.cognitive_network.update_node(node)

    # -------------------------------------------------------------------------
    # Final Pruning for Ghosted Nodes
    # -------------------------------------------------------------------------
    async def handle_final_pruning(
        self,
        node: Union[BodyMemoryNode, CognitiveMemoryNode],
        network: str
    ) -> None:
        """
        Permanently remove a ghosted node if its strength is below final_prune_threshold.
        """
        if not node.ghosted or node.strength > self.final_prune_threshold:
            return

        self.logger.info(
            f"[GhostManager] Node {node.node_id} in {network} is below final prune threshold; preparing removal."
        )

        if node.parent_node_id:
            await self._store_ghost_state_in_parent(node, network)

        if network == "body":
            await self.body_network.remove_node(node.node_id)
        else:
            await self.cognitive_network.remove_node(node.node_id)
            
        self.logger.debug(
            f"[GhostManager] Completed removal of {network} node {node.node_id} during consolidation."
        )

    async def _maybe_prune_node(
        self,
        node: Union[BodyMemoryNode, CognitiveMemoryNode],
        network: str
    ) -> None:
        """
        Check if a ghosted node should be permanently removed (final prune).
        """
        if not node.ghosted or node.strength > self.final_prune_threshold:
            return

        self.logger.info(
            f"[GhostManager] Node {node.node_id} in {network} is below final prune threshold; removing."
        )
        if node.parent_node_id:
            await self._store_ghost_state_in_parent(node, network)
        if network == "body":
            await self.body_network.remove_node(node.node_id)
        else:
            await self.cognitive_network.remove_node(node.node_id)

    async def _store_ghost_state_in_parent(
        self,
        node: Union[BodyMemoryNode, CognitiveMemoryNode],
        network: str
    ) -> None:
        """
        Store a final ghost snapshot in the node's parent.
        """
        if network == "body":
            parent = self.body_network.nodes.get(str(node.parent_node_id))
        else:
            parent = self.cognitive_network.nodes.get(str(node.parent_node_id))

        if parent:
            ghost_state = {
                "timestamp": node.timestamp,
                "raw_state": node.raw_state,
                "processed_state": node.processed_state,
                "strength": node.strength,
                "derived_from_node": node.node_id,
            }
            if isinstance(node, CognitiveMemoryNode):
                ghost_state["text_content"] = node.text_content
                ghost_state["embedding"] = node.embedding
                ghost_state["echo_dampening"] = node.echo_dampening
                ghost_state["formation_source"] = node.formation_source

            parent.ghost_states.append(ghost_state)
            if network == "body":
                await self.body_network.update_node(parent)
            else:
                await self.cognitive_network.update_node(parent)

    # -------------------------------------------------------------------------
    # Revival of Ghosted Nodes
    # -------------------------------------------------------------------------
    async def _maybe_revive_node(
        self,
        node: Union[BodyMemoryNode, CognitiveMemoryNode],
        network: str
    ) -> None:
        """
        Revive a ghosted node if its strength exceeds the revive threshold.
        """
        if not node.ghosted or node.strength < self.revive_threshold:
            return

        self.logger.info(
            f"[GhostManager] Node {node.node_id} in {network} revived from ghost (strength={node.strength:.2f})."
        )
        node.ghosted = False
        if network == "body":
            await self.body_network.update_node(node)
        else:
            await self.cognitive_network.update_node(node)
