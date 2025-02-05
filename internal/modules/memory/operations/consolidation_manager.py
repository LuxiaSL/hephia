"""
internal\\modules\\memory\\operations\\consolidation_manager.py

Manages network consolidation operations including:
- Activity pattern tracking & analysis
- Consolidation timing & triggers
- Coordination with merge/ghost managers 
- Network health maintenance
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import time

from ..nodes.base_node import BaseMemoryNode
from ..operations.ghost_manager import GhostManager
from ..operations.merge_manager import MergeManager

from loggers.loggers import MemoryLogger


@dataclass
class ConsolidationConfig:
    """Configuration for consolidation behavior."""
    activity_window: float = 3600  # 1 hour
    min_activity_ratio: float = 0.7  # For consolidation triggers
    strength_threshold: float = 0.3  # When to consider consolidation
    max_nodes_ratio: float = 1.2     # Ratio to ideal node count
    consolidation_cooldown: float = 300  # 5 minutes between consolidations
    min_active_nodes: int = 5  # Minimum nodes to maintain


class ConsolidationManager:
    """
    Manages network consolidation and health through activity monitoring
    and coordinated operations with other managers.
    """

    def __init__(
        self,
        merge_manager: MergeManager,
        ghost_manager: GhostManager,
        config: Optional[ConsolidationConfig] = None
    ):
        self.merge_manager = merge_manager
        self.ghost_manager = ghost_manager
        self.config = config or ConsolidationConfig()

        # Activity tracking
        self.activity_patterns: Dict[str, Dict[str, Any]] = {
            "cognitive": {"recent_avg": 0, "last_consolidation": 0},
            "body": {"recent_avg": 0, "last_consolidation": 0}
        }

        self.logger = MemoryLogger

    async def check_consolidation_needed(
        self,
        network_type: str,
        current_nodes: List[BaseMemoryNode]
    ) -> bool:
        """
        Determine if consolidation should be triggered based on:
        - Current activity levels
        - Network size
        - Time since last consolidation
        - Node strength distribution
        """
        if len(current_nodes) < self.config.min_active_nodes:
            return False

        pattern = self.activity_patterns[network_type]
        current_time = time.time()

        # Check cooldown period
        if current_time - pattern["last_consolidation"] < self.config.consolidation_cooldown:
            return False

        # Count active nodes (based on last_accessed within activity_window)
        active_count = len([
            n for n in current_nodes 
            if n.last_accessed and current_time - n.last_accessed < self.config.activity_window
        ])

        activity_ratio = (active_count / pattern["recent_avg"]) if pattern["recent_avg"] > 0 else 1.0

        # Determine how many nodes are weak (strength below threshold)
        weak_nodes = [n for n in current_nodes if n.strength < self.config.strength_threshold]
        strength_ratio = len(weak_nodes) / len(current_nodes)

        return (
            activity_ratio < self.config.min_activity_ratio or
            strength_ratio > 0.3  # If more than 30% of nodes are weak
        )

    async def update_activity_patterns(
        self,
        network_type: str,
        current_nodes: List[BaseMemoryNode]
    ) -> None:
        """Update rolling average of network activity."""
        current_time = time.time()
        active_count = len([
            n for n in current_nodes 
            if n.last_accessed and current_time - n.last_accessed < self.config.activity_window
        ])

        pattern = self.activity_patterns[network_type]
        # Update using exponential moving average
        pattern["recent_avg"] = pattern.get("recent_avg", active_count) * 0.9 + active_count * 0.1

    async def run_consolidation_cycle(
        self,
        network_type: str,
        nodes: List[BaseMemoryNode]
    ) -> None:
        """
        Run a complete consolidation cycle:
        1. Update activity tracking.
        2. Check if consolidation is needed.
        3. Coordinate merge/ghost operations.
        4. Verify and update network health.
        """
        try:
            # Update activity tracking
            await self.update_activity_patterns(network_type, nodes)

            # Check if consolidation is needed
            if not await self.check_consolidation_needed(network_type, nodes):
                return

            self.logger.info(f"Starting consolidation cycle for {network_type} network")
            self.activity_patterns[network_type]["last_consolidation"] = time.time()

            # Process nodes in order of increasing strength (weak nodes first)
            weak_nodes = sorted(
                [n for n in nodes if n.strength < self.config.strength_threshold],
                key=lambda x: x.strength
            )

            for node in weak_nodes:
                # Attempt merging first via merge manager
                if await self.merge_manager.handle_weak_node(node):
                    continue

                # If merging is not possible, consider ghosting via ghost manager
                await self.ghost_manager.consider_ghosting(node)

            # Finally, verify overall network health
            await self._verify_network_health(network_type, nodes)

        except Exception as e:
            self.logger.error(f"Consolidation cycle failed: {e}")

    async def _verify_network_health(
        self,
        network_type: str,
        current_nodes: List[BaseMemoryNode]
    ) -> None:
        """
        Verify and maintain network health:
        - Ensure node count is within a reasonable range.
        - Check that strength distribution is balanced.
        - Validate connection density.
        - Confirm that the ghost/active ratio is healthy.
        """
        active_nodes = [n for n in current_nodes if not n.ghosted]
        ghost_nodes = [n for n in current_nodes if n.ghosted]

        # Check ghost ratio
        ghost_ratio = len(ghost_nodes) / (len(current_nodes) or 1)
        if ghost_ratio > 0.5:  # If more than 50% of nodes are ghosted, prune excess.
            await self._prune_excess_ghosts(ghost_nodes)

        # Check node strength distribution
        strengths = [n.strength for n in active_nodes]
        if strengths:
            avg_strength = sum(strengths) / len(strengths)
            if avg_strength < 0.4:  # If average strength is too low, boost network strength.
                await self._boost_network_strength(active_nodes)

    async def _prune_excess_ghosts(
        self,
        ghost_nodes: List[BaseMemoryNode]
    ) -> None:
        """Remove oldest or weakest ghost nodes to maintain a healthy ratio."""
        # Sort ghosts by a combination of age and strength (lower strength and older preferred for removal)
        sorted_ghosts = sorted(
            ghost_nodes,
            key=lambda x: (x.strength * 0.3 + (1 / (time.time() - x.timestamp)) * 0.7)
        )
        # Retain top 30% of ghosts and prune the rest
        num_to_keep = int(len(sorted_ghosts) * 0.3)
        to_prune = sorted_ghosts[num_to_keep:]
        for node in to_prune:
            await self.ghost_manager.handle_final_pruning(node)

    async def _boost_network_strength(
        self,
        active_nodes: List[BaseMemoryNode]
    ) -> None:
        """
        Apply a small strength boost to maintain network health when average strength is low.
        """
        boost = 0.1
        for node in active_nodes:
            node.strength = min(1.0, node.strength + boost)
