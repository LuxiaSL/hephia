import time
import asyncio
from typing import Optional, TYPE_CHECKING

from .base_manager import (
    BaseConnectionManager,
    ConnectionOperation
)

from loggers.loggers import MemoryLogger

if TYPE_CHECKING:
    from ..body_network import BodyMemoryNetwork

from ...nodes.body_node import BodyMemoryNode
from ...nodes.node_utils import (
    calculate_emotional_similarity,
    calculate_mood_similarity,
    calculate_needs_similarity,
)


class BodyConnectionManager(BaseConnectionManager[BodyMemoryNode]):
    """
    Concrete connection manager for BodyMemoryNodes.

    Uses shared utilities to compute:
class BodyConnectionManager(BaseConnectionManager['BodyMemoryNode']):
      - Mood similarity (calculate_mood_similarity)
      - Needs similarity (calculate_needs_similarity)
      - Behavior similarity (via a simple equality check)
      
    The final connection weight is computed as a weighted sum:
        weight = 0.35 * emotional_sim +
                 0.25 * mood_sim +
                 0.20 * behavior_sim +
                 0.20 * needs_sim
    """
    def __init__(self, network: 'BodyMemoryNetwork', metrics_orchestrator, config=None):
        super().__init__(metrics_orchestrator, config)
        self.network = network
        self.weights = {
            'emotional': 0.35,
            'mood': 0.25,
            'behavior': 0.20,
            'needs': 0.20
        }
        self.logger = MemoryLogger
    
    async def _calculate_connection_weight(
        self,
        node_a: BodyMemoryNode,
        node_b: BodyMemoryNode,
        operation: ConnectionOperation
    ) -> float:
        """Ensure order-independent weight calculation."""
        try:
            # Validate nodes and raw states exist
            if not node_a or not node_b:
                self.logger.error("Invalid nodes provided for connection weight calculation")
                return 0.0

            if not hasattr(node_a, 'raw_state') or not hasattr(node_b, 'raw_state'):
                self.logger.error(f"Missing raw_state - Node A: {bool(getattr(node_a, 'raw_state', None))}, Node B: {bool(getattr(node_b, 'raw_state', None))}")
                return 0.0

            # Calculate bi-directional similarities and use the average
            try:
                emotional_sim_ab = calculate_emotional_similarity(
                    node_a.raw_state.get("emotional_vectors", []),
                    node_b.raw_state.get("emotional_vectors", [])
                )
                emotional_sim_ba = calculate_emotional_similarity(
                    node_b.raw_state.get("emotional_vectors", []),
                    node_a.raw_state.get("emotional_vectors", [])
                )
                emotional_sim = (emotional_sim_ab + emotional_sim_ba) / 2
            except Exception as e:
                self.logger.error(f"Error calculating emotional similarity: {e}")
                emotional_sim = 0.0

            try:
                mood_sim = calculate_mood_similarity(
                    node_a.raw_state.get("mood", {}),
                    node_b.raw_state.get("mood", {})
                )
            except Exception as e:
                self.logger.error(f"Error calculating mood similarity: {e}")
                mood_sim = 0.0

            try:
                behavior_a = node_a.raw_state.get("behavior")
                behavior_b = node_b.raw_state.get("behavior")
                behavior_sim = 1.0 if behavior_a and behavior_b and behavior_a == behavior_b else 0.0
            except Exception as e:
                self.logger.error(f"Error comparing behaviors: {e}")
                behavior_sim = 0.0

            try:
                needs_sim = calculate_needs_similarity(
                    node_a.raw_state.get("needs", {}),
                    node_b.raw_state.get("needs", {})
                )
            except Exception as e:
                self.logger.error(f"Error calculating needs similarity: {e}")
                needs_sim = 0.0

            # Calculate final weight
            base_weight = (
                self.weights.get("emotional", 0.25) * emotional_sim +
                self.weights.get("mood", 0.25) * mood_sim +
                self.weights.get("behavior", 0.25) * behavior_sim +
                self.weights.get("needs", 0.25) * needs_sim
            )

            # Ensure weight is in valid range
            return max(0.0, min(1.0, base_weight))

        except Exception as e:
            self.logger.error(f"Unexpected error in connection weight calculation: {str(e)}")
            return 0.0
    
    def _get_node_by_id(self, node_id: str) -> Optional[BodyMemoryNode]:
        """
        Retrieve a BodyMemoryNode from the network by its ID.
        """
        return self.network.nodes.get(node_id)
    
    async def transfer_connections(
        self,
        from_node: BodyMemoryNode,
        to_node: BodyMemoryNode,
        transfer_weight: float = 0.9
    ) -> None:
        """
        Transfer connections from one node to another with proper weight adjustment.
        
        Args:
            from_node: Source node losing connections.
            to_node: Target node receiving connections.
            transfer_weight: Dampening factor for transferred connections.
        """
        source_connections = from_node.get_connected_nodes(min_weight=0.2)
        for conn_id, weight in source_connections.items():
            if conn_id == to_node.node_id:
                continue  # Skip self-connection
                
            new_weight = weight * transfer_weight
            if conn_id in to_node.connections:
                to_node.connections[conn_id] = max(new_weight, to_node.connections[conn_id])
            else:
                to_node.connections[conn_id] = new_weight
        to_node.last_connection_update = time.time()
        from_node.connections.clear()
        from_node.last_connection_update = time.time()

    