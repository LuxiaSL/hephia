import time
import hashlib
from typing import Optional, TYPE_CHECKING

from .base_manager import (
    BaseConnectionManager,
    ConnectionOperation
)

from loggers.loggers import MemoryLogger
from ...async_lru_cache import async_lru_cache

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
    Concrete connection manager for BodyMemoryNodes with intelligent caching.

    Uses shared utilities to compute:
      - Emotional similarity (calculate_emotional_similarity)
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

    def _generate_connection_cache_key(self, node_a: BodyMemoryNode, node_b: BodyMemoryNode, operation: ConnectionOperation) -> str:
        """Generate cache key for body connection weight calculations."""
        try:
            # Create ordered pair for consistent caching
            id_a, id_b = str(node_a.node_id), str(node_b.node_id)
            ordered_ids = tuple(sorted([id_a, id_b]))
            
            # Include emotional state signatures for cache invalidation
            try:
                # Use emotional vectors as signature (main component of body similarity)
                emot_a = node_a.raw_state.get("emotional_vectors", [])
                emot_b = node_b.raw_state.get("emotional_vectors", [])
                
                # Create simple signature from first emotional vector if present
                if emot_a and len(emot_a) > 0 and isinstance(emot_a[0], dict):
                    sig_a = f"{emot_a[0].get('valence', 0):.1f}_{emot_a[0].get('arousal', 0):.1f}"
                else:
                    sig_a = "no_emot"
                    
                if emot_b and len(emot_b) > 0 and isinstance(emot_b[0], dict):
                    sig_b = f"{emot_b[0].get('valence', 0):.1f}_{emot_b[0].get('arousal', 0):.1f}"
                else:
                    sig_b = "no_emot"
                    
            except Exception:
                sig_a = sig_b = "emot_error"
            
            # Include operation type for different contexts
            op_type = str(operation.value) if hasattr(operation, 'value') else str(operation)
            
            return f"body_conn:{ordered_ids[0]}:{ordered_ids[1]}:emot:{sig_a}:{sig_b}:op:{op_type}"
            
        except Exception:
            return f"body_fallback:{hash((str(node_a.node_id), str(node_b.node_id), str(operation)))}"

    @async_lru_cache(
        maxsize=1500, 
        ttl=7200,
        key_func=lambda self, node_a, node_b, operation: self._generate_connection_cache_key(node_a, node_b, operation)
    )
    async def _calculate_connection_weight_cached(
        self,
        node_a: BodyMemoryNode,
        node_b: BodyMemoryNode,
        operation: ConnectionOperation
    ) -> float:
        """Cached connection weight calculation for body nodes."""
        return await self._calculate_connection_weight_internal(node_a, node_b, operation)

    async def _calculate_connection_weight_internal(
        self,
        node_a: BodyMemoryNode,
        node_b: BodyMemoryNode,
        operation: ConnectionOperation
    ) -> float:
        """Internal connection weight calculation (the actual computation)."""
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
                behavior_a = node_a.raw_state.get("behavior", {})
                behavior_b = node_b.raw_state.get("behavior", {})
                
                # Compare behavior names if they exist
                if isinstance(behavior_a, dict) and isinstance(behavior_b, dict):
                    name_a = behavior_a.get("name")
                    name_b = behavior_b.get("name")
                    behavior_sim = 1.0 if name_a and name_b and name_a == name_b else 0.0
                else:
                    behavior_sim = 1.0 if behavior_a == behavior_b else 0.0
                    
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
                self.weights.get("emotional", 0.35) * emotional_sim +
                self.weights.get("mood", 0.25) * mood_sim +
                self.weights.get("behavior", 0.20) * behavior_sim +
                self.weights.get("needs", 0.20) * needs_sim
            )

            # Ensure weight is in valid range
            return max(0.0, min(1.0, base_weight))

        except Exception as e:
            self.logger.error(f"Unexpected error in body connection weight calculation: {str(e)}")
            return 0.0

    async def _calculate_connection_weight(
        self,
        node_a: BodyMemoryNode,
        node_b: BodyMemoryNode,
        operation: ConnectionOperation
    ) -> float:
        """Calculate connection weight with intelligent caching."""
        try:
            return await self._calculate_connection_weight_cached(node_a, node_b, operation)
        except Exception as e:
            self.logger.error(f"Cached body connection calculation failed: {e}")
            # Fallback to direct calculation
            return await self._calculate_connection_weight_internal(node_a, node_b, operation)
    
    def _get_node_by_id(self, node_id: str) -> Optional[BodyMemoryNode]:
        """
        Retrieve a BodyMemoryNode from the network by its ID.
        """
        return self.network.nodes.get(node_id)
    
    async def _persist_node_through_network(self, node: BodyMemoryNode, lock_acquired: bool = False) -> None:
        """Persist node through the cognitive network with proper lock handling."""
        try:
            await self.network._persist_node(node, lock_acquired=lock_acquired)
        except Exception as e:
            self.logger.error(f"Failed to persist node {node.node_id}: {e}")
    
    async def transfer_connections(
        self,
        from_node: BodyMemoryNode,
        to_node: BodyMemoryNode,
        transfer_weight: float = 0.9
    ) -> None:
        """
        Transfer connections from one node to another with proper weight adjustment and max_connections enforcement.
        
        Args:
            from_node: Source node losing connections.
            to_node: Target node receiving connections.
            transfer_weight: Dampening factor for transferred connections.
        """
        source_connections = from_node.get_connections_with_weights(min_weight=0.2)
        
        # Calculate new connection weights
        potential_transfers = {}
        for conn_id, weight in source_connections.items():
            if conn_id == to_node.node_id:
                continue  # Skip self-connection
            new_weight = weight * transfer_weight
            if conn_id in to_node.connections:
                potential_transfers[conn_id] = max(new_weight, to_node.connections[conn_id])
            else:
                potential_transfers[conn_id] = new_weight
        
        # Enforce connection limits
        final_transfers = self._enforce_connection_limit(to_node, potential_transfers)
        
        # Apply the transfers that made the cut
        to_node.connections.update(final_transfers)
        to_node.last_connection_update = time.time()
        
        # Clear source connections
        from_node.connections.clear()
        from_node.last_connection_update = time.time()
        
        transferred_count = len(final_transfers)
        total_potential = len(potential_transfers)
        if transferred_count < total_potential:
            self.logger.debug(f"Connection transfer: transferred {transferred_count}/{total_potential} connections due to max_connections limit")