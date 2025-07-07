import time
import asyncio

from asyncio import TimeoutError as AsyncTimeoutError
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from .base_manager import (
    BaseConnectionManager,
    ConnectionOperation
)

if TYPE_CHECKING:
    from ..cognitive_network import CognitiveMemoryNetwork

from ...async_lru_cache import async_lru_cache
from ...nodes.cognitive_node import CognitiveMemoryNode
from ...nodes.node_utils import (
    calculate_emotional_similarity,
    calculate_needs_similarity,
    calculate_temporal_weight
)

class CognitiveConnectionManager(BaseConnectionManager[CognitiveMemoryNode]):
    """
    Concrete connection manager for CognitiveMemoryNodes.

    For cognitive memory we use a four-part weighting:
      - Semantic similarity (0.4): computed via the metrics orchestrator.
      - Emotional resonance (0.2): computed with calculate_emotional_similarity.
      - State alignment (0.2): here we use needs similarity as a proxy.
      - Temporal proximity (0.2): computed via calculate_temporal_weight.
    """
    def __init__(self, network: 'CognitiveMemoryNetwork', metrics_orchestrator, config=None):
        super().__init__(metrics_orchestrator, config)
        self.network = network
        self.weights = {
            'semantic': 0.45,
            'emotional': 0.2,
            'state': 0.2,
            'temporal': 0.15
        }
        self._calc_semaphore = asyncio.Semaphore(20)

    def _generate_connection_cache_key(self, node_a: CognitiveMemoryNode, node_b: CognitiveMemoryNode) -> str:
        """Generate cache key for connection weight calculations."""
        try:
            # Create ordered pair for consistent caching regardless of call order
            id_a, id_b = str(node_a.node_id), str(node_b.node_id)
            ordered_ids = tuple(sorted([id_a, id_b]))
            
            # Include node strengths and timestamps in key to invalidate when nodes change significantly
            strength_a = round(node_a.strength, 3)  # Round to avoid minor differences
            strength_b = round(node_b.strength, 3)
            
            # Use recent timestamps to ensure cache invalidation for major changes
            time_a = int(node_a.timestamp / 3600)  # Hour-level granularity  
            time_b = int(node_b.timestamp / 3600)
            
            return f"conn:{ordered_ids[0]}:{ordered_ids[1]}:s:{strength_a}:{strength_b}:t:{time_a}:{time_b}"
            
        except Exception:
            # Fallback to basic hash
            return f"fallback:{hash((str(node_a.node_id), str(node_b.node_id)))}"

    @async_lru_cache(
        maxsize=3000, 
        ttl=7200, 
        key_func=lambda self, node_a, node_b: self._generate_connection_cache_key(node_a, node_b)
    )
    async def _calculate_connection_weight_cached(
        self, 
        node_a: CognitiveMemoryNode,
        node_b: CognitiveMemoryNode
    ) -> float:
        """
        Cached connection weight calculation using decorator.
        """
        return await self._calculate_connection_weight_internal(
            node_a_id=str(node_a.node_id),
            node_b_id=str(node_b.node_id), 
            node_a_embedding=node_a.embedding,
            node_b_embedding=node_b.embedding,
            node_a_raw_state=node_a.raw_state,
            node_b_raw_state=node_b.raw_state,
            node_a_text=node_a.text_content,
            node_b_text=node_b.text_content,
            node_a_timestamp=node_a.timestamp,
            node_b_timestamp=node_b.timestamp
        )

    async def _calculate_connection_weight(self, node_a: CognitiveMemoryNode, node_b: CognitiveMemoryNode, operation: ConnectionOperation) -> float:
        """
        Calculate bidirectional connection weight with intelligent caching.
        Uses custom key generation for robust caching.
        """
        try:
            return await self._calculate_connection_weight_cached(node_a, node_b)
        except Exception as e:
            self.logger.error(f"Cached connection weight calculation failed: {e}")
            # Fallback to basic calculation
            return 0.0
    
    async def _calculate_connection_weight_internal(
        self,
        node_a_id: str, 
        node_b_id: str,
        node_a_embedding: List[float],
        node_b_embedding: List[float], 
        node_a_raw_state: Dict[str, Any],
        node_b_raw_state: Dict[str, Any],
        node_a_text: str,
        node_b_text: str,
        node_a_timestamp: float,
        node_b_timestamp: float
    ) -> float:
        """
        Internal connection weight calculation.
        Uses cached metrics orchestrator for expensive operations.
        """
        async with self._calc_semaphore:
            try:
                # Use metrics orchestrator with caching for semantic similarity
                semantic_sim = 0.0
                if self.metrics.embedding_manager:
                    try:
                        semantic_sim = await self.metrics.embedding_manager.calculate_similarity_cached(
                            node_a_embedding, node_b_embedding
                        )
                    except Exception as e:
                        self.logger.error(f"Cached embedding similarity failed: {e}")
                        semantic_sim = self.metrics.embedding_manager._calculate_similarity_internal(
                            node_a_embedding, node_b_embedding
                        )

                # Calculate emotional similarity with timeout
                emotional_sim = 0.0
                try:
                    emotional_ab = await asyncio.wait_for(
                        asyncio.to_thread(
                            lambda: calculate_emotional_similarity(
                                node_a_raw_state.get("emotional_vectors", []),
                                node_b_raw_state.get("emotional_vectors", [])
                            )
                        ),
                        timeout=0.5
                    )
                    emotional_ba = await asyncio.wait_for(
                        asyncio.to_thread(
                            lambda: calculate_emotional_similarity(
                                node_b_raw_state.get("emotional_vectors", []),
                                node_a_raw_state.get("emotional_vectors", [])
                            )
                        ),
                        timeout=0.5
                    )
                    emotional_sim = (emotional_ab + emotional_ba) / 2
                except asyncio.TimeoutError:
                    self.logger.warning("Emotional similarity calculation timed out")
                    emotional_sim = 0.0
                except Exception as e:
                    self.logger.error(f"Error calculating emotional similarity: {e}")
                    emotional_sim = 0.0

                # Calculate state similarity (needs)
                state_sim = 0.0
                try:
                    state_sim = calculate_needs_similarity(
                        node_a_raw_state.get("needs", {}),
                        node_b_raw_state.get("needs", {})
                    )
                except Exception as e:
                    self.logger.error(f"Needs similarity calculation failed: {e}")
                    state_sim = 0.0

                # Calculate temporal similarity
                temporal_sim = 0.0
                decay_hours = 24.0
                if self.config and hasattr(self.config, 'thresholds') and hasattr(self.config.thresholds, 'temporal_window'):
                    decay_hours = self.config.thresholds.temporal_window / 3600.0

                try:
                    temporal_sim = calculate_temporal_weight(
                        node_a_timestamp,
                        node_b_timestamp,
                        decay_hours=decay_hours
                    )
                except Exception as e:
                    self.logger.error(f"Temporal similarity calculation failed: {e}")
                    temporal_sim = 0.0

                # Calculate final weight using all components
                weight = (
                    self.weights["semantic"] * semantic_sim +
                    self.weights["emotional"] * emotional_sim +
                    self.weights["state"] * state_sim +
                    self.weights["temporal"] * temporal_sim
                )
                
                return weight
                
            except Exception as e:
                self.logger.error(f"Connection weight calculation failed: {e}")
                return 0.0

    def _get_node_by_id(self, node_id: str) -> Optional[CognitiveMemoryNode]:
        """
        Retrieve a CognitiveMemoryNode from the network by its ID.
        """
        return self.network.nodes.get(node_id)

    async def transfer_connections(
        self,
        from_node: CognitiveMemoryNode,
        to_node: CognitiveMemoryNode,
        transfer_weight: float = 0.9
    ) -> None:
        """
        Transfer connections from one node to another with proper weight adjustment and max_connections enforcement.
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
