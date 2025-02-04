import time
import asyncio
from asyncio import TimeoutError as AsyncTimeoutError
from typing import Optional, TYPE_CHECKING
from functools import lru_cache

from .base_manager import (
    BaseConnectionManager,
    ConnectionOperation,
    MergeCandidate
)

if TYPE_CHECKING:
    from ..cognitive_network import CognitiveMemoryNetwork

from ...nodes.cognitive_node import CognitiveMemoryNode
from ...nodes.node_utils import (
    softmax,
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
            'semantic': 0.4,
            'emotional': 0.2,
            'state': 0.2,
            'temporal': 0.2
        }
        self._calc_semaphore = asyncio.Semaphore(20)

    async def _calculate_connection_weight(self, node_a: CognitiveMemoryNode, node_b: CognitiveMemoryNode, operation: ConnectionOperation) -> float:
        """
        Calculate bidirectional connection weight between two CognitiveMemoryNodes using the metrics orchestrator.
        Falls back to direct calculations if orchestrator fails.
        """
        async with self._calc_semaphore:
            # Try using metrics orchestrator first for both directions
            try:
                if not isinstance(node_a, CognitiveMemoryNode) or not isinstance(node_b, CognitiveMemoryNode):
                    raise ValueError(f"Invalid node types provided for connection weight calculation {node_a.__class__} {node_b.__class__}")
                
                # Set up temporary config to force detailed metrics
                temp_config = self.metrics.config
                temp_config.detailed_metrics = True

                # Calculate metrics in both directions
                metrics_ab = await self.metrics.calculate_metrics(
                    target_node=node_a,
                    comparison_state=node_b.raw_state,
                    query_text=node_b.text_content,
                    query_embedding=node_b.embedding,
                    override_config=temp_config
                )
                metrics_ba = await self.metrics.calculate_metrics(
                    target_node=node_b,
                    comparison_state=node_a.raw_state,
                    query_text=node_a.text_content,
                    query_embedding=node_a.embedding,
                    override_config=temp_config
                )

                if isinstance(metrics_ab, dict) and isinstance(metrics_ba, dict) and \
                'component_metrics' in metrics_ab and 'component_metrics' in metrics_ba:
                    
                    # Process both directions
                    def extract_components(metrics):
                        components = metrics['component_metrics']
                        semantic = components.get('semantic', {}).get('embedding_similarity', 0.0)
                        emotional = components.get('emotional', {}).get('vector_similarity', 0.0)
                        
                        # Aggregate state similarity if available
                        state_sim = 0.0
                        if 'state' in components:
                            state_values = [v for d in components['state'].values() 
                                        if isinstance(d, dict) 
                                        for v in d.values() 
                                        if isinstance(v, (int, float))]
                            if state_values:
                                state_sim = sum(state_values) / len(state_values)
                                
                        temporal = next((v for v in components.get('temporal', {}).values() 
                                    if isinstance(v, (int, float))), 0.0)
                        
                        return semantic, emotional, state_sim, temporal

                    # Get components from both directions
                    sem_ab, emo_ab, state_ab, temp_ab = extract_components(metrics_ab)
                    sem_ba, emo_ba, state_ba, temp_ba = extract_components(metrics_ba)

                    # Average the components
                    semantic_sim = (sem_ab + sem_ba) / 2
                    emotional_sim = (emo_ab + emo_ba) / 2
                    state_sim = (state_ab + state_ba) / 2
                    temporal_sim = (temp_ab + temp_ba) / 2

                    # Calculate final weight using averaged components
                    weight = (
                        self.weights["semantic"] * semantic_sim +
                        self.weights["emotional"] * emotional_sim +
                        self.weights["state"] * state_sim +
                        self.weights["temporal"] * temporal_sim
                    )
                    return weight

            except Exception as e:
                self.logger.error(f"Metrics orchestrator calculation failed: {e}")

            # If orchestrator fails, fall back to direct calculations
            self.logger.debug(f"Falling back to direct calculations {node_a.node_id} <-> {node_b.node_id}: {metrics_ab} {metrics_ba}") 

            # Calculate semantic similarity
            try:
                semantic_ab = self.metrics.embedding_manager.calculate_similarity(
                    node_a.embedding,
                    node_b.embedding
                )
                semantic_ba = self.metrics.embedding_manager.calculate_similarity(
                    node_b.embedding,
                    node_a.embedding
                )
                semantic_sim = (semantic_ab + semantic_ba) / 2
            except Exception as e:
                self.logger.error(f"Semantic similarity calculation failed: {e}")
                semantic_sim = 0.0

            # Calculate emotional similarity
            try:
                emotional_ab = await asyncio.wait_for(
                    asyncio.to_thread(
                        lambda: calculate_emotional_similarity(
                            node_a.raw_state.get("emotional_vectors", []),
                            node_b.raw_state.get("emotional_vectors", [])
                        )
                    ),
                    timeout=0.5
                )
                emotional_ba = await asyncio.wait_for(
                    asyncio.to_thread(
                        lambda: calculate_emotional_similarity(
                            node_b.raw_state.get("emotional_vectors", []),
                            node_a.raw_state.get("emotional_vectors", [])
                        )
                    ),
                    timeout=0.5
                )
                emotional_sim = (emotional_ab + emotional_ba) / 2
            except AsyncTimeoutError:
                self.logger.warning("Emotional similarity calculation timed out")
                emotional_sim = 0.0
            except Exception as e:
                self.logger.error(f"Error calculating emotional similarity: {e}")
                emotional_sim = 0.0

            # Calculate state similarity (needs)
            try:
                state_sim = calculate_needs_similarity(
                    node_a.raw_state.get("needs", {}),
                    node_b.raw_state.get("needs", {})
                )
            except Exception as e:
                self.logger.error(f"Needs similarity calculation failed: {e}")
                state_sim = 0.0

            # Calculate temporal similarity
            decay_hours = 24.0
            if self.config and hasattr(self.config, 'thresholds') and hasattr(self.config.thresholds, 'temporal_window'):
                decay_hours = self.config.thresholds.temporal_window / 3600.0

            try:
                temporal_sim = calculate_temporal_weight(
                    node_a.timestamp,
                    node_b.timestamp,
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

    def _get_node_by_id(self, node_id: str) -> Optional[CognitiveMemoryNode]:
        """
        Retrieve a CognitiveMemoryNode from the network by its ID.
        """
        return self.network.nodes.get(node_id)

    async def transfer_connections(self, from_node: CognitiveMemoryNode, to_node: CognitiveMemoryNode, transfer_weight: float = 0.9) -> None:
        """
        Transfer connections from one node to another with proper weight adjustment.
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
