"""
\\metrics\\orchestrator.py

Orchestrates the complete metrics calculation process for memory retrieval.
Coordinates semantic, emotional, state, temporal, and strength calculations.

Key capabilities:
- Unified metrics calculation
- Component management and coordination
- Flexible configuration
- Comprehensive error handling
- Result normalization and weighting
"""

import time
import asyncio
import hashlib
import concurrent.futures as cf

from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
from enum import Enum, auto

from .semantic import SemanticMetricsCalculator 
from .emotional import EmotionalMetricsCalculator
from .state import StateMetricsCalculator
from .temporal import TemporalMetricsCalculator
from .strength import StrengthMetricsCalculator
from ..embedding_manager import EmbeddingManager
from ..state.signatures import EmotionalStateSignature
from ..async_lru_cache import async_lru_cache

from loggers.loggers import MemoryLogger

class MetricComponent(Enum):
    """Available metric calculation components."""
    SEMANTIC = auto()
    EMOTIONAL = auto()
    STATE = auto()
    TEMPORAL = auto()
    STRENGTH = auto()

@dataclass
class MetricsConfiguration:
    """Configuration for metrics calculation."""
    enabled_components: List[MetricComponent] = None
    component_weights: Dict[MetricComponent, float] = None
    semantic_options: Dict[str, Any] = None
    emotional_options: Dict[str, Any] = None
    state_options: Dict[str, Any] = None
    temporal_options: Dict[str, Any] = None
    strength_options: Dict[str, Any] = None
    detailed_metrics: bool = False
    include_strength: bool = True

    def __post_init__(self):
        """Set defaults if not provided."""
        if self.enabled_components is None:
            self.enabled_components = list(MetricComponent)
            
        if self.component_weights is None:
            self.component_weights = {
                MetricComponent.SEMANTIC: 0.425,
                MetricComponent.EMOTIONAL: 0.2, 
                MetricComponent.STATE: 0.225,
                MetricComponent.TEMPORAL: 0.1,
                MetricComponent.STRENGTH: 0.05
            }
            
        # Initialize empty option dicts if None
        self.semantic_options = self.semantic_options or {}
        self.emotional_options = self.emotional_options or {}
        self.state_options = self.state_options or {}
        self.temporal_options = self.temporal_options or {}
        self.strength_options = self.strength_options or {}

class RetrievalMetricsOrchestrator:
    """
    Coordinates calculation of all retrieval metrics.
    Acts as the central point for configuring and executing
    memory similarity calculations.
    """
    
    def __init__(
        self,
        body_memory_reference=None,
        embedding_manager: Optional[EmbeddingManager] = None,
        config: Optional[MetricsConfiguration] = None
    ):
        """
        Initialize orchestrator with required components.
        
        Args:
            embedding_manager: Optional, for semantic analysis
            body_memory_reference: Optional reference to body memory system
            config: Optional custom configuration
        """
        self.config = config or MetricsConfiguration()
        self.logger = MemoryLogger
        self.body_memory_reference = body_memory_reference
        self.embedding_manager = embedding_manager
        
        # Initialize calculators
        self.calculators = {}
        
        if (MetricComponent.SEMANTIC in self.config.enabled_components and 
            self.embedding_manager is not None):
            self.calculators[MetricComponent.SEMANTIC] = SemanticMetricsCalculator(
                embedding_manager=self.embedding_manager
            )
            
        if MetricComponent.EMOTIONAL in self.config.enabled_components:
            self.calculators[MetricComponent.EMOTIONAL] = EmotionalMetricsCalculator()
            
        if MetricComponent.STATE in self.config.enabled_components:
            self.calculators[MetricComponent.STATE] = StateMetricsCalculator(
                emotional_calculator=self.calculators.get(MetricComponent.EMOTIONAL),
                **self.config.state_options
            )
            
        if MetricComponent.TEMPORAL in self.config.enabled_components:
            self.calculators[MetricComponent.TEMPORAL] = TemporalMetricsCalculator(
                **self.config.temporal_options
            )
            
        if MetricComponent.STRENGTH in self.config.enabled_components:
            self.calculators[MetricComponent.STRENGTH] = StrengthMetricsCalculator()

    def _generate_cache_key(
        self,
        target_node: Any,
        comparison_state: Dict[str, Any],
        query_text: str,
        query_embedding: List[float],
        body_node_id: Optional[str] = None
    ) -> str:
        """
        Generate a stable cache key for metrics calculation.
        Uses content hashing to ensure consistency across calls.
        """
        try:
            # Create a stable representation of the calculation inputs
            key_components = []
            
            # Node identifier
            if hasattr(target_node, 'node_id') and target_node.node_id:
                key_components.append(f"node:{target_node.node_id}")
            else:
                # Fallback to content hash for nodes without IDs
                node_content = getattr(target_node, 'text_content', '') or str(getattr(target_node, 'raw_state', {}))
                node_hash = hashlib.md5(node_content.encode('utf-8')).hexdigest()[:8]
                key_components.append(f"content:{node_hash}")
            
            # Comparison state hash
            if comparison_state:
                state_str = str(sorted(comparison_state.items()))
                state_hash = hashlib.md5(state_str.encode('utf-8')).hexdigest()[:8]
                key_components.append(f"state:{state_hash}")
            
            # Query content hash
            if query_text:
                query_hash = hashlib.md5(query_text.encode('utf-8')).hexdigest()[:8]
                key_components.append(f"query:{query_hash}")
            
            # Embedding hash (first few values as representative)
            if query_embedding and len(query_embedding) > 0:
                emb_sample = str(query_embedding[:5])  # First 5 values as signature
                emb_hash = hashlib.md5(emb_sample.encode('utf-8')).hexdigest()[:6]
                key_components.append(f"emb:{emb_hash}")
            
            # Body node reference
            if body_node_id:
                key_components.append(f"body:{body_node_id}")
            
            return "|".join(key_components)
            
        except Exception as e:
            # Fallback to a basic key if hashing fails
            self.logger.log_error(f"Cache key generation failed: {e}")
            return f"fallback:{hash((str(target_node), str(comparison_state), query_text))}"

    @async_lru_cache(
        maxsize=2000, 
        ttl=7200, 
        key_func=lambda self, target_node, comparison_state, query_text, query_embedding, body_node_id=None, preserved_signature=None, override_config=None: 
            self._generate_cache_key(target_node, comparison_state, query_text, query_embedding, body_node_id)
    )
    async def calculate_metrics(
        self,
        target_node: Any,
        comparison_state: Dict[str, Any],
        query_text: str,
        query_embedding: List[float],
        body_node_id: Optional[str] = None,
        preserved_signature: Optional[EmotionalStateSignature] = None,
        override_config: Optional[MetricsConfiguration] = None
    ) -> Union[float, Dict[str, Any]]:
        """
        Calculate comprehensive retrieval metrics with intelligent caching.
        Uses custom cache key generation via decorator.
        """
        return await self._calculate_metrics_internal(
            target_node, comparison_state, query_text, query_embedding,
            body_node_id, preserved_signature, override_config
        )

    async def _calculate_metrics_internal(
        self,
        target_node: Any,
        comparison_state: Dict[str, Any],
        query_text: str,
        query_embedding: List[float],
        body_node_id: Optional[str] = None,
        preserved_signature: Optional[EmotionalStateSignature] = None,
        override_config: Optional[MetricsConfiguration] = None
    ) -> Union[float, Dict[str, Any]]:
        """
        Internal implementation of metrics calculation.
        This is the actual computation that gets cached.
        """
        try:
            config = override_config or self.config
            metrics = {}

            tasks = []
            components = []
            
            if MetricComponent.SEMANTIC in config.enabled_components:
                tasks.append(self._calculate_semantic_metrics(
                    target_node, query_text, query_embedding, config.semantic_options
                ))
                components.append(MetricComponent.SEMANTIC)
                
            loop = asyncio.get_event_loop()
            with cf.ThreadPoolExecutor() as executor:
                if MetricComponent.EMOTIONAL in config.enabled_components:
                    tasks.append(loop.run_in_executor(
                    executor,
                    self._calculate_emotional_metrics,
                    target_node, comparison_state, body_node_id,
                    preserved_signature, config.emotional_options
                ))
                components.append(MetricComponent.EMOTIONAL)
                
                if MetricComponent.STATE in config.enabled_components:
                    tasks.append(loop.run_in_executor(
                        executor,
                        self._calculate_state_metrics,
                        target_node, comparison_state, body_node_id,
                        preserved_signature, config.state_options
                    ))
                    components.append(MetricComponent.STATE)
                    
                if MetricComponent.TEMPORAL in config.enabled_components:
                    tasks.append(loop.run_in_executor(
                        executor,
                        self._calculate_temporal_metrics,
                        target_node, config.temporal_options
                    ))
                    components.append(MetricComponent.TEMPORAL)
                    
                if (MetricComponent.STRENGTH in config.enabled_components and 
                    config.include_strength):
                    tasks.append(loop.run_in_executor(
                        executor,
                        self._calculate_strength_metrics,
                        target_node, config.strength_options
                    ))
                    components.append(MetricComponent.STRENGTH)

                # Run all tasks concurrently
                all_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for i, component in enumerate(components):
                if isinstance(all_results[i], Exception):
                    self.logger.log_error(f"{component.name} metrics failed: {str(all_results[i])}")
                    metrics[component.name.lower()] = {'error': str(all_results[i])}
                else:
                    metrics[component.name.lower()] = all_results[i]
                    
            # Return detailed metrics or compute final score
            if config.detailed_metrics:
                return {
                    'final_score': self._compute_final_score(metrics, config),
                    'component_metrics': metrics,
                    'component_weights': config.component_weights
                }
            else:
                return self._compute_final_score(metrics, config)
                
        except Exception as e:
            self.logger.log_error(f"Metrics calculation failed: {str(e)}")
            if config.detailed_metrics:
                return {
                    'error': str(e),
                    'final_score': 0.0,
                    'component_metrics': {}
                }
            return 0.0
        
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache performance statistics from decorators."""
        stats = {}
        
        try:
            # Get stats from decorated methods
            if hasattr(self.calculate_metrics, 'cache_info'):
                stats['metrics_cache'] = self.calculate_metrics.cache_info()
                
            if self.embedding_manager and hasattr(self.embedding_manager.calculate_similarity_cached, 'cache_info'):
                stats['embedding_cache'] = self.embedding_manager.calculate_similarity_cached.cache_info()
                
            if hasattr(self.calculators.get(MetricComponent.SEMANTIC), '_calculate_text_relevance_cached'):
                semantic_calc = self.calculators[MetricComponent.SEMANTIC]
                if hasattr(semantic_calc._calculate_text_relevance_cached, 'cache_info'):
                    stats['semantic_cache'] = semantic_calc._calculate_text_relevance_cached.cache_info()
            
            if hasattr(self, 'body_memory_reference') and self.body_memory_reference:
                try:
                    body_network = getattr(self.body_memory_reference, 'body_network', None)
                    if body_network and hasattr(body_network, 'connection_manager'):
                        body_conn_mgr = body_network.connection_manager
                        if hasattr(body_conn_mgr, '_calculate_connection_weight_cached') and hasattr(body_conn_mgr._calculate_connection_weight_cached, 'cache_info'):
                            stats['body_connection_cache'] = body_conn_mgr._calculate_connection_weight_cached.cache_info()
                except Exception as e:
                    stats['body_connection_cache_error'] = str(e)
            
            stats['cache_enabled'] = True
            
        except Exception as e:
            stats = {'cache_enabled': False, 'error': str(e)}
        
        return stats

    async def clear_all_caches(self) -> Dict[str, bool]:
        """Clear all decorator-managed caches."""
        results = {}
        
        try:
            # Clear decorator caches
            if hasattr(self.calculate_metrics, 'cache_clear'):
                await self.calculate_metrics.cache_clear()
                results['metrics_cache'] = True
                
            if self.embedding_manager and hasattr(self.embedding_manager.calculate_similarity_cached, 'cache_clear'):
                await self.embedding_manager.calculate_similarity_cached.cache_clear()
                results['embedding_cache'] = True
                
            if hasattr(self.calculators.get(MetricComponent.SEMANTIC), '_calculate_text_relevance_cached'):
                semantic_calc = self.calculators[MetricComponent.SEMANTIC]
                if hasattr(semantic_calc._calculate_text_relevance_cached, 'cache_clear'):
                    await semantic_calc._calculate_text_relevance_cached.cache_clear()
                    results['semantic_cache'] = True
            
            if hasattr(self, 'body_memory_reference') and self.body_memory_reference:
                try:
                    body_network = getattr(self.body_memory_reference, 'body_network', None)
                    if body_network and hasattr(body_network, 'connection_manager'):
                        body_conn_mgr = body_network.connection_manager
                        if hasattr(body_conn_mgr, '_calculate_connection_weight_cached') and hasattr(body_conn_mgr._calculate_connection_weight_cached, 'cache_clear'):
                            await body_conn_mgr._calculate_connection_weight_cached.cache_clear()
                            results['body_connection_cache'] = True
                except Exception as e:
                    results['body_connection_cache_error'] = str(e)
                
        except Exception as e:
            self.logger.log_error(f"Cache clearing failed: {e}")
            results['error'] = str(e)
        
        return results

    async def _calculate_semantic_metrics(
        self,
        node: Any,
        query_text: Optional[str],
        query_embedding: Optional[List[float]],
        options: Dict[str, Any]
    ) -> Dict[str, float]:
        """Calculate semantic metrics with error handling."""
        try:
            # Return default metrics for nodes without text content
            if not hasattr(node, 'text_content') or node.text_content is None:
                return {
                    'embedding_similarity': 0.0,
                    'text_similarity': 0.0,
                    'combined_similarity': 0.0
                }
                
            calculator = self.calculators[MetricComponent.SEMANTIC]
            return await calculator.calculate_metrics(
                text_content=node.text_content,
                embedding=node.embedding,
                query_text=query_text,
                query_embedding=query_embedding,
                **options
            )
        except Exception as e:
            self.logger.log_error(f"Semantic metrics failed: {str(e)}")
            return {'error': str(e)}

    def _calculate_emotional_metrics(
        self,
        node: Any,
        comparison_state: Dict[str, Any],
        body_node_id: Optional[str],
        preserved_signature: Optional[EmotionalStateSignature],
        options: Dict[str, Any]
    ) -> Dict[str, float]:
        """Calculate emotional metrics with error handling."""
        try:
            calculator = self.calculators[MetricComponent.EMOTIONAL]
            return calculator.calculate_metrics(
                node_state=node.raw_state,
                comparison_state=comparison_state,
                **options
            )
        except Exception as e:
            self.logger.log_error(f"Emotional metrics failed: {str(e)}")
            return {'error': str(e)}

    def _calculate_state_metrics(
        self,
        node: Any,
        comparison_state: Dict[str, Any],
        body_node_id: Optional[str],
        preserved_signature: Optional[EmotionalStateSignature],
        options: Dict[str, Any]
    ) -> Dict[str, Dict[str, float]]:
        """Calculate state metrics with error handling."""
        try:
            calculator = self.calculators[MetricComponent.STATE]
            return calculator.calculate_metrics(
                node_state={
                    'raw_state': node.raw_state,
                    'processed_state': node.processed_state
                },
                comparison_state=comparison_state,
                preserved_signature=preserved_signature,
                **options
            )
        except Exception as e:
            self.logger.log_error(f"State metrics failed: {str(e)}")
            return {'error': str(e)}

    def _calculate_temporal_metrics(
        self,
        node: Any,
        options: Dict[str, Any]
    ) -> Dict[str, float]:
        """Calculate temporal metrics with error handling."""
        try:
            calculator = self.calculators[MetricComponent.TEMPORAL]
            return calculator.calculate_metrics(
                node_timestamp=node.timestamp,
                last_accessed=getattr(node, 'last_accessed', None),
                last_echo_time=getattr(node, 'last_echo_time', None),
                echo_dampening=getattr(node, 'echo_dampening', None),
                **options
            )
        except Exception as e:
            self.logger.log_error(f"Temporal metrics failed: {str(e)}")
            return {'error': str(e)}

    def _calculate_strength_metrics(
        self,
        node: Any,
        options: Dict[str, Any]
    ) -> Dict[str, float]:
        """Calculate strength/network metrics with error handling."""
        try:
            calculator = self.calculators[MetricComponent.STRENGTH]
            # Get connected node strengths if possible
            connected_strengths = []
            if hasattr(node, '_get_node_by_id'):
                connected_strengths = [
                    n.strength for n in 
                    [node._get_node_by_id(nid) for nid in node.connections.keys()]
                    if n is not None
                ]
            
            return calculator.calculate_metrics(
                node_strength=node.strength,
                connections=node.connections,
                is_ghosted=node.ghosted,
                ghost_nodes=node.ghost_nodes,
                connected_strengths=connected_strengths,
                **options
            )
        except Exception as e:
            self.logger.log_error(f"Strength metrics failed: {str(e)}")
            return {'error': str(e)}

    def _compute_final_score(
        self,
        metrics: Dict[str, Any],
        config: MetricsConfiguration
    ) -> float:
        """
        Compute final similarity score from component metrics.
        Handles missing metrics and applies component weights.
        """
        try:
            score = 0.0
            total_weight = 0.0
            
            for component, weight in config.component_weights.items():
                if component.name.lower() not in metrics:
                    continue
                    
                component_metrics = metrics[component.name.lower()]
                if isinstance(component_metrics, dict):
                    if 'error' in component_metrics:
                        continue
                        
                    # Extract main score based on component type
                    if component == MetricComponent.SEMANTIC:
                        component_score = component_metrics.get('embedding_similarity', 0.0)
                    elif component == MetricComponent.EMOTIONAL:
                        component_score = component_metrics.get('vector_similarity', 0.0)
                    elif component == MetricComponent.STATE:
                        # Average sub-components
                        sub_scores = []
                        for sub_dict in component_metrics.values():
                            if isinstance(sub_dict, dict):
                                sub_scores.extend(
                                    v for v in sub_dict.values() 
                                    if isinstance(v, (int, float))
                                )
                        component_score = (
                            sum(sub_scores) / len(sub_scores) 
                            if sub_scores else 0.0
                        )
                    else:
                        # Use first numeric value as score
                        component_score = next(
                            (v for v in component_metrics.values() 
                             if isinstance(v, (int, float))),
                            0.0
                        )
                        
                    score += component_score * weight
                    total_weight += weight
                    
            return score / total_weight if total_weight > 0 else 0.0
            
        except Exception as e:
            self.logger.log_error(f"Final score computation failed: {str(e)}")
            return 0.0
        
    async def compare_nodes(
        self,
        nodeA: Any,
        nodeB: Any,
        override_config: Optional[MetricsConfiguration] = None
    ) -> Dict[str, float]:
        """
        Pairwise comparison of two nodes without using an external query.
        Each node computes its detailed metrics using the other node's state
        as the comparison_state. The result is a dissonance score per component.
        
        Returns:
            Dict mapping component names ('semantic', 'emotional', etc.) to a
            float representing the average absolute difference in that component.
        """
        # Ensure detailed metrics are enabled so we get component breakdowns.
        config = override_config or self.config
        # Force detailed_metrics on for this computation.
        config.detailed_metrics = True

        # For pairwise, we pass empty query info. The idea is that the node's own data
        # and the other node's stored state (e.g., raw_state) will drive the comparison.
        # For semantic comparison, use the other node's embedding
        metricsA = await self.calculate_metrics(
            target_node=nodeA,
            comparison_state=getattr(nodeB, 'raw_state', {}),
            query_text="",
            query_embedding=getattr(nodeB, 'embedding', [0.0] * (nodeA.embedding.shape[0] if hasattr(nodeA, 'embedding') else 0)),
            override_config=config
        )
        metricsB = await self.calculate_metrics(
            target_node=nodeB,
            comparison_state=getattr(nodeA, 'raw_state', {}),
            query_text="",
            query_embedding=getattr(nodeA, 'embedding', [0.0] * (nodeB.embedding.shape[0] if hasattr(nodeB, 'embedding') else 0)),
            override_config=config
        )

        # Extract detailed component metrics.
        detailedA = metricsA.get('component_metrics', {})
        detailedB = metricsB.get('component_metrics', {})

        dissonance = {}
        # Iterate over the expected components.
        for component in config.component_weights.keys():
            comp_key = component.name.lower()
            compA_val = detailedA.get(comp_key, {})
            compB_val = detailedB.get(comp_key, {})
            # Compute a difference value between the two component dicts.
            dissonance[comp_key] = self._compare_component_dicts(compA_val, compB_val)
        return dissonance

    def _compare_component_dicts(self, dataA: Any, dataB: Any) -> float:
        """
        Recursively compare two data structures (expected to be dicts or numeric values)
        and return an average absolute difference.
        
        If both values are numeric, the difference is the absolute difference.
        If they are dicts, we compute the difference over their union of keys.
        Otherwise, if one or both values are missing or non-numeric, we assume zero.
        """
        # If both are numeric values, return absolute difference.
        if isinstance(dataA, (int, float)) and isinstance(dataB, (int, float)):
            return abs(dataA - dataB)
        
        # If both are dicts, compare recursively key by key.
        if isinstance(dataA, dict) and isinstance(dataB, dict):
            differences = []
            all_keys = set(dataA.keys()) | set(dataB.keys())
            for key in all_keys:
                diff = self._compare_component_dicts(
                    dataA.get(key, 0),
                    dataB.get(key, 0)
                )
                differences.append(diff)
            return sum(differences) / len(differences) if differences else 0.0
        
        # For any non-numeric and non-dict types, return 0 difference.
        return 0.0

    def update_configuration(
        self,
        new_config: MetricsConfiguration
    ) -> None:
        """
        Update orchestrator configuration and reinitialize calculators as needed.
        
        Args:
            new_config: New configuration to apply
        """
        # Store old config for comparison
        old_components = set(self.config.enabled_components)
        
        # Update config
        self.config = new_config
        
        # Check for calculator changes
        new_components = set(new_config.enabled_components)
        removed = old_components - new_components
        added = new_components - old_components
        
        # Remove disabled calculators
        for component in removed:
            self.calculators.pop(component, None)
            
        # Initialize new calculators
        for component in added:
            if component == MetricComponent.SEMANTIC:
                self.calculators[component] = SemanticMetricsCalculator(
                    embedding_manager=self.embedding_manager
                )
            elif component == MetricComponent.EMOTIONAL:
                self.calculators[component] = EmotionalMetricsCalculator()
            elif component == MetricComponent.STATE:
                self.calculators[component] = StateMetricsCalculator(
                    emotional_calculator=self.calculators.get(MetricComponent.EMOTIONAL),
                    **self.config.state_options
                )
            elif component == MetricComponent.TEMPORAL:
                self.calculators[component] = TemporalMetricsCalculator(
                    **self.config.temporal_options
                )
            elif component == MetricComponent.STRENGTH:
                self.calculators[component] = StrengthMetricsCalculator()