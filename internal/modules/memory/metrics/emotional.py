"""
\metrics\emotional.py

Implements emotional similarity and resonance calculations for memory retrieval.
Handles both stored emotional states and body memory relationships.

Key capabilities:
- Emotional vector similarity calculation
- State signature preservation
- Neutral state handling ("connecting foam")
- Intensity and complexity analysis
"""

from typing import Dict, List, Optional, Union, Any
from dataclasses import dataclass
import time
import math

from internal.modules.memory.nodes.body_node import BodyMemoryNode
from internal.modules.memory.state.signatures import EmotionalStateSignature

from loggers.loggers import MemoryLogger

class EmotionalMetricsError(Exception):
    """Base exception for emotional metrics calculation failures."""
    pass

class EmotionalMetricsCalculator:
    """
    Calculates emotional similarity metrics between memory nodes.
    Handles both stored states and body memory relationships.
    """
    
    def __init__(self):
        """
        Initialize calculator
        """
        self.logger = MemoryLogger
        
    def calculate_metrics(
        self,
        node_state: Dict[str, Any],
        comparison_state: Dict[str, Any]
    ) -> Dict[str, float]:
        """
        Calculate comprehensive emotional similarity metrics.
        
        Args:
            node_state: Node's stored emotional state
            comparison_state: State to compare against
            
        Returns:
            Dict containing emotional metrics:
            - vector_similarity: Direct emotional vector comparison
            - emotional_complexity: Complexity of emotional states
            - valence_shift: Changes in emotional valence
            - intensity_delta: Intensity differences
            - neutral_alignment: Similarity despite neutral states
        """
        try:
            metrics = {}
            
            # Get emotional vectors from states
            node_vectors = node_state.get('emotional_vectors', [])
            comp_vectors = comparison_state.get('emotional_vectors', [])
            
            # Calculate base vector similarity
            metrics.update(self._calculate_vector_similarity(
                node_vectors,
                comp_vectors
            ))
            
            # Handle body memory integration if available
            # what did she mean by this ?! !? 
            
            # Calculate emotional complexity
            metrics['emotional_complexity'] = self._calculate_emotional_complexity(
                node_vectors,
                comp_vectors
            )
            
            # Calculate valence shifts
            metrics['valence_shift'] = self._calculate_valence_shift(
                node_vectors,
                comp_vectors
            )
            
            # Add intensity comparison
            metrics['intensity_delta'] = self._calculate_intensity_delta(
                node_vectors,
                comp_vectors
            )
            
            return metrics
            
        except Exception as e:
            self.logger.log_error(f"Emotional metrics calculation failed: {str(e)}")
            return self._get_fallback_metrics()
            
    def _calculate_vector_similarity(
        self,
        vectors1: List[Dict[str, Any]],
        vectors2: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """
        Calculate similarity between emotional vector sets.
        Handles both present and absent emotions.
        """
        metrics = {}
        
        # Handle neutral states
        if not vectors1 or not vectors2:
            metrics['neutral_alignment'] = self._calculate_neutral_similarity(
                vectors1 or vectors2
            )
            metrics['vector_similarity'] = 0.0
            return metrics
            
        # Calculate pairwise similarities
        similarities = []
        for v1 in vectors1:
            for v2 in vectors2:
                # Compare valence-arousal pairs
                valence_sim = 1.0 - abs(
                    v1.get('valence', 0.0) - v2.get('valence', 0.0)
                )
                arousal_sim = 1.0 - abs(
                    v1.get('arousal', 0.0) - v2.get('arousal', 0.0)
                )
                intensity_sim = 1.0 - abs(
                    v1.get('intensity', 0.0) - v2.get('intensity', 0.0)
                )
                
                # Weight components
                sim = (
                    valence_sim * 0.425 +
                    arousal_sim * 0.425 +
                    intensity_sim * 0.15
                )
                similarities.append(sim)
                
        metrics['vector_similarity'] = (
            sum(similarities) / len(similarities)
            if similarities else 0.0
        )
        
        return metrics
        
    def _calculate_neutral_similarity(
        self,
        vectors: List[Dict[str, Any]]
    ) -> float:
        """
        Calculate similarity contribution for neutral states.
        Implements "connecting foam" concept.
        """
        if not vectors:
            return 0.3  # Base neutral alignment
            
        # Calculate reduced weight based on vectors present
        total_intensity = sum(v.get('intensity', 0.0) for v in vectors)
        return 0.3 * (1.0 - min(1.0, total_intensity))
            
    def _calculate_temporal_weight(self, timestamp: float) -> float:
        """Calculate temporal weighting for preserved signatures."""
        time_diff = time.time() - timestamp
        return max(0.1, math.exp(-time_diff / (24 * 3600)))  # 24-hour decay
        
    def _calculate_emotional_complexity(
        self,
        vectors1: List[Dict[str, Any]],
        vectors2: List[Dict[str, Any]]
    ) -> float:
        """
        Calculate emotional complexity comparison.
        
        EXPANSION POINT: Enhanced complexity analysis
        - Emotional pattern recognition
        - State transition analysis
        - Emotional rhythm detection
        """
        if not vectors1 or not vectors2:
            return 0.0
            
        # Compare number of distinct emotions
        complexity1 = len(vectors1) / 5.0  # Normalize to max 5 emotions
        complexity2 = len(vectors2) / 5.0
        
        # Calculate complexity similarity
        return 1.0 - abs(complexity1 - complexity2)
        
    def _calculate_valence_shift(
        self,
        vectors1: List[Dict[str, Any]],
        vectors2: List[Dict[str, Any]]
    ) -> float:
        """Calculate shift in emotional valence between states."""
        if not vectors1 or not vectors2:
            return 0.0
            
        try:
            # Calculate weighted average valence for each state
            def get_weighted_valence(vectors):
                total_intensity = sum(v.get('intensity', 0.0) for v in vectors)
                if total_intensity == 0:
                    return 0.0
                    
                weighted_sum = sum(
                    v.get('valence', 0.0) * v.get('intensity', 0.0)
                    for v in vectors
                )
                return weighted_sum / total_intensity
                
            valence1 = get_weighted_valence(vectors1)
            valence2 = get_weighted_valence(vectors2)
            
            return abs(valence1 - valence2)
            
        except Exception as e:
            self.logger.log_error(f"Valence shift calculation failed: {str(e)}")
            return 0.0
            
    def _calculate_intensity_delta(
        self,
        vectors1: List[Dict[str, Any]],
        vectors2: List[Dict[str, Any]]
    ) -> float:
        """Calculate difference in emotional intensity between states."""
        try:
            intensity1 = sum(v.get('intensity', 0.0) for v in vectors1)
            intensity2 = sum(v.get('intensity', 0.0) for v in vectors2)
            
            max_intensity = max(intensity1, intensity2)
            if max_intensity == 0:
                return 0.0
                
            return abs(intensity1 - intensity2) / max_intensity
            
        except Exception as e:
            self.logger.log_error(f"Intensity delta calculation failed: {str(e)}")
            return 0.0
            
    def _get_fallback_metrics(self) -> Dict[str, float]:
        """Provide safe fallback metrics if calculations fail."""
        return {
            'vector_similarity': 0.0,
            'emotional_complexity': 0.0,
            'valence_shift': 0.0,
            'intensity_delta': 0.0,
            'neutral_alignment': 0.3,
            'body_state_similarity': 0.0
        }