"""
\metrics\state.py

Implements unified state metrics calculations for memory retrieval.
Handles needs, behavior, mood and emotional state comparisons.

Key capabilities:
- Needs similarity and urgency analysis
- Behavior pattern matching
- Mood state comparison
- Emotional state integration 
- Support for body/cognitive memory integration
"""

from typing import Dict, Any, Optional, List
import math
import time

from .emotional import EmotionalMetricsCalculator, EmotionalStateSignature

from loggers.loggers import MemoryLogger

class StateMetricsError(Exception):
    """Base exception for state metrics calculation failures."""
    pass

class StateMetricsCalculator:
    """
    Calculates and unifies state-based similarity metrics for memory retrieval.
    Deals with 'needs', 'mood', 'behavior', and optional emotional vectors.

    By consolidating these methods in one place, both CognitiveMemory and BodyMemory
    can rely on the same logic for comparing states, preventing duplication.
    """

    def __init__(
        self,
        emotional_calculator: EmotionalMetricsCalculator
    ):
        """
        Args:
            emotional_calculator: Instance of EmotionalMetricsCalculator for emotional state comparisons
        """
        self.logger = MemoryLogger
        self.emotional_calculator = emotional_calculator

    def calculate_metrics(
        self,
        node_state: Dict[str, Any],
        comparison_state: Dict[str, Any],
        **kwargs
    ) -> Dict[str, Dict[str, float]]:
        """
        Main entry point. Calculates state-based metrics across four categories:
        - "needs"
        - "behavior"
        - "mood"
        - "emotional"

        Each sub-dict can contain multiple numeric fields that the retrieval system 
        can combine/weight as needed.

        Args:
            node_state: Dict with at least {'raw_state': {}, 'processed_state': {}}
            comparison_state: Same structure, representing the "other" memory or current state
            kwargs: Extra options or expansions

        Returns:
            Dict[str, Dict[str, float]] with sub-keys for each category.
        """
        try:
            # Safely unpack
            raw_node = node_state.get('raw_state', {})
            raw_comp = comparison_state.get('raw_state', {})

            metrics = {
                "needs": {},
                "behavior": {},
                "mood": {},
                "emotional": {}
            }

            # 1) Needs
            if 'needs' in raw_node and 'needs' in raw_comp:
                needs_metrics = self._calculate_needs_metrics(raw_node['needs'], raw_comp['needs'])
                metrics["needs"].update(needs_metrics)

            # 2) Behavior
            if 'behavior' in raw_node and 'behavior' in raw_comp:
                behavior_metrics = self._calculate_behavior_metrics(raw_node['behavior'], raw_comp['behavior'])
                metrics["behavior"].update(behavior_metrics)

            # 3) Mood
            if 'mood' in raw_node and 'mood' in raw_comp:
                mood_metrics = self._calculate_mood_metrics(raw_node['mood'], raw_comp['mood'])
                metrics["mood"].update(mood_metrics)

            # 4) Emotional Vectors
            emo_metrics = self.emotional_calculator.calculate_metrics(
                node_state=raw_node,
                comparison_state=raw_comp
            )
            metrics["emotional"].update(emo_metrics)

            # eventually do cluster based analysis on directly related body memory node 
            # can additionally consider "cognitive" as a part of the state somehow, perhaps, far off...

            return metrics

        except Exception as e:
            if self.logger:
                self.logger.log_error(f"State metrics calculation failed: {str(e)}")
            return {"error": str(e)}

    # --------------------------------------------------------------------------
    #  Implementation: Needs
    # --------------------------------------------------------------------------
    def _calculate_needs_metrics(
        self,
        needs1: Dict[str, Any],
        needs2: Dict[str, Any]
    ) -> Dict[str, float]:
        """
        Compare two sets of needs. 
        Borrowing logic from both BodyMemory and CognitiveMemory approaches:
         - We can compute satisfaction similarity or delta
         - We can compute urgency or "state shifts"
        
        Example needs dict format:
        {
            'hunger': {'value': 50, 'satisfaction': 0.5},
            'thirst': {'value': 30, 'satisfaction': 0.7},
            'stamina': {'value': 80, 'satisfaction': 0.8}
        }

        Returns a dict like:
        {
            'satisfaction_similarity': 0.85,  # How similar the need states are (0-1)
            'urgency_levels': 0.4,           # Average urgency of needs (0-1)
            'state_shifts': 0.15             # Magnitude of state differences (0-1)
        }
        """
        metrics = {}

        try:
            # Basic similarity measure
            similarity = self._calculate_needs_similarity(needs1, needs2)
            metrics['satisfaction_similarity'] = similarity

            # Overall "urgency" for node1's needs
            urgency_1 = self._calculate_need_urgency(needs1)
            metrics['urgency_levels'] = urgency_1

            # State shifts measure how different these needs sets are
            metrics['state_shifts'] = abs(similarity - 1.0)

        except Exception as e:
            if self.logger:
                self.logger.log_error(f"Needs metrics failed: {str(e)}")

        return metrics

    def _calculate_needs_similarity(
        self,
        needs1: Dict[str, Any],
        needs2: Dict[str, Any]
    ) -> float:
        """
        Calculate similarity between two need states.
        
        Example inputs:
        needs1 = {
            'hunger': {'value': 50},
            'thirst': {'current_value': 30}
        }
        needs2 = {
            'hunger': {'value': 60},
            'thirst': {'value': 35}
        }
        
        Returns [0-1] measure of how similar these sets of needs are.
        1.0 = identical, 0.0 = completely different
        """
        if not needs1 or not needs2:
            return 0.0

        common_needs = set(needs1.keys()) & set(needs2.keys())
        if not common_needs:
            return 0.0

        similarities = []
        for need in common_needs:
            # Handle both value formats
            val1 = needs1[need].get('value', needs1[need].get('current_value', 50))
            val2 = needs2[need].get('value', needs2[need].get('current_value', 50))

            val1 = float(val1)
            val2 = float(val2)
            diff = abs(val1 - val2) / 100.0
            similarities.append(1.0 - diff)

        return sum(similarities) / len(similarities) if similarities else 0.0

    def _calculate_need_urgency(self, needs: Dict[str, Any]) -> float:
        """
        Calculate overall urgency level of needs state.
        Matches NeedsManager satisfaction calculations.
        
        Example input:
        {
            'hunger': {'value': 70},          # High value = more urgent
            'thirst': {'value': 50},          # High value = more urgent
            'stamina': {'value': 30}          # Low value = more urgent
        }
        
        Returns [0-1] urgency level where:
        0.0 = all needs satisfied
        1.0 = all needs maximally urgent
        """
        if not needs:
            return 0.0

        urgency_sum = 0.0
        count = len(needs)

        for need_name, need_data in needs.items():
            current_value = float(need_data.get('value', need_data.get('current_value', 50)))

            # Match NeedsManager satisfaction calculation
            if need_name.lower() == 'stamina':
                satisfaction = current_value / 100.0
            else:
                satisfaction = 1.0 - (current_value / 100.0)

            if 'satisfaction' in need_data:
                satisfaction = need_data['satisfaction']

            urgency = (1.0 - satisfaction)
            urgency_sum += urgency

        return urgency_sum / count if count else 0.0

    # --------------------------------------------------------------------------
    #  Implementation: Behavior
    # --------------------------------------------------------------------------
    def _calculate_behavior_metrics(
        self,
        behavior1: Dict[str, Any],
        behavior2: Dict[str, Any]
    ) -> Dict[str, float]:
        """
        Compare behaviors. Typically a boolean match approach, but you can
        expand for more nuanced logic.

        Returns something like:
            { 'matching': 1 or 0,
              'transition_significance': ... }
        """
        metrics = {}
        try:
            name1 = behavior1.get('name', None)
            name2 = behavior2.get('name', None)

            matching = (name1 == name2)
            metrics['matching'] = float(matching)

            # A simple measure of "transition significance"
            metrics['transition_significance'] = 1.0 - float(matching)

        except Exception as e:
            if self.logger:
                self.logger.log_error(f"Behavior metrics failed: {str(e)}")
        return metrics

    # --------------------------------------------------------------------------
    #  Implementation: Mood
    # --------------------------------------------------------------------------
    def _calculate_mood_metrics(
        self,
        mood1: Dict[str, Any],
        mood2: Dict[str, Any]
    ) -> Dict[str, float]:
        """
        Compare mood vectors similarly to how we handle emotion.
        Returns e.g.:
            { 
              'similarity': ...,
              'intensity_delta': ...
            }
        """
        metrics = {}
        try:
            similarity = self._calculate_mood_similarity(mood1, mood2)
            metrics['similarity'] = similarity

            intensity_delta = abs(
                self._get_mood_intensity(mood1) - 
                self._get_mood_intensity(mood2)
            )
            metrics['intensity_delta'] = intensity_delta

        except Exception as e:
            if self.logger:
                self.logger.log_error(f"Mood metrics failed: {str(e)}")

        return metrics

    def _calculate_mood_similarity(
        self,
        mood1: Dict[str, Any],
        mood2: Dict[str, Any]
    ) -> float:
        """
        Derived from the logic in body_memory.py & cognitive_memory.
        Typically a valence/arousal 2D check.

        Returns [0..1].
        """
        if not mood1 or not mood2:
            return 0.0

        try:
            valence_diff = abs(mood1.get('valence', 0.0) - mood2.get('valence', 0.0))
            arousal_diff = abs(mood1.get('arousal', 0.0) - mood2.get('arousal', 0.0))

            # The smaller the sum of diffs, the higher the similarity
            max_val = 2.0  # valence + arousal => each up to 1.0 difference
            sim = 1.0 - ((valence_diff + arousal_diff) / max_val)
            return max(0.0, min(1.0, sim))
        except:
            return 0.0

    def _get_mood_intensity(self, mood: Dict[str, Any]) -> float:
        if not mood:
            return 0.0
        val = float(mood.get('valence', 0.0))
        aro = float(mood.get('arousal', 0.0))
        return math.sqrt((val * val) + (aro * aro)) / math.sqrt(2)
