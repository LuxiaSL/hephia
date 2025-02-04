# internal/internal_context.py
"""
InternalContext module.

Provides an access layer to retrieve the internal's current state (mood, behavior, needs, etc.)
and memory context. This version is updated to work with the async MemorySystemOrchestrator.
"""

import time
from typing import TYPE_CHECKING
from config import Config

if TYPE_CHECKING:
    from internal import Internal


class InternalContext:
    def __init__(self, internal: "Internal") -> None:
        """
        Initializes the InternalContext with a reference to the internal instance.
        
        Args:
            internal (Internal): The internal instance.
        """
        self.internal = internal
        self._recent_emotions_cache = None
        self._recent_emotions_cache_timestamp = 0.0

    async def get_api_context(self) -> dict:
        """Gets complete JSON-serializable context with type verification."""
        try:
            # Get mood state
            mood = self.internal.mood_synthesizer
            current_mood = mood.get_current_mood()
            mood_data = {
                'name': mood.get_current_mood_name(),
                'valence': float(current_mood.valence),
                'arousal': float(current_mood.arousal)
            }
            
            # Get behavior state
            behavior = self.internal.behavior_manager.current_behavior
            behavior_data = {
                'name': behavior.name if behavior else None,
                'active': behavior is not None
            }
            
            # Get emotional state from recent body memories (await the async call)
            emotional_state = await self.get_recent_emotions(use_timeframe=True)
            
            # Get needs state
            needs = self.internal.needs_manager.get_needs_summary()
            
            return {
                'mood': mood_data,
                'needs': needs,
                'behavior': behavior_data,
                'emotional_state': emotional_state
            }
        except Exception as e:
            print(f"DEBUG - Error in get_api_context: {str(e)}")
            raise

    async def get_recent_emotions(self, use_timeframe: bool = False) -> list:
        """
        Gets current emotional state from recent body memory nodes,
        and optionally current emotional stimuli, if they exist.
        
        Args:
            use_timeframe (bool): If True, query by timeframe; otherwise, get top 5 most recent.
        
        Returns:
            list: List of active emotional states from recent memory.
        """
        try:
            if use_timeframe:
                # Always bypass cache: perform a fresh query by time window.
                current_time = time.time()
                start_time = current_time - Config.get_exo_min_interval() * 1.75
                recent_nodes = await self.internal.memory_system.query_by_time_window(
                    start_time,
                    current_time,
                    network_type="body",
                    include_ghosted=False
                )
                # Process and return the emotional state from recent_nodes.
                emotional_state = []
                for node in recent_nodes:
                    if 'emotional_state' in node.processed_state:
                        emotional_data = node.processed_state['emotional_state']
                        if emotional_data and isinstance(emotional_data, list):
                            aggregate_state = emotional_data[0]
                            if aggregate_state:
                                emotional_state.append({
                                    'name': aggregate_state.get('name'),
                                    'intensity': float(aggregate_state.get('intensity', 0.0)),
                                    'valence': float(aggregate_state.get('valence', 0.0)), 
                                    'arousal': float(aggregate_state.get('arousal', 0.0))
                                })
                return emotional_state

            else:
                # For the default case, use caching.
                # Determine the latest timestamp from body memory nodes.
                nodes = self.internal.memory_system.body_network.nodes.values()
                latest_node_timestamp = max((node.timestamp for node in nodes), default=0.0)
                
                # If cache exists and nothing new has been added, return cached result.
                if (
                    self._recent_emotions_cache is not None and 
                    latest_node_timestamp <= self._recent_emotions_cache_timestamp
                ):
                    return self._recent_emotions_cache

                # Otherwise, perform the query for the top 5 most recent memories.
                recent_nodes = await self.internal.memory_system.get_recent_memories(
                    count=5,
                    network_type="body",
                    include_ghosted=False
                )
                emotional_state = []
                for node in recent_nodes:
                    if 'emotional_state' in node.processed_state:
                        emotional_data = node.processed_state['emotional_state']
                        if emotional_data and isinstance(emotional_data, list):
                            aggregate_state = emotional_data[0]
                            if aggregate_state:
                                emotional_state.append({
                                    'name': aggregate_state.get('name'),
                                    'intensity': float(aggregate_state.get('intensity', 0.0)),
                                    'valence': float(aggregate_state.get('valence', 0.0)), 
                                    'arousal': float(aggregate_state.get('arousal', 0.0))
                                })
                
                # Update cache.
                self._recent_emotions_cache = emotional_state
                self._recent_emotions_cache_timestamp = latest_node_timestamp
                
                return emotional_state

        except Exception as e:
            print(f"DEBUG - Error getting emotional state: {str(e)}")
            return []

    async def get_memory_context(self, is_cognitive: bool = False) -> dict:
        """
        Gets raw and processed state context for memory formation.
        
        Raw state captures exact system values for reconstruction,
        while processed state provides human-readable interpretations.
        
        Args:
            is_cognitive (bool): Whether to include cognitive information.
        
        Returns:
            dict: Contains 'raw_state' and 'processed_state' with relevant data.
        """
        try:
            # Get current raw states from various managers
            raw_state = {
                'needs': self.internal.needs_manager.get_needs_state(),
                'behavior': self.internal.behavior_manager.get_behavior_state(),
                'emotions': self.internal.emotional_processor.get_emotional_state(),
                'mood': self.internal.mood_synthesizer.get_mood_state()
            }
            if is_cognitive:
                # For cognitive context, get additional info from cognitive_bridge
                cog_state = self.internal.cognitive_bridge.get_cognitive_state()
                raw_state['cognitive'] = cog_state.get('raw_state', {})
            # Processed state: get readable versions
            current_mood = self.internal.mood_synthesizer.get_current_mood()
            current_behavior = self.internal.behavior_manager.current_behavior
            emotional_state = []
            if hasattr(self.internal.emotional_processor, 'current_stimulus'):
                stimulus = self.internal.emotional_processor.current_stimulus
                if stimulus and (abs(stimulus.valence) >= 0.001 or 
                                 abs(stimulus.arousal) >= 0.001 or 
                                 stimulus.intensity >= 0.001):
                    overall_state = {
                        'name': self.internal.emotional_processor._get_emotion_name_by_category(
                            self.internal.emotional_processor._categorize_stimulus(stimulus)
                        ),
                        'valence': float(stimulus.valence),
                        'arousal': float(stimulus.arousal),
                        'intensity': float(stimulus.intensity)
                    }
                    emotional_state.append(overall_state)
                    for vector in stimulus.active_vectors:
                        if vector.intensity >= 0.1:
                            emotional_state.append({
                                'name': vector.name,
                                'category': self.internal.emotional_processor._categorize_vector(
                                    vector.valence, vector.arousal
                                ),
                                'intensity': float(vector.intensity),
                                'source': vector.source_type
                            })
                else:
                    emotional_state = []
            processed_state = {
                'mood': {
                    'name': self.internal.mood_synthesizer.get_current_mood_name(),
                    'valence': float(current_mood.valence),
                    'arousal': float(current_mood.arousal)
                },
                'behavior': {
                    'name': current_behavior.name if current_behavior else None,
                    'active': current_behavior is not None
                },
                'needs': self.internal.needs_manager.get_needs_summary(),
                'emotional_state': emotional_state
            }
            if is_cognitive:
                processed_state['cognitive'] = self.internal.cognitive_bridge.get_cognitive_state().get('processed_state', {})
            return {
                'raw_state': raw_state,
                'processed_state': processed_state
            }
        except Exception as e:
            print(f"DEBUG - Error in get_memory_context: {str(e)}")
            raise

    async def get_full_context(self) -> dict:
        """Retrieve the complete internal context."""
        return {
            'mood': self.get_current_mood(),
            'recent_emotions': await self.get_recent_emotions(),
            'needs': self.get_current_needs(),
            'current_behavior': self.get_current_behavior()
        }

    def get_current_mood(self) -> dict:
        """Retrieves the current mood."""
        return {
            "name": self.internal.mood_synthesizer.get_current_mood_name(),
            "mood_object": self.internal.mood_synthesizer.get_current_mood()
        }

    def get_current_behavior(self):
        """Retrieves the current behavior."""
        return self.internal.behavior_manager.get_current_behavior()

    def get_current_needs(self) -> dict:
        """Retrieves current needs summary."""
        return self.internal.needs_manager.get_needs_summary()
