# internal/internal_context.py
"""
InternalContext module.

Provides an access layer to retrieve the internal's current internal state, such as mood,
behavior, and needs. It acts as a snapshot provider without managing or storing
the actual state values.
"""
from config import Config
import time

class InternalContext:

    def __init__(self, internal):
        """
        Initializes the InternalContext with a reference to the internal instance.

        Args:
            internal (Internal): The internal instance.
        """
        self.internal = internal

    def get_api_context(self):
        """Gets complete JSON-serializable context with type verification."""
        try:
            # Get mood state
            mood = self.internal.mood_synthesizer
            current_mood = mood.get_current_mood()
            mood_data = {
                'name': mood.get_current_mood_name(),
                'valence': float(current_mood.valence),  # Ensure numeric
                'arousal': float(current_mood.arousal)   # Ensure numeric
            }
            
            # Get behavior state
            behavior = self.internal.behavior_manager.current_behavior
            behavior_data = {
                'name': behavior.name if behavior else None,
                'active': behavior is not None
            }
            
            # Get emotional state from body memory's current state
            emotional_state = self.get_recent_emotions(use_timeframe=True)
            
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
        
    def get_recent_emotions(self, use_timeframe=False):
        """
        Gets current emotional state from recent body memory nodes.
        
        Args:
            use_timeframe (bool): If True, query by timeframe, otherwise get top 5 most recent
        
        Returns:
            list: List of active emotional states from recent memory
        """
        try:
            emotional_state = []
            
            if use_timeframe:
                current_time = time.time()
                # Wide interval to catch any recent emotions
                start_time = current_time - Config.EXO_MIN_INTERVAL * 1.75
                
                # Get nodes in timeframe
                recent_nodes = self.internal.memory_system.body_memory.query_by_time_window(
                    start_time,
                    current_time,
                    include_ghosted=False
                )
            else:
                # Get 5 most recent memories
                recent_nodes = self.internal.memory_system.body_memory.get_recent_memories(
                    count=5,
                    include_ghosted=False
                )
            
            # Extract emotional states from memory nodes
            for node in recent_nodes:
                # Get the aggregate emotional state from the memory node
                if 'emotional_state' in node.processed_state:
                    emotional_data = node.processed_state['emotional_state']
                    # The first item is always the overall state, others are individual vectors
                    if emotional_data and isinstance(emotional_data, list):
                        aggregate_state = emotional_data[0]  # Get the overall emotional state
                        if aggregate_state:
                            emotional_state.append({
                                'name': aggregate_state.get('name'),
                                'intensity': float(aggregate_state.get('intensity', 0.0)),
                                'valence': float(aggregate_state.get('valence', 0.0)), 
                                'arousal': float(aggregate_state.get('arousal', 0.0))
                            })

            return emotional_state
            
        except Exception as e:
            print(f"DEBUG - Error getting emotional state: {str(e)}")
            return []

    def get_memory_context(self, is_cognitive=False):
        """
        Gets raw and processed state context for memory formation.
        
        Raw state captures exact system values for reconstruction,
        while processed state provides human-readable interpretations.
        
        Args:
            get_cognitive (bool): Whether to include cognitive information
            
        Returns:
            dict: Contains 'raw_state' and 'processed_state' sections with relevant data
        """
        try:
            # Get current raw states
            raw_state = {
                'needs': self.internal.needs_manager.get_needs_state(),
                'behavior': self.internal.behavior_manager.get_behavior_state(),
                'emotions': self.internal.emotional_processor.get_emotional_state(),
                'mood': self.internal.mood_synthesizer.get_mood_state()
            }

            if is_cognitive:
                cog_state = self.internal.cognitive_bridge.get_cognitive_state()
                raw_state['cognitive'] = cog_state.get('raw_state', {})

            # Get current processed/readable states
            current_mood = self.internal.mood_synthesizer.get_current_mood()
            current_behavior = self.internal.behavior_manager.current_behavior
            
            # Build emotional stimulus data for processed state
            # NOTE: The EmotionalProcessor class doesn't expose these values as properties
            # and relies on internal methods for categorization and naming
            # Consider adding properties to EmotionalProcessor/EmotionalStimulus for:
            # - emotion_name
            # - emotion_category 
            # - vector_categories
            emotional_state = []
            if hasattr(self.internal.emotional_processor, 'current_stimulus'):
                stimulus = self.internal.emotional_processor.current_stimulus
                
                # Check if stimulus exists and has meaningful values
                if stimulus and (abs(stimulus.valence) >= 0.001 or 
                               abs(stimulus.arousal) >= 0.001 or 
                               stimulus.intensity >= 0.001):
                    # Get overall emotional state
                    overall_state = {
                        'name': self.internal.emotional_processor._get_emotion_name_by_category(
                            self.internal.emotional_processor._categorize_stimulus(stimulus)
                        ),
                        'valence': float(stimulus.valence),
                        'arousal': float(stimulus.arousal),
                        'intensity': float(stimulus.intensity)
                    }
                    emotional_state.append(overall_state)
                else:
                    # Return an empty emotional state if no meaningful stimulus exists
                    emotional_state = []
                
                # Include active vector categories for detailed view
                for vector in stimulus.active_vectors:
                    if vector.intensity >= 0.1:  # Only include significant contributors
                        emotional_state.append({
                            'name': vector.name,
                            'category': self.internal.emotional_processor._categorize_vector(
                                vector.valence, vector.arousal
                            ),
                            'intensity': float(vector.intensity),
                            'source': vector.source_type
                        })

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
                processed_state['cognitive'] = cog_state.get('processed_state', {})

            return {
                'raw_state': raw_state,
                'processed_state': processed_state
            }
            
        except Exception as e:
            print(f"DEBUG - Error in get_memory_context: {str(e)}")
            raise

    def get_current_mood(self):
        """
        Retrieves the internal's current mood from the MoodSynthesizer in a structured format.

        Returns:
            dict: A dictionary with the mood name and mood object.
        """
        return {
            "name": self.internal.mood_synthesizer.get_current_mood_name(),
            "mood_object": self.internal.mood_synthesizer.get_current_mood()
        }

    def get_current_behavior(self):
        """
        Retrieves the internal's current behavior from the BehaviorManager.

        Returns:
            Behavior: The current behavior of the internal.
        """
        return self.internal.behavior_manager.get_current_behavior()

    def get_current_needs(self):
        """
        Retrieves the internal's current needs from the NeedsManager.

        Returns:
            dict: A dictionary of need names to their current values and satisfaction levels
        """
        return self.internal.needs_manager.get_needs_summary()

    def get_full_context(self):
        return {
            'mood': self.get_current_mood(),
            'recent_emotions': self.get_recent_emotions(),
            'needs': self.get_current_needs(),
            'current_behavior': self.get_current_behavior()
        }

