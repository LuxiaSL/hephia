# internal/internal_context.py
"""
InternalContext module.

Provides an access layer to retrieve the internal's current internal state, such as mood,
behavior, and needs. It acts as a snapshot provider without managing or storing
the actual state values.
"""

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
            mood = self.internal.mood_synthesizer
            current_mood = mood.get_current_mood()
            mood_data = {
                'name': mood.get_current_mood_name(),
                'valence': float(current_mood.valence),  # Ensure numeric
                'arousal': float(current_mood.arousal)   # Ensure numeric
            }
            
            behavior = self.internal.behavior_manager.current_behavior
            behavior_data = {
                'name': behavior.name if behavior else None,
                'active': behavior is not None
            }
            
            emotions = self.internal.memory_system.body_memory.get_recent_emotions()
            emotions_data = [
                {
                    'name': str(emotion.name),
                    'intensity': float(emotion.intensity),
                    'timestamp': emotion.timestamp if hasattr(emotion, 'timestamp') else None
                }
                for emotion in emotions
            ]
            
            needs = self.internal.needs_manager.get_needs_summary()
            
            return {
                'mood': mood_data,
                'needs': needs,
                'behavior': behavior_data,
                'emotions': emotions_data
            }
        except Exception as e:
            print(f"DEBUG - Error in get_api_context: {str(e)}")
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

    def get_recent_emotions(self):
        """
        Retrieves the recent emotions from the MemorySystem's BodyMemory.

        Returns:
            list: A list of recent Emotion instances.
        """
        return self.internal.memory_system.body_memory.get_recent_emotions()

    def get_full_context(self):
        return {
            'mood': self.get_current_mood(),
            'recent_emotions': self.get_recent_emotions(),
            'needs': self.get_current_needs(),
            'current_behavior': self.get_current_behavior()
        }

