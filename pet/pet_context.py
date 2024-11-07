# pet/pet_context.py
"""
PetContext module.

Provides an access layer to retrieve the pet's current internal state, such as mood,
behavior, and needs. It acts as a snapshot provider without managing or storing
the actual state values.
"""

class PetContext:

    def __init__(self, pet):
        """
        Initializes the PetContext with a reference to the pet instance.

        Args:
            pet (Pet): The pet instance.
        """
        self.pet = pet

    def get_api_context(self):
        """Gets complete JSON-serializable context with type verification."""
        try:
            mood = self.pet.mood_synthesizer
            current_mood = mood.get_current_mood()
            mood_data = {
                'name': mood.get_current_mood_name(),
                'valence': float(current_mood.valence),  # Ensure numeric
                'arousal': float(current_mood.arousal)   # Ensure numeric
            }
            
            behavior = self.pet.behavior_manager.current_behavior
            behavior_data = {
                'name': behavior.name if behavior else None,
                'active': behavior is not None
            }
            
            emotions = self.pet.memory_system.body_memory.get_recent_emotions()
            emotions_data = [
                {
                    'name': str(emotion.name),
                    'intensity': float(emotion.intensity),
                    'timestamp': emotion.timestamp.isoformat() if hasattr(emotion, 'timestamp') else None
                }
                for emotion in emotions
            ]
            
            needs = self.pet.needs_manager.get_needs_summary()
            
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
        Retrieves the pet's current mood from the MoodSynthesizer in a structured format.

        Returns:
            dict: A dictionary with the mood name and mood object.
        """
        return {
            "name": self.pet.mood_synthesizer.get_current_mood_name(),
            "mood_object": self.pet.mood_synthesizer.get_current_mood()
        }

    def get_current_behavior(self):
        """
        Retrieves the pet's current behavior from the BehaviorManager.

        Returns:
            Behavior: The current behavior of the pet.
        """
        return self.pet.behavior_manager.get_current_behavior()

    def get_current_needs(self):
        """
        Retrieves the pet's current needs from the NeedsManager.

        Returns:
            dict: A dictionary of need names to their current values and satisfaction levels
        """
        return self.pet.needs_manager.get_needs_summary()

    def get_recent_emotions(self):
        """
        Retrieves the recent emotions from the MemorySystem's BodyMemory.

        Returns:
            list: A list of recent Emotion instances.
        """
        return self.pet.memory_system.body_memory.get_recent_emotions()

    def get_full_context(self):
        return {
            'mood': self.get_current_mood(),
            'recent_emotions': self.get_recent_emotions(),
            'needs': self.get_current_needs(),
            'current_behavior': self.get_current_behavior()
        }

