# pet/pet_context.py
"""
PetContext module.

Provides an access layer to retrieve the pet's current internal state, such as mood,
behavior, and needs. It acts as a snapshot provider without managing or storing
the actual state values.
"""

class PetContext:
    """
    PetContext serves as an access point to the pet's internal state.

    It retrieves up-to-date snapshots of the pet's current mood, behavior, and needs
    by querying the relevant modules. It does not manage or store these values.
    """

    def __init__(self, pet):
        """
        Initializes the PetContext with a reference to the pet instance.

        Args:
            pet (Pet): The pet instance.
        """
        self.pet = pet

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

    def get_visualization_context(self):
        return {
            'mood': self.get_current_mood(),
            'needs': self.get_current_needs(),
            'current_behavior': self.get_current_behavior().name if self.get_current_behavior() else None
        }

