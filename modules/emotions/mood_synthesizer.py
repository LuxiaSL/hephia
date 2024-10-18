# modules/emotions/mood_synthesizer.py

from .emotion import Emotion
from event_dispatcher import global_event_dispatcher, Event

class Mood:
    """
    Represents the pet's current mood.
    """

    def __init__(self, valence=0.0, arousal=0.0):
        """
        Initializes a Mood instance.

        Args:
            valence (float): The valence of the mood (-1 to 1).
            arousal (float): The arousal level of the mood (-1 to 1).
        """
        self.valence = valence
        self.arousal = arousal

class MoodSynthesizer:
    """
    Generates and updates the pet's mood over time.
    """

    def __init__(self, memory_system, needs_manager):
        """
        Initializes the MoodSynthesizer.

        Args:
            memory_system (MemorySystem): Reference to the MemorySystem.
            needs_manager (NeedsManager): Reference to the NeedsManager.
        """
        self.memory_system = memory_system
        self.needs_manager = needs_manager
        self.current_mood = Mood()
        self.current_emotional_state = 'neutral'

    def get_current_mood(self):
        """
        Returns the current mood.

        Returns:
            Mood: The current mood.
        """
        return self.current_mood

    def get_current_emotional_state(self):
        """
        Returns the current emotional state.

        Returns:
            str: The current emotional state.
        """
        return self.current_emotional_state

    def update_mood(self):
        """
        Updates the mood based on recent emotions and needs.
        """
        # Get recent emotions from body memory
        recent_emotions = self.memory_system.body_memory.get_recent_emotions()

        if recent_emotions:
            # Compute weighted average valence and arousal
            total_intensity = sum(e.intensity for e in recent_emotions)
            if total_intensity == 0:
                avg_valence = 0.0
                avg_arousal = 0.0
            else:
                avg_valence = sum(e.valence * e.intensity for e in recent_emotions) / total_intensity
                avg_arousal = sum(e.arousal * e.intensity for e in recent_emotions) / total_intensity
        else:
            # Default to neutral if no recent emotions
            avg_valence = 0.0
            avg_arousal = 0.0

        # Update current mood
        old_mood = self.current_mood
        self.current_mood = Mood(avg_valence, avg_arousal)

        # Determine emotional state string based on mood
        new_emotional_state = self.map_mood_to_emotional_state(avg_valence, avg_arousal)

        # Override emotional state if specific needs are very low or high
        hunger = self.needs_manager.get_need_value('hunger')
        boredom = self.needs_manager.get_need_value('boredom')
        stamina = self.needs_manager.get_need_value('stamina')

        if hunger > 80:
            new_emotional_state = 'hungry'
        elif boredom > 80:
            new_emotional_state = 'bored'
        elif stamina < 20:
            new_emotional_state = 'tired'

        # Update the pet's emotional state if it has changed
        old_state = self.current_emotional_state
        if new_emotional_state != old_state:
            self.current_emotional_state = new_emotional_state
            # Dispatch event
            global_event_dispatcher.dispatch_event_sync(Event("mood:changed", {
                "old_state": old_state,
                "new_state": new_emotional_state
            }))

    def map_mood_to_emotional_state(self, valence, arousal):
        """
        Maps mood valence and arousal to an emotional state string.

        Args:
            valence (float): The valence of the mood.
            arousal (float): The arousal level of the mood.

        Returns:
            str: The emotional state string.
        """
        # Simple mapping example
        if valence > 0.5 and arousal > 0.5:
            return 'excited'
        elif valence > 0.0:
            return 'happy'
        elif valence < -0.5 and arousal > 0.5:
            return 'angry'
        elif valence < 0.0:
            return 'sad'
        else:
            return 'neutral'
