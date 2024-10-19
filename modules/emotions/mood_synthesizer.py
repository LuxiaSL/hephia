# modules/emotions/mood_synthesizer.py

from .emotion import Emotion
from event_dispatcher import global_event_dispatcher, Event
import time

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
    Synthesizes the pet's mood based on recent emotions, needs, and behavior.
    """

    def __init__(self, pet_context):
        """
        Initializes the MoodSynthesizer.

        Args:
            pet_context (PetContext): methods to access info from other modules
        """
        self.context = pet_context
        self.current_mood = Mood()
        self.current_mood_name = 'neutral'
        self.decay_half_life = 300  # 5 minutes in seconds
        self.weights = {
            'emotions': 0.5,
            'needs': 0.3,
            'behavior': 0.2
        }
        self.setup_event_listeners()

    def setup_event_listeners(self):
        global_event_dispatcher.add_listener("need:changed", self.update_mood)
        global_event_dispatcher.add_listener("action:completed", self.update_mood)
        global_event_dispatcher.add_listener("behavior:changed", self.update_mood)
        global_event_dispatcher.add_listener("emotion:new", self.update_mood)

    def update_mood(self, event):
        # updates the pet's mood based on holistic information
        event_type = event.event_type
        event_data = event.data

        # retrieve info from context
        current_needs = self.pet_context.get_current_needs()
        recent_emotions = self.pet_context.get_recent_emotions()
        current_behavior = self.pet_context.get_current_behavior()
        # Calculate weighted valence and arousal from recent emotions
        new_mood = self._calculate_mood(event_type, event_data, current_needs, recent_emotions, current_behavior)
        
        if new_mood.valence != self.current_mood.valence or new_mood.arousal != self.current_mood.arousal:
            self.current_mood = new_mood
            new_emotional_state = self._map_mood_to_emotional_state(new_mood)
            if new_emotional_state != self.current_emotional_state:
                old_state = self.current_emotional_state
                self.current_emotional_state = new_emotional_state
                global_event_dispatcher.dispatch_event_sync(Event("mood:changed", {
                    "old_state": old_state,
                    "new_state": new_emotional_state
                }))
        
        # Dispatch event if emotional state has changed
        if new_emotional_state != self.current_emotional_state:
            old_state = self.current_emotional_state
            self.current_emotional_state = new_emotional_state
            global_event_dispatcher.dispatch_event_sync(Event("mood:changed", {
                "old_state": old_state,
                "new_state": new_emotional_state
            }))

    def _calculate_mood(self, event_type, event_data, current_needs, recent_emotions, current_behavior):
        weighted_valence = 0.0
        weighted_arousal = 0.0

        # Calculate emotion contribution
        if recent_emotions:
            emotion_valence = sum(e.valence * e.intensity for e in recent_emotions) / len(recent_emotions)
            emotion_arousal = sum(e.arousal * e.intensity for e in recent_emotions) / len(recent_emotions)
            weighted_valence += self.weights['emotions'] * emotion_valence
            weighted_arousal += self.weights['emotions'] * emotion_arousal

        # Calculate needs contribution
        need_satisfaction = sum(1 - value / 100 for value in current_needs.values()) / len(current_needs)
        need_valence = need_satisfaction * 2 - 1  # Map [0, 1] to [-1, 1]
        need_arousal = abs(need_valence)  # Higher arousal when needs are very satisfied or very unsatisfied
        weighted_valence += self.weights['needs'] * need_valence
        weighted_arousal += self.weights['needs'] * need_arousal

        # Calculate behavior contribution
        behavior_valence = 0.1 if current_behavior.__class__.__name__ == 'WalkBehavior' else -0.1
        behavior_arousal = 0.2 if current_behavior.__class__.__name__ == 'WalkBehavior' else -0.2
        weighted_valence += self.weights['behavior'] * behavior_valence
        weighted_arousal += self.weights['behavior'] * behavior_arousal

        return Mood(
            valence=max(-1.0, min(1.0, weighted_valence)),
            arousal=max(-1.0, min(1.0, weighted_arousal))
        )

    def _map_mood_to_name(self, valence, arousal):
        """
        Maps mood valence and arousal to a string.

        Args:
            valence (float): The valence of the mood.
            arousal (float): The arousal level of the mood.

        Returns:
            str: The resulting mood string.
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
        
    def get_current_mood(self):
        # raw mood state
        return self.current_mood

    def get_current_mood_name(self):
        # relevant string
        return self.current_mood_name
