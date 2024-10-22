# modules/emotions/mood_synthesizer.py

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

    MOOD_MAPPINGS = {
        'excited': {'valence': 0.75, 'arousal': 0.75},
        'happy': {'valence': 0.5, 'arousal': 0.25},
        'content': {'valence': 0.25, 'arousal': -0.25},
        'calm': {'valence': 0.1, 'arousal': -0.5},
        'neutral': {'valence': 0.0, 'arousal': 0.0},
        'bored': {'valence': -0.25, 'arousal': -0.5},
        'sad': {'valence': -0.5, 'arousal': -0.25},
        'angry': {'valence': -0.75, 'arousal': 0.75},
        'frustrated': {'valence': -0.5, 'arousal': 0.5}
    }

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
        
        # if the mood has changed been altered
        if new_mood.valence != self.current_mood.valence or new_mood.arousal != self.current_mood.arousal:
            # find relevant name
            self.current_mood = new_mood
            new_name = self._map_mood_to_name(new_mood)
            # if name is updated
            if new_name != self.current_mood_name:
                old_name = self.current_mood_name
                self.current_mood_name = new_name
                #set and send
                global_event_dispatcher.dispatch_event_sync(Event("mood:changed", {
                    "old_name": old_name,
                    "new_name": new_name
                }))

    def _calculate_mood(self, event_type, event_data, current_needs, recent_emotions, current_behavior):
        weighted_valence = 0.0
        weighted_arousal = 0.0

        # emotion contribution
        if recent_emotions:
            emotion_valence = sum(e.valence * e.intensity for e in recent_emotions) / len(recent_emotions)
            emotion_arousal = sum(e.arousal * e.intensity for e in recent_emotions) / len(recent_emotions)
            weighted_valence += self.weights['emotions'] * emotion_valence
            weighted_arousal += self.weights['emotions'] * emotion_arousal

        # need contribution 
        total_satisfaction = 0
        need_count = 0
        for need_info in current_needs.values():
            total_satisfaction += need_info['satisfaction']
            need_count += 1

        avg_satisfaction = total_satisfaction / need_count if need_count > 0 else 0
        # map to valence [0, 1] -> [-1, 1] and arousal
        need_valence = avg_satisfaction * 2 - 1
        need_arousal = abs(need_valence)

        # Apply weights to valence and arousal contributions
        weighted_valence += self.weights['needs'] * need_valence
        weighted_arousal += self.weights['needs'] * need_arousal

        # calculate behavior contribution
        behavior_valence = 0.1 if current_behavior.__class__.__name__ == 'WalkBehavior' else -0.1
        behavior_arousal = 0.2 if current_behavior.__class__.__name__ == 'WalkBehavior' else -0.2
        weighted_valence += self.weights['behavior'] * behavior_valence
        weighted_arousal += self.weights['behavior'] * behavior_arousal

        # eventually below, calculate the memory influence based on sentiment analysis via cognitive processing of non-body memory
        # when considering episodic memory; gradient scaling

        # done, send it
        return Mood(
            valence=max(-1.0, min(1.0, weighted_valence)),
            arousal=max(-1.0, min(1.0, weighted_arousal))
        )

    def _map_mood_to_name(self, valence, arousal):
        """
        Maps mood valence and arousal to a string.

        Args:
            valence (float): The valence of the mood (-1 to 1).
            arousal (float): The arousal level of the mood (-1 to 1).

        Returns:
            str: The resulting mood string.
        """
        # estimate vicinity to mood
        distances = {
            mood: ((v['valence'] - valence) ** 2 + (v['arousal'] - arousal) ** 2) ** 0.5
            for mood, v in self.MOOD_MAPPINGS.items()
        }
        
        # return closest match
        return min(distances, key=distances.get)
    
    def _map_name_to_mood(self, mood_name):
        """
        Maps a mood name to a Mood object.

        Args:
            mood_name (str): The name of the mood.

        Returns:
            Mood: A Mood object with the corresponding valence and arousal.

        Raises:
            ValueError: If an invalid mood name is provided.
        """
        # Convert the mood name to lowercase for case-insensitive matching
        mood_name = mood_name.lower()
        
        # Check if the mood name exists in our mapping
        if mood_name in self.MOOD_MAPPINGS:
            mood_values = self.MOOD_MAPPINGS[mood_name]
            return Mood(valence=mood_values['valence'], arousal=mood_values['arousal'])
        else:
            # If an invalid mood name is provided, raise a ValueError
            raise ValueError(f"Invalid mood name: {mood_name}")
        
    def get_current_mood(self):
        # raw mood state
        return self.current_mood

    def get_current_mood_name(self):
        # relevant string
        return self.current_mood_name
