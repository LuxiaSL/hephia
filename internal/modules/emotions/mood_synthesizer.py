# modules/emotions/mood_synthesizer.py

"""
Mood Synthesis System:

Generates the ongoing emotional state by integrating:
- Recent emotional experiences {immediate feelings} -> {short-term impact}
- Need satisfaction levels {physical state} -> {baseline mood}
- Current behavior {activity influence} -> {embodied state}

Unlike emotions which respond to specific events, mood represents a more
stable emotional baseline that colors perception of events and
influences its behavioral tendencies.

Future integrations:
- Memory-based mood influences through cognitive processing
- Cognitive shaping of mood over time, indirectly via emotion influence & direct via mood influence
"""

from event_dispatcher import global_event_dispatcher, Event
import time

class Mood:
    """
    Represents the current mood.
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
    Synthesizes the internal's mood based on recent emotions, needs, and behavior.
    """

    def __init__(self, internal_context):
        """
        Initializes the MoodSynthesizer.

        Args:
            internal_context (InternalContext): methods to access info from other modules
        """
        self.context = internal_context
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

    BEHAVIOR_MOOD_INFLUENCES = {
        'idle': {'valence': 0.1, 'arousal': -0.2},  # default state
        'walk': {'valence': 0.2, 'arousal': 0.3},   # Active, engaged
        'chase': {'valence': 0.6, 'arousal': 0.7},   # Energetic, excited
        'sleep': {'valence': 0.1, 'arousal': -0.7},  # Peaceful, low arousal
        'relax': {'valence': 0.3, 'arousal': -0.3},  # resting from action, slightly low arousal
        # Future behaviors...
    }

    def setup_event_listeners(self):
        global_event_dispatcher.add_listener("need:changed", self.update_mood)
        global_event_dispatcher.add_listener("action:completed", self.update_mood)
        global_event_dispatcher.add_listener("behavior:changed", self.update_mood)
        global_event_dispatcher.add_listener("emotion:new", self.update_mood)

        global_event_dispatcher.add_listener("memory:echo", self._handle_memory_echo)
        global_event_dispatcher.add_listener("cognitive:mood:meditation", self.process_meditation)

    def update_mood(self, event):
        # Retrieve current needs, emotions, and behavior from the context
        current_needs = self.context.get_current_needs()
        recent_emotions = self.context.get_recent_emotions()
        current_behavior = self.context.get_current_behavior()

        # Calculate the new mood object based on current context
        new_mood = self._calculate_mood(current_needs, recent_emotions, current_behavior)

        # Fallback to neutral mood if no change is detected or no emotions are present
        if new_mood.valence == 0 and new_mood.arousal == 0:
            new_mood = Mood(valence=0.0, arousal=0.0)  # Explicitly set to neutral

        # Check if mood has changed in terms of valence and arousal
        if new_mood.valence != self.current_mood.valence or new_mood.arousal != self.current_mood.arousal:
            self.current_mood = new_mood
            new_name = self._map_mood_to_name(new_mood)  # Map mood object to its descriptive name
            
            # If the mood name has also updated
            if new_name != self.current_mood_name:
                old_name = self.current_mood_name
                self.current_mood_name = new_name
                
                # Dispatch the mood change event with consistent dictionary formatting
                global_event_dispatcher.dispatch_event_sync(Event("mood:changed", {
                    "old_name": old_name,
                    "new_name": new_name
                }))

    def _calculate_mood(self, current_needs, recent_emotions, current_behavior):
        weighted_valence = 0.0
        weighted_arousal = 0.0

        # emotion contribution
        if recent_emotions:
            emotion_valence = sum(e['valence'] * e['intensity'] for e in recent_emotions) / len(recent_emotions)
            emotion_arousal = sum(e['arousal'] * e['intensity'] for e in recent_emotions) / len(recent_emotions)
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
        behavior_name = current_behavior.name
        if behavior_name in self.BEHAVIOR_MOOD_INFLUENCES:
            influence = self.BEHAVIOR_MOOD_INFLUENCES[behavior_name]
            weighted_valence += self.weights['behavior'] * influence['valence']
            weighted_arousal += self.weights['behavior'] * influence['arousal']

        # eventually below, calculate the memory influence based on sentiment analysis via cognitive processing of non-body memory
        # cognitive shaping of mood over time, indirectly via emotion influence & direct via mood influence
        # done, send it
        return Mood(
            valence=max(-1.0, min(1.0, weighted_valence)),
            arousal=max(-1.0, min(1.0, weighted_arousal))
        )

    def _map_mood_to_name(self, _mood):
        """
        Maps mood to a string.

        Args:
            _mood (Mood): contains valence/arousal to use

        Returns:
            str: The resulting mood string.
        """
        # estimate vicinity to mood
        distances = {
            mood: ((v['valence'] - _mood.valence) ** 2 + (v['arousal'] - _mood.arousal) ** 2) ** 0.5
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

    def get_mood_state(self):
        """Gets the persistent state of the mood system."""
        return {
            "mood": {
                "valence": self.current_mood.valence,
                "arousal": self.current_mood.arousal,
                "name": self.current_mood_name
            }
        }

    def set_mood_state(self, state):
        """Sets the mood system state."""
        if not state:
            return
            
        mood_data = state.get("mood", {})
        self.current_mood = Mood(
            valence=mood_data.get("valence", 0.0),
            arousal=mood_data.get("arousal", 0.0)
        )
        self.current_mood_name = mood_data.get("name", "neutral")

    def _handle_memory_echo(self, event):
        """
        Handle memory echo effects on mood by creating a subtle resonance influence
        based on the remembered emotional state.
        """
        echo_data = event.data
        if not echo_data or 'metadata' not in echo_data:
            return
            
        metadata = echo_data['metadata']
        
        # Extract mood influence from metadata if available
        mood_data = metadata.get('mood', {}).get('mood', {})
        if mood_data and 'valence' in mood_data and 'arousal' in mood_data:
            # Scale echo influence by the echo intensity 
            echo_intensity = echo_data.get('intensity', 0.3) * 0.4  # Reduced influence
            echo_valence = mood_data['valence'] * echo_intensity  
            echo_arousal = mood_data['arousal'] * echo_intensity

            # Create temporary mood nudge combining current mood and echo
            echo_mood = Mood(
                valence=self.current_mood.valence + (echo_valence * self.weights['emotions']),
                arousal=self.current_mood.arousal + (echo_arousal * self.weights['emotions'])
            )

            # Update mood if the echo influence is significant
            if abs(echo_mood.valence - self.current_mood.valence) > 0.1 or \
               abs(echo_mood.arousal - self.current_mood.arousal) > 0.1:
                
                # Clamp values
                echo_mood.valence = max(-1.0, min(1.0, echo_mood.valence))
                echo_mood.arousal = max(-1.0, min(1.0, echo_mood.arousal))
                
                # Apply the echo-influenced mood
                old_name = self.current_mood_name
                self.current_mood = echo_mood
                new_name = self._map_mood_to_name(echo_mood)
                
                if new_name != old_name:
                    self.current_mood_name = new_name
                    global_event_dispatcher.dispatch_event_sync(Event("mood:changed", {
                        "old_name": old_name,
                        "new_name": new_name,
                        "mood_object": echo_mood,
                        "source": "memory_echo"
                    }))

    def process_meditation(self, event):
        """
        Process meditation effects on mood by creating a directed mood shift based on 
        meditation type and intensity.
        """
        try:
            meditation_data = event.data
            meditation_type = meditation_data.get('type')
            intensity = meditation_data.get('intensity', 0.5)
            duration = meditation_data.get('duration', 1)

            # Map meditation types to mood influences
            meditation_influences = {
                'elevation': {'valence': 0.3, 'arousal': 0.2},
                'calming': {'valence': 0.1, 'arousal': -0.4},
                'deepening': {'valence': -0.2, 'arousal': -0.3},
                'focusing': {'valence': 0.2, 'arousal': 0.3},
                'activation': {'valence': 0.3, 'arousal': 0.4}
            }

            if meditation_type in meditation_influences:
                influence = meditation_influences[meditation_type]
                
                # Scale influence by intensity and duration
                scaled_valence = influence['valence'] * intensity * (1 + 0.2 * duration)
                scaled_arousal = influence['arousal'] * intensity * (1 + 0.2 * duration)
                
                # Create meditation-influenced mood
                meditation_mood = Mood(
                    valence=self.current_mood.valence + (scaled_valence * self.weights['emotions']),
                    arousal=self.current_mood.arousal + (scaled_arousal * self.weights['emotions'])
                )
                
                # Clamp values
                meditation_mood.valence = max(-1.0, min(1.0, meditation_mood.valence))
                meditation_mood.arousal = max(-1.0, min(1.0, meditation_mood.arousal))
                
                # Update mood if change is significant
                if abs(meditation_mood.valence - self.current_mood.valence) > 0.1 or \
                   abs(meditation_mood.arousal - self.current_mood.arousal) > 0.1:
                    
                    old_name = self.current_mood_name
                    self.current_mood = meditation_mood
                    new_name = self._map_mood_to_name(meditation_mood)
                    
                    if new_name != old_name:
                        self.current_mood_name = new_name
                        global_event_dispatcher.dispatch_event_sync(Event("mood:changed", {
                            "old_name": old_name,
                            "new_name": new_name,
                            "mood_object": meditation_mood,
                            "source": "meditation"
                        }))

        except Exception as e:
            print(f"Error processing meditation effect: {str(e)}")