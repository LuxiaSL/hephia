# modules/emotions/emotional_processor.py

from event_dispatcher import global_event_dispatcher, Event
from .emotion import Emotion

class EmotionalProcessor:
    """
    Processes events and generates appropriate emotional responses.
    """

    def __init__(self, cognitive_processor, mood_synthesizer, memory_system):
        """
        Initializes the EmotionalProcessor.

        Args:
            cognitive_processor (CognitiveProcessor): Reference to the pet's cognitive processor.
            mood_synthesizer (MoodSynthesizer): Reference to the pet's mood synthesizer.
            memory_system (MemorySystem): Reference to the pet's memory system.
        """
        self.cognitive_processor = cognitive_processor
        self.mood_synthesizer = mood_synthesizer
        self.memory_system = memory_system
        self.body_memory = self.memory_system.body_memory
        self.setup_event_listeners()

    def setup_event_listeners(self):
        """
        Sets up event listeners for various relevant events.
        """
        global_event_dispatcher.add_listener("need:changed", self.on_need_update)
        global_event_dispatcher.add_listener("action:completed", self.on_action_performed)
        global_event_dispatcher.add_listener("behavior:changed", self.on_behavior_changed)
        # Add more listeners as needed

    def on_need_update(self, event):
        """
        Handles need update events.
        """
        need_name = event.data['need_name']
        old_value = event.data['old_value']
        new_value = event.data['new_value']
        self.process_event('need_update', {
            'need_name': need_name,
            'old_value': old_value,
            'new_value': new_value
        })

    def on_action_performed(self, event):
        """
        Handles action performed events.
        """
        action_name = event.data['action_name']
        self.process_event('action_performed', {'action_name': action_name})

    def on_behavior_changed(self, event):
        """
        Handles behavior change events.
        """
        new_behavior = event.data['new_behavior']
        self.process_event('behavior_changed', {'new_behavior': new_behavior})

    def process_event(self, event_type, event_data):
        """
        Processes an event and generates an appropriate emotional response.

        Args:
            event_type (str): The type of event.
            event_data (dict): Data associated with the event.
        """
        # Generate initial emotional response
        initial_response = self.generate_initial_response(event_type, event_data)
        if initial_response is None:
        # No significant emotion generated; do nothing
            return
        #print(f"Initial emotional response: {initial_response}")
        self.body_memory.log(initial_response)

        # Gather context for cognitive mediation
        context = self.gather_context()
        # Mediate the emotional response through cognitive processing
        mediated_response = self.cognitive_processor.mediate_emotion(initial_response, event_type, event_data, context)
        # Apply mood influence to the mediated response
        final_response = self.apply_mood_influence(mediated_response)
        #print(f"Final emotional response: {final_response}")

        # Log the final emotional response and dispatch an event
        self.body_memory.log(final_response)
        global_event_dispatcher.dispatch_event_sync(Event("emotion:new", {
            "emotion": final_response
        }))

    def generate_initial_response(self, event_type, event_data):
        """
        Generates an initial emotional response based on the event type and data.

        Args:
            event_type (str): The type of event.
            event_data (dict): Data associated with the event.

        Returns:
            Emotion: The initial emotional response, or None if no response is generated.
        """
        # This is a simplified implementation. In a real scenario, this would be much more complex,
        # taking into account the pet's current state, past experiences, etc.
        emotion_mapping = {
            'need_update': self._process_need_update,
            'action_performed': self._process_action_performed,
            'behavior_changed': self._process_behavior_changed
        }

        # Get the appropriate processing function or default to neutral emotion
        process_func = emotion_mapping.get(event_type, lambda x: Emotion('neutral', 0, 0, 0.1))
        return process_func(event_data)

    def _process_need_update(self, data):
        """
        Processes a need update event and generates an appropriate emotional response.

        Args:
            data (dict): Data about the need update.

        Returns:
            Emotion: The generated emotional response, or None if the change is insignificant.
        """
        need_name = data['need_name']
        old_value = data['old_value']
        new_value = data['new_value']
        change = new_value - old_value

        # Ignore small changes (hypersensitivity)
        if abs(change) < 5:
            return None

        # Generate emotions based on the type of need and direction of change
        if need_name in ['hunger', 'thirst']:
            if change > 0:
                return Emotion('discomfort', -0.3, 0.2, min(abs(change) / 20, 1))
            else:
                return Emotion('satisfaction', 0.3, -0.1, min(abs(change) / 20, 1))
        elif need_name == 'boredom':
            if change > 0:
                return Emotion('frustration', -0.2, 0.3, min(abs(change) / 20, 1))
            else:
                return Emotion('interest', 0.2, 0.3, min(abs(change) / 20, 1))
        elif need_name == 'stamina':
            if change > 0:
                return Emotion('energetic', 0.2, 0.4, min(abs(change) / 20, 1))
            else:
                return Emotion('tired', -0.2, -0.3, min(abs(change) / 20, 1))

    def _process_action_performed(self, data):
        """
        Processes an action performed event and generates an appropriate emotional response.

        Args:
            data (dict): Data about the action performed.

        Returns:
            Emotion: The generated emotional response.
        """
        action_name = data['action_name']
        # Map actions to emotions
        action_emotions = {
            'feed': Emotion('contentment', 0.5, 0.2, 0.6),
            'give_water': Emotion('refreshed', 0.4, 0.1, 0.5),
            'play': Emotion('joy', 0.7, 0.8, 0.7),
            'rest': Emotion('relaxation', 0.3, -0.4, 0.5)
        }
        return action_emotions.get(action_name, Emotion('interest', 0.1, 0.3, 0.2))

    def _process_behavior_changed(self, data):
        """
        Processes a behavior change event and generates an appropriate emotional response.

        Args:
            data (dict): Data about the behavior change.

        Returns:
            Emotion: The generated emotional response.
        """
        new_behavior = data['new_behavior']
        # Map behaviors to emotions
        behavior_emotions = {
            'idle': Emotion('calm', 0.1, -0.2, 0.3),
            'walk': Emotion('curious', 0.3, 0.4, 0.5),
            # Add more behaviors as they are implemented
        }
        return behavior_emotions.get(new_behavior, Emotion('curiosity', 0.2, 0.4, 0.3))

    def gather_context(self):
        """
        Gathers contextual information for cognitive processing.

        Returns:
            dict: A dictionary containing current mood, recent emotions, needs, and current behavior.
        """
        return {
            'current_mood': self.mood_synthesizer.get_current_mood(),
            'recent_emotions': self.body_memory.get_recent_emotions(),
            'needs': {need: self.mood_synthesizer.needs_manager.get_need_value(need)
                      for need in ['hunger', 'thirst', 'boredom', 'stamina']},
            'current_behavior': self.mood_synthesizer.needs_manager.behavior_manager.get_current_behavior()
        }

    def apply_mood_influence(self, emotion):
        """
        Applies the influence of the current mood to an emotion.

        Args:
            emotion (Emotion): The emotion to be influenced.

        Returns:
            Emotion: The emotion after mood influence has been applied.
        """
        current_mood = self.mood_synthesizer.get_current_mood()
        
        # Adjust emotion valence and arousal towards the current mood
        adjusted_valence = (emotion.valence + current_mood.valence) / 2
        adjusted_arousal = (emotion.arousal + current_mood.arousal) / 2
        
        # Adjust intensity based on how different the emotion is from the current mood
        intensity_factor = 1 + (abs(emotion.valence - current_mood.valence) + 
                                abs(emotion.arousal - current_mood.arousal)) / 2
        adjusted_intensity = min(emotion.intensity * intensity_factor, 1.0)

        return Emotion(
            name=emotion.name,
            valence=adjusted_valence,
            arousal=adjusted_arousal,
            intensity=adjusted_intensity
        )
