# modules/emotions/emotional_processor.py

from event_dispatcher import global_event_dispatcher, Event
from .emotion import Emotion

class EmotionalProcessor:
    """
    Processes various events to generate emotional responses.
    """
    #ignore this. shouldnt be used yet unless we have a need for it in the future. otherwise, control throttle of emotions visual-side.
    FOREGROUND_THRESHOLD = 0.5  # Threshold for significant emotional changes

    def __init__(self, cognitive_processor, mood_synthesizer, memory_system):
        """
        Initializes the EmotionalProcessor.

        Args:
            cognitive_processor (CognitiveProcessor): Reference to the CognitiveProcessor.
            mood_synthesizer (MoodSynthesizer): Reference to the MoodSynthesizer.
            memory_system (MemorySystem): Reference to the MemorySystem.
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
        Processes an event to generate an emotional response.

        Args:
            event_type (str): The type of event.
            event_data (dict): The event data.
        """
        initial_response = self.generate_initial_response(event_type, event_data)
        if initial_response is None:
        # No significant emotion generated; do nothing
            return
        #print(f"Initial emotional response: {initial_response}")
        self.body_memory.log(initial_response)

        mediated_response = self.cognitive_processor.mediate_emotion(initial_response, event_type, event_data)
        #print(f"Mediated emotional response: {mediated_response}")
        final_response = self.apply_mood_influence(mediated_response)
        #print(f"Final emotional response: {final_response}")

        print(f"Dispatching new emotion event: {final_response}")
        global_event_dispatcher.dispatch_event_sync(Event("emotion:new", {
            "emotion": final_response
        }))

    def generate_initial_response(self, event_type, event_data):
        """
        Generates the initial emotional response based on the event.

        Args:
            event_type (str): The type of event.
            event_data (dict): The event data.

        Returns:
            Emotion: The initial emotional response.
        """
        # This is a simplified implementation. In a real scenario, this would be much more complex,
        # taking into account the pet's current state, past experiences, etc.
        emotion_mapping = {
            'need_update': self._process_need_update,
            'action_performed': self._process_action_performed,
            'behavior_changed': self._process_behavior_changed
        }

        process_func = emotion_mapping.get(event_type, lambda x: Emotion('neutral', 0, 0, 0.1))
        return process_func(event_data)

    def _process_need_update(self, data):
        need_name = data['need_name']
        old_value = data['old_value']
        new_value = data['new_value']
        change = abs(new_value - old_value)
        SIGNIFICANCE_THRESHOLD = 5.0  

        if change < SIGNIFICANCE_THRESHOLD:
        # Negligible change; do not generate an emotion
            return None 
        
        #here; need specific logic to determine whether the need *wants* to be going positive or negative.
        #perhaps we can work the needs such that their definitions all align with them wanting to go in the same direction
        
        if new_value > old_value:
            return Emotion('concern', -0.3, 0.2, 0.4)
        else:
            return Emotion('satisfaction', 0.3, 0.1, 0.3)

    def _process_action_performed(self, data):
        action_name = data['action_name']
        # Simplified mapping of actions to emotions
        print(action_name)
        action_emotions = {
            'feed': Emotion('contentment', 0.5, 0.2, 0.6),
            'play': Emotion('joy', 0.7, 0.8, 0.7),
            'rest': Emotion('relaxation', 0.3, -0.4, 0.5)
        }
        return action_emotions.get(action_name, Emotion('interest', 0.1, 0.3, 0.2))

    def _process_behavior_changed(self, data):
        new_behavior = data['new_behavior']
        # Simplified emotional response to behavior changes
        return Emotion('curiosity', 0.2, 0.4, 0.3)

    def apply_mood_influence(self, emotion):
        """
        Modifies the emotion based on the current mood.

        Args:
            emotion (Emotion): The mediated emotional response.

        Returns:
            Emotion: The emotion after mood influence.
        """
        current_mood = self.mood_synthesizer.get_current_mood()
        
        # Adjust emotion valence and arousal slightly towards mood
        adjusted_valence = (emotion.valence + current_mood.valence) / 2
        adjusted_arousal = (emotion.arousal + current_mood.arousal) / 2

        # Create a new Emotion instance with adjusted values
        adjusted_emotion = Emotion(
            name=emotion.name,
            valence=adjusted_valence,
            arousal=adjusted_arousal,
            intensity=emotion.intensity
        )
        return adjusted_emotion
