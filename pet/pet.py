# pet/pet.py

from pet.pet_state import PetState
from pet.pet_context import PetContext
from modules.needs.needs_manager import NeedsManager
from modules.behaviors.behavior_manager import BehaviorManager
from modules.actions.action_manager import ActionManager
from modules.emotions.emotional_processor import EmotionalProcessor
from modules.cognition.cognitive_processor import CognitiveProcessor
from modules.emotions.mood_synthesizer import MoodSynthesizer
from modules.memory.memory_system import MemorySystem
from event_dispatcher import global_event_dispatcher, Event

class Pet:
    """
    orchestrates the pet's overall functionality
    """

    def __init__(self):
        """
        Initializes the Pet instance and its managers.
        """
        # Initialize internals & externals
        self.context = PetContext(self)
        self.state = PetState()

        # Initialize managers
        self.needs_manager = NeedsManager()
        self.behavior_manager = BehaviorManager(self.state, self.needs_manager)
        self.action_manager = ActionManager(self.needs_manager)

        # Initialize cognitive and emotional modules
        self.memory_system = MemorySystem()
        self.mood_synthesizer = MoodSynthesizer(self.memory_system, self.needs_manager)
        self.cognitive_processor = CognitiveProcessor()
        self.emotional_processor = EmotionalProcessor(
            self.cognitive_processor,
            self.mood_synthesizer,
            self.memory_system
        )

        # Set up event listeners
        self.setup_event_listeners()

        # Initialize other attributes if needed
        self.is_active = True

    def setup_event_listeners(self):
        global_event_dispatcher.add_listener("need:changed", self.on_need_change)
        global_event_dispatcher.add_listener("behavior:changed", self.on_behavior_change)
        global_event_dispatcher.add_listener("action:performed", self.on_action_performed)
        global_event_dispatcher.add_listener("mood:changed", self.on_mood_change)
        global_event_dispatcher.add_listener("emotion:new", self.on_new_emotion)

    def on_need_change(self, event):
        print(f"Pet: Need '{event.data['need_name']}' changed to {event.data['new_value']}")

    def on_behavior_change(self, event):
        print(f"Pet: Behavior changed to {event.data['new_behavior']}")

    def on_action_performed(self, event):
        print(f"Pet: Action '{event.data['action_name']}' performed")

    def on_mood_change(self, event):
        print(f"Pet: Mood changed to {event.data.get('new_state')}")

    def on_new_emotion(self, event):
        print(f"Pet: Emotion experienced: {event.data.get('emotion').name}")

    def update(self):
        """
        coordinating periodic updates across modules
        """
        if not self.is_active:
            return
        
        self.needs_manager.update_needs()
        self.cognitive_processor.periodic_check()

        global_event_dispatcher.dispatch_event_sync(Event("pet:updated", {"pet": self}))


    def perform_action(self, action_name):
        """
        Performs a user-initiated action.

        Args:
            action_name (str): The name of the action to perform.
        """
        self.action_manager.perform_action(action_name)

    def shutdown(self):
        """
        Shuts down the pet's activities gracefully.
        """
        self.is_active = False
        # Perform any necessary cleanup
        self.behavior_manager.current_behavior.stop()
        global_event_dispatcher.dispatch_event_sync(Event("pet:shutdown"))