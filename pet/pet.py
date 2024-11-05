# pet/pet.py

from pet.pet_context import PetContext
from modules.needs.needs_manager import NeedsManager
from modules.behaviors.behavior_manager import BehaviorManager
from modules.actions.action_manager import ActionManager
from modules.emotions.emotional_processor import EmotionalProcessor
from modules.cognition.cognitive_processor import CognitiveProcessor
from modules.emotions.mood_synthesizer import MoodSynthesizer
from modules.memory.memory_system import MemorySystem
from event_dispatcher import global_event_dispatcher, Event
from global_timer import GlobalTimer
from config import Config

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

        # Initialize managers
        self.needs_manager = NeedsManager()
        self.behavior_manager = BehaviorManager(self.context, self.needs_manager)
        self.action_manager = ActionManager(self.needs_manager)

        # Initialize cognitive and emotional modules
        self.memory_system = MemorySystem()
        self.mood_synthesizer = MoodSynthesizer(self.context)
        self.cognitive_processor = CognitiveProcessor()
        self.emotional_processor = EmotionalProcessor(
            self.context,
            self.cognitive_processor,
            self.memory_system
        )

        # Set up event listeners
        self.setup_event_listeners()

        # establish timers
        self.timer = GlobalTimer()

        self.timer.add_task(Config.NEED_UPDATE_TIMER, self.tick_needs)
        self.timer.add_task(Config.EMOTION_UPDATE_TIMER, self.tick_emotions)

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
        print(f"Pet: Mood changed to {event.data.get('new_name')}")

    def on_new_emotion(self, event):
        print(f"Pet: Emotion experienced: {event.data.get('emotion').name}")

    async def tick_needs(self):
        self.needs_manager.update_needs()

    async def tick_emotions(self):
        self.emotional_processor.update()

    async def start(self):
        await self.timer.run()
        
    def stop(self):
        self.timer.stop()
        self.is_active = False

    def perform_action(self, action_name):
        """
        performs desired action (used by either user or pet)

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