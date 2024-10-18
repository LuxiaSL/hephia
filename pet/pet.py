# pet/pet.py

from pet.pet_state import PetState
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
    The Pet class orchestrates the pet's overall functionality,
    integrating needs, behaviors, and actions.
    """

    def __init__(self):
        """
        Initializes the Pet instance and its managers.
        """
        self.state = PetState()

        # Initialize managers
        self.needs_manager = NeedsManager()
        self.behavior_manager = BehaviorManager(self.state, self.needs_manager)
        self.action_manager = ActionManager(self.needs_manager)

        # Initialize cognitive and emotional modules
        self.memory_system = MemorySystem()
        self.mood_synthesizer = MoodSynthesizer()
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
        """
        Sets up event listeners for the pet.
        """
        global_event_dispatcher.add_listener("need:changed", self.on_need_change)
        global_event_dispatcher.add_listener("behavior:changed", self.on_behavior_change)
        global_event_dispatcher.add_listener("action:performed", self.on_action_performed)
        global_event_dispatcher.add_listener("mood:changed", self.on_mood_change)
        global_event_dispatcher.add_listener("emotion:new", self.on_new_emotion)

    def on_need_change(self, event):
        """
        Handles need change events.
        """
        need_name = event.data['need_name']
        new_value = event.data['new_value']
        print(f"Pet: Need '{need_name}' changed to {new_value}")

        #STRONG NOTE: THE EMOTIONAL STATE FUNCTION HERE IS **NOT** RELATED TO THE EMOTIONAL MODULE. IT IS A VERY POORLY NAMED METHOD FOR UPDATING THE CURRENT VISUAL STATE, WHICH WOULD BE HANDLED BY BOTH THE MOOD/EMOTIONAL/COGNITIVE MODULES
        self.update_emotional_state()

    def on_behavior_change(self, event):
        """
        Handles behavior change events.
        """
        new_behavior = event.data['new_behavior']
        print(f"Pet: Behavior changed to {new_behavior}")

    def on_action_performed(self, event):
        """
        Handles action performed events.
        """
        action_name = event.data['action_name']
        print(f"Pet: Action '{action_name}' performed")

    def on_mood_change(self, event):
        """
        Handles mood change events.

        Args:
            event (Event): The mood change event containing mood data.
        """
        # Extract mood information from the event
        new_mood = event.data.get('new_mood')
        
        # Log the mood change
        print(f"Pet: Mood changed to {new_mood}")
        
        # Update the pet's emotional state based on the new mood
        self.update_emotional_state()

    def on_new_emotion(self, event):
        """
        Handles new emotion events.

        Args:
            event (Event): The new emotion event containing emotion data.
        """
        # Extract emotion information from the event
        new_emotion = event.data.get('emotion')
        
        # Log the new emotion
        print(f"Pet: New emotion experienced: {new_emotion.name}")
        
        # Update the pet's emotional state based on the new emotion
        self.update_emotional_state()

    def update(self):
        """
        Updates the pet's state by coordinating updates across managers.
        """
        if not self.is_active:
            return

         # Update needs
        self.needs_manager.update_needs()

        # Update behaviors
        self.behavior_manager.update()

        # Update mood
        self.mood_synthesizer.update_mood()

        # Update emotional state based on needs and behaviors
        # NOTE:: PLACE THE BELOW FUNCTIONALITY INTO THE MOOD SYNTHESIZER AND EXTEND BASED ON SUM OF RECENT EMOTIONS BEING POSITIVE OR NEGATIVE AND INTENSITY OR W.E
        # NOTE:: THE SUM FUNCTIONALITY SIMULATES THE MEMORY FUNCTIONALITY, WHICH WILL LIKELY FOLLOW EMOTIONAL MODULE IN IMPLEMENTATION
        # NOTE:: ALSO CAN OVERRIDE WHEN SPECIFIC NEEDS ARE V LOW TO SPECIFIC STATES
        self.update_emotional_state()

        # Dispatch a pet updated event
        # See EVENT_CATALOG.md for full event details
        global_event_dispatcher.dispatch_event_sync(Event("pet:updated", {"pet": self}))

    def perform_action(self, action_name):
        """
        Performs a user-initiated action.

        Args:
            action_name (str): The name of the action to perform.
        """
        self.action_manager.perform_action(action_name)

    def update_emotional_state(self):
        """
        Updates the pet's emotional state based on current needs and behaviors.
        """
        # Simple example logic for emotional state determination
        hunger = self.needs_manager.get_need_value('hunger')
        boredom = self.needs_manager.get_need_value('boredom')
        stamina = self.needs_manager.get_need_value('stamina')

        old_state = self.state.emotional_state

        if hunger > 80:
            new_state = 'hungry'
        elif boredom > 80:
            new_state = 'bored'
        elif stamina < 20:
            new_state = 'tired'
        else:
            new_state = 'happy'

        if new_state != old_state:
            self.state.update_emotional_state(new_state)
            global_event_dispatcher.dispatch_event_sync(Event("pet:emotional_state_changed", {
                "old_state": old_state,
                "new_state": new_state
            }))

    def shutdown(self):
        """
        Shuts down the pet's activities gracefully.
        """
        self.is_active = False
        # Perform any necessary cleanup
        self.behavior_manager.current_behavior.stop()
        global_event_dispatcher.dispatch_event_sync(Event("pet:shutdown"))