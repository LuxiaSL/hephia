# pet/pet.py

from pet.pet_state import PetState
from modules.needs.needs_manager import NeedsManager
from modules.behaviors.behavior_manager import BehaviorManager
from modules.actions.action_manager import ActionManager
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
        # Initialize the pet's state
        self.state = PetState()

        # Initialize managers
        self.needs_manager = NeedsManager()
        self.behavior_manager = BehaviorManager(self.state, self.needs_manager)
        self.action_manager = ActionManager(self.needs_manager)

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

    def on_need_change(self, event):
        """
        Handles need change events.
        """
        need_name = event.data['need_name']
        new_value = event.data['new_value']
        print(f"Pet: Need '{need_name}' changed to {new_value}")
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

        # Update emotional state based on needs and behaviors
        self.update_emotional_state()

        # Dispatch a pet updated event
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