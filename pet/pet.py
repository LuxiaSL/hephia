# pet/pet.py

from pet.pet_state import PetState
from modules.needs.needs_manager import NeedsManager
from modules.behaviors.behavior_manager import BehaviorManager
from modules.actions.action_manager import ActionManager

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

        # Subscribe behavior manager to need changes
        self.needs_manager.subscribe_to_need('hunger', self.behavior_manager.on_need_change)
        self.needs_manager.subscribe_to_need('thirst', self.behavior_manager.on_need_change)
        self.needs_manager.subscribe_to_need('boredom', self.behavior_manager.on_need_change)
        self.needs_manager.subscribe_to_need('stamina', self.behavior_manager.on_need_change)

        # Initialize other attributes if needed
        self.is_active = True 

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

        # Additional updates can be added here
        # e.g., checking for state transitions, handling events

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

        if hunger > 80:
            self.state.update_emotional_state('hungry')
        elif boredom > 80:
            self.state.update_emotional_state('bored')
        elif stamina < 20:
            self.state.update_emotional_state('tired')
        else:
            self.state.update_emotional_state('happy')

    def shutdown(self):
        """
        Shuts down the pet's activities gracefully.
        """
        self.is_active = False
        # Perform any necessary cleanup
        self.behavior_manager.current_behavior.stop()
