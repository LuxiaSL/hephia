# modules/behaviors/behavior_manager.py

from .idle import IdleBehavior
from .walk import WalkBehavior
from event_dispatcher import global_event_dispatcher, Event

class BehaviorManager:
    """
    Manages the pet's behaviors.
    """

    def __init__(self, pet_state, needs_manager):
        """
        Initializes the BehaviorManager.

        Args:
            pet_state (PetState): The pet's current state.
            needs_manager (NeedsManager): The NeedsManager instance.
        """
        self.pet_state = pet_state
        self.needs_manager = needs_manager
        self.current_behavior = None

        # Initialize behaviors
        self.idle_behavior = IdleBehavior(self, pet_state)
        self.walk_behavior = WalkBehavior(self, pet_state)
        # Other behaviors...

        # Start with idle behavior
        self.change_behavior(self.idle_behavior)

        # Set up event listeners
        self.setup_event_listeners()

    def setup_event_listeners(self):
        """
        Sets up event listeners for the BehaviorManager.
        """
        global_event_dispatcher.add_listener("need:changed", self.on_need_change)

    def update(self):
        """
        Updates the current behavior.
        """
        if self.current_behavior:
            self.current_behavior.update()

    def change_behavior(self, new_behavior):
        """
        Changes the current behavior.

        Args:
            new_behavior (Behavior): The new behavior to activate.
        """
        if self.current_behavior:
            self.current_behavior.stop()
        old_behavior = self.current_behavior
        self.current_behavior = new_behavior
        self.current_behavior.start()

        # Dispatch behavior:changed event
        # See EVENT_CATALOG.md for full event details
        global_event_dispatcher.dispatch_event_sync(Event("behavior:changed", {
            "old_behavior": old_behavior.__class__.__name__ if old_behavior else None,
            "new_behavior": new_behavior.__class__.__name__
        }))

    def on_need_change(self, event):
        """
        Reacts to changes in needs.

        Args:
            event (Event): The need change event.
        """
        need_name = event.data['need_name']
        new_value = event.data['new_value']

        # Example logic: if hunger is high, switch to seek food behavior
        if need_name == 'hunger' and new_value > 80:
            # Implement logic to change behavior accordingly
            pass
        # Add more behavior change logic based on needs

    def get_current_behavior(self):
        """
        Returns the current behavior.

        Returns:
            Behavior: The current behavior instance.
        """
        return self.current_behavior