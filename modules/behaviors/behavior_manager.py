# modules/behaviors/behavior_manager.py

from .idle import IdleBehavior
from .walk import WalkBehavior

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
        self.current_behavior = new_behavior
        self.current_behavior.start()

    # Methods to react to need changes...
    def on_need_change(self, need):
        """
        Reacts to changes in needs.

        Args:
            need (Need): The need that has changed.
        """
        # Example logic: if hunger is high, switch to seek food behavior
        if need.name == 'hunger' and need.value > 80:
            # Implement logic to change behavior accordingly
            pass

    def subscribe_to_behavior(self, event_type, observer):
        """
        Subscribes an observer to a specific behavior event.

        Args:
            event_type (str): The type of event ('start', 'update', 'stop').
            observer (callable): The observer function or method.
        """
        if self.current_behavior:
            self.current_behavior.subscribe(observer)

    def unsubscribe_from_behavior(self, event_type, observer):
        """
        Unsubscribes an observer from a specific behavior event.

        Args:
            event_type (str): The type of event ('start', 'update', 'stop').
            observer (callable): The observer function or method to remove.
        """
        if self.current_behavior:
            self.current_behavior.unsubscribe(observer)
