# modules/behaviors/behavior_manager.py

from .idle import IdleBehavior
from .walk import WalkBehavior
from event_dispatcher import global_event_dispatcher, Event

class BehaviorManager:
    """
    Manages the pet's behaviors.
    """
    def __init__(self, pet_context, needs_manager):
        """
        Initializes the BehaviorManager.

        Args:
            pet_context (PetContext): methods to retrieve pet's current internal state
            needs_manager (NeedsManager): The NeedsManager instance (used by behaviors to manage decay rates)
        """
        self.pet_context = pet_context
        self.needs_manager = needs_manager
        self.current_behavior = None

        self.behaviors = {
            'idle': IdleBehavior(self),
            'walk': WalkBehavior(self),
            # Add more behaviors here
        }
        # Other behaviors...

        # Start with idle behavior
        self.change_behavior('idle')

        # Set up event listeners
        self.setup_event_listeners()

    def setup_event_listeners(self):
        global_event_dispatcher.add_listener("need:changed", self.determine_behavior)
        global_event_dispatcher.add_listener("action:completed", self.determine_behavior)
        global_event_dispatcher.add_listener("mood:changed", self.determine_behavior)
        global_event_dispatcher.add_listener("emotion:new", self.determine_behavior)

    def update(self):
        """
        Updates the current behavior.
        """
        if self.current_behavior:
            self.current_behavior.update()

    def change_behavior(self, new_behavior_name):
        """
        Changes the current behavior.

        Args:
            new_behavior_name (str): selected behavior to activate
        """
        if self.current_behavior:
            old_behavior = self.current_behavior
            self.current_behavior.stop()
        self.current_behavior = self.behaviors[new_behavior_name]
        self.current_behavior.start()

        global_event_dispatcher.dispatch_event_sync(Event("behavior:changed", {
            "old_behavior": old_behavior.__class__.__name__ if old_behavior else None,
            "new_behavior": new_behavior_name
        }))

    def determine_behavior(self, event):
        """
        alters behavior based on event received, if necessary.

        args:
            event (Event): *any* given event from above
        """
        event_type = event.event_type
        event_data = event.data

        # get holistic accounting
        current_needs = self.pet_context.get_current_needs()
        current_mood = self.pet_context.get_current_mood()
        recent_emotions = self.pet_context.get_recent_emotions()
        
        # what to do...
        new_behavior = self._calculate_behavior(event_type, event_data, current_needs, current_mood, recent_emotions)

        if new_behavior != self.current_behavior.__class__.__name__:
            self.change_behavior(new_behavior)

    def _calculate_behavior(self, event_type, event_data, current_needs, current_mood, recent_emotions):
        # Simple logic for demonstration purposes
        # In a real implementation, this would be more complex
        if current_needs['hunger'] > 80 or current_needs['thirst'] > 80:
            return 'walk'  # Assume walking leads to finding food/water
        elif current_needs['boredom'] > 70:
            return 'walk'
        else:
            return 'idle'

    def get_current_behavior(self):
        """
        Returns the current behavior.

        Returns:
            Behavior: The current behavior instance.
        """
        return self.current_behavior