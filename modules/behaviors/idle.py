# modules/behaviors/idle.py

from .behavior import Behavior
from event_dispatcher import global_event_dispatcher, Event

class IdleBehavior(Behavior):
    """
    Represents the pet's idle state.
    """

    def __init__(self, behavior_manager):
        super().__init__(behavior_manager)
        # Initialize idle-specific properties
        self.base_rate_modifiers = {}
        self.multiplier_modifiers = {}

    def start(self):
        """Starts the idle behavior."""
        super().start()
        print("IdleBehavior started.")
        # Apply need decay modifiers
        self.apply_need_modifiers()

    def update(self):
        """Updates the idle behavior."""
        super().update()
        if not self.active:
            return
        # Update idle-specific animations or logic

    def stop(self):
        """Stops the idle behavior."""
        print("IdleBehavior stopped.")
        # Remove need decay modifiers
        self.remove_need_modifiers()
        super().stop()

    def apply_need_modifiers(self):
        """
        Applies decay rate modifiers specific to idle behavior.
        """
        needs_manager = self.behavior_manager.needs_manager
        # Decrease physical need decay rates
        self.base_rate_modifiers['hunger'] = -0.05
        self.base_rate_modifiers['thirst'] = -0.05
        self.multiplier_modifiers['hunger'] = 0.9  # 10% decrease
        self.multiplier_modifiers['thirst'] = 0.9  # 10% decrease
        
        for need, modifier in self.base_rate_modifiers.items():
            needs_manager.alter_base_decay_rate(need, modifier)
        
        for need, modifier in self.multiplier_modifiers.items():
            needs_manager.alter_decay_rate_multiplier(need, modifier)

        # Dispatch behavior:idle:modifiers_applied event
        # See EVENT_CATALOG.md for full event details
        global_event_dispatcher.dispatch_event_sync(Event("behavior:idle:modifiers_applied", {
            "base_modifiers": self.base_rate_modifiers,
            "multiplier_modifiers": self.multiplier_modifiers
        }))

    def remove_need_modifiers(self):
        """
        Removes decay rate modifiers applied by idle behavior.
        """
        needs_manager = self.behavior_manager.needs_manager
        
        for need, modifier in self.base_rate_modifiers.items():
            needs_manager.alter_base_decay_rate(need, -modifier)  # Reverse the modification
        
        for need, modifier in self.multiplier_modifiers.items():
            needs_manager.alter_decay_rate_multiplier(need, 1/modifier)  # Reverse the modification
        
        # Dispatch behavior:idle:modifiers_removed event
        # See EVENT_CATALOG.md for full event details
        global_event_dispatcher.dispatch_event_sync(Event("behavior:idle:modifiers_removed", {
            "base_modifiers": self.base_rate_modifiers,
            "multiplier_modifiers": self.multiplier_modifiers
        }))

        self.base_rate_modifiers.clear()
        self.multiplier_modifiers.clear()
