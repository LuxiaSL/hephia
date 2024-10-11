from .behavior import Behavior

class WalkBehavior(Behavior):
    """
    Represents the pet's walking state.
    """

    def __init__(self, behavior_manager, pet_state):
        super().__init__(behavior_manager, pet_state)
        # Initialize walk-specific properties
        self.base_rate_modifiers = {}
        self.multiplier_modifiers = {}

    def start(self):
        """Starts the walk behavior."""
        self.active = True
        print("WalkBehavior started.")
        # Apply need decay modifiers
        self.apply_need_modifiers()
        # Notify observers
        self.notify_observers('start')

    def update(self):
        """Updates the walk behavior."""
        if not self.active:
            return
        # Update walk-specific logic (e.g., moving position)
        # ...
        # Notify observers
        self.notify_observers('update')

    def stop(self):
        """Stops the walk behavior."""
        self.active = False
        print("WalkBehavior stopped.")
        # Remove need decay modifiers
        self.remove_need_modifiers()
        # Notify observers
        self.notify_observers('stop')

    def apply_need_modifiers(self):
        """
        Applies decay rate modifiers specific to walk behavior.
        """
        needs_manager = self.behavior_manager.needs_manager
        # Increase physical need decay rates
        self.base_rate_modifiers['hunger'] = 0.05
        self.base_rate_modifiers['thirst'] = 0.05
        self.multiplier_modifiers['hunger'] = 1.1  # 10% increase
        self.multiplier_modifiers['thirst'] = 1.1  # 10% increase
        
        for need, modifier in self.base_rate_modifiers.items():
            needs_manager.alter_base_decay_rate(need, modifier)
        
        for need, modifier in self.multiplier_modifiers.items():
            needs_manager.alter_decay_rate_multiplier(need, modifier)

    def remove_need_modifiers(self):
        """
        Removes decay rate modifiers applied by walk behavior.
        """
        needs_manager = self.behavior_manager.needs_manager
        
        for need, modifier in self.base_rate_modifiers.items():
            needs_manager.alter_base_decay_rate(need, -modifier)  # Reverse the modification
        
        for need, modifier in self.multiplier_modifiers.items():
            needs_manager.alter_decay_rate_multiplier(need, 1/modifier)  # Reverse the modification
        
        self.base_rate_modifiers.clear()
        self.multiplier_modifiers.clear()
