# modules/actions/drink.py

from .action import Action

class GiveWaterAction(Action):
    """
    Action to give water to the pet.
    """

    def __init__(self, action_manager, needs_manager, water_value=20.0):
        super().__init__(action_manager, needs_manager)
        self.water_value = water_value  # Amount to reduce thirst

    def perform(self):
        """Performs the giving water action by reducing thirst."""
        print("Performing GiveWaterAction.")
        self.dispatch_event("action:give_water:started")
        
        initial_thirst = self.needs_manager.get_need_value('thirst')
        self.needs_manager.alter_need('thirst', -self.water_value)
        final_thirst = self.needs_manager.get_need_value('thirst')
        
        result = {
            "initial_thirst": initial_thirst,
            "final_thirst": final_thirst,
            "thirst_reduced": initial_thirst - final_thirst
        }
        
        self.dispatch_event("action:give_water:completed", result)
        return result