# modules/actions/drink.py

from .action import Action

class DrinkAction(Action):
    """
    Action to give drink.
    """

    def __init__(self, action_manager, needs_manager, drink_value=20.0):
        super().__init__(action_manager, needs_manager)
        self.drink_value = drink_value  # Amount to reduce thirst

    def perform(self):
        """Performs the drink action by reducing thirst."""
        print("Performing DrinkAction.")
        
        self.dispatch_event("action:drink:started")
        
        initial_thirst = self.needs_manager.get_need_value('thirst')
        self.needs_manager.alter_need('thirst', -self.drink_value)
        final_thirst = self.needs_manager.get_need_value('thirst')
        
        result = {
            "initial_thirst": initial_thirst,
            "final_thirst": final_thirst,
            "thirst_reduced": initial_thirst - final_thirst
        }
        
        self.dispatch_event("action:drink:completed", result)
        return result