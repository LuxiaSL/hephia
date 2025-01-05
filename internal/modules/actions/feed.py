# modules/actions/feed.py

from .action import Action

class FeedAction(Action):
    """
    Action to feed.
    """

    def __init__(self, action_manager, needs_manager, food_value=20.0):
        super().__init__(action_manager, needs_manager)
        self.food_value = food_value  # Amount to reduce hunger

    def perform(self):
        """Performs the feeding action by reducing hunger."""
        print("Performing FeedAction.")
        
        self.dispatch_event("action:feed:started")
        
        initial_hunger = self.needs_manager.get_need_value('hunger')
        self.needs_manager.alter_need('hunger', -self.food_value)
        final_hunger = self.needs_manager.get_need_value('hunger')
        
        result = {
            "initial_hunger": initial_hunger,
            "final_hunger": final_hunger,
            "hunger_reduced": initial_hunger - final_hunger
        }
        
        self.dispatch_event("action:feed:completed", result)
        return result