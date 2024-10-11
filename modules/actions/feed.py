from .action import Action

class FeedAction(Action):
    """
    Action to feed the pet.
    """

    def __init__(self, action_manager, needs_manager, food_value=20.0):
        super().__init__(action_manager, needs_manager)
        self.food_value = food_value  # Amount to reduce hunger

    def perform(self):
        """Performs the feeding action by reducing hunger."""
        print("Performing FeedAction.")
        self.needs_manager.alter_need('hunger', -self.food_value)
        self.notify_observers('perform')