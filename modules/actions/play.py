from .action import Action

class PlayAction(Action):
    """
    Action to play with the pet.
    """

    def __init__(self, action_manager, needs_manager, play_value=15.0, stamina_cost=10.0):
        super().__init__(action_manager, needs_manager)
        self.play_value = play_value      # Amount to reduce boredom
        self.stamina_cost = stamina_cost  # Amount to reduce stamina

    def perform(self):
        """Performs the play action by reducing boredom and stamina."""
        print("Performing PlayAction.")
        self.needs_manager.alter_need('boredom', -self.play_value)
        self.needs_manager.alter_need('stamina', -self.stamina_cost)
        self.notify_observers('perform')