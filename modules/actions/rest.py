from .action import Action

class RestAction(Action):
    """
    Action to let the pet rest.
    """

    def __init__(self, action_manager, needs_manager, stamina_gain=20.0):
        super().__init__(action_manager, needs_manager)
        self.stamina_gain = stamina_gain  # Amount to increase stamina

    def perform(self):
        """Performs the rest action by increasing stamina."""
        print("Performing RestAction.")
        self.needs_manager.alter_need('stamina', self.stamina_gain)
        self.notify_observers('perform')