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
        self.needs_manager.alter_need('thirst', -self.water_value)
        self.notify_observers('perform')