# modules/actions/rest.py

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
        self.dispatch_event("action:rest:started")
        
        initial_stamina = self.needs_manager.get_need_value('stamina')
        self.needs_manager.alter_need('stamina', self.stamina_gain)
        final_stamina = self.needs_manager.get_need_value('stamina')
        
        result = {
            "initial_stamina": initial_stamina,
            "final_stamina": final_stamina,
            "stamina_gained": final_stamina - initial_stamina
        }
        
        self.dispatch_event("action:rest:completed", result)
        return result