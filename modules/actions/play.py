# modules/actions/play.py

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
        
        # Dispatch action:play:started event
        # See EVENT_CATALOG.md for full event details
        self.dispatch_event("action:play:started")
        
        initial_boredom = self.needs_manager.get_need_value('boredom')
        initial_stamina = self.needs_manager.get_need_value('stamina')
        
        self.needs_manager.alter_need('boredom', -self.play_value)
        self.needs_manager.alter_need('stamina', -self.stamina_cost)
        
        final_boredom = self.needs_manager.get_need_value('boredom')
        final_stamina = self.needs_manager.get_need_value('stamina')
        
        result = {
            "initial_boredom": initial_boredom,
            "final_boredom": final_boredom,
            "boredom_reduced": initial_boredom - final_boredom,
            "initial_stamina": initial_stamina,
            "final_stamina": final_stamina,
            "stamina_reduced": initial_stamina - final_stamina
        }
        
        # Dispatch action:play:completed event
        # See EVENT_CATALOG.md for full event details
        self.dispatch_event("action:play:completed", result)
        return result
