# modules/actions/play.py

from typing import Dict, Union, Tuple
from .action import Action

class PlayAction(Action):
    """Action to play with enhanced validation and stamina management."""

    def __init__(self, needs_manager, play_value: float = 15.0, stamina_cost: float = 10.0):
        super().__init__(needs_manager)
        self.play_value = float(play_value)
        self.stamina_cost = float(stamina_cost)
        self._base_cooldown = 45.0  # 45 second cooldown for playing

    def validate(self) -> Tuple[bool, Union[str, None]]:
        """Validate if playing is possible based on stamina levels."""
        try:
            current_stamina = self.needs_manager.get_need_value('stamina')
            if current_stamina < self.stamina_cost:
                return False, "Insufficient stamina to play"
            
            current_boredom = self.needs_manager.get_need_value('boredom')
            if current_boredom <= self.needs_manager.needs['boredom'].min_value:
                return False, "Not bored enough to play"
                
            return True, None
        except Exception as e:
            return False, f"Validation error: {str(e)}"

    def perform(self) -> Dict[str, float]:
        """Performs the play action with dynamic recovery and stamina cost."""
        try:
            initial_boredom = self.needs_manager.get_need_value('boredom')
            initial_stamina = self.needs_manager.get_need_value('stamina')
            
            # Calculate boredom recovery based on current level
            boredom_recovery = self.calculate_recovery_amount(initial_boredom, 'boredom')
            
            # Calculate stamina cost (higher when more tired)
            stamina_critical = 1 - (initial_stamina / self.needs_manager.needs['stamina'].max_value)
            adjusted_stamina_cost = self.stamina_cost * (1 + (stamina_critical * 0.5))
            
            # Apply boredom reduction, ensuring it doesn't go below the minimum
            boredom_reduction = min(boredom_recovery, initial_boredom - self.needs_manager.needs['boredom'].min_value)
            self.needs_manager.alter_need('boredom', -boredom_reduction)
            
            # Apply stamina cost, ensuring it doesn't go below zero
            stamina_reduction = min(adjusted_stamina_cost, initial_stamina - self.needs_manager.needs['stamina'].min_value)
            self.needs_manager.alter_need('stamina', -stamina_reduction)
            
            final_boredom = self.needs_manager.get_need_value('boredom')
            final_stamina = self.needs_manager.get_need_value('stamina')
            
            result = {
                "initial_boredom": initial_boredom,
                "final_boredom": final_boredom,
                "boredom_reduced": boredom_reduction,
                "initial_stamina": initial_stamina,
                "final_stamina": final_stamina,
                "stamina_cost": adjusted_stamina_cost,
                "stamina_reduced": stamina_reduction,
                "recovery_amount": boredom_recovery # still reporting the potential recovery
            }
            
            self.dispatch_event("action:play:completed", result)
            return result
            
        except Exception as e:
            self.dispatch_event("action:play:error", {"error": str(e)})
            raise RuntimeError(f"Play action failed: {str(e)}")