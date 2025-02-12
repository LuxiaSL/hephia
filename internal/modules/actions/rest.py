# modules/actions/rest.py

from typing import Dict, Union, Tuple
from .action import Action

class RestAction(Action):
    """Action to rest with enhanced validation and recovery scaling."""

    def __init__(self, needs_manager, stamina_gain: float = 20.0):
        super().__init__(needs_manager)
        self.stamina_gain = float(stamina_gain)
        self._base_cooldown = 60.0  # 60 second cooldown for resting

    def validate(self) -> Tuple[bool, Union[str, None]]:
        """Validate if resting is possible."""
        try:
            current_stamina = self.needs_manager.get_need_value('stamina')
            if current_stamina >= self.needs_manager.needs['stamina'].max_value:
                return False, "Already fully rested"
            return True, None
        except Exception as e:
            return False, f"Validation error: {str(e)}"

    def perform(self) -> Dict[str, float]:
        """Performs the rest action with dynamic recovery."""
        try:
            initial_stamina = self.needs_manager.get_need_value('stamina')
            
            # Calculate recovery amount based on current stamina
            recovery = self.calculate_recovery_amount(initial_stamina, 'stamina')
            
            # Ensure stamina does not exceed maximum
            max_stamina = self.needs_manager.needs['stamina'].max_value
            recovery = min(recovery, max_stamina - initial_stamina)
            
            self.needs_manager.alter_need('stamina', recovery)
            final_stamina = self.needs_manager.get_need_value('stamina')
            
            result = {
                "initial_stamina": initial_stamina,
                "final_stamina": final_stamina,
                "stamina_gained": final_stamina - initial_stamina,
                "recovery_amount": recovery
            }
            
            self.dispatch_event("action:rest:completed", result)
            return result
            
        except Exception as e:
            self.dispatch_event("action:rest:error", {"error": str(e)})
            raise RuntimeError(f"Rest action failed: {str(e)}")