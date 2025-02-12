# modules/actions/drink.py

from typing import Dict, Union, Tuple
from .action import Action

class DrinkAction(Action):
    """Action to give drink with enhanced validation and recovery scaling."""

    def __init__(self, needs_manager, drink_value: float = 20.0):
        super().__init__(needs_manager)
        self.drink_value = float(drink_value)
        self._base_cooldown = 25.0  # 25 second cooldown for drinking

    def validate(self) -> Tuple[bool, Union[str, None]]:
        """Validate if drinking is possible."""
        try:
            current_thirst = self.needs_manager.get_need_value('thirst')
            if current_thirst <= self.needs_manager.needs['thirst'].min_value:
                return False, "Already fully hydrated"
            return True, None
        except Exception as e:
            return False, f"Validation error: {str(e)}"

    def perform(self) -> Dict[str, float]:
        """Performs the drink action with dynamic recovery and minimum value protection."""
        try:
            initial_thirst = self.needs_manager.get_need_value('thirst')
            min_thirst = self.needs_manager.needs['thirst'].min_value
            
            # Calculate recovery amount based on current thirst
            recovery = self.calculate_recovery_amount(initial_thirst, 'thirst')
            
            # Ensure we don't recover beyond the minimum value
            if initial_thirst - recovery < min_thirst:
                recovery = initial_thirst - min_thirst
            
            self.needs_manager.alter_need('thirst', -recovery)
            final_thirst = self.needs_manager.get_need_value('thirst')
            
            result = {
                "initial_thirst": initial_thirst,
                "final_thirst": final_thirst,
                "thirst_reduced": initial_thirst - final_thirst,
                "recovery_amount": recovery
            }
            
            self.dispatch_event("action:drink:completed", result)
            return result
            
        except Exception as e:
            self.dispatch_event("action:drink:error", {"error": str(e)})
            raise RuntimeError(f"Drink action failed: {str(e)}")