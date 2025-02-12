# modules/actions/feed.py

from typing import Dict, Union, Tuple
from .action import Action

class FeedAction(Action):
    """Action to feed with enhanced validation and recovery scaling."""

    def __init__(self, needs_manager, food_value: float = 20.0):
        super().__init__(needs_manager)
        self.food_value = float(food_value)
        self._base_cooldown = 30.0  # 30 second cooldown for feeding

    def validate(self) -> Tuple[bool, Union[str, None]]:
        """Validate if feeding is possible."""
        try:
            current_hunger = self.needs_manager.get_need_value('hunger')
            if current_hunger <= self.needs_manager.needs['hunger'].min_value:
                return False, "Already fully fed"
            return True, None
        except Exception as e:
            return False, f"Validation error: {str(e)}"

    def perform(self) -> Dict[str, float]:
        """Performs the feeding action with dynamic recovery."""
        try:
            initial_hunger = self.needs_manager.get_need_value('hunger')
            
            # Calculate recovery amount based on current hunger
            recovery = self.calculate_recovery_amount(initial_hunger, 'hunger')
            
            # Ensure hunger doesn't drop below the minimum value
            recovery = min(recovery, initial_hunger - self.needs_manager.needs['hunger'].min_value)
            
            self.needs_manager.alter_need('hunger', -recovery)
            final_hunger = self.needs_manager.get_need_value('hunger')
            
            result = {
                "initial_hunger": initial_hunger,
                "final_hunger": final_hunger,
                "hunger_reduced": initial_hunger - final_hunger,
                "recovery_amount": recovery
            }
            
            self.dispatch_event("action:feed:completed", result)
            return result
            
        except Exception as e:
            self.dispatch_event("action:feed:error", {"error": str(e)})
            raise RuntimeError(f"Feed action failed: {str(e)}")