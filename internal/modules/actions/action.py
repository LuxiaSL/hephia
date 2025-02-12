# modules/actions/action.py

from abc import ABC, abstractmethod
import time
from internal.modules.needs.needs_manager import NeedsManager
from event_dispatcher import global_event_dispatcher, Event

class Action(ABC):
    """
    Abstract base class for all actions with enhanced stability controls.
    """

    def __init__(self, needs_manager: NeedsManager):
        """
        Initializes the Action.

        Args:
            needs_manager (NeedsManager): Reference to the NeedsManager.
        """
        self.needs_manager = needs_manager
        self._base_cooldown = 10  # Default 10 second cooldown

    @abstractmethod
    def perform(self):
        """
        Performs the action. Must be implemented by concrete actions.
        
        Returns:
            dict: Result of the action including any relevant data.
        """
        pass

    def validate(self):
        """
        Validates if the action can be performed.
        
        Returns:
            tuple: (bool, str) - (can_execute, reason if cannot execute)
        """
        return True, None

    def get_cooldown(self):
        """
        Get the cooldown duration for this action.
        
        Returns:
            float: Cooldown time in seconds
        """
        return self._base_cooldown

    def calculate_recovery_amount(self, current_value, target_need):
        """
        Calculates recovery amount based on current need value.
        Implements progressive recovery - more recovery when needs are critical.
        
        Args:
            current_value (float): Current value of the need
            target_need (str): Name of the need being recovered
        
        Returns:
            float: Amount to recover by
        """
        # Get need maximum from needs manager
        need_max = self.needs_manager.needs[target_need].max_value
        need_min = self.needs_manager.needs[target_need].min_value
        
        # Calculate how critical the need is (0 to 1, where 1 is most critical)
        critical_level = (need_max - current_value) / (need_max - need_min)
        
        # Base recovery amount (can be tuned)
        base_recovery = 20.0
        
        # Enhanced recovery for critical needs
        if critical_level > 0.8:  # More than 80% depleted
            return base_recovery * 2.0
        elif critical_level > 0.6:  # More than 60% depleted
            return base_recovery * 1.5
        
        return base_recovery

    def dispatch_event(self, event_type, data=None):
        """
        Dispatches an event related to this action.

        Args:
            event_type (str): The type of event to dispatch.
            data (dict, optional): Additional data to include with the event.
        """
        event_data = {
            "action_name": self.__class__.__name__,
            "timestamp": time.time()
        }
        if data:
            event_data.update(data)
        global_event_dispatcher.dispatch_event(Event(event_type, event_data))