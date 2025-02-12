# modules/actions/action_manager.py

from .feed import FeedAction
from .drink import DrinkAction
from .play import PlayAction
from .rest import RestAction
from internal.modules.needs.needs_manager import NeedsManager
from event_dispatcher import global_event_dispatcher, Event
import time

class ActionManager:
    """
    Manages and executes actions with stability controls.
    """

    def __init__(self, needs_manager: NeedsManager):
        """
        Initializes the ActionManager.

        Args:
            needs_manager (NeedsManager): Reference to the NeedsManager.
        """
        self.needs_manager = needs_manager
        self.available_actions = {
            'feed': FeedAction(self.needs_manager),
            'give_water': DrinkAction(self.needs_manager),
            'play': PlayAction(self.needs_manager),
            'rest': RestAction(self.needs_manager)
        }
        self.action_history = {}
        self.initialize_history()

    def initialize_history(self):
        """Initialize tracking for all available actions."""
        for action_name in self.available_actions:
            self.action_history[action_name] = {
                'last_execution': 0,
                'total_executions': 0,
                'successful_executions': 0,
                'failed_executions': 0
            }

    def perform_action(self, action_name):
        """
        Executes the specified action with validation and tracking.

        Args:
            action_name (str): The name of the action to perform.

        Returns:
            dict: Result of the action execution including success status and details.

        Raises:
            ValueError: If the action_name is not recognized.
        """
        action = self.available_actions.get(action_name)
        if not action:
            raise ValueError(f"Action '{action_name}' is not available.")

        # Check cooldown
        if not self._check_cooldown(action_name):
            return {
                'success': False,
                'error': 'Action is on cooldown',
                'remaining_cooldown': self._get_remaining_cooldown(action_name)
            }

        # Validate action prerequisites
        can_execute, reason = action.validate()
        if not can_execute:
            self._update_history(action_name, False)
            return {
                'success': False,
                'error': reason
            }

        try:
            print(f"ActionManager: Executing '{action_name}' action.")
            global_event_dispatcher.dispatch_event_sync(Event("action:started", {"action_name": action_name}))
            
            result = action.perform()
            
            # Update history and dispatch completion event
            self._update_history(action_name, True)
            global_event_dispatcher.dispatch_event_sync(Event("action:completed", {
                "action_name": action_name,
                "result": result,
                "success": True
            }))
            
            return {
                'success': True,
                'result': result
            }
            
        except Exception as e:
            self._update_history(action_name, False)
            error_msg = str(e)
            global_event_dispatcher.dispatch_event_sync(Event("action:error", {
                "action_name": action_name,
                "error": error_msg
            }))
            return {
                'success': False,
                'error': error_msg
            }

    def _check_cooldown(self, action_name):
        """Check if enough time has passed since last execution."""
        last_execution = self.action_history[action_name]['last_execution']
        cooldown = self.available_actions[action_name].get_cooldown()
        return time.time() >= last_execution + cooldown

    def _get_remaining_cooldown(self, action_name):
        """Get remaining cooldown time in seconds."""
        last_execution = self.action_history[action_name]['last_execution']
        cooldown = self.available_actions[action_name].get_cooldown()
        remaining = (last_execution + cooldown) - time.time()
        return max(0, remaining)

    def _update_history(self, action_name, success):
        """Update action history with execution results."""
        self.action_history[action_name]['last_execution'] = time.time()
        self.action_history[action_name]['total_executions'] += 1
        if success:
            self.action_history[action_name]['successful_executions'] += 1
        else:
            self.action_history[action_name]['failed_executions'] += 1

    def get_action_status(self, action_name=None):
        """
        Get status information about actions.

        Args:
            action_name (str, optional): Specific action to get status for.

        Returns:
            dict: Status information including cooldowns and history.
        """
        if action_name:
            if action_name not in self.available_actions:
                raise ValueError(f"Action '{action_name}' not found.")
            return {
                'history': self.action_history[action_name],
                'on_cooldown': not self._check_cooldown(action_name),
                'remaining_cooldown': self._get_remaining_cooldown(action_name)
            }
        
        return {name: {
            'history': self.action_history[name],
            'on_cooldown': not self._check_cooldown(name),
            'remaining_cooldown': self._get_remaining_cooldown(name)
        } for name in self.available_actions}