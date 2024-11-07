# modules/actions/action_manager.py

from .feed import FeedAction
from .drink import DrinkAction
from .play import PlayAction
from .rest import RestAction
from event_dispatcher import global_event_dispatcher, Event

class ActionManager:
    """
    Manages and executes actions.
    """

    def __init__(self, needs_manager):
        """
        Initializes the ActionManager.

        Args:
            needs_manager (NeedsManager): Reference to the NeedsManager.
        """
        self.needs_manager = needs_manager
        self.available_actions = {
            'feed': FeedAction(self, self.needs_manager),
            'give_water': DrinkAction(self, self.needs_manager),
            'play': PlayAction(self, self.needs_manager),
            'rest': RestAction(self, self.needs_manager)
            # Add more actions as needed
        }

    def perform_action(self, action_name):
        """
        Executes the specified action.

        Args:
            action_name (str): The name of the action to perform.

        Raises:
            ValueError: If the action_name is not recognized.
        """
        action = self.available_actions.get(action_name)
        if not action:
            raise ValueError(f"Action '{action_name}' is not available.")
        
        print(f"ActionManager: Executing '{action_name}' action.")
        global_event_dispatcher.dispatch_event_sync(Event("action:started", {"action_name": action_name}))
        
        result = action.perform()
        
        global_event_dispatcher.dispatch_event_sync(Event("action:completed", {
            "action_name": action_name,
            "result": result
        }))