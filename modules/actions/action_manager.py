# modules/actions/action_manager.py

from .feed import FeedAction
from .drink import GiveWaterAction
from .play import PlayAction
from .rest import RestAction

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
            'give_water': GiveWaterAction(self, self.needs_manager),
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
        action.perform()

    def subscribe_to_action(self, action_name, observer):
        """
        Subscribes an observer to a specific action.

        Args:
            action_name (str): The name of the action.
            observer (callable): The observer function or method.
        """
        action = self.available_actions.get(action_name)
        if action:
            action.subscribe(observer)
        else:
            raise ValueError(f"Action '{action_name}' is not available.")

    def unsubscribe_from_action(self, action_name, observer):
        """
        Unsubscribes an observer from a specific action.

        Args:
            action_name (str): The name of the action.
            observer (callable): The observer function or method to remove.
        """
        action = self.available_actions.get(action_name)
        if action:
            action.unsubscribe(observer)
        else:
            raise ValueError(f"Action '{action_name}' is not available.")
