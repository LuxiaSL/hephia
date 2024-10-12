# modules/actions/action_manager.py

from .feed import FeedAction
from .drink import GiveWaterAction
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
        global_event_dispatcher.dispatch_event_sync(Event("action:started", {"action_name": action_name}))
        
        result = action.perform()
        
        global_event_dispatcher.dispatch_event_sync(Event("action:completed", {
            "action_name": action_name,
            "result": result
        }))

    def register_action(self, action_name, action_instance):
        """
        Registers a new action with the ActionManager.

        Args:
            action_name (str): The name of the action.
            action_instance (Action): The action instance to register.
        """
        self.available_actions[action_name] = action_instance
        global_event_dispatcher.dispatch_event_sync(Event("action:registered", {"action_name": action_name}))

    def unregister_action(self, action_name):
        """
        Unregisters an action from the ActionManager.

        Args:
            action_name (str): The name of the action to unregister.
        """
        if action_name in self.available_actions:
            del self.available_actions[action_name]
            global_event_dispatcher.dispatch_event_sync(Event("action:unregistered", {"action_name": action_name}))