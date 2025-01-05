# modules/actions/action.py

from abc import ABC, abstractmethod
from event_dispatcher import global_event_dispatcher, Event

class Action(ABC):
    """
    Abstract base class for all actions.
    """

    def __init__(self, action_manager, needs_manager):
        """
        Initializes the Action.

        Args:
            action_manager (ActionManager): Reference to the ActionManager.
            needs_manager (NeedsManager): Reference to the NeedsManager.
        """
        self.action_manager = action_manager
        self.needs_manager = needs_manager

    @abstractmethod
    def perform(self):
        """Performs the action."""
        pass

    def dispatch_event(self, event_type, data=None):
        """
        Dispatches an event related to this action.

        Args:
            event_type (str): The type of event to dispatch.
            data (dict, optional): Additional data to include with the event.
        """
        event_data = {"action_name": self.__class__.__name__}
        if data:
            event_data.update(data)
        global_event_dispatcher.dispatch_event_sync(Event(event_type, event_data))