# modules/actions/action.py

from abc import ABC, abstractmethod

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
        self.observers = []

    @abstractmethod
    def perform(self):
        """Performs the action."""
        pass

    def subscribe(self, observer):
        """
        Subscribes an observer to this action.

        Args:
            observer (callable): The observer function or method.
        """
        if observer not in self.observers:
            self.observers.append(observer)

    def unsubscribe(self, observer):
        """
        Unsubscribes an observer from this action.

        Args:
            observer (callable): The observer function or method to remove.
        """
        if observer in self.observers:
            self.observers.remove(observer)

    def notify_observers(self, event_type):
        """
        Notifies all subscribed observers about an event.

        Args:
            event_type (str): The type of event ('start', 'perform', 'end').
        """
        for observer in self.observers:
            observer(self, event_type)
