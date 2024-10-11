# modules/behaviors/behavior.py

from abc import ABC, abstractmethod

class Behavior(ABC):
    """
    Abstract base class for all behaviors.
    """

    def __init__(self, behavior_manager, pet_state):
        """
        Initializes the Behavior.

        Args:
            behavior_manager (BehaviorManager): Reference to the BehaviorManager.
            pet_state (PetState): The pet's current state.
        """
        self.behavior_manager = behavior_manager
        self.pet_state = pet_state
        self.active = False

        # Observers subscribing to this behavior
        self.observers = []

    @abstractmethod
    def start(self):
        """Starts the behavior."""
        pass

    @abstractmethod
    def update(self):
        """Updates the behavior."""
        pass

    @abstractmethod
    def stop(self):
        """Stops the behavior."""
        pass

    def subscribe(self, observer):
        """
        Subscribes an observer to this behavior.

        Args:
            observer (callable): The observer function or method.
        """
        if observer not in self.observers:
            self.observers.append(observer)

    def unsubscribe(self, observer):
        """
        Unsubscribes an observer from this behavior.

        Args:
            observer (callable): The observer function or method to remove.
        """
        if observer in self.observers:
            self.observers.remove(observer)

    def notify_observers(self, event_type):
        """
        Notifies all subscribed observers about an event.

        Args:
            event_type (str): The type of event ('start', 'update', 'stop').
        """
        for observer in self.observers:
            observer(self, event_type)
