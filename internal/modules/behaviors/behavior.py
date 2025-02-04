# modules/behaviors/behavior.py

from abc import ABC, abstractmethod
from typing import Optional
from event_dispatcher import global_event_dispatcher, Event

class Behavior(ABC):
    """
    Abstract base class for all behaviors.
    """

    def __init__(self, behavior_manager):
        """
        Initializes the Behavior.

        Args:
            behavior_manager (BehaviorManager): Reference to the BehaviorManager.
        """
        self.behavior_manager = behavior_manager
        self.name: Optional[str] = None
        self.active = False

    @abstractmethod
    def start(self):
        """Starts the behavior."""
        self.active = True
        global_event_dispatcher.dispatch_event_sync(Event(f"behavior:{self.name}:started"))

    @abstractmethod
    def update(self):
        """Updates the behavior."""
        if self.active:
            global_event_dispatcher.dispatch_event_sync(Event(f"behavior:{self.name}:updated"))

    @abstractmethod
    def stop(self):
        """Stops the behavior."""
        self.active = False
        global_event_dispatcher.dispatch_event_sync(Event(f"behavior:{self.name}:stopped"))