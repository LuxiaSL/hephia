# event_dispatcher.py

import asyncio
from collections import defaultdict
import re
from typing import Any, Callable, Dict, List, Optional

class Event:
    """
    Represents an event with type, data, and metadata.
    """

    def __init__(self, event_type: str, data: Any = None, metadata: Optional[Dict[str, Any]] = None):
        """
        Initializes an Event instance.

        Args:
            event_type (str): The type of the event, using colon-separated namespacing.
            data (Any, optional): The data associated with the event.
            metadata (Dict[str, Any], optional): Additional metadata for the event.
        """
        self.event_type = event_type
        self.data = data
        self.metadata = metadata or {}

class EventDispatcher:
    """
    Manages event listeners, dispatching, and listener prioritization.
    """

    def __init__(self):
        """
        Initializes the EventDispatcher.
        """
        self.listeners: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self.wildcard_listeners: List[Dict[str, Any]] = []

    def add_listener(self, event_type: str, callback: Callable, priority: int = 0) -> None:
        """
        Adds a listener for a specific event type.

        Args:
            event_type (str): The event type to listen for. Can include wildcards (*).
            callback (Callable): The function to call when the event is dispatched.
            priority (int, optional): The priority of the listener. Higher priority listeners are called first.
        """
        listener = {"callback": callback, "priority": priority}
        if '*' in event_type:
            self.wildcard_listeners.append({"pattern": re.compile(event_type.replace('*', '.*')), **listener})
        else:
            self.listeners[event_type].append(listener)
            self.listeners[event_type].sort(key=lambda x: x["priority"], reverse=True)

    def remove_listener(self, event_type: str, callback: Callable) -> None:
        """
        Removes a listener for a specific event type.

        Args:
            event_type (str): The event type to remove the listener from.
            callback (Callable): The callback function to remove.
        """
        if '*' in event_type:
            self.wildcard_listeners = [l for l in self.wildcard_listeners if l["callback"] != callback]
        else:
            self.listeners[event_type] = [l for l in self.listeners[event_type] if l["callback"] != callback]

    def dispatch_event(self, event: Event) -> None:
        """
        Dispatches an event to all registered listeners.

        This method handles both specific event type listeners and wildcard listeners.
        It sorts listeners by priority and calls them in order.

        Args:
            event (Event): The event to dispatch.

        Raises:
            None, but prints error messages for exceptions in listeners.
        """
        # Create a copy of the listeners for this event type
        listeners_to_call = self.listeners[event.event_type].copy()
        
        # Add matching wildcard listeners to the list
        listeners_to_call.extend(
            [l for l in self.wildcard_listeners if l["pattern"].match(event.event_type)]
        )
        
        # Sort listeners by priority (highest first)
        listeners_to_call.sort(key=lambda x: x["priority"], reverse=True)

        # Call each listener's callback with the event
        for listener in listeners_to_call:
            try:
                listener["callback"](event)
            except Exception as e:
                # Print any errors that occur in listeners
                print(f"Error in event listener: {e}")

    # For compatibility, dispatch_event_sync can be an alias to dispatch_event
    dispatch_event_sync = dispatch_event

# Global event dispatcher instance
global_event_dispatcher = EventDispatcher()
