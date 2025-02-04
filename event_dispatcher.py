# event_dispatcher.py

from collections import defaultdict
import re
import traceback
from typing import Any, Callable, Dict, List, Optional
from loggers import EventLogger
import asyncio
import inspect

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
        # Event types to filter out (won't be logged)
        self.event_filter: List[str] = ['timer', 'state', 'need', 'emotion']
        # Optionally, a list of event types to select for logging
        self.event_select: Optional[List[str]] = None

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
            self.wildcard_listeners.append({
                "pattern": re.compile(event_type.replace('*', '.*')),
                **listener
            })
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

    def _get_listeners(self, event: Event) -> List[Dict[str, Any]]:
        """
        Returns the list of listeners that match the event, including wildcard matches,
        sorted by priority (highest first).
        """
        listeners_to_call = self.listeners[event.event_type].copy()
        listeners_to_call.extend(
            [l for l in self.wildcard_listeners if l["pattern"].match(event.event_type)]
        )
        listeners_to_call.sort(key=lambda x: x["priority"], reverse=True)
        return listeners_to_call
    
    def _log_event(self, event: Event) -> None:
        """
        Performs event logging based on filter/selection rules.
        """
        # Log events that are not filtered out.
        if self.event_filter:
            if not any(event.event_type.startswith(prefix + ':') for prefix in self.event_filter):
                if event.event_type not in ["memory:echo_requested", "cognitive:context_update"]:
                    EventLogger.log_event_dispatch(event.event_type, event.data, event.metadata)
        # Log events that are selected.
        if self.event_select:
            if any(event.event_type.startswith(prefix + ':') for prefix in self.event_select):
                EventLogger.log_event_dispatch(event.event_type, event.data, event.metadata)
    
    def dispatch_event_sync(self, event: Event) -> None:
        """
        Synchronously dispatches an event to all registered listeners.
        
        For legacy code: if a callback returns an awaitable (i.e. is async),
        this method will try to run it in a blocking manner if no event loop is running,
        or schedule it (fire-and-forget) if an event loop is detected.
        """
        self._log_event(event)
        listeners_to_call = self._get_listeners(event)
        for listener in listeners_to_call:
            try:
                result = listener["callback"](event)
                if inspect.isawaitable(result):
                    if not isinstance(result, asyncio.Task):
                        try:
                            # If no event loop is running, we can block using asyncio.run.
                            loop = asyncio.get_running_loop()
                            # If there is a running loop, we cannot block. Schedule it as a task.
                            loop.create_task(result)
                        except RuntimeError:
                            # No running loop; safe to run synchronously.
                            asyncio.run(result)
            except Exception as e:
                print("Error in event listener:")
                print("Event Data:", event.data)
                print("Event Type:", event.event_type)
                print("Listener Callback:", listener["callback"])
                print("Error Type:", type(e).__name__)
                print("Error Message:", str(e))
                print("Traceback:")
                traceback.print_exc()

    async def dispatch_event_async(self, event: Event) -> None:
        """
        Asynchronously dispatches an event to all registered listeners.
        
        This version awaits any asynchronous listener callbacks, ensuring that the
        asynchronous flow is preserved.
        """
        self._log_event(event)
        listeners_to_call = self._get_listeners(event)
        for listener in listeners_to_call:
            try:
                result = listener["callback"](event)
                if inspect.isawaitable(result) and not isinstance(result, asyncio.Task):
                    await result
            except Exception as e:
                print("Error in event listener (async):")
                print("Event Data:", event.data)
                print("Event Type:", event.event_type)
                print("Listener Callback:", listener["callback"])
                print("Error Type:", type(e).__name__)
                print("Error Message:", str(e))
                print("Traceback:")
                traceback.print_exc()

    dispatch_event = dispatch_event_sync

# Global event dispatcher instance
global_event_dispatcher = EventDispatcher()
