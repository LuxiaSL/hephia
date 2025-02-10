"""
brain/cognition/notification.py
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, Any, List, Callable, Awaitable
from abc import ABC, abstractmethod
import asyncio

@dataclass
class Notification:
    content: Dict[str, Any]    # Raw data about what happened
    source_interface: str      # Which interface created it
    timestamp: datetime        # When it happened

class NotificationManager:
    def __init__(self, max_age: timedelta = timedelta(hours=1)):
        self._notifications: List[Notification] = []
        self._interface_last_check: Dict[str, datetime] = {}
        self._max_age = max_age
        self._lock = asyncio.Lock()
        self._summary_formatters: Dict[str, Callable[[List[Notification]], Awaitable[str]]] = {}
    
    def register_interface(
        self, 
        interface_id: str, 
        summary_formatter: Callable[[List[Notification]], Awaitable[str]]
    ) -> None:
        """Register an interface's summary formatter."""
        self._summary_formatters[interface_id] = summary_formatter
    
    async def add_notification(self, notification: Notification) -> None:
        """Add a new notification."""
        async with self._lock:
            self._notifications.append(notification)
            self._cleanup_old_notifications()
    
    async def get_updates_for_interface(self, interface_id: str) -> str:
        """Get formatted summaries of what other interfaces have done."""
        async with self._lock:
            last_check = self._interface_last_check.get(interface_id, datetime.min)
            self._interface_last_check[interface_id] = datetime.now()
            
            # Group notifications by source interface
            interface_notifications: Dict[str, List[Notification]] = {}
            for notification in self._notifications:
                if (notification.source_interface != interface_id and 
                    notification.timestamp > last_check):
                    notifications = interface_notifications.setdefault(notification.source_interface, [])
                    notifications.append(notification)
            
            # Get formatted summaries for each interface's notifications
            summaries = []
            for src_interface, notifications in interface_notifications.items():
                formatter = self._summary_formatters.get(src_interface)
                if formatter:
                    summary = await formatter(notifications)
                    if summary:
                        summaries.append(summary)
            
            return "\n\n".join(summaries) if summaries else "No recent updates from other interfaces"
    
    def _cleanup_old_notifications(self) -> None:
        """Remove notifications older than max_age."""
        current_time = datetime.now()
        self._notifications = [
            n for n in self._notifications
            if (current_time - n.timestamp) <= self._max_age
        ]

class NotificationInterface(ABC):
    def __init__(self, interface_id: str, notification_manager: NotificationManager):
        self.interface_id = interface_id
        self.notification_manager = notification_manager
        # Register this interface's summary formatter
        self.notification_manager.register_interface(
            interface_id, 
            self._generate_summary
        )
    
    async def create_notification(self, content: Dict[str, Any]) -> Notification:
        """Create and store a notification about what this interface did."""
        notification = Notification(
            content=content,
            source_interface=self.interface_id,
            timestamp=datetime.now()
        )
        await self.notification_manager.add_notification(notification)
        return notification
    
    @abstractmethod
    async def _generate_summary(self, notifications: List[Notification]) -> str:
        """Each interface implements its own summary generation."""
        pass