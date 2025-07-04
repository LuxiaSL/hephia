"""
Context window management for #N message references.

This module handles the context window system that allows LLMs to reference
messages using simple #N notation without dealing with Discord message IDs.
"""

import asyncio
import logging
from typing import Dict, Optional, List
from datetime import datetime
import discord

from .bot_models import ContextWindow, EnhancedMessage
from .bot_exceptions import (
    NoContextWindowError,
    ContextWindowExpiredError, 
    InvalidWindowTimestampError,
    ReferenceNotInWindowError,
    InvalidMessageIdFormatInWindow,
    ReferencedMessageNotFound,
    ReferencedMessageForbidden
)

logger = logging.getLogger(__name__)


class ContextWindowManager:
    """
    Manages context windows for #N message references.
    
    This service maintains temporary windows of message history that allow
    LLMs to reference specific messages using simple #N notation instead
    of Discord message IDs.
    """
    
    def __init__(self, bot_client: discord.Client, expiry_minutes: int = 5):
        self.bot = bot_client
        self.expiry_minutes = expiry_minutes
        
        # Active context windows: channel_id -> ContextWindow
        self.windows: Dict[str, ContextWindow] = {}
        
        # Cleanup task
        self._cleanup_task: Optional[asyncio.Task] = None
        
    async def initialize(self) -> None:
        """Initialize the context window manager."""
        logger.info("Initializing context window manager...")
        
        # Start the cleanup task
        self._cleanup_task = asyncio.create_task(self._periodic_cleanup())
        
        logger.info("Context window manager initialized")
    
    async def shutdown(self) -> None:
        """Shutdown the context window manager."""
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Context window manager shutdown")
    
    async def create_window(
        self, 
        channel_id: str, 
        channel_path: str, 
        messages: List[EnhancedMessage]
    ) -> ContextWindow:
        """
        Create a new context window for a channel.
        
        Args:
            channel_id: Discord channel ID
            channel_path: LLM-friendly channel path
            messages: List of messages to include in window
            
        Returns:
            Created ContextWindow
        """
        # Create the context window
        window = ContextWindow(
            channel_id=channel_id,
            channel_path=channel_path,
            expiry_minutes=self.expiry_minutes
        )
        
        # Add message references (#1 = newest, #2 = second newest, etc.)
        for i, message in enumerate(messages):
            reference = f"#{i + 1}"
            window.add_message_reference(reference, message.id)
        
        # Store the window
        self.windows[channel_id] = window
        
        logger.debug(f"Created context window for {channel_path} with {len(messages)} messages")
        return window
    
    async def get_window(self, channel_id: str) -> Optional[ContextWindow]:
        """
        Get the context window for a channel.
        
        Args:
            channel_id: Discord channel ID
            
        Returns:
            ContextWindow if exists and not expired, None otherwise
        """
        window = self.windows.get(channel_id)
        if not window:
            return None
            
        if window.is_expired():
            # Remove expired window
            del self.windows[channel_id]
            logger.debug(f"Removed expired context window for channel {channel_id}")
            return None
            
        return window
    
    async def resolve_reference(
        self, 
        channel_id: str, 
        reference: str,
        channel_obj: discord.TextChannel = None
    ) -> discord.Message:
        """
        Resolve a #N reference to a Discord message.
        
        This is the core method that handles the #N -> message_id -> discord.Message
        conversion, with comprehensive error handling.
        
        Args:
            channel_id: Discord channel ID
            reference: The #N reference string
            channel_obj: Optional channel object for message fetching
            
        Returns:
            discord.Message object
            
        Raises:
            NoContextWindowError: If no window exists
            InvalidWindowTimestampError: If window timestamp is malformed
            ContextWindowExpiredError: If window has expired
            ReferenceNotInWindowError: If reference not in window
            InvalidMessageIdFormatInWindow: If message ID is malformed
            ReferencedMessageNotFound: If Discord message not found
            ReferencedMessageForbidden: If Discord access forbidden
        """
        # Validate reference format
        if not (reference.startswith("#") and reference[1:].isdigit()):
            raise ValueError(f"Invalid reference format: {reference}")
        
        # Get the window
        window = self.windows.get(channel_id)
        if not window:
            channel_name = channel_obj.name if channel_obj else f"ID:{channel_id}"
            raise NoContextWindowError(
                f"No context window available for channel '{channel_name}'. "
                f"Please fetch history via enhanced-history first."
            )
        
        # Validate window timestamp
        if not isinstance(window.timestamp, datetime):
            logger.warning(f"Invalid timestamp in context window for channel {channel_id}")
            if channel_id in self.windows:
                del self.windows[channel_id]
            channel_name = channel_obj.name if channel_obj else f"ID:{channel_id}"
            raise InvalidWindowTimestampError(
                f"Corrupted context window timestamp for channel '{channel_name}'. "
                f"Window cleared. Please refresh history."
            )
        
        # Check if window has expired
        if window.is_expired():
            logger.info(f"Context window for channel {channel_id} expired for reference {reference}")
            channel_name = channel_obj.name if channel_obj else f"ID:{channel_id}"
            raise ContextWindowExpiredError(
                f"Context window for channel '{channel_name}' has expired. "
                f"Please refresh history."
            )
        
        # Resolve the reference
        message_id_str = window.resolve_reference(reference)
        if not message_id_str:
            logger.warning(f"Reference {reference} not found in window for channel {channel_id}")
            channel_name = channel_obj.name if channel_obj else f"ID:{channel_id}"
            raise ReferenceNotInWindowError(
                f"Reference '{reference}' not found in the current context window "
                f"for channel '{channel_name}'. Please refresh history."
            )
        
        # Convert message ID to integer
        try:
            message_id_int = int(message_id_str)
        except ValueError:
            logger.error(f"Invalid message ID format '{message_id_str}' for reference {reference}")
            channel_name = channel_obj.name if channel_obj else f"ID:{channel_id}"
            raise InvalidMessageIdFormatInWindow(
                f"Invalid message ID format in context window for reference '{reference}' "
                f"in channel '{channel_name}'."
            )
        
        # Get the channel object if not provided
        if not channel_obj:
            channel_obj = self.bot.get_channel(int(channel_id))
            if not channel_obj:
                raise ValueError(f"Channel with ID {channel_id} not found")
        
        # Try to fetch the message from Discord
        try:
            message_obj = await channel_obj.fetch_message(message_id_int)
            logger.debug(f"Successfully fetched message {message_id_str} (ref: {reference}) from API")
            return message_obj
        except discord.NotFound:
            logger.warning(f"Message {message_id_str} (ref: {reference}) not found in channel {channel_obj.name}")
            raise ReferencedMessageNotFound(
                f"Message for reference '{reference}' (ID: {message_id_str}) "
                f"not found in channel '{channel_obj.name}'. It might have been deleted."
            )
        except discord.Forbidden:
            logger.warning(f"Forbidden to fetch message {message_id_str} (ref: {reference}) from channel {channel_obj.name}")
            raise ReferencedMessageForbidden(
                f"Bot lacks permissions to fetch the referenced message "
                f"(ref: '{reference}', ID: {message_id_str}) in channel '{channel_obj.name}'."
            )
    
    async def resolve_any_reference(
        self, 
        channel_id: str,
        reference: str,
        channel_obj: discord.TextChannel = None,
        cache_manager = None  # MessageCacheManager instance
    ) -> Optional[discord.Message]:
        """
        Resolve any type of message reference.
        
        This method handles both #N references (via context windows) and
        other reference types like "latest", "latest-from:user", "contains:text".
        
        Args:
            channel_id: Discord channel ID
            reference: Message reference string
            channel_obj: Optional channel object
            cache_manager: MessageCacheManager instance for fallback
            
        Returns:
            discord.Message if found, None otherwise
        """
        # Handle #N references via context windows
        if reference.startswith("#") and reference[1:].isdigit():
            try:
                return await self.resolve_reference(channel_id, reference, channel_obj)
            except Exception as e:
                logger.error(f"Error resolving #N reference {reference}: {e}")
                return None
        
        # Handle other reference types
        if not channel_obj:
            channel_obj = self.bot.get_channel(int(channel_id))
            if not channel_obj:
                return None
        
        try:
            # Handle "latest" reference
            if reference.lower() == "latest":
                if cache_manager:
                    # Use the new cache system
                    try:
                        cache = cache_manager.get_cache(channel_obj)
                        async for message in cache.history(limit=1):
                            return message
                    except Exception as e:
                        logger.debug(f"Cache fallback failed for latest: {e}")
                
                # Fallback to API
                async for message in channel_obj.history(limit=1):
                    return message
                return None
            
            # Handle "latest-from:username"
            if reference.lower().startswith("latest-from:"):
                username = reference[12:].strip().lower()
                if not username:
                    return None
                
                # Check cache first
                if cache_manager:
                    try:
                        cache = cache_manager.get_cache(channel_obj)
                        async for message in cache.history(limit=100):
                            if (message.author.name.lower() == username or
                                username in message.author.name.lower() or
                                username in message.author.display_name.lower()):
                                return message
                    except Exception as e:
                        logger.debug(f"Cache fallback failed for latest-from: {e}")
                
                # Fallback to API
                async for message in channel_obj.history(limit=100):
                    if (message.author.name.lower() == username or
                        username in message.author.name.lower() or
                        username in message.author.display_name.lower()):
                        return message
                return None
            
            # Handle "contains:text"
            if reference.lower().startswith("contains:"):
                search_text = reference[9:].strip().lower()
                if not search_text:
                    return None
                
                # Check cache first
                if cache_manager:
                    try:
                        cache = cache_manager.get_cache(channel_obj)
                        async for message in cache.history(limit=100):
                            if search_text in message.content.lower():
                                return message
                    except Exception as e:
                        logger.debug(f"Cache fallback failed for contains: {e}")
                
                # Fallback to API
                async for message in channel_obj.history(limit=100):
                    if search_text in message.content.lower():
                        return message
                return None
            
            # Unknown reference type
            logger.warning(f"Unknown reference format: {reference}")
            return None
            
        except Exception as e:
            logger.error(f"Error resolving reference {reference}: {e}")
            return None
    
    async def _periodic_cleanup(self) -> None:
        """Periodically clean up expired context windows."""
        while True:
            try:
                await asyncio.sleep(60)  # Check every minute
                
                now = datetime.now()
                expired_channels = []
                
                for channel_id, window in self.windows.items():
                    if window.is_expired():
                        expired_channels.append(channel_id)
                
                for channel_id in expired_channels:
                    if channel_id in self.windows:
                        del self.windows[channel_id]
                        logger.debug(f"Cleaned up expired context window for channel {channel_id}")
                
                if expired_channels:
                    logger.info(f"Cleaned up {len(expired_channels)} expired context windows")
                
            except asyncio.CancelledError:
                logger.info("Context window cleanup task cancelled")
                break
            except Exception as e:
                logger.error(f"Error during context window cleanup: {e}")
                await asyncio.sleep(120)  # Wait longer on error
    
    def get_window_stats(self) -> Dict[str, int]:
        """Get statistics about current context windows."""
        active_windows = len(self.windows)
        expired_windows = sum(1 for window in self.windows.values() if window.is_expired())
        
        return {
            "active_windows": active_windows,
            "expired_windows": expired_windows,
            "total_references": sum(len(window.messages) for window in self.windows.values())
        }
    
    def clear_all_windows(self) -> None:
        """Clear all context windows (for testing/debugging)."""
        count = len(self.windows)
        self.windows.clear()
        logger.info(f"Cleared {count} context windows")
    
    def get_window_info(self, channel_id: str) -> Optional[Dict[str, any]]:
        """Get information about a specific context window."""
        window = self.windows.get(channel_id)
        if not window:
            return None
        
        return {
            "channel_id": window.channel_id,
            "channel_path": window.channel_path,
            "created": window.timestamp.isoformat(),
            "expires": window.expires_at.isoformat(),
            "is_expired": window.is_expired(),
            "message_count": len(window.messages),
            "references": list(window.messages.keys())
        }