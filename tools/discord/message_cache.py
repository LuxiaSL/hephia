"""
Enhanced message cache with real-time updates and sophisticated caching. It provides:

- Real-time message updates (edits, deletions)
- Linked-list sparse indexing for efficient traversal
- Smart cache + API fallback
- Cache coherency during concurrent operations
- Integration with name mapping for format conversion

Key differences from basic caching:
- Uses SortedDict for sparse indexing and efficient range queries
- Maintains linked-list structure for cache coherency
- Handles real-time updates without losing cache consistency
- Provides fallback to Discord API when cache is incomplete
"""

import asyncio
import logging
from typing import Dict, List, Optional, AsyncIterator, Callable, Tuple, Any
from datetime import datetime, timedelta
from collections import defaultdict
import discord
from sortedcontainers import SortedDict

from .bot_models import EnhancedMessage, MessageCacheStats, BotConfig
from .bot_exceptions import CacheError, CacheTimeoutError
from .name_mapping import NameMappingService

logger = logging.getLogger(__name__)


class EnhancedChannelCache:
    """
    Enhanced channel cache with real-time updates and linked-list structure.
    
    Key features:
    - Sparse indexing with SortedDict for efficient range queries
    - Real-time updates for message edits and deletions
    - Cache coherency during concurrent operations
    - Smart fallback to Discord API when cache is incomplete
    """
    
    def __init__(self, channel: discord.TextChannel, name_mapping: NameMappingService):
        self.channel = channel
        self.name_mapping = name_mapping
        self.channel_id = str(channel.id)
        
        # Core cache storage
        # message_id -> discord.Message (None for deleted messages)
        self.messages: Dict[int, Optional[discord.Message]] = {}
        
        # Sparse linked-list structure for efficient traversal
        # message_id -> bool (True if this message links to the next one)
        # This allows us to traverse the cache efficiently without storing
        # explicit next/prev pointers for every message
        self.sparse: SortedDict[int, bool] = SortedDict()
        
        # Whether we have a complete view of the channel from the latest message
        self.up_to_date = False
        
        # Statistics
        self.stats = MessageCacheStats()
        
        # Lock for thread safety
        self._lock = asyncio.Lock()
    
    def _set_prev(self, index: int, func: Callable[[bool, bool], bool]) -> bool:
        """
        Set the previous link in the sparse linked-list structure.
        
        This is a key method from Chapter 2 that maintains cache coherency
        during updates. It handles the complex logic of updating the linked-list
        structure when messages are added or removed.
        
        Args:
            index: Message ID to update
            func: Function to apply to the previous link state
            
        Returns:
            Previous state of the link
        """
        if len(self.sparse) == 0 or index >= self.sparse.keys()[-1]:
            # This is the newest message or cache is empty
            old = self.up_to_date
            self.up_to_date = func(old, True)
            return old
        else:
            # Find the previous message in reverse chronological order
            try:
                prev = next(
                    self.sparse.irange(
                        minimum=index, reverse=False, inclusive=(False, False)
                    )
                )
                old = self.sparse[prev]
                self.sparse[prev] = func(old, False)
                return old
            except StopIteration:
                # No previous message found
                old = self.up_to_date
                self.up_to_date = func(old, True)
                return old
    
    async def update(self, message: discord.Message, latest: bool = False) -> None:
        """
        Update the cache with a new or edited message.
        
        This method handles both new messages and edits, maintaining cache
        coherency and the linked-list structure.
        
        Args:
            message: Discord message to cache
            latest: Whether this is the latest message in the channel
        """
        async with self._lock:
            try:
                # Handle case where message comes in after deletion
                if self.messages.get(message.id, True) is None:
                    logger.debug(f"Ignoring update for previously deleted message {message.id}")
                    return
                
                # Check if we already have this message
                if old := self.messages.get(message.id):
                    # Only update if this is a more recent edit
                    if (message.edited_at and old.edited_at and 
                        message.edited_at <= old.edited_at):
                        logger.debug(f"Ignoring older edit for message {message.id}")
                        return
                
                # Store the message
                self.messages[message.id] = message
                self.stats.total_messages += 1
                
                # Update sparse linked-list structure
                if latest:
                    # This is the newest message, update the chain
                    self.sparse[message.id] = self._set_prev(
                        message.id, lambda prev, last: last or prev
                    )
                else:
                    # This is an older message, just mark it as not linked
                    if message.id not in self.sparse:
                        self.sparse[message.id] = False
                
                logger.debug(f"Updated cache for message {message.id} (latest={latest})")
                
            except Exception as e:
                logger.error(f"Error updating cache for message {message.id}: {e}")
                raise CacheError(f"Failed to update cache: {e}")
    
    async def delete(self, message_id: int) -> None:
        """
        Handle message deletion in the cache.
        
        This method properly updates the linked-list structure when a message
        is deleted, maintaining cache coherency.
        
        Args:
            message_id: ID of the deleted message
        """
        async with self._lock:
            try:
                # Mark message as deleted (None)
                self.messages[message_id] = None
                
                if message_id in self.sparse:
                    # Update the linked-list structure
                    # The logic here maintains the chain by updating the previous link
                    self._set_prev(message_id, lambda prev, last: prev & self.sparse[message_id])
                    del self.sparse[message_id]
                
                logger.debug(f"Deleted message {message_id} from cache")
                
            except Exception as e:
                logger.error(f"Error deleting message {message_id} from cache: {e}")
                raise CacheError(f"Failed to delete from cache: {e}")
    
    async def history(
        self,
        limit: Optional[int] = 100,
        before: Optional[discord.Message] = None,
        after: Optional[discord.Message] = None,
        oldest_first: bool = False
    ) -> AsyncIterator[discord.Message]:
        """
        Get message history with smart cache + API fallback.
        
        This is the core method that provides efficient history traversal
        using the cached data when possible and falling back to the Discord API
        when necessary.
        
        Args:
            limit: Maximum number of messages to return
            before: Get messages before this message
            after: Get messages after this message
            oldest_first: Whether to return oldest messages first
            
        Yields:
            discord.Message objects in the requested order
        """
        async with self._lock:
            try:
                remaining = limit
                beforeid: Optional[int] = before and before.id
                afterid: Optional[int] = after and after.id
                
                # Track messages yielded from cache
                cached_messages = []
                last_cached_id: Optional[int] = None
                
                # First, try to serve from cache
                if (before is None and self.up_to_date) or self.sparse.get(beforeid):
                    logger.debug(f"Serving history from cache for channel {self.channel_id}")
                    self.stats.cache_hits += 1
                    
                    # Create a copy of sparse keys to avoid modification during iteration
                    sparse_items = [
                        (k, self.sparse[k])
                        for k in self.sparse.irange(
                            minimum=afterid,
                            maximum=beforeid,
                            inclusive=(True, False),
                            reverse=True,  # Newest first
                        )
                    ]
                    
                    for message_id, has_next in sparse_items:
                        if message_id == afterid:
                            continue
                        
                        # Get message from cache
                        message = self.messages.get(message_id)
                        if message:  # Skip deleted messages (None)
                            cached_messages.append(message)
                            last_cached_id = message_id
                            
                            if remaining is not None:
                                remaining -= 1
                                if remaining <= 0:
                                    break
                        
                        # If this message doesn't link to the next, we need API fallback
                        if not has_next:
                            break
                
                # If we need more messages, fall back to Discord API
                if remaining is None or remaining > 0:
                    logger.debug(f"Falling back to API for channel {self.channel_id}")
                    self.stats.cache_misses += 1
                    self.stats.api_calls += 1
                    
                    # Determine the starting point for API fetch
                    api_before = None
                    if last_cached_id:
                        api_before = discord.Object(last_cached_id)
                    elif before:
                        api_before = before
                    
                    # Fetch from Discord API
                    try:
                        async with asyncio.timeout(30.0):  # 30 second timeout
                            api_messages = []
                            async for message in self.channel.history(
                                limit=remaining,
                                before=api_before,
                                after=after,
                                oldest_first=False  # Always fetch newest first
                            ):
                                api_messages.append(message)
                                
                                # Update cache with fetched messages
                                await self.update(message, len(api_messages) == 1 and not api_before)
                            
                            # Combine cached and API messages
                            all_messages = cached_messages + api_messages
                            
                    except asyncio.TimeoutError:
                        logger.error(f"API timeout fetching history for channel {self.channel_id}")
                        raise CacheTimeoutError("Discord API timeout")
                    except Exception as e:
                        logger.error(f"API error fetching history for channel {self.channel_id}: {e}")
                        # Still return cached messages if we have them
                        all_messages = cached_messages
                else:
                    # We have enough from cache
                    all_messages = cached_messages
                
                # Sort messages by timestamp (newest first by default)
                all_messages.sort(key=lambda m: m.created_at, reverse=not oldest_first)
                
                # Yield messages in the requested order
                for message in all_messages:
                    yield message
                    
            except Exception as e:
                logger.error(f"Error in history traversal for channel {self.channel_id}: {e}")
                raise CacheError(f"Failed to get history: {e}")
    
    async def get_message(self, message_id: int) -> Optional[discord.Message]:
        """
        Get a specific message by ID.
        
        Args:
            message_id: Discord message ID
            
        Returns:
            discord.Message if found, None otherwise
        """
        async with self._lock:
            # Check cache first
            if message_id in self.messages:
                message = self.messages[message_id]
                if message is not None:  # Not deleted
                    self.stats.cache_hits += 1
                    return message
            
            # Fall back to API
            try:
                self.stats.cache_misses += 1
                self.stats.api_calls += 1
                
                message = await self.channel.fetch_message(message_id)
                await self.update(message)
                return message
                
            except discord.NotFound:
                # Message doesn't exist
                return None
            except Exception as e:
                logger.error(f"Error fetching message {message_id}: {e}")
                return None
    
    def get_stats(self) -> MessageCacheStats:
        """Get cache statistics."""
        return self.stats
    
    def clear(self) -> None:
        """Clear all cached data."""
        with self._lock:
            self.messages.clear()
            self.sparse.clear()
            self.up_to_date = False
            self.stats = MessageCacheStats()
            logger.debug(f"Cleared cache for channel {self.channel_id}")


class MessageCacheManager:
    """
    Manages message caches for all channels with enhanced features.
    
    This is the main interface for message caching, providing:
    - Per-channel cache management
    - Real-time event handling
    - Format conversion integration
    - Cache statistics and monitoring
    """
    
    def __init__(self, bot_client: discord.Client, name_mapping: NameMappingService, config: BotConfig):
        self.bot = bot_client
        self.name_mapping = name_mapping
        self.config = config
        
        # Channel caches: channel_id -> EnhancedChannelCache
        self.channel_caches: Dict[str, EnhancedChannelCache] = {}
        
        # Global statistics
        self.global_stats = MessageCacheStats()
        
        # Cleanup task
        self._cleanup_task: Optional[asyncio.Task] = None
        
        # Lock for cache management
        self._lock = asyncio.Lock()
    
    async def initialize(self) -> None:
        """Initialize the cache manager."""
        logger.info("Initializing message cache manager...")
        
        # Start cleanup task
        self._cleanup_task = asyncio.create_task(self._periodic_cleanup())
        
        # Pre-populate caches for all accessible channels
        await self.populate_all_caches()
        
        logger.info("Message cache manager initialized")
    
    async def shutdown(self) -> None:
        """Shutdown the cache manager."""
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Message cache manager shutdown")
    
    def get_cache(self, channel: discord.TextChannel) -> EnhancedChannelCache:
        """
        Get or create a cache for a channel.
        
        Args:
            channel: Discord channel
            
        Returns:
            EnhancedChannelCache for the channel
        """
        channel_id = str(channel.id)
        
        if channel_id not in self.channel_caches:
            self.channel_caches[channel_id] = EnhancedChannelCache(channel, self.name_mapping)
            logger.debug(f"Created cache for channel {channel.name} ({channel_id})")
        
        return self.channel_caches[channel_id]
    
    async def populate_all_caches(self, messages_per_channel: int = None) -> None:
        """
        Pre-populate caches for all accessible channels.
        
        Args:
            messages_per_channel: Number of messages to cache per channel
        """
        if messages_per_channel is None:
            messages_per_channel = self.config.max_message_cache_size
        
        logger.info(f"Pre-populating caches with {messages_per_channel} messages per channel")
        
        tasks = []
        semaphore = asyncio.Semaphore(5)  # Limit concurrent fetches
        
        async def populate_channel(channel: discord.TextChannel):
            async with semaphore:
                try:
                    cache = self.get_cache(channel)
                    
                    # Fetch recent messages
                    messages = []
                    async for message in channel.history(limit=messages_per_channel):
                        messages.append(message)
                    
                    # Update cache with messages (newest first)
                    for i, message in enumerate(messages):
                        await cache.update(message, i == 0)  # First message is latest
                    
                    logger.debug(f"Populated cache for {channel.name} with {len(messages)} messages")
                    
                except Exception as e:
                    logger.error(f"Error populating cache for {channel.name}: {e}")
        
        # Create tasks for all text channels
        for guild in self.bot.guilds:
            for channel in guild.channels:
                if isinstance(channel, discord.TextChannel):
                    tasks.append(populate_channel(channel))
        
        # Execute all tasks
        await asyncio.gather(*tasks, return_exceptions=True)
        
        logger.info(f"Finished populating {len(self.channel_caches)} channel caches")
    
    # Event handlers for real-time updates
    
    async def handle_message_create(self, message: discord.Message) -> None:
        """Handle new message creation."""
        if isinstance(message.channel, discord.TextChannel):
            cache = self.get_cache(message.channel)
            await cache.update(message, latest=True)
    
    async def handle_message_edit(self, before: discord.Message, after: discord.Message) -> None:
        """Handle message edit."""
        if isinstance(after.channel, discord.TextChannel):
            cache = self.get_cache(after.channel)
            await cache.update(after, latest=False)
    
    async def handle_message_delete(self, message: discord.Message) -> None:
        """Handle message deletion."""
        if isinstance(message.channel, discord.TextChannel):
            cache = self.get_cache(message.channel)
            await cache.delete(message.id)
    
    # Enhanced methods that integrate with name mapping
    
    async def get_enhanced_history(
        self,
        channel: discord.TextChannel,
        limit: int = 100,
        before: Optional[discord.Message] = None,
        after: Optional[discord.Message] = None,
        convert_format: bool = True
    ) -> List[EnhancedMessage]:
        """
        Get enhanced message history with format conversion.
        
        Args:
            channel: Discord channel
            limit: Maximum number of messages
            before: Get messages before this message
            after: Get messages after this message
            convert_format: Whether to convert to LLM format
            
        Returns:
            List of EnhancedMessage objects
        """
        cache = self.get_cache(channel)
        enhanced_messages = []
        
        async for message in cache.history(limit=limit, before=before, after=after):
            # Convert to enhanced message
            enhanced = await self._to_enhanced_message(message, convert_format)
            enhanced_messages.append(enhanced)
        
        return enhanced_messages
    
    async def _to_enhanced_message(self, message: discord.Message, convert_format: bool = True) -> EnhancedMessage:
        """
        Convert a Discord message to an EnhancedMessage.
        
        Args:
            message: Discord message
            convert_format: Whether to convert to LLM format
            
        Returns:
            EnhancedMessage object
        """
        guild_id = str(message.guild.id) if message.guild else None
        channel_path = self.name_mapping.get_channel_path(str(message.channel.id))
        
        # Get author name
        author_name = self.name_mapping.get_user_name(str(message.author.id))
        if not author_name:
            author_name = message.author.display_name
        
        # Convert message content if requested
        if convert_format and guild_id:
            format_result = await self.name_mapping.discord_to_llm_format(message.content, guild_id)
            llm_content = format_result.llm_content
        else:
            llm_content = message.content
        
        return EnhancedMessage(
            id=str(message.id),
            author=author_name,
            discord_author_id=str(message.author.id),
            content=llm_content,
            raw_content=message.content,
            timestamp=message.created_at,
            channel_path=channel_path or f"Unknown:{message.channel.name}",
            channel_id=str(message.channel.id),
            reply_to=str(message.reference.message_id) if message.reference else None,
            attachments=[],  # TODO: Handle attachments
            embeds=[]        # TODO: Handle embeds
        )
    
    async def _periodic_cleanup(self) -> None:
        """Periodic cleanup of old cache entries."""
        while True:
            try:
                await asyncio.sleep(self.config.cache_cleanup_interval)
                
                # Clean up old caches and update statistics
                current_time = datetime.now()
                
                for channel_id, cache in list(self.channel_caches.items()):
                    # Remove very old messages to limit memory usage
                    if len(cache.messages) > self.config.max_message_cache_size:
                        # Remove oldest messages
                        sorted_ids = sorted(cache.messages.keys())
                        to_remove = sorted_ids[:len(cache.messages) - self.config.max_message_cache_size]
                        
                        for msg_id in to_remove:
                            if msg_id in cache.messages:
                                del cache.messages[msg_id]
                            if msg_id in cache.sparse:
                                del cache.sparse[msg_id]
                
                logger.debug("Completed cache cleanup")
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in cache cleanup: {e}")
                await asyncio.sleep(60)  # Wait before retrying
    
    def get_global_stats(self) -> Dict[str, Any]:
        """Get global cache statistics."""
        total_messages = sum(len(cache.messages) for cache in self.channel_caches.values())
        total_cache_hits = sum(cache.stats.cache_hits for cache in self.channel_caches.values())
        total_cache_misses = sum(cache.stats.cache_misses for cache in self.channel_caches.values())
        total_api_calls = sum(cache.stats.api_calls for cache in self.channel_caches.values())
        
        return {
            "total_channels": len(self.channel_caches),
            "total_messages": total_messages,
            "total_cache_hits": total_cache_hits,
            "total_cache_misses": total_cache_misses,
            "total_api_calls": total_api_calls,
            "hit_rate": total_cache_hits / (total_cache_hits + total_cache_misses) if (total_cache_hits + total_cache_misses) > 0 else 0,
            "api_efficiency": total_messages / total_api_calls if total_api_calls > 0 else 0
        }
    
    def clear_all_caches(self) -> None:
        """Clear all caches (for testing/debugging)."""
        for cache in self.channel_caches.values():
            cache.clear()
        self.channel_caches.clear()
        logger.info("Cleared all message caches")