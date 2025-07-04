"""
Discord bot client for WebSocket event handling.

This module provides the core Discord client that handles WebSocket events
and delegates processing to the appropriate service modules. It maintains
clean separation of concerns while preserving all existing functionality.

Key responsibilities:
- Discord WebSocket connection management
- Event handling (on_ready, on_message, on_message_edit, on_message_delete)
- Delegation to service modules
- Bot lifecycle management
"""

import asyncio
import logging
from typing import Optional, Dict, Any
from datetime import datetime
import discord
import aiohttp

from .bot_exceptions import CacheError
from .bot_models import BotConfig, BotStatus
from .name_mapping import NameMappingService
from .message_cache import MessageCacheManager
from .context_windows import ContextWindowManager
from .message_logic import MessageProcessor

logger = logging.getLogger(__name__)


class DiscordBotClient(discord.Client):
    """
    Core Discord bot client that handles WebSocket events.
    
    This client focuses purely on Discord event handling and delegates
    all business logic to the appropriate service modules.
    """
    
    def __init__(
        self,
        config: BotConfig,
        session: Optional[aiohttp.ClientSession] = None,
        **kwargs
    ):
        # Set up Discord intents
        intents = discord.Intents.all()
        super().__init__(intents=intents, **kwargs)
        
        self.config = config
        self.session = session
        self.status = BotStatus()
        
        # Service modules (will be initialized after client is ready)
        self.name_mapping: Optional[NameMappingService] = None
        self.cache_manager: Optional[MessageCacheManager] = None
        self.context_manager: Optional[ContextWindowManager] = None
        self.message_processor: Optional[MessageProcessor] = None
        
        # Initialization flag
        self._services_initialized = False
        
        logger.info("Discord bot client initialized")
    
    async def setup_services(self):
        """Initialize all service modules after bot is ready."""
        if self._services_initialized:
            return
            
        logger.info("Setting up service modules...")
        
        try:
            # Initialize services in dependency order
            self.name_mapping = NameMappingService(self)
            await self.name_mapping.initialize()
            
            self.cache_manager = MessageCacheManager(self, self.name_mapping, self.config)
            await self.cache_manager.initialize()
            
            self.context_manager = ContextWindowManager(self, self.config.context_window_expiry_minutes)
            await self.context_manager.initialize()
            
            self.message_processor = MessageProcessor(
                self, self.name_mapping, self.cache_manager, self.context_manager, self.config, self.session
            )
            await self.message_processor.initialize()
            
            self._services_initialized = True
            logger.info("All service modules initialized successfully")
            
        except Exception as e:
            logger.error(f"Error setting up services: {e}")
            raise
    
    async def shutdown_services(self):
        """Shutdown all service modules."""
        logger.info("Shutting down service modules...")
        
        if self.message_processor:
            await self.message_processor.shutdown()
            
        if self.context_manager:
            await self.context_manager.shutdown()
            
        if self.cache_manager:
            await self.cache_manager.shutdown()
            
        # name_mapping doesn't have async shutdown
        
        logger.info("All service modules shut down")
    
    # Discord event handlers
    
    async def on_ready(self):
        """Handle bot ready event."""
        logger.info(f"Bot logged in as {self.user} (id={self.user.id})")
        
        # Update status
        self.status.is_ready = True
        self.status.connected_guilds = len(self.guilds)
        self.status.uptime_start = datetime.now()
        
        # Set up service modules
        await self.setup_services()
        
        # Update status with service info
        if self.cache_manager:
            cache_stats = self.cache_manager.get_global_stats()
            self.status.total_channels = cache_stats.get("total_channels", 0)
            self.status.cached_messages = cache_stats.get("total_messages", 0)
        
        if self.context_manager:
            context_stats = self.context_manager.get_window_stats()
            self.status.active_context_windows = context_stats.get("active_windows", 0)
        
        logger.info(f"Bot ready - Connected to {self.status.connected_guilds} guilds")
        logger.info(f"Cached {self.status.cached_messages} messages across {self.status.total_channels} channels")
        
        # Set up periodic tasks
        if self.name_mapping:
            asyncio.create_task(self._periodic_mapping_updates())
    
    async def on_message(self, message: discord.Message):
        """Handle new message events."""
        try:
            # Update status
            self.status.last_message_time = datetime.now()
            
            # Ensure services are initialized
            if not self._services_initialized:
                logger.warning("Services not initialized, skipping message processing")
                return
            
            # Update cache first (important for consistency)
            if self.cache_manager and isinstance(message.channel, discord.TextChannel):
                await self.cache_manager.handle_message_create(message)
            
            # Process message through message processor
            if self.message_processor:
                await self.message_processor.process_message(message)
                
        except Exception as e:
            logger.error(f"Error processing message {message.id}: {e}")
            # Add error reaction to message if possible
            try:
                if isinstance(message.channel, discord.TextChannel):
                    await message.add_reaction("⚠️")
            except:
                pass  # Ignore reaction failures
    
    async def on_message_edit(self, before: discord.Message, after: discord.Message):
        """Handle message edit events."""
        try:
            # Update cache
            if self.cache_manager and isinstance(after.channel, discord.TextChannel):
                await self.cache_manager.handle_message_edit(before, after)
                
            logger.debug(f"Message {after.id} edited in {after.channel.name}")
            
        except Exception as e:
            logger.error(f"Error handling message edit for {after.id}: {e}")
    
    async def on_message_delete(self, message: discord.Message):
        """Handle message deletion events."""
        try:
            # Update cache
            if self.cache_manager and isinstance(message.channel, discord.TextChannel):
                await self.cache_manager.handle_message_delete(message)
                
            logger.debug(f"Message {message.id} deleted from {message.channel.name}")
            
        except Exception as e:
            logger.error(f"Error handling message deletion for {message.id}: {e}")
    
    async def on_guild_join(self, guild: discord.Guild):
        """Handle guild join events."""
        try:
            logger.info(f"Bot joined guild: {guild.name} (id={guild.id})")
            
            # Update status
            self.status.connected_guilds = len(self.guilds)
            
            # Update mappings
            if self.name_mapping:
                await self.name_mapping.update_all_mappings()
                
        except Exception as e:
            logger.error(f"Error handling guild join for {guild.name}: {e}")
    
    async def on_guild_remove(self, guild: discord.Guild):
        """Handle guild removal events."""
        try:
            logger.info(f"Bot removed from guild: {guild.name} (id={guild.id})")
            
            # Update status
            self.status.connected_guilds = len(self.guilds)
            
            # Update mappings
            if self.name_mapping:
                await self.name_mapping.update_all_mappings()
                
        except Exception as e:
            logger.error(f"Error handling guild removal for {guild.name}: {e}")
    
    async def on_guild_channel_create(self, channel: discord.abc.GuildChannel):
        """Handle channel creation events."""
        try:
            if isinstance(channel, discord.TextChannel):
                logger.info(f"New text channel created: {channel.name} in {channel.guild.name}")
                
                # Update mappings
                if self.name_mapping:
                    await self.name_mapping.update_channel_mappings()
                    
        except Exception as e:
            logger.error(f"Error handling channel creation for {channel.name}: {e}")
    
    async def on_guild_channel_delete(self, channel: discord.abc.GuildChannel):
        """Handle channel deletion events."""
        try:
            if isinstance(channel, discord.TextChannel):
                logger.info(f"Text channel deleted: {channel.name} from {channel.guild.name}")
                
                # Clean up cache
                if self.cache_manager:
                    channel_id = str(channel.id)
                    if channel_id in self.cache_manager.channel_caches:
                        del self.cache_manager.channel_caches[channel_id]
                
                # Update mappings
                if self.name_mapping:
                    await self.name_mapping.update_channel_mappings()
                    
        except Exception as e:
            logger.error(f"Error handling channel deletion for {channel.name}: {e}")
    
    async def on_member_join(self, member: discord.Member):
        """Handle member join events."""
        try:
            logger.debug(f"Member joined: {member.name} in {member.guild.name}")
            
            # Update user mappings
            if self.name_mapping:
                await self.name_mapping.update_user_mappings()
                
        except Exception as e:
            logger.error(f"Error handling member join for {member.name}: {e}")
    
    async def on_member_remove(self, member: discord.Member):
        """Handle member removal events."""
        try:
            logger.debug(f"Member left: {member.name} from {member.guild.name}")
            
            # Update user mappings
            if self.name_mapping:
                await self.name_mapping.update_user_mappings()
                
        except Exception as e:
            logger.error(f"Error handling member removal for {member.name}: {e}")
    
    async def on_member_update(self, before: discord.Member, after: discord.Member):
        """Handle member update events (name changes, etc.)."""
        try:
            # Check if display name changed
            if before.display_name != after.display_name:
                logger.debug(f"Member name changed: {before.display_name} -> {after.display_name}")
                
                # Update user mappings
                if self.name_mapping:
                    await self.name_mapping.update_user_mappings()
                    
        except Exception as e:
            logger.error(f"Error handling member update for {after.name}: {e}")
    
    async def on_guild_emojis_update(self, guild: discord.Guild, before: list, after: list):
        """Handle emoji update events."""
        try:
            logger.debug(f"Emojis updated in guild: {guild.name}")
            
            # Update emoji mappings
            if self.name_mapping:
                await self.name_mapping.update_emoji_mappings()
                
        except Exception as e:
            logger.error(f"Error handling emoji update for {guild.name}: {e}")
    
    async def on_error(self, event: str, *args, **kwargs):
        """Handle Discord client errors."""
        logger.error(f"Discord client error in event {event}: {args}")
        
        # Try to add error reaction if this was a message event
        if event == "on_message" and args:
            try:
                message = args[0]
                if isinstance(message, discord.Message) and isinstance(message.channel, discord.TextChannel):
                    await message.add_reaction("⚠️")
            except:
                pass
    
    async def _periodic_mapping_updates(self):
        """Periodic task to update name mappings."""
        while True:
            try:
                await asyncio.sleep(300)  # Update every 5 minutes
                
                if self.name_mapping and self.is_ready():
                    await self.name_mapping.periodic_update()
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in periodic mapping update: {e}")
                await asyncio.sleep(60)  # Wait before retrying on error
    
    # Utility methods
    
    def get_status(self) -> BotStatus:
        """Get current bot status."""
        if self.cache_manager:
            cache_stats = self.cache_manager.get_global_stats()
            self.status.cached_messages = cache_stats.get("total_messages", 0)
            self.status.total_channels = cache_stats.get("total_channels", 0)
        
        if self.context_manager:
            context_stats = self.context_manager.get_window_stats()
            self.status.active_context_windows = context_stats.get("active_windows", 0)
        
        return self.status
    
    def get_detailed_stats(self) -> Dict[str, Any]:
        """Get detailed statistics about the bot."""
        stats = {
            "status": self.get_status().__dict__,
            "guilds": len(self.guilds),
            "text_channels": sum(1 for guild in self.guilds for channel in guild.channels if isinstance(channel, discord.TextChannel)),
            "voice_channels": sum(1 for guild in self.guilds for channel in guild.channels if isinstance(channel, discord.VoiceChannel)),
            "users": sum(guild.member_count for guild in self.guilds if guild.member_count),
        }
        
        if self.cache_manager:
            stats["cache"] = self.cache_manager.get_global_stats()
            
        if self.name_mapping:
            stats["mappings"] = self.name_mapping.get_mapping_stats()
            
        if self.context_manager:
            stats["context_windows"] = self.context_manager.get_window_stats()
            
        return stats
    
    async def close(self):
        """Close the bot client and clean up resources."""
        logger.info("Closing Discord bot client...")
        
        # Shutdown services first
        await self.shutdown_services()
        
        # Close Discord connection
        await super().close()
        
        logger.info("Discord bot client closed")