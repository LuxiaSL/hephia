"""
Message processing logic for engagement decisions and forwarding.

This module handles the core message processing logic including:
- Message counting and engagement decisions
- Forwarding messages to Hephia server
- High message count notifications
- Engagement probability calculations
"""

import logging
import math
import random
from typing import Dict, Optional, List, Any
import discord
import aiohttp

from .bot_models import BotConfig
from .name_mapping import NameMappingService
from .message_cache import MessageCacheManager
from .context_windows import ContextWindowManager

logger = logging.getLogger(__name__)


class MessageProcessor:
    """
    Handles message processing logic including engagement decisions and forwarding.
    
    This class encapsulates all the logic for deciding when to engage with messages
    and how to forward them to the Hephia server.
    """
    
    def __init__(
        self,
        bot_client: discord.Client,
        name_mapping: NameMappingService,
        cache_manager: MessageCacheManager,
        context_manager: ContextWindowManager,
        config: BotConfig,
        session: Optional[aiohttp.ClientSession] = None
    ):
        self.bot = bot_client
        self.name_mapping = name_mapping
        self.cache_manager = cache_manager
        self.context_manager = context_manager
        self.config = config
        self.session = session
        
        # Message counting per channel
        self.new_messages: Dict[str, int] = {}
        
        # Hephia server configuration
        self.hephia_server_url = "http://localhost:5517"  # Default from original
        
        # Statistics
        self.messages_processed = 0
        self.messages_forwarded = 0
        self.high_count_notifications = 0
        
        logger.info("Message processor initialized")
    
    async def initialize(self):
        """Initialize the message processor."""
        logger.info("Initializing message processor...")
        # No async initialization needed currently
        logger.info("Message processor initialized")
    
    async def shutdown(self):
        """Shutdown the message processor."""
        logger.info("Shutting down message processor...")
        # No async cleanup needed currently
        logger.info("Message processor shut down")
    
    async def process_message(self, message: discord.Message):
        """
        Process a new message and decide whether to engage.
        
        This is the main entry point for message processing, containing
        all the engagement logic from the original discord_bot.py.
        
        Args:
            message: Discord message to process
        """
        try:
            self.messages_processed += 1
            channel_id = str(message.channel.id)
            
            logger.debug(f"Processing message from {message.author.name} in {message.channel.name}")
            
            # Skip messages that start with a period (command messages)
            if message.content.startswith('.'):
                logger.debug("Skipping message that starts with '.'")
                return
            
            # Initialize channel counter if it doesn't exist
            if channel_id not in self.new_messages:
                self.new_messages[channel_id] = 0
            
            # Skip messages from ourselves or system messages
            if message.author == self.bot.user:
                logger.debug("Resetting counter for own message")
                self.new_messages[channel_id] = 0
                return
            
            # Increment message counter for this channel
            self.new_messages[channel_id] += 1
            
            # Check if we should notify about high message count
            if self.new_messages[channel_id] >= self.config.high_message_threshold:
                await self._notify_high_message_count(message.channel, self.new_messages[channel_id])
                self.new_messages[channel_id] = 0  # Reset after notification
            
            # Determine if we should engage with this message
            should_engage = await self._should_engage(message)
            
            if should_engage:
                logger.info(f"Engaging with message in {message.channel.name}")
                await self._forward_to_hephia(message)
                self.messages_forwarded += 1
                # Reset counter after engagement
                self.new_messages[channel_id] = 0
            
        except Exception as e:
            logger.error(f"Error processing message {message.id}: {e}")
            raise
    
    async def _should_engage(self, message: discord.Message) -> bool:
        """
        Determine if we should engage with a message.
        
        This implements the engagement logic from the original discord_bot.py:
        - Direct mentions or replies
        - Contains "hephia" (75% chance)
        - Sigmoid-scaled probability based on message count
        
        Args:
            message: Discord message to evaluate
            
        Returns:
            True if we should engage, False otherwise
        """
        try:
            channel_id = str(message.channel.id)
            current_count = self.new_messages.get(channel_id, 0)
            
            # Check for direct mention or reply
            if (self.bot.user.mentioned_in(message) or 
                (message.reference and message.reference.resolved and 
                 message.reference.resolved.author == self.bot.user)):
                logger.debug("Engaging due to direct mention or reply")
                return True
            
            # Check for "hephia" keyword (75% chance)
            if "hephia" in message.content.lower():
                if random.random() < 0.75:
                    logger.debug("Engaging due to 'hephia' keyword")
                    return True
                else:
                    logger.debug("'hephia' keyword found but random chance failed")
                    return False
            
            # Calculate sigmoid-scaled probability based on message count
            # Original logic: sigmoid approaches 75% as count nears 100
            x = (current_count - 50) / 15.0  # Normalize and center around 50
            sigmoid = 0.75 / (1 + math.exp(-x))  # Sigmoid scaled to max 75%
            
            if random.random() < sigmoid:
                logger.debug(f"Engaging due to sigmoid probability {sigmoid:.2f} (count: {current_count})")
                return True
            
            logger.debug(f"Not engaging (sigmoid: {sigmoid:.2f}, count: {current_count})")
            return False
            
        except Exception as e:
            logger.error(f"Error in engagement decision: {e}")
            return False
    
    async def _notify_high_message_count(self, channel: discord.TextChannel, count: int):
        """
        Notify Hephia server about high message count in a channel.
        
        Args:
            channel: Discord channel with high message count
            count: Number of new messages
        """
        try:
            if not self.session:
                logger.warning("No session available for high message count notification")
                return
            
            self.high_count_notifications += 1
            
            url = f"{self.hephia_server_url}/discord_channel_update"
            data = {
                "channel_id": str(channel.id),
                "new_message_count": count,
                "channel_name": channel.name,
                "guild_name": channel.guild.name if channel.guild else "DM"
            }
            
            logger.debug(f"Sending high message count notification: {count} messages in {channel.name}")
            
            async with self.session.post(url, json=data) as resp:
                if resp.status != 200:
                    error_text = await resp.text()
                    logger.warning(f"High message count notification failed: {resp.status} {error_text}")
                else:
                    logger.debug("High message count notification sent successfully")
                    
        except Exception as e:
            logger.error(f"Failed to notify about high message count: {e}")
    
    async def _forward_to_hephia(self, message: discord.Message):
        """
        Forward a Discord message to Hephia's /discord_inbound endpoint.
        
        This includes recent message history for context and handles format conversion.
        
        Args:
            message: Discord message to forward
        """
        try:
            if not self.session:
                logger.warning("No session available for message forwarding")
                return
            
            logger.debug(f"Forwarding message {message.id} to Hephia")
            
            # Get author information
            author_ref = self._get_author_reference(message.author)
            
            # Convert message content to LLM format
            guild_id = str(message.guild.id) if message.guild else None
            content = message.content
            
            # Clean up bot mentions in the message
            bot_mention = f'<@{self.bot.user.id}>'
            bot_mention_bang = f'<@!{self.bot.user.id}>'
            content = content.replace(bot_mention, f'@{self.bot.user.name}')
            content = content.replace(bot_mention_bang, f'@{self.bot.user.name}')
            
            # Convert to LLM format if possible
            if guild_id and self.name_mapping:
                format_result = await self.name_mapping.discord_to_llm_format(content, guild_id)
                if format_result.success:
                    content = format_result.llm_content
                else:
                    logger.warning(f"Format conversion failed: {format_result.errors}")
            
            # Get message history for context
            history_messages = await self._get_context_history(message)
            
            # Get current message count
            channel_id = str(message.channel.id)
            current_count = self.new_messages.get(channel_id, 0)
            
            # Prepare the data to send to Hephia
            inbound_data = {
                "channel_id": channel_id,
                "message_id": str(message.id),
                "author": author_ref,
                "author_id": str(message.author.id),
                "content": content,
                "timestamp": message.created_at.isoformat(),
                "context": {
                    "recent_history": history_messages,
                    "channel_name": message.channel.name,
                    "guild_name": message.guild.name if message.guild else "DM",
                    "message_count": current_count,
                    "history_source": "enhanced_cache"
                }
            }
            
            # Send to Hephia
            url = f"{self.hephia_server_url}/discord_inbound"
            
            async with self.session.post(url, json=inbound_data) as resp:
                if resp.status != 200:
                    error_text = await resp.text()
                    logger.warning(f"Error forwarding to Hephia: {resp.status} {error_text}")
                else:
                    logger.info(f"Successfully forwarded message {message.id} to Hephia")
                    
        except Exception as e:
            logger.error(f"Failed to forward message to Hephia: {e}")
    
    def _get_author_reference(self, author: discord.User) -> str:
        """
        Get the appropriate author reference for Hephia.
        
        Args:
            author: Discord user
            
        Returns:
            Author reference string
        """
        # Use the new Discord system if available
        if hasattr(author, 'global_name') and author.global_name:
            return author.global_name
        elif hasattr(author, 'name'):
            return author.name
        else:
            # Legacy system
            return f"{author.display_name}#{author.discriminator}"
    
    async def _get_context_history(self, message: discord.Message, limit: int = 1000) -> List[Dict[str, Any]]:
        """
        Get recent message history for context.
        
        Args:
            message: The current message
            limit: Maximum number of history messages to include
            
        Returns:
            List of formatted message dictionaries
        """
        try:
            if not isinstance(message.channel, discord.TextChannel):
                return []
            
            # Get history from cache
            cache = self.cache_manager.get_cache(message.channel)
            history_messages = []
            
            # Get messages before the current message
            count = 0
            async for hist_msg in cache.history(limit=limit, before=message):
                if count >= limit:
                    break
                
                # Format message for Hephia
                formatted_msg = await self._format_message_for_hephia(hist_msg)
                history_messages.append(formatted_msg)
                count += 1
            
            # Reverse to get chronological order (oldest first)
            history_messages.reverse()
            
            logger.debug(f"Retrieved {len(history_messages)} context messages")
            return history_messages
            
        except Exception as e:
            logger.error(f"Error getting context history: {e}")
            return []
    
    async def _format_message_for_hephia(self, message: discord.Message) -> Dict[str, Any]:
        """
        Format a Discord message for Hephia consumption.
        
        Args:
            message: Discord message to format
            
        Returns:
            Formatted message dictionary
        """
        try:
            # Get author reference
            author_ref = self._get_author_reference(message.author)
            
            # Get message content
            content = message.content
            
            # Clean up bot mentions
            bot_mention = f'<@{self.bot.user.id}>'
            bot_mention_bang = f'<@!{self.bot.user.id}>'
            content = content.replace(bot_mention, f'@{self.bot.user.name}')
            content = content.replace(bot_mention_bang, f'@{self.bot.user.name}')
            
            # Convert to LLM format if possible
            guild_id = str(message.guild.id) if message.guild else None
            if guild_id and self.name_mapping:
                format_result = await self.name_mapping.discord_to_llm_format(content, guild_id)
                if format_result.success:
                    content = format_result.llm_content
            
            return {
                "id": str(message.id),
                "author": author_ref,
                "author_id": str(message.author.id),
                "content": content,
                "timestamp": message.created_at.isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error formatting message for Hephia: {e}")
            return {
                "id": str(message.id),
                "author": "Unknown",
                "author_id": str(message.author.id),
                "content": message.content,
                "timestamp": message.created_at.isoformat()
            }
    
    # Statistics and monitoring
    
    def get_stats(self) -> Dict[str, Any]:
        """Get message processor statistics."""
        return {
            "messages_processed": self.messages_processed,
            "messages_forwarded": self.messages_forwarded,
            "high_count_notifications": self.high_count_notifications,
            "engagement_rate": (
                self.messages_forwarded / self.messages_processed 
                if self.messages_processed > 0 else 0
            ),
            "active_channels": len(self.new_messages),
            "total_pending_messages": sum(self.new_messages.values())
        }
    
    def get_channel_stats(self, channel_id: str) -> Dict[str, Any]:
        """Get statistics for a specific channel."""
        return {
            "channel_id": channel_id,
            "pending_messages": self.new_messages.get(channel_id, 0),
            "threshold": self.config.high_message_threshold
        }
    
    def reset_channel_counter(self, channel_id: str) -> None:
        """Reset the message counter for a specific channel."""
        if channel_id in self.new_messages:
            old_count = self.new_messages[channel_id]
            self.new_messages[channel_id] = 0
            logger.info(f"Reset counter for channel {channel_id}: {old_count} -> 0")
    
    def reset_all_counters(self) -> None:
        """Reset all message counters."""
        total_reset = sum(self.new_messages.values())
        self.new_messages.clear()
        logger.info(f"Reset all counters: {total_reset} total messages")
    
    # Configuration updates
    
    def update_hephia_server_url(self, url: str) -> None:
        """Update the Hephia server URL."""
        self.hephia_server_url = url
        logger.info(f"Updated Hephia server URL to: {url}")
    
    def update_config(self, config: BotConfig) -> None:
        """Update the bot configuration."""
        self.config = config
        logger.info("Updated message processor configuration")