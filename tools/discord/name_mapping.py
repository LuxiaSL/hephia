"""
Name/ID mapping service for bidirectional Discord abstraction.

This module provides the core abstraction layer that allows LLMs to work with
human-readable names while the bot handles Discord's internal ID system.

Key features:
- Bidirectional user name <-> ID mapping (@john <-> <@123456>)
- Channel path <-> ID mapping (Server:channel <-> channel_id)
- Emoji name <-> ID mapping (:emoji: <-> <:emoji:789>)
- Fallback strategies for missing mappings
- Message format conversion (LLM format <-> Discord format)
"""

import re
import logging
from typing import Dict, List, Optional, Tuple, Set
import discord

from .bot_models import (
    UserMapping, ChannelMapping, EmojiMapping, 
    MessageFormatResult
)
from .bot_exceptions import (
    UserNotFoundError, ChannelNotFoundError, EmojiNotFoundError
)

logger = logging.getLogger(__name__)


class NameMappingService:
    """
    Core service for bidirectional name/ID mapping.
    
    This service maintains mappings between Discord IDs and human-readable names,
    allowing LLMs to work with simple names while the bot handles Discord's 
    internal ID system.
    """
    
    def __init__(self, bot_client: discord.Client):
        self.bot = bot_client
        
        # Channel mappings (enhanced from existing logic)
        self.channel_mappings: Dict[str, ChannelMapping] = {}  # channel_id -> ChannelMapping
        self.path_to_id: Dict[str, str] = {}  # path -> channel_id
        
        # User mappings (new functionality)
        self.user_mappings: Dict[str, UserMapping] = {}  # user_id -> UserMapping
        self.name_to_user_id: Dict[str, str] = {}  # name -> user_id (per guild)
        self.guild_users: Dict[str, Set[str]] = {}  # guild_id -> set of user_ids
        
        # Emoji mappings (new functionality)
        self.emoji_mappings: Dict[str, EmojiMapping] = {}  # emoji_id -> EmojiMapping
        self.emoji_name_to_id: Dict[str, str] = {}  # name -> emoji_id (per guild)
        
        # Regex patterns for message format conversion
        self.discord_mention_pattern = re.compile(r'<@!?(\d+)>')
        self.discord_emoji_pattern = re.compile(r'<a?:(\w+):(\d+)>')
        self.llm_mention_pattern = re.compile(r'@(\w+)')
        self.llm_emoji_pattern = re.compile(r':(\w+):')
        
        # Cache for recent conversions
        self.conversion_cache: Dict[str, MessageFormatResult] = {}
        self.cache_max_size = 1000
        
    async def initialize(self) -> None:
        """Initialize the mapping service with current Discord state."""
        logger.info("Initializing name mapping service...")
        await self.update_all_mappings()
        logger.info("Name mapping service initialized")
    
    async def update_all_mappings(self) -> None:
        """Update all mappings from current Discord state."""
        await self.update_channel_mappings()
        await self.update_user_mappings()
        await self.update_emoji_mappings()
    
    # Channel Mapping Methods (enhanced from existing logic)
    
    async def update_channel_mappings(self) -> None:
        """Update channel mappings from current Discord state."""
        logger.info("Updating channel mappings...")
        
        # Clear existing mappings
        self.channel_mappings.clear()
        self.path_to_id.clear()
        
        for guild in self.bot.guilds:
            guild_id = str(guild.id)
            guild_name = guild.name
            
            for channel in guild.channels:
                if isinstance(channel, discord.TextChannel):
                    channel_id = str(channel.id)
                    channel_name = channel.name
                    
                    # Create mapping
                    mapping = ChannelMapping(
                        discord_id=channel_id,
                        guild_id=guild_id,
                        channel_name=channel_name,
                        guild_name=guild_name,
                        channel_type='text'
                    )
                    
                    self.channel_mappings[channel_id] = mapping
                    
                    # Add all path variations
                    for path in mapping.path_variations:
                        self.path_to_id[path] = channel_id
        
        logger.info(f"Updated {len(self.channel_mappings)} channel mappings")
    
    def find_channel_id(self, path: str) -> Tuple[str, bool]:
        """
        Find channel ID from a path string.
        Enhanced version of existing find_channel_id method.
        
        Args:
            path: Channel path like "Guild:channel" or raw channel ID
            
        Returns:
            Tuple of (channel_id, success)
        """
        # If it's already an ID, return as-is
        if path.isdigit():
            return path, True
        
        # Clean the path
        path = path.strip()
        if (path.startswith('"') and path.endswith('"')) or \
           (path.startswith("'") and path.endswith("'")):
            path = path[1:-1]
        
        # Direct lookup
        if path in self.path_to_id:
            return self.path_to_id[path], True
        
        # Case-insensitive lookup
        path_lower = path.lower()
        if path_lower in self.path_to_id:
            return self.path_to_id[path_lower], True
        
        return "", False
    
    def find_channel_id_strict(self, path: str) -> str:
        """
        Find channel ID from path, raising exception if not found.
        
        Args:
            path: Channel path like "Guild:channel"
            
        Returns:
            Channel ID
            
        Raises:
            ChannelNotFoundError: If path cannot be found
        """
        channel_id, found = self.find_channel_id(path)
        if not found:
            raise ChannelNotFoundError(path)
        return channel_id
    
    def get_channel_path(self, channel_id: str) -> Optional[str]:
        """Get LLM-friendly path for a channel ID."""
        mapping = self.channel_mappings.get(channel_id)
        return mapping.llm_path if mapping else None
    
    # User Mapping Methods (new functionality)
    
    async def update_user_mappings(self) -> None:
        """Update user mappings from current Discord state."""
        logger.info("Updating user mappings...")
        
        # Clear existing mappings
        self.user_mappings.clear()
        self.name_to_user_id.clear()
        self.guild_users.clear()
        
        for guild in self.bot.guilds:
            guild_id = str(guild.id)
            self.guild_users[guild_id] = set()
            
            for member in guild.members:
                user_id = str(member.id)
                self.guild_users[guild_id].add(user_id)
                
                # Create user mapping
                mapping = UserMapping(
                    discord_id=user_id,
                    username=member.name,
                    display_name=member.display_name,
                    global_name=getattr(member, 'global_name', None),
                    guild_id=guild_id
                )
                
                self.user_mappings[user_id] = mapping
                
                # Add name variations for lookup
                names_to_try = [
                    member.display_name.lower(),
                    member.name.lower()
                ]
                
                if hasattr(member, 'global_name') and member.global_name:
                    names_to_try.append(member.global_name.lower())
                
                # Use the first available name (prefer display name)
                for name in names_to_try:
                    if name and name not in self.name_to_user_id:
                        self.name_to_user_id[name] = user_id
                        break
        
        logger.info(f"Updated {len(self.user_mappings)} user mappings")
    
    def find_user_id(self, username: str, guild_id: str = None) -> Optional[str]:
        """
        Find user ID from a username.
        
        Args:
            username: User's display name or username
            guild_id: Optional guild ID to limit search
            
        Returns:
            User ID if found, None otherwise
            
        Raises:
            UserNotFoundError: If username cannot be found (when strict=True)
        """
        username_lower = username.lower()
        
        # Direct lookup
        if username_lower in self.name_to_user_id:
            user_id = self.name_to_user_id[username_lower]
            
            # If guild specified, verify user is in that guild
            if guild_id and user_id not in self.guild_users.get(guild_id, set()):
                return None
                
            return user_id
        
        # Fallback: search through all users
        for user_id, mapping in self.user_mappings.items():
            if guild_id and mapping.guild_id != guild_id:
                continue
                
            if (mapping.username.lower() == username_lower or
                mapping.display_name.lower() == username_lower or
                (mapping.global_name and mapping.global_name.lower() == username_lower)):
                return user_id
        
        return None
    
    def find_user_id_strict(self, username: str, guild_id: str = None) -> str:
        """
        Find user ID from username, raising exception if not found.
        
        Args:
            username: User's display name or username
            guild_id: Optional guild ID to limit search
            
        Returns:
            User ID
            
        Raises:
            UserNotFoundError: If username cannot be found
        """
        user_id = self.find_user_id(username, guild_id)
        if user_id is None:
            guild_name = None
            if guild_id:
                for mapping in self.channel_mappings.values():
                    if mapping.guild_id == guild_id:
                        guild_name = mapping.guild_name
                        break
            raise UserNotFoundError(username, guild_name)
        return user_id
    
    def get_user_name(self, user_id: str) -> Optional[str]:
        """Get LLM-friendly name for a user ID."""
        mapping = self.user_mappings.get(user_id)
        return mapping.llm_name if mapping else None
    
    # Emoji Mapping Methods (new functionality)
    
    async def update_emoji_mappings(self) -> None:
        """Update emoji mappings from current Discord state."""
        logger.info("Updating emoji mappings...")
        
        # Clear existing mappings
        self.emoji_mappings.clear()
        self.emoji_name_to_id.clear()
        
        for guild in self.bot.guilds:
            guild_id = str(guild.id)
            
            for emoji in guild.emojis:
                emoji_id = str(emoji.id)
                
                # Create emoji mapping
                mapping = EmojiMapping(
                    discord_id=emoji_id,
                    name=emoji.name,
                    guild_id=guild_id,
                    animated=emoji.animated
                )
                
                self.emoji_mappings[emoji_id] = mapping
                
                # Add name lookup (guild-specific)
                emoji_key = f"{guild_id}:{emoji.name.lower()}"
                self.emoji_name_to_id[emoji_key] = emoji_id
        
        logger.info(f"Updated {len(self.emoji_mappings)} emoji mappings")
    
    def find_emoji_id(self, emoji_name: str, guild_id: str) -> Optional[str]:
        """Find emoji ID from name in a specific guild."""
        emoji_key = f"{guild_id}:{emoji_name.lower()}"
        return self.emoji_name_to_id.get(emoji_key)
    
    def find_emoji_id_strict(self, emoji_name: str, guild_id: str) -> str:
        """
        Find emoji ID from name, raising exception if not found.
        
        Args:
            emoji_name: Emoji name
            guild_id: Guild ID
            
        Returns:
            Emoji ID
            
        Raises:
            EmojiNotFoundError: If emoji cannot be found
        """
        emoji_id = self.find_emoji_id(emoji_name, guild_id)
        if emoji_id is None:
            guild_name = None
            for mapping in self.channel_mappings.values():
                if mapping.guild_id == guild_id:
                    guild_name = mapping.guild_name
                    break
            raise EmojiNotFoundError(emoji_name, guild_name)
        return emoji_id
    
    def get_emoji_name(self, emoji_id: str) -> Optional[str]:
        """Get LLM-friendly name for an emoji ID."""
        mapping = self.emoji_mappings.get(emoji_id)
        return mapping.llm_format if mapping else None
    
    # Message Format Conversion (core bidirectional functionality)
    
    async def discord_to_llm_format(self, content: str, guild_id: str) -> MessageFormatResult:
        """
        Convert Discord message format to LLM-friendly format.
        
        Converts:
        - <@123456> -> @username
        - <:emoji:789> -> :emoji:
        - Preserves other Discord formatting
        
        Args:
            content: Discord message content
            guild_id: Guild ID for context
            
        Returns:
            MessageFormatResult with converted content
        """
        # Check cache first
        cache_key = f"d2l:{guild_id}:{hash(content)}"
        if cache_key in self.conversion_cache:
            return self.conversion_cache[cache_key]
        
        result = MessageFormatResult(
            success=True,
            llm_content=content,
            discord_content=content  # Original format
        )
        
        try:
            converted_content = content
            
            # Convert mentions: <@123456> -> @username
            for match in self.discord_mention_pattern.finditer(content):
                user_id = match.group(1)
                mention_text = match.group(0)
                
                username = self.get_user_name(user_id)
                if username:
                    converted_content = converted_content.replace(mention_text, f"@{username}")
                else:
                    result.add_warning(f"Could not resolve user ID {user_id}")
            
            # Convert emojis: <:emoji:789> -> :emoji:
            for match in self.discord_emoji_pattern.finditer(content):
                emoji_name = match.group(1)
                emoji_id = match.group(2)
                emoji_text = match.group(0)
                
                # Just use the emoji name for LLM
                converted_content = converted_content.replace(emoji_text, f":{emoji_name}:")
            
            result.llm_content = converted_content
            
        except Exception as e:
            result.add_error(f"Error converting Discord to LLM format: {str(e)}")
            logger.error(f"Discord to LLM conversion error: {e}")
        
        # Cache the result
        self._cache_conversion_result(cache_key, result)
        return result
    
    async def llm_to_discord_format(self, content: str, guild_id: str, strict: bool = False) -> MessageFormatResult:
        """
        Convert LLM message format to Discord format.
        
        Converts:
        - @username -> <@123456>
        - :emoji: -> <:emoji:789>
        - Handles fallbacks for missing mappings
        
        Args:
            content: LLM message content
            guild_id: Guild ID for context
            strict: If True, raise exceptions on mapping failures
            
        Returns:
            MessageFormatResult with converted content
            
        Raises:
            UserNotFoundError: If strict=True and username not found
            EmojiNotFoundError: If strict=True and emoji not found
        """
        # Check cache first
        cache_key = f"l2d:{guild_id}:{hash(content)}"
        if cache_key in self.conversion_cache:
            return self.conversion_cache[cache_key]
        
        result = MessageFormatResult(
            success=True,
            llm_content=content,  # Original format
            discord_content=content
        )
        
        try:
            converted_content = content
            
            # Convert mentions: @username -> <@123456>
            for match in self.llm_mention_pattern.finditer(content):
                username = match.group(1)
                mention_text = match.group(0)
                
                try:
                    if strict:
                        user_id = self.find_user_id_strict(username, guild_id)
                    else:
                        user_id = self.find_user_id(username, guild_id)
                    
                    if user_id:
                        converted_content = converted_content.replace(mention_text, f"<@{user_id}>")
                    else:
                        result.add_warning(f"Could not resolve username '{username}' - mention won't ping")
                        # Keep the @username format as fallback
                        
                except UserNotFoundError as e:
                    if strict:
                        raise
                    result.add_warning(f"Could not resolve username '{username}' - mention won't ping")
            
            # Convert emojis: :emoji: -> <:emoji:789>
            for match in self.llm_emoji_pattern.finditer(content):
                emoji_name = match.group(1)
                emoji_text = match.group(0)
                
                try:
                    if strict:
                        emoji_id = self.find_emoji_id_strict(emoji_name, guild_id)
                    else:
                        emoji_id = self.find_emoji_id(emoji_name, guild_id)
                    
                    if emoji_id:
                        mapping = self.emoji_mappings.get(emoji_id)
                        if mapping:
                            converted_content = converted_content.replace(emoji_text, mapping.discord_format)
                    else:
                        result.add_warning(f"Could not resolve emoji ':{emoji_name}:' - will show as text")
                        # Keep the :emoji: format as fallback
                        
                except EmojiNotFoundError as e:
                    if strict:
                        raise
                    result.add_warning(f"Could not resolve emoji ':{emoji_name}:' - will show as text")
            
            result.discord_content = converted_content
            
        except (UserNotFoundError, EmojiNotFoundError):
            # Re-raise strict mode exceptions
            raise
        except Exception as e:
            result.add_error(f"Error converting LLM to Discord format: {str(e)}")
            logger.error(f"LLM to Discord conversion error: {e}")
        
        # Cache the result
        self._cache_conversion_result(cache_key, result)
        return result
    
    def _cache_conversion_result(self, cache_key: str, result: MessageFormatResult) -> None:
        """Cache a conversion result with size limit."""
        if len(self.conversion_cache) >= self.cache_max_size:
            # Remove oldest entries (simple FIFO)
            keys_to_remove = list(self.conversion_cache.keys())[:100]
            for key in keys_to_remove:
                del self.conversion_cache[key]
        
        self.conversion_cache[cache_key] = result
    
    # Utility Methods
    
    def get_mapping_stats(self) -> Dict[str, int]:
        """Get statistics about current mappings."""
        return {
            "channels": len(self.channel_mappings),
            "users": len(self.user_mappings),
            "emojis": len(self.emoji_mappings),
            "guilds": len(self.guild_users),
            "cache_size": len(self.conversion_cache)
        }
    
    def clear_cache(self) -> None:
        """Clear the conversion cache."""
        self.conversion_cache.clear()
        logger.info("Conversion cache cleared")
    
    async def periodic_update(self) -> None:
        """Periodic update of mappings (call every 5 minutes)."""
        try:
            await self.update_all_mappings()
            logger.debug("Periodic mapping update completed")
        except Exception as e:
            logger.error(f"Error in periodic mapping update: {e}")