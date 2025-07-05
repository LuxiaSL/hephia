"""
HTTP API handlers for Discord bot endpoints.

This module provides all the HTTP endpoint handlers that allow external services
(like core/discord_service.py) to interact with the Discord bot. It provides
detailed error reporting and integrates with all the new modular components.

Key features:
- Comprehensive error reporting with specific error codes
- Integration with name mapping for bidirectional format conversion
- Enhanced caching for performance
- Context window management for #N references
- Detailed logging for debugging
"""

import asyncio
import logging
import traceback
from typing import Dict, Any, Optional, List
from datetime import datetime
from aiohttp import web
import discord

from bot_exceptions import CacheTimeoutError, ContextWindowError
from name_mapping import NameMappingService
from message_cache import MessageCacheManager
from context_windows import ContextWindowManager
from bot_client import DiscordBotClient

logger = logging.getLogger(__name__)


class APIResponse:
    """Helper class for creating standardized API responses."""
    
    @staticmethod
    def success(data: Any = None, message: str = "Success") -> Dict[str, Any]:
        """Create a success response."""
        response = {
            "success": True,
            "message": message,
            "timestamp": datetime.utcnow().isoformat(),
            "error_code": None
        }
        if data is not None:
            response["data"] = data
        return response
    
    @staticmethod
    def error(
        error_code: str,
        message: str,
        details: str = None,
        status_code: int = 400,
        suggestions: List[str] = None,
        data: Any = None
    ) -> Dict[str, Any]:
        """Create an error response with detailed information."""
        response = {
            "success": False,
            "message": message,
            "error_code": error_code,
            "status_code": status_code,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        if details:
            response["details"] = details
        if suggestions:
            response["suggestions"] = suggestions
        if data:
            response["debug_data"] = data
            
        return response


class DiscordAPIHandlers:
    """
    HTTP API handlers for Discord bot endpoints.
    
    This class provides all the HTTP endpoint handlers and integrates with
    the modular components for clean separation of concerns.
    """
    
    def __init__(
        self,
        bot_client: DiscordBotClient,
        name_mapping: NameMappingService,
        cache_manager: MessageCacheManager,
        context_manager: ContextWindowManager
    ):
        self.bot = bot_client
        self.name_mapping = name_mapping
        self.cache_manager = cache_manager
        self.context_manager = context_manager
        
        # Request tracking for debugging
        self.request_count = 0
        
    async def handle_list_guilds_with_channels(self, request: web.Request) -> web.Response:
        """
        GET /guilds-with-channels
        
        Returns hierarchical guild and channel information.
        Enhanced with detailed error reporting.
        """
        self.request_count += 1
        request_id = f"req_{self.request_count}"
        
        logger.info(f"[{request_id}] Processing list_guilds_with_channels request")
        
        try:
            if not self.bot.is_ready():
                return web.json_response(
                    APIResponse.error(
                        "BOT_NOT_READY",
                        "Discord bot is not ready yet",
                        details="Bot is still connecting to Discord servers",
                        status_code=503,
                        suggestions=["Wait a few seconds and try again"]
                    ),
                    status=503
                )
            
            result = []
            
            for guild in self.bot.guilds:
                guild_data = {
                    "id": str(guild.id),
                    "name": guild.name,
                    "channels": []
                }
                
                # Get bot member for permission checking
                bot_member = guild.get_member(self.bot.user.id)
                if not bot_member:
                    logger.warning(f"[{request_id}] Bot member not found in guild {guild.name}")
                    continue
                
                # Process text channels
                for channel in guild.channels:
                    if isinstance(channel, discord.TextChannel):
                        # Check permissions
                        channel_perms = channel.permissions_for(bot_member)
                        
                        if channel_perms.view_channel and channel_perms.send_messages:
                            channel_data = {
                                "id": str(channel.id),
                                "name": channel.name,
                                "path": f"{guild.name}:{channel.name}",
                                "permissions": {
                                    "view_channel": channel_perms.view_channel,
                                    "send_messages": channel_perms.send_messages,
                                    "read_message_history": channel_perms.read_message_history
                                }
                            }
                            guild_data["channels"].append(channel_data)
                
                # Only include guilds with accessible channels
                if guild_data["channels"]:
                    result.append(guild_data)
            
            logger.info(f"[{request_id}] Successfully listed {len(result)} guilds with channels")
            
            return web.json_response(
                APIResponse.success(
                    data=result,
                    message=f"Listed {len(result)} guilds with accessible channels"
                )
            )
            
        except Exception as e:
            logger.error(f"[{request_id}] Error in list_guilds_with_channels: {e}")
            return web.json_response(
                APIResponse.error(
                    "INTERNAL_ERROR",
                    "Failed to list guilds and channels",
                    details=str(e),
                    status_code=500,
                    suggestions=["Check bot permissions", "Verify bot is connected to Discord"]
                ),
                status=500
            )
    
    async def handle_send_by_path(self, request: web.Request) -> web.Response:
        """
        POST /send-by-path
        
        Send a message using friendly path format with format conversion.
        Enhanced with detailed error reporting and format conversion.
        """
        self.request_count += 1
        request_id = f"req_{self.request_count}"
        
        logger.info(f"[{request_id}] Processing send_by_path request")
        
        try:
            # Parse request body
            try:
                data = await request.json()
            except Exception as e:
                logger.warning(f"[{request_id}] Invalid JSON in request: {e}")
                return web.json_response(
                    APIResponse.error(
                        "INVALID_JSON",
                        "Invalid JSON in request body",
                        details=str(e),
                        status_code=400,
                        suggestions=["Check JSON syntax", "Ensure Content-Type is application/json"]
                    ),
                    status=400
                )
            
            # Validate required fields
            path = data.get("path", "").strip()
            content = data.get("content", "").strip()
            
            if not path:
                return web.json_response(
                    APIResponse.error(
                        "MISSING_PATH",
                        "Path parameter is required",
                        details="Path should be in 'Guild:channel' format",
                        status_code=400,
                        suggestions=["Provide path in format 'ServerName:channelname'"]
                    ),
                    status=400
                )
            
            if not content:
                return web.json_response(
                    APIResponse.error(
                        "MISSING_CONTENT",
                        "Content parameter is required",
                        details="Message content cannot be empty",
                        status_code=400,
                        suggestions=["Provide message content"]
                    ),
                    status=400
                )
            
            logger.debug(f"[{request_id}] Sending message to '{path}' with content length {len(content)}")
            
            # Resolve channel path
            channel_id, found = self.name_mapping.find_channel_id(path)
            if not found:
                logger.warning(f"[{request_id}] Channel path '{path}' not found")
                return web.json_response(
                    APIResponse.error(
                        "CHANNEL_NOT_FOUND",
                        f"Channel path '{path}' not found",
                        details="Channel path should be in 'Guild:channel' format",
                        status_code=404,
                        suggestions=[
                            "Use /guilds-with-channels to see available channels",
                            "Check spelling of guild and channel names",
                            "Ensure bot has access to the channel"
                        ]
                    ),
                    status=404
                )
            
            # Get channel object
            channel = self.bot.get_channel(int(channel_id))
            if not channel or not isinstance(channel, discord.TextChannel):
                logger.warning(f"[{request_id}] Channel {channel_id} not found or not a text channel")
                return web.json_response(
                    APIResponse.error(
                        "CHANNEL_INVALID",
                        "Channel not found or not a text channel",
                        details=f"Channel ID {channel_id} resolved from path '{path}'",
                        status_code=404,
                        suggestions=["Verify channel exists and is a text channel"]
                    ),
                    status=404
                )
            
            # Convert message format from LLM to Discord
            guild_id = str(channel.guild.id)
            format_result = await self.name_mapping.llm_to_discord_format(content, guild_id)
            
            if not format_result.success:
                logger.warning(f"[{request_id}] Format conversion failed: {format_result.errors}")
                return web.json_response(
                    APIResponse.error(
                        "FORMAT_CONVERSION_ERROR",
                        "Failed to convert message format",
                        details="; ".join(format_result.errors),
                        status_code=400,
                        suggestions=["Check username and emoji references"],
                        data={"warnings": format_result.warnings}
                    ),
                    status=400
                )
            
            # Handle message length limits
            MAX_LENGTH = 2000
            discord_content = format_result.discord_content
            
            if len(discord_content) > MAX_LENGTH:
                logger.warning(f"[{request_id}] Message too long ({len(discord_content)} chars)")
                discord_content = discord_content[:MAX_LENGTH]
            
            # Send the message
            try:
                sent_message = await channel.send(discord_content)
                logger.info(f"[{request_id}] Successfully sent message {sent_message.id} to {channel.name}")
                
                # Update cache
                await self.cache_manager.handle_message_create(sent_message)
                
                # Prepare response data
                response_data = {
                    "message_id": str(sent_message.id),
                    "channel_id": channel_id,
                    "channel_path": path,
                    "guild_name": channel.guild.name,
                    "channel_name": channel.name,
                    "content_length": len(content),
                    "discord_content_length": len(discord_content),
                    "format_warnings": format_result.warnings
                }
                
                return web.json_response(
                    APIResponse.success(
                        data=response_data,
                        message=f"Message sent successfully to {channel.name}"
                    )
                )
                
            except discord.Forbidden as e:
                logger.error(f"[{request_id}] Permission denied sending to {channel.name}: {e}")
                return web.json_response(
                    APIResponse.error(
                        "PERMISSION_DENIED",
                        "Permission denied sending message",
                        details=str(e),
                        status_code=403,
                        suggestions=[
                            "Check bot permissions in the channel",
                            "Ensure bot has 'Send Messages' permission",
                            "Verify bot is not muted or restricted"
                        ]
                    ),
                    status=403
                )
            except discord.HTTPException as e:
                logger.error(f"[{request_id}] Discord API error: {e}")
                return web.json_response(
                    APIResponse.error(
                        "DISCORD_API_ERROR",
                        "Discord API error",
                        details=str(e),
                        status_code=500,
                        suggestions=["Try again in a few moments", "Check Discord API status"]
                    ),
                    status=500
                )
            
        except Exception as e:
            logger.error(f"[{request_id}] Unexpected error in send_by_path: {e}")
            logger.error(traceback.format_exc())
            return web.json_response(
                APIResponse.error(
                    "INTERNAL_ERROR",
                    "Internal server error",
                    details=str(e),
                    status_code=500,
                    suggestions=["Contact administrator", "Check server logs"]
                ),
                status=500
            )
    
    async def handle_enhanced_history(self, request: web.Request) -> web.Response:
        """
        GET /enhanced-history?path=Server:channel&limit=50
        
        Get enhanced message history with context window creation.
        Enhanced with detailed error reporting and format conversion.
        """
        self.request_count += 1
        request_id = f"req_{self.request_count}"
        
        logger.info(f"[{request_id}] Processing enhanced_history request")
        
        try:
            # Parse query parameters
            path = request.query.get("path", "").strip()
            limit_str = request.query.get("limit", "50")
            
            if not path:
                return web.json_response(
                    APIResponse.error(
                        "MISSING_PATH",
                        "Path parameter is required",
                        details="Path should be in 'Guild:channel' format",
                        status_code=400,
                        suggestions=["Provide path as ?path=ServerName:channelname"]
                    ),
                    status=400
                )
            
            # Parse limit
            try:
                limit = int(limit_str)
                if limit <= 0:
                    limit = 50
                elif limit > 1000:
                    limit = 1000
            except ValueError:
                limit = 50
            
            logger.debug(f"[{request_id}] Getting history for '{path}' with limit {limit}")
            
            # Resolve channel path
            channel_id, found = self.name_mapping.find_channel_id(path)
            if not found:
                logger.warning(f"[{request_id}] Channel path '{path}' not found")
                return web.json_response(
                    APIResponse.error(
                        "CHANNEL_NOT_FOUND",
                        f"Channel path '{path}' not found",
                        details="Channel path should be in 'Guild:channel' format",
                        status_code=404,
                        suggestions=[
                            "Use /guilds-with-channels to see available channels",
                            "Check spelling of guild and channel names"
                        ]
                    ),
                    status=404
                )
            
            # Get channel object
            channel = self.bot.get_channel(int(channel_id))
            if not channel or not isinstance(channel, discord.TextChannel):
                logger.warning(f"[{request_id}] Channel {channel_id} not found or not a text channel")
                return web.json_response(
                    APIResponse.error(
                        "CHANNEL_INVALID",
                        "Channel not found or not a text channel",
                        details=f"Channel ID {channel_id} resolved from path '{path}'",
                        status_code=404,
                        suggestions=["Verify channel exists and is a text channel"]
                    ),
                    status=404
                )
            
            # Get enhanced history
            try:
                enhanced_messages = await self.cache_manager.get_enhanced_history(
                    channel, limit=limit, convert_format=True
                )
                
                # Format messages for response
                formatted_messages = []
                message_references = {}
                
                for i, enhanced_msg in enumerate(enhanced_messages):
                    # Add #N reference
                    reference = f"#{i + 1}"
                    message_references[reference] = enhanced_msg.id
                    
                    # Format for LLM
                    formatted_msg = enhanced_msg.to_llm_format()
                    formatted_msg["reference"] = reference
                    formatted_messages.append(formatted_msg)
                
                formatted_messages.reverse()  # Newest first

                # Create context window
                if message_references:
                    await self.context_manager.create_window(
                        channel_id, path, enhanced_messages
                    )
                
                # Prepare response
                response_data = {
                    "path": path,
                    "channel_id": channel_id,
                    "guild_name": channel.guild.name,
                    "channel_name": channel.name,
                    "message_count": len(formatted_messages),
                    "messages": formatted_messages,
                    "display_order": "oldest_first",
                    "numbering": "newest_first",
                    "context_window_created": len(message_references) > 0,
                    "cache_stats": self.cache_manager.get_cache(channel).get_stats().to_dict()
                }
                
                logger.info(f"[{request_id}] Successfully retrieved {len(formatted_messages)} messages")
                
                return web.json_response(
                    APIResponse.success(
                        data=response_data,
                        message=f"Retrieved {len(formatted_messages)} messages from {channel.name}"
                    )
                )
                
            except CacheTimeoutError as e:
                logger.error(f"[{request_id}] Cache timeout: {e}")
                return web.json_response(
                    APIResponse.error(
                        "CACHE_TIMEOUT",
                        "Request timed out while fetching history",
                        details=str(e),
                        status_code=504,
                        suggestions=["Try again with a smaller limit", "Check Discord API status"]
                    ),
                    status=504
                )
            except discord.Forbidden as e:
                logger.error(f"[{request_id}] Permission denied: {e}")
                return web.json_response(
                    APIResponse.error(
                        "PERMISSION_DENIED",
                        "Permission denied accessing channel history",
                        details=str(e),
                        status_code=403,
                        suggestions=[
                            "Check bot permissions in the channel",
                            "Ensure bot has 'Read Message History' permission"
                        ]
                    ),
                    status=403
                )
            
        except Exception as e:
            logger.error(f"[{request_id}] Unexpected error in enhanced_history: {e}")
            logger.error(traceback.format_exc())
            return web.json_response(
                APIResponse.error(
                    "INTERNAL_ERROR",
                    "Internal server error",
                    details=str(e),
                    status_code=500,
                    suggestions=["Contact administrator", "Check server logs"]
                ),
                status=500
            )
    
    async def handle_find_message(self, request: web.Request) -> web.Response:
        """
        GET /find-message?path=Server:channel&reference=<reference>
        
        Find a specific message using various reference types.
        Enhanced with detailed error reporting and context window integration.
        """
        self.request_count += 1
        request_id = f"req_{self.request_count}"
        
        logger.info(f"[{request_id}] Processing find_message request")
        
        try:
            # Parse query parameters
            path = request.query.get("path", "").strip()
            raw_reference = request.query.get("reference", "").strip()
            
            if not path:
                return web.json_response(
                    APIResponse.error(
                        "MISSING_PATH",
                        "Path parameter is required",
                        status_code=400,
                        suggestions=["Provide path as ?path=ServerName:channelname"]
                    ),
                    status=400
                )
            
            if not raw_reference:
                return web.json_response(
                    APIResponse.error(
                        "MISSING_REFERENCE",
                        "Reference parameter is required",
                        details="Reference can be #N, 'latest', 'latest-from:user', or 'contains:text'",
                        status_code=400,
                        suggestions=["Provide reference as ?reference=#1 or ?reference=latest"]
                    ),
                    status=400
                )
            
            reference = self._sanitize_reference(raw_reference)
            logger.debug(f"[{request_id}] Finding message in '{path}' with reference '{reference}'")
            
            # Resolve channel path
            channel_id, found = self.name_mapping.find_channel_id(path)
            if not found:
                return web.json_response(
                    APIResponse.error(
                        "CHANNEL_NOT_FOUND",
                        f"Channel path '{path}' not found",
                        status_code=404,
                        suggestions=["Use /guilds-with-channels to see available channels"]
                    ),
                    status=404
                )
            
            # Get channel object
            channel = self.bot.get_channel(int(channel_id))
            if not channel or not isinstance(channel, discord.TextChannel):
                return web.json_response(
                    APIResponse.error(
                        "CHANNEL_INVALID",
                        "Channel not found or not a text channel",
                        status_code=404
                    ),
                    status=404
                )
            
            # Resolve the reference
            try:
                message = await self.context_manager.resolve_any_reference(
                    channel_id, reference, channel, self.cache_manager
                )
                
                if not message:
                    return web.json_response(
                        APIResponse.error(
                            "MESSAGE_NOT_FOUND",
                            f"No message found matching reference '{reference}'",
                            details="Message may have been deleted or reference may be invalid",
                            status_code=404,
                            suggestions=[
                                "Try a different reference",
                                "Use /enhanced-history to see available messages",
                                "Check if message was deleted"
                            ]
                        ),
                        status=404
                    )
                
                # Convert to enhanced message
                enhanced_msg = await self.cache_manager._to_enhanced_message(message, convert_format=True)
                formatted_msg = enhanced_msg.to_llm_format()
                
                logger.info(f"[{request_id}] Successfully found message {message.id}")
                
                return web.json_response(
                    APIResponse.success(
                        data=formatted_msg,
                        message=f"Found message matching reference '{reference}'"
                    )
                )
                
            except ContextWindowError as e:
                logger.warning(f"[{request_id}] Context window error: {e}")
                return web.json_response(
                    APIResponse.error(
                        "CONTEXT_WINDOW_ERROR",
                        str(e),
                        details="Context window issue with #N reference",
                        status_code=e.status_code,
                        suggestions=[
                            "Use /enhanced-history to refresh context window",
                            "Try a different reference type"
                        ]
                    ),
                    status=e.status_code
                )
            
        except Exception as e:
            logger.error(f"[{request_id}] Unexpected error in find_message: {e}")
            logger.error(traceback.format_exc())
            return web.json_response(
                APIResponse.error(
                    "INTERNAL_ERROR",
                    "Internal server error",
                    details=str(e),
                    status_code=500
                ),
                status=500
            )
    
    async def handle_reply_to_message(self, request: web.Request) -> web.Response:
        """
        POST /reply-to-message
        
        Reply to a specific message with format conversion.
        Enhanced with detailed error reporting and context window integration.
        """
        self.request_count += 1
        request_id = f"req_{self.request_count}"
        
        logger.info(f"[{request_id}] Processing reply_to_message request")
        
        try:
            # Parse request body
            try:
                data = await request.json()
            except Exception as e:
                return web.json_response(
                    APIResponse.error(
                        "INVALID_JSON",
                        "Invalid JSON in request body",
                        details=str(e),
                        status_code=400
                    ),
                    status=400
                )
            
            # Validate required fields
            path = data.get("path", "").strip()
            raw_reference = data.get("reference", "").strip()
            content = data.get("content", "").strip()
            
            if not all([path, raw_reference, content]):
                return web.json_response(
                    APIResponse.error(
                        "MISSING_PARAMETERS",
                        "Path, reference, and content are all required",
                        details="All three parameters must be provided",
                        status_code=400,
                        suggestions=["Provide path, reference, and content in JSON body"]
                    ),
                    status=400
                )
            
            reference = self._sanitize_reference(raw_reference)
            logger.debug(f"[{request_id}] Replying to '{reference}' in '{path}'")
            
            # Resolve channel path
            channel_id, found = self.name_mapping.find_channel_id(path)
            if not found:
                return web.json_response(
                    APIResponse.error(
                        "CHANNEL_NOT_FOUND",
                        f"Channel path '{path}' not found",
                        status_code=404,
                        suggestions=["Use /guilds-with-channels to see available channels"]
                    ),
                    status=404
                )
            
            # Get channel object
            channel = self.bot.get_channel(int(channel_id))
            if not channel or not isinstance(channel, discord.TextChannel):
                return web.json_response(
                    APIResponse.error(
                        "CHANNEL_INVALID",
                        "Channel not found or not a text channel",
                        status_code=404
                    ),
                    status=404
                )
            
            # Find the message to reply to
            try:
                target_message = await self.context_manager.resolve_any_reference(
                    channel_id, reference, channel, self.cache_manager
                )
                
                if not target_message:
                    return web.json_response(
                        APIResponse.error(
                            "TARGET_MESSAGE_NOT_FOUND",
                            f"No message found matching reference '{reference}'",
                            details="Cannot reply to a message that doesn't exist",
                            status_code=404,
                            suggestions=[
                                "Use /find-message to verify the reference",
                                "Check if message was deleted"
                            ]
                        ),
                        status=404
                    )
                
                # Convert reply content format
                guild_id = str(channel.guild.id)
                format_result = await self.name_mapping.llm_to_discord_format(content, guild_id)
                
                if not format_result.success:
                    return web.json_response(
                        APIResponse.error(
                            "FORMAT_CONVERSION_ERROR",
                            "Failed to convert reply format",
                            details="; ".join(format_result.errors),
                            status_code=400,
                            data={"warnings": format_result.warnings}
                        ),
                        status=400
                    )
                
                # Send the reply
                discord_content = format_result.discord_content
                MAX_LENGTH = 2000
                if len(discord_content) > MAX_LENGTH:
                    discord_content = discord_content[:MAX_LENGTH]
                
                reply_message = await channel.send(discord_content, reference=target_message)
                
                # Update cache
                await self.cache_manager.handle_message_create(reply_message)
                
                # Convert both messages to enhanced format
                reply_enhanced = await self.cache_manager._to_enhanced_message(reply_message, convert_format=True)
                target_enhanced = await self.cache_manager._to_enhanced_message(target_message, convert_format=True)
                
                response_data = {
                    "reply_message": reply_enhanced.to_llm_format(),
                    "target_message": target_enhanced.to_llm_format(),
                    "channel_path": path,
                    "format_warnings": format_result.warnings
                }
                
                logger.info(f"[{request_id}] Successfully sent reply {reply_message.id}")
                
                return web.json_response(
                    APIResponse.success(
                        data=response_data,
                        message=f"Reply sent successfully to message {target_message.id}"
                    )
                )
                
            except ContextWindowError as e:
                logger.warning(f"[{request_id}] Context window error: {e}")
                return web.json_response(
                    APIResponse.error(
                        "CONTEXT_WINDOW_ERROR",
                        str(e),
                        status_code=e.status_code,
                        suggestions=[
                            "Use /enhanced-history to refresh context window",
                            "Try a different reference type"
                        ]
                    ),
                    status=e.status_code
                )
            except discord.Forbidden as e:
                logger.error(f"[{request_id}] Permission denied: {e}")
                return web.json_response(
                    APIResponse.error(
                        "PERMISSION_DENIED",
                        "Permission denied sending reply",
                        details=str(e),
                        status_code=403,
                        suggestions=["Check bot permissions in the channel"]
                    ),
                    status=403
                )
            
        except Exception as e:
            logger.error(f"[{request_id}] Unexpected error in reply_to_message: {e}")
            logger.error(traceback.format_exc())
            return web.json_response(
                APIResponse.error(
                    "INTERNAL_ERROR",
                    "Internal server error",
                    details=str(e),
                    status_code=500
                ),
                status=500
            )
    
    async def handle_get_user_list(self, request: web.Request) -> web.Response:
        """
        GET /user-list?path=Server:channel
        
        Get a list of users in a channel with enhanced error reporting.
        """
        self.request_count += 1
        request_id = f"req_{self.request_count}"
        
        logger.info(f"[{request_id}] Processing get_user_list request")
        
        try:
            path = request.query.get("path", "").strip()
            
            if not path:
                return web.json_response(
                    APIResponse.error(
                        "MISSING_PATH",
                        "Path parameter is required",
                        status_code=400,
                        suggestions=["Provide path as ?path=ServerName:channelname"]
                    ),
                    status=400
                )
            
            # Resolve channel path
            channel_id, found = self.name_mapping.find_channel_id(path)
            if not found:
                return web.json_response(
                    APIResponse.error(
                        "CHANNEL_NOT_FOUND",
                        f"Channel path '{path}' not found",
                        status_code=404,
                        suggestions=["Use /guilds-with-channels to see available channels"]
                    ),
                    status=404
                )
            
            # Get channel object
            channel = self.bot.get_channel(int(channel_id))
            if not channel or not isinstance(channel, discord.TextChannel):
                return web.json_response(
                    APIResponse.error(
                        "CHANNEL_INVALID",
                        "Channel not found or not a text channel",
                        status_code=404
                    ),
                    status=404
                )
            
            # Get users from recent history
            recent_users = {}
            try:
                cache = self.cache_manager.get_cache(channel)
                count = 0
                async for message in cache.history(limit=100):
                    if count >= 100:
                        break
                    
                    user_id = str(message.author.id)
                    if user_id not in recent_users:
                        user_name = self.name_mapping.get_user_name(user_id)
                        recent_users[user_id] = {
                            "id": user_id,
                            "name": user_name or message.author.name,
                            "display_name": message.author.display_name,
                            "last_active": message.created_at.isoformat()
                        }
                    count += 1
                
                # Get other channel members
                other_users = []
                for member in channel.members:
                    if str(member.id) not in recent_users:
                        user_name = self.name_mapping.get_user_name(str(member.id))
                        other_users.append({
                            "id": str(member.id),
                            "name": user_name or member.name,
                            "display_name": member.display_name,
                            "status": str(member.status) if hasattr(member, "status") else "unknown"
                        })
                
                response_data = {
                    "path": path,
                    "channel_id": channel_id,
                    "total_users": len(recent_users) + len(other_users),
                    "recently_active": list(recent_users.values()),
                    "other_members": other_users
                }
                
                logger.info(f"[{request_id}] Successfully listed {response_data['total_users']} users")
                
                return web.json_response(
                    APIResponse.success(
                        data=response_data,
                        message=f"Listed {response_data['total_users']} users in {channel.name}"
                    )
                )
                
            except Exception as e:
                logger.error(f"[{request_id}] Error getting user list: {e}")
                return web.json_response(
                    APIResponse.error(
                        "USER_LIST_ERROR",
                        "Failed to get user list",
                        details=str(e),
                        status_code=500
                    ),
                    status=500
                )
            
        except Exception as e:
            logger.error(f"[{request_id}] Unexpected error in get_user_list: {e}")
            logger.error(traceback.format_exc())
            return web.json_response(
                APIResponse.error(
                    "INTERNAL_ERROR",
                    "Internal server error",
                    details=str(e),
                    status_code=500
                ),
                status=500
            )

    async def handle_health_check(self, request: web.Request) -> web.Response:
        """GET /health or /status"""
        try:
            detailed_stats = self.bot.get_detailed_stats()
            
            stats = {
                **detailed_stats,
                "request_count": self.request_count,
                "timestamp": datetime.utcnow().isoformat()
            }
            
            return web.json_response(
                APIResponse.success(data=stats, message="Discord bot is healthy")
            )
            
        except Exception as e:
            logger.error(f"Health check error: {e}")
            return web.json_response(
                APIResponse.error("HEALTH_CHECK_ERROR", "Health check failed", details=str(e), status_code=500),
                status=500
            )

    def _sanitize_reference(self, reference: str) -> str:
        """
        Sanitize reference string by removing surrounding quotes.
        Matches the behavior of the original discord_bot.py find_message_by_reference method.
        """
        reference = reference.strip()
        
        # Remove surrounding quotes if present (same logic as original discord_bot.py)
        if (reference.startswith('"') and reference.endswith('"')) or \
        (reference.startswith("'") and reference.endswith("'")):
            reference = reference[1:-1].strip()
        
        return reference

def create_routes(handlers: DiscordAPIHandlers) -> List[web.RouteDef]:
    """Create all the HTTP routes for the Discord bot API."""
    return [
        web.get("/guilds-with-channels", handlers.handle_list_guilds_with_channels),
        web.post("/send-by-path", handlers.handle_send_by_path),
        web.get("/enhanced-history", handlers.handle_enhanced_history),
        web.get("/find-message", handlers.handle_find_message),
        web.post("/reply-to-message", handlers.handle_reply_to_message),
        web.get("/user-list", handlers.handle_get_user_list),
        web.get("/health", handlers.handle_health_check),
        web.get("/status", handlers.handle_health_check)
    ]