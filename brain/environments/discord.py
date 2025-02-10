from typing import Dict, Any, List

from .base_environment import BaseEnvironment
from brain.commands.model import (
    CommandDefinition,
    Parameter,
    Flag,
    ParameterType,
    CommandResult,
    CommandValidationError
)

from core.discord_service import DiscordService

class DiscordEnvironment(BaseEnvironment):
    """
    Environment for communicating with the local Discord bot server.

    Supports listing guilds, listing channels, sending messages,
    and retrieving specific messages or channel histories.
    """

    def __init__(self, discord_service: DiscordService):
        super().__init__()
        self.discord_service = discord_service
        self.help_text = """
        The discord environment lets you communicate with Discord.
        You can list guilds, list channels, read messages, or post messages.
        Example usage:
        - discord list_guilds
        - discord list_channels <guild_id>
        - discord get_message <channel_id> <message_id>
        - discord get_history <channel_id> --limit=10
        - discord send_message <channel_id> "<content>"
        """

    def _register_commands(self) -> None:
        """Register all Discord commands with full validation and help information."""
        
        # List Guilds - Show available Discord servers
        self.register_command(
            CommandDefinition(
                name="list_guilds",
                description="List all Discord servers (guilds) the bot has access to",
                parameters=[],
                flags=[],
                examples=[
                    "discord list_guilds"
                ],
                related_commands=["list_channels", "get_history"],
                failure_hints={
                    "network": "Unable to reach Discord bot. Is it running?",
                    "auth": "Bot may not have proper permissions"
                },
                category="Discovery"
            )
        )

        # List Channels - Show channels in a guild
        self.register_command(
            CommandDefinition(
                name="list_channels",
                description="List all channels in a specified Discord server",
                parameters=[
                    Parameter(
                        name="guild_id",
                        description="The Discord server ID to list channels from",
                        type=ParameterType.STRING,
                        required=True
                    )
                ],
                flags=[],
                examples=[
                    "discord list_channels <guild_id>"
                ],
                related_commands=["list_guilds", "get_history", "send_message"],
                failure_hints={
                    "invalid_guild": "Server ID not found. Use list_guilds to see available servers",
                    "permissions": "Bot may not have access to this server"
                },
                category="Discovery"
            )
        )

        # Get Message History
        self.register_command(
            CommandDefinition(
                name="get_history",
                description="Retrieve recent message history from a Discord channel",
                parameters=[
                    Parameter(
                        name="channel_id",
                        description="The Discord channel ID to fetch messages from",
                        type=ParameterType.STRING,
                        required=True
                    )
                ],
                flags=[
                    Flag(
                        name="limit",
                        description="Maximum number of messages to retrieve (1-100)",
                        type=ParameterType.INTEGER,
                        required=False,
                        default=50
                    )
                ],
                examples=[
                    "discord get_history <channel_id>",
                    "discord get_history <channel_id> --limit=<number>"
                ],
                related_commands=["list_channels", "get_message"],
                failure_hints={
                    "invalid_channel": "Channel not found. Use list_channels to see available channels",
                    "limit_exceeded": "Limit must be between 1 and 100 messages",
                    "permissions": "Bot may not have permission to read this channel"
                },
                category="Messages"
            )
        )

        # Get Specific Message
        self.register_command(
            CommandDefinition(
                name="get_message",
                description="Retrieve a specific Discord message by its ID",
                parameters=[
                    Parameter(
                        name="channel_id",
                        description="The Discord channel ID containing the message",
                        type=ParameterType.STRING,
                        required=True
                    ),
                    Parameter(
                        name="message_id",
                        description="The specific Discord message ID to retrieve",
                        type=ParameterType.STRING,
                        required=True
                    )
                ],
                flags=[],
                examples=[
                    "discord get_message <channel_id> <message_id>"
                ],
                related_commands=["get_history", "send_message"],
                failure_hints={
                    "invalid_channel": "Channel not found. Use list_channels to see available channels",
                    "invalid_message": "Message not found or has been deleted",
                    "permissions": "Bot may not have permission to read this message"
                },
                category="Messages"
            )
        )

        # Send Message
        self.register_command(
            CommandDefinition(
                name="send_message",
                description="Send a new message to a Discord channel",
                parameters=[
                    Parameter(
                        name="channel_id",
                        description="The Discord channel ID to send the message to",
                        type=ParameterType.STRING,
                        required=True
                    ),
                    Parameter(
                        name="content",
                        description="The message text to send (use quotes for spaces)",
                        type=ParameterType.STRING,
                        required=True
                    )
                ],
                flags=[],
                examples=[
                    'discord send_message <channel_id> "<message>"'
                ],
                related_commands=["list_channels", "get_history"],
                failure_hints={
                    "invalid_channel": "Channel not found. Use list_channels to see available channels",
                    "permissions": "Bot may not have permission to send messages to this channel",
                    "content_empty": "Message content cannot be empty",
                    "content_invalid": "Message content must be enclosed in quotes if it contains spaces"
                },
                category="Messages"
            )
        )

    async def _execute_command(
        self,
        action: str,
        params: List[str],
        flags: Dict[str, Any],
        context: Dict[str, Any]
    ) -> CommandResult:
        try:
            if action == "list_guilds":
                return await self._list_guilds()
            elif action == "list_channels":
                return await self._list_channels(params)
            elif action == "get_message":
                return await self._get_message(params)
            elif action == "get_history":
                return await self._get_history(params, flags)
            elif action == "send_message":
                return await self._send_message(params)
            else:
                error = CommandValidationError(
                    message=f"Unknown action: {action}",
                    suggested_fixes=["Check command spelling", "Use help to see available commands"],
                    related_commands=["discord help", "discord list_guilds"],
                    examples=["discord list_guilds", "discord list_channels <guild_id>"]
                )
                return CommandResult(
                    success=False,
                    message=f"Unknown command '{action}'",
                    suggested_commands=["discord help", "discord list_guilds"],
                    error=error
                )
        except CommandValidationError as cve:
            return CommandResult(
                success=False,
                message=f"Validation error: {str(cve)}",
                suggested_commands=["discord help", "help"],
                error=cve,
                data={"error_type": "validation", "error_details": str(cve)}
            )
        except Exception as e:
            error = CommandValidationError(
                message=str(e),
                suggested_fixes=["Try a simpler command", "Check command parameters"],
                related_commands=["discord help", "discord list_guilds"],
                examples=["discord list_guilds"]
            )
            return CommandResult(
                success=False,
                message=f"Unexpected error: {e}",
                suggested_commands=["discord help", "discord list_guilds"],
                error=error,
                data={"error_type": "unexpected", "error_details": str(e)}
            )

    async def _list_guilds(self) -> CommandResult:
        data, status_code = await self.discord_service.list_guilds()
        if status_code != 200 or data is None:
            error = CommandValidationError(
                message=f"Failed to list guilds: HTTP {status_code}",
                suggested_fixes=["Check bot permissions", "Verify bot is running"],
                related_commands=["discord help"],
                examples=["discord list_guilds"]
            )
            return CommandResult(
                success=False,
                message=f"Failed to list guilds: {status_code}",
                suggested_commands=["discord help"],
                error=error,
                data={"status_code": status_code}
            )

        lines = ["Available Discord Servers:", "---"]
        for guild in data:
            lines.append(f"• {guild.get('name', 'Unnamed')} (ID: {guild['id']})")
            if 'member_count' in guild:
                lines.append(f"  Members: {guild['member_count']}")
        lines.append("---")
        lines.append("Use 'discord list_channels <guild_id>' to see available channels")

        return CommandResult(
            success=True,
            message="\n".join(lines),
            data=data,
            suggested_commands=[
                f"discord list_channels {guild['id']}" for guild in data[:3]
            ]
        )

    async def _list_channels(self, params: List[str]) -> CommandResult:
        if len(params) < 1:
            error = CommandValidationError(
                message="Guild ID required", 
                suggested_fixes=["Provide a guild ID", "Use list_guilds to find guild IDs"],
                related_commands=["discord list_guilds"],
                examples=["discord list_channels 123456789"]
            )
            return CommandResult(
                success=False,
                message="Guild ID is required",
                suggested_commands=["discord list_guilds"],
                error=error
            )

        guild_id = params[0]
        data, status_code = await self.discord_service.list_channels(guild_id)
        if status_code != 200 or data is None:
            error = CommandValidationError(
                message=f"Failed to list channels: HTTP {status_code}",
                suggested_fixes=["Check guild ID", "Verify bot permissions"],
                related_commands=["discord list_guilds"],
                examples=["discord list_channels <valid_guild_id>"]
            )
            return CommandResult(
                success=False,
                message=f"Failed to list channels: {status_code}",
                suggested_commands=["discord list_guilds"],
                error=error,
                data={"status_code": status_code, "guild_id": guild_id}
            )

        # Format the message for direct LLM consumption
        lines = ["Available Discord Channels:"]
        lines.append("---")
        for channel in data:
            # Basic channel info
            channel_type = channel.get('type', 'Unknown')
            lines.append(f"• {channel.get('name', 'Unnamed')} (ID: {channel['id']})")
            # Add channel details if available
            if channel.get('topic'):
                lines.append(f"  Topic: {channel['topic']}")
            if channel.get('nsfw'):
                lines.append("  [NSFW]")
            # Add channel type
            lines.append(f"  Type: {channel_type}")
        lines.append("---")
        lines.append("Use 'discord get_history <channel_id>' to view message history")

        return CommandResult(
            success=True,
            message="\n".join(lines),
            data=data,
            suggested_commands=[
                f"discord get_history {channel['id']}"
                for channel in data[:3]  # Suggest first 3 channels
            ]
        )

    async def _get_message(self, params: List[str]) -> CommandResult:
        if len(params) < 2:
            error = CommandValidationError(
                message="Channel ID and Message ID required",
                suggested_fixes=["Provide both channel and message IDs"],
                related_commands=["discord list_channels", "discord get_history"],
                examples=["discord get_message <channel_id> <message_id>"]
            )
            return CommandResult(
                success=False,
                message="Both channel ID and message ID are required",
                suggested_commands=["discord list_channels", "discord get_history"],
                error=error
            )

        channel_id, message_id = params
        channel_id, message_id = params
        data, status_code = await self.discord_service.get_message(channel_id, message_id)
        if status_code != 200 or data is None:
            error = CommandValidationError(
                message=f"Failed to get message: HTTP {status_code}",
                suggested_fixes=["Check message ID", "Verify channel access"],
                related_commands=["discord get_history"],
                examples=["discord get_history <channel_id>"]
            )
            return CommandResult(
                success=False,
                message=f"Failed to get message: {status_code}",
                suggested_commands=["discord get_history <channel_id>"],
                error=error,
                data={"status_code": status_code, "channel_id": channel_id, "message_id": message_id}
            )

        # Format message data for LLM consumption
        lines = ["Discord Message:"]
        lines.append("---")
        lines.append(f"Author: {data.get('author', 'Unknown')}")
        lines.append(f"Channel ID: {channel_id}")
        lines.append(f"Sent at: {data.get('timestamp', 'Unknown')}")
        lines.append("Content:")
        lines.append(data.get('content', 'No content'))
        
        lines.append("---")
        lines.append("Available Actions:")
        lines.append("• Get channel history")
        lines.append("• Send a reply")

        return CommandResult(
            success=True,
            message="\n".join(lines),
            data=data,
            suggested_commands=[
                f"discord get_history {channel_id}",
                f"discord send_message {channel_id} \"@{data.get('author', 'Unknown')} ...\""
            ]
        )

    async def _get_history(self, params: List[str], flags: Dict[str, Any]) -> CommandResult:
        if len(params) < 1:
            error = CommandValidationError(
                message="Channel ID required",
                suggested_fixes=["Provide a channel ID", "Use list_channels to find channel IDs"],
                related_commands=["discord list_channels"],
                examples=["discord get_history <channel_id>"]
            )
            return CommandResult(
                success=False,
                message="Channel ID is required",
                suggested_commands=["discord list_channels"],
                error=error
            )

        channel_id = params[0]
        limit = flags.get("limit", 50)
        data, status_code = await self.discord_service.get_history(channel_id, limit)
        if status_code != 200 or data is None:
            error = CommandValidationError(
                message=f"Failed to get history: HTTP {status_code}",
                suggested_fixes=["Check channel ID", "Verify channel access"],
                related_commands=["discord list_channels"],
                examples=["discord get_history <valid_channel_id>"]
            )
            return CommandResult(
                success=False,
                message=f"Failed to get channel history: {status_code}",
                suggested_commands=["discord list_channels"],
                error=error,
                data={"status_code": status_code, "channel_id": channel_id}
            )
            
        # Constants for message formatting
        MAX_MSG_LENGTH = 500  # Characters per message
        MAX_TOTAL_TOKENS = 2000  # Reserve room for other content
        MAX_TOTAL_CHARS = MAX_TOTAL_TOKENS * 4  

        # Format messages for LLM consumption
        lines = ["Discord Channel History:"]
        lines.append("---")
        
        # Pre-format all messages
        formatted_messages = []
        for msg in data:
            try:
                timestamp = msg.get('timestamp', '')[:16]  # YYYY-MM-DD HH:MM
            except Exception:
                timestamp = 'Unknown time'
            
            author = msg.get('author', 'Unknown')
            content = msg.get('content', '').replace('\n', ' ')
            if len(content) > MAX_MSG_LENGTH:
                content = content[:MAX_MSG_LENGTH-3] + "..."
            
            msg_line = f"{msg['id']} {author} [{timestamp}]: {content}"
            formatted_messages.append(msg_line)

        # Start from the most recent messages and work backwards
        total_chars = len("\n".join(lines))
        message_count = 0
        truncated = False

        # Process messages from newest to oldest, but will display oldest to newest
        kept_messages = []
        for msg_line in reversed(formatted_messages):
            if total_chars + len(msg_line) + 2 > MAX_TOTAL_CHARS:
                truncated = True
                break
            
            kept_messages.append(msg_line)
            total_chars += len(msg_line) + 1
            message_count += 1

        # Add messages in chronological order (oldest first)
        lines.extend(reversed(kept_messages))

        # Add summary footer
        lines.append("---")
        status = f"Showing {message_count} of {len(data)} messages"
        if truncated:
            status += " (older messages truncated to conserve tokens)"
        lines.append(status)
        lines.append(f"Channel ID: {channel_id}")
        
        return CommandResult(
            success=True,
            message="\n".join(lines),
            data=data,
            suggested_commands=[
                f"discord send_message {channel_id} \"Reply...\"",
                f"discord get_history {channel_id} --limit={min(100, limit + 10)}"
            ]
        )

    async def _send_message(self, params: List[str]) -> CommandResult:
        if len(params) < 2:
            error = CommandValidationError(
                message="Channel ID and message content required", 
                suggested_fixes=["Provide channel ID and message content", "Use quotes around message content"],
                related_commands=["discord list_channels"],
                examples=['discord send_message <channel_id> "Hello\nworld"'] 
            )
            return CommandResult(
                success=False,
                message="Channel ID and message content are required",
                suggested_commands=["discord list_channels"],
                error=error
            )

        channel_id, content = params
        data, status_code = await self.discord_service.send_message_immediate(channel_id, content)

        if status_code != 200 or data is None:
            error = CommandValidationError(
                message=f"Failed to send message: HTTP {status_code}",
                suggested_fixes=["Check channel ID", "Verify bot permissions"],
                related_commands=["discord list_channels"],
                examples=['discord send_message <valid_channel_id> "Hello\nworld"']  # Added multiline example
            )
            return CommandResult(
                success=False,
                message=f"Failed to send message: {status_code}",
                suggested_commands=["discord list_channels"],
                error=error,
                data={"status_code": status_code, "channel_id": channel_id}
            )

        lines = ["Message Sent Successfully:", "---"]
        lines.append(f"Channel ID: {channel_id}")
        preview = content[:100] + ('...' if len(content) > 100 else '')
        lines.append("Content:")
        lines.append(preview)  # Changed to preserve newlines in preview
        if 'message_id' in data:
            lines.append(f"Message ID: {data['message_id']}")
        lines.append("---")
        lines.append("Available Actions:")
        lines.append("• View channel history")
        lines.append("• Send another message")

        return CommandResult(
            success=True,
            message="\n".join(lines),
            data=data,
            suggested_commands=[
                f"discord get_history {channel_id}",
                f"discord send_message {channel_id} \"Another message...\""
            ]
        )
