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
from brain.prompting.loader import get_prompt

from config import Config
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
        self.help_text = """###
The Discord environment provides a seamless interface to Discord chat functionality.

Key Commands:
• Discovery:
    - discord list_servers     | View all accessible servers and channels
    - discord list_users "Server:channel"      | List users in a specific channel

• Messages:
    - discord send "Server:channel" "message here"    | Send a message
    - discord history "Server:channel"                | Get channel history
    - discord show "Server:channel" "#3"              | View specific message
    - discord reply "Server:channel" "#2" "response"  | Reply to a message

Note that the numbers for replies may shift references due to the discrete nature of your interactions.
###"""

    def _register_commands(self) -> None:
        """Register all Discord commands with full validation and help information."""
        
        # LIST SERVERS - Primary discovery command
        self.register_command(
            CommandDefinition(
                name="list_servers",
                description="List all Discord servers and channels",
                parameters=[],
                flags=[],
                examples=[
                    "discord list_servers"
                ],
                related_commands=["send", "history"],
                failure_hints={
                    "network": "Unable to reach Discord bot. Is it running?",
                    "auth": "Bot may not have proper permissions"
                },
                category="Discovery"
            )
        )

        # LIST USERS - List users in a specific channel, autonomous interaction capability
        self.register_command(
            CommandDefinition(
                name="list_users",
                description="List all users in a specific Discord channel",
                parameters=[
                    Parameter(
                        name="channel",
                        description="Channel to list users from, in 'Server:channel' format",
                        type=ParameterType.STRING,
                        required=True
                    )
                ],
                flags=[],
                examples=[
                    'discord list_users "ServerName:channel"',
                ],
                related_commands=["list_servers", "send", "history", "show"],
                failure_hints={
                    "invalid_channel": "Channel not found. Use list_servers to see available channels",
                    "permissions": "Bot may not have permission to read this channel"
                },
                category="Discovery"
            )
        )

        # SEND - Primary message creation command
        self.register_command(
            CommandDefinition(
                name="send",
                description="Send a Discord message to a channel",
                parameters=[
                    Parameter(
                        name="channel",
                        description="Where to send the message, in 'Server:channel' format",
                        type=ParameterType.STRING,
                        required=True
                    ),
                    Parameter(
                        name="message",
                        description="The message to send (use quotes for spaces)",
                        type=ParameterType.STRING,
                        required=True
                    )
                ],
                flags=[],
                examples=[
                    'discord send "ServerName:channel" "message here..."',
                ],
                related_commands=["list_servers", "history", "reply"],
                failure_hints={
                    "invalid_channel": "Channel not found. Use list_servers to see available channels",
                    "permissions": "Bot may not have permission to send messages to this channel",
                    "content_empty": "Message content cannot be empty"
                },
                category="Messages"
            )
        )

        # HISTORY - View channel conversation
        self.register_command(
            CommandDefinition(
                name="history",
                description="Retrieve recent message history from a Discord channel",
                parameters=[
                    Parameter(
                        name="channel",
                        description="Channel to fetch messages from, in 'Server:channel' format",
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
                    'discord history "MyServer:general"',
                    'discord history "MyServer:announcements" --limit=20'
                ],
                related_commands=["list_servers", "send", "show", "reply"],
                failure_hints={
                    "invalid_channel": "Channel not found. Use list_servers to see available channels",
                    "permissions": "Bot may not have permission to read this channel"
                },
                category="Messages"
            )
        )

        # SHOW - Find and display specific messages
        self.register_command(
            CommandDefinition(
                name="show",
                description="Find and display a specific message from a Discord channel",
                parameters=[
                    Parameter(
                        name="channel",
                        description="Channel to find message in, in 'Server:channel' format",
                        type=ParameterType.STRING,
                        required=True
                    ),
                    Parameter(
                        name="reference",
                        description="Reference to the message (see examples)",
                        type=ParameterType.STRING,
                        required=False,
                        default="latest"
                    )
                ],
                flags=[],
                examples=[
                    'discord show "MyServer:general"',  # Shows latest message
                    'discord show "MyServer:general" "#3"',  # Shows message #3 from history
                    'discord show "MyServer:general" "latest-from:username"',  # Latest from user
                    'discord show "MyServer:general" "contains:keyword"'  # Message containing text
                ],
                related_commands=["history", "reply"],
                failure_hints={
                    "invalid_channel": "Channel not found. Use list_servers to see available channels",
                    "invalid_reference": "Message reference not found or invalid"
                },
                category="Messages"
            )
        )

        # REPLY - Respond to specific messages
        self.register_command(
            CommandDefinition(
                name="reply",
                description="Reply to a specific message in a Discord channel",
                parameters=[
                    Parameter(
                        name="channel",
                        description="Channel containing the message, in 'Server:channel' format",
                        type=ParameterType.STRING,
                        required=True
                    ),
                    Parameter(
                        name="reference",
                        description="Reference to the message to reply to",
                        type=ParameterType.STRING,
                        required=True
                    ),
                    Parameter(
                        name="content",
                        description="Reply content (use quotes for spaces)",
                        type=ParameterType.STRING,
                        required=True
                    )
                ],
                flags=[],
                examples=[
                    'discord reply "MyServer:general" "#3" "Replying to the third message"',
                    'discord reply "MyServer:announcements" "contains:event" "Will this event be recorded?"'
                ],
                related_commands=["show", "send", "history"],
                failure_hints={
                    "invalid_channel": "Channel not found. Use list_servers to see available channels",
                    "invalid_reference": "Message reference not found or invalid",
                    "content_empty": "Reply content cannot be empty"
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
        """Execute a Discord command."""
        try:
            # Process primary commands first
            if action == "list_servers":
                return await self._list_servers()
            elif action == "list_users":
                return await self._list_users(params)
            elif action == "send":
                return await self._send(params)
            elif action == "history":
                return await self._history(params, flags)
            elif action == "show":
                return await self._show(params)
            elif action == "reply":
                return await self._reply(params)
            else:
                error = CommandValidationError(
                    message=f"Unknown action: {action}",
                    suggested_fixes=["Check command spelling", "Use 'discord help' to see available commands"],
                    related_commands=["discord help", "discord list_servers"],
                    examples=["discord list_servers"]
                )
                return CommandResult(
                    success=False,
                    message=f"Unknown command '{action}'",
                    suggested_commands=["discord help", "discord list_servers"],
                    error=error
                )
        except CommandValidationError as cve:
            return CommandResult(
                success=False,
                message=f"Validation error: {str(cve)}",
                suggested_commands=["discord help"],
                error=cve,
                data={"error_type": "validation", "error_details": str(cve)}
            )
        except Exception as e:
            error = CommandValidationError(
                message=str(e),
                suggested_fixes=["Try a simpler command", "Check command parameters"],
                related_commands=["discord help", "discord list_servers"],
                examples=["discord list_servers"]
            )
            return CommandResult(
                success=False,
                message=f"Unexpected error: {e}",
                suggested_commands=["discord help", "discord list_servers"],
                error=error,
                data={"error_type": "unexpected", "error_details": str(e)}
            )

    async def _list_servers(self) -> CommandResult:
        """List servers and their channels in a user-friendly format."""
        data, status_code = await self.discord_service.list_guilds_with_channels()
        if status_code != 200 or data is None:
            error = CommandValidationError(
                message=f"Failed to list Discord servers: HTTP {status_code}",
                suggested_fixes=["Check bot permissions", "Verify bot is running"],
                related_commands=["discord help"],
                examples=["discord list_servers"]
            )
            return CommandResult(
                success=False,
                message=f"Failed to list Discord servers: {status_code}",
                suggested_commands=["discord help"],
                error=error,
                data={"status_code": status_code}
            )

        lines = ["Available Discord Servers and Channels:", "###"]
        
        # Build list of suggested commands for common channels
        suggested_commands = []
        
        for guild in data:
            guild_name = guild.get('name', 'Unnamed Server')
            lines.append(f"• {guild_name}")
            
            # Sort channels alphabetically
            channels = sorted(guild.get('channels', []), key=lambda c: c.get('name', ''))
            
            for channel in channels:
                channel_name = channel.get('name', 'unnamed')
                path = channel.get('path', f"{guild_name}:{channel_name}")
                
                lines.append(f"  - #{channel_name} (use as \"{path}\")")
                
                # Add first channel from each server to suggested commands
                if len(suggested_commands) < 5 and channel == channels[0]:
                    suggested_commands.append(f'discord send "{path}" "Hello there!"')
                    suggested_commands.append(f'discord history "{path}"')
        
        lines.append("###")
        lines.append("Use these server:channel names with the following commands:")
        lines.append('- discord send "ServerName:channel-name" "Your message here"')
        lines.append('- discord history "ServerName:channel-name"')
        lines.append('- discord show "ServerName:channel-name"')

        return CommandResult(
            success=True,
            message="\n".join(lines),
            data=data,
            suggested_commands=suggested_commands[:5]  # Limit to 5 suggestions
        )
    
    async def _list_users(self, params: List[str]) -> CommandResult:
        """List users in a specific channel."""
        if len(params) < 1:
            error = CommandValidationError(
                message="Channel path required",
                suggested_fixes=["Provide a channel path in 'Server:channel' format", 
                                "Use list_servers to see available channels"],
                related_commands=["discord list_servers"],
                examples=['discord list_users "ServerName:channel"']
            )
            return CommandResult(
                success=False,
                message="Channel path is required",
                suggested_commands=["discord list_servers"],
                error=error
            )

        # Get and clean the path
        raw_path = params[0]
        path = raw_path.strip()
        if (path.startswith('"') and path.endswith('"')) or (path.startswith("'") and path.endswith("'")):
            path = path[1:-1]
        
        data, status_code = await self.discord_service.get_user_list(path)
        
        if status_code != 200 or data is None:
            error = CommandValidationError(
                message=f"Failed to list users: HTTP {status_code}",
                suggested_fixes=["Check channel path spelling", "Use list_servers to see available channels"],
                related_commands=["discord list_servers"],
                examples=['discord list_users "ServerName:channel"']
            )
            return CommandResult(
                success=False,
                message=f"Failed to list users in {path}: {status_code}",
                suggested_commands=["discord list_servers"],
                error=error,
                data={"status_code": status_code, "path": path}
            )

        # Extract users from data
        recent_users = [f"{user['display_name']}" for user in data.get('recently_active', [])]
        other_users = [f"{user['display_name']}" for user in data.get('other_members', [])]

        num_recent = len(recent_users)
        num_other = len(other_users)
        recent_list_str = ", ".join(recent_users) if recent_users else "None"
        other_list_str = ", ".join(other_users) if other_users else "None"
        example_user = recent_users[0] if recent_users else other_users[0] if other_users else "No users found"

        response = get_prompt("interfaces.exo.environments.discord.commands.list_users",
            model=Config.get_cognitive_model(),
            vars={
                "num_recent": num_recent,
                "num_other": num_other,
                "recent_list": recent_list_str,
                "other_list": other_list_str,
                "example_user": example_user,
                "channel": path
            }
        )
        
        return CommandResult(
            success=True,
            message=response,
            data=data,
            suggested_commands=[
                f'discord send "{path}" "Hello @{example_user}!"',
                f'discord history "{path}"',
                f'discord show "{path}" "latest-from:{example_user}"',
            ]
        )
    
    async def _send(self, params: List[str]) -> CommandResult:
        """Send a message using the friendly path format."""
        if len(params) < 2:
            error = CommandValidationError(
                message="Channel path and message content required", 
                suggested_fixes=["Provide both channel path and message content", 
                            "Use quotes around message content"],
                related_commands=["discord list_servers"],
                examples=['discord send "Server:channel" "Hello world!"'] 
            )
            return CommandResult(
                success=False,
                message="Both channel path and message content are required",
                suggested_commands=["discord list_servers"],
                error=error
            )

        # Get the raw path and content, ensuring proper cleanup
        raw_path, content = params
        
        # Clean up the path - strip any surrounding quotes that might have been included
        path = raw_path.strip()
        if (path.startswith('"') and path.endswith('"')) or (path.startswith("'") and path.endswith("'")):
            path = path[1:-1]
        
        # Try sending the message
        data, status_code = await self.discord_service.send_message_by_path(path, content)

        if status_code != 200 or data is None:
            # Enhanced error message with path details
            error_detail = f"Failed to send to '{path}'"
            if isinstance(data, dict) and "error" in data:
                error_detail += f": {data['error']}"
            
            error = CommandValidationError(
                message=f"Failed to send message: HTTP {status_code}",
                suggested_fixes=[
                    "Check channel path spelling", 
                    "Use list_servers to see available channels",
                    f"Tried to send to: '{path}'"
                ],
                related_commands=["discord list_servers"],
                examples=['discord send "Server:general" "Hello world!"']
            )
            return CommandResult(
                success=False,
                message=f"Failed to send message to {path}: {status_code}",
                suggested_commands=["discord list_servers"],
                error=error,
                data={"status_code": status_code, "path": path, "error": data if isinstance(data, dict) else None}
            )

        lines = ["Message Sent Successfully:", "###"]
        lines.append(f"To: {path}")
        preview = content[:100] + ('...' if len(content) > 100 else '')
        lines.append("Content:")
        lines.append(preview)
        lines.append("###")
        lines.append("Available Actions:")
        lines.append("• View channel history")
        lines.append("• Send another message")

        return CommandResult(
            success=True,
            message="\n".join(lines),
            data=data,
            suggested_commands=[
                f'discord history "{path}"',
                f'discord send "{path}" "Another message..."'
            ]
        )

    async def _history(self, params: List[str], flags: Dict[str, Any]) -> CommandResult:
        """Get message history using friendly path format."""
        if len(params) < 1:
            error = CommandValidationError(
                message="Channel path required",
                suggested_fixes=["Provide a channel path in 'Server:channel' format", 
                                "Use list_servers to see available channels"],
                related_commands=["discord list_servers"],
                examples=['discord history "Server:general"']
            )
            return CommandResult(
                success=False,
                message="Channel path is required",
                suggested_commands=["discord list_servers"],
                error=error
            )

        # Get and clean the path
        raw_path = params[0]
        path = raw_path.strip()
        if (path.startswith('"') and path.endswith('"')) or (path.startswith("'") and path.endswith("'")):
            path = path[1:-1]
        
        limit = flags.get("limit", 50)

        data, status_code = await self.discord_service.get_enhanced_history(path, limit)
        
        if status_code != 200 or data is None:
            # Enhanced error message with path details
            error_detail = f"Failed to get history from '{path}'"
            if isinstance(data, dict) and "error" in data:
                error_detail += f": {data['error']}"
                
            error = CommandValidationError(
                message=f"Failed to get history: HTTP {status_code}",
                suggested_fixes=[
                    "Check channel path spelling", 
                    "Verify channel access", 
                    "Use list_servers to see available channels"
                ],
                related_commands=["discord list_servers"],
                examples=['discord history "Server:general"']
            )
            return CommandResult(
                success=False,
                message=f"Failed to get channel history for {path}: {status_code}",
                suggested_commands=["discord list_servers"],
                error=error,
                data={"status_code": status_code, "path": path, "error": data if isinstance(data, dict) else None}
            )
            
        # Format messages for LLM consumption
        lines = ["Discord Channel History:"]
        lines.append("###")
        lines.append(f"Channel: {path}")
        
        # Format messages (adjusted to match enhanced_history response format)
        messages = data.get("messages", [])
        if not messages:
            lines.append("No messages found in channel history.")
        else:
            for msg in messages:
                try:
                    ref = msg.get('reference', '')
                    timestamp = msg.get('timestamp', '')[:16]  # YYYY-MM-DD HH:MM
                    author = msg.get('author', 'Unknown')
                    content = msg.get('content', '').replace('\n', ' ')
                    if len(content) > 500:  # Truncate very long messages
                        content = content[:500] + "..."
                    
                    # Include reference ID for easy referencing in commands
                    lines.append(f"[{timestamp}] {ref} {author}: {content}")
                except Exception as e:
                    lines.append(f"[Error formatting message: {e}]")

        # Add summary footer
        lines.append("###")
        status = f"Showing {len(messages)} messages"
        lines.append(status)
        lines.append("You can reference messages by their number (e.g., \"#1\") in commands:")
        lines.append('Example: discord show "' + path + '" "#1"')
        lines.append('Example: discord reply "' + path + '" "#2" "Your reply"')
        
        return CommandResult(
            success=True,
            message="\n".join(lines),
            data=data,
            suggested_commands=[
                f'discord send "{path}" "New message..."',
                f'discord reply "{path}" "#1" "Reply to the first message..."',
                f'discord history "{path}" --limit={min(100, limit + 10)}'
            ]
        )

    async def _show(self, params: List[str]) -> CommandResult:
        """Find and display a specific message."""
        if len(params) < 1:
            error = CommandValidationError(
                message="Channel path is required",
                suggested_fixes=["Provide a channel path in 'Server:channel' format"],
                related_commands=["discord list_servers"],
                examples=['discord show "MyServer:general"']
            )
            return CommandResult(
                success=False,
                message="Channel path is required",
                suggested_commands=["discord list_servers"],
                error=error
            )
        
        # Get and clean the path (same pattern as other methods)
        raw_path = params[0]
        path = raw_path.strip()
        if (path.startswith('"') and path.endswith('"')) or (path.startswith("'") and path.endswith("'")):
            path = path[1:-1]
        
        reference = params[1] if len(params) > 1 else "latest"
        
        data, status_code = await self.discord_service.find_message(path, reference)
        
        if status_code != 200 or data is None:
            error_msg = "Failed to find message"
            if isinstance(data, dict) and "error" in data:
                error_msg += f": {data['error']}"
                
            error = CommandValidationError(
                message=f"{error_msg}: HTTP {status_code}",
                suggested_fixes=[
                    "Check the channel path and message reference",
                    "Try a different reference format",
                    "Ensure the message exists"
                ],
                related_commands=["discord history", "discord list_servers"],
                examples=[
                    'discord show "MyServer:general"',
                    'discord show "MyServer:general" "#1"'
                ]
            )
            return CommandResult(
                success=False,
                message=f"Failed to find message: {error_msg}",
                suggested_commands=["discord history", "discord list_servers"],
                error=error
            )
        
        # Format message data for display
        lines = ["Discord Message:", "###"]
        lines.append(f"Channel: {path}")
        lines.append(f"Author: {data.get('author', 'Unknown')}")
        
        # Format timestamp nicely if available
        timestamp = data.get('timestamp', '')
        if timestamp:
            try:
                date_part = timestamp.split('T')[0]
                time_part = timestamp.split('T')[1][:8]
                lines.append(f"Time: {date_part} {time_part}")
            except IndexError:
                lines.append(f"Time: {timestamp}")
        
        lines.append("Content:")
        content = data.get('content', 'No content')
        lines.append(content)
        
        lines.append("###")
        lines.append("Available Actions:")
        lines.append(f'• Reply: discord reply "{path}" "{reference}" "Your reply"')
        lines.append(f'• View more: discord history "{path}"')
        
        return CommandResult(
            success=True,
            message="\n".join(lines),
            data=data,
            suggested_commands=[
                f'discord reply "{path}" "{reference}" "Your reply here"',
                f'discord history "{path}"'
            ]
        )

    async def _reply(self, params: List[str]) -> CommandResult:
        """Reply to a specific message."""
        if len(params) < 3:
            error = CommandValidationError(
                message="Channel path, message reference, and content are required",
                suggested_fixes=["Provide all required parameters"],
                related_commands=["discord show", "discord history"],
                examples=['discord reply "MyServer:general" "latest" "Your reply here"']
            )
            return CommandResult(
                success=False,
                message="Channel path, message reference, and content are all required",
                suggested_commands=["discord show", "discord history"],
                error=error
            )
        
        # Get and clean the path and reference
        raw_path = params[0]
        raw_reference = params[1]
        content = params[2]
        
        # Clean up path
        path = raw_path.strip()
        if (path.startswith('"') and path.endswith('"')) or (path.startswith("'") and path.endswith("'")):
            path = path[1:-1]
        
        reference = raw_reference.strip()
        if (reference.startswith('"') and reference.endswith('"')) or (reference.startswith("'") and reference.endswith("'")):
            reference = reference[1:-1]
        
        data, status_code = await self.discord_service.reply_to_message(path, reference, content)
        
        if status_code != 200 or data is None:
            error_msg = "Failed to send reply"
            if isinstance(data, dict) and "error" in data:
                error_msg += f": {data['error']}"
                
            error = CommandValidationError(
                message=f"{error_msg}: HTTP {status_code}",
                suggested_fixes=[
                    "Check the channel path and message reference",
                    "Make sure the message to reply to exists",
                    "Verify that the bot has permission to send messages"
                ],
                related_commands=["discord show", "discord history"],
                examples=['discord reply "MyServer:general" "#3" "Your reply here"']
            )
            return CommandResult(
                success=False,
                message=f"Failed to send reply: {error_msg}",
                suggested_commands=["discord show", "discord history"],
                error=error
            )
        
        # Format success message
        lines = ["Reply Sent Successfully:", "###"]
        lines.append(f"Channel: {path}")
        
        # Show what we replied to
        if "replied_to" in data:
            replied_to = data["replied_to"]
            lines.append(f"Replied to: {replied_to.get('author', 'Unknown')}")
            original_content = replied_to.get('content', '')
            if len(original_content) > 100:
                original_content = original_content[:97] + "..."
            lines.append(f"Original: {original_content}")
        
        # Show our reply
        lines.append("Your reply:")
        lines.append(content)
        
        lines.append("###")
        lines.append("Available Actions:")
        lines.append(f'• View history: discord history "{path}"')
        lines.append(f'• Send another message: discord send "{path}" "Message"')
        
        return CommandResult(
            success=True,
            message="\n".join(lines),
            data=data,
            suggested_commands=[
                f'discord history "{path}"',
                f'discord show "{path}" "latest"'
            ]
        )