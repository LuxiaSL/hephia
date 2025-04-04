import math
import os
import random
import sys
import logging
import asyncio
from typing import Optional
import aiohttp
import discord
from aiohttp import web
from dotenv import load_dotenv
from datetime import datetime

###############################################################################
# LOGGING SETUP
###############################################################################

# Create the logs directory if it doesn't exist
LOG_DIR = "./data/logs/discord"
os.makedirs(LOG_DIR, exist_ok=True)

# Create a timestamped log file
timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
log_filename = os.path.join(LOG_DIR, f"discord_{timestamp_str}.log")

logger = logging.getLogger("DiscordBotLogger")
logger.setLevel(logging.DEBUG)  # Adjust level as needed (DEBUG, INFO, WARNING, etc.)

# File handler
file_handler = logging.FileHandler(log_filename)
file_handler.setLevel(logging.DEBUG)
file_format = logging.Formatter(
    "[%(asctime)s] %(levelname)s in %(module)s:%(lineno)d: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
file_handler.setFormatter(file_format)
logger.addHandler(file_handler)

# Console handler (optional)
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)
console_format = logging.Formatter(
    "%(asctime)s [%(levelname)s] %(message)s", 
    datefmt="%H:%M:%S"
)
console_handler.setFormatter(console_format)
logger.addHandler(console_handler)

###############################################################################
# CONFIGURATION
###############################################################################

load_dotenv()
DISCORD_TOKEN = os.getenv("DISCORD_BOT_TOKEN")  # Your bot's token
HEPHIA_SERVER_URL = "http://localhost:5517"     # Where Hephia server listens
BOT_HTTP_PORT = 5518                            # Port where *this* bot listens

# For convenience, define a global reference to the bot
bot: "RealTimeDiscordBot" = None
persistent_session: aiohttp.ClientSession = None  # We'll reuse this session for all outbound requests

###############################################################################
# REAL-TIME DISCORD BOT (CLIENT)
###############################################################################

class RealTimeDiscordBot(discord.Client):
    """
    Discord client that connects via WebSocket. Receives events in real-time.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.new_messages = {}
        self.guild_name_to_id = {}  # Maps guild names to IDs
        self.channel_path_to_id = {}  # Maps "guild:channel" paths to channel IDs
        self.id_to_guild_name = {}  # Maps guild IDs to names
        self.id_to_channel_name = {}  # Maps channel IDs to names (including guild context)
        # We'll store the persistent aiohttp session reference
        self.session = kwargs.get("session", None)

    async def on_ready(self):
        logger.info(f"[Bot] Logged in as {self.user} (id={self.user.id})")
        await self.update_name_mappings()
        await initialize_message_cache(self)

    async def on_message(self, message: discord.Message):
        """
        Called whenever a new message is created in Discord.
        Tracks message counts per channel and notifies when thresholds are reached.
        Also handles random engagement with sigmoid scaling.
        """
        channel_id = str(message.channel.id)
        logger.debug(f"[Bot] New message in {message.channel.name} from {message.author.name}")

        # 0) Ignore messages that start with a period
        if message.content.startswith('.'):
            return

        # 1) Initialize channel counter if it doesn't exist
        if channel_id not in self.new_messages:
            self.new_messages[channel_id] = 0

        # 2) Ignore messages from ourselves or system messages
        if message.author == self.user:
            self.new_messages[channel_id] = 0
            return
        
        # 3) Increment message counter for this channel
        self.new_messages[channel_id] += 1
        
        # 4) Check if we should notify about high message count (threshold = 100)
        if self.new_messages[channel_id] >= 100:
            await self.notify_high_message_count(message.channel, self.new_messages[channel_id])
            # Reset counter after notification
            self.new_messages[channel_id] = 0
        
        # 5) Check for mentions/keywords and calculate response probability
        should_respond = False
        
        # Direct mention or reply
        if self.user.mentioned_in(message) or (message.reference and message.reference.resolved and message.reference.resolved.author == self.user):
            should_respond = True
        # Contains "hephia" but not a direct mention/reply
        elif "hephia" in message.content.lower():
            # 85% chance to respond
            if random.random() < 0.85:
                should_respond = True
        else:
            # Calculate sigmoid-scaled probability based on message count
            x = (self.new_messages[channel_id] - 50) / 15.0  # Normalize and center around 50
            sigmoid = 0.75 / (1 + math.exp(-x))  # Sigmoid scaled to max 75%
            if random.random() < sigmoid:  # Will approach 75% chance as count nears 100
                should_respond = True
        
        # 6) Forward message if we should respond
        if should_respond:
            logger.info(f"[Bot] Triggered response in channel {message.channel.name}")
            await self.forward_to_hephia(message)
            # Reset counter for this channel
            self.new_messages[channel_id] = 0

    async def notify_high_message_count(self, channel: discord.TextChannel, count: int):
        """
        Notifies Hephia server about high message count in a channel.
        """
        if not self.session:
            logger.error("[Bot] Session not defined, cannot notify high message count.")
            return
        
        url = f"{HEPHIA_SERVER_URL}/discord_channel_update"
        data = {
            "channel_id": str(channel.id),
            "new_message_count": count,
            "channel_name": channel.name
        }
        try:
            async with self.session.post(url, json=data) as resp:
                if resp.status != 200:
                    err = await resp.text()
                    logger.warning(f"[Bot] High message count notification failed: {resp.status} {err}")
                else:
                    logger.debug("[Bot] High message count notification successful")
        except Exception as e:
            logger.exception(f"[Bot] Failed to notify about high message count: {e}")

    async def forward_to_hephia(self, message: discord.Message):
        """
        Forwards a Discord message to Hephia's /discord_inbound endpoint.
        Includes recent message history for context.
        """
        logger.info(f"[Bot] Forwarding message from {message.author.name} in {message.channel.name}")

        if not self.session:
            logger.error("[Bot] Session not defined, cannot forward message.")
            return

        # Get the proper user reference format
        if hasattr(message.author, 'global_name'):  # New Discord system
            author_ref = message.author.name
        else:  # Legacy Discord system
            author_ref = f"{message.author.display_name}#{message.author.discriminator}"

        # Clean up message content
        content = message.content
        bot_mention = f'<@{self.user.id}>'
        bot_mention_bang = f'<@!{self.user.id}>'
        content = content.replace(bot_mention, f'@{self.user.name}')
        content = content.replace(bot_mention_bang, f'@{self.user.name}')

        # Get recent message history for context (last 100 messages)
        try:
            logger.debug(f"[Bot] Fetching message history for context before {message.id}")
            history = []
            message_count = 0
            
            async for hist_msg in message.channel.history(limit=100, before=message):
                message_count += 1
                if hasattr(hist_msg.author, 'global_name'):
                    hist_author = hist_msg.author.name
                else:
                    hist_author = f"{hist_msg.author.display_name}#{hist_msg.author.discriminator}"
                
                hist_content = hist_msg.content
                hist_content = hist_content.replace(bot_mention, f'@{self.user.name}')
                hist_content = hist_content.replace(bot_mention_bang, f'@{self.user.name}')
                
                history.append({
                    "id": str(hist_msg.id),
                    "author": hist_author,
                    "author_id": str(hist_msg.author.id),
                    "content": hist_content,
                    "timestamp": str(hist_msg.created_at.isoformat())
                })
            
            logger.debug(f"[Bot] Found {message_count} context messages")
            history.reverse()  # Oldest first
        except Exception as e:
            logger.warning(f"[Bot] Could not fetch message history: {e}")
            history = []

        channel_id = str(message.channel.id)
        current_count = self.new_messages.get(channel_id, 0)
        logger.debug(f"[Bot] Channel {message.channel.name} has {current_count} new messages recorded")

        inbound_data = {
            "channel_id": channel_id,
            "message_id": str(message.id),
            "author": author_ref,
            "author_id": str(message.author.id),
            "content": content,
            "timestamp": str(message.created_at.isoformat()),
            "context": {
                "recent_history": history,
                "channel_name": message.channel.name,
                "guild_name": message.guild.name if message.guild else "DM",
                "message_count": current_count
            }
        }

        url = f"{HEPHIA_SERVER_URL}/discord_inbound"
        try:
            async with self.session.post(url, json=inbound_data) as resp:
                if resp.status != 200:
                    err = await resp.text()
                    logger.warning(f"[Bot] Error forwarding to Hephia: {resp.status} {err}")
                else:
                    logger.info("[Bot] Successfully forwarded message to Hephia")
        except Exception as e:
            logger.exception(f"[Bot] Failed to forward message to Hephia: {e}")

    async def update_name_mappings(self):
        """Update all name-to-ID and ID-to-name mappings."""
        logger.info("[Bot] Updating guild and channel name mappings")
        
        # Clear existing mappings
        self.guild_name_to_id.clear()
        self.id_to_guild_name.clear()
        self.channel_path_to_id.clear()
        self.id_to_channel_name.clear()
        
        # Update guild mappings
        for guild in self.guilds:
            guild_id = str(guild.id)
            guild_name = guild.name
            
            # Handle potential duplicate guild names by appending ID to duplicates
            if guild_name in self.guild_name_to_id:
                logger.warning(f"[Bot] Duplicate guild name detected: {guild_name}")
                # Keep original mapping but log the conflict
                self.id_to_guild_name[guild_id] = guild_name
            else:
                self.guild_name_to_id[guild_name] = guild_id
                self.id_to_guild_name[guild_id] = guild_name
            
            # Update channel mappings for this guild
            for channel in guild.channels:
                if isinstance(channel, discord.TextChannel):
                    channel_id = str(channel.id)
                    channel_name = channel.name
                    
                    # Create path formats: "Guild:channel" and "guild:channel"
                    path_case_sensitive = f"{guild_name}:{channel_name}"
                    path_lower = path_case_sensitive.lower()
                    
                    # Store both case-sensitive and lowercase mappings for better matching
                    self.channel_path_to_id[path_case_sensitive] = channel_id
                    self.channel_path_to_id[path_lower] = channel_id
                    
                    # Store ID to path mapping
                    self.id_to_channel_name[channel_id] = path_case_sensitive
        
        logger.info(f"[Bot] Mapped {len(self.guild_name_to_id)} guilds and {len(self.channel_path_to_id)//2} channels")

    def find_channel_id(self, path: str) -> tuple[str, bool]:
        """
        Find channel ID from a path string like "Guild:channel".
        Returns (channel_id, success) tuple.
        
        Supports various formats:
        - "Guild:channel" - Exact case match
        - "guild:channel" - Case-insensitive match
        - Raw channel ID
        """
        # If it's already an ID, return as-is
        if path.isdigit():
            return path, True
        
        # Check direct path matches
        if path in self.channel_path_to_id:
            return self.channel_path_to_id[path], True
        
        # Try case-insensitive matching
        path_lower = path.lower()
        if path_lower in self.channel_path_to_id:
            return self.channel_path_to_id[path_lower], True
        
        # Try to parse as guild:channel format
        if ":" in path:
            guild_name, channel_name = path.split(":", 1)
            
            # Try to find guild ID
            guild_id = None
            if guild_name in self.guild_name_to_id:
                guild_id = self.guild_name_to_id[guild_name]
            else:
                # Try case-insensitive guild match
                for name, id in self.guild_name_to_id.items():
                    if name.lower() == guild_name.lower():
                        guild_id = id
                        break
            
            if guild_id:
                # Find matching channel in that guild
                guild = self.get_guild(int(guild_id))
                if guild:
                    for channel in guild.channels:
                        if isinstance(channel, discord.TextChannel):
                            if channel.name.lower() == channel_name.lower():
                                return str(channel.id), True
        
        # Nothing found
        return "", False
    
    async def find_message_by_reference(self, channel, reference: str) -> Optional[discord.Message]:
        """
        Find a message in a channel based on a user-friendly reference.
        
        Supported reference formats:
        - "latest" - The most recent message
        - "latest-from:<username>" - Latest message from a specific user
        - "contains:<text>" - Latest message containing specific text
        - "#<number>" - The Nth message from recent history (1-based, where #1 is newest)
        
        Args:
            channel: Discord channel object
            reference: User-friendly reference string
            
        Returns:
            discord.Message if found, None otherwise
        """
        try:
            if not isinstance(channel, discord.TextChannel):
                logger.error(f"[Bot] Invalid channel type: {type(channel)}")
                return None
            
            # Sanitize the reference string first
            reference = reference.strip()
            
            # Remove surrounding quotes if present
            if (reference.startswith('"') and reference.endswith('"')) or \
            (reference.startswith("'") and reference.endswith("'")):
                reference = reference[1:-1].strip()
                
            logger.debug(f"[Bot] Searching for message with sanitized reference: '{reference}'")
                
            # Handle "latest" reference - simplest case
            if reference.lower() == "latest":
                async for message in channel.history(limit=1):
                    return message
                return None
                
            # Handle "#N" reference (message number)
            if reference.startswith("#"):
                try:
                    logger.debug(f"[Bot] Parsing message number from reference: '{reference}'")
                    index = int(reference[1:]) - 1  # Convert to 0-based
                    logger.debug(f"[Bot] Converted to index: {index}")
                    
                    if index < 0:
                        logger.warning(f"[Bot] Invalid negative index: {index}")
                        return None
                        
                    # Fetch messages newest-first, so #1 is the newest message
                    logger.debug(f"[Bot] Fetching messages newest-first for reference #{index + 1}")
                    messages = []
                    
                    # Use oldest_first=False to get newest messages first
                    # This ensures #1 is the newest message, #2 is the second newest, etc.
                    async for msg in channel.history(limit=index + 1 + 5, oldest_first=False):
                        messages.append(msg)
                    
                    logger.debug(f"[Bot] Fetched {len(messages)} messages, need index {index}")
                    if index < len(messages):
                        logger.debug(f"[Bot] Found message at index {index} from {messages[index].author.name}")
                        return messages[index]
                    else:
                        logger.warning(f"[Bot] Index {index} exceeds available message count {len(messages)}")
                    return None
                except ValueError:
                    logger.error(f"[Bot] Invalid message number reference: {reference}")
                    return None
                
            # Handle "latest-from:<username>" - already fetches newest first
            if reference.lower().startswith("latest-from:"):
                username = reference[12:].strip().lower()
                if not username:
                    return None
                    
                async for message in channel.history(limit=100):
                    author_name = message.author.name.lower()
                    # Try multiple forms of the username
                    if (username == author_name or 
                        username in author_name or
                        (hasattr(message.author, 'display_name') and 
                        username in message.author.display_name.lower())):
                        return message
                return None
                
            # Handle "contains:<text>" - already fetches newest first
            if reference.lower().startswith("contains:"):
                search_text = reference[9:].strip().lower()
                if not search_text:
                    return None
                    
                async for message in channel.history(limit=100):
                    if search_text in message.content.lower():
                        return message
                return None
                
            # If no recognized format, log with more details
            logger.warning(f"[Bot] Unrecognized message reference format: '{reference}' (length: {len(reference)})")
            logger.debug(f"[Bot] First few characters as hex: {':'.join(hex(ord(c)) for c in reference[:5])}")
            return None
        except Exception as e:
            logger.error(f"[Bot] Error finding message by reference: {e}")
            return None
        
    def format_message_for_display(self, message: discord.Message, index: Optional[int] = None) -> dict:
        """
        Format a Discord message for display with user-friendly references.
        
        Args:
            message: Discord message to format
            index: Optional message index/number for referencing
            
        Returns:
            Dictionary with formatted message data
        """
        if hasattr(message.author, 'global_name'):
            author_ref = message.author.name
        else:
            author_ref = f"{message.author.display_name}#{message.author.discriminator}"
            
        # Clean up message content by replacing bot mentions
        bot_mention = f'<@{self.user.id}>'
        bot_mention_bang = f'<@!{self.user.id}>'
        content = message.content.replace(bot_mention, f'@{self.user.name}')
        content = content.replace(bot_mention_bang, f'@{self.user.name}')
        
        # Get channel path if available
        channel_id = str(message.channel.id)
        channel_path = self.id_to_channel_name.get(channel_id, f"Unknown:{message.channel.name}")
        
        result = {
            "id": str(message.id),  # Keep ID for internal use
            "author": author_ref,
            "content": content,
            "timestamp": str(message.created_at.isoformat()),
            "channel_path": channel_path
        }
        
        # Add index reference if provided
        if index is not None:
            result["reference"] = f"#{index + 1}"  # 1-based for user display
            
        return result


###############################################################################
# AIOHTTP SERVER ROUTES: OUTBOUND COMMANDS
###############################################################################
async def handle_list_guilds_with_channels(request: web.Request) -> web.Response:
    """
    GET /guilds-with-channels
    Returns JSON with guild and channel information in hierarchical format.
    Useful for displaying available destinations to users.
    """
    result = []
    
    for guild in bot.guilds:
        guild_data = {
            "id": str(guild.id),
            "name": guild.name,
            "channels": []
        }
        
        # Only include text channels
        for channel in guild.channels:
            if isinstance(channel, discord.TextChannel):
                guild_data["channels"].append({
                    "id": str(channel.id),
                    "name": channel.name,
                    "path": f"{guild.name}:{channel.name}"
                })
        
        # Only include guilds with at least one text channel
        if guild_data["channels"]:
            result.append(guild_data)
    
    return web.json_response(result)

async def handle_send_by_path(request: web.Request) -> web.Response:
    """
    POST /send-by-path
    JSON body: { "path": "Guild:channel", "content": "Hello world!" }
    Returns: { "status": "ok", "message_id": "<ID>", "channel_id": "<ID>" }
    
    Handles sending messages using the friendly path format.
    """
    try:
        data = await request.json()
    except Exception as e:
        return web.json_response({"error": f"Invalid JSON: {str(e)}"}, status=400)
    
    path = data.get("path", "").strip()
    content = data.get("content", "").strip()
    
    if not path or not content:
        return web.json_response({"error": "Path and content are required"}, status=400)
    
    channel_id, found = bot.find_channel_id(path)
    if not found:
        return web.json_response({"error": f"Channel path '{path}' not found"}, status=404)
    
    # Reuse the existing send_message endpoint logic
    channel = bot.get_channel(int(channel_id))
    if not channel or not isinstance(channel, discord.TextChannel):
        return web.json_response({"error": "Channel not found or not a text channel"}, status=404)
    
    try:
        # Discord's message length limit (2000 chars)
        MAX_LENGTH = 2000
        if len(content) > MAX_LENGTH:
            logger.warning(f"[Bot] Message too long ({len(content)} chars), truncating to {MAX_LENGTH}")
            content = content[:MAX_LENGTH]
            
        sent_msg = await channel.send(content)
        logger.info(f"[Bot] Message sent in channel {channel.name} with ID {sent_msg.id}")
        return web.json_response({
            "status": "ok", 
            "message_id": str(sent_msg.id),
            "channel_id": channel_id,
            "path": bot.id_to_channel_name.get(channel_id, path)
        })
    except Exception as e:
        logger.exception(f"[Bot] Failed to send message: {e}")
        return web.json_response({"error": f"Failed to send message: {e}"}, status=500)
    
async def handle_find_message(request: web.Request) -> web.Response:
    """
    GET /find-message?path=Server:channel&reference=<reference>
    
    Find a message using a user-friendly reference:
    - "latest" - The most recent message
    - "latest-from:<username>" - Latest message from a specific user
    - "contains:<text>" - Latest message containing specific text
    - "#<number>" - The Nth message from the recent history
    
    Returns the message if found.
    """
    path = request.query.get("path", "").strip()
    reference = request.query.get("reference", "").strip()
    
    if not path or not reference:
        return web.json_response(
            {"error": "Both path and reference parameters are required"}, 
            status=400
        )
    
    channel_id, found = bot.find_channel_id(path)
    if not found:
        return web.json_response(
            {"error": f"Channel path '{path}' not found"}, 
            status=404
        )
    
    channel = bot.get_channel(int(channel_id))
    if not channel or not isinstance(channel, discord.TextChannel):
        return web.json_response(
            {"error": "Channel not found or not a text channel"}, 
            status=404
        )
    
    try:
        message = await bot.find_message_by_reference(channel, reference)
        if not message:
            return web.json_response(
                {"error": f"No message found matching reference '{reference}'"}, 
                status=404
            )
        
        result = bot.format_message_for_display(message)
        return web.json_response(result)
    except Exception as e:
        logger.exception(f"[Bot] Error in find_message: {e}")
        return web.json_response(
            {"error": f"Error finding message: {str(e)}"}, 
            status=500
        )

async def handle_enhanced_history(request: web.Request) -> web.Response:
    """
    GET /enhanced-history?path=Server:channel&limit=50
    
    Returns history with user-friendly message references and additional metadata.
    """
    path = request.query.get("path", "").strip()
    if not path:
        return web.json_response({"error": "Path parameter is required"}, status=400)
    
    limit_str = request.query.get("limit", "50")
    try:
        limit = min(int(limit_str), 100)  # Some safe cap
    except ValueError:
        limit = 50
    
    channel_id, found = bot.find_channel_id(path)
    if not found:
        return web.json_response(
            {"error": f"Channel path '{path}' not found"}, 
            status=404
        )
    
    channel = bot.get_channel(int(channel_id))
    if not channel or not isinstance(channel, discord.TextChannel):
        return web.json_response(
            {"error": "Channel not found or not a text channel"}, 
            status=404
        )
    
    try:
        # Fetch messages with a timeout - getting NEWEST messages first
        history_iter = channel.history(limit=limit, oldest_first=False)
        async with asyncio.timeout(30.0):
            messages = [msg async for msg in history_iter]
        
        # Format messages with indices for easy referencing
        formatted_messages = []
        
        # Process in reverse order to ensure consistent numbering
        # Message #1 should always be the newest message
        for i, msg in enumerate(messages):
            # Note: we're numbering from 1, not 0
            reference_number = i + 1
            formatted = bot.format_message_for_display(msg, i)
            
            # Ensure the reference is set
            formatted["reference"] = f"#{reference_number}"
            formatted_messages.append(formatted)
            
        # IMPORTANT: Now reverse the order for display so oldest are first
        # This keeps the NUMBERING consistent (#1 = newest) but displays oldest first
        formatted_messages.reverse()
        
        result = {
            "path": path,
            "channel_id": channel_id,  # Keep for backward compatibility
            "message_count": len(formatted_messages),
            "messages": formatted_messages,
            "display_order": "oldest_first",  # Document the display order
            "numbering": "newest_first"       # Document the numbering scheme
        }
        
        return web.json_response(result)
    except asyncio.TimeoutError:
        return web.json_response(
            {"error": "Request timed out while fetching history"}, 
            status=504
        )
    except Exception as e:
        logger.exception(f"[Bot] Error in enhanced_history: {e}")
        return web.json_response(
            {"error": f"Failed to fetch history: {str(e)}"}, 
            status=500
        )

async def handle_reply_to_message(request: web.Request) -> web.Response:
    """
    POST /reply-to-message
    JSON body: {
        "path": "Server:channel",
        "reference": "<reference>",
        "content": "Reply message"
    }
    
    Reply to a specific message identified by a user-friendly reference.
    """
    try:
        data = await request.json()
        logger.info(f"[Bot] Received reply-to-message request")
    except Exception as e:
        error_msg = f"Invalid JSON payload: {e}"
        logger.error(f"[Bot] {error_msg}")
        return web.json_response({"error": error_msg}, status=400)
    
    path = data.get("path", "").strip()
    reference = data.get("reference", "").strip()
    content = data.get("content", "").strip()
    
    logger.info(f"[Bot] Attempting to reply to message: path='{path}', reference='{reference}'")
    logger.debug(f"[Bot] Reply content length: {len(content)}")
    
    if not path or not reference or not content:
        error_msg = "Path, reference, and content are all required"
        logger.error(f"[Bot] {error_msg}")
        return web.json_response({"error": error_msg}, status=400)
    
    # Find the channel
    channel_id, found = bot.find_channel_id(path)
    if not found:
        error_msg = f"Channel path '{path}' not found"
        logger.error(f"[Bot] {error_msg}")
        return web.json_response({"error": error_msg}, status=404)
    
    # Log the resolved channel
    channel = bot.get_channel(int(channel_id))
    if not channel:
        error_msg = f"Channel with ID {channel_id} not found in bot's cache"
        logger.error(f"[Bot] {error_msg}")
        return web.json_response({"error": error_msg}, status=404)
        
    logger.info(f"[Bot] Resolved path '{path}' to channel '{channel.name}' in guild '{channel.guild.name if channel.guild else 'DM'}' with ID {channel_id}")
    
    if not isinstance(channel, discord.TextChannel):
        error_msg = "Channel is not a text channel"
        logger.error(f"[Bot] {error_msg}")
        return web.json_response({"error": error_msg}, status=404)
    
    try:
        # Find the message to reply to
        logger.info(f"[Bot] Finding message with reference: '{reference}'")
        referenced_message = await bot.find_message_by_reference(channel, reference)
        if not referenced_message:
            error_msg = f"No message found matching reference '{reference}'"
            logger.error(f"[Bot] {error_msg}")
            return web.json_response({"error": error_msg}, status=404)
        
        logger.info(f"[Bot] Found message from {referenced_message.author.name} to reply to")
        
        # Discord's message length limit (2000 chars)
        MAX_LENGTH = 2000
        if len(content) > MAX_LENGTH:
            logger.warning(f"[Bot] Message too long ({len(content)} chars), truncating to {MAX_LENGTH}")
            content = content[:MAX_LENGTH]
        
        # Send the reply
        logger.info(f"[Bot] Sending reply to message in {channel.name}")
        sent_msg = await channel.send(content, reference=referenced_message)
        logger.info(f"[Bot] Reply sent with ID {sent_msg.id}")
        
        # Format the reply for response
        result = bot.format_message_for_display(sent_msg)
        result["replied_to"] = bot.format_message_for_display(referenced_message)
        result["status"] = "ok"
        
        return web.json_response(result)
    except Exception as e:
        logger.exception(f"[Bot] Failed to send reply: {e}")
        return web.json_response({"error": f"Failed to send reply: {e}"}, status=500)

###############################################################################
# SETTING UP THE AIOHTTP WEB SERVER
###############################################################################

def create_app() -> web.Application:
    """
    Creates an aiohttp Application and registers all routes.
    """
    app = web.Application()

    app.router.add_get("/guilds-with-channels", handle_list_guilds_with_channels)
    app.router.add_post("/send-by-path", handle_send_by_path)
    app.router.add_get("/find-message", handle_find_message)
    app.router.add_get("/enhanced-history", handle_enhanced_history)
    app.router.add_post("/reply-to-message", handle_reply_to_message)
    app.router.add_get("/health", lambda _: web.Response(text="OK"))

    return app

###############################################################################
# MESSAGE CACHE INITIALIZATION
###############################################################################

async def initialize_message_cache(
    bot: RealTimeDiscordBot, 
    messages_per_channel: int = 100, 
    concurrency_limit: int = 5
):
    """
    Pre-fetches message history for all accessible text channels.
    Uses a semaphore to limit concurrency.
    
    Args:
        bot: The Discord bot instance
        messages_per_channel: How many messages to fetch per channel
        concurrency_limit: Max concurrent fetches to avoid rate limit issues
    """
    logger.info("[Bot] Starting message cache initialization...")

    # Prepare tasks
    tasks = []
    sem = asyncio.Semaphore(concurrency_limit)

    async def fetch_channel_history(channel):
        async with sem:
            try:
                logger.debug(f"[Bot] Fetching history for {channel.guild.name}/{channel.name}")
                messages = [msg async for msg in channel.history(limit=messages_per_channel)]
                bot.new_messages[str(channel.id)] = len(messages)
                logger.debug(f"[Bot] Cached {len(messages)} messages from {channel.name}")
            except Exception as e:
                logger.warning(f"[Bot] Failed to fetch history for {channel.name}: {e}")

    for guild in bot.guilds:
        for channel in guild.channels:
            if isinstance(channel, discord.TextChannel):
                tasks.append(asyncio.create_task(fetch_channel_history(channel)))

    await asyncio.gather(*tasks)
    logger.info("[Bot] Message cache initialization complete")

###############################################################################
# MAIN ENTRY POINT
###############################################################################

async def main():
    global bot, persistent_session
    # Create a persistent aiohttp session for all outbound requests
    persistent_session = aiohttp.ClientSession()

    # Pass the session to the bot so it can be re-used for all calls
    bot = RealTimeDiscordBot(
        intents=discord.Intents.all(),
        session=persistent_session
    )

    # 1) Start the aiohttp server in a background task
    app = create_app()
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "0.0.0.0", BOT_HTTP_PORT)
    await site.start()
    logger.info(f"[HTTP] Aiohttp server running at http://0.0.0.0:{BOT_HTTP_PORT}")

    async def periodic_mapping_updates():
        while True:
            try:
                await asyncio.sleep(300)  # Update every 5 minutes
                await bot.update_name_mappings()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"[Bot] Error updating mappings: {e}")
                await asyncio.sleep(60)  # Retry sooner after error

    mapping_update_task = asyncio.create_task(periodic_mapping_updates())

    # 2) Start the Discord bot
    if not DISCORD_TOKEN:
        logger.error("[Bot] DISCORD_BOT_TOKEN is not set. Exiting.")
        return

    try:
        await bot.start(DISCORD_TOKEN)
    except KeyboardInterrupt:
        logger.info("[Bot] Received KeyboardInterrupt, shutting down...")
    finally:
        logger.info("[Bot] Closing Discord bot connection...")
        await bot.close()

        logger.info("[Bot] Stopping periodic mapping updates...")
        mapping_update_task.cancel()
        try:
            await mapping_update_task
        except asyncio.CancelledError:
            pass

        logger.info("[Bot] Cleaning up the web server...")
        await runner.cleanup()

        # Close the persistent session
        logger.info("[Bot] Closing persistent aiohttp session...")
        if not persistent_session.closed:
            await persistent_session.close()
        logger.info("[Bot] Shutdown complete.")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as e:
        logger.exception("[Bot] Fatal error in main: %s", e)