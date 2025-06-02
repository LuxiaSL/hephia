import math
import os
import random
import sys
import logging
import asyncio
from typing import Optional, Dict, List
import aiohttp
import discord
from aiohttp import web
from dotenv import load_dotenv
from datetime import datetime

################################################################################
# CUSTOM EXCEPTIONS
################################################################################
class ContextWindowError(Exception):
    """Base class for context window related errors."""
    def __init__(self, message: str, status_code: int = 400):
        super().__init__(message)
        self.status_code = status_code

class NoContextWindowError(ContextWindowError):
    """Raised when no context window exists for a channel."""
    def __init__(self, message: str = "No context window available. Please fetch history first.", status_code: int = 404):
        super().__init__(message, status_code)

class ContextWindowExpiredError(ContextWindowError):
    """Raised when a context window has expired."""
    def __init__(self, message: str = "Context window has expired. Please refresh history.", status_code: int = 400): # 400 or 409 Conflict
        super().__init__(message, status_code)

class InvalidWindowTimestampError(ContextWindowError):
    """Raised when a context window has an invalid timestamp."""
    def __init__(self, message: str = "Corrupted context window timestamp. Window cleared. Please refresh history.", status_code: int = 500):
        super().__init__(message, status_code)

class ReferenceNotInWindowError(ContextWindowError):
    """Raised when a #N reference is not found in an active window."""
    def __init__(self, message: str = "Reference not found in the current context window. Please refresh history.", status_code: int = 404):
        super().__init__(message, status_code)

class ReferencedMessageNotFound(ContextWindowError):
    """Raised when the message ID from window is not found in Discord (e.g., deleted)."""
    def __init__(self, message: str = "Referenced message not found. It might have been deleted.", status_code: int = 404):
        super().__init__(message, status_code)

class ReferencedMessageForbidden(ContextWindowError):
    """Raised when bot is forbidden to fetch the message ID from window."""
    def __init__(self, message: str = "Bot lacks permissions to fetch the referenced message.", status_code: int = 403):
        super().__init__(message, status_code)

class InvalidMessageIdFormatInWindow(ContextWindowError):
    """Raised if the message ID in the window is not a valid integer."""
    def __init__(self, message: str = "Invalid message ID format in context window. Please refresh context.", status_code: int = 500):
        super().__init__(message, status_code)

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
MAX_MESSAGE_CACHE_SIZE = 1000
CONTEXT_WINDOW_EXPIRY_MINUTES = 5

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
        self.new_messages: Dict[str, int] = {}
        self.message_cache: Dict[str, List[discord.Message]] = {}
        self.guild_name_to_id = {}  # Maps guild names to IDs
        self.channel_path_to_id = {}  # Maps "guild:channel" paths to channel IDs
        self.id_to_guild_name = {}  # Maps guild IDs to names
        self.id_to_channel_name = {}  # Maps channel IDs to names (including guild context)
        # We'll store the persistent aiohttp session reference
        self.session = kwargs.get("session", None)

        self.history_windows: Dict[str, Dict[str, any]] = {} #stores references for use

    async def on_ready(self):
        logger.info(f"[Bot] Logged in as {self.user} (id={self.user.id})")
        await self.update_name_mappings()
        await self.populate_full_message_cache()

    async def on_message(self, message: discord.Message):
        """
        Called whenever a new message is created in Discord.
        Tracks message counts per channel and notifies when thresholds are reached.
        Also handles random engagement with sigmoid scaling.
        """
        channel_id = str(message.channel.id)
        logger.debug(f"[Bot] New message in {message.channel.name} from {message.author.name}")

        if channel_id not in self.message_cache:
            self.message_cache[channel_id] = []
            logger.info(f"[Bot] Initialized empty message cache for new/unseen channel {message.channel.name} (ID: {channel_id})")

        self.message_cache[channel_id].insert(0, message)

        if len(self.message_cache[channel_id]) > MAX_MESSAGE_CACHE_SIZE:
            self.message_cache[channel_id] = self.message_cache[channel_id][:MAX_MESSAGE_CACHE_SIZE]

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
        
        # 4) Check if we should notify about high message count
        if self.new_messages[channel_id] >= 50:
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
            # 75% chance to notify
            if random.random() < 0.75:
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

    async def populate_full_message_cache(self, messages_per_channel: int = MAX_MESSAGE_CACHE_SIZE, concurrency_limit: int = 5):
        """
        Pre-fetches message history for all accessible text channels and populates self.message_cache.
        Uses a semaphore to limit concurrency.
        """
        logger.info(f"[Bot] Populating message cache with at most {messages_per_channel} messages per channel")
        tasks = []
        sem = asyncio.Semaphore(concurrency_limit)

        async def fetch_and_cache_channel_history(channel: discord.TextChannel):
            async with sem:
                try:
                    logger.debug(f"[Bot] Caching history for {channel.guild.name}/{channel.name} (ID: {channel.id})")
                    # Fetch newest messages first to easily cap the cache if needed,
                    # and to align with how on_message will prepend messages.
                    # messages_per_channel defaults to MAX_MESSAGE_CACHE_SIZE
                    history_messages = [msg async for msg in channel.history(limit=messages_per_channel, oldest_first=False)]

                    # Store the fetched discord.Message objects directly.
                    # The list is already newest-first.
                    self.message_cache[str(channel.id)] = history_messages 
                    logger.debug(f"[Bot] Cached {len(history_messages)} message objects from {channel.guild.name}/{channel.name}")

                except discord.Forbidden:
                    logger.warning(f"[Bot] Permission denied: Cannot cache history for {channel.guild.name}/{channel.name} (ID: {channel.id}). Ensure the bot has 'View Channel' and 'Read Message History' permissions.")
                except Exception as e:
                    logger.warning(f"[Bot] Failed to cache history for {channel.guild.name}/{channel.name} (ID: {channel.id}): {e}")

        for guild in self.guilds:
            for channel in guild.channels:
                tasks.append(asyncio.create_task(fetch_and_cache_channel_history(channel)))

        await asyncio.gather(*tasks, return_exceptions=True)
        logger.info(f"[Bot] Finished populating message cache. Total channels cached: {len(self.message_cache)}")

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

        history_message_objects = []
        source = "cache"

        channel_id_str = str(message.channel.id)
        desired_history_limit = 1000

        if channel_id_str in self.message_cache:
            cached_messages = self.message_cache[channel_id_str]

            try:
                current_message_index_in_cache = -1
                for i, cached_msg in enumerate(cached_messages):
                    if cached_msg.id == message.id:
                        current_message_index_in_cache = i
                        break
                
                if current_message_index_in_cache != -1:
                    history_message_objects = cached_messages[current_message_index_in_cache + 1 : current_message_index_in_cache + 1 + desired_history_limit]
                else:
                    history_message_objects = cached_messages[:desired_history_limit]

                if history_message_objects:
                    logger.debug(f"[Bot] Using {len(history_message_objects)} messages from cache for Hephia context (before message {message.id}).")
            except Exception as e:
                logger.warning(f"[Bot] Error processing cache for Hephia context: {e}. Will attempt live fetch.")
                history_message_objects = []
        
        if not history_message_objects:
            source = "live_api_fallback"
            logger.info(f"[Bot] Cache did not provide sufficient context for Hephia for message {message.id}. Fetching live history.")
            try:
                live_fetched_list = [msg async for msg in message.channel.history(limit=desired_history_limit, before=message)]
                history_message_objects = live_fetched_list
                logger.debug(f"[Bot] Fetched {len(history_message_objects)} live context messages for Hephia.")
            except discord.Forbidden:
                logger.warning(f"[Bot] Permission denied fetching live history for Hephia context for channel {message.channel.name}.")
                history_message_objects = [] # Ensure empty on permission error
            except Exception as e:
                logger.warning(f"[Bot] Could not fetch live message history for Hephia context: {e}")
                history_message_objects = [] # Ensure empty on other errors

        formatted_history_for_hephia = []
        if history_message_objects:
            message_objects_oldest_first = list(reversed(history_message_objects))

            async for hist_msg in message_objects_oldest_first:
                if hasattr(hist_msg.author, 'global_name'):
                    hist_author = hist_msg.author.name
                else:
                    hist_author = f"{hist_msg.author.display_name}#{hist_msg.author.discriminator}"

                hist_content = hist_msg.content 
                hist_content = hist_content.replace(bot_mention, f'@{self.user.name}')
                hist_content = hist_content.replace(bot_mention_bang, f'@{self.user.name}')
                
                formatted_history_for_hephia.append({
                    "id": str(hist_msg.id),
                    "author": hist_author,
                    "author_id": str(hist_msg.author.id),
                    "content": hist_content,
                    "timestamp": str(hist_msg.created_at.isoformat())
                })

        current_count = self.new_messages.get(channel_id_str, 0) # Existing logic
        logger.debug(f"[Bot] Channel {message.channel.name} has {current_count} new messages recorded by counter.")

        inbound_data = {
            "channel_id": channel_id_str,
            "message_id": str(message.id),
            "author": author_ref,
            "author_id": str(message.author.id),
            "content": content,
            "timestamp": str(message.created_at.isoformat()),
            "context": {
                "recent_history": formatted_history_for_hephia, # Use the newly processed history
                "channel_name": message.channel.name,
                "guild_name": message.guild.name if message.guild else "DM",
                "message_count": current_count, # This is from self.new_messages counter
                "history_source": source # Added for debugging
            }
        }

        url = f"{HEPHIA_SERVER_URL}/discord_inbound"
        try:
            async with self.session.post(url, json=inbound_data) as resp:
                if resp.status != 200:
                    err_text = await resp.text()
                    logger.warning(f"[Bot] Error forwarding to Hephia: {resp.status} {err_text}")
                else:
                    logger.info(f"[Bot] Successfully forwarded message {message.id} to Hephia (history from {source}).")
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
            
            channel_id_str = str(channel.id)
            cached_messages = self.message_cache.get(channel_id_str, []) # Newest first

            # Sanitize the reference string first
            reference = reference.strip()
            
            # Remove surrounding quotes if present
            if (reference.startswith('"') and reference.endswith('"')) or \
            (reference.startswith("'") and reference.endswith("'")):
                reference = reference[1:-1].strip()
                
            logger.debug(f"[Bot] Searching for message with sanitized reference: '{reference}'")
                
            # Handle "latest" reference
            if reference.lower() == "latest":
                if cached_messages:
                    logger.debug(f"[Bot find_message_by_reference] 'latest' found in cache for {channel.name}.")
                    return cached_messages[0]
                else:
                    logger.debug(f"[Bot find_message_by_reference] 'latest' not in cache for {channel.name}, falling back to API.")
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
                        
                    if index < len(cached_messages):
                        logger.debug(f"[Bot find_message_by_reference] '#{index+1}' found in cache for {channel.name}.")
                        return cached_messages[index]
                    else:
                        logger.debug(f"[Bot find_message_by_reference] '#{index+1}' not in cache or index out of bounds ({len(cached_messages)} items) for {channel.name}, falling back to API.")
                        # Fallback: Fetch enough messages to cover the index
                        # The original logic fetched index + 1 + 5; we'll stick to that buffer.
                        # Ensure oldest_first=False to align with cache structure (newest first).
                        messages_from_api = []
                        async for msg in channel.history(limit=index + 1 + 5, oldest_first=False):
                            messages_from_api.append(msg)
                        
                        if index < len(messages_from_api):
                            return messages_from_api[index]
                        else:
                            logger.warning(f"[Bot find_message_by_reference] Index {index} exceeds API fetched message count {len(messages_from_api)} for {channel.name}")
                            return None
                except ValueError:
                    logger.error(f"[Bot] Invalid message number reference: {reference}")
                    return None
                
            # Handle "latest-from:<username>"
            if reference.lower().startswith("latest-from:"):
                username = reference[12:].strip().lower()
                if not username:
                    return None
                
                search_limit = min(len(cached_messages), 100)
                for i in range(search_limit):
                    msg = cached_messages[i]
                    author_name = msg.author.name.lower()
                    display_name_lower = msg.author.display_name.lower() if hasattr(msg.author, 'display_name') else ""
                    if (username == author_name or 
                        username in author_name or
                        (display_name_lower and username in display_name_lower)):
                        logger.debug(f"[Bot find_message_by_reference] 'latest-from' found in cache for {channel.name}.")
                        return msg

                logger.debug(f"[Bot find_message_by_reference] 'latest-from:{username}' not in cache for {channel.name}, falling back to API.")
                async for message in channel.history(limit=100):
                    author_name = message.author.name.lower()
                    # Try multiple forms of the username
                    if (username == author_name or 
                        username in author_name or
                        (display_name_lower and username in display_name_lower)):
                        logger.debug(f"[Bot find_message_by_reference] 'latest-from:{username}' found in API for {channel.name}.")
                        return message
                return None
                
            # Handle "contains:<text>" 
            if reference.lower().startswith("contains:"):
                search_text = reference[9:].strip().lower()
                if not search_text:
                    return None
                
                search_limit = min(len(cached_messages), 100)
                for i in range(search_limit):
                    msg = cached_messages[i]
                    if search_text in msg.content.lower():
                        logger.debug(f"[Bot find_message_by_reference] 'contains:{search_text}' found in cache for {channel.name}.")
                        return msg

                logger.debug(f"[Bot find_message_by_reference] 'contains:{search_text}' not in cache for {channel.name}, falling back to API.")
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
        
        # Check for embeds with text content and append to content
        if message.embeds:
            embed_texts = []
            for embed in message.embeds:
                if embed.description:
                    embed_texts.append(embed.description)
            if embed_texts:
                content = f"{content}\n[embedded info: {' '.join(embed_texts)}]"
        
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
    
    async def periodic_cleanup_history_windows(self):
        """Periodically cleans up expired history windows."""
        while True:
            await asyncio.sleep(60) # Check every minute
            try:
                now = datetime.now()
                expired_channels = []
                for channel_id, window_data in self.history_windows.items():
                    # Ensure timestamp is a datetime object
                    if isinstance(window_data.get("timestamp"), datetime):
                        age = now - window_data["timestamp"]
                        if age.total_seconds() > CONTEXT_WINDOW_EXPIRY_MINUTES * 60:
                            expired_channels.append(channel_id)
                    else:
                        # Log or handle cases where timestamp might not be set or is invalid
                        logger.warning(f"[Bot] Invalid or missing timestamp for history window in channel {channel_id}. Marking for removal.")
                        expired_channels.append(channel_id) # Mark for removal if timestamp is problematic

                for channel_id in expired_channels:
                    if channel_id in self.history_windows:
                        del self.history_windows[channel_id]
                        logger.info(f"[Bot] Expired history window for channel {channel_id} removed.")
                
                if expired_channels:
                    logger.debug(f"[Bot] History window cleanup complete. Removed {len(expired_channels)} expired windows.")
                else:
                    logger.debug(f"[Bot] History window cleanup: No expired windows found.")

            except asyncio.CancelledError:
                logger.info("[Bot] History window cleanup task cancelled.")
                break
            except Exception as e:
                logger.error(f"[Bot] Error during history window cleanup: {e}", exc_info=True)
                await asyncio.sleep(120) 

    async def get_message_from_window_reference(
        self, 
        channel_id_str: str, 
        reference_param: str, 
        channel_obj: discord.TextChannel
    ) -> discord.Message:
        """
        Retrieves a discord.Message object using a "#N" reference from the history window.

        Args:
            channel_id_str: The ID of the channel.
            reference_param: The "#N" reference string (e.g., "#1").
            channel_obj: The discord.TextChannel object.

        Returns:
            A discord.Message object.

        Raises:
            NoContextWindowError: If no window exists for the channel.
            InvalidWindowTimestampError: If the window timestamp is malformed.
            ContextWindowExpiredError: If the window has expired.
            ReferenceNotInWindowError: If the reference is not in the (valid) window.
            InvalidMessageIdFormatInWindow: If the message ID in window is malformed.
            ReferencedMessageNotFound: If Discord API reports message not found.
            ReferencedMessageForbidden: If Discord API reports access forbidden.
            discord.HTTPException: For other Discord API errors during fetch.
        """
        if not (reference_param.startswith("#") and reference_param[1:].isdigit()):
            # This method is specifically for #N references, caller should ensure this.
            # However, an internal check or different error type could be used.
            # For now, assume caller validates the format or let it proceed and fail at lookup.
            logger.warning(f"[get_message_from_window_reference] Called with non-#N reference: {reference_param}")
            # Or raise ValueError("Reference must be in '#N' format.")

        window_data = self.history_windows.get(channel_id_str)
        if not window_data:
            logger.info(f"[Bot] No active context window for channel {channel_id_str} for reference {reference_param}.")
            raise NoContextWindowError(f"No context window available for channel '{channel_obj.name}'. Please fetch history via /enhanced-history first.")

        window_timestamp = window_data.get("timestamp")
        if not isinstance(window_timestamp, datetime):
            logger.warning(f"[Bot] Invalid timestamp in context window for channel {channel_id_str}.")
            if channel_id_str in self.history_windows: # Check before deleting
                 del self.history_windows[channel_id_str]
            raise InvalidWindowTimestampError(f"Corrupted context window timestamp for channel '{channel_obj.name}'. Window cleared. Please refresh history.")

        age = datetime.now() - window_timestamp
        if age.total_seconds() > CONTEXT_WINDOW_EXPIRY_MINUTES * 60:
            logger.info(f"[Bot] Context window for channel {channel_id_str} expired for reference {reference_param}.")
            raise ContextWindowExpiredError(f"Context window for channel '{channel_obj.name}' has expired. Please refresh history.")

        message_id_str = window_data.get("messages", {}).get(reference_param)
        if not message_id_str:
            logger.warning(f"[Bot] Reference {reference_param} not found in active window for channel {channel_id_str}.")
            raise ReferenceNotInWindowError(f"Reference '{reference_param}' not found in the current context window for channel '{channel_obj.name}'. Please refresh history.")

        try:
            message_id_int = int(message_id_str)
        except ValueError:
            logger.error(f"[Bot] Invalid message ID format '{message_id_str}' for reference {reference_param} in window for channel {channel_id_str}.")
            raise InvalidMessageIdFormatInWindow(f"Invalid message ID format in context window for reference '{reference_param}' in channel '{channel_obj.name}'.")

        # Try to get from local message_cache first
        if channel_id_str in self.message_cache:
            for msg_in_cache in self.message_cache[channel_id_str]:
                if msg_in_cache.id == message_id_int:
                    logger.info(f"[Bot] Found message {message_id_str} for reference {reference_param} in local message cache.")
                    return msg_in_cache
        
        # If not in local cache, fetch from Discord
        logger.info(f"[Bot] Message {message_id_str} for reference {reference_param} not in local cache, fetching from API.")
        try:
            message_obj = await channel_obj.fetch_message(message_id_int)
            logger.info(f"[Bot] Successfully fetched message {message_id_str} (ref: {reference_param}) from API.")
            return message_obj
        except discord.NotFound:
            logger.warning(f"[Bot] Message with ID {message_id_str} (from window reference {reference_param}) not found in channel {channel_obj.name}.")
            raise ReferencedMessageNotFound(f"Message for reference '{reference_param}' (ID: {message_id_str}) not found in channel '{channel_obj.name}'. It might have been deleted.")
        except discord.Forbidden:
            logger.warning(f"[Bot] Forbidden to fetch message ID {message_id_str} for reference {reference_param} from channel {channel_obj.name}.")
            raise ReferencedMessageForbidden(f"Bot lacks permissions to fetch the referenced message (ref: '{reference_param}', ID: {message_id_str}) in channel '{channel_obj.name}'.")


###############################################################################
# AIOHTTP SERVER ROUTES: OUTBOUND COMMANDS
###############################################################################
async def handle_list_guilds_with_channels(request: web.Request) -> web.Response:
    """
    GET /guilds-with-channels
    Returns JSON with guild and channel information in hierarchical format.
    Only includes channels where the bot has both VIEW_CHANNEL and SEND_MESSAGES permissions.
    Useful for displaying available destinations to users.
    """
    result = []
    
    for guild in bot.guilds:
        guild_data = {
            "id": str(guild.id),
            "name": guild.name,
            "channels": []
        }
        
        # Only include text channels where bot has required permissions
        for channel in guild.channels:
            if isinstance(channel, discord.TextChannel):
                # Get bot's permissions for this channel
                bot_member = guild.get_member(bot.user.id)
                if not bot_member:
                    continue
                    
                channel_perms = channel.permissions_for(bot_member)
                
                # Check if bot has permission to both view and send messages
                if channel_perms.view_channel and channel_perms.send_messages:
                    guild_data["channels"].append({
                        "id": str(channel.id),
                        "name": channel.name,
                        "path": f"{guild.name}:{channel.name}"
                    })
        
        # Only include guilds with at least one accessible text channel
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

    For "#N" references, uses the context window. Other references use live/cached data.
    
    Returns the message if found.
    """
    path = request.query.get("path", "").strip()
    reference_param = request.query.get("reference", "").strip()
    
    if not path or not reference_param:
        return web.json_response(
            {"error": "Both path and reference parameters are required"}, 
            status=400
        )
    
    channel_id_str, found = bot.find_channel_id(path)
    if not found:
        return web.json_response(
            {"error": f"Channel path '{path}' not found"}, 
            status=404
        )
    
    channel_obj = bot.get_channel(int(channel_id_str))
    if not channel_obj or not isinstance(channel_obj, discord.TextChannel):
        return web.json_response(
            {"error": "Channel not found or not a text channel"}, 
            status=404
        )
    
    message_to_format: Optional[discord.Message] = None # Type hint

    # Check if it's a #N reference to use the history window
    if reference_param.startswith("#") and reference_param[1:].isdigit():
        logger.info(f"[API /find-message] Using window reference: {reference_param} for path {path}")
        try:
            message_to_format = await bot.get_message_from_window_reference(
                channel_id_str, reference_param, channel_obj
            )
        except ContextWindowError as e:
            logger.warning(f"[API /find-message] Context window error for '{reference_param}' in '{path}': {e}")
            return web.json_response({"error": str(e)}, status=e.status_code)
        except discord.HTTPException as e: # Catch other potential Discord errors like rate limits
            logger.exception(f"[API /find-message] Discord HTTP error resolving window reference '{reference_param}' in '{path}': {e}")
            return web.json_response({"error": f"Discord API error: {e.text if hasattr(e, 'text') else str(e)}"}, status=e.status if hasattr(e, 'status') else 500)
        except Exception as e: # Catch-all for unexpected errors from the helper
            logger.exception(f"[API /find-message] Unexpected error resolving window reference '{reference_param}' in '{path}': {e}")
            return web.json_response({"error": f"Unexpected error: {str(e)}"}, status=500)
    else:
        # For other reference types, use the existing find_message_by_reference method
        logger.info(f"[API /find-message] Using find_message_by_reference for non-#N reference: '{reference_param}' in path {path}")
        try:
            message_to_format = await bot.find_message_by_reference(channel_obj, reference_param)
        except Exception as e: # Catch potential errors from find_message_by_reference
            logger.exception(f"[API /find-message] Error in bot.find_message_by_reference for '{reference_param}': {e}")
            return web.json_response(
                {"error": f"Error processing reference '{reference_param}': {str(e)}"},
                status=500
            )

    # Common response part
    if not message_to_format:
        # This path should ideally be hit less for #N if errors are returned above,
        # but could be hit if find_message_by_reference returns None for other types.
        logger.warning(f"[API /find-message] No message found matching reference '{reference_param}' in channel {path} after all checks.")
        return web.json_response(
            {"error": f"No message found matching reference '{reference_param}'"}, 
            status=404
        )
    
    try:
        result = bot.format_message_for_display(message_to_format)
        return web.json_response(result)
    except Exception as e: # Should not happen if message_to_format is a valid Message object
        logger.exception(f"[API /find-message] Error formatting message for display: {e}")
        return web.json_response(
            {"error": f"Error formatting found message: {str(e)}"}, 
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
    
    limit_str = request.query.get("limit", "100")
    try:
        limit = min(int(limit_str), MAX_MESSAGE_CACHE_SIZE, 1000)  # Some safe cap
        if limit <= 0:
            limit = 100
    except ValueError:
        limit = 100
    
    channel_id, found = bot.find_channel_id(path)
    if not found:
        return web.json_response(
            {"error": f"Channel path '{path}' not found"}, 
            status=404
        )
    
    messages_to_format = []
    source = "cache"

    cached_messages_for_channel = bot.message_cache.get(str(channel_id), [])

    if cached_messages_for_channel:
        messages_to_format = cached_messages_for_channel[:limit]
        logger.debug(f"[API /enhanced-history] Served {len(messages_to_format)} messages for {path} from cache.")
    else:
        logger.warning(f"[API /enhanced-history] No cached messages for channel {path} (ID: {channel_id}): attempting fallback.")
        source = "live_api_fallback"

        channel = bot.get_channel(int(channel_id))
        if not channel or not isinstance(channel, discord.TextChannel):
            return web.json_response(
                {"error": f"Channel not found or not a text channel for ID {channel_id} during fallback."},
                status=404
            )
        try:
            # Fetch messages with a timeout - getting NEWEST messages first
            history_iter = channel.history(limit=limit, oldest_first=False)
            async with asyncio.timeout(30.0):
                messages_to_format = [msg async for msg in history_iter]
            logger.info(f"[API /enhanced-history] Served {len(messages_to_format)} messages for {path} from live API fallback.")
        except asyncio.TimeoutError:
            logger.error(f"[API /enhanced-history] Request timed out during live API fallback for {path}.")
            return web.json_response(
                {"error": "Request timed out while fetching history via fallback"}, 
                status=504 # Gateway Timeout
            )
        except discord.Forbidden:
            logger.error(f"[API /enhanced-history] Permission denied during live API fallback for {path}.")
            return web.json_response(
                {"error": "Permission denied while fetching history via fallback"},
                status=403 # Forbidden
            )
        except Exception as e:
            logger.exception(f"[API /enhanced-history] Error during live API fallback for {path}: {e}")
            return web.json_response(
                {"error": f"Failed to fetch history via fallback: {str(e)}"}, 
                status=500 # Internal Server Error
            )
    
    # Format messages with indices for easy referencing
    formatted_messages = []
    message_references_for_window: Dict[str, str] = {}
    
    # Process in reverse order to ensure consistent numbering
    # Message #1 should always be the newest message
    for i, msg_obj in enumerate(messages_to_format):
        formatted = bot.format_message_for_display(msg_obj, index=i) 
        # The 'reference' key (e.g., "#1") is correctly set by format_message_for_display
        formatted_messages.append(formatted)

        if "reference" in formatted and "id" in formatted:
            message_references_for_window[formatted["reference"]] = formatted["id"]
    
    if message_references_for_window:
        bot.history_windows[channel_id] = {
            "timestamp": datetime.now(),
            "message_references": message_references_for_window
        }
    elif not messages_to_format:
        bot.history_windows[channel_id] = {
            "timestamp": datetime.now(),
            "messages": {} # Empty messages
        }
        logger.info(f"[API /enhanced-history] History window for channel {channel_id} ({path}) set to empty as no messages were returned.")

    # IMPORTANT: Now reverse the order for display so oldest are first
    # This keeps the NUMBERING consistent (#1 = newest) but displays oldest first
    formatted_messages.reverse()
        
    result = {
        "path": path,
        "channel_id": channel_id,  # Keep for backward compatibility
        "message_count": len(formatted_messages),
        "messages": formatted_messages,
        "display_order": "oldest_first",  # Document the display order
        "numbering": "newest_first",      # Document the numbering scheme
        "source": source
    }
    
    return web.json_response(result)

async def handle_reply_to_message(request: web.Request) -> web.Response:
    """
    POST /reply-to-message
    JSON body: {
        "path": "Server:channel",
        "reference": "<reference>",
        "content": "Reply message"
    }
    
    Reply to a specific message. For "#N" references, uses the context window 
    via a helper method.
    """
    try:
        data = await request.json()
        logger.info(f"[API /reply-to-message] Received reply request.")
        logger.debug(f"[API /reply-to-message] Reply data: {data}")
    except Exception as e:
        error_msg = f"Invalid JSON payload: {e}"
        logger.error(f"[API /reply-to-message] {error_msg}")
        return web.json_response({"error": error_msg}, status=400)
    
    path = data.get("path", "").strip()
    payload_reference = data.get("reference", "").strip()
    content = data.get("content", "").strip()
    
    logger.info(f"[API /reply-to-message] Attempting to reply: path='{path}', reference='{payload_reference}'")
    logger.debug(f"[API /reply-to-message] Reply content length: {len(content)}")
    
    if not path or not payload_reference or not content:
        error_msg = "Path, reference, and content are all required"
        logger.error(f"[API /reply-to-message] {error_msg}")
        return web.json_response({"error": error_msg}, status=400)
    
    channel_id_str, found = bot.find_channel_id(path)
    if not found:
        error_msg = f"Channel path '{path}' not found"
        logger.error(f"[API /reply-to-message] {error_msg}")
        return web.json_response({"error": error_msg}, status=404)
    
    channel_obj = bot.get_channel(int(channel_id_str))
    if not channel_obj:
        error_msg = f"Channel with ID {channel_id_str} (from path '{path}') not found in bot's cache"
        logger.error(f"[API /reply-to-message] {error_msg}")
        return web.json_response({"error": error_msg}, status=404)
        
    logger.info(f"[API /reply-to-message] Resolved path '{path}' to channel '{channel_obj.name}' in guild '{channel_obj.guild.name if channel_obj.guild else 'DM'}' with ID {channel_id_str}")
    
    if not isinstance(channel_obj, discord.TextChannel):
        error_msg = "Channel is not a text channel"
        logger.error(f"[API /reply-to-message] {error_msg}")
        return web.json_response({"error": error_msg}, status=404)

    referenced_message: Optional[discord.Message] = None

    if payload_reference.startswith("#") and payload_reference[1:].isdigit():
        logger.info(f"[API /reply-to-message] Using window reference: {payload_reference} for path {path}")
        try:
            referenced_message = await bot.get_message_from_window_reference(
                channel_id_str, payload_reference, channel_obj
            )
        except ContextWindowError as e:
            logger.warning(f"[API /reply-to-message] Context window error for reply reference '{payload_reference}' in '{path}': {e}")
            return web.json_response({"error": str(e)}, status=e.status_code)
        except discord.HTTPException as e: 
            logger.exception(f"[API /reply-to-message] Discord HTTP error resolving window reference for reply '{payload_reference}' in '{path}': {e}")
            return web.json_response({"error": f"Discord API error: {e.text if hasattr(e, 'text') else str(e)}"}, status=e.status if hasattr(e, 'status') else 500)
        except Exception as e:
            logger.exception(f"[API /reply-to-message] Unexpected error resolving window reference for reply '{payload_reference}' in '{path}': {e}")
            return web.json_response({"error": f"Unexpected error: {str(e)}"}, status=500)
    else:
        logger.info(f"[API /reply-to-message] Using legacy find_message_by_reference for non-#N reply reference: '{payload_reference}' in path {path}")
        try:
            referenced_message = await bot.find_message_by_reference(channel_obj, payload_reference)
        except Exception as e:
            logger.exception(f"[API /reply-to-message] Error in bot.find_message_by_reference for reply reference '{payload_reference}': {e}")
            return web.json_response(
                {"error": f"Error processing reply reference '{payload_reference}': {str(e)}"},
                status=500
            )

    if not referenced_message:
        logger.error(f"[API /reply-to-message] No message found matching reference '{payload_reference}' in channel {path} to reply to, after all checks.")
        return web.json_response(
            {"error": f"No message found matching reference '{payload_reference}' to reply to."}, 
            status=404
        )
    
    try:
        logger.info(f"[API /reply-to-message] Found message ID {referenced_message.id} from {referenced_message.author.name} to reply to.")
        
        MAX_LENGTH = 2000
        if len(content) > MAX_LENGTH:
            logger.warning(f"[API /reply-to-message] Message too long ({len(content)} chars), truncating to {MAX_LENGTH}")
            content = content[:MAX_LENGTH]
        
        logger.info(f"[API /reply-to-message] Sending reply to message in {channel_obj.name}")
        sent_msg = await channel_obj.send(content, reference=referenced_message)
        logger.info(f"[API /reply-to-message] Reply sent with ID {sent_msg.id}")
        
        result = bot.format_message_for_display(sent_msg)
        replied_to_info = bot.format_message_for_display(referenced_message)
        replied_to_info.pop("reference", None) 
        result["replied_to"] = replied_to_info
        result["status"] = "ok"
        
        return web.json_response(result)
        
    except discord.Forbidden as e:
        logger.exception(f"[API /reply-to-message] Permission error when trying to send reply: {e}")
        return web.json_response({"error": f"Permission error sending reply: {e.text}"}, status=403)
    except discord.HTTPException as e:
        logger.exception(f"[API /reply-to-message] HTTP error when trying to send reply: {e}")
        return web.json_response({"error": f"Network or API error sending reply: {e.text if hasattr(e, 'text') else str(e)}"}, status=e.status if hasattr(e, 'status') else 500)
    except Exception as e:
        logger.exception(f"[API /reply-to-message] Failed to send reply: {e}")
        return web.json_response({"error": f"Failed to send reply: {str(e)}"}, status=500)
    
async def handle_get_user_list(request: web.Request) -> web.Response:
    """
    GET /user-list?path=Server:channel
    
    Returns a list of users in two categories:
    1. Recently active (from message history)
    2. Present but not recently active
    Limited to 100 users total.
    """
    path = request.query.get("path", "").strip()
    if not path:
        return web.json_response({"error": "Path parameter is required"}, status=400)
    
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
        # First get recently active users from history
        recent_users = set()
        recent_user_data = []
        
        async with asyncio.timeout(30.0):
            async for message in channel.history(limit=100):
                if len(recent_users) >= 100:  # Safety limit
                    break
                    
                # Skip if we've already recorded this user
                author_id = str(message.author.id)
                if author_id in recent_users:
                    continue
                    
                recent_users.add(author_id)
                recent_user_data.append({
                    "id": author_id,
                    "name": message.author.name,
                    "display_name": message.author.display_name,
                    "last_active": message.created_at.isoformat()
                })
        
        # Sort recent users alphabetically by display name
        recent_user_data.sort(key=lambda x: x["display_name"].lower())
        
        # Now get all current members of the channel
        channel_users = []
        for member in channel.members:
            # Skip if user was in recent history
            if str(member.id) in recent_users:
                continue
                
            channel_users.append({
                "id": str(member.id),
                "name": member.name,
                "display_name": member.display_name,
                "status": str(member.status) if hasattr(member, "status") else "unknown"
            })
            
            if len(recent_user_data) + len(channel_users) >= 100:
                break
        
        # Sort channel users alphabetically
        channel_users.sort(key=lambda x: x["display_name"].lower())
        
        result = {
            "path": path,
            "channel_id": channel_id,
            "total_users": len(recent_user_data) + len(channel_users),
            "recently_active": recent_user_data,
            "other_members": channel_users
        }
        
        return web.json_response(result)
        
    except asyncio.TimeoutError:
        return web.json_response(
            {"error": "Request timed out while fetching users"}, 
            status=504
        )
    except Exception as e:
        logger.exception(f"[Bot] Error in get_user_list: {e}")
        return web.json_response(
            {"error": f"Failed to fetch user list: {str(e)}"}, 
            status=500
        )


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
    app.router.add_get("/user-list", handle_get_user_list)
    app.router.add_get("/health", lambda _: web.Response(text="OK"))

    return app


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
                if bot.is_ready():
                    await bot.update_name_mappings()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"[Bot] Error updating mappings: {e}")
                await asyncio.sleep(60)  # Retry sooner after error

    mapping_update_task = asyncio.create_task(periodic_mapping_updates())

    history_window_cleanup_task = asyncio.create_task(bot.periodic_cleanup_history_windows())

    # 2) Start the Discord bot
    if not DISCORD_TOKEN:
        logger.error("[Bot] DISCORD_BOT_TOKEN is not set. Exiting.")
        mapping_update_task.cancel()
        history_window_cleanup_task.cancel()
        try:
            await asyncio.gather(mapping_update_task, history_window_cleanup_task, return_exceptions=True)
        except asyncio.CancelledError:
            pass
        await runner.cleanup()
        if persistent_session and not persistent_session.closed:
            await persistent_session.close()
        return

    try:
        await bot.start(DISCORD_TOKEN)
    except KeyboardInterrupt:
        logger.info("[Bot] Received KeyboardInterrupt, shutting down...")
    finally:
        logger.info("[Bot] Closing Discord bot connection...")
        if bot and not bot.is_closed():
            await bot.close()

        logger.info("[Bot] Stopping periodic mapping updates...")
        mapping_update_task.cancel()
        
        logger.info("[Bot] Stopping history window cleanup task...")
        history_window_cleanup_task.cancel()

        try:
            await asyncio.gather(mapping_update_task, history_window_cleanup_task, return_exceptions=True)
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