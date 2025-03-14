import os
import sys
import logging
import asyncio
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
        # We'll store the persistent aiohttp session reference
        self.session = kwargs.get("session", None)

    async def on_ready(self):
        logger.info(f"[Bot] Logged in as {self.user} (id={self.user.id})")
        await initialize_message_cache(self)

    async def on_message(self, message: discord.Message):
        """
        Called whenever a new message is created in Discord.
        Tracks message counts per channel and notifies when thresholds are reached.
        """
        channel_id = str(message.channel.id)
        logger.debug(f"[Bot] New message in {message.channel.name} from {message.author.name}")

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
        
        # 5) Always forward if bot is mentioned
        if self.user.mentioned_in(message):
            logger.info(f"[Bot] Mention detected in channel {message.channel.name}")
            await self.forward_to_hephia(message)
            # new hephia message requested, reset counter for channel
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

###############################################################################
# AIOHTTP SERVER ROUTES: OUTBOUND COMMANDS
###############################################################################

async def handle_list_guilds(request: web.Request) -> web.Response:
    """
    GET /guilds -> Returns a JSON list of guilds the bot is in.
    Example response:
    [
      {"id": "1234567890", "name": "MyGuild"},
      ...
    ]
    """
    guilds = []
    logger.debug("[Bot] Listing guilds")
    for g in bot.guilds:
        guilds.append({"id": str(g.id), "name": g.name})
    logger.debug(f"[Bot] Found {len(guilds)} guilds")
    return web.json_response(guilds)


async def handle_list_channels(request: web.Request) -> web.Response:
    """
    GET /guilds/{guild_id}/channels
    Returns a JSON list of channels in the specified guild.
    Example response:
    [
      {"id": "111111", "name": "#general", "type": "text"},
      ...
    ]
    """
    guild_id = request.match_info["guild_id"]
    guild = bot.get_guild(int(guild_id))
    if not guild:
        return web.json_response({"error": "Guild not found"}, status=404)

    channels_info = []
    logger.debug(f"[Bot] Listing channels for guild {guild.name}")
    for channel in guild.channels:
        channel_type = str(channel.type)  # "text", "voice", "category", etc.
        channels_info.append({
            "id": str(channel.id),
            "name": channel.name,
            "type": channel_type
        })
    logger.debug(f"[Bot] Found {len(channels_info)} channels in guild {guild.name}")
    return web.json_response(channels_info)


async def handle_get_channel_history(request: web.Request) -> web.Response:
    """
    GET /channels/{channel_id}/history?limit=50
    Returns the last N messages from the channel, in chronological order (oldest first).
    Example response:
    [
      {"id": "...", "author": "username#1234", "content": "...", "timestamp": ...},
      ...
    ]
    """
    try:
        channel_id = str(request.match_info["channel_id"])
        logger.debug(f"[Bot] Fetching history for channel {channel_id}")
        limit_str = request.query.get("limit", "50")
        try:
            limit = min(int(limit_str), 100)  # Some safe cap
        except ValueError:
            limit = 50

        try:
            channel = bot.get_channel(int(channel_id))
            if channel is None:
                logger.debug(f"[Bot] Channel {channel_id} not found in cache, attempting fetch_channel")
                channel = await bot.fetch_channel(int(channel_id))
        except discord.NotFound:
            return web.json_response({"error": f"Channel {channel_id} not found"}, status=404)
        except discord.Forbidden:
            return web.json_response({"error": "No permission to access channel"}, status=403)
            
        if not isinstance(channel, discord.TextChannel):
            return web.json_response({"error": "Channel exists but is not a text channel"}, status=404)

        # Fetch up to 'limit' messages, most recent first
        history_data = []
        
        try:
            history_iter = channel.history(limit=limit, oldest_first=False)
            async with asyncio.timeout(30.0):  # 30 seconds timeout
                messages = [msg async for msg in history_iter]
            messages.reverse()  # earliest first
        except asyncio.TimeoutError:
            return web.json_response({"error": "Request timed out while fetching history"}, status=504)

        bot_mention = f'<@{bot.user.id}>'
        bot_mention_bang = f'<@!{bot.user.id}>'

        for msg in messages:
            # Get the proper user reference format
            if hasattr(msg.author, 'global_name'):  # New Discord system
                author_ref = msg.author.name
            else:
                author_ref = f"{msg.author.display_name}#{msg.author.discriminator}"
            
            # Clean up message content by replacing bot mentions
            content = msg.content.replace(bot_mention, f'@{bot.user.name}')
            content = content.replace(bot_mention_bang, f'@{bot.user.name}')
            
            history_data.append({
                "id": str(msg.id),
                "author": author_ref,
                "content": content,
                "timestamp": str(msg.created_at.isoformat())
            })

        bot.new_messages[channel_id] = 0  # Reset new message counter for this channel
        return web.json_response(history_data)

    except Exception as e:
        logger.exception(f"[Bot] Error in get_channel_history: {e}")
        return web.json_response(
            {"error": f"Failed to fetch history: {str(e)}"}, 
            status=500
        )


async def handle_get_message_by_id(request: web.Request) -> web.Response:
    """
    GET /channels/{channel_id}/messages/{message_id}
    Returns a single message by ID, if it exists.
    Example response:
    {
      "id": "987654321", 
      "author": "username#1234",
      "content": "Hello",
      "timestamp": "2025-01-25T12:34:56.789Z"
    }
    """
    channel_id = request.match_info["channel_id"]
    message_id = request.match_info["message_id"]

    channel = bot.get_channel(int(channel_id))
    if not channel or not isinstance(channel, discord.TextChannel):
        return web.json_response({"error": "Channel not found or not a text channel"}, status=404)

    try:
        msg = await channel.fetch_message(int(message_id))
        if msg is None:
            return web.json_response({"error": "Message not found"}, status=404)

        if hasattr(msg.author, 'global_name'):  # New Discord system
            author_ref = msg.author.name
        else:
            author_ref = f"{msg.author.display_name}#{msg.author.discriminator}"

        bot_mention = f'<@{bot.user.id}>'
        bot_mention_bang = f'<@!{bot.user.id}>'
        content = msg.content.replace(bot_mention, f'@{bot.user.name}')
        content = content.replace(bot_mention_bang, f'@{bot.user.name}')

        result = {
            "id": str(msg.id),
            "author": author_ref,
            "content": content,
            "timestamp": str(msg.created_at.isoformat())
        }
        return web.json_response(result)

    except discord.NotFound:
        return web.json_response({"error": "Message not found"}, status=404)
    except Exception as e:
        logger.exception(f"[Bot] Error fetching message: {e}")
        return web.json_response({"error": f"Failed to fetch message: {e}"}, status=500)


async def handle_send_message(request: web.Request) -> web.Response:
    """
    POST /channels/{channel_id}/send_message
    JSON body: { "content": "Hello world!" }
    Returns: { "status": "ok", "message_id": "<ID>" }
    """
    channel_id = request.match_info["channel_id"]
    try:
        data = await request.json()
    except Exception as e:
        error_msg = f"Invalid JSON payload: {e}"
        logger.error(f"[Bot] {error_msg}")
        return web.json_response({"error": error_msg}, status=400)

    content = data.get("content", "").strip()

    # Validate non-empty content before proceeding.
    if not content:
        error_msg = "Message content cannot be empty."
        logger.error(f"[Bot] {error_msg}")
        return web.json_response({"error": error_msg}, status=400)

    channel = bot.get_channel(int(channel_id))
    if not channel or not isinstance(channel, discord.TextChannel):
        return web.json_response({"error": "Channel not found or not a text channel"}, status=404)

    try:
        sent_msg = await channel.send(content)
        logger.info(f"[Bot] Message sent in channel {channel.name} with ID {sent_msg.id}")
        return web.json_response({"status": "ok", "message_id": str(sent_msg.id)})
    except Exception as e:
        logger.exception(f"[Bot] Failed to send message: {e}")
        return web.json_response({"error": f"Failed to send message: {e}"}, status=500)

###############################################################################
# SETTING UP THE AIOHTTP WEB SERVER
###############################################################################

def create_app() -> web.Application:
    """
    Creates an aiohttp Application and registers all routes.
    """
    app = web.Application()

    # Routes for listing servers, channels, etc.
    app.router.add_get("/guilds", handle_list_guilds)
    app.router.add_get("/guilds/{guild_id}/channels", handle_list_channels)
    app.router.add_get("/channels/{channel_id}/history", handle_get_channel_history)
    app.router.add_get("/channels/{channel_id}/messages/{message_id}", handle_get_message_by_id)
    app.router.add_post("/channels/{channel_id}/send_message", handle_send_message)

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
