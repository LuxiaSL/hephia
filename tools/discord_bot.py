import os
import discord
import asyncio
import aiohttp

from aiohttp import web
from dotenv import load_dotenv
from typing import Optional, List, Dict, Any

###############################################################################
# CONFIGURATION
###############################################################################

load_dotenv()
DISCORD_TOKEN = os.getenv("DISCORD_BOT_TOKEN")  # Your bot's token
HEPHIA_SERVER_URL = "http://localhost:8000"     # Where Hephia server listens
BOT_HTTP_PORT = 9001                            # Port where *this* bot listens

# For convenience, define a global reference to the bot, so HTTP handlers
# can access it easily. We could also pass the bot object around, or attach
# it to the app object.
bot: "RealTimeDiscordBot" = None  

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

    async def on_ready(self):
        print(f"[Bot] Logged in as {self.user} (id={self.user.id})")
        await initialize_message_cache(self)

    async def on_message(self, message: discord.Message):
        """
        Called whenever a new message is created in Discord.
        Tracks message counts per channel and notifies when thresholds are reached.
        """
        print(f"got a new message from {message.author.name}")
        # 1) Initialize channel counter if it doesn't exist
        channel_id = str(message.channel.id)
        if channel_id not in self.new_messages:
            self.new_messages[channel_id] = 0

        # 2) Ignore messages from ourselves or system messages
        if message.author == self.user:
            self.new_messages[str(message.channel.id)] = 0
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
            print("got a mention")
            await self.forward_to_hephia(message)
            # new hephia message requested, reset counter for channel
            self.new_messages[channel_id] = 0

    async def notify_high_message_count(self, channel: discord.TextChannel, count: int):
        """
        Notifies Hephia server about high message count in a channel.
        """
        async with aiohttp.ClientSession() as session:
            url = f"{HEPHIA_SERVER_URL}/discord_channel_update"
            data = {
                "channel_id": str(channel.id),
                "new_message_count": count,
                "channel_name": channel.name
            }
            try:
                await session.post(url, json=data)
            except Exception as e:
                print(f"[Bot] Failed to notify about high message count: {e}")

    async def forward_to_hephia(self, message: discord.Message):
        """
        Forwards a Discord message to Hephia's /discord_inbound endpoint.
        Includes recent message history for context.
        """
        print(f"[Bot] Forwarding message from {message.author.name} in {message.channel.name}")

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

        # Get recent message history for context (last 50 messages)
        try:
            print(f"[Bot] Fetching history before message {message.id}")
            history = []
            message_count = 0
            
            # First try to get messages from before the current message
            async for hist_msg in message.channel.history(limit=50, before=message):
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
            
            print(f"[Bot] Found {message_count} messages in history")
            history.reverse()  # Oldest first
            
        except Exception as e:
            print(f"[Bot] Warning: Could not fetch message history: {e}")
            history = []

        channel_id = str(message.channel.id)
        current_count = self.new_messages.get(channel_id, 0)
        print(f"[Bot] Channel {message.channel.name} has {current_count} new messages")

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

        print(f"[Bot] Sending data to Hephia with {len(history)} context messages")
        async with aiohttp.ClientSession() as session:
            url = f"{HEPHIA_SERVER_URL}/discord_inbound" 
            try:
                resp = await session.post(url, json=inbound_data)
                if resp.status != 200:
                    err = await resp.text()
                    print(f"[Bot] Error forwarding to Hephia: {resp.status} {err}")
                else:
                    print("[Bot] Successfully forwarded message to Hephia")
            except Exception as e:
                print(f"[Bot] Failed to forward message to Hephia: {e}")

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
    for g in bot.guilds:
        guilds.append({"id": str(g.id), "name": g.name})
    return web.json_response(guilds)


async def handle_list_channels(request: web.Request) -> web.Response:
    """
    GET /guilds/{guild_id}/channels
    Returns a JSON list of channels in the specified guild.
    Example response:
    [
      {"id": "111111", "name": "#general", "type": "text"},
      {"id": "222222", "name": "#random", "type": "text"},
      ...
    ]
    """
    guild_id = request.match_info["guild_id"]
    guild = bot.get_guild(int(guild_id))
    if not guild:
        return web.json_response({"error": "Guild not found"}, status=404)

    channels_info = []
    for channel in guild.channels:
        # You can further filter only text channels, or include categories, voice, etc.
        channel_type = str(channel.type)  # "text", "voice", "category", etc.
        channels_info.append({
            "id": str(channel.id),
            "name": channel.name,
            "type": channel_type
        })

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
        print(f"[Bot] Fetching history for channel {channel_id}")
        limit_str = request.query.get("limit", "50")
        try:
            limit = min(int(limit_str), 100)  # Some safe cap
        except ValueError:
            limit = 50

        try:
            channel = bot.get_channel(int(channel_id))
            if channel is None:
                print(f"[Bot] Channel {channel_id} not found in cache, attempting fetch")
                channel = await bot.fetch_channel(int(channel_id))
        except discord.NotFound:
            return web.json_response({"error": f"Channel {channel_id} not found"}, status=404)
        except discord.Forbidden:
            return web.json_response({"error": "No permission to access channel"}, status=403)
            
        if not isinstance(channel, discord.TextChannel):
            return web.json_response({"error": "Channel exists but is not a text channel"}, status=404)

        # Fetch up to 'limit' messages, most recent first
        history_data = []
        
        # WARNING: This calls the REST API if not cached. Consider caching if you do this a lot.
        try:
            history_iter = channel.history(limit=limit, oldest_first=False)
            async with asyncio.timeout(30.0):  # 30 seconds timeout
                messages = [msg async for msg in history_iter]
            # We'll reverse them so the earliest is first
            messages.reverse()
        except asyncio.TimeoutError:
            return web.json_response({"error": "Request timed out while fetching history"}, status=504)

        # Bot mention patterns
        bot_mention = f'<@{bot.user.id}>'
        bot_mention_bang = f'<@!{bot.user.id}>'

        for msg in messages:
            # Get the proper user reference format
            if hasattr(msg.author, 'global_name'):  # New Discord system
                author_ref = msg.author.name  # Uses unique username
            else:  # Legacy Discord system
                author_ref = f"{msg.author.display_name}#{msg.author.discriminator}"
            
            # Clean up message content by replacing bot mentions with friendly format
            content = msg.content
            content = content.replace(bot_mention, f'@{bot.user.name}')
            content = content.replace(bot_mention_bang, f'@{bot.user.name}')
            
            # Convert each message to a dict
            history_data.append({
                "id": str(msg.id),
                "author": author_ref,
                "content": content,
                "timestamp": str(msg.created_at.isoformat())
            })

        bot.new_messages[channel_id] = 0  # Reset new message counter for this channel
        return web.json_response(history_data)

    except Exception as e:
        print(f"[Bot] Error in get_channel_history: {e}")
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

        # Get the proper user reference format
        if hasattr(msg.author, 'global_name'):  # New Discord system
            author_ref = msg.author.name  # Uses unique username
        else:  # Legacy Discord system
            author_ref = f"{msg.author.display_name}#{msg.author.discriminator}"

        # Clean up message content by replacing bot mentions with friendly format
        content = msg.content
        bot_mention = f'<@{bot.user.id}>'
        bot_mention_bang = f'<@!{bot.user.id}>'
        content = content.replace(bot_mention, f'@{bot.user.name}')
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
        return web.json_response({"error": f"Failed to fetch message: {e}"}, status=500)


async def handle_send_message(request: web.Request) -> web.Response:
    """
    POST /channels/{channel_id}/send_message
    JSON body: { "content": "Hello world!" }
    Returns: { "status": "ok", "message_id": "<ID>" }
    """
    channel_id = request.match_info["channel_id"]
    data = await request.json()
    content = data.get("content", "")

    channel = bot.get_channel(int(channel_id))
    if not channel or not isinstance(channel, discord.TextChannel):
        return web.json_response({"error": "Channel not found or not a text channel"}, status=404)

    try:
        sent_msg = await channel.send(content)
        return web.json_response({"status": "ok", "message_id": str(sent_msg.id)})
    except Exception as e:
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
# MAIN ENTRY POINT
###############################################################################

async def initialize_message_cache(bot: RealTimeDiscordBot, messages_per_channel: int = 100, 
                                delay_between_channels: float = 1.0):
    """
    Pre-fetches message history for all accessible channels.
    Uses delays to be nice to Discord's API.
    
    Args:
        bot: The Discord bot instance
        messages_per_channel: How many messages to fetch per channel
        delay_between_channels: Delay in seconds between channel fetches
    """
    print("[Bot] Starting message cache initialization...")
    
    for guild in bot.guilds:
        for channel in guild.channels:
            if not isinstance(channel, discord.TextChannel):
                continue
                
            try:
                print(f"[Bot] Fetching history for {guild.name}/{channel.name}")
                messages = [msg async for msg in channel.history(limit=messages_per_channel)]
                # Store message count for this channel
                bot.new_messages[str(channel.id)] = len(messages)
                print(f"[Bot] Cached {len(messages)} messages from {channel.name}")
                
                # Be nice to Discord API
                await asyncio.sleep(delay_between_channels)
                
            except Exception as e:
                print(f"[Bot] Failed to fetch history for {channel.name}: {e}")
                continue
    
    print("[Bot] Message cache initialization complete")

async def main():
    global bot
    bot = RealTimeDiscordBot(intents=discord.Intents.all())

    # 1) Start the aiohttp server in a background task
    app = create_app()
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "0.0.0.0", BOT_HTTP_PORT)
    await site.start()
    print(f"[HTTP] Aiohttp server running at http://0.0.0.0:{BOT_HTTP_PORT}")

    # 2) Start the Discord bot
    if not DISCORD_TOKEN:
        print("[Bot] ERROR: DISCORD_BOT_TOKEN is not set")
        return

    try:
        # 3) Connect to Discord and run forever
        await bot.start(DISCORD_TOKEN)
    except KeyboardInterrupt:
        await bot.close()
    finally:
        await runner.cleanup()

if __name__ == "__main__":
    asyncio.run(main())
