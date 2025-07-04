"""
Discord Bot - Modular Architecture

This is the main entry point for the Discord bot that provides LLM-friendly
HTTP API endpoints for autonomous Discord interaction.

Key Features:
- Bidirectional format conversion (@john ↔ <@123456>)
- Enhanced message caching with real-time updates
- Context window system for #N message references
- Detailed error reporting with specific error codes
- Clean modular architecture with proper separation of concerns

Architecture:
- bot_client.py: Discord WebSocket client and event coordination
- api_handlers.py: HTTP API endpoints with detailed error reporting
- message_cache.py: Sophisticated caching with real-time updates
- name_mapping.py: Bidirectional ID/name abstraction layer
- context_windows.py: #N reference system for LLM tool usage
- message_logic.py: Engagement decisions and Hephia forwarding
- bot_models.py: Core data structures
- bot_exceptions.py: Error handling hierarchy
"""

import math
import os
import sys
import logging
import asyncio
from typing import Optional, Dict, List
import aiohttp
import discord
from aiohttp import web
from dotenv import load_dotenv
from datetime import datetime

# Import all our modular components
from bot_client import DiscordBotClient
from api_handlers import DiscordAPIHandlers, create_routes
from message_cache import MessageCacheManager
from name_mapping import NameMappingService
from context_windows import ContextWindowManager
from message_logic import MessageProcessor
from bot_models import BotConfig, BotStatus
from bot_exceptions import *

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
logger.setLevel(logging.DEBUG)

# File handler
file_handler = logging.FileHandler(log_filename)
file_handler.setLevel(logging.DEBUG)
file_format = logging.Formatter(
    "[%(asctime)s] %(levelname)s in %(module)s:%(lineno)d: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
file_handler.setFormatter(file_format)
logger.addHandler(file_handler)

# Console handler
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

# Bot configuration
DISCORD_TOKEN = os.getenv("DISCORD_BOT_TOKEN")
HEPHIA_SERVER_URL = os.getenv("HEPHIA_SERVER_URL", "http://localhost:5517")
BOT_HTTP_PORT = int(os.getenv("BOT_HTTP_PORT", "5518"))

# Create bot configuration
bot_config = BotConfig(
    max_message_cache_size=int(os.getenv("MAX_MESSAGE_CACHE_SIZE", "1000")),
    context_window_expiry_minutes=int(os.getenv("CONTEXT_WINDOW_EXPIRY_MINUTES", "5")),
    max_concurrent_requests=int(os.getenv("MAX_CONCURRENT_REQUESTS", "5")),
    enable_debug_logging=os.getenv("ENABLE_DEBUG_LOGGING", "false").lower() == "true",
    high_message_threshold=int(os.getenv("HIGH_MESSAGE_THRESHOLD", "50")),
    random_engagement_rate=float(os.getenv("RANDOM_ENGAGEMENT_RATE", "0.75")),
    cache_cleanup_interval=int(os.getenv("CACHE_CLEANUP_INTERVAL", "300")),
    max_cache_age_hours=int(os.getenv("MAX_CACHE_AGE_HOURS", "24")),
    requests_per_minute=int(os.getenv("REQUESTS_PER_MINUTE", "60")),
    burst_size=int(os.getenv("BURST_SIZE", "10"))
)

# Global references
bot_client: Optional[DiscordBotClient] = None
persistent_session: Optional[aiohttp.ClientSession] = None
api_handlers: Optional[DiscordAPIHandlers] = None

###############################################################################
# DISCORD BOT SERVICE
###############################################################################

class DiscordBotService:
    """
    Main service that coordinates the Discord bot and HTTP API server.
    
    This service manages the lifecycle of all components and provides
    a clean interface for starting and stopping the bot.
    """
    
    def __init__(self, config: BotConfig):
        self.config = config
        self.bot_client: Optional[DiscordBotClient] = None
        self.session: Optional[aiohttp.ClientSession] = None
        self.api_handlers: Optional[DiscordAPIHandlers] = None
        self.web_app: Optional[web.Application] = None
        self.web_runner: Optional[web.AppRunner] = None
        self.web_site: Optional[web.TCPSite] = None
        
        logger.info("Discord bot service initialized")
    
    async def start(self, token: str, port: int):
        """Start the Discord bot service."""
        try:
            logger.info("Starting Discord bot service...")
            
            # Create persistent session
            self.session = aiohttp.ClientSession()
            
            # Create Discord bot client
            self.bot_client = DiscordBotClient(
                config=self.config,
                session=self.session
            )
            
            # Wait for bot to be ready before setting up HTTP server
            logger.info("Starting Discord bot client...")
            bot_task = asyncio.create_task(self.bot_client.start(token))
            
            # Wait for bot to be ready
            while not self.bot_client.is_ready():
                await asyncio.sleep(0.1)
            
            logger.info("Discord bot client ready, setting up HTTP server...")
            
            # Create API handlers
            self.api_handlers = DiscordAPIHandlers(
                bot_client=self.bot_client,
                name_mapping=self.bot_client.name_mapping,
                cache_manager=self.bot_client.cache_manager,
                context_manager=self.bot_client.context_manager
            )
            
            # Set up HTTP server
            await self._setup_http_server(port)
            
            logger.info(f"Discord bot service started successfully on port {port}")
            logger.info(f"Bot connected to {len(self.bot_client.guilds)} guilds")
            
            # Wait for bot to finish
            await bot_task
            
        except Exception as e:
            logger.error(f"Error starting Discord bot service: {e}")
            await self.stop()
            raise
    
    async def _setup_http_server(self, port: int):
        """Set up the HTTP API server."""
        try:
            # Create web application
            self.web_app = web.Application()
            
            # Add routes
            routes = create_routes(self.api_handlers)
            self.web_app.add_routes(routes)
            
            # Add CORS middleware if needed
            self.web_app.middlewares.append(self._cors_middleware)
            
            # Set up runner
            self.web_runner = web.AppRunner(self.web_app)
            await self.web_runner.setup()
            
            # Start server
            self.web_site = web.TCPSite(self.web_runner, "0.0.0.0", port)
            await self.web_site.start()
            
            logger.info(f"HTTP API server started on http://0.0.0.0:{port}")
            
        except Exception as e:
            logger.error(f"Error setting up HTTP server: {e}")
            raise
    
    async def stop(self):
        """Stop the Discord bot service."""
        logger.info("Stopping Discord bot service...")
        
        try:
            # Stop HTTP server
            if self.web_site:
                await self.web_site.stop()
            if self.web_runner:
                await self.web_runner.cleanup()
            
            # Stop Discord bot
            if self.bot_client and not self.bot_client.is_closed():
                await self.bot_client.close()
            
            # Close session
            if self.session and not self.session.closed:
                await self.session.close()
            
            logger.info("Discord bot service stopped")
            
        except Exception as e:
            logger.error(f"Error stopping Discord bot service: {e}")
    
    async def get_status(self) -> Dict[str, any]:
        """Get detailed status of the bot service."""
        if not self.bot_client:
            return {"status": "not_started"}
        
        try:
            bot_status = self.bot_client.get_detailed_stats()
            
            # Add HTTP server status
            bot_status["http_server"] = {
                "running": self.web_site is not None,
                "port": BOT_HTTP_PORT
            }
            
            # Add service-level info
            bot_status["service"] = {
                "config": self.config.__dict__,
                "session_closed": self.session.closed if self.session else True,
                "handlers_initialized": self.api_handlers is not None
            }
            
            return bot_status
            
        except Exception as e:
            logger.error(f"Error getting status: {e}")
            return {"status": "error", "error": str(e)}
    
    @web.middleware
    async def _cors_middleware(self, request, handler):
        """Simple CORS middleware for API requests."""
        response = await handler(request)
        response.headers['Access-Control-Allow-Origin'] = '*'
        response.headers['Access-Control-Allow-Methods'] = 'GET, POST, PUT, DELETE, OPTIONS'
        response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization'
        return response

###############################################################################
# MAIN ENTRY POINT
###############################################################################

async def main():
    """Main entry point for the Discord bot."""
    global bot_client, persistent_session, api_handlers
    
    # Validate configuration
    if not DISCORD_TOKEN:
        logger.error("DISCORD_BOT_TOKEN environment variable not set")
        logger.error("Please set your Discord bot token in the .env file")
        sys.exit(1)
    
    # Log configuration
    logger.info("Starting Discord bot with configuration:")
    logger.info(f"  HTTP Port: {BOT_HTTP_PORT}")
    logger.info(f"  Hephia Server: {HEPHIA_SERVER_URL}")
    logger.info(f"  Cache Size: {bot_config.max_message_cache_size}")
    logger.info(f"  Context Window Expiry: {bot_config.context_window_expiry_minutes} minutes")
    logger.info(f"  Debug Logging: {bot_config.enable_debug_logging}")
    
    # Create and start the service
    service = DiscordBotService(bot_config)
    
    try:
        await service.start(DISCORD_TOKEN, BOT_HTTP_PORT)
        
    except KeyboardInterrupt:
        logger.info("Received KeyboardInterrupt, shutting down...")
        
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        
    finally:
        await service.stop()
        logger.info("Discord bot shutdown complete")

###############################################################################
# DEVELOPMENT AND TESTING UTILITIES
###############################################################################

async def run_health_check():
    """Run a health check against the bot API."""
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(f"http://localhost:{BOT_HTTP_PORT}/health") as resp:
                if resp.status == 200:
                    data = await resp.json()
                    print("✅ Health check passed")
                    print(f"Bot ready: {data.get('data', {}).get('bot_ready', False)}")
                    print(f"Guilds: {data.get('data', {}).get('guild_count', 0)}")
                    return True
                else:
                    print(f"❌ Health check failed: {resp.status}")
                    return False
    except Exception as e:
        print(f"❌ Health check error: {e}")
        return False

async def run_test_message():
    """Send a test message to verify the bot is working."""
    try:
        async with aiohttp.ClientSession() as session:
            # First get available guilds
            async with session.get(f"http://localhost:{BOT_HTTP_PORT}/guilds-with-channels") as resp:
                if resp.status != 200:
                    print("❌ Failed to get guilds")
                    return False
                
                guilds = await resp.json()
                if not guilds.get('data'):
                    print("❌ No guilds available")
                    return False
                
                # Use first available channel
                first_guild = guilds['data'][0]
                if not first_guild.get('channels'):
                    print("❌ No channels available")
                    return False
                
                channel_path = first_guild['channels'][0]['path']
                
                # Send test message
                test_data = {
                    "path": channel_path,
                    "content": "🤖 Test message from modular Discord bot!"
                }
                
                async with session.post(
                    f"http://localhost:{BOT_HTTP_PORT}/send-by-path",
                    json=test_data
                ) as resp:
                    if resp.status == 200:
                        print(f"✅ Test message sent to {channel_path}")
                        return True
                    else:
                        error = await resp.text()
                        print(f"❌ Test message failed: {resp.status} - {error}")
                        return False
                        
    except Exception as e:
        print(f"❌ Test message error: {e}")
        return False

###############################################################################
# CLI INTERFACE
###############################################################################

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Discord Bot with Modular Architecture")
    parser.add_argument("--health-check", action="store_true", help="Run health check")
    parser.add_argument("--test-message", action="store_true", help="Send test message")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    
    args = parser.parse_args()
    
    # Update debug logging if requested
    if args.debug:
        bot_config.enable_debug_logging = True
        logger.setLevel(logging.DEBUG)
        console_handler.setLevel(logging.DEBUG)
    
    # Handle CLI commands
    if args.health_check:
        print("Running health check...")
        result = asyncio.run(run_health_check())
        sys.exit(0 if result else 1)
    
    elif args.test_message:
        print("Sending test message...")
        result = asyncio.run(run_test_message())
        sys.exit(0 if result else 1)
    
    else:
        # Run the main bot
        try:
            asyncio.run(main())
        except KeyboardInterrupt:
            logger.info("Shutdown complete")
        except Exception as e:
            logger.error(f"Fatal error: {e}")
            sys.exit(1)