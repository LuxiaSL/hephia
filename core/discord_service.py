# core/discord_service.py

import asyncio
import aiohttp
import time
from typing import Optional, Dict, Any, List, Tuple, Callable
from dataclasses import dataclass
from enum import Enum

from config import Config
from loggers import SystemLogger


class DiscordServiceStatus(Enum):
    """Track the status of Discord integration."""
    DISABLED = "disabled"
    STARTING = "starting"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    ERROR = "error"
    RECONNECTING = "reconnecting"


@dataclass
class DiscordServiceConfig:
    """Configuration for Discord integration."""
    enabled: bool = False
    bot_url: str = Config.DISCORD_BOT_URL
    max_retries: int = 3
    retry_delay: float = 0.5
    health_check_interval: float = 30.0
    connection_timeout: float = 10.0
    max_queue_size: int = 100
    max_concurrent_requests: int = 5


class DiscordConnectionManager:
    """
    Manages connections to the Discord bot service with resilient
    connection handling and health monitoring.
    """

    def __init__(self, config: DiscordServiceConfig):
        self.config = config
        self.status = (
            DiscordServiceStatus.DISABLED
            if not config.enabled
            else DiscordServiceStatus.STARTING
        )
        self.session: Optional[aiohttp.ClientSession] = None
        self._lock = asyncio.Lock()
        self._health_check_task: Optional[asyncio.Task] = None
        self._last_successful_connection = 0
        self.status_callbacks: List[Callable[[DiscordServiceStatus], None]] = []

    async def initialize(self):
        """Initialize the connection manager if Discord is enabled."""
        if not self.config.enabled:
            SystemLogger.info("Discord integration is disabled, skipping initialization.")
            return
        self._update_status(DiscordServiceStatus.STARTING)
        await self._ensure_session()
        self._health_check_task = asyncio.create_task(self._health_check_loop())

    async def _ensure_session(self) -> Optional[aiohttp.ClientSession]:
        """Ensure a valid session exists, creating a new one if needed."""
        async with self._lock:
            if self.session is None or self.session.closed:
                timeout = aiohttp.ClientTimeout(total=self.config.connection_timeout)
                self.session = aiohttp.ClientSession(timeout=timeout)
            return self.session

    def _update_status(self, new_status: DiscordServiceStatus):
        """Update status and notify callbacks."""
        self.status = new_status
        for callback in self.status_callbacks:
            try:
                callback(new_status)
            except Exception as e:
                SystemLogger.error(f"Error in status callback: {e}")

    async def _health_check_loop(self):
        """Periodically check Discord bot service health."""
        while True:
            try:
                await asyncio.sleep(self.config.health_check_interval)
                await self.check_health()
            except asyncio.CancelledError:
                break
            except Exception as e:
                SystemLogger.error(f"Health check error: {e}")
                await asyncio.sleep(self.config.retry_delay)

    async def check_health(self) -> bool:
        """Check if Discord bot service is healthy by calling /guilds."""
        if not self.config.enabled:
            return False
        try:
            session = await self._ensure_session()
            async with session.get(f"{self.config.bot_url}/guilds") as resp:
                healthy = (resp.status == 200)
                if healthy:
                    self._last_successful_connection = time.time()
                    if self.status != DiscordServiceStatus.CONNECTED:
                        self._update_status(DiscordServiceStatus.CONNECTED)
                else:
                    SystemLogger.warning(f"Discord health check returned status {resp.status}")
                return healthy
        except Exception as e:
            SystemLogger.error(f"Health check failed: {e}")
            if self.status == DiscordServiceStatus.CONNECTED:
                self._update_status(DiscordServiceStatus.RECONNECTING)
            return False

    async def request(
        self,
        method: str,
        endpoint: str,
        retry_count: int = 0,
        **kwargs
    ) -> Optional[aiohttp.ClientResponse]:
        """
        Make a request to the Discord bot service with retry logic.
        Returns the raw ClientResponse or None if error/not found.
        """
        if not self.config.enabled:
            return None
        session = await self._ensure_session()
        url = f"{self.config.bot_url}{endpoint}"

        try:
            async with session.request(method, url, **kwargs) as resp:
                if resp.status == 200:
                    return resp
                elif resp.status == 404:
                    return None
                elif resp.status >= 500 and retry_count < self.config.max_retries:
                    await asyncio.sleep(self.config.retry_delay * (retry_count + 1))
                    return await self.request(method, endpoint, retry_count + 1, **kwargs)
                else:
                    SystemLogger.error(f"Discord request to {url} returned status {resp.status}")
                    return None
        except aiohttp.ClientError as ce:
            SystemLogger.error(f"Client error on {url}: {ce}")
            if retry_count < self.config.max_retries:
                await asyncio.sleep(self.config.retry_delay * (retry_count + 1))
                return await self.request(method, endpoint, retry_count + 1, **kwargs)
        except Exception as e:
            SystemLogger.error(f"Unexpected error on {url}: {e}")

        return None

    async def cleanup(self):
        """Clean up connections and tasks."""
        if self._health_check_task:
            self._health_check_task.cancel()
            try:
                await self._health_check_task
            except asyncio.CancelledError:
                pass

        if self.session and not self.session.closed:
            await self.session.close()


class DiscordRequestQueue:
    """
    Manages queued requests to the Discord bot service with rate limiting.
    Typically used for sending messages but not necessarily for read operations.
    """

    def __init__(self, config: DiscordServiceConfig, connection: DiscordConnectionManager):
        self.config = config
        self.connection = connection
        self._queue = asyncio.PriorityQueue(maxsize=config.max_queue_size)
        self._semaphore = asyncio.Semaphore(config.max_concurrent_requests)
        self._processing_task: Optional[asyncio.Task] = None

    def start(self):
        """Start processing the queue."""
        if not self.config.enabled:
            return
        if not self._processing_task:
            self._processing_task = asyncio.create_task(self._process_queue())

    async def stop(self):
        """Stop queue processing."""
        if self._processing_task:
            self._processing_task.cancel()
            try:
                await self._processing_task
            except asyncio.CancelledError:
                pass

    async def _process_queue(self):
        while True:
            try:
                priority, (method, endpoint, kwargs) = await self._queue.get()
                async with self._semaphore:
                    try:
                        resp = await self.connection.request(method, endpoint, **kwargs)
                        if resp:
                            if resp.status == 200:
                                SystemLogger.debug(f"Successfully processed request to {endpoint}")
                            elif resp.status == 404:
                                SystemLogger.warning(f"Request to {endpoint} returned 404 Not Found")
                            else:
                                SystemLogger.error(f"Request to {endpoint} returned status {resp.status}")
                    except Exception as e:
                        SystemLogger.error(f"Error processing Discord request: {e}")
                    finally:
                        self._queue.task_done()
            except asyncio.CancelledError:
                break
            except Exception as e:
                SystemLogger.error(f"Error in queue processing: {e}")
                await asyncio.sleep(1)

    async def enqueue(self, priority: int, method: str, endpoint: str, **kwargs):
        """
        Enqueue a Discord request at a given priority (lower = higher priority).
        """
        if not self.config.enabled:
            return
        try:
            await self._queue.put((priority, (method, endpoint, kwargs)))
        except asyncio.QueueFull:
            SystemLogger.warning(f"Discord request queue is full. Dropping request to {endpoint}")


class DiscordService:
    """
    Main Discord integration service that coordinates the connection manager
    and a request queue (for sends) plus direct methods for reading data.
    """

    def __init__(self):
        self.config = DiscordServiceConfig(
            enabled=Config.get_discord_enabled(),
            bot_url=Config.DISCORD_BOT_URL,
            max_retries=3,
            retry_delay=0.5,
            health_check_interval=30.0,
            connection_timeout=10.0,
            max_queue_size=100,
            max_concurrent_requests=5
        )
        self.connection = DiscordConnectionManager(self.config)
        self.queue = DiscordRequestQueue(self.config, self.connection)

    async def initialize(self):
        """Initialize Discord integration."""
        if not self.config.enabled:
            return
        await self.connection.initialize()
        self.queue.start()

    async def cleanup(self):
        """Shut down tasks and sessions."""
        if not self.config.enabled:
            return
        await self.queue.stop()
        await self.connection.cleanup()

    # ----------------------------------------------------------------------
    # HELPER: Synchronous-like request that returns (data, status_code)
    # ----------------------------------------------------------------------
    async def _perform_request(
        self,
        method: str,
        endpoint: str,
        **kwargs
    ) -> Tuple[Optional[Any], Optional[int]]:
        """
        Wraps connection.request(...) and:
          - returns (json_data, status_code)
          - returns (None, status_code) if e.g. 404 or request fails
          - logs errors/warnings appropriately
        """
        resp = await self.connection.request(method, endpoint, **kwargs)
        if resp is None:
            SystemLogger.error(f"Failed to make request to {endpoint} with method {method}")
            return None, None

        status = resp.status
        try:
            data = await resp.json()
        except Exception as e:
            SystemLogger.warning(f"Failed to parse JSON response from {endpoint}: {e}")
            data = None

        if status != 200:
            if status == 404:
                SystemLogger.warning(f"Request to {endpoint} returned 404 Not Found")
            else:
                SystemLogger.error(f"Request to {endpoint} returned status {status}")
        elif data is None:
            SystemLogger.warning(f"Request to {endpoint} returned 200 but no parseable data")
        return data, status

    # ----------------------------------------------------------------------
    # EXACT METHODS PARALLELING YOUR ENVIRONMENT
    # ----------------------------------------------------------------------

    async def list_guilds(self) -> Tuple[Optional[List[dict]], Optional[int]]:
        """GET /guilds -> returns ([{id, name, ...}], status_code)."""
        return await self._perform_request("GET", "/guilds")

    async def list_channels(
        self,
        guild_id: str
    ) -> Tuple[Optional[List[dict]], Optional[int]]:
        """GET /guilds/{guild_id}/channels -> returns ([{id, name, ...}], status)."""
        endpoint = f"/guilds/{guild_id}/channels"
        return await self._perform_request("GET", endpoint)

    async def get_message(
        self,
        channel_id: str,
        message_id: str
    ) -> Tuple[Optional[dict], Optional[int]]:
        """GET /channels/{channel_id}/messages/{message_id} -> returns (data, status)."""
        endpoint = f"/channels/{channel_id}/messages/{message_id}"
        return await self._perform_request("GET", endpoint)

    async def get_history(
        self,
        channel_id: str,
        limit: int = 50
    ) -> Tuple[Optional[List[dict]], Optional[int]]:
        """GET /channels/{channel_id}/history?limit=X -> returns ([...], status)."""
        endpoint = f"/channels/{channel_id}/history"
        params = {"limit": str(limit)}
        return await self._perform_request("GET", endpoint, params=params)

    async def send_message(
        self,
        channel_id: str,
        content: str,
        priority: int = 5
    ) -> None:
        """
        Fire-and-forget style. We queue up a POST to /channels/{channel_id}/send_message.
        If you want an immediate response or message_id, you'd do a direct _perform_request
        instead of queueing.
        """
        if not self.config.enabled:
            SystemLogger.warning("Attempted to send a message while Discord integration is disabled.")
            return
        endpoint = f"/channels/{channel_id}/send_message"
        await self.queue.enqueue(
            priority,
            "POST",
            endpoint,
            json={"content": content}
        )

    async def send_message_immediate(
        self,
        channel_id: str,
        content: str
    ) -> Tuple[Optional[dict], Optional[int]]:
        """
        Directly POST to /channels/{channel_id}/send_message, returning (json, status).
        No queue used here, so you can parse the response right away.
        """
        endpoint = f"/channels/{channel_id}/send_message"
        return await self._perform_request("POST", endpoint, json={"content": content})
