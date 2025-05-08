# core/discord_service.py

import asyncio
import json
import aiohttp
import time
from typing import Optional, Any, List, Tuple, Callable
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
    connection_timeout: float = 15.0
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
            async with session.get(f"{self.config.bot_url}/health") as resp:
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
        Wraps an HTTP request with:
        - Session validation and timeout handling
        - Detailed logging of request/response state
        - Enhanced error reporting for path issues
        Returns (json_data, status_code) or (None, status_code) on error.
        """
        full_url = f"{self.config.bot_url}{endpoint}"
        SystemLogger.debug(f"Initiating {method} request to {endpoint} with kwargs: {kwargs}")
        SystemLogger.debug(f"Full URL: {full_url}")
        SystemLogger.debug(f"Connection status: {self.connection.status}")

        try:
            # Ensure session exists and is valid
            if not self.connection.session or self.connection.session.closed:
                SystemLogger.warning("Session was closed or missing; creating new session")
                await self.connection._ensure_session()

            SystemLogger.debug(f"Session state: {'Active' if self.connection.session and not self.connection.session.closed else 'Inactive'}")

            # Make request with timeout handling
            try:
                async with asyncio.timeout(self.config.connection_timeout):
                    async with self.connection.session.request(method, full_url, **kwargs) as resp:
                        status = resp.status
                        SystemLogger.debug(f"Response received | Status: {status} | Headers: {resp.headers}")

                        try:
                            raw_text = await resp.text()
                            SystemLogger.debug(f"Raw response text (first 1000 chars): {raw_text[:1000]}...")

                            if status != 200:
                                if status == 404:
                                    SystemLogger.warning(f"Not Found | Method: {method} | Endpoint: {endpoint}")
                                    return {"error": f"Endpoint not found: {endpoint}"}, status
                                else:
                                    SystemLogger.error(
                                        f"Non-200 status | Code: {status} | Method: {method} | "
                                        f"Endpoint: {endpoint} | Response: {raw_text[:200]}"
                                    )

                            try:
                                data = json.loads(raw_text)
                                SystemLogger.debug(
                                    f"Successfully parsed JSON response | Data keys: "
                                    f"{list(data.keys()) if isinstance(data, dict) else 'array'}"
                                )
                                return data, status
                            except json.JSONDecodeError as e:
                                SystemLogger.error(
                                    f"Invalid JSON | Endpoint: {endpoint} | Status: {status} | "
                                    f"Error: {str(e)} | Position: {e.pos} | "
                                    f"Raw text snippet: {raw_text[:200]}"
                                )
                                return raw_text, status

                        except Exception as e:
                            SystemLogger.error(f"Error processing response: {e}")
                            return None, status

            except asyncio.TimeoutError:
                SystemLogger.error(f"Request timed out after {self.config.connection_timeout}s")
                return None, 504

        except Exception as e:
            SystemLogger.error(
                f"Request failed | Method: {method} | Endpoint: {endpoint} | "
                f"Error: {str(e)} | Connection Status: {self.connection.status}",
                exc_info=True
            )
            return None, None
        
    def _sanitize_path(self, path: str) -> str:
        """
        Sanitize a Discord path to ensure consistent formatting.
        Removes quotes, trims whitespace, etc.
        """
        # Strip whitespace
        path = path.strip()
        
        # Remove surrounding quotes if present
        if (path.startswith('"') and path.endswith('"')) or (path.startswith("'") and path.endswith("'")):
            path = path[1:-1]
            
        return path

    async def list_guilds_with_channels(self) -> Tuple[Optional[List[dict]], Optional[int]]:
        """
        GET /guilds-with-channels -> returns hierarchical data of guilds and their channels
        """
        return await self._perform_request("GET", "/guilds-with-channels")
    
    # Queue-based version of send_message_by_path
    async def queue_message_by_path(
        self,
        path: str,
        content: str,
        priority: int = 5
    ) -> None:
        """
        Fire-and-forget style. Send a message to a path using the queue system.
        Messages longer than 1750 characters are automatically split into multiple messages.
        
        Args:
            path: The target in 'Guild:channel' format (e.g., 'MyServer:general')
            content: Message to send
            priority: Message priority (lower values = higher priority)
        """
        if not self.config.enabled:
            SystemLogger.warning("Attempted to send a message while Discord integration is disabled.")
            return

        # Discord's message length limit
        MAX_LENGTH = 1750
        endpoint = f"/send-by-path"

        path = self._sanitize_path(path)
        SystemLogger.info(f"Queueing message to '{path}' with content length {len(content)}")

        # Split message if needed
        if len(content) <= MAX_LENGTH:
            await self.queue.enqueue(
                priority,
                "POST",
                endpoint,
                json={"path": path, "content": content}
            )
        else:
            # Split content into chunks of MAX_LENGTH while preserving words
            chunks = []
            remaining = content
            while remaining:
                if len(remaining) <= MAX_LENGTH:
                    chunks.append(remaining)
                    break
                
                # Find last space before MAX_LENGTH
                split_point = remaining.rfind(' ', 0, MAX_LENGTH)
                if split_point == -1:  # No space found, force split at MAX_LENGTH
                    split_point = MAX_LENGTH
                
                chunks.append(remaining[:split_point])
                remaining = remaining[split_point:].lstrip()

            # Queue each chunk with increasing priority to maintain order
            for i, chunk in enumerate(chunks):
                # Adjust priority slightly for each chunk to maintain sequence
                chunk_priority = priority + (i * 0.001)
                await self.queue.enqueue(
                    chunk_priority,
                    "POST",
                    endpoint,
                    json={"path": path, "content": chunk}
                )

    async def send_message_by_path(
        self,
        path: str,
        content: str,
        priority: int = 5
    ) -> Tuple[Optional[dict], Optional[int]]:
        """
        Send a message using the friendly 'Server:channel' path format.
        No queue used here, returns immediately with response data.
        """
        # Sanitize the path to ensure it's clean
        path = self._sanitize_path(path)
        SystemLogger.info(f"Sending message to '{path}' with content length {len(content)}")
        
        # Add specific debugging for this method
        try:
            # Construct the request to the bot
            endpoint = "/send-by-path"
            full_url = f"{self.config.bot_url}{endpoint}"
            SystemLogger.debug(f"Making request to: {full_url}")
            SystemLogger.debug(f"Request payload: path='{path}', content_length={len(content)}")
            
            # Make the request with detailed error trapping
            result, status = await self._perform_request("POST", endpoint, json={
                "path": path,
                "content": content
            })
            
            # Log the outcome
            if status == 200:
                SystemLogger.info(f"Successfully sent message to '{path}'")
            else:
                SystemLogger.error(f"Failed to send message to '{path}': status={status}")
                if result:
                    SystemLogger.error(f"Error details: {result}")
            
            return result, status
        except Exception as e:
            SystemLogger.error(f"Exception in send_message_by_path: {e}", exc_info=True)
            return None, 500
    
    async def find_message(
        self,
        path: str,
        reference: str
    ) -> Tuple[Optional[dict], Optional[int]]:
        """Find a specific message using a user-friendly reference."""
        # Sanitize the path
        path = self._sanitize_path(path)
        SystemLogger.info(f"Finding message in '{path}' with reference '{reference}'")
        
        endpoint = "/find-message"
        params = {
            "path": path,
            "reference": reference
        }
        return await self._perform_request("GET", endpoint, params=params)

    async def get_enhanced_history(
        self,
        path: str,
        limit: int = 50
    ) -> Tuple[Optional[dict], Optional[int]]:
        """
        Get enhanced message history with user-friendly references.
        
        Args:
            path: Channel path in 'Server:channel' format
            limit: Maximum number of messages to retrieve
            
        Returns:
            Tuple of (history_data, status_code)
        """
        path = self._sanitize_path(path)
        SystemLogger.info(f"Retrieving enhanced history for '{path}' with limit {limit}")

        endpoint = "/enhanced-history"
        params = {
            "path": path,
            "limit": str(limit)
        }
        return await self._perform_request("GET", endpoint, params=params)

    async def reply_to_message(
        self,
        path: str,
        reference: str,
        content: str
    ) -> Tuple[Optional[dict], Optional[int]]:
        """
        Reply to a specific message identified by a user-friendly reference.
        
        Args:
            path: Channel path in 'Server:channel' format
            reference: Reference to message (e.g., "#1", "latest")
            content: Reply content
            
        Returns:
            Tuple of (response_data, status_code)
        """
        # Sanitize inputs
        path = self._sanitize_path(path)
        reference = reference.strip()
        
        # Log exact values being used
        SystemLogger.info(f"Replying to message: path='{path}', reference='{reference}'")
        
        # Construct request data
        endpoint = "/reply-to-message" 
        payload = {
            "path": path,
            "reference": reference,
            "content": content
        }
        
        # Perform the request with detailed error handling
        try:
            result, status = await self._perform_request("POST", endpoint, json=payload)
            
            if status != 200:
                SystemLogger.error(f"Reply request failed: status={status}, response={result}")
                # Include more information in error
                if isinstance(result, dict) and "error" in result:
                    error_msg = f"Reply failed: {result['error']}"
                else:
                    error_msg = f"Endpoint not found: {endpoint}"
                    if self.config.bot_url:
                        error_msg += f" (full URL: {self.config.bot_url}{endpoint})"
                
                # Return structured error
                return {"error": error_msg}, status
            
            return result, status
            
        except Exception as e:
            SystemLogger.error(f"Exception in reply_to_message: {e}", exc_info=True)
            return {"error": f"Exception: {str(e)}"}, 500
        
    async def get_user_list(
        self,
        path: str
    ) -> Tuple[Optional[List[dict]], Optional[int]]:
        """
        Get a list of active usernames in a specific channel.

        Args:
            path: Channel path in 'Server:channel' format
        
        Returns:
            Tuple of (user_list, status_code)
        """
        path = self._sanitize_path(path)
        SystemLogger.info(f"Retrieving user list for '{path}'")

        endpoint = "/user-list"
        params = {
            "path": path
        }
        return await self._perform_request("GET", endpoint, params=params)

