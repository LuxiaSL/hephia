"""
api_clients.py - Centralized API communication management for Hephia.

Provides unified interfaces for multiple API services while maintaining
provider-specific optimizations and requirements.
"""

import platform
import aiohttp
from aiohttp import UnixConnector, TCPConnector
import os
from typing import Dict, Any, List, Optional, Union
import json
import asyncio
from abc import ABC, abstractmethod

from loggers import SystemLogger, PromptLogger
from config import Config


class BaseAPIClient(ABC):
    """Enhanced base class for API clients with robust error handling."""
    
    def __init__(self, api_key: str, base_url: str, service_name: str):
        self.api_key = api_key
        self.base_url = base_url
        self.service_name = service_name
        self.max_retries = 3
        self.base_retry_delay = 1  # seconds
        self.max_retry_delay = 32  # Maximum delay after exponential backoff
        self.read_timeout_multiplier = 1.5  # multiplier for socket read timeout
        
    async def _make_request(
        self,
        endpoint: str,
        method: str = "POST",
        payload: Optional[Dict] = None,
        extra_headers: Optional[Dict] = None,
        timeout: float = 30.0
    ) -> Dict[str, Any]:
        """Make API request with enhanced retry logic and error handling."""
        headers = self._get_headers(extra_headers)
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        last_exception = None

        # Log prompt before making the request
        if (payload and "messages" in payload) and Config.get_log_prompts():
            temp_messages = payload["messages"]
            if self.service_name == "Anthropic":
                # Anthropic uses a different format for system messages
                temp_messages = [
                    {"role": "system", "content": payload["system"]},
                ] + temp_messages

            PromptLogger.log_prompt(
                service=self.service_name,
                messages=temp_messages,
                model=payload.get("model", "unknown"),
                metadata={
                    "temperature": payload.get("temperature"),
                    "max_tokens": payload.get("max_tokens"),
                    "endpoint": endpoint
                }
            )

        for attempt in range(self.max_retries):
            try:
                # Increase overall timeout for each retry attempt
                current_timeout = timeout * (1 + (attempt * 0.5))
                client_timeout = aiohttp.ClientTimeout(
                    total=current_timeout,
                    connect=current_timeout / 3,
                    sock_read=current_timeout * self.read_timeout_multiplier
                )
                # Using TCP keepalive to help maintain the connection
                SystemLogger.debug(
                    f"[{self.service_name}] Attempt {attempt + 1}: Starting request to {endpoint}\n"
                    f"Timeout settings: {client_timeout}"
                )
                tcp_connector = TCPConnector(
                    keepalive_timeout=60,
                    force_close=False,
                    enable_cleanup_closed=True
                )
                async with aiohttp.ClientSession(
                    connector=tcp_connector,
                    timeout=client_timeout
                ) as session:
                    try:
                        async with session.request(
                            method,
                            url,
                            headers=headers,
                            json=payload
                        ) as response:
                            # First try to read the raw bytes
                            try:
                                raw_bytes = await response.read()
                                SystemLogger.debug(
                                    f"[{self.service_name}] Raw response received: {len(raw_bytes)} bytes"
                                )
                            except aiohttp.ClientPayloadError as e:
                                raise Exception(f"Failed to read response payload: {str(e)}")

                            # Then try to decode as text
                            try:
                                response_text = raw_bytes.decode('utf-8')
                                SystemLogger.debug(
                                    f"[{self.service_name}] Decoded response length: {len(response_text)}"
                                )
                            except UnicodeDecodeError as e:
                                raise Exception(f"Failed to decode response as UTF-8: {str(e)}")

                            # Log response details before processing
                            SystemLogger.debug(
                                f"[{self.service_name}] Response details:\n"
                                f"Status: {response.status}\n"
                                f"Headers: {dict(response.headers)}\n"
                                f"Content-Length: {response.headers.get('Content-Length')}\n"
                                f"Transfer-Encoding: {response.headers.get('Transfer-Encoding')}\n"
                                f"Connection: {response.headers.get('Connection')}"
                            )

                            if response.status == 200:
                                try:
                                    response_data = json.loads(response_text)
                                    SystemLogger.log_api_request(
                                        self.service_name,
                                        endpoint,
                                        response.status
                                    )
                                    return response_data
                                except json.JSONDecodeError as e:
                                    error_msg = (
                                        f"JSON decode failed for {self.service_name}:\n"
                                        f"Error: {str(e)}\n"
                                        f"Response preview: {response_text[:200]}..."
                                    )
                                    SystemLogger.error(error_msg)
                                    raise Exception(error_msg)

                            if response.status == 429:
                                retry_after = int(response.headers.get('Retry-After', current_timeout))
                                SystemLogger.log_api_retry(
                                    self.service_name,
                                    attempt + 1,
                                    self.max_retries,
                                    f"Rate limited, waiting {retry_after}s"
                                )
                                await asyncio.sleep(retry_after)
                                continue

                            if response.status >= 500:
                                delay = min(self.base_retry_delay * (2 ** attempt), self.max_retry_delay)
                                SystemLogger.log_api_retry(
                                    self.service_name,
                                    attempt + 1,
                                    self.max_retries,
                                    f"Server error {response.status}, waiting {delay}s"
                                )
                                await asyncio.sleep(delay)
                                continue

                            error_msg = (
                                f"API error ({self.service_name}): Status {response.status}\n"
                                f"Response: {response_text}"
                            )
                            raise Exception(error_msg)

                    except aiohttp.ClientResponseError as e:
                        raise Exception(f"Response error: {str(e)}")
                    except aiohttp.ClientConnectionError as e:
                        raise Exception(f"Connection error: {str(e)}")

            except asyncio.TimeoutError as e:
                last_exception = e
                SystemLogger.warning(
                    f"Timeout on attempt {attempt + 1}/{self.max_retries} ({self.service_name}):\n"
                    f"Type: TimeoutError\n"
                    f"Error: {str(e)}"
                )
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(self.base_retry_delay * (2 ** attempt))
                    continue

            except Exception as e:
                last_exception = e
                SystemLogger.error(
                    f"Error on attempt {attempt + 1}/{self.max_retries} ({self.service_name}):\n"
                    f"Type: {type(e).__name__}\n"
                    f"Error: {str(e)}"
                )
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(self.base_retry_delay * (2 ** attempt))
                    continue

        error_msg = f"All retries failed for {self.service_name}: {str(last_exception)}"
        SystemLogger.error(error_msg)
        raise Exception(error_msg)
    
    @abstractmethod
    def _get_headers(self, extra_headers: Optional[Dict] = None) -> Dict[str, str]:
        """Get headers specific to this provider."""
        pass
    
    @abstractmethod
    def _extract_message_content(self, response: Dict[str, Any]) -> str:
        """Extract message content from provider-specific response format."""
        pass


class OpenAIClient(BaseAPIClient):
    """Client for OpenAI API interactions."""
    
    def __init__(self, api_key: str):
        super().__init__(
            api_key=api_key,
            base_url="https://api.openai.com/v1",
            service_name="OpenAI"
        )
    
    def _get_headers(self, extra_headers: Optional[Dict] = None) -> Dict[str, str]:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        if extra_headers:
            headers.update(extra_headers)
        return headers

    def _extract_message_content(self, response: Dict[str, Any]) -> str:
        return response["choices"][0]["message"]["content"]

    async def create_completion(
        self,
        messages: List[Dict[str, str]],
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 150,
        return_content_only: bool = False
    ) -> Union[Dict[str, Any], str]:
        """Create chat completion via OpenAI."""
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        response = await self._make_request("chat/completions", payload=payload)
        return self._extract_message_content(response) if return_content_only else response


class AnthropicClient(BaseAPIClient):
    """Client for Anthropic API interactions."""
    
    def __init__(self, api_key: str):
        super().__init__(
            api_key=api_key,
            base_url="https://api.anthropic.com/v1",
            service_name="Anthropic"
        )
    
    def _get_headers(self, extra_headers: Optional[Dict] = None) -> Dict[str, str]:
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json"
        }
        if extra_headers:
            headers.update(extra_headers)
        return headers

    def _extract_message_content(self, response: Dict[str, Any]) -> str:
        return response["content"][0]["text"]

    async def create_completion(
        self,
        messages: List[Dict[str, str]],
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 150,
        return_content_only: bool = False
    ) -> Union[Dict[str, Any], str]:
        """Create chat completion via Anthropic."""
        # Combine all system messages if present
        system_messages = [msg["content"] for msg in messages if msg["role"] == "system"]
        combined_system = " ".join(system_messages) if system_messages else None
        
        payload = {
            "model": model,
            "messages": [m for m in messages if m["role"] != "system"],
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        if combined_system:
            payload["system"] = combined_system

        response = await self._make_request("messages", payload=payload)
        return self._extract_message_content(response) if return_content_only else response


class GoogleClient(BaseAPIClient):
    """Client for Google AI interactions."""
    
    def __init__(self, api_key: str):
        super().__init__(
            api_key=api_key,
            base_url="https://generativelanguage.googleapis.com/v1",
            service_name="Google"
        )
    
    def _get_headers(self, extra_headers: Optional[Dict] = None) -> Dict[str, str]:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        if extra_headers:
            headers.update(extra_headers)
        return headers

    def _extract_message_content(self, response: Dict[str, Any]) -> str:
        return response["candidates"][0]["content"]["parts"][0]["text"]

    async def create_completion(
        self,
        messages: List[Dict[str, str]],
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 150,
        return_content_only: bool = False
    ) -> Union[Dict[str, Any], str]:
        """Create chat completion via Google."""
        formatted_messages = [{
            "role": msg["role"],
            "parts": [{"text": msg["content"]}]
        } for msg in messages]
        
        payload = {
            "messages": formatted_messages,
            "temperature": temperature,
            "maxOutputTokens": max_tokens
        }
        
        response = await self._make_request(
            f"models/{model}:generateContent",
            payload=payload
        )
        return self._extract_message_content(response) if return_content_only else response


class OpenRouterClient(BaseAPIClient):
    """Client for OpenRouter API interactions."""
    
    def __init__(self, api_key: str):
        super().__init__(
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1",
            service_name="OpenRouter"
        )
    
    def _get_headers(self, extra_headers: Optional[Dict] = None) -> Dict[str, str]:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://localhost:5517",
            "X-Title": "Hephia"
        }
        if extra_headers:
            headers.update(extra_headers)
        return headers

    def _extract_message_content(self, response: Dict[str, Any]) -> str:
        return response["choices"][0]["message"]["content"]

    async def create_completion(
        self,
        messages: List[Dict[str, str]],
        model: str = "openai/gpt-3.5-turbo",
        temperature: float = 0.7,
        max_tokens: int = 150,
        return_content_only: bool = False
    ) -> Union[Dict[str, Any], str]:
        """Create chat completion via OpenRouter."""
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        
        response = await self._make_request("chat/completions", payload=payload)
        return self._extract_message_content(response) if return_content_only else response


class OpenPipeClient(BaseAPIClient):
    """Client for OpenPipe API interactions"""
    
    def __init__(self, api_key: str):
        super().__init__(
            api_key=api_key,
            base_url="https://api.openpipe.ai/api/v1",
            service_name="OpenPipe"
        )
    
    def _get_headers(self, extra_headers: Optional[Dict] = None) -> Dict[str, str]:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        if extra_headers:
            headers.update(extra_headers)
        return headers

    def _extract_message_content(self, response: Dict[str, Any]) -> str:
        return response["choices"][0]["message"]["content"]

    async def create_completion(
        self,
        messages: List[Dict[str, str]],
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 125,
        return_content_only: bool = False
    ) -> Union[Dict[str, Any], str]:
        """Create chat completion via OpenPipe."""
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        
        response = await self._make_request("chat/completions", payload=payload)
        return self._extract_message_content(response) if return_content_only else response


class PerplexityClient(BaseAPIClient):
    """Client for Perplexity API interactions."""
    
    def __init__(self, api_key: str):
        super().__init__(
            api_key=api_key,
            base_url="https://api.perplexity.ai",
            service_name="Perplexity"
        )
    
    def _get_headers(self, extra_headers: Optional[Dict] = None) -> Dict[str, str]:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        if extra_headers:
            headers.update(extra_headers)
        return headers
    
    def _extract_message_content(self, response: Dict[str, Any]) -> str:
        return response["choices"][0]["message"]["content"]

    async def create_completion(
        self,
        messages: List[Dict[str, str]],
        model: str = "llama-3.1-sonar-small-128k-online",
        temperature: float = 0.7,
        max_tokens: int = 400,
        return_content_only: bool = False
    ) -> Union[Dict[str, Any], str]:
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        response = await self._make_request("chat/completions", payload=payload)
        return self._extract_message_content(response) if return_content_only else response


class UnixSocketClient(BaseAPIClient):
    """Base client for Unix socket communication."""
    
    def __init__(self, socket_path: str, service_name: str):
        super().__init__(
            api_key="N/A",
            base_url="http://localhost",  # Hostname is irrelevant for Unix sockets
            service_name=service_name
        )
        self.socket_path = socket_path
        self.max_retries = 10
    
    def _get_headers(self, extra_headers: Optional[Dict] = None) -> Dict[str, str]:
        """Get headers for Unix socket requests"""
        headers = {"Content-Type": "application/json"}
        if extra_headers:
            headers.update(extra_headers)
        return headers

    def _extract_message_content(self, response: Dict[str, Any]) -> str:
        """Extract message from Unix socket response"""
        return response["choices"][0]["message"]["content"]
        
    async def _make_request(
        self,
        endpoint: str,
        method: str = "POST", 
        payload: Optional[Dict] = None, 
        extra_headers: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """Make request via Unix socket with retry logic."""
        headers = self._get_headers(extra_headers)
        
        for attempt in range(self.max_retries):
            try:
                connector = UnixConnector(path=self.socket_path)
                async with aiohttp.ClientSession(connector=connector) as session:
                    url = f"http://localhost/{endpoint.lstrip('/')}"
                    async with session.request(
                        method,
                        url,
                        headers=headers,
                        json=payload
                    ) as response:
                        if response.status == 200:
                            SystemLogger.log_api_request(
                                self.service_name, 
                                endpoint,
                                response.status
                            )
                            return await response.json()
                        
                        error_text = await response.text()
                        SystemLogger.log_api_request(
                            self.service_name,
                            endpoint,
                            response.status,
                            error_text
                        )
                        
                        if response.status >= 500:
                            SystemLogger.log_api_retry(
                                self.service_name,
                                attempt + 1,
                                self.max_retries,
                                f"Server error"
                            )
                            await asyncio.sleep(0.2)  # 200ms backoff
                            continue
                            
                        raise Exception(f"Unix socket error: Status {response.status}")
                        
            except Exception as e:
                SystemLogger.log_api_retry(
                    self.service_name,
                    attempt + 1, 
                    self.max_retries,
                    str(e)
                )
                await asyncio.sleep(0.2)  # 200ms backoff
                if attempt == self.max_retries - 1:
                    raise


class Chapter2Client(BaseAPIClient):
    """Client for Chapter2 API supporting both HTTP and Unix socket."""
    
    def __init__(self, socket_path: str = None, http_port: int = None):
        self.is_unix = platform.system() != "Windows"
        self.socket_path = socket_path or Config.get_chapter2_socket_path()
        self.http_port = http_port or Config.get_chapter2_http_port()
        
        # Initialize base class with HTTP configuration first
        super().__init__(
            api_key="N/A", 
            base_url=f"http://localhost:{self.http_port}/v1",
            service_name="Chapter2"
        )
        
        # Then check if we should use Unix socket
        self.use_unix = self.is_unix and os.path.exists(self.socket_path)
        if self.use_unix:
            self.unix_client = UnixSocketClient(self.socket_path, "Chapter2")
        else:
            self.unix_client = None
    
    def _get_headers(self, extra_headers: Optional[Dict] = None) -> Dict[str, str]:
        """Get headers for HTTP requests"""
        headers = {"Content-Type": "application/json"}
        if extra_headers:
            headers.update(extra_headers)
        return headers

    def _extract_message_content(self, response: Dict[str, Any]) -> str:
        """Extract message content from response"""
        return response["choices"][0]["message"]["content"]
    
    async def create_completion(
        self,
        messages: List[Dict[str, str]], 
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 150,
        return_content_only: bool = False,
        **kwargs
    ) -> Union[Dict[str, Any], str]:
        """Route completion request to appropriate client."""
        # Add name: system for user messages
        formatted_messages = [
            {**msg, "name": "system"} if msg.get("role") == "user" else msg 
            for msg in messages
        ]
        
        if self.use_unix:
            response = await self.unix_client._make_request(
                "v1/chat/completions",
                payload={
                    "model": model,
                    "messages": formatted_messages,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    **kwargs
                }
            )
            return self._extract_message_content(response) if return_content_only else response
            
        # For HTTP, simply use the base class _make_request
        return await super().create_completion(
            messages=formatted_messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            return_content_only=return_content_only,
            **kwargs
        )


class APIManager:
    """
    Central manager for all API clients.
    Handles initialization and provides access to different services.
    """
    
    def __init__(
        self,
        openai_key: Optional[str] = None,
        anthropic_key: Optional[str] = None,
        google_key: Optional[str] = None,
        openrouter_key: Optional[str] = None,
        openpipe_key: Optional[str] = None,
        perplexity_key: Optional[str] = None
    ):
        self.clients = {}
        if openai_key:
            self.clients["openai"] = OpenAIClient(openai_key)
        if anthropic_key:
            self.clients["anthropic"] = AnthropicClient(anthropic_key)
        if google_key:
            self.clients["google"] = GoogleClient(google_key)
        if openrouter_key:
            self.clients["openrouter"] = OpenRouterClient(openrouter_key)
        if openpipe_key:
            self.clients["openpipe"] = OpenPipeClient(openpipe_key)
        if perplexity_key:
            self.clients["perplexity"] = PerplexityClient(perplexity_key)
        self.clients["chapter2"] = Chapter2Client()
    
    @classmethod
    def from_env(cls):
        """Create APIManager from environment variables."""
        import os
        return cls(
            openai_key=os.getenv("OPENAI_API_KEY"),
            anthropic_key=os.getenv("ANTHROPIC_API_KEY"),
            google_key=os.getenv("GOOGLE_API_KEY"),
            openrouter_key=os.getenv("OPENROUTER_API_KEY"),
            openpipe_key=os.getenv("OPENPIPE_API_KEY"),
            perplexity_key=os.getenv("PERPLEXITY_API_KEY")
        )
    
    def get_client(self, provider: str) -> BaseAPIClient:
        """Get specific client by provider name."""
        if provider not in self.clients:
            raise ValueError(f"Unknown provider: {provider}")
        return self.clients[provider]
    
    async def create_completion(
        self,
        provider: str,
        messages: List[Dict[str, str]],
        **kwargs
    ) -> Union[Dict[str, Any], str]:
        """Create completion using specified provider."""
        if provider not in self.clients:
            raise ValueError(f"Unknown provider: {provider}")
        return await self.clients[provider].create_completion(messages, **kwargs)
