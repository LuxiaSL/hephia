"""
api_clients.py - Centralized API communication management for Hephia.

Provides unified interfaces for multiple API services while maintaining
provider-specific optimizations and requirements.
"""

import aiohttp
from typing import Dict, Any, List, Optional, Union
import json
import asyncio
from abc import ABC, abstractmethod
from loggers import SystemLogger

class BaseAPIClient(ABC):
    """Base class for API clients with common functionality."""
    
    def __init__(self, api_key: str, base_url: str, service_name: str):
        self.api_key = api_key
        self.base_url = base_url
        self.service_name = service_name
        self.max_retries = 3
        self.retry_delay = 1  # seconds
    
    async def _make_request(
        self,
        endpoint: str,
        method: str = "POST",
        payload: Optional[Dict] = None,
        extra_headers: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """Make API request with retry logic and error handling."""
        headers = self._get_headers(extra_headers)
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        
        for attempt in range(self.max_retries):
            try:
                async with aiohttp.ClientSession() as session:
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
                        
                        if response.status == 429:  # Rate limit
                            retry_after = int(response.headers.get('Retry-After', self.retry_delay))
                            SystemLogger.log_api_retry(
                                self.service_name,
                                attempt + 1,
                                self.max_retries,
                                f"Rate limited, waiting {retry_after}s"
                            )
                            await asyncio.sleep(retry_after)
                            continue
                            
                        if response.status >= 500:  # Server error, retry
                            delay = self.retry_delay * (attempt + 1)
                            SystemLogger.log_api_retry(
                                self.service_name,
                                attempt + 1,
                                self.max_retries,
                                f"Server error, waiting {delay}s"
                            )
                            await asyncio.sleep(delay)
                            continue
                            
                        raise Exception(f"API error ({self.service_name}): Status {response.status}")
                        
            except Exception as e:
                SystemLogger.log_api_retry(
                    self.service_name,
                    attempt + 1,
                    self.max_retries,
                    str(e)
                )
                if attempt == self.max_retries - 1:
                    raise
                await asyncio.sleep(self.retry_delay * (attempt + 1))

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
        # Extract system message if present
        system_message = next(
            (msg["content"] for msg in messages if msg["role"] == "system"),
            None
        )
        
        payload = {
            "model": model,
            "messages": [m for m in messages if m["role"] != "system"],
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        
        if system_message:
            payload["system"] = system_message
            
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
            "HTTP-Referer": "http://localhost:8000",
            "X-Title": "Hephia Project"
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
