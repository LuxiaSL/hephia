"""
api_clients.py - Centralized API communication management for Hephia.
"""

import aiohttp
from typing import Dict, Any, List, Optional
import json
import asyncio
from datetime import datetime
import logging
from abc import ABC, abstractmethod
from config import Config

logger = logging.getLogger('hephia.api')

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
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        if extra_headers:
            headers.update(extra_headers)
            
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
                            return await response.json()
                        
                        error_text = await response.text()
                        logger.error(f"{self.service_name} API error (Status {response.status})")
                        
                        if response.status == 429:  # Rate limit
                            retry_after = int(response.headers.get('Retry-After', self.retry_delay))
                            await asyncio.sleep(retry_after)
                            continue
                            
                        if response.status >= 500:  # Server error, retry
                            await asyncio.sleep(self.retry_delay * (attempt + 1))
                            continue
                            
                        raise Exception(f"API error ({self.service_name}): Status {response.status}")
                        
            except Exception as e:
                logger.error(f"{self.service_name} request failed (attempt {attempt + 1}/{self.max_retries}): {str(e)}")
                if attempt == self.max_retries - 1:
                    raise
                await asyncio.sleep(self.retry_delay * (attempt + 1))

class OpenRouterClient(BaseAPIClient):
    """Client for OpenRouter API interactions."""
    
    def __init__(self, api_key: str):
        super().__init__(
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1",
            service_name="OpenRouter"
        )
    
    async def create_completion(
        self,
        messages: List[Dict[str, str]],
        model: str,
        temperature: float = 0.7
    ) -> Dict[str, Any]:
        """Create chat completion via OpenRouter."""
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature
        }
        
        extra_headers = {
            "HTTP-Referer": "http://localhost:8000",
            "X-Title": "Hephia Project"
        }
        
        return await self._make_request(
            "chat/completions",
            payload=payload,
            extra_headers=extra_headers
        )
    
class OpenPipeClient(BaseAPIClient):
    """Client for OpenPipe API interactions (main LLM interactions)."""
    
    def __init__(self, api_key: str):
        super().__init__(
            api_key=api_key,
            base_url="https://api.openpipe.ai/api/v1",
            service_name="OpenPipe"
        )
    
    async def create_completion(
        self,
        messages: List[Dict[str, str]],
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 125
    ) -> Dict[str, Any]:
        """Create chat completion via OpenPipe."""
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        
        return await self._make_request(
            "chat/completions",
            payload=payload
        )

class PerplexityClient(BaseAPIClient):
    """Client for Perplexity API interactions."""
    
    def __init__(self, api_key: str):
        super().__init__(
            api_key=api_key,
            base_url="https://api.perplexity.ai",
            service_name="Perplexity"
        )
    
    async def create_completion(
        self,
        messages: List[Dict[str, str]],
        model: str = "llama-3.1-sonar-small-128k-online",
        max_tokens: int = 400,
        temperature: float = 0.7
    ) -> Dict[str, Any]:
        """Create chat completion via Perplexity."""
        payload = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature
        }
        
        return await self._make_request(
            "chat/completions",
            payload=payload
        )

class APIManager:
    """
    Central manager for all API clients.
    Handles initialization and provides access to different services.
    """
    
    def __init__(
        self,
        openpipe_key: str,
        openrouter_key: str,
        perplexity_key: str
    ):
        self.openpipe = OpenPipeClient(openpipe_key)
        self.openrouter = OpenRouterClient(openrouter_key)
        self.perplexity = PerplexityClient(perplexity_key)
        
    @classmethod
    def from_env(cls):
        """Create APIManager from environment variables."""
        import os
        return cls(
            openpipe_key=os.getenv("OPENPIPE_API_KEY"),
            openrouter_key=os.getenv("OPENROUTER_API_KEY"),
            perplexity_key=os.getenv("PERPLEXITY_API_KEY")
        )

