"""
api_clients.py - Centralized API communication management for Hephia.

Provides unified interfaces for multiple API services while maintaining
provider-specific optimizations and requirements.
"""

import copy
import platform
import random
import socket
import time
import traceback
import uuid
import aiohttp
from aiohttp import UnixConnector, TCPConnector
import os
from typing import Dict, Any, List, Optional, Union
import json
import asyncio
from abc import ABC, abstractmethod

from loggers import SystemLogger
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
    """Enhanced OpenAI client supporting both Responses and Chat Completions APIs."""

    # Models that support Responses API well
    RESPONSES_API_MODELS = {
        "gpt-5", "gpt-5-mini", "gpt-5-nano", "gpt-5-pro",
        "gpt-4.1", "gpt-4.1-mini", "gpt-4.1-nano",
        "gpt-4o", "gpt-4o-mini",
        "o3", "o3-mini", "o4-mini"
    }

    # Models that have issues with json_schema in text.format
    RESPONSES_NO_JSON_SCHEMA = {
        "gpt-5-chat-latest"
    }

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
        """Extract assistant text from either Chat Completions or Responses API shapes."""
        # Chat Completions shape
        if isinstance(response, dict) and "choices" in response:
            try:
                return response["choices"][0]["message"]["content"]
            except (KeyError, IndexError, TypeError):
                pass
        
        # Responses API shape
        try:
            output = response.get("output", [])
            for item in reversed(output):
                if item.get("type") == "message" and item.get("role") == "assistant":
                    parts = item.get("content", [])
                    texts = [p.get("text", "") for p in parts if isinstance(p, dict) and p.get("text")]
                    if texts:
                        return "".join(texts)
            # Some Responses variants may include a top-level convenience field
            if "output_text" in response and isinstance(response["output_text"], str):
                return response["output_text"]
        except Exception:
            pass
        
        return ""

    def _messages_to_responses_input(self, messages: List[Dict[str, str]]) -> List[Dict[str, Any]]:
        """DEPRECATED: Use _split_messages_for_responses. Kept for compatibility."""
        return self._split_messages_for_responses(messages)[1]

    def _split_messages_for_responses(self, messages: List[Dict[str, str]]) -> (Optional[str], List[Dict[str, Any]]):
        """Convert messages to Responses API fields: (instructions, input[]).

        - system messages -> concatenated instructions
        - user messages -> role user with content type input_text
        - assistant messages -> role assistant with content type output_text
        """
        instructions_parts: List[str] = []
        input_items: List[Dict[str, Any]] = []

        for msg in messages:
            role = msg.get("role") or "user"
            content = msg.get("content") or ""

            if role == "system":
                if content:
                    instructions_parts.append(content)
                continue

            part_type = "input_text" if role == "user" else "output_text" if role == "assistant" else "input_text"
            input_items.append({
                "role": role if role in ("user", "assistant") else "user",
                "content": [
                    {"type": part_type, "text": content}
                ]
            })

        instructions = "\n".join(instructions_parts) if instructions_parts else None
        return instructions, input_items

    def _should_use_responses_api(self, model: str, kwargs: Dict[str, Any]) -> bool:
        """Determine if we should use Responses API based on model and parameters."""
        model_name = model.split("/")[-1].lower()

        for supported_model in self.RESPONSES_API_MODELS:
            if model_name.startswith(supported_model.lower()):
                # Special case: if using json_schema with problematic models, use Chat Completions
                if any(no_json in model_name for no_json in self.RESPONSES_NO_JSON_SCHEMA):
                    rf = kwargs.get("response_format") or {}
                    if isinstance(rf, dict) and rf.get("type") == "json_schema":
                        SystemLogger.debug(
                            f"Model {model} doesn't support json_schema in Responses API, using Chat Completions"
                        )
                        return False

                # If has GPT-5 specific params, prefer Responses API
                if any(k in kwargs for k in ["reasoning_effort", "verbosity", "reasoning"]):
                    return True

                return True

        return False

    def _convert_response_format_to_text_format(self, response_format: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Convert Chat Completions response_format to Responses API text.format."""
        if not response_format:
            return None

        format_type = response_format.get("type")

        if format_type == "json_object":
            return {"type": "text"}

        if format_type == "json_schema":
            json_schema = response_format.get("json_schema", {}) or {}
            return {
                "type": "json_schema",
                "name": json_schema.get("name", "response"),
                "schema": json_schema.get("schema", {}),
                "strict": json_schema.get("strict", True)
            }

        return None

    def _build_responses_payload(
        self,
        messages: List[Dict[str, str]],
        model: str,
        temperature: float,
        max_tokens: int,
        **kwargs
    ) -> Dict[str, Any]:
        """Build payload for Responses API with all parameters."""
        instructions, input_items = self._split_messages_for_responses(messages)

        payload: Dict[str, Any] = {
            "model": model,
            "input": input_items,
            "max_output_tokens": max_tokens
        }

        if instructions:
            payload["instructions"] = instructions

        if "response_format" in kwargs:
            text_format = self._convert_response_format_to_text_format(kwargs["response_format"])
            if text_format:
                payload.setdefault("text", {})
                payload["text"]["format"] = text_format

        if "reasoning_effort" in kwargs:
            if "reasoning" not in payload:
                payload["reasoning"] = {}
            payload["reasoning"]["effort"] = kwargs["reasoning_effort"]

        if "verbosity" in kwargs:
            payload["verbosity"] = kwargs["verbosity"]

        for key in [
            "stream", "store", "truncation", "tool_choice", "tools",
            "parallel_tool_calls", "top_p", "metadata"
        ]:
            if key in kwargs:
                payload[key] = kwargs[key]

        # gpt-5 family defaults: minimal reasoning and low verbosity unless explicitly provided
        model_name = model.split("/")[-1].lower()
        if model_name.startswith("gpt-5"):
            if "reasoning_effort" not in kwargs and "reasoning" not in payload:
                payload["reasoning"] = {"effort": "minimal"}
            payload.setdefault("text", {})
            if "verbosity" not in payload["text"]:
                payload["text"]["verbosity"] = "low"

        return payload

    def _build_chat_completions_payload(
        self,
        messages: List[Dict[str, str]],
        model: str,
        temperature: float,
        max_tokens: int,
        **kwargs
    ) -> Dict[str, Any]:
        """Build payload for Chat Completions API."""
        payload: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
        }

        if "max_completion_tokens" in kwargs:
            payload["max_completion_tokens"] = kwargs["max_completion_tokens"]
        elif any(model.startswith(prefix) for prefix in ["gpt-5", "gpt-4.1", "o3", "o4"]):
            payload["max_completion_tokens"] = max_tokens
        else:
            payload["max_tokens"] = max_tokens

        valid_params = [
            "response_format", "stream", "stop", "logprobs", "top_logprobs",
            "top_p", "frequency_penalty", "presence_penalty", "seed",
            "tools", "tool_choice", "user", "logit_bias"
        ]

        for key in valid_params:
            if key in kwargs:
                payload[key] = kwargs[key]

        return payload


    def _responses_to_chat_completions(self, responses_obj: Dict[str, Any], model: str) -> Dict[str, Any]:
        """Adapt a Responses API object to a Chat Completions–like response for backward compatibility."""
        assistant_text = self._extract_message_content(responses_obj) or ""
        usage = responses_obj.get("usage", {}) or {}
        # Map usage fields if present
        prompt_tokens = usage.get("input_tokens") if isinstance(usage, dict) else None
        completion_tokens = usage.get("output_tokens") if isinstance(usage, dict) else None
        total_tokens = None
        if isinstance(prompt_tokens, int) and isinstance(completion_tokens, int):
            total_tokens = prompt_tokens + completion_tokens
        chat_usage = None
        if prompt_tokens is not None or completion_tokens is not None:
            chat_usage = {
                "prompt_tokens": prompt_tokens or 0,
                "completion_tokens": completion_tokens or 0,
                "total_tokens": total_tokens or ((prompt_tokens or 0) + (completion_tokens or 0))
            }
        adapted = {
            "id": responses_obj.get("id", f"chatcmpl-{uuid.uuid4().hex[:12]}"),
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": assistant_text},
                    "finish_reason": responses_obj.get("status", "stop")
                }
            ]
        }
        if chat_usage is not None:
            adapted["usage"] = chat_usage
        return adapted

    async def create_completion(
        self,
        messages: List[Dict[str, str]],
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 150,
        return_content_only: bool = False,
        **kwargs
    ) -> Union[Dict[str, Any], str]:
        """Create completion with automatic API selection and proper parameter handling."""
        use_responses = self._should_use_responses_api(model, kwargs)

        if use_responses:
            SystemLogger.debug(f"Using Responses API for model {model}")
            try:
                payload = self._build_responses_payload(messages, model, temperature, max_tokens, **kwargs)
                responses_obj = await self._make_request("responses", payload=payload)
                adapted = self._responses_to_chat_completions(responses_obj, model)
                if return_content_only:
                    return self._extract_message_content(adapted)
                return adapted
            except Exception as responses_err:
                error_msg = str(responses_err)
                SystemLogger.warning(
                    f"Responses API failed for {model}: {error_msg[:200]}"
                )
                # If parameter-related error, fall through to Chat Completions
                if not any(keyword in error_msg.lower() for keyword in [
                    "invalid parameter", "not supported", "unknown parameter",
                    "text.format", "json_schema"
                ]):
                    # Re-raise for non-parameter issues (auth, rate limit, etc.)
                    raise

        SystemLogger.debug(f"Using Chat Completions API for model {model}")
        payload = self._build_chat_completions_payload(messages, model, temperature, max_tokens, **kwargs)

        try:
            response = await self._make_request("chat/completions", payload=payload)
            if return_content_only:
                return self._extract_message_content(response)
            return response
        except Exception as chat_err:
            err_text = str(chat_err)
            if "max_tokens" in err_text and "max_completion_tokens" in err_text:
                SystemLogger.debug("Retrying with max_completion_tokens instead of max_tokens")
                if "max_tokens" in payload:
                    payload["max_completion_tokens"] = payload.pop("max_tokens")
                    response = await self._make_request("chat/completions", payload=payload)
                    if return_content_only:
                        return self._extract_message_content(response)
                    return response
            raise


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
            "HTTP-Referer": "https://github.com/LuxiaSL/hephia",
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

class LocalInferenceClient(BaseAPIClient):
    def __init__(self, base_url: str):
        super().__init__(
            api_key="N/A",  # No API key needed for local inference
            base_url=base_url,
            service_name="LocalInference"
        )

    def _get_headers(self, extra_headers: Optional[Dict] = None) -> Dict[str, str]:
        headers = {
            "Content-Type": "application/json"
        }
        if extra_headers:
            headers.update(extra_headers)
        return headers
    
    def _extract_message_content(self, response: Dict[str, Any]) -> str:
        """Extract message content from local inference response."""
        if "choices" in response and len(response["choices"]) > 0:
            return response["choices"][0]["message"]["content"]
        return ""
    
    async def create_completion(
        self,
        messages: List[Dict[str, str]],
        model: str = "none",
        temperature: float = 0.7,
        max_tokens: int = 150,
        return_content_only: bool = False
    ) -> Union[Dict[str, Any], str]:
        """Create chat completion via local inference."""
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        
        response = await self._make_request("chat/completions", payload=payload, timeout=Config.LLM_TIMEOUT) #more permissive timeout for generation
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
        self.advanced_logging = Config.get_advanced_c2_logging()
        self.connect_timeout = 10
        self.read_timeout = 45
        self.total_timeout = 60
    
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
        request_id = str(uuid.uuid4())[:8]
        
        # Basic request logging - always done
        SystemLogger.info(f"API Request {request_id} ({self.service_name}): "
                         f"Endpoint: {endpoint}, Method: {method}")
        
        # Payload logging
        if self.advanced_logging and payload:
            payload_log = copy.deepcopy(payload) if payload else {}
            if "messages" in payload_log:
                # Summarize messages to reduce log verbosity
                msg_count = len(payload_log["messages"])
                if msg_count > 2:
                    # Show first and last message only
                    first_msg = payload_log["messages"][0]
                    last_msg = payload_log["messages"][-1]
                    # Truncate content if too long
                    if "content" in first_msg and len(first_msg["content"]) > 150:
                        first_msg["content"] = first_msg["content"][:150] + "..."
                    if "content" in last_msg and len(last_msg["content"]) > 150:
                        last_msg["content"] = last_msg["content"][:150] + "..."
                    payload_log["messages"] = [
                        first_msg,
                        {"role": "system", "content": f"... ({msg_count-2} messages omitted) ..."},
                        last_msg
                    ]
            SystemLogger.debug(f"API Request {request_id} Payload Summary: {json.dumps(payload_log)}")
            SystemLogger.debug(f"API Request {request_id} Socket Path: {self.socket_path}")

        # Socket existence check
        if not os.path.exists(self.socket_path):
            SystemLogger.error(f"API Request {request_id} Failed: Socket path {self.socket_path} does not exist")
            raise FileNotFoundError(f"Socket path {self.socket_path} does not exist")

        for attempt in range(self.max_retries):
            start_time = time.time()
            connector = None
            
            try:
                # Socket lifecycle tracking
                SystemLogger.debug(f"Socket lifecycle {request_id}: Opening connection to {self.socket_path}")

                # Create connector with careful error handling
                try:
                    connector = UnixConnector(path=self.socket_path)
                    SystemLogger.debug(f"Socket lifecycle {request_id}: Connector created successfully")
                except Exception as conn_err:
                    SystemLogger.error(f"Socket lifecycle {request_id}: Failed to create connector: {str(conn_err)}")
                    raise

                # Configure timeouts for different phases of the request
                timeout = aiohttp.ClientTimeout(
                    total=self.total_timeout,
                    connect=self.connect_timeout,
                    sock_read=self.read_timeout
                )
                
                # Create session with specific connector
                session_start = time.time()
                SystemLogger.debug(f"Socket lifecycle {request_id}: Creating session")

                async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
                    url = f"http://localhost/{endpoint.lstrip('/')}"
                    
                    SystemLogger.debug(f"Socket lifecycle {request_id}: Sending request to {url}")
                                        
                    # Make the actual request
                    request_start = time.time()
                    async with session.request(
                        method,
                        url,
                        headers=headers,
                        json=payload
                    ) as response:
                        # headers received
                        headers_time = time.time() - request_start
                        SystemLogger.debug(
                            f"Socket lifecycle {request_id}: Received response headers in {headers_time:.2f}s, "
                            f"Status: {response.status}"
                        )
                        # Read response body with timeout protection
                        try:
                            body_start = time.time()
                            SystemLogger.debug(f"Socket lifecycle {request_id}: Starting response body read")
                            
                            # Read raw bytes first
                            raw_bytes = await asyncio.wait_for(
                                response.read(), 
                                timeout=self.read_timeout
                            )
                            
                            body_time = time.time() - body_start
                            SystemLogger.debug(
                                f"Socket lifecycle {request_id}: Completed body read in {body_time:.2f}s, "
                                f"Size: {len(raw_bytes)} bytes"
                            )
                            
                            # Decode as text
                            try:
                                response_text = raw_bytes.decode('utf-8')
                                SystemLogger.debug(
                                    f"Socket lifecycle {request_id}: Decoded UTF-8 content, "
                                    f"Length: {len(response_text)}"
                                )
                            except UnicodeDecodeError as e:
                                SystemLogger.error(f"Socket lifecycle {request_id}: UTF-8 decode error: {str(e)}")
                                raise Exception(f"Failed to decode response as UTF-8: {str(e)}")
                            
                        except asyncio.TimeoutError:
                            elapsed = time.time() - body_start
                            SystemLogger.error(
                                f"Socket lifecycle {request_id}: Response body read timed out after {elapsed:.2f}s"
                            )
                            raise

                        total_time = time.time() - start_time
                        
                        # Success case - HTTP 200
                        if response.status == 200:
                            try:
                                response_data = json.loads(response_text)
                                SystemLogger.info(
                                    f"API Response {request_id} ({self.service_name}): "
                                    f"Success, Time: {total_time:.2f}s"
                                )
                                
                                # Log response summary for successful requests
                                if self.advanced_logging:
                                    # Create a summary of the response for logging
                                    if isinstance(response_data, dict) and "choices" in response_data:
                                        choices = response_data.get("choices", [])
                                        if choices and len(choices) > 0:
                                            msg = choices[0].get("message", {})
                                            content = msg.get("content", "")
                                            content_preview = content[:100] + "..." if len(content) > 100 else content
                                            SystemLogger.debug(
                                                f"API Response {request_id} Content: {content_preview}"
                                            )
                                
                                # Record successful API request
                                SystemLogger.log_api_request(
                                    self.service_name,
                                    endpoint,
                                    response.status
                                )
                                
                                return response_data
                                
                            except json.JSONDecodeError as e:
                                SystemLogger.error(
                                    f"API Response {request_id} JSON parse error: {str(e)}, "
                                    f"Response text: {response_text[:200]}..."
                                )
                                raise Exception(f"Failed to parse JSON response: {str(e)}")
                        
                        # Error case - standard error logging
                        error_log = f"API Error {request_id} ({self.service_name}): "
                        error_log += f"Status: {response.status}, Time: {total_time:.2f}s"
                        SystemLogger.error(error_log)
                        
                        # Advanced error details if enabled
                        if self.advanced_logging:
                            SystemLogger.error(f"API Error {request_id} Headers: {dict(response.headers)}")
                            
                            # Log full response content on error
                            error_content = f"Full response: {response_text[:5000]}"
                            if len(response_text) > 5000:
                                error_content += f" ... (truncated, total length: {len(response_text)})"
                            SystemLogger.error(f"API Error {request_id} Response: {error_content}")
                        
                        # Standard error logging for system logger
                        SystemLogger.log_api_request(
                            self.service_name,
                            endpoint,
                            response.status,
                            error=response_text[:1000]  # Truncate very long errors
                        )
                        
                        if response.status >= 500:
                            retry_msg = f"Server error: {response.status}"
                            SystemLogger.log_api_retry(
                                self.service_name,
                                attempt + 1,
                                self.max_retries,
                                retry_msg
                            )
                            
                            backoff_time = 0.5
                            
                            if self.advanced_logging:
                                SystemLogger.debug(f"API Request {request_id} Retry backoff: {backoff_time:.2f}s")
                                
                            await asyncio.sleep(backoff_time)
                            continue
                            
                        raise Exception(f"Unix socket error: Status {response.status}, Response: {response_text[:200]}")
                        
            except asyncio.TimeoutError as te:
                elapsed_time = time.time() - start_time
                phase = "connection" if elapsed_time < self.connect_timeout else "read"

                SystemLogger.error(
                    f"API Timeout {request_id} ({self.service_name}): "
                    f"Timeout during {phase} phase after {elapsed_time:.2f}s, "
                    f"Attempt {attempt+1}/{self.max_retries}"
                )
                
                if attempt < self.max_retries - 1:
                    backoff_time = 0.5 
                    SystemLogger.log_api_retry(
                        self.service_name,
                        attempt + 1, 
                        self.max_retries,
                        f"Timeout during {phase}, waiting {backoff_time}s"
                    )
                    await asyncio.sleep(backoff_time)
                    continue
                    
                raise Exception(f"Unix socket timeout after {self.max_retries} attempts: {str(te)}")
                
            except Exception as e:
                # Standard exception logging
                exc_type = type(e).__name__
                elapsed_time = time.time() - start_time

                SystemLogger.error(
                    f"API Exception {request_id} ({self.service_name}): "
                    f"Type: {exc_type}, Message: {str(e)}, "
                    f"Attempt: {attempt+1}/{self.max_retries}, "
                    f"Elapsed time: {elapsed_time:.2f}s"
                )
                
                if self.advanced_logging:
                    exc_tb = traceback.format_exc()
                    SystemLogger.debug(f"API Exception {request_id} Traceback: {exc_tb}")
                
                SystemLogger.log_api_retry(
                    self.service_name,
                    attempt + 1, 
                    self.max_retries,
                    f"{exc_type}: {str(e)}"
                )
                
                if attempt < self.max_retries - 1:
                    backoff_time = 0.5 
                    await asyncio.sleep(backoff_time)
                    continue
                    
                # On final attempt, re-raise with detailed error
                raise Exception(f"Unix socket error after {self.max_retries} attempts: {exc_type}: {str(e)}")
            
            finally:
                # Ensure connector is closed properly
                if connector:
                    SystemLogger.debug(f"Socket lifecycle {request_id}: Closing connector")
                    try:
                        connector.close()
                    except Exception as close_err:
                        SystemLogger.error(
                            f"Socket lifecycle {request_id}: Failed to close connector: {str(close_err)}"
                        )


class Chapter2Client(BaseAPIClient):
    """Client for Chapter2 API supporting both HTTP and Unix socket."""
    
    def __init__(self, socket_path: str = None, http_port: int = None):
        self.is_unix = platform.system() != "Windows"
        self.socket_path = socket_path or Config.get_chapter2_socket_path()
        self.http_port = http_port or Config.get_chapter2_http_port()
        self.advanced_logging = Config.get_advanced_c2_logging()
        
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
            if self.advanced_logging:
                SystemLogger.info(f"Chapter2Client initialized with Unix socket: {self.socket_path}")
        else:
            self.unix_client = None
            if self.advanced_logging:
                SystemLogger.info(f"Chapter2Client initialized with HTTP: localhost:{self.http_port}")
    
    def _get_headers(self, extra_headers: Optional[Dict] = None) -> Dict[str, str]:
        """Get headers for HTTP requests"""
        headers = {"Content-Type": "application/json"}
        if extra_headers:
            headers.update(extra_headers)
        return headers

    def _extract_message_content(self, response: Dict[str, Any]) -> str:
        """Extract message content from response"""
        return response["choices"][0]["message"]["content"]
    
    def _check_socket_health(self) -> bool:
        """Check if the Unix socket is healthy and accessible."""
        if not self.use_unix:
            return False
            
        if not os.path.exists(self.socket_path):
            SystemLogger.error(f"Socket health check failed: {self.socket_path} does not exist")
            return False
            
        try:
            # Basic connectivity test
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
                sock.settimeout(2)
                sock.connect(self.socket_path)
                return True
        except Exception as e:
            SystemLogger.error(f"Socket health check failed: Connection error: {str(e)}")
            return False
    
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
        request_id = str(uuid.uuid4())[:8]
        
        # Add name: system for user messages
        formatted_messages = [
            {**msg, "name": "system"} if msg.get("role") == "user" else msg 
            for msg in messages
        ]
        
        # Basic request logging - always done
        log_prefix = f"Completion Request {request_id}: "
        basic_info = f"Model: {model}, Messages: {len(formatted_messages)}"
        
        if self.advanced_logging:
            basic_info += f", Temperature: {temperature}, Max Tokens: {max_tokens}, Using Unix: {self.use_unix}"
        
        SystemLogger.info(log_prefix + basic_info)
        
        # Detailed message logging - only if advanced logging is enabled
        if self.advanced_logging:
            msg_count = len(formatted_messages)
            if msg_count > 0:
                first_msg = formatted_messages[0]
                first_content = first_msg.get("content", "")
                SystemLogger.debug(f"Completion Request {request_id} First message: "
                                 f"Role: {first_msg.get('role')}, "
                                 f"Content: {first_content[:100]}...")
                
                if msg_count > 1:
                    last_msg = formatted_messages[-1]
                    last_content = last_msg.get("content", "")
                    SystemLogger.debug(f"Completion Request {request_id} Last message: "
                                     f"Role: {last_msg.get('role')}, "
                                     f"Content: {last_content[:100]}...")
                    
        if self.use_unix and not self._check_socket_health():
            SystemLogger.warning(f"Completion Request {request_id}: Socket health check failed")
        
        start_time = time.time()
        
        try:
            if self.use_unix:
                SystemLogger.debug(f"Completion Request {request_id}: Using Unix socket client")
                    
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
            else:
                SystemLogger.debug(f"Completion Request {request_id}: Using HTTP client")
                    
                # For HTTP, use the base class _make_request
                response = await super()._make_request(
                    "chat/completions",
                    payload={
                        "model": model,
                        "messages": formatted_messages,
                        "temperature": temperature,
                        "max_tokens": max_tokens,
                        **kwargs
                    }
                )
            
            # Success logging
            elapsed_time = time.time() - start_time
            
            if isinstance(response, dict) and "choices" in response:
                content = self._extract_message_content(response)
                content_preview = content[:100] + "..." if len(content) > 100 else content
                
                SystemLogger.info(
                    f"Completion Success {request_id}: "
                    f"Time: {elapsed_time:.2f}s, "
                    f"Content length: {len(content)}"
                )
                
                if self.advanced_logging:
                    SystemLogger.debug(f"Completion Content {request_id}: {content_preview}")
                    
                    if "usage" in response:
                        usage = response["usage"]
                        SystemLogger.debug(f"Completion Usage {request_id}: {usage}")
            else:
                SystemLogger.info(f"Completion Success {request_id}: Time: {elapsed_time:.2f}s")
            
            # Return appropriate format
            if return_content_only:
                return self._extract_message_content(response)
            return response
            
        except Exception as e:
            # Error logging
            exc_type = type(e).__name__
            elapsed_time = time.time() - start_time
            
            SystemLogger.error(
                f"Completion Error {request_id}: Type: {exc_type}, "
                f"Message: {str(e)}, Time: {elapsed_time:.2f}s"
            )
            
            if self.advanced_logging:
                exc_tb = traceback.format_exc()
                SystemLogger.debug(f"Completion Error {request_id} Traceback: {exc_tb}")
            # Re-raise the exception
            raise


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
        perplexity_key: Optional[str] = None,
        local_inference_base_url: Optional[str] = None
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
        if local_inference_base_url:
            self.clients["local"] = LocalInferenceClient(local_inference_base_url)
    
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
            perplexity_key=os.getenv("PERPLEXITY_API_KEY"),
            local_inference_base_url=os.getenv("LOCAL_INFERENCE_BASE_URL")
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
