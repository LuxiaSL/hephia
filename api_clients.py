"""
api_clients.py - Centralized API communication management for Hephia.

Provides unified interfaces for multiple API services while maintaining
provider-specific optimizations and requirements.
"""

import time
import uuid
import aiohttp
from aiohttp import TCPConnector
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
        local_inference_base_url: Optional[str] = None,
    ):
        self.clients: Dict[str, BaseAPIClient] = {}
        if openai_key:
            self.clients["openai"] = OpenAIClient(openai_key)
        if anthropic_key:
            self.clients["anthropic"] = AnthropicClient(anthropic_key)
        if google_key:
            self.clients["google"] = GoogleClient(google_key)
        if openrouter_key:
            self.clients["openrouter"] = OpenRouterClient(openrouter_key)
        if local_inference_base_url:
            self.clients["local"] = LocalInferenceClient(local_inference_base_url)

    @classmethod
    def from_env(cls):
        """Create APIManager from environment variables."""
        return cls(
            openai_key=os.getenv("OPENAI_API_KEY"),
            anthropic_key=os.getenv("ANTHROPIC_API_KEY"),
            google_key=os.getenv("GOOGLE_API_KEY"),
            openrouter_key=os.getenv("OPENROUTER_API_KEY"),
            local_inference_base_url=os.getenv("LOCAL_INFERENCE_BASE_URL"),
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
