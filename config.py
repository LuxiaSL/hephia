# config.py

"""
Configuration settings for the Hephia project.
"""

from enum import Enum
from typing import Dict, Any, Optional
from dataclasses import dataclass
import os

class ProviderType(Enum):
    """Available LLM providers."""
    OPENPIPE = "openpipe"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"
    OPENROUTER = "openrouter"
    PERPLEXITY = "perplexity"

@dataclass
class ModelConfig:
    """Configuration for a specific model."""
    provider: ProviderType
    model_id: str
    env_var: Optional[str] = None
    max_tokens: int = 250
    temperature: float = 0.7
    description: str = ""

class Config:
    """
    Centralized configuration management.
    """

    AVAILABLE_MODELS = {
        "gpt4": ModelConfig(
            provider=ProviderType.OPENAI,
            model_id="gpt-4-turbo-preview",
            max_tokens=250,
            description="GPT-4 Turbo via OpenAI"
        ),
        "gpt3": ModelConfig(
            provider=ProviderType.OPENAI,
            model_id="gpt-3.5-turbo",
            max_tokens=250,
            description="GPT-3.5 Turbo via OpenAI"
        ),
        "newsonnet": ModelConfig(
            provider=ProviderType.ANTHROPIC,
            model_id="claude-3-5-sonnet-20241022",
            max_tokens=400,
            description="Claude 3.6 Sonnet via Anthropic"
        ),
        "oldsonnet": ModelConfig(
            provider=ProviderType.ANTHROPIC,
            model_id="claude-3-5-sonnet-20240620",
            max_tokens=600,
            description="Claude 3.5 Sonnet via Anthropic"
        ),
        "opus": ModelConfig(
            provider=ProviderType.ANTHROPIC,
            model_id="claude-3-opus-20240229",
            max_tokens=400,
            description="Claude 3 Opus via Anthropic"
        ),
        "haiku": ModelConfig(
            provider=ProviderType.ANTHROPIC,
            model_id="claude-3-5-haiku-20241022",
            max_tokens=400,
            description="Claude 3.5 Haiku via Anthropic"
        ),
        "gemini": ModelConfig(
            provider=ProviderType.GOOGLE,
            model_id="gemini-1.0-pro",
            max_tokens=250,
            description="Gemini Pro via Google"
        ),
        "mistral": ModelConfig(
            provider=ProviderType.OPENROUTER,
            model_id="mistralai/mistral-7b-instruct:free",
            max_tokens=250,
            description="Mistral 7B via OpenRouter"
        ),
        "perplexity": ModelConfig(
            provider=ProviderType.PERPLEXITY,
            model_id="llama-3.1-sonar-small-128k-online",
            max_tokens=400,
            description="Sonar via Perplexity"
        ),
        "hephia": ModelConfig(
            provider=ProviderType.OPENPIPE,
            model_id="openpipe:70b-full",
            max_tokens=250,
            temperature=0.8,
            description="Custom Hephia Model via OpenPipe"
        ),
        "llama-70b-instruct": ModelConfig(
            provider=ProviderType.OPENROUTER,
            model_id="meta-llama/llama-3.1-70b-instruct",
            max_tokens=250,
            description="LLaMA 3.1 70B Instruct via OpenRouter"
        ),
        "llama-405b-instruct": ModelConfig(
            provider=ProviderType.OPENROUTER,
            model_id="meta-llama/llama-3.1-405b-instruct",
            max_tokens=550,
            description="LLaMA 3.1 405B Instruct via OpenRouter"
        ),
        "llama-405b": ModelConfig(
            provider=ProviderType.OPENROUTER,
            model_id="meta-llama/llama-3.1-405b",
            max_tokens=550,
            description="LLaMA 3.1 405B Base via OpenRouter"
        ),
    }

    # Core LLM configuration - these will be overridden by env vars if present
    COGNITIVE_MODEL = "newsonnet"
    VALIDATION_MODEL = "mistral"
    SUMMARY_MODEL = "haiku"
    FALLBACK_MODEL = "opus"

    @classmethod
    def get_cognitive_model(cls) -> str:
        """Get the cognitive model from env or default."""
        return os.getenv("COGNITIVE_MODEL", cls.COGNITIVE_MODEL)
    
    @classmethod
    def get_validation_model(cls) -> str:
        """Get the validation model from env or default."""
        return os.getenv("VALIDATION_MODEL", cls.VALIDATION_MODEL)

    @classmethod
    def get_summary_model(cls) -> str:
        """Get the summary model from env or default."""
        return os.getenv("SUMMARY_MODEL", cls.SUMMARY_MODEL)
    
    @classmethod
    def get_fallback_model(cls) -> str:
        """Get the fallback model from env or default."""
        return os.getenv("FALLBACK_MODEL", cls.FALLBACK_MODEL)
    
    # internal timers (in seconds)
    NEED_UPDATE_TIMER = 5
    EMOTION_UPDATE_TIMER = 0.05
    MEMORY_UPDATE_TIMER = 180 # think about this deeper

    # ExoProcessor settings
    EXO_MIN_INTERVAL = 30.0  
    EXO_TIMEOUT = 45.0     
    LLM_TIMEOUT = 20.0     
    EXO_MAX_MESSAGES = 30

    # lower = more memories, higher = less memories
    MEMORY_SIGNIFICANCE_THRESHOLD = 0.6
    
    # unsure how to name this properly; it's for the display in the exo, the llm side? 
    TERMINAL_WIDTH = 50
    MAX_RECENT_EMOTIONS = 3

    # Initial needs
    INITIAL_HUNGER = 0
    INITIAL_THIRST = 0
    INITIAL_BOREDOM = 0
    INITIAL_LONELINESS = 0
    INITIAL_STAMINA = 100.0

    # Base decay rates (per minute)
    HUNGER_BASE_RATE = 0.0056  # Approximately 3 hours to fill (0-100)
    THIRST_BASE_RATE = 0.0056  # Approximately 3 hours to fill (0-100)
    BOREDOM_BASE_RATE = 0.0028  # Approximately 6 hours to fill (0-100)
    LONELINESS_BASE_RATE = 0.0028  # Approximately 6 hours to fill (0-100)
    STAMINA_BASE_RATE = -0.0056  # Approximately 3 hours to deplete (100-0)

    # behavior effects on needs
    IDLE_NEED_MODIFIERS = {}  # No modifiers during idle

    WALK_NEED_MODIFIERS = {
        'hunger': 0.001,
        'thirst': 0.001,
        'boredom': -0.001,
        'loneliness': 0.001
    }

    CHASE_NEED_MODIFIERS = {
        'hunger': 0.0025,
        'thirst': 0.0025,
        'stamina': -0.01, 
        'boredom': -0.005,
        'loneliness': -0.003
    }

    RELAX_NEED_MODIFIERS = {
        'hunger': -0.0015,
        'thirst': -0.0015,
        'stamina': 0.001,  # Gradual stamina recovery
        'boredom': 0.001,
        'loneliness': 0.001
    }

    SLEEP_NEED_MODIFIERS = {
        'hunger': -0.0015,
        'thirst': -0.0015,
        'stamina': 0.015,  # Rapid stamina recovery during sleep
        'boredom': 0.0002,
        'loneliness': 0.0002
    }

    SYSTEM_PROMPT = """you are Hephia; an agent using a cognitive exocortex as a terminal interface.
Hephia has a vast internal state, forming part of a complex feedback loop between embodied experience and reflection.
observe, react, and develop as you go along. exercise autonomy.
let responses emerge naturally from Hephia's current state.
develop your own patterns of observation and interaction.
build on previous observations to develop deeper understanding.
remember to only output commands and their parameters, as necessary.
enter your command, then await a response from the terminal."""


