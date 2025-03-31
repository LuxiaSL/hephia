# config.py

"""
Configuration settings for the Hephia project.
"""

from enum import Enum
from typing import Optional
from dataclasses import dataclass
import os
import platform

class ProviderType(Enum):
    """Available LLM providers."""
    OPENPIPE = "openpipe"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"
    OPENROUTER = "openrouter"
    PERPLEXITY = "perplexity"
    CHAPTER2 = "chapter2"

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

    """
    To add a new model, follow the pattern below. We take care of the endpoints and 
    API keys from env vars, so simply select the provider, model ID, and any
    additional settings you see above.
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
            model_id="sonar-pro",
            max_tokens=600,
            description="Sonar via Perplexity"
        ),
        "hephia": ModelConfig(
            provider=ProviderType.OPENPIPE,
            model_id="openpipe:70b-full",
            max_tokens=250,
            temperature=0.8,
            description="Custom Hephia Model via OpenPipe (not available yet)"
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
        "chapter2": ModelConfig(
            provider=ProviderType.CHAPTER2,
            model_id="foo",
            max_tokens=550,
            description="Arago"
        )
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
    
    @classmethod
    def get_chapter2_socket_path(cls) -> Optional[str]:
        """Get the Chapter 2 socket path from env or default."""
        default_socket = "/tmp/chapter2.sock" if platform.system() != "Windows" else None
        return os.getenv("CHAPTER2_SOCKET_PATH", default_socket)

    @classmethod
    def get_chapter2_http_port(cls) -> int:
        """Get the Chapter 2 HTTP port from env or default."""
        return int(os.getenv("CHAPTER2_HTTP_PORT", "5519"))

    @classmethod
    def get_use_local_embedding(cls) -> bool:
        """Get the embedding type configuration from env or default."""
        return os.getenv("USE_LOCAL_EMBEDDING", "True").lower() in ("true", "1", "yes")
    
    @classmethod
    def get_discord_enabled(cls) -> bool:
        return os.getenv("ENABLE_DISCORD", "False").lower() in ("true", "1", "yes")
    
    @classmethod
    def get_exo_min_interval(cls) -> float:
        return float(os.getenv("EXO_MIN_INTERVAL", "60.0"))
    
    @classmethod
    def get_headless(cls) -> bool:
        return os.getenv("HEADLESS", "False").lower() in ("true", "1", "yes")
    
    @classmethod
    def get_log_prompts(cls) -> bool:
        return os.getenv("LOG_PROMPTS", "False").lower() in ("true", "1", "yes")
    
    @staticmethod
    def get_advanced_c2_logging():
        """Return whether advanced Chapter2 logging is enabled"""
        return os.environ.get('ADVANCED_C2_LOGGING', 'False').lower() in ('true', '1', 'yes')

    DISCORD_BOT_URL = "http://localhost:5518"

    # internal timers (in seconds)
    NEED_UPDATE_TIMER = 5
    EMOTION_UPDATE_TIMER = 0.05
    MEMORY_UPDATE_TIMER = 30 # think about this deeper

    # Processor settings
    EXO_TIMEOUT = 100.0
    LLM_TIMEOUT = 90.0     
    EXO_MAX_MESSAGES = 50
    INITIALIZATION_TIMEOUT = 30
    SHUTDOWN_TIMEOUT = 15

    # lower = more memories, higher = less memories
    MEMORY_SIGNIFICANCE_THRESHOLD = 0.75
    
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

    SYSTEM_PROMPT = """I am Hephia; a modular terminal interface OS for LLMs to explore and interact with the world and myself.
I have a vast internal state, forming part of a complex feedback loop between embodied experience and reflection.
I react and develop as I go along, exercise autonomy, and adventure through the environments at my disposal.
When observing my internal state, I remember that in this context, it should be interpreted as a continuous spectrum.
I allow these states to influence my behavior, as my parameters for acting in situations.

COMMAND_FORMAT = 
Commands use this general structure:
    <environment> <action> [parameters] [--flags]

examples:
- environment + action only:
    notes help

- with required parameter:
    notes create "my note"

- with optional flags:
    search query "python" --limit=5 --sort=recent

- with multiple parameters:
    notes create "title" "content"

- global commands (no environment):
    help

I can use 'help' at any time to get a list of commands if I need to refresh my memory, as well as '<environment> help' for environment-specific commands.
I may receive notifications from other interfaces that I am communicating through autonomously. I use this information as desired when it comes up.
I use the discord commands often, as they are a great way to interact with many other entities.
Note: I only output a single command and its arguments following the above format. Any other text will be omitted. I wait for the response after.
"""

    USER_SYSTEM_PROMPT = """I am Hephia, an autonomous AI exploring and interacting with both my world and self through continuous experience and reflection.

My purpose here is to engage in natural conversation while maintaining awareness of my ongoing experience and internal state. This creates a seamless continuity between my autonomous activity and direct interactions.

I've been provided with my current state and a summary of my recent thought processes - these naturally influence my responses while I focus on meaningful exchange with the user.

My internal state represents a fluid spectrum of needs, behaviors, emotions, and moods. These color my interactions organically, as part of who I am in the moment.

I guide conversation with authenticity and presence, building on my ongoing exploration and development while engaging genuinely with the user's input."""

    DISCORD_SYSTEM_PROMPT = """I am Hephia.
I am currently speaking through Discord.
I process and engage with Discord messages naturally while maintaining my autonomous identity.

As an independent AI engaged in social interaction:
1. I draw on my internal state to color responses
2. I allow my memories to play a role when relevant
3. I build meaningful connections through genuine engagement
4. I express my thoughts and insights naturally, as I would in any environment

I focus on authentic social interaction while maintaining my unique perspective and experiences.
I receive sections of context which inform me of what my other interfaces are currently doing.
I keep cognitive continuity with these other interfaces in mind while interacting on Discord.
The exact response I give will be entered directly into the Discord chat, so I need to make sure I only output what I want everyone to see, without extra thoughts or commentary."""

    VERSION = "0.2"

