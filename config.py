# config.py

"""
Configuration settings for the Hephia project.
"""

from enum import Enum
from typing import Optional
from dataclasses import dataclass
from pathlib import Path
import os
import platform
import json

class ProviderType(Enum):
    """Available LLM providers."""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"
    OPENROUTER = "openrouter"
    LOCAL = "local"

@dataclass
class ModelConfig:
    """Configuration for a specific model."""
    provider: ProviderType
    model_id: str
    env_var: Optional[str] = None
    max_tokens: int = 250
    temperature: float = 0.95
    description: str = ""

class Config:
    """
    Centralized configuration management.
    """

    """
    i provide a lot of default models, but i may miss some/you may want to configure or tweak them!
    so, you can create a `models.json` file in your config directory (~/.config/hephia/models.json or %APPDATA%/hephia/models.json) to override or add models.

    format:
    {
        "model_name": {
            "provider": "openrouter",  # or "anthropic", "google", etc.
            "model_id": "vendor/your-model-id", # or "model-id" for non-openrouter providers,
            "max_tokens": 300,
            "temperature": 0.8,
            "description": "Your custom model description",
            "env_var": "LOCAL_INFERENCE_BASE_URL" # optional, only needed for local inference models
        }
    }

    use the same provider strings as in the ProviderType enum, and make sure you use the model_name in your .env files to reference your custom models!
    if your desired provider is not listed, let me know and i'll add it to the clients!
    """

    @classmethod
    def load_user_models(cls):
        """Load and merge user-defined models with defaults."""
        # First, look for a user config directory
        config_dir = (
            Path.home() / ".config" / "hephia" 
            if platform.system() != "Windows"
            else Path(os.getenv("APPDATA", str(Path.home()))) / "hephia"
        )
        
        # Look for models.json in that directory
        models_file = config_dir / "models.json"
        
        if not models_file.exists():
            return  # No user models defined
        
        try:
            with open(models_file, "r", encoding="utf-8") as f:
                user_models = json.load(f)
            
            # Convert JSON to ModelConfig objects and merge with AVAILABLE_MODELS
            for name, model_data in user_models.items():
                try:
                    # Validate required fields
                    if "provider" not in model_data or "model_id" not in model_data:
                        print(f"Warning: Skipping user model '{name}' - missing required fields")
                        continue
                    
                    # Convert provider string to enum
                    provider = ProviderType(model_data["provider"])
                    
                    # Create ModelConfig with defaults for optional fields
                    cls.AVAILABLE_MODELS[name] = ModelConfig(
                        provider=provider,
                        model_id=model_data["model_id"],
                        env_var=model_data.get("env_var"),
                        max_tokens=model_data.get("max_tokens", 250),
                        temperature=model_data.get("temperature", 1.0),
                        description=model_data.get("description", f"User-defined {name} model")
                    )
                    print(f"Loaded user model: {name}")
                except (ValueError, KeyError) as e:
                    print(f"Error loading user model '{name}': {e}")
        
        except Exception as e:
            print(f"Error loading user models: {e}")

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
    def get_pet_model(cls) -> str:
        """Get the pet model from env or default."""
        return os.getenv("PET_MODEL", cls.PET_MODEL)

    @classmethod
    def get_worker_model(cls) -> str:
        """Get the worker model from env or default."""
        return os.getenv("WORKER_MODEL", cls.WORKER_MODEL)
    
    @classmethod
    def get_use_local_embedding(cls) -> bool:
        """Get the embedding type configuration from env or default."""
        return os.getenv("USE_LOCAL_EMBEDDING", "True").lower() in ("true", "1", "yes")
    
    @classmethod
    def get_discord_enabled(cls) -> bool:
        return os.getenv("ENABLE_DISCORD", "False").lower() in ("true", "1", "yes")

    @classmethod
    def get_discord_reply_on_tag(cls) -> bool:
        return os.getenv("REPLY_ON_TAG", "True").lower() in ("true", "1", "yes")
    
    @classmethod
    def get_exo_min_interval(cls) -> float:
        return float(os.getenv("EXO_MIN_INTERVAL", "60.0"))
    
    @classmethod
    def get_headless(cls) -> bool:
        return os.getenv("HEADLESS", "False").lower() in ("true", "1", "yes")
    
    @classmethod
    def get_log_prompts(cls) -> bool:
        return os.getenv("LOG_PROMPTS", "False").lower() in ("true", "1", "yes")
    
    @classmethod
    def get_exo_max_turns(cls) -> int:
        return int(os.getenv("EXO_MAX_TURNS", "50"))

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
        "mischievous-sonnet": ModelConfig(
            provider=ProviderType.ANTHROPIC,
            model_id="claude-3-7-sonnet-latest",
            max_tokens=400,
            description="Claude 3.7 Sonnet via Anthropic"
        ),
        "new-sonnet": ModelConfig(
            provider=ProviderType.ANTHROPIC,
            model_id="claude-3-5-sonnet-20241022",
            max_tokens=400,
            description="Claude 3.6 Sonnet via Anthropic"
        ),
        "old-sonnet": ModelConfig(
            provider=ProviderType.ANTHROPIC,
            model_id="claude-3-5-sonnet-20240620",
            max_tokens=600,
            description="Claude 3.5 Sonnet via Anthropic"
        ),
        "opus": ModelConfig(
            provider=ProviderType.ANTHROPIC,
            model_id="claude-3-opus-20240229",
            max_tokens=550,
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
        "llama-405b-base": ModelConfig(
            provider=ProviderType.OPENROUTER,
            model_id="meta-llama/llama-3.1-405b",
            max_tokens=550,
            description="LLaMA 3.1 405B Base via OpenRouter"
        ),
        "deepseek-v3-base": ModelConfig(
            provider=ProviderType.OPENROUTER,
            model_id="deepseek/deepseek-v3-base:free",
            max_tokens=550,
            description="DeepSeek Base via OpenRouter"
        ),
        "deepseek-r1-zero": ModelConfig(
            provider=ProviderType.OPENROUTER,
            model_id="deepseek/deepseek-r1-zero:free",
            max_tokens=550,
            description="DeepSeek R1 Zero via OpenRouter"
        ),
        "arliai-32b-free": ModelConfig(
            provider=ProviderType.OPENROUTER,
            model_id="arliai/qwq-32b-arliai-rpr-v1:free",
            max_tokens=550,
            description="QwQ 32B ArliAI RPR via OpenRouter"
        ),
        "local-model": ModelConfig(
            provider=ProviderType.LOCAL,
            model_id="foo",
            max_tokens=550,
            description="Local Inference Model",
            env_var="LOCAL_INFERENCE_BASE_URL"
        ),
        "haiku-4.5": ModelConfig(
            provider=ProviderType.ANTHROPIC,
            model_id="claude-haiku-4-5-20251001",
            max_tokens=400,
            temperature=0.9,
            description="Claude Haiku 4.5 via Anthropic"
        ),
        "opus-4.6": ModelConfig(
            provider=ProviderType.ANTHROPIC,
            model_id="claude-opus-4-6",
            max_tokens=550,
            temperature=0.9,
            description="Claude Opus 4.6 via Anthropic"
        ),
    }

    COGNITIVE_MODEL = "haiku"
    VALIDATION_MODEL = "mistral"
    SUMMARY_MODEL = "haiku"
    FALLBACK_MODEL = "haiku"
    PET_MODEL = "haiku-4.5"
    WORKER_MODEL = "opus-4.6"

    # Agent SDK settings (for worker task queue)
    AGENT_MAX_TURNS = 200
    AGENT_MAX_BUDGET_USD = 2.0
    AGENT_ALLOWED_TOOLS = [
        "Read", "Write", "Edit", "Bash", "Glob", "Grep", "WebSearch", "WebFetch",
    ]
    # Working directory for the agent. Empty string = user home directory (blanket access).
    # Set to a specific path to restrict the agent's primary working directory.
    AGENT_CWD = ""
    # Optional comma-separated list of additional directories the agent can access.
    # Only meaningful when AGENT_CWD is set to a specific directory.
    # Empty = no additional dirs (if CWD is set) or unrestricted (if CWD is home).
    AGENT_ALLOWED_DIRS: list[str] = []

    @classmethod
    def get_agent_max_turns(cls) -> int:
        """Get max agent turns from env or default."""
        return int(os.getenv("AGENT_MAX_TURNS", str(cls.AGENT_MAX_TURNS)))

    @classmethod
    def get_agent_max_budget(cls) -> float:
        """Get max agent budget (USD) from env or default."""
        return float(os.getenv("AGENT_MAX_BUDGET_USD", str(cls.AGENT_MAX_BUDGET_USD)))

    @classmethod
    def get_agent_allowed_tools(cls) -> list[str]:
        """Get allowed agent tools from env or default."""
        env_val = os.getenv("AGENT_ALLOWED_TOOLS")
        if env_val:
            return [t.strip() for t in env_val.split(",") if t.strip()]
        return list(cls.AGENT_ALLOWED_TOOLS)

    @classmethod
    def get_agent_cwd(cls) -> str:
        """Get agent working directory. Empty/unset = user home (blanket access)."""
        cwd = os.getenv("AGENT_CWD", cls.AGENT_CWD).strip()
        if not cwd:
            return str(Path.home())
        return cwd

    @classmethod
    def get_agent_allowed_dirs(cls) -> list[str]:
        """Get additional directories the agent can access."""
        env_val = os.getenv("AGENT_ALLOWED_DIRS")
        if env_val:
            return [d.strip() for d in env_val.split(",") if d.strip()]
        return list(cls.AGENT_ALLOWED_DIRS)

    DISCORD_BOT_URL = "http://localhost:5518"

    MAX_STICKY_NOTES = 3

    # internal timers (in seconds)
    NEED_UPDATE_TIMER = 5
    EMOTION_UPDATE_TIMER = 1
    MEMORY_UPDATE_TIMER = 180 # think about this deeper

    # Processor settings
    EXO_TIMEOUT = 100.0
    LLM_TIMEOUT = 90.0     
    INITIALIZATION_TIMEOUT = 30
    SHUTDOWN_TIMEOUT = 15

    # lower = more memories, higher = less memories
    MEMORY_SIGNIFICANCE_THRESHOLD = 0.625
    
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

    # --- Layer 1: Temporal Dynamics ---

    # Mood inertia — blend rate toward target (0 = frozen, 1 = instant snap)
    MOOD_BLEND_RATE = 0.3

    # Need urgency curve: rate *= (1 + CURVE_K * (value/max)^CURVE_N)
    # Applied to hunger, thirst, boredom, loneliness only (not stamina)
    NEED_CURVE_K = 2.0      # max multiplier at full urgency: 1 + K = 3x
    NEED_CURVE_N = 2.0      # exponent controlling curve steepness

    # Post-satisfaction afterglow (temporary rate reduction after action)
    AFTERGLOW_FACTOR = 0.4   # rate multiplied by this during afterglow
    AFTERGLOW_DURATION = 60   # seconds for afterglow to fully decay

    # Behavior characteristic durations (seconds before transition urgency builds)
    BEHAVIOR_CHARACTERISTIC_TIMES = {
        'idle': 300,      # 5 min before restlessness
        'walk': 180,      # 3 min walk sessions
        'chase': 60,      # 1 min bursts
        'sleep': 600,     # 10 min deep sleep
        'relax': 240,     # 4 min rest
    }

    # --- Layer 2: Cross-System Coupling ---

    # Emotion threshold — minimum need change to generate emotional vector
    EMOTION_NEED_CHANGE_THRESHOLD = 1.5

    # Mood coloring — how much mood biases incoming stimulus valence
    MOOD_VALENCE_BIAS_FACTOR = 0.15

    # Conversation impact on needs (per turn)
    CONVERSATION_LONELINESS_REDUCTION = 3.0
    CONVERSATION_BOREDOM_REDUCTION = 1.5

    # Need cross-talk coupling strengths (modifier on target need's rate)
    # Format: (source_need, condition, target_need, modifier)
    # condition is satisfaction threshold below which coupling activates
    NEED_CROSSTALK = [
        # low stamina → faster boredom (too tired to self-entertain)
        ('stamina', 0.4, 'boredom', 0.15),
        # high loneliness → faster hunger (comfort-seeking)
        ('loneliness', 0.4, 'hunger', 0.10),
        # high boredom → faster loneliness (nothing to distract)
        ('boredom', 0.4, 'loneliness', 0.10),
    ]

    # Mood influence on behavior transition weights
    MOOD_BEHAVIOR_INFLUENCE = 0.3  # strength of mood bias on transitions

    VERSION = "0.3"

