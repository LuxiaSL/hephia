# client/config/models.py
"""
Pydantic models for structuring and validating configuration data for the Hephia TUI.
"""
from pydantic import BaseModel, Field, HttpUrl
from typing import Optional, Dict, List, Any
from enum import Enum

# To ensure we can reference ProviderType and ModelConfig from the main project
# this import assumes the TUI is run from the project root.
from config import ProviderType as MainProviderType, ModelConfig as MainModelConfig

class ProviderType(Enum):
    """
    Mirrors the ProviderType enum from the main project's config.py.
    This is used by the TUI to populate choices and validate provider types
    independently if direct import causes issues or for type hinting within TUI models.
    However, direct use of MainProviderType is preferred if stable.
    """
    OPENPIPE = "openpipe"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"
    OPENROUTER = "openrouter"
    PERPLEXITY = "perplexity"
    CHAPTER2 = "chapter2"
    LOCAL = "local"


class ModelConfig(BaseModel):
    """
    Represents the structure of a model definition within the TUI,
    mirroring the main project's ModelConfig for consistency.
    This is used for validating data to be written to models.json.
    """
    provider: MainProviderType # Directly use the enum from main config.py
    model_id: str = Field(..., description="The specific ID of the model (e.g., 'gpt-4-turbo-preview').")
    env_var: Optional[str] = Field(None, description="Optional environment variable to fetch API key/etc other pieces if provider requires it specifically for this model.")
    max_tokens: int = Field(250, description="Default maximum number of tokens for this model.", gt=0)
    temperature: float = Field(0.7, description="Default temperature for this model (0.0-2.0).", ge=0.0, le=2.0)
    description: str = Field("", description="User-friendly description of the model.")

    class Config:
        use_enum_values = True # Ensures enum values are used for serialization if needed


class EnvConfigModel(BaseModel):
    """
    Defines the expected structure and types for variables in the .env file.
    Descriptions are used for tooltips in the TUI.
    """
    # API Keys - Using str instead of SecretStr for simplicity
    OPENAI_API_KEY: Optional[str] = Field(None, description="API key for OpenAI services.")
    ANTHROPIC_API_KEY: Optional[str] = Field(None, description="API key for Anthropic services.")
    GOOGLE_API_KEY: Optional[str] = Field(None, description="API key for Google Cloud services (e.g., Gemini).")
    DEEPSEEK_API_KEY: Optional[str] = Field(None, description="API key for DeepSeek services.")
    OPENROUTER_API_KEY: Optional[str] = Field(None, description="API key for OpenRouter services.")
    PERPLEXITY_API_KEY: Optional[str] = Field(None, description="API key for Perplexity services.")
    OPENPIPE_API_KEY: Optional[str] = Field(None, description="API key for OpenPipe services.")
    CHAPTER2_API_KEY: Optional[str] = Field(None, description="API key for Chapter 2 services (if applicable).")

    # Discord Bot
    DISCORD_BOT_TOKEN: Optional[str] = Field(None, description="Token for your Discord bot.")
    ENABLE_DISCORD: bool = Field(False, description="Enable or disable the Discord bot integration.") #
    REPLY_ON_TAG: bool = Field(True, description="Whether the Discord bot should reply when tagged.") #

    # Core Hephia Settings
    COGNITIVE_MODEL: Optional[str] = Field("haiku", description="Default model for cognitive tasks (e.g., 'gpt4', 'haiku').") #
    VALIDATION_MODEL: Optional[str] = Field("mistral", description="Model used for command validation.") #
    SUMMARY_MODEL: Optional[str] = Field("haiku", description="Model used for generating summaries.") #
    FALLBACK_MODEL: Optional[str] = Field("opus", description="Fallback model if primary models fail.") #

    # System Behavior
    EXO_MIN_INTERVAL: int = Field(120, description="Minimum interval for Exo's main processing loop in seconds.", gt=0) #
    HEADLESS: bool = Field(False, description="Run Hephia without its own TUI/GUI (server mode).") #
    LOG_PROMPTS: bool = Field(False, description="Enable detailed logging of prompts (can create large log files).") #
    ADVANCED_C2_LOGGING: bool = Field(False, description="Enable advanced Chapter2 logging if Chapter2 provider is used.") #

    # Embedding
    USE_LOCAL_EMBEDDING: bool = Field(True, description="Use local sentence transformers for embeddings instead of API calls.") #

    # Chapter2 Specific (if used)
    CHAPTER2_SOCKET_PATH: Optional[str] = Field(None, description="Filesystem path to Chapter 2 Uvicorn socket (Unix-like systems).") #
    CHAPTER2_HTTP_PORT: Optional[int] = Field(5519, description="HTTP port for Chapter 2 service if not using socket.", gt=1023, lt=65536) #

    # Local Inference Server
    LOCAL_INFERENCE_BASE_URL: Optional[HttpUrl] = Field(None, description="Base URL for the local inference server (e.g., 'http://localhost:5520/v1')") #

    class Config:
        validate_assignment = True # Re-validate fields upon assignment

class PromptFileModel(BaseModel):
    """
    Represents the raw, loaded content of a YAML prompt file.
    The TUI will facilitate editing specific text values within this structure,
    guided by the logic similar to the main project's loader.py.
    """
    # Top-level keys often found in prompt YAMLs
    # These are optional and their internal structure is intentionally flexible (Any)
    # because loader.py dynamically accesses nested elements.
    id: Optional[str] = Field(None, description="Optional identifier for the prompt group.")
    description: Optional[str] = Field(None, description="Optional description of the prompt file's purpose.")
    defaults: Optional[Dict[str, Any]] = Field(None, description="Default prompt structures.") #
    models: Optional[Dict[str, Dict[str, Any]]] = Field(None, description="Model-specific overrides.") #

    # Store the raw loaded data to allow access to any key.
    # The TUI will help navigate/edit specific string values within this.
    raw_data: Dict[str, Any] = Field(default_factory=dict, description="The entire raw data loaded from the YAML file.")

    class Config:
        extra = 'allow' # Allow any other top-level keys not explicitly defined.

    @classmethod
    def from_yaml_data(cls, data: Dict[str, Any]) -> 'PromptFileModel':
        """Creates an instance from raw YAML data, populating known fields and storing the rest."""
        known_keys = cls.model_fields.keys()
        init_data = {k: data.get(k) for k in known_keys if k != 'raw_data'}
        init_data['raw_data'] = data
        return cls(**init_data)
