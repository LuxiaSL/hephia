from typing import Dict, Any
from pathlib import Path
import yaml

from .formats import (
    CognitiveFormat,
    InterfaceFormat,
    MemoryFormat,
    Message,
    PromptTemplate,
    BaseMessageFormat
)

class PromptManager:
    """
    Manages prompt templates and their application.
    Handles loading configurations and coordinating formatters.
    """
    
    def __init__(self, config_path: Path):
        self.templates: Dict[str, PromptTemplate] = {}
        self.formatters: Dict[str, BaseMessageFormat] = {}
        self._load_config(config_path)

    def _load_config(self, path: Path) -> None:
        """Load and validate configuration from YAML."""
        with open(path) as f:
            config = yaml.safe_load(f)
            
        core_prompts = config.get("prompts", {})
        for name, content in core_prompts.items():
            self.templates[name] = PromptTemplate(
                name=name,
                content=content["content"],
                required_data=content.get("required_data", []),
                metadata=content.get("metadata", {})
            )

        # Initialize formatters
        for fmt_name, fmt_config in config.get("formats", {}).items():
            formatter_class = self._get_formatter_class(fmt_config["class"])
            self.formatters[fmt_name] = formatter_class(fmt_config)

    def _get_formatter_class(self, class_name: str) -> type:
        """Map format names to concrete formatter classes."""
        # Will be expanded as we add formatters
        formatters = {
            "CognitiveFormat": CognitiveFormat,
            "InterfaceFormat": InterfaceFormat,
            "MemoryFormat": MemoryFormat
        }
        return formatters[class_name]

    async def build_prompt(
        self,
        template_name: str,
        data: Dict[str, Any],
        format_name: str
    ) -> str:
        """
        Build a prompt using specified template and formatter.
        Validates required data is present.
        """
        if template_name not in self.templates:
            raise ValueError(f"Unknown template: {template_name}")
        if format_name not in self.formatters:
            raise ValueError(f"Unknown format: {format_name}")

        template = self.templates[template_name]
        formatter = self.formatters[format_name]

        # Validate required data
        missing = [key for key in template.required_data if key not in data]
        if missing:
            raise ValueError(f"Missing required data: {missing}")

        # Create message from template
        message = Message(
            role="system",
            content=template.content,
            metadata=template.metadata
        )

        # Let formatter handle the specifics
        return formatter.render(message, data)