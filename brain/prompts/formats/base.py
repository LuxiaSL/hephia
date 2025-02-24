"""
Base classes for Hephia's prompt management system.
Handles efficient message formatting while maintaining separation of concerns.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional

@dataclass
class Message:
    """Core message structure."""
    role: str
    content: str
    metadata: Optional[Dict[str, Any]] = None

@dataclass
class PromptTemplate:
    """Core template with variations and context requirements."""
    name: str
    base_template: str
    required_context: List[str]
    format_class: Optional[str] = None  # Name of formatter to use
    model_variants: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def render(self, data: Dict[str, Any], model: Optional[str] = None) -> str:
        """
        Render template with data, using model-specific variant if available.
        
        Args:
            data: Context data for template
            model: Optional model identifier for variants
        """
        # Validate required context
        missing = [key for key in self.required_context if key not in data]
        if missing:
            raise ValueError(f"Missing required context: {missing}")
            
        # Get appropriate template
        template = self.model_variants.get(model, self.base_template)
        
        # Apply template
        try:
            return template.format(**data)
        except KeyError as e:
            raise ValueError(f"Missing template variable: {e}")

class BaseMessageFormat:
    """Base formatter - handles template application."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.templates: Dict[str, PromptTemplate] = {}
        self._load_templates(config.get("templates", {}))
    
    def _load_templates(self, template_config: Dict[str, Any]) -> None:
        """Load and validate templates from config."""
        for name, config in template_config.items():
            if isinstance(config, str):
                # Simple string template
                template = PromptTemplate(
                    name=name,
                    base_template=config,
                    required_context=[],
                    format_class=self.__class__.__name__
                )
            else:
                # Full template configuration
                template = PromptTemplate(
                    name=name,
                    base_template=config["template"],
                    required_context=config.get("required_context", []),
                    format_class=self.__class__.__name__,
                    model_variants=config.get("model_variants", {}),
                    metadata=config.get("metadata", {})
                )
            self.templates[name] = template
    
    def render(self, message: Message, data: Dict[str, Any], model: Optional[str] = None) -> str:
        """
        Render message using appropriate template.
        Base implementation just returns message content.
        """
        return message.content
    
    def parse(self, content: str) -> List[Message]:
        """Parse content back into messages."""
        return [Message(role="assistant", content=content.strip())]

    def merge(self, messages: List[Message], max_length: Optional[int] = None) -> Message:
        """Combine messages respecting length."""
        combined = "\n\n".join(msg.content for msg in messages)
        if max_length:
            combined = combined[:max_length]
        return Message(
            role=messages[-1].role if messages else "system",
            content=combined
        )