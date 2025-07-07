"""
brain/cognition/memory/significance.py

Handles significance analysis for memory formation across different
interaction types. Works with interface-specific significance checks.
"""

from dataclasses import dataclass, field
from typing import Dict, Any
from datetime import datetime
from enum import Enum

from brain.commands.model import ParsedCommand
from config import Config

class SourceType(Enum):
    COMMAND = "command"        # ExoProcessor commands
    DIRECT_CHAT = "direct"     # User interface chat
    DISCORD = "discord"        # Discord messages
    ENVIRONMENT = "environment"  # Environment transitions

@dataclass
class MemoryData:
    """Standardized structure for memory check events."""
    interface_id: str
    content: str               # The main content to be remembered
    context: Dict[str, Any]    # Current cognitive state/context
    source_type: SourceType
    metadata: Dict[str, Any]   # Source-specific metadata
    timestamp: datetime = field(default_factory=datetime.now)

    def to_event_data(self) -> Dict[str, Any]:
        """Convert to an event-safe dictionary structure."""
        return {
            "interface_id": self.interface_id,
            "content": self.content,
            "context": self.context,
            "source_type": self.source_type.value,
            "metadata": self.metadata,
            "timestamp": self.timestamp.isoformat()
        }

    @classmethod
    def from_event_data(cls, data: Dict[str, Any]) -> 'MemoryData':
        """Reconstruct a MemoryData object from event data."""
        return cls(
            interface_id=data["interface_id"],
            content=data["content"],
            context=data["context"],
            source_type=SourceType(data["source_type"]),
            metadata=data["metadata"],
            timestamp=datetime.fromisoformat(data["timestamp"])
        )
    
class SignificanceAnalyzer:
    """Analyzes memory significance across different source types.
    This should eventually pull from the memory network metrics as well."""
    
    def __init__(self):
        self.thresholds: Dict[str, float] = {
            "exo_processor": Config.MEMORY_SIGNIFICANCE_THRESHOLD,
            "discord": Config.MEMORY_SIGNIFICANCE_THRESHOLD * 0.75,
            "user": Config.MEMORY_SIGNIFICANCE_THRESHOLD * 0.75
        }

    def analyze_significance(self, memory_data: MemoryData) -> bool:
        """Analyze significance based on source type and content."""
        score = 0.0
        
        if memory_data.source_type == SourceType.COMMAND:
            score = self._analyze_command_significance(memory_data)
        elif memory_data.source_type == SourceType.DISCORD:
            score = self._analyze_social_significance(memory_data)
        elif memory_data.source_type == SourceType.DIRECT_CHAT:
            score = self._analyze_social_significance(memory_data)
        elif memory_data.source_type == SourceType.ENVIRONMENT:
            score = self._analyze_environment_significance(memory_data)
            
        threshold = self.thresholds.get(memory_data.interface_id, 0.5)
        return score > threshold

    def _analyze_command_significance(self, data: MemoryData) -> float:
        score = 0.0
        metadata = data.metadata

        # Command complexity (0.3)
        command = metadata.get('command')
        if isinstance(command, ParsedCommand) and command.parameters:
            # Each parameter contributes 0.1, up to 0.3 maximum.
            score += min(0.3, len(command.parameters) * 0.1)

        # Response impact (0.3)
        response = metadata.get('response', '')
        if isinstance(response, str):
            words = response.split()
            # Up to 0.15 for word count impact.
            score += min(0.15, len(words) / 100)
        # Additional impact if the command did not succeed.
        if not metadata.get('success', True):
            score += 0.15

        # Environment context (0.2)
        if command and getattr(command, 'environment', None):
            score += 0.2

        # Result impact (0.2)
        result = metadata.get('result')
        if result:
            # Coerce potential None values to defaults.
            message = (getattr(result, 'message', '') or '')
            data_val = (getattr(result, 'data', {}) or {})
            state_changes = (getattr(result, 'state_changes', {}) or {})
            
            impact_score = (len(message.split()) + len(data_val) * 2 + len(state_changes) * 3) / 25
            # Only add if the command wasn’t a trivial help or version request.
            if command and command.action not in ['help', 'version', 'list']:
                score += min(0.2, impact_score)

        return score

    def _analyze_social_significance(self, data: MemoryData) -> float:
        """
        Analyze significance of social interactions with path-based channel references.
        """
        score = 0.0
        metadata = data.metadata
        
        if data.source_type == SourceType.DISCORD:
            # Message length (0.5)
            message = metadata.get('message', {}).get('content', '')
            msg_words = message.split()
            score += min(0.5, len(msg_words) / 50)

            # Interaction depth (0.5)
            if len(metadata.get('history', [])) >= 2:
                score += 0.25
            if metadata.get('mentions_bot'):
                score += 0.25
        else:  # DIRECT_CHAT
            # Conversation depth (0.5)
            conv_data = metadata.get('conversation', {})
            if conv_data.get('has_multi_turn'):
                score += 0.2
            score += min(0.2, conv_data.get('total_messages', 0) * 0.05)

            # Message content (0.5)
            if last_msg := conv_data.get('last_user_message'):
                score += min(0.5, len(last_msg.split()) / 50)

        score = min(score, 1.0)
        return score

    def _analyze_environment_significance(self, data: MemoryData) -> float:
        score = 0.0
        metadata = data.metadata
        
        # Session length (0.4)
        history = metadata.get('history', [])
        score += min(0.4, len(history) * 0.1)
        
        # Command variety (0.3)
        unique_commands = len(set(
            cmd.get('action') for cmd in history 
            if isinstance(cmd, dict) and 'action' in cmd
        ))
        score += min(0.3, unique_commands * 0.1)
        
        # Success rate (0.3)
        successes = sum(1 for cmd in history if cmd.get('success', False))
        if history:
            score += 0.3 * (successes / len(history))
            
        return score