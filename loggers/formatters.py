"""
Formatters for different log types in Hephia.
Handles consistent formatting across different logging streams.
"""

import json
import logging

class InternalFormatter(logging.Formatter):
    """Formatter for internal and system events."""
    
    def format(self, record):
        timestamp = self.formatTime(record)
        return (
            f"[{timestamp}] {record.levelname:8} "
            f"{record.name.split('.')[-1]:10} | {record.getMessage()}"
        )

class ExoLoopFormatter(logging.Formatter):
    """Formatter for brain/exo events with clear section breaks."""
    
    def format(self, record):
        timestamp = self.formatTime(record)
        msg = record.getMessage()
        
        # Add section breaks for major events
        if "LLM EXCHANGE" in msg:
            return f"\n{'='*80}\n{timestamp} - {msg}\n{'='*80}"
        elif "COMMAND PROCESSING" in msg:
            return f"\n{'-'*80}\n{timestamp} - {msg}\n{'-'*80}"
        else:
            return f"{timestamp} - {msg}"

class MemoryFormatter(logging.Formatter):
    """Formatter for memory system events."""
    
    def format(self, record):
        timestamp = self.formatTime(record)
        msg = record.getMessage()
        return f"[{timestamp}] {record.levelname:8} MEMORY | {msg}"

class PromptFormatter(logging.Formatter):
    """Formatter for prompt logging with clear sections and JSON formatting."""
    
    def format(self, record):
        timestamp = self.formatTime(record)
        msg = record.getMessage()
        
        # Format message as JSON with proper indentation
        try:
            data = json.loads(msg)
            formatted_json = json.dumps(data, indent=2)
            return f"[{timestamp}]\n{formatted_json}\n{'='*80}"
        except json.JSONDecodeError:
            return f"[{timestamp}] {msg}"
    
class EventFormatter(logging.Formatter):
    """Formatter for event dispatcher"""
    
    def format(self, record):
        timestamp = self.formatTime(record)
        return f"[{timestamp}] {record.levelname:8} EVENT | {record.getMessage()}"

class ConsoleFormatter(logging.Formatter):
    """Minimal formatter for console output."""
    
    def format(self, record):
        if record.levelno >= logging.WARNING:
            return f"❗ {record.getMessage()}"
        return f"📝 {record.getMessage()}"