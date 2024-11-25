"""
Logging utilities for Hephia - handles log organization and formatting.
"""

import logging
import json
import sys
from typing import Dict, Any, List, Optional
from datetime import datetime
from pathlib import Path

class LogFormatter:
    """Custom formatters for different log outputs."""
    
    @staticmethod
    def get_file_formatter():
        return logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    @staticmethod
    def get_console_formatter():
        return logging.Formatter('=== %(message)s ===')

class LogManager:
    """Manages log file creation and organization."""
    
    @staticmethod
    def get_log_path() -> Path:
        """Create and return path for current session's log file."""
        log_dir = Path('data/logs')
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # Create timestamped log file
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        return log_dir / f"hephia_{timestamp}.log"

def setup_logging():
    """Setup centralized logging with file and console outputs."""
    # Get root logger
    logger = logging.getLogger('hephia')
    logger.setLevel(logging.DEBUG)
    
    # Clear any existing handlers
    logger.handlers = []
    
    # File handler with full detail
    file_handler = logging.FileHandler(LogManager.get_log_path())
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(LogFormatter.get_file_formatter())
    
    # Console handler with less detail
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(LogFormatter.get_console_formatter())
    
    # Add handlers
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

class ExoLogger:
    """Enhanced logging utility for exo processor operations."""
    
    @staticmethod
    def log_state_update(state: Dict[str, Any]):
        """Log state updates with detailed formatting."""
        logger = logging.getLogger('hephia.exo')
        
        # Detailed file log
        formatted_state = json.dumps(state, indent=2)
        logger.debug(f"State Update:\n{formatted_state}")
        
        # Concise console output
        mood = state.get('pet_state', {}).get('mood', {}).get('name', 'unknown')
        behavior = state.get('pet_state', {}).get('behavior', {}).get('name', 'unknown')
        logger.info(f"State Updated - Mood: {mood}, Behavior: {behavior}")
    
    @staticmethod
    def log_llm_exchange(messages: List[Dict[str, str]], response: str):
        """Log LLM interactions with context."""
        logger = logging.getLogger('hephia.exo')
        
        # Format messages for logging
        messages_formatted = json.dumps(messages[-3:], indent=2)  # Last 3 messages for context
        
        logger.debug(
            f"LLM Exchange:\n"
            f"Context Messages:\n{messages_formatted}\n"
            f"Response: {response}"
        )
        logger.info(f"LLM Response: {response[:100]}...")
    
    @staticmethod
    def log_command_processing(raw_command: str, processed_command: Optional[str], help_text: Optional[str]):
        """Log command processing with clear formatting."""
        logger = logging.getLogger('hephia.exo')
        logger.debug(
            "Command Processing:\n"
            f"Raw Input: {raw_command}\n"
            f"Processed: {processed_command or 'None'}\n"
            f"Help/Error: {help_text or 'None'}"
        )
        
        if processed_command:
            logger.info(f"Processed Command: {processed_command}")
        elif help_text:
            logger.info(f"Command Error: {help_text}")
    
    @staticmethod
    def log_turn_start():
        """Log the start of an exo loop turn."""
        logger = logging.getLogger('hephia.exo')
        logger.debug("Starting new exo loop turn")
    
    @staticmethod
    def log_turn_end(success: bool, reason: Optional[str] = None):
        """Log the end of an exo loop turn."""
        logger = logging.getLogger('hephia.exo')
        status = "completed successfully" if success else "failed"
        if reason:
            logger.debug(f"Exo loop turn {status}: {reason}")
        else:
            logger.debug(f"Exo loop turn {status}")