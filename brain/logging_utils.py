# In logging_utils.py
import logging
import json
import sys
from typing import Dict, Any, List, Optional

# Create formatters for different outputs
file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_formatter = logging.Formatter('=== %(message)s ===')

def setup_logging():
    """Setup centralized logging for Hephia."""
    # Get root logger
    logger = logging.getLogger('hephia')
    logger.setLevel(logging.DEBUG)
    
    # Clear any existing handlers
    logger.handlers = []
    
    # File handler with full detail
    file_handler = logging.FileHandler('hephia.log')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(file_formatter)
    
    # Console handler with less detail
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(console_formatter)
    
    # Add handlers
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

class ExoLogger:
    """Logging utility for exo processor operations."""
    
    @staticmethod
    def log_state_update(state: Dict[str, Any]):
        logger = logging.getLogger('hephia')
        # Detailed log to file
        logger.debug(f"State Update:\n{json.dumps(state, indent=2)}")
        # Simple console output
        logger.info(f"Pet State Updated")
    
    @staticmethod
    def log_llm_exchange(messages: List[Dict[str, str]], response: str):
        logger = logging.getLogger('hephia')
        # Detailed log to file
        logger.debug(f"LLM Exchange:\nMessages: {json.dumps(messages, indent=2)}\nResponse: {response}")
        # Simple console output
        logger.info(f"LLM Response: {response[:100]}...")
    
    @staticmethod
    def log_command_processing(raw_command: str, processed_command: Optional[str], help_text: Optional[str]):
        logger = logging.getLogger('hephia')
        logger.debug(
            f"Command Processing:\n"
            f"Raw: {raw_command}\n"
            f"Processed: {processed_command}\n"
            f"Help: {help_text}"
        )