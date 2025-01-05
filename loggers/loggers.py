"""
Specialized loggers for different Hephia subsystems.
Provides clean interfaces for specific logging needs.
"""

import logging
import json
from typing import Dict, Any, List, Optional

class SystemLogger:
    """Logger for internal system operations."""
    
    @staticmethod
    def log_api_request(
        service: str,
        endpoint: str,
        status: Optional[int] = None,
        error: Optional[str] = None
    ):
        """Log API request details."""
        logger = logging.getLogger('hephia.system')
        
        if error:
            logger.error(
                f"API Error ({service}):\n"
                f"  Endpoint: {endpoint}\n"
                f"  Status: {status}\n"
                f"  Error: {error}"
            )
        else:
            logger.debug(
                f"API Request ({service}):\n"
                f"  Endpoint: {endpoint}\n"
                f"  Status: {status}"
            )

    @staticmethod
    def log_api_retry(service: str, attempt: int, max_retries: int, reason: str):
        """Log API retry attempts."""
        logger = logging.getLogger('hephia.system')
        logger.warning(
            f"API Retry ({service}):\n"
            f"  Attempt: {attempt}/{max_retries}\n"
            f"  Reason: {reason}"
        )

class InternalLogger:
    """Logger for internal state changes."""
    
    @staticmethod
    def log_state_change(component: str, old_value: Any, new_value: Any):
        """Log a state change with before/after values."""
        logger = logging.getLogger('hephia.internal')
        logger.debug(
            f"State Change: {component}\n"
            f"  From: {old_value}\n"
            f"  To:   {new_value}"
        )
        logger.info(f"Internal: {component} changed to {new_value}")
    
    @staticmethod
    def log_behavior(behavior: str, context: Optional[Dict] = None):
        """Log behavior execution with context."""
        logger = logging.getLogger('hephia.internal')
        if context:
            logger.debug(
                f"Behavior: {behavior}\n"
                f"Context: {json.dumps(context, indent=2)}"
            )
        else:
            logger.debug(f"Behavior: {behavior}")
        logger.info(f"Internal: Performed {behavior}")

class BrainLogger:
    """Enhanced logger for cognitive operations."""
    
    @staticmethod
    def log_llm_exchange(messages: List[Dict], response: str):
        """Log LLM interaction with clear formatting."""
        logger = logging.getLogger('hephia.brain')
        logger.debug(
            "LLM EXCHANGE\n"
            f"Context (last 5 messages):\n"
            f"{json.dumps(messages[-5:], indent=2)}\n"
            f"Response:\n{response}"
        )
        # Clean version for console
        logger.info(f"LLM: {response[:100]}...")
    
    @staticmethod
    def log_command_processing(
        raw_input: Dict,
        processed: Optional[str],
        validation_result: Optional[str]
    ):
        """Log command processing with clear stages."""
        logger = logging.getLogger('hephia.brain')
        logger.debug(
            "COMMAND PROCESSING\n"
            f"Raw Input:\n{json.dumps(raw_input, indent=2)}\n"
            f"Processed Command: {processed}\n"
            f"Validation: {validation_result or 'Success'}"
        )
        if processed:
            logger.info(f"Command: {processed}")
        if validation_result:
            logger.warning(f"Command failed: {validation_result}")
    
    @staticmethod
    def log_turn_start():
        """Log the start of an exo loop turn."""
        logger = logging.getLogger('hephia.brain')
        logger.debug("Starting new exo loop turn")
    
    @staticmethod
    def log_turn_end(success: bool, reason: Optional[str] = None):
        """Log the end of an exo loop turn."""
        logger = logging.getLogger('hephia.brain')
        status = "completed successfully" if success else "failed"
        if reason:
            logger.debug(f"Exo loop turn {status}: {reason}")
        else:
            logger.debug(f"Exo loop turn {status}")