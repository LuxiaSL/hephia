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
    def error(message: str):
        """Log error-level brain system events."""
        logger = logging.getLogger('hephia.brain')
        logger.error(f"Brain Error: {message}")

    @staticmethod
    def warning(message: str):
        """Log warning-level brain system events."""
        logger = logging.getLogger('hephia.brain')
        logger.warning(f"Brain Warning: {message}")

    @staticmethod
    def debug(message: str):
        """Log debug-level brain system events."""
        logger = logging.getLogger('hephia.brain')
        logger.debug(f"Brain Debug: {message}")

    @staticmethod
    def info(message: str):
        """Log info-level brain system events."""
        logger = logging.getLogger('hephia.brain')
        logger.info(f"Brain Info: {message}")
    
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
        raw_input: str,
        processed: Optional[str],
        validation_result: Optional[str]
    ):
        """Log command processing with clear stages."""
        logger = logging.getLogger('hephia.brain')
        logger.debug(
            "COMMAND PROCESSING\n"
            f"Raw Input:\n{raw_input}\n"
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

class MemoryLogger:
    """Logger for memory system operations."""

    @staticmethod
    def log_error(message: str):
        logger = logging.getLogger('hephia.memory')
        logger.error(message)

    @staticmethod
    def error(message: str):
        """Log error-level memory system events."""
        logger = logging.getLogger('hephia.memory')
        logger.error(f"Memory Error: {message}")

    @staticmethod
    def warning(message: str):
        """Log warning-level memory system events."""
        logger = logging.getLogger('hephia.memory')
        logger.warning(f"Memory Warning: {message}")

    @staticmethod
    def debug(message: str):
        """Log debug-level memory system events."""
        logger = logging.getLogger('hephia.memory')
        logger.debug(f"Memory Debug: {message}")

    @staticmethod
    def info(message: str):
        """Log info-level memory system events."""
        logger = logging.getLogger('hephia.memory')
        logger.info(f"Memory Info: {message}")

    @staticmethod
    def log_memory_formation(memory_type: str, memory_id: str, details: Dict[str, Any]):
        """Log the formation of a new memory node."""
        logger = logging.getLogger('hephia.memory')
        logger.info(
            f"Memory Formation ({memory_type}):\n"
            f"  Memory ID: {memory_id}\n"
            f"  Details: {json.dumps(details, indent=2)}"
        )

    @staticmethod
    def log_memory_retrieval(memory_type: str, memory_id: str, success: bool, query: Optional[str] = None):
        """Log retrieval attempts of a memory node."""
        logger = logging.getLogger('hephia.memory')
        if success:
            logger.debug(
                f"Memory Retrieval Success ({memory_type}):\n"
                f"  Memory ID: {memory_id}\n"
                f"  Query: {query}"
            )
        else:
            logger.warning(
                f"Memory Retrieval Failure ({memory_type}):\n"
                f"  Memory ID: {memory_id}\n"
                f"  Query: {query}"
            )

    @staticmethod
    def log_memory_decay(memory_type: str, memory_id: str, new_strength: float):
        """Log the decay of a memory node."""
        logger = logging.getLogger('hephia.memory')
        logger.info(
            f"Memory Decay ({memory_type}):\n"
            f"  Memory ID: {memory_id}\n"
            f"  New Strength: {new_strength}"
        )

    @staticmethod
    def log_memory_merge(memory_type: str, from_memory_id: str, to_memory_id: str):
        """Log the merging of two memory nodes."""
        logger = logging.getLogger('hephia.memory')
        logger.info(
            f"Memory Merge ({memory_type}):\n"
            f"  From Memory ID: {from_memory_id}\n"
            f"  To Memory ID: {to_memory_id}"
        )

class EventLogger:
    """Logger for event system operations."""
    @staticmethod
    def error(message: str):
        """Log error-level event system events."""
        logger = logging.getLogger('hephia.events')
        logger.error(f"Event Error: {message}")

    @staticmethod
    def warning(message: str):
        """Log warning-level event system events."""
        logger = logging.getLogger('hephia.events')
        logger.warning(f"Event Warning: {message}")

    @staticmethod
    def debug(message: str):
        """Log debug-level event system events."""
        logger = logging.getLogger('hephia.events')
        logger.debug(f"Event Debug: {message}")

    @staticmethod
    def info(message: str):
        """Log info-level event system events."""
        logger = logging.getLogger('hephia.events')
        logger.info(f"Event Info: {message}")

    @staticmethod
    def log_event_dispatch(event_type: str, data: Any = None, metadata: Optional[Dict] = None):
        """Log event dispatch with formatted data and metadata."""
        logger = logging.getLogger('hephia.events')
        
        # Build message components
        components = [f"Event Dispatched: {event_type}"]
        
        if data is not None:
            if isinstance(data, (dict, list)):
                data_str = json.dumps(data, indent=2)
                components.append(f"Data:\n{data_str}")
            else:
                components.append(f"Data: {str(data)}")
                
        if metadata:
            meta_str = json.dumps(metadata, indent=2)
            components.append(f"Metadata:\n{meta_str}")
        
        # Join with newlines for debug, keep compact for info
        logger.debug('\n'.join(components))
        logger.info(f"Dispatched: {event_type}")