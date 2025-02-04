"""
Specialized loggers for different Hephia subsystems.
Provides clean interfaces for specific logging needs.
"""

import logging
import json
import sys
import re
from typing import Dict, Any, List, Optional, Union
from brain.commands.model import ParsedCommand

def strip_emojis(text: str) -> str:
    emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
        u"\U00002702-\U000027B0"  # additional symbols
        u"\U000024C2-\U0001F251"
        u"\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs (includes the brain emoji U+1F9E0)
    "]+", flags=re.UNICODE)
    return emoji_pattern.sub('', text)


# --- Custom Safe Handler for Streams ---
class SafeStreamHandler(logging.StreamHandler):
    """
    A stream handler that uses UTF-8 encoding and replaces unencodable characters.
    This prevents UnicodeEncodeError when logging emoji or other characters.
    """
    def emit(self, record):
        try:
            msg = self.format(record)
            stream = self.stream
            # Use 'replace' to substitute characters that can’t be encoded
            stream.write(msg.encode('utf-8', errors='replace').decode('utf-8') + self.terminator)
            self.flush()
        except Exception:
            self.handleError(record)

# --- Logger classes ---
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
                f"API Error ({strip_emojis(service)}):\n"
                f"  Endpoint: {strip_emojis(endpoint)}\n"
                f"  Status: {status}\n"
                f"  Error: {strip_emojis(error)}"
            )
        else:
            logger.debug(
                f"API Request ({strip_emojis(service)}):\n"
                f"  Endpoint: {strip_emojis(endpoint)}\n"
                f"  Status: {status}"
            )

    @staticmethod
    def log_api_retry(service: str, attempt: int, max_retries: int, reason: str):
        """Log API retry attempts."""
        logger = logging.getLogger('hephia.system')
        logger.warning(
            f"API Retry ({strip_emojis(service)}):\n"
            f"  Attempt: {attempt}/{max_retries}\n"
            f"  Reason: {strip_emojis(reason)}"
        )

    @staticmethod
    def error(message: str):
        """Log error-level system events."""
        logger = logging.getLogger('hephia.system')
        logger.error(f"System Error: {strip_emojis(message)}")

    @staticmethod
    def warning(message: str):
        """Log warning-level system events."""
        logger = logging.getLogger('hephia.system')
        logger.warning(f"System Warning: {strip_emojis(message)}")

    @staticmethod
    def debug(message: str):
        """Log debug-level system events."""
        logger = logging.getLogger('hephia.system')
        logger.debug(f"System Debug: {strip_emojis(message)}")

    @staticmethod
    def info(message: str):
        """Log info-level system events."""
        logger = logging.getLogger('hephia.system')
        logger.info(f"System Info: {strip_emojis(message)}")

class InternalLogger:
    """Logger for internal state changes."""
    
    @staticmethod
    def log_state_change(component: str, old_value: Any, new_value: Any):
        logger = logging.getLogger('hephia.internal')
        logger.debug(
            f"State Change: {component}\n"
            f"  From: {old_value}\n"
            f"  To:   {new_value}"
        )
        logger.info(f"Internal: {component} changed to {new_value}")
    
    @staticmethod
    def log_behavior(behavior: str, context: Optional[Dict] = None):
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
        logger = logging.getLogger('hephia.brain')
        logger.error(f"Brain Error: {strip_emojis(message)}")

    @staticmethod
    def warning(message: str):
        logger = logging.getLogger('hephia.brain')
        logger.warning(f"Brain Warning: {strip_emojis(message)}")

    @staticmethod
    def debug(message: str):
        logger = logging.getLogger('hephia.brain')
        logger.debug(f"Brain Debug: {strip_emojis(message)}")

    @staticmethod
    def info(message: str):
        logger = logging.getLogger('hephia.brain')
        logger.info(f"Brain Info: {strip_emojis(message)}")
    
    @staticmethod
    def log_llm_exchange(messages: List[Dict], response: str):
        logger = logging.getLogger('hephia.brain')
        clean_messages = [{k: strip_emojis(str(v)) for k, v in msg.items()} for msg in messages[-5:]]
        clean_response = strip_emojis(response)
        logger.debug(
            "LLM EXCHANGE\n"
            f"Context (last 5 messages):\n"
            f"{json.dumps(clean_messages, indent=2)}\n"
            f"Response:\n{clean_response}"
        )
        logger.info(f"LLM: {clean_response[:100]}...")
    
    @staticmethod
    def log_command_processing(
        raw_input: str,
        processed: Optional[Union[str, 'ParsedCommand']],
        validation_result: Optional[str]
    ):
        logger = logging.getLogger('hephia.brain')
        processed_str = str(processed) if processed else None
        logger.debug(
            "COMMAND PROCESSING\n"
            f"Raw Input:\n{strip_emojis(raw_input)}\n"
            f"Processed Command: {strip_emojis(processed_str) if processed_str else None}\n"
            f"Validation: {strip_emojis(validation_result) if validation_result else 'Success'}"
        )
        if validation_result:
            logger.warning(f"Command failed: {strip_emojis(validation_result)}")
    
    @staticmethod
    def log_turn_start():
        logger = logging.getLogger('hephia.brain')
        logger.debug("Starting new exo loop turn")
    
    @staticmethod
    def log_turn_end(success: bool, reason: Optional[str] = None):
        logger = logging.getLogger('hephia.brain')
        status = "completed successfully" if success else "failed"
        if reason:
            logger.debug(f"Exo loop turn {status}: {strip_emojis(reason)}")
        else:
            logger.debug(f"Exo loop turn {status}")

class MemoryLogger:
    """Logger for memory system operations."""

    @staticmethod
    def log_error(message: str):
        logger = logging.getLogger('hephia.memory')
        logger.error(strip_emojis(message))

    @staticmethod
    def error(message: str):
        logger = logging.getLogger('hephia.memory')
        logger.error(f"Memory Error: {strip_emojis(message)}")

    @staticmethod
    def warning(message: str):
        logger = logging.getLogger('hephia.memory')
        logger.warning(f"Memory Warning: {strip_emojis(message)}")

    @staticmethod
    def debug(message: str):
        logger = logging.getLogger('hephia.memory')
        logger.debug(f"Memory Debug: {strip_emojis(message)}")

    @staticmethod
    def info(message: str):
        logger = logging.getLogger('hephia.memory')
        logger.info(f"Memory Info: {strip_emojis(message)}")

    @staticmethod
    def log_memory_formation(memory_type: str, memory_id: str, details: Dict[str, Any]):
        logger = logging.getLogger('hephia.memory')
        logger.info(
            f"Memory Formation ({strip_emojis(memory_type)}):\n"
            f"  Memory ID: {strip_emojis(memory_id)}\n"
            f"  Details: {json.dumps({k: strip_emojis(str(v)) for k,v in details.items()}, indent=2)}"
        )

    @staticmethod
    def log_memory_retrieval(memory_type: str, memory_id: str, success: bool, query: Optional[str] = None):
        logger = logging.getLogger('hephia.memory')
        if success:
            logger.debug(
                f"Memory Retrieval Success ({strip_emojis(memory_type)}):\n"
                f"  Memory ID: {strip_emojis(memory_id)}\n"
                f"  Query: {strip_emojis(query) if query else None}"
            )
        else:
            logger.warning(
                f"Memory Retrieval Failure ({strip_emojis(memory_type)}):\n"
                f"  Memory ID: {strip_emojis(memory_id)}\n"
                f"  Query: {strip_emojis(query) if query else None}"
            )

    @staticmethod
    def log_memory_decay(memory_type: str, memory_id: str, new_strength: float):
        logger = logging.getLogger('hephia.memory')
        logger.info(
            f"Memory Decay ({strip_emojis(memory_type)}):\n"
            f"  Memory ID: {strip_emojis(memory_id)}\n"
            f"  New Strength: {new_strength}"
        )

    @staticmethod
    def log_memory_merge(memory_type: str, from_memory_id: str, to_memory_id: str):
        logger = logging.getLogger('hephia.memory')
        logger.info(
            f"Memory Merge ({strip_emojis(memory_type)}):\n"
            f"  From Memory ID: {strip_emojis(from_memory_id)}\n"
            f"  To Memory ID: {strip_emojis(to_memory_id)}"
        )

class EventLogger:
    """Logger for event system operations."""
    @staticmethod
    def error(message: str):
        logger = logging.getLogger('hephia.events')
        logger.error(f"Event Error: {strip_emojis(message)}")

    @staticmethod
    def warning(message: str):
        logger = logging.getLogger('hephia.events')
        logger.warning(f"Event Warning: {strip_emojis(message)}")

    @staticmethod
    def debug(message: str):
        logger = logging.getLogger('hephia.events')
        logger.debug(f"Event Debug: {strip_emojis(message)}")

    @staticmethod
    def info(message: str):
        logger = logging.getLogger('hephia.events')
        logger.info(f"Event Info: {strip_emojis(message)}")

    @staticmethod
    def log_event_dispatch(event_type: str, data: Any = None, metadata: Optional[Dict] = None):
        logger = logging.getLogger('hephia.events')
        components = [f"Event Dispatched: {strip_emojis(event_type)}"]
        if data is not None:
            if isinstance(data, dict):
                processed_data = {k: strip_emojis(str(v)) for k, v in data.items()}
                data_str = json.dumps(processed_data, indent=2)
                components.append(f"Data:\n{data_str}")
            elif isinstance(data, list):
                processed_data = [strip_emojis(str(item)) for item in data]
                data_str = json.dumps(processed_data, indent=2)
                components.append(f"Data:\n{data_str}")
            else:
                components.append(f"Data: {strip_emojis(str(data))}")
        if metadata:
            meta_str = json.dumps({k: strip_emojis(str(v)) for k,v in metadata.items()}, indent=2)
            components.append(f"Metadata:\n{meta_str}")
        logger.debug('\n'.join(components))
        logger.info(f"Dispatched: {strip_emojis(event_type)}")
