"""
Central logging management for Hephia.
Handles logger setup and configuration.
"""

import logging
import os
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
from .formatters import InternalFormatter, ExoLoopFormatter, MemoryFormatter, EventFormatter

class LogManager:
    """Enhanced log management with multiple output streams."""
    
    @staticmethod
    def setup_logging():
        """Initialize all loggers with appropriate handlers and configurable levels."""
        # Load environment variables from .env file
        load_dotenv()
        
        # Mapping from level name (string) to logging level (int)
        LEVELS = {
            'CRITICAL': logging.CRITICAL,
            'ERROR': logging.ERROR,
            'WARNING': logging.WARNING,
            'INFO': logging.INFO,
            'DEBUG': logging.DEBUG,
        }

        # Create log directories
        log_base = Path('data/logs')
        internal_dir = log_base / 'internal'
        exoloop_dir = log_base / 'brain'
        system_dir = log_base / 'system'
        memory_dir = log_base / 'memory'
        event_dir = log_base / 'events'
        for directory in [internal_dir, exoloop_dir, system_dir, memory_dir, event_dir]:
            directory.mkdir(parents=True, exist_ok=True)
            
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Define logger configuration: each logger gets a FileHandler with UTF-8.
        config = {
            'hephia.internal': (internal_dir / f"internal_{timestamp}.log", InternalFormatter()),
            'hephia.system': (system_dir / f"system_{timestamp}.log", InternalFormatter()),
            'hephia.brain': (exoloop_dir / f"brain_{timestamp}.log", ExoLoopFormatter()),
            'hephia.memory': (memory_dir / f"memory_{timestamp}.log", MemoryFormatter()),
            'hephia.events': (event_dir / f"events_{timestamp}.log", EventFormatter())
        }
        
        for logger_name, (log_file, formatter) in config.items():
            logger = logging.getLogger(logger_name)
            
            # Construct the environment variable key from the logger name
            # e.g., 'hephia.system' -> 'LOG_LEVEL_HEPHIA_SYSTEM'
            env_var_key = f"LOG_LEVEL_{logger_name.upper().replace('.', '_')}"
            
            # Get the log level from the environment, defaulting to 'INFO' if not set
            log_level_name = os.getenv(env_var_key, 'INFO')
            log_level = LEVELS.get(log_level_name.upper(), logging.INFO)
            
            # Set the logger's level based on the configuration
            logger.setLevel(log_level)
            
            # File handler with UTF-8 encoding
            file_handler = logging.FileHandler(log_file, encoding='utf-8')
            file_handler.setFormatter(formatter)
            
            # Avoid adding duplicate handlers if this function is called more than once
            if not logger.hasHandlers():
                logger.addHandler(file_handler)