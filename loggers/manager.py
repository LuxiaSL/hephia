"""
Central logging management for Hephia.
Handles logger setup and configuration.
"""

import logging
import sys
from pathlib import Path
from datetime import datetime
from .formatters import InternalFormatter, ExoLoopFormatter, ConsoleFormatter, MemoryFormatter, EventFormatter

class LogManager:
    """Enhanced log management with multiple output streams."""
    
    @staticmethod
    def setup_logging():
        """Initialize all loggers with appropriate handlers."""
        # Create log directories
        log_base = Path('data/logs')
        internal_dir = log_base / 'internal'
        exoloop_dir = log_base / 'exoloop'
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
            logger.setLevel(logging.DEBUG)
            
            # File handler with UTF-8 encoding
            file_handler = logging.FileHandler(log_file, encoding='utf-8')
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
            
            # Remove any default stream handlers to prevent unwanted console output.
            # If you wish to add a console handler later, use our SafeStreamHandler.
            for handler in logger.handlers[:]:
                if isinstance(handler, logging.StreamHandler) and not isinstance(handler, logging.FileHandler):
                    logger.removeHandler(handler)
