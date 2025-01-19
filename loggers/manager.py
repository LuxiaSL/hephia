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
        
        # Setup all file handlers first
        loggers = {
            'hephia.internal': (internal_dir / f"internal_{timestamp}.log", InternalFormatter()),
            'hephia.system': (system_dir / f"system_{timestamp}.log", InternalFormatter()),
            'hephia.brain': (exoloop_dir / f"brain_{timestamp}.log", ExoLoopFormatter()),
            'hephia.memory': (memory_dir / f"memory_{timestamp}.log", MemoryFormatter()),
            'hephia.events': (event_dir / f"events_{timestamp}.log", EventFormatter())
        }
        
        for logger_name, (log_file, formatter) in loggers.items():
            logger = logging.getLogger(logger_name)
            logger.setLevel(logging.DEBUG)
            
            # File handler
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
            
            for handler in logger.handlers[:]:
                if isinstance(handler, logging.StreamHandler) and not isinstance(handler, logging.FileHandler):
                    logger.removeHandler(handler)
        
        # Instead of adding console handlers, we'll only log critical errors to stderr
        error_console = logging.StreamHandler(sys.stderr)
        error_console.setLevel(logging.CRITICAL)
        error_console.setFormatter(ConsoleFormatter())
        
        # Only add stderr handler to system logger for critical errors
        system_logger = logging.getLogger('hephia.system')
        system_logger.addHandler(error_console)