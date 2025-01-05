"""
Central logging management for Hephia.
Handles logger setup and configuration.
"""

import logging
import sys
from pathlib import Path
from datetime import datetime
from .formatters import InternalFormatter, ExoLoopFormatter, ConsoleFormatter

class LogManager:
    """Enhanced log management with multiple output streams."""
    
    @staticmethod
    def setup_logging():
        """Initialize all loggers with appropriate handlers."""
        # Create log directories
        log_base = Path('data/logs')
        internal_dir = log_base / 'internal'
        exoloop_dir = log_base / 'exoloop'
        for directory in [internal_dir, exoloop_dir]:
            directory.mkdir(parents=True, exist_ok=True)
            
        # Setup specific loggers
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Internal/System Logger
        internal_logger = logging.getLogger('hephia.internal')
        internal_logger.setLevel(logging.DEBUG)
        internal_handler = logging.FileHandler(
            internal_dir / f"internal_{timestamp}.log"
        )
        internal_handler.setFormatter(InternalFormatter())
        internal_logger.addHandler(internal_handler)
        
        system_logger = logging.getLogger('hephia.system')
        system_logger.setLevel(logging.DEBUG)
        system_handler = logging.FileHandler(
            internal_dir / f"system_{timestamp}.log"
        )
        system_handler.setFormatter(InternalFormatter())
        system_logger.addHandler(system_handler)
        
        # Brain/Exo Logger
        brain_logger = logging.getLogger('hephia.brain')
        brain_logger.setLevel(logging.DEBUG)
        brain_handler = logging.FileHandler(
            exoloop_dir / f"brain_{timestamp}.log"
        )
        brain_handler.setFormatter(ExoLoopFormatter())
        brain_logger.addHandler(brain_handler)
        
        # Console Handler (for important events)
        console = logging.StreamHandler(sys.stdout)
        console.setLevel(logging.INFO)
        console.setFormatter(ConsoleFormatter())
        
        # Add console handler to all loggers
        for logger in [internal_logger, system_logger, brain_logger]:
            logger.addHandler(console)