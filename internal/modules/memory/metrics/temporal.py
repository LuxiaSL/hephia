"""
\metrics\temporal.py

Implements temporal analysis for memory retrieval and pattern detection.
Handles decay curves, echo timing, and access patterns.

Key capabilities:
- Basic time decay calculations
- Echo recency and dampening effects
- Access history analysis
- Support for temporal pattern detection
"""

from typing import Dict, Optional, Any, List, Union
import math
import time



from loggers.loggers import MemoryLogger

class TemporalMetricsError(Exception):
    """Base exception for temporal metrics calculation failures."""
    pass

class TemporalMetricsCalculator:
    """
    Calculates temporal metrics for memory retrieval.
    Handles formation time, access patterns, and echo effects.
    """
    
    def __init__(self, echo_window: float = 180.0):
        """
        Initialize calculator with timing parameters.
        
        Args:
            echo_window: Time window (seconds) for echo dampening
        """
        self.echo_window = echo_window
        self.logger = MemoryLogger
        
    def calculate_metrics(
        self,
        node_timestamp: float,
        current_time: Optional[float] = None,
        last_accessed: Optional[float] = None,
        last_echo_time: Optional[float] = None,
        echo_dampening: Optional[float] = None,
        is_cognitive: bool = True
    ) -> Dict[str, float]:
        """
        Calculate comprehensive temporal metrics.
        
        Args:
            node_timestamp: Formation time of the memory
            current_time: Current time (defaults to time.time())
            last_accessed: Last access timestamp (if available)
            last_echo_time: Last echo timestamp (cognitive only)
            echo_dampening: Current echo dampening factor
            is_cognitive: Whether this is a cognitive memory
            
        Returns:
            Dict containing temporal metrics:
            - recency: Basic time decay score
            - access_recency: Time since last access
            - echo_recency: Time since last echo (cognitive only)
            - echo_dampening: Current dampening factor (cognitive only)
        """
        try:
            if current_time is None:
                current_time = time.time()
                
            metrics = {}
            
            # Basic time decay (1-hour decay curve)
            time_diff = current_time - node_timestamp
            metrics['recency'] = math.exp(-time_diff / 3600)
            
            # Access history
            if last_accessed is not None:
                access_diff = current_time - last_accessed
                metrics['access_recency'] = math.exp(-access_diff / 3600)
                
            # Echo effects (cognitive only)
            if is_cognitive and last_echo_time is not None:
                echo_diff = current_time - last_echo_time
                metrics['echo_recency'] = math.exp(-echo_diff / self.echo_window)
                
                if echo_dampening is not None:
                    metrics['echo_dampening'] = echo_dampening
                    
                # Add echo potential rating
                metrics['echo_potential'] = self._calculate_echo_potential(
                    echo_diff,
                    echo_dampening or 1.0
                )
            
            # EXPANSION POINT: Pattern detection across time windows
            # EXPANSION POINT: Rhythm analysis in access patterns
            # EXPANSION POINT: Memory consolidation timing
            
            return metrics
            
        except Exception as e:
            self.logger.log_error(f"Temporal metrics calculation failed: {str(e)}")
            return self._get_fallback_metrics(is_cognitive)
            
    def _calculate_echo_potential(
        self,
        time_since_echo: float,
        current_dampening: float
    ) -> float:
        """
        Calculate potential for new echo effect.
        
        Args:
            time_since_echo: Seconds since last echo
            current_dampening: Current echo dampening factor
            
        Returns:
            float: Echo potential score [0,1]
        """
        if time_since_echo > self.echo_window:
            # Full echo potential after window
            return 1.0
        
        # Progressive recovery within window
        recovery = time_since_echo / self.echo_window
        base_potential = min(1.0, recovery * 1.5)  # Allow faster initial recovery
        
        # Apply current dampening
        return base_potential * current_dampening
        
    def analyze_temporal_patterns(
        self,
        timestamps: List[float],
        current_time: Optional[float] = None
    ) -> Dict[str, float]:
        """
        Analyze patterns in temporal data.
        
        EXPANSION POINT: Enhanced pattern recognition
        - Access rhythm detection
        - Memory consolidation patterns
        - Temporal clustering analysis
        
        Args:
            timestamps: List of relevant timestamps
            current_time: Current time reference
            
        Returns:
            Dict of pattern metrics
        """
        if not timestamps:
            return {}
            
        if current_time is None:
            current_time = time.time()
            
        try:
            patterns = {}
            
            # Sort timestamps
            sorted_times = sorted(timestamps)
            
            # Calculate basic statistics
            intervals = [
                t2 - t1 
                for t1, t2 in zip(sorted_times[:-1], sorted_times[1:])
            ]
            
            if intervals:
                # Average interval
                patterns['avg_interval'] = sum(intervals) / len(intervals)
                
                # Interval consistency
                if len(intervals) > 1:
                    variance = sum(
                        (i - patterns['avg_interval']) ** 2 
                        for i in intervals
                    ) / len(intervals)
                    patterns['interval_consistency'] = math.exp(-variance / 3600)
                
            # Recent activity density
            hour_ago = current_time - 3600
            recent_count = sum(1 for t in sorted_times if t > hour_ago)
            patterns['recent_density'] = recent_count / len(timestamps)
            
            return patterns
            
        except Exception as e:
            self.logger.log_error(f"Pattern analysis failed: {str(e)}")
            return {}
            
    def _get_fallback_metrics(self, is_cognitive: bool) -> Dict[str, float]:
        """Provide safe fallback metrics if calculations fail."""
        metrics = {
            'recency': 0.0,
            'access_recency': 0.0
        }
        
        if is_cognitive:
            metrics.update({
                'echo_recency': 0.0,
                'echo_dampening': 1.0,
                'echo_potential': 0.0
            })
            
        return metrics