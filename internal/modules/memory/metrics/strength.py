"""
\metrics\strength.py

Implements strength-based analysis for memory networks.
Handles node strength, network position, and ghost relationships.

Key capabilities:
- Basic strength metrics
- Network position analysis
- Ghost node relationships
- Connection strength patterns
"""

from typing import Dict, List, Optional, Any
from loggers.loggers import MemoryLogger

class StrengthMetricsError(Exception):
    """Base exception for strength metrics calculation failures."""
    pass

class StrengthMetricsCalculator:
    """
    Calculates strength-based metrics for memory nodes.
    Analyzes both individual strength and network position.
    """
    
    def __init__(self):
        """Initialize calculator with required dependencies."""
        self.logger = MemoryLogger
        
    def calculate_metrics(
        self,
        node_strength: float,
        connections: Dict[str, float],
        is_ghosted: bool = False,
        ghost_nodes: Optional[List[Dict]] = None,
        connected_strengths: Optional[List[float]] = None
    ) -> Dict[str, float]:
        """
        Calculate comprehensive strength metrics.
        
        Args:
            node_strength: Current strength value of the node
            connections: Dict mapping node IDs to connection strengths
            is_ghosted: Whether node is in ghost state
            ghost_nodes: List of ghost node data
            connected_strengths: Pre-calculated strengths of connected nodes
            
        Returns:
            Dict containing strength metrics:
            - current_strength: Raw strength value
            - relative_strength: Position in local network
            - ghost_metrics: Ghost state information
            - network_metrics: Connection-based calculations
        """
        try:
            metrics = {
                'current_strength': node_strength
            }
            
            # Network position analysis
            if connected_strengths:
                network_metrics = self._analyze_network_position(
                    node_strength,
                    connected_strengths,
                    connections
                )
                metrics.update(network_metrics)
                
            # Ghost state metrics
            ghost_metrics = self._calculate_ghost_metrics(
                is_ghosted,
                ghost_nodes or []
            )
            metrics.update(ghost_metrics)
            
            # EXPANSION POINT: Field effect analysis
            # EXPANSION POINT: Connection pattern recognition
            # EXPANSION POINT: Strength distribution analysis
            
            return metrics
            
        except Exception as e:
            self.logger.log_error(f"Strength metrics calculation failed: {str(e)}")
            return self._get_fallback_metrics()
            
    def _analyze_network_position(
        self,
        node_strength: float,
        connected_strengths: List[float],
        connections: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Analyze node's position in local network.
        
        EXPANSION POINT: Enhanced network analysis
        - Cluster strength patterns
        - Connection topology
        - Field effect calculations
        """
        metrics = {}
        
        try:
            if connected_strengths:
                # Calculate relative strength
                avg_connected = sum(connected_strengths) / len(connected_strengths)
                metrics['relative_strength'] = node_strength / avg_connected if avg_connected > 0 else 0.0
                
                # Connection strength analysis
                connection_values = list(connections.values())
                if connection_values:
                    metrics['avg_connection_strength'] = sum(connection_values) / len(connection_values)
                    metrics['max_connection_strength'] = max(connection_values)
                    
                # Network influence potential
                metrics['network_influence'] = (
                    node_strength * 
                    metrics['avg_connection_strength'] * 
                    len(connections) / 10  # Normalize for typical connection counts
                )
                
            return metrics
            
        except Exception as e:
            self.logger.log_error(f"Network position analysis failed: {str(e)}")
            return {}
            
    def _calculate_ghost_metrics(
        self,
        is_ghosted: bool,
        ghost_nodes: List[Dict]
    ) -> Dict[str, float]:
        """
        Calculate ghost-related strength metrics.
        
        EXPANSION POINT: Enhanced ghost analysis
        - Ghost node patterns
        - Resurrection potential
        - Merge history analysis
        """
        metrics = {}
        
        try:
            # Basic ghost state
            metrics['is_ghosted'] = float(is_ghosted)
            metrics['ghost_nodes_count'] = len(ghost_nodes)
            
            # Analyze ghost node strengths if present
            if ghost_nodes:
                ghost_strengths = [
                    g.get('strength', 0.0) 
                    for g in ghost_nodes
                ]
                if ghost_strengths:
                    metrics['avg_ghost_strength'] = sum(ghost_strengths) / len(ghost_strengths)
                    metrics['max_ghost_strength'] = max(ghost_strengths)
                    
            return metrics
            
        except Exception as e:
            self.logger.log_error(f"Ghost metrics calculation failed: {str(e)}")
            return {
                'is_ghosted': float(is_ghosted),
                'ghost_nodes_count': len(ghost_nodes)
            }
            
    def _get_fallback_metrics(self) -> Dict[str, float]:
        """Provide safe fallback metrics if calculations fail."""
        return {
            'current_strength': 0.0,
            'relative_strength': 0.0,
            'is_ghosted': 0.0,
            'ghost_nodes_count': 0,
        }