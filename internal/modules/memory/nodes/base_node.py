"""
base_node.py

Defines the base memory node class that provides common functionality for both
body and cognitive memory nodes. Handles core attributes, serialization,
connection management, and state transitions that are shared across node types.

Key capabilities:
- Core node attribute management
- Serialization for database storage
- Connection tracking and management
- Basic strength and decay mechanics
- Ghost state transitions
"""

from typing import Dict, List, Any, Optional
import time
import json
from abc import ABC, abstractmethod

class MemoryNodeError(Exception):
    """Base exception for memory node errors."""
    pass

class BaseMemoryNode(ABC):
    """
    Abstract base class for memory nodes with shared functionality.
    Both body and cognitive nodes inherit from this base.
    
    Core attributes:
    - timestamp: When the memory was formed
    - raw_state: Complete state snapshot at formation
    - processed_state: Interpreted/summarized state data
    - strength: Current memory strength [0-1]
    - node_id: Optional unique identifier
    - connections: Weighted links to other nodes
    - ghost management: Flags and data for memory transitions
    """
    def __init__(
        self,
        timestamp: float,
        raw_state: Dict[str, Any],
        processed_state: Dict[str, Any],
        strength: float,
        node_id: Optional[str] = None,
        ghosted: bool = False,
        parent_node_id: Optional[str] = None,
        ghost_nodes: Optional[List[Dict]] = None,
        ghost_states: Optional[List[Dict]] = None,
        connections: Optional[Dict[str, float]] = None,
        last_connection_update: Optional[float] = None
    ):
        self.timestamp = timestamp
        self.raw_state = raw_state
        self.processed_state = processed_state
        self.strength = strength
        self.node_id = node_id
        self.ghosted = ghosted
        self.parent_node_id = parent_node_id
        self.ghost_nodes = ghost_nodes or []
        self.ghost_states = ghost_states or []
        self.connections = connections or {}
        self.last_connection_update = last_connection_update or time.time()
        self.last_accessed = time.time()

    def __post_init__(self):
        """Initialize collections if None."""
        self.ghost_nodes = self.ghost_nodes or []
        self.ghost_states = self.ghost_states or []
        self.connections = self.connections or {}

    # -------------------------------------------------------------------------
    # Serialization Interface
    # -------------------------------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        """Convert node to database-friendly format."""
        return {
            'timestamp': self.timestamp,
            'raw_state': json.dumps(self.raw_state),
            'processed_state': json.dumps(self.processed_state),
            'strength': self.strength,
            'ghosted': self.ghosted,
            'parent_node_id': self.parent_node_id,
            'ghost_nodes': json.dumps(self.ghost_nodes),
            'ghost_states': json.dumps(self.ghost_states),
            'connections': json.dumps(self.connections),
            'last_connection_update': self.last_connection_update,
            'last_accessed': self.last_accessed
        }

    @classmethod
    @abstractmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BaseMemoryNode':
        """
        Create node from database format. Must be implemented by subclasses
        to handle specialized fields.
        """
        pass

    # -------------------------------------------------------------------------
    # Connection Management
    # -------------------------------------------------------------------------
    def add_connection(self, node_id: str, weight: float) -> None:
        """
        Add or update a connection to another node.
        
        Args:
            node_id: ID of node to connect to
            weight: Connection weight [0-1]
        """
        if not 0 <= weight <= 1:
            raise ValueError("Connection weight must be between 0 and 1")
        self.connections[node_id] = weight
        self.last_connection_update = time.time()

    def remove_connection(self, node_id: str) -> None:
        """Remove a connection if it exists."""
        self.connections.pop(node_id, None)
        self.last_connection_update = time.time()

    def get_connected_nodes(self, min_weight: float = 0.0) -> List[str]:
        """
        Get IDs of connected nodes, optionally filtered by minimum weight.
        
        Args:
            min_weight: Minimum connection weight to include
            
        Returns:
            List of connected node IDs
        """
        return [
            node_id for node_id, weight in self.connections.items()
            if weight >= min_weight
        ]

    # -------------------------------------------------------------------------
    # Strength & Decay
    # -------------------------------------------------------------------------
    def decay(self, rate: float) -> float:
        """
        Apply decay to node strength and determine resulting state.
        
        Args:
            rate: Base decay rate to apply
            
        Returns:
            New strength after decay
        """
        # Apply base decay
        self.strength *= (1.0 - rate)
        
        if self.ghosted:
            # Ghosted nodes decay faster
            self.strength *= (1.0 - rate * 0.95)
        
        return self.strength
    
    def boost_strength(self, amount: float) -> None:
        """
        Safely increase node strength.
        
        Args:
            amount: Amount to increase strength by
        """
        self.strength = min(1.0, self.strength + amount)

    # -------------------------------------------------------------------------
    # Ghost State Management 
    # -------------------------------------------------------------------------
    def merge_into_parent(self, parent: 'BaseMemoryNode') -> None:
        """
        Handle node-level state changes when being merged into a parent.
        Note: This doesn't handle network-level or DB updates - those
        should be handled by the appropriate managers.
        
        Only handles:
        - Basic state ghosting
        - Connection transfers
        - Strength redistribution
        """
        self.ghosted = True
        self.parent_node_id = parent.node_id
        
        # Record this node as a ghost in the parent
        parent.ghost_nodes.append({
            'node_id': self.node_id,
            'timestamp': self.timestamp,
            'raw_state': self.raw_state,
            'processed_state': self.processed_state,
            'strength': self.strength,
            'ghost_nodes': self.ghost_nodes  # Preserve any nested ghosts
        })
        
        # Transfer remaining strength to parent
        parent.strength = min(1.0, parent.strength + (self.strength * 0.3))
        
        # Merge connections - connections to this node should now point to parent
        for conn_id, weight in self.connections.items():
            if conn_id in parent.connections:
                # If connection already exists, strengthen it
                parent.connections[conn_id] = min(
                    1.0, 
                    parent.connections[conn_id] + (weight * 0.5)
                )
            else:
                # Transfer connection with reduced weight
                parent.connections[conn_id] = weight * 0.85

    @abstractmethod
    def get_state_signature(self) -> Dict[str, Any]:
        """
        Get a complete signature of the node's current state.
        Must be implemented by subclasses to handle their specific state data.
        """
        pass

    def __hash__(self) -> int:
        """Enable use in sets/dicts based on node_id."""
        return hash(self.node_id)

    def __eq__(self, other: object) -> bool:
        """Nodes are equal if they have the same ID."""
        if not isinstance(other, BaseMemoryNode):
            return NotImplemented
        return self.node_id == other.node_id