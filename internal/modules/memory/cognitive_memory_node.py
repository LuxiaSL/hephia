"""
cognitive_memory_node.py

Defines the CognitiveMemoryNode dataclass for storing semantic (LLM-interpreted)
experiences in memory. Each node represents a cognitively processed memory with 
semantic meaning, embedding vectors for retrieval, and connection tracking.

Key capabilities:
- Stores semantic interpretations & embeddings from LLM processing
- Manages memory strength and decay over time 
- Handles ghost node transitions for weak/merged memories
- Calculates and generates memory echo effects
- Tracks connections to other cognitive nodes
- Provides serialization for database storage

Body-memory relationships are handled externally via the memory_links table in 
the parent CognitiveMemory manager.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
import time
import json

class MemorySystemError(Exception):
    """Generic memory system error."""
    pass

@dataclass
class CognitiveMemoryNode:
    """
    A cognitive memory node representing a semantically interpreted experience.
    
    Key points:
      - text_content: LLM's interpretation or summary
      - embedding: vector for semantic retrieval
      - raw_state: raw system state at formation
      - processed_state: assembled state summary
      - connections: adjacency to other cognitive nodes
      - ghost_nodes & ghost_states: merges and version history; graveyard that can be resurrected within reason
      - echo fields: used to calculate satiation for repeated echos
    """
    # --- Core Node Fields ---
    timestamp: float
    text_content: str
    embedding: List[float]
    raw_state: Dict[str, Any]
    processed_state: Dict[str, Any]
    strength: float
    node_id: Optional[str] = None

    # --- Ghost / Merge Fields ---
    ghosted: bool = False
    parent_node_id: Optional[str] = None
    ghost_nodes: List[Dict] = field(default_factory=list)  # Child node data
    ghost_states: List[Dict] = field(default_factory=list) # Prior states/versions
    
    # --- Cognitive Connections ---
    connections: Dict[str, float] = field(default_factory=dict)

    # --- Echo Fields ---
    last_echo_time: Optional[float] = None
    echo_dampening: float = 1.0
    
    # --- Additional Info ---
    semantic_context: Dict[str, Any] = field(default_factory=dict)     # Additional semantic metadata
    last_accessed: Optional[float] = None                              # For tracking LLM attention
    formation_source: Optional[str] = None                             # Event/trigger that created this memory

    # -------------------------------------------------------------------------
    # Initialization & Conversion
    # -------------------------------------------------------------------------
    def __post_init__(self):
        """Initialize optional fields if not provided."""
        if self.ghost_nodes is None:
            self.ghost_nodes = []
        if self.ghost_states is None:
            self.ghost_states = []
        if self.connections is None:
            self.connections = {}
        if self.semantic_context is None:
            self.semantic_context = {}
        if self.last_accessed is None:
            self.last_accessed = self.timestamp
        if self.last_echo_time is None:
            self.last_echo_time = self.timestamp

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert node to database-friendly format for insertion/update.
        JSON fields are serialized. 
        """
        return {
            'timestamp': self.timestamp,
            'text_content': self.text_content,
            'embedding': json.dumps(self.embedding),
            'raw_state': json.dumps(self.raw_state),
            'processed_state': json.dumps(self.processed_state),
            'strength': self.strength,
            'ghosted': self.ghosted,
            'parent_node_id': self.parent_node_id,
            'ghost_nodes': json.dumps(self.ghost_nodes),
            'ghost_states': json.dumps(self.ghost_states),
            'connections': json.dumps(self.connections),
            'semantic_context': json.dumps(self.semantic_context),
            'last_accessed': self.last_accessed,
            'formation_source': self.formation_source,
            'last_echo_time': self.last_echo_time,
            'echo_dampening': self.echo_dampening
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CognitiveMemoryNode':
        """
        Create node from a database row dict. 
        Deserializes JSON fields to restore lists/dicts.
        """
        try:
            node_data = {
                'timestamp': data['timestamp'],
                'text_content': data['text_content'],
                'embedding': json.loads(data['embedding']),
                'raw_state': json.loads(data['raw_state']),
                'processed_state': json.loads(data['processed_state']),
                'strength': data['strength'],
                'node_id': data.get('id'),
                'ghosted': data.get('ghosted', False),
                'parent_node_id': data.get('parent_node_id'),
                'ghost_nodes': json.loads(data['ghost_nodes']),
                'ghost_states': json.loads(data['ghost_states']),
                'connections': json.loads(data['connections']),
                'semantic_context': json.loads(data['semantic_context']),
                'last_accessed': data.get('last_accessed'),
                'formation_source': data.get('formation_source'),
                'last_echo_time': data.get('last_echo_time'),
                'echo_dampening': data.get('echo_dampening', 1.0)
            }
            return cls(**node_data)
        except (json.JSONDecodeError, KeyError) as e:
            raise MemorySystemError(f"Failed to create cognitive node from data: {e}")

    # -------------------------------------------------------------------------
    # Merge & Decay Logic
    # -------------------------------------------------------------------------
    def merge_into_parent(self, parent: 'CognitiveMemoryNode') -> None:
        """
        Merge this node into a parent node, marking self as ghosted.
        """
        self.ghosted = True
        self.parent_node_id = parent.node_id
        
        # Record this node as a ghost in the parent
        parent.ghost_nodes.append({
            'node_id': self.node_id,
            'timestamp': self.timestamp,
            'text_content': self.text_content,
            'raw_state': self.raw_state,
            'processed_state': self.processed_state,
            'strength': self.strength,
            'ghost_nodes': self.ghost_nodes
        })
        
        # Transfer strength with cognitive adjustment
        parent.strength = min(1.0, parent.strength + (self.strength * 0.4))
        
        # Merge cognitive connections
        for conn_id, weight in self.connections.items():
            if conn_id in parent.connections:
                parent.connections[conn_id] = min(1.0, parent.connections[conn_id] + (weight * 0.6))
            else:
                parent.connections[conn_id] = weight * 0.8
                
    def decay(self, rate: float, min_strength: float, min_ghost_strength: float) -> str:
        """
        Apply cognitive-specific decay to node strength.
        
        Args:
            rate: Base decay rate
            min_strength: Threshold for initial ghosting
            min_ghost_strength: Threshold for ghost state conversion
            
        Returns:
            str: Current decay state ('active', 'ghost', 'final_prune')
        """
        # Apply base decay
        self.strength *= (1.0 - rate)
        
        # Apply cognitive-specific modifiers
        if self.last_accessed:
            time_factor = min(1.0, (time.time() - self.last_accessed) / (72 * 3600))  # 72hr scaling
            self.strength *= (1.0 - (rate * time_factor * 0.5))
        
        if self.ghosted:
            # Ghosted nodes decay faster but preserve more of their connections
            self.strength *= (1.0 - rate * 0.7)
            return 'final_prune' if self.strength < min_ghost_strength else 'ghost'
        
        return 'ghost' if self.strength < min_strength else 'active'

    # -------------------------------------------------------------------------
    # Echo Management
    # -------------------------------------------------------------------------
    def calculate_echo(self) -> Dict[str, Any]:
        """
        Calculate potential echo effect based on stored state.
        Returns echo signature if node can produce significant effect.
        
        Returns:
            Dict containing echo intensity and characteristics
        """
        if self.ghosted:
            return None
            
        # Start with base intensity from node strength
        echo_data = {
            "intensity": self.strength,
            "source_node": self.node_id,
            "components": {}
        }
        
        # Emotional component if available
        if "emotional_vectors" in self.raw_state:
            echo_data["components"]["emotional"] = {
                "vectors": self.raw_state["emotional_vectors"],
                "weight": 0.4  # Emotional resonance weighted heavily
            }
        
        # Need state component if available
        if "needs" in self.raw_state:
            echo_data["components"]["needs"] = {
                "states": self.raw_state["needs"],
                "weight": 0.3
            }
        
        # Behavioral component if available
        if "behavior" in self.raw_state:
            echo_data["components"]["behavior"] = {
                "state": self.raw_state["behavior"],
                "weight": 0.3
            }
            
        return echo_data

    def activate_echo(self, intensity: float):
        """
        Generate echo events based on stored state.
        Called after node is selected as most relevant echo source.
        """
        if not self.raw_state:
            return
            
        # Base echo metadata all events will share
        echo_meta = {
            "intensity": intensity,
            "source_node": self.node_id,
            "timestamp": time.time()
        }
        
        # Construct complete echo metadata from internal state components
        echo_data = {
            **echo_meta,
            "metadata": {
                "emotional": self.raw_state.get("emotions", {}).get("emotional_vectors", []),
                "mood": self.raw_state.get("mood", {}),
                "needs": self.raw_state.get("needs", {}),
                "behavior": self.raw_state.get("behavior", {}),
            },
            "processed_state": self.processed_state,
        }

        # Dispatch single comprehensive echo event
        global_event_dispatcher.dispatch_event(Event(
            "memory:echo",
            echo_data
        ))
