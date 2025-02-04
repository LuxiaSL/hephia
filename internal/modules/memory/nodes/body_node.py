"""
body_node.py

Implementation of BodyMemoryNode for storing raw emotional/physical experiences.
Focuses on managing unmediated bodily states while inheriting core memory
functionality from BaseMemoryNode.

Key capabilities:
- Stores complete emotional/physical state snapshots
- Manages raw body-based experiences
- Provides emotional similarity comparisons
- Generates state signatures for preservation
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
import time
from .base_node import BaseMemoryNode, MemoryNodeError
from .node_utils import (
    calculate_emotional_similarity,
    calculate_needs_similarity,
    calculate_mood_similarity,
    deserialize_state
)

@dataclass
class BodyMemoryNode(BaseMemoryNode):
    """
    A single body memory experience with complete state information.
    Contains both raw and processed state data for complete recall.
    
    Extends BaseMemoryNode with:
    - Formation metadata
    - Body-specific state comparisons
    - Specialized strength calculations
    - State signature generation
    """
    
    # Inherited fields explicitly declared
    timestamp: float
    raw_state: Dict[str, Any]
    processed_state: Dict[str, Any]
    strength: float
    node_id: Optional[str] = None
    ghosted: bool = False
    parent_node_id: Optional[str] = None
    ghost_nodes: List[Dict] = field(default_factory=list)
    ghost_states: List[Dict] = field(default_factory=list)
    connections: Dict[str, float] = field(default_factory=dict)
    last_connection_update: Optional[float] = None

    # BodyMemoryNode-specific field
    formation_metadata: Dict[str, Any] = field(default_factory=dict)

    
    def __post_init__(self):
        """Initialize and validate body-specific state."""
        super().__init__(
            timestamp=self.timestamp,
            raw_state=self.raw_state,
            processed_state=self.processed_state,
            strength=self.strength,
            node_id=self.node_id,
            ghosted=self.ghosted,
            parent_node_id=self.parent_node_id,
            ghost_nodes=self.ghost_nodes,
            ghost_states=self.ghost_states,
            connections=self.connections,
            last_connection_update=self.last_connection_update
        )

        try:
            super().__post_init__()
        except AttributeError:
            pass
        
        
        if not self.raw_state:
            raise MemoryNodeError(f"""Body node requires raw state {self.node_id} / {self.raw_state}""")
        
        if not isinstance(self.timestamp, (int, float)):
            raise MemoryNodeError("Timestamp must be a number")
        
        # Validate raw_state structure for emotional vectors
        if 'emotions' in self.raw_state:
            vectors = self.raw_state['emotions'].get('active_vectors', [])
            for vector in vectors:
                if not all(isinstance(vector.get(k), (int, float)) 
                        for k in ['valence', 'arousal', 'intensity']):
                    raise MemoryNodeError("Vector values must be numbers")
                if vector.get('source_data') and not isinstance(
                    vector['source_data'], (dict, str, int, float, bool)
                ):
                    raise MemoryNodeError("Invalid source_data type")
            
        required_components = {'emotions', 'needs', 'behavior', 'mood'}
        missing = required_components - set(self.raw_state.keys())
        if missing:
            raise MemoryNodeError(f"Missing required state components: {missing}")

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BodyMemoryNode':
        """Create node from database format."""
        try:
            # Deserialize JSON fields
            raw_state = deserialize_state(data['raw_state'])
            processed_state = deserialize_state(data['processed_state'])
            ghost_nodes = deserialize_state(data.get('ghost_nodes', '[]'))
            ghost_states = deserialize_state(data.get('ghost_states', '[]'))
            connections = deserialize_state(data.get('connections', '{}'))
            
            # Extract formation metadata if present
            formation_metadata = {}
            if 'formation_metadata' in data:
                formation_metadata = deserialize_state(data['formation_metadata'])
            last_accessed = data.get('last_accessed', data.get('timestamp', time.time()))
            node = cls(
                timestamp=data['timestamp'],
                raw_state=raw_state,
                processed_state=processed_state,
                strength=data['strength'],
                node_id=data.get('id'),
                ghosted=data.get('ghosted', False),
                parent_node_id=data.get('parent_node_id'),
                ghost_nodes=ghost_nodes,
                ghost_states=ghost_states,
                connections=connections,
                last_connection_update=data.get('last_connection_update'),
                formation_metadata=formation_metadata
            )
            # Set the new attribute explicitly
            node.last_accessed = last_accessed
            return node
        except Exception as e:
            raise MemoryNodeError(f"Failed to create body node from data: {e}")

    def calculate_state_similarity(self, other: 'BodyMemoryNode') -> float:
        """
        Calculate overall similarity between this node's state and another's.
        
        Args:
            other: Node to compare against
            
        Returns:
            float: Similarity score [0-1]
        """
        similarity = 0.0
        components = 0
        
        raw1, raw2 = self.raw_state, other.raw_state
        
        # Emotional state comparison (highest weight)
        if ('emotional_vectors' in raw1 and 'emotional_vectors' in raw2 and
            raw1['emotional_vectors'] and raw2['emotional_vectors']):
            emotional_sim = calculate_emotional_similarity(
                raw1['emotional_vectors'],
                raw2['emotional_vectors']
            )
            similarity += emotional_sim * 0.35
            components += 0.35

        # Mood comparison
        if ('mood' in raw1 and 'mood' in raw2):
            mood_sim = calculate_mood_similarity(raw1['mood'], raw2['mood'])
            similarity += mood_sim * 0.25 
            components += 0.25
            
        # Behavior state comparison
        if ('behavior' in raw1 and 'behavior' in raw2):
            behavior1 = raw1['behavior'].get('name')
            behavior2 = raw2['behavior'].get('name')
            behavior_sim = 1.0 if behavior1 and behavior1 == behavior2 else 0.0
            similarity += behavior_sim * 0.2
            components += 0.2
            
        # Needs similarity
        if ('needs' in raw1 and 'needs' in raw2):
            needs_sim = calculate_needs_similarity(raw1['needs'], raw2['needs'])
            similarity += needs_sim * 0.2
            components += 0.2
        
        return similarity / components if components > 0 else 0.0

    def get_state_signature(self) -> Dict[str, Any]:
        """
        Generate a complete signature of the node's current state.
        Used for state preservation and cognitive memory references.
        
        Returns:
            Dict containing complete state signature
        """
        raw = self.raw_state
        
        return {
            'timestamp': self.timestamp,
            'source_node_id': self.node_id,
            'emotional_signature': {
                'emotional_vectors': raw.get('emotional_vectors', []),
                'timestamp': self.timestamp
            },
            'need_states': raw.get('needs', {}),
            'behavior': raw.get('behavior', {}),
            'mood_state': raw.get('mood', {}),
            'source_strength': self.strength,
            'connection_weights': self.connections.copy(),
            'formation_context': {
                'timestamp': self.timestamp,
                'strength': self.strength,
                'metadata': self.formation_metadata,
                'connected_nodes': list(self.connections.keys())
            }
        }