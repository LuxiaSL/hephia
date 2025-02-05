# memory/types.py

from dataclasses import dataclass
from typing import Dict, Any, List, TYPE_CHECKING

if TYPE_CHECKING:
    from ..nodes.body_node import BodyMemoryNode


@dataclass
class EmotionalStateSignature:
    """
    Represents preserved emotional information extracted from a body node.
    Typically nested within the full BodyStateSignature.
    """
    emotional_vectors: List[Dict[str, Any]]
    timestamp: float

    @classmethod
    def from_body_node(cls, body_node: 'BodyMemoryNode') -> 'EmotionalStateSignature':
        """
        Build an emotional signature from the raw_state in a BodyMemoryNode.
        """
        raw = body_node.raw_state or {}
        return cls(
            emotional_vectors=[
                v.to_dict() for v in raw.get('emotional_vectors', [])
            ],
            timestamp=body_node.timestamp
        )


@dataclass
class BodyStateSignature:
    """
    A complete preserved snapshot of a body memory node’s state:
    - Emotional data (nested in EmotionalStateSignature)
    - Needs, behaviors, and other relevant fields
    - Connection info & formation context
    """
    timestamp: float
    source_node_id: str

    # Nested emotional signature (optional, but generally expected)
    emotional_signature: EmotionalStateSignature

    # Additional body-level states
    need_states: Dict[str, Dict[str, Any]]
    behavior: Dict[str, Any]
    mood_state: Dict[str, float]

    # High-level node metrics
    source_strength: float
    connection_weights: Dict[str, float]

    # Metadata about how/why this signature was formed
    formation_context: Dict[str, Any]

    @classmethod
    def from_body_node(cls, body_node: 'BodyMemoryNode') -> 'BodyStateSignature':
        """
        Construct a full BodyStateSignature from an active BodyMemoryNode.
        """
        # Build the nested emotional signature
        emotional_sig = EmotionalStateSignature.from_body_node(body_node)

        raw = body_node.raw_state or {}
        return cls(
            timestamp=body_node.timestamp,
            source_node_id=body_node.node_id,
            emotional_signature=emotional_sig,
            need_states=raw.get('needs', {}),
            behavior=raw.get('behavior', {}),
            mood=raw.get('mood', {}),
            source_strength=body_node.strength,
            connection_weights=body_node.connections.copy(),
            formation_context={
                'timestamp': body_node.timestamp,
                'strength': body_node.strength,
                'connected_nodes': list(body_node.connections.keys()),
            }
        )

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize this signature into a dictionary that can be stored as JSON.
        """
        return {
            'timestamp': self.timestamp,
            'source_node_id': self.source_node_id,
            'emotional_signature': {
                'emotional_vectors': self.emotional_signature.emotional_vectors,
                'timestamp': self.emotional_signature.timestamp
            },
            'need_states': self.need_states,
            'behavior': self.behavior,
            'mood_state': self.mood_state,
            'source_strength': self.source_strength,
            'connection_weights': self.connection_weights,
            'formation_context': self.formation_context
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BodyStateSignature':
        """
        Reconstruct a BodyStateSignature from a dictionary (e.g., loaded JSON).
        """
        em_sig_data = data['emotional_signature']
        emotional_sig = EmotionalStateSignature(
            emotional_vectors=em_sig_data['emotional_vectors'],
            timestamp=em_sig_data['timestamp']
        )

        return cls(
            timestamp=data['timestamp'],
            source_node_id=data['source_node_id'],
            emotional_signature=emotional_sig,
            need_states=data['need_states'],
            behavior=data['behavior'],
            mood_state=data['mood_state'],
            source_strength=data['source_strength'],
            connection_weights=data['connection_weights'],
            formation_context=data['formation_context']
        )
