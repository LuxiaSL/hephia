"""
cognitive_node.py

Implementation of CognitiveMemoryNode for semantically interpreted experiences.
Extends BaseMemoryNode with LLM-processed content, embeddings, and echo mechanics
while maintaining links to originating body states.

Key capabilities:
- Stores semantic interpretations & embeddings
- Manages echo effects and dampening
- Preserves body state signatures
- Handles cognitive-specific serialization
- Provides synthesis & conflict hooks
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
import time
import json

from .base_node import BaseMemoryNode, MemoryNodeError
from .body_node import BodyMemoryNode
from .node_utils import deserialize_state
from ..state.references import BodyNodeReference

@dataclass
class CognitiveMemoryNode(BaseMemoryNode):
    """
    A cognitive memory node representing a semantically interpreted experience.
    
    Extends BaseMemoryNode with:
    - LLM-interpreted content
    - Semantic embeddings
    - Echo mechanics
    - Body state references
    - Formation context
    """
   # Non-default fields (required):
    timestamp: float
    raw_state: Dict[str, Any]
    processed_state: Dict[str, Any]
    strength: float
    text_content: str
    embedding: List[float]

    # Default fields inherited from BaseMemoryNode:
    node_id: Optional[str] = None
    ghosted: bool = False
    parent_node_id: Optional[str] = None
    ghost_nodes: List[Dict] = field(default_factory=list)
    ghost_states: List[Dict] = field(default_factory=list)
    connections: Dict[str, float] = field(default_factory=dict)
    last_connection_update: Optional[float] = None

    # CognitiveMemoryNode-specific optional fields:
    last_echo_time: Optional[float] = None
    echo_dampening: float = 1.0
    semantic_context: Dict[str, Any] = field(default_factory=dict)
    last_accessed: Optional[float] = None
    formation_source: Optional[str] = None
    body_references: List[BodyNodeReference] = field(default_factory=list)

    def __post_init__(self):
        """Initialize optional fields and validate cognitive state."""
        # Call the parent initializer to ensure BaseMemoryNode state is set up
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
        
        # Optionally call parent's post init if needed (if defined)
        try:
            super().__post_init__()
        except AttributeError:
            pass
        
        # Initialize collections explicitly (if not already set)
        self.semantic_context = self.semantic_context or {}
        self.body_references = self.body_references or []
        
        # Set access time if not provided
        if self.last_accessed is None:
            self.last_accessed = self.timestamp
            
        if self.last_echo_time is None:
            self.last_echo_time = self.timestamp
            
        # Validate required cognitive fields
        if not self.text_content:
            raise MemoryNodeError("Cognitive node requires text content")
            
        if not self.embedding:
            raise MemoryNodeError("Cognitive node requires embedding")

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CognitiveMemoryNode':
        """Create node from database format."""
        try:
            # Deserialize JSON fields
            raw_state = deserialize_state(data['raw_state'])
            processed_state = deserialize_state(data['processed_state'])
            ghost_nodes = deserialize_state(data.get('ghost_nodes', '[]'))
            ghost_states = deserialize_state(data.get('ghost_states', '[]'))
            connections = deserialize_state(data.get('connections', '{}'))
            embedding = deserialize_state(data['embedding'])
            semantic_context = deserialize_state(data.get('semantic_context', '{}'))
            
            # Rebuild body references
            references = []
            if 'body_references' in data and data['body_references']:
                try:
                    ref_list = deserialize_state(data['body_references'])
                    for ref_data in ref_list:
                        references.append(BodyNodeReference.from_dict(ref_data))
                except (json.JSONDecodeError, KeyError):
                    references = []
            
            return cls(
                timestamp=data['timestamp'],
                raw_state=raw_state,
                processed_state=processed_state,
                strength=data['strength'],
                text_content=data['text_content'],
                embedding=embedding,
                node_id=data.get('id'),
                ghosted=data.get('ghosted', False),
                parent_node_id=data.get('parent_node_id'),
                ghost_nodes=ghost_nodes,
                ghost_states=ghost_states,
                connections=connections,
                last_connection_update=data.get('last_connection_update'),
                semantic_context=semantic_context,
                last_accessed=data.get('last_accessed'),
                formation_source=data.get('formation_source'),
                last_echo_time=data.get('last_echo_time'),
                echo_dampening=data.get('echo_dampening', 1.0),
                body_references=references
            )
        except Exception as e:
            raise MemoryNodeError(f"Failed to create cognitive node from data: {e}")

    def to_dict(self) -> Dict[str, Any]:
        """Convert node to database-friendly format."""
        data = super().to_dict()
        
        # Add cognitive-specific fields
        data.update({
            'text_content': self.text_content,
            'embedding': json.dumps(self.embedding),
            'semantic_context': json.dumps(self.semantic_context),
            'last_accessed': self.last_accessed,
            'formation_source': self.formation_source,
            'last_echo_time': self.last_echo_time,
            'echo_dampening': self.echo_dampening,
            'body_references': json.dumps([
                ref.to_dict() for ref in self.body_references
            ])
        })
        
        return data

    # -------------------------------------------------------------------------
    # Body State Reference Management
    # -------------------------------------------------------------------------
    def attach_body_reference(self, body_node_id: str, preserve_signature: bool = True) -> None:
        """
        Create a new body reference with optional immediate signature preservation.
        
        Args:
            body_node_id: ID of body node to reference
            preserve_signature: Whether to immediately preserve state signature
        """
        ref = BodyNodeReference(
            body_node_id=str(body_node_id),
            formation_timestamp=time.time()
        )
        self.body_references.append(ref)

    def preserve_body_signature(self, body_node: 'BodyMemoryNode') -> None:
        """
        Preserve current signature of a referenced body node.
        
        Args:
            body_node: Body node to preserve signature from
        """
        for ref in self.body_references:
            if ref.body_node_id == str(body_node.node_id):
                ref.preserve_signature(body_node)
                break

    def get_body_context(self, body_node_id: str) -> Dict[str, Any]:
        """
        Get preserved signature for a specific body reference.
        
        Args:
            body_node_id: ID of referenced body node
            
        Returns:
            Dict containing signature if preserved
        """
        for ref in self.body_references:
            if ref.body_node_id == str(body_node_id):
                if ref.has_valid_signature():
                    return {
                        'signature': ref.preserved_signature,
                        'timestamp': ref.signature_timestamp
                    }
        return {}

    # -------------------------------------------------------------------------
    # Echo Management
    # -------------------------------------------------------------------------
    def calculate_echo(self) -> Optional[Dict[str, Any]]:
        """
        Calculate potential echo effect based on current state.
        
        Returns:
            Dict containing echo parameters if node can produce effect
        """
        if self.ghosted:
            return None
            
        # Calculate base intensity from strength
        echo_data = {
            "intensity": self.strength,
            "source_node": self.node_id,
            "components": {}
        }
        
        # Add emotional component if available
        if "emotional_vectors" in self.raw_state:
            echo_data["components"]["emotional"] = {
                "vectors": self.raw_state["emotional_vectors"],
                "weight": 0.4
            }
        
        # Add need states if available
        if "needs" in self.raw_state:
            echo_data["components"]["needs"] = {
                "states": self.raw_state["needs"],
                "weight": 0.3
            }
        
        # Add behavioral component if available
        if "behavior" in self.raw_state:
            echo_data["components"]["behavior"] = {
                "state": self.raw_state["behavior"],
                "weight": 0.3
            }
            
        return echo_data

    def update_echo_state(self, intensity: float) -> None:
        """
        Update node state after producing an echo effect.
        
        Args:
            intensity: Intensity of the echo produced
        """
        # Record echo time
        self.last_echo_time = time.time()
        
        # Update dampening
        echo_window = 180  # seconds
        time_since_last = self.last_echo_time - (self.last_echo_time or 0)
        
        if time_since_last > echo_window:
            self.echo_dampening = 1.0
        else:
            # Progressive dampening with floor
            self.echo_dampening = max(0.1, self.echo_dampening * 0.75)

    # -------------------------------------------------------------------------
    # State Management Helpers
    # -------------------------------------------------------------------------
    def get_state_signature(self) -> Dict[str, Any]:
        """Generate complete state signature including cognitive context."""
        sig = super().get_state_signature()
        
        # Add cognitive-specific data
        sig.update({
            'text_content': self.text_content,
            'semantic_context': self.semantic_context,
            'formation_source': self.formation_source,
            'body_references': [
                ref.preserved_signature.to_dict() 
                for ref in self.body_references 
                if ref.has_valid_signature()
            ]
        })
        
        return sig

    def __hash__(self) -> int:
        """Enable use in sets/dicts based on node_id."""
        return hash(self.node_id)

    def __eq__(self, other: object) -> bool:
        """Nodes are equal if they have the same ID."""
        if not isinstance(other, CognitiveMemoryNode):
            return NotImplemented
        return self.node_id == other.node_id