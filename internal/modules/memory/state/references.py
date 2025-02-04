from dataclasses import dataclass
from typing import Optional, TYPE_CHECKING
import time

if TYPE_CHECKING:
    from ..nodes.body_node import BodyMemoryNode

from .signatures import BodyStateSignature

@dataclass
class BodyNodeReference:
    """
    Tracks the relationship between a cognitive node and its associated body state.
    If the body node merges or ghosts, we can preserve the last known signature.
    """
    body_node_id: str
    formation_timestamp: float
    preserved_signature: Optional[BodyStateSignature] = None
    signature_timestamp: Optional[float] = None

    def preserve_signature(self, body_node: 'BodyMemoryNode') -> None:
        """
        Capture a complete state signature from the given body node,
        storing it locally to survive merges/ghosting.
        """
        self.preserved_signature = BodyStateSignature.from_body_node(body_node)
        self.signature_timestamp = time.time()

    def has_valid_signature(self) -> bool:
        """Check if we've preserved a signature for later reference."""
        return (self.preserved_signature is not None and
                self.signature_timestamp is not None)