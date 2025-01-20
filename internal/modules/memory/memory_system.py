"""
Memory System Implementation.

Provides both body and cognitive memory systems with realistic memory formation,
decay, and interaction patterns. Body memory maintains complete state records
that inform the broader cognitive memory network.

The memory system aims to model natural cognitive processes through a hyperdimensional
lattice of interconnected memory nodes that self-organize based on multiple factors.
Each memory has a point in space with different weights that interact with all other
memory nodes around it, affecting the space and creating clusters.
"""

from .body_memory import BodyMemory
from .cognitive_memory import CognitiveMemory

class MemorySystem:
    """
    Manages different types of memory and their interactions.
    Coordinates body memory and cognitive memory processes.
    """

    def __init__(self, internal_context, api_manager, db_path: str = 'data/memory.db'):
        """Initialize the memory system"""
        self.body_memory = BodyMemory(internal_context, db_path)
        self.cognitive_memory = CognitiveMemory(internal_context, self.body_memory, api_manager, db_path)
        
    def update(self) -> None:
        """Update all memory subsystems"""
        self.body_memory.update()
        self.cognitive_memory.update()
        
    # get/set methods here won't exist because memory is stored in its own persistent db(s).
    # instead we perform retrieval on a given system depending on the call.
    # we can also eventually remove calls from body/cognitive memory into here as handlers akin to other patterns