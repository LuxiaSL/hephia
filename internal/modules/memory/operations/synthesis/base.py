"""
base.py

Defines abstract/base classes for the synthesis system, plus any shared data structures.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict

class ISynthesisHandler(ABC):
    """
    Interface for a synthesis handler that merges or synthesizes new nodes
    from conflicting or combined states.
    """

    @abstractmethod
    def handle_conflict_synthesis(
        self,
        conflict_data: Dict[str, Any],
        child: Any,
        parent: Any,
        additional_strength: float = 0.0
    ) -> str:
        """
        Takes conflicting nodes along with conflict details and 
        returns a newly created 'synthesis node' ID.
        
        Args:
            conflict_data: Key information describing the conflict or advanced analysis
            child: The 'child' (secondary) node to merge
            parent: The 'parent' (primary) node
            additional_strength: Extra strength contributed by child/parent 
                                 or from conflict severity
        Returns:
            The new node ID
        """
        pass

    @abstractmethod
    def handle_synthesis_complete(
        self,
        synthesis_node_id: str,
        constituents: list
    ) -> None:
        """
        Optional post-synthesis step, e.g. dispatch an event, or further merges.
        """
        pass
