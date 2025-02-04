import json
import time
from typing import List, Optional, Dict, Any, Tuple

from .schema import SYNTHESIS_TYPES
from loggers.loggers import MemoryLogger
from .operations import BaseDBOperations, DatabaseError
from internal.modules.memory.nodes.body_node import BodyMemoryNode
from internal.modules.memory.nodes.cognitive_node import CognitiveMemoryNode


class BodyDBManager:
    """
    Domain-specific interface for BodyMemoryNode operations.
    Leverages BaseDBOperations for actual DB reads/writes.
    """

    def __init__(self, db_ops: BaseDBOperations):
        self.db = db_ops
        self.logger = MemoryLogger

    async def create_node(self, node: BodyMemoryNode) -> int:
        """
        Save a brand-new BodyMemoryNode to the database.
        Returns the newly assigned node_id, which is also set on the node.
        """
        node_data = node.to_dict()
        new_id = await self.db.create_body_node(node_data)
        node.node_id = new_id
        return new_id

    async def update_node(self, node: BodyMemoryNode) -> None:
        """
        Update an existing BodyMemoryNode in the database.
        """
        if not node.node_id:
            raise ValueError("BodyMemoryNode must have a valid node_id to update.")
        node_data = node.to_dict()
        await self.db.update_body_node(node.node_id, node_data)

    async def load_node(self, node_id: int) -> Optional[BodyMemoryNode]:
        """
        Retrieve a BodyMemoryNode from DB and reconstitute it as an object.
        """
        record = await self.db.get_body_node(node_id)
        if not record:
            return None
        return BodyMemoryNode.from_dict(record)

    async def delete_node(self, node: BodyMemoryNode) -> None:
        """
        Remove the node from DB entirely.
        """
        if not node.node_id:
            raise ValueError("BodyMemoryNode must have a valid node_id to be deleted.")
        await self.db.delete_body_node(node.node_id)

    async def load_all_nodes(self) -> List[BodyMemoryNode]:
        """
        Fetch all nodes from the DB and reconstitute them.
        """
        records = await self.db.load_all_body_nodes()
        nodes = []
        for rec in records:
            try:
                node_obj = BodyMemoryNode.from_dict(rec)
                nodes.append(node_obj)
            except Exception as e:
                self.logger.warning(f"Failed to load BodyMemoryNode ID={rec.get('id')}: {e}")
        return nodes

    async def merge_nodes(self, child_node: BodyMemoryNode, parent_node: BodyMemoryNode) -> None:
        """
        Merge child_node into parent_node and persist changes.
        """
        child_node.merge_into_parent(parent_node)
        await self.update_node(child_node)
        await self.update_node(parent_node)


class CognitiveDBManager:
    """
    Domain-specific interface for CognitiveMemoryNode operations.
    Uses BaseDBOperations for DB interaction.
    """

    def __init__(self, db_ops: BaseDBOperations):
        self.db = db_ops
        self.logger = MemoryLogger()

    async def create_node(self, node: CognitiveMemoryNode) -> int:
        """
        Save a new CognitiveMemoryNode. Returns assigned node_id, also set on the node.
        """
        node_data = node.to_dict()
        new_id = await self.db.create_cognitive_node(node_data)
        node.node_id = new_id
        return new_id

    async def update_node(self, node: CognitiveMemoryNode) -> None:
        """
        Update an existing CognitiveMemoryNode in the DB.
        """
        if not node.node_id:
            raise ValueError("CognitiveMemoryNode must have a valid node_id to update.")
        node_data = node.to_dict()
        await self.db.update_cognitive_node(node.node_id, node_data)

    async def load_node(self, node_id: int) -> Optional[CognitiveMemoryNode]:
        """
        Retrieve a CognitiveMemoryNode by ID.
        """
        record = await self.db.get_cognitive_node(node_id)
        if not record:
            return None
        return CognitiveMemoryNode.from_dict(record)

    async def delete_node(self, node: CognitiveMemoryNode) -> None:
        """
        Remove a CognitiveMemoryNode from DB.
        """
        if not node.node_id:
            raise ValueError("CognitiveMemoryNode must have a valid node_id to be deleted.")
        await self.db.delete_cognitive_node(node.node_id)

    async def load_all_nodes(self) -> List[CognitiveMemoryNode]:
        """
        Fetch all CognitiveMemoryNodes from the DB.
        """
        records = await self.db.load_all_cognitive_nodes()
        nodes = []
        for rec in records:
            try:
                node_obj = CognitiveMemoryNode.from_dict(rec)
                nodes.append(node_obj)
            except Exception as e:
                self.logger.warning(f"Failed to load CognitiveMemoryNode ID={rec.get('id')}: {e}")
        return nodes

    async def record_synthesis_resurrection(self, node: CognitiveMemoryNode, parent_id: int) -> None:
        """
        Record a resurrection synthesis relation.
        """
        if not node.node_id:
            raise ValueError("CognitiveMemoryNode must be persisted first.")
        metadata = {"timestamp": node.timestamp}
        await self.db.add_synthesis_relation(
            synthesis_node_id=node.node_id,
            constituent_node_id=parent_id,
            relationship_type="resurrection",
            metadata=metadata
        )

    async def handle_conflict_synthesis(self, synthesis_node: CognitiveMemoryNode, constituents: List[int], details: Tuple) -> None:
        """
        Handle conflict synthesis by recording relationships.
        """
        if not synthesis_node.node_id:
            raise ValueError("Synthesis node must be persisted first.")
        for cid in constituents:
            meta = {
                "timestamp": synthesis_node.timestamp,
                "synthesis_type": "conflict_resolution",
                "conflict_metrics": details[0] if details else None,
                "similarity_metrics": details[1] if details else None
            }
            await self.db.add_synthesis_relation(
                synthesis_node_id=synthesis_node.node_id,
                constituent_node_id=cid,
                relationship_type="conflict_synthesis",
                metadata=meta
            )

    async def get_body_links(self, cognitive_node_id: int) -> List[Dict[str, Any]]:
        """Get all body links for a cognitive node."""
        return await self.db.get_memory_links_for_cognitive(cognitive_node_id)

    async def transfer_body_links(self, from_id: int, to_id: int, strength_modifier: float = 0.9) -> None:
        """Transfer body links during merge operations."""
        await self.db.transfer_body_links(from_id, to_id, strength_modifier)

    async def merge_nodes(self, source: CognitiveMemoryNode, target: CognitiveMemoryNode) -> None:
        """
        Handle database updates for node merges.
        Uses transaction context for atomic operations.
        """
        try:
            async with self.db.transaction() as conn:
                # Update the nodes themselves
                await self.update_node(source)
                await self.update_node(target)

                # Transfer body links
                await self.db.transfer_body_links(
                    from_cog_id=int(source.node_id),
                    to_cog_id=int(target.node_id)
                )

                # Record merge relationship in synthesis relations
                metadata = {
                    "merge_timestamp": time.time(),
                    "source_strength": source.strength,
                    "target_strength": target.strength
                }

                await self.db.add_synthesis_relation(
                    synthesis_node_id=int(target.node_id),
                    constituent_node_id=int(source.node_id),
                    relationship_type=SYNTHESIS_TYPES['MERGE'],
                    metadata=metadata
                )
        except Exception as e:
            raise DatabaseError(f"Failed to merge nodes in DB: {e}")


class SynthesisRelationManager:
    """
    Manages synthesis relations between memory nodes, including:
    - Conflict resolution syntheses
    - Memory merges 
    - Node resurrections
    - Tracking synthesis hierarchies
    """

    def __init__(self, db_ops: BaseDBOperations):
        self.db = db_ops
        self.logger = MemoryLogger()

    async def add_relation(
        self,
        synthesis_node_id: int,
        constituent_node_id: int,
        relationship_type: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Create a synthesis relationship record."""
        if relationship_type not in SYNTHESIS_TYPES.values():
            raise ValueError(f"Invalid synthesis type: {relationship_type}")
        await self.db.add_synthesis_relation(
            synthesis_node_id=synthesis_node_id,
            constituent_node_id=constituent_node_id,
            relationship_type=relationship_type,
            metadata=metadata or {}
        )

    async def get_relations(self, node_id: int, as_synthesis: bool = True) -> List[Dict[str, Any]]:
        """Get synthesis relations for a node."""
        return await self.db.get_synthesis_relations(node_id, as_synthesis)

    async def delete_relations(self, node_id: int) -> None:
        """Remove all synthesis relations for a node."""
        await self.db.delete_synthesis_relations(node_id)

    async def transfer_relations(self, from_id: int, to_id: int, modifier: float = 0.9) -> None:
        """Transfer synthesis relations during merges."""
        async with self.db.transaction() as conn:
            relations = await self.get_relations(from_id, True)
            for relation in relations:
                metadata = relation.get('metadata') or {}
                metadata.update({
                    'transferred_from': from_id,
                    'transfer_strength_mod': modifier
                })
                await self.add_relation(
                    synthesis_node_id=to_id,
                    constituent_node_id=relation['node_id'],
                    relationship_type=relation['relationship_type'],
                    metadata=metadata
                )
            await self.delete_relations(from_id)

    async def get_node_ancestry(self, node_id: int, is_body_node: bool = False) -> List[int]:
        """
        Gets chain of synthesis parent nodes.

        Args:
            node_id: Node to get ancestry for
            is_body_node: Whether node is body or cognitive memory

        Returns:
            List of ancestor node IDs in synthesis hierarchy
        """
        return await self.db.get_node_ancestry(node_id, is_body_node)


class MemoryDBManager:
    """
    Orchestrates the different memory managers.
    """

    def __init__(self, db_path: str = 'data/memory.db'):
        self.db_operations = BaseDBOperations(db_path)
        self.cognitive_manager = CognitiveDBManager(self.db_operations)
        self.body_manager = BodyDBManager(self.db_operations)
        self.synthesis_relation_manager = SynthesisRelationManager(self.db_operations)

    async def init_database(self) -> None:
        await self.db_operations.init_database()
        # Initialize additional components if necessary
