"""
cognitive_network.py

Concrete implementation of a CognitiveMemoryNetwork, specializing BaseNetwork to handle
CognitiveMemoryNode creation, updates, merges, and maintenance. Integrates with 
CognitiveDBManager for persistence, and optionally a SynthesisManager for advanced merges,
and a CognitiveConnectionManager for connection management.
"""

from __future__ import annotations
import time
from typing import Optional, Dict, Any
import asyncio

from loggers.loggers import MemoryLogger
from .connections.cognitive_manager import CognitiveConnectionManager
from .base_network import BaseNetwork, NetworkConfig, NetworkError
from ..nodes.cognitive_node import CognitiveMemoryNode
from ..db.managers import CognitiveDBManager


class CognitiveMemoryNetwork(BaseNetwork[CognitiveMemoryNode]):
    """
    A specialized network for cognitive memory nodes, handling:
      - Node creation/persistence via CognitiveDBManager
      - Connection management via CognitiveConnectionManager
      - Maintenance tasks (decay, ghost transitions, merges)
    """
    def __init__(
        self,
        db_manager: CognitiveDBManager,
        metrics_orchestrator: Any,
        config: Optional[NetworkConfig] = None,
        working_dir: Optional[str] = None
    ) -> None:
        """
        Initialize the CognitiveMemoryNetwork.

        Args:
            db_manager: CognitiveDBManager for database access (create, update, delete, load).
            metrics_orchestrator: System for calculating retrieval & usage metrics.
            config: Optional network configuration.
            working_dir: Optional working directory for path operations.
        """
        super().__init__(db_manager, metrics_orchestrator, config, working_dir)
        self.cog_db_manager = db_manager
        self.metrics_orchestrator = metrics_orchestrator
        self.connection_manager = CognitiveConnectionManager(self, metrics_orchestrator)
        self.logger = MemoryLogger

    @classmethod
    async def create(cls, db_manager: CognitiveDBManager, metrics_orchestrator: Any, 
                     config: Optional[NetworkConfig] = None, working_dir: Optional[str] = None) -> CognitiveMemoryNetwork:
        instance = cls(db_manager, metrics_orchestrator, config, working_dir)
        await instance._load_existing_nodes()
        instance.logger.debug("CognitiveMemoryNetwork initialized.")
        return instance

    async def _load_existing_nodes(self) -> None:
        loaded_nodes = await self.cog_db_manager.load_all_nodes()
        current_time = time.time()
        valid_node_ids = set()

        # First pass: load all nodes and collect valid IDs.
        for node in loaded_nodes:
            node.node_id = str(node.node_id)
            valid_node_ids.add(node.node_id)
            node.last_accessed = node.last_accessed or current_time
            node.last_connection_update = node.last_connection_update or current_time
            self.nodes[node.node_id] = node

        # Second pass: validate and clean up connections.
        total_invalid = 0
        for node in self.nodes.values():
            invalid_connections = []
            for conn_id, weight in node.connections.items():
                str_conn_id = str(conn_id)
                if (str_conn_id not in valid_node_ids or
                    weight < self.config.min_connection_strength or
                    str_conn_id == node.node_id):
                    invalid_connections.append(conn_id)
            for invalid_id in invalid_connections:
                node.connections.pop(invalid_id, None)
                total_invalid += 1
            if invalid_connections:
                await self.cog_db_manager.update_node(node)
        self.logger.info(
            f"Loaded {len(loaded_nodes)} CognitiveMemoryNodes from DB. "
            f"Cleaned up {total_invalid} invalid connections."
        )

    # -------------------------------------------------------------------------
    # Core CRUD Operations
    # -------------------------------------------------------------------------
    async def add_node(self, node: CognitiveMemoryNode) -> str:
        """Add a new CognitiveMemoryNode to the network."""
        async with self.network_operation("add_node"):
            # Step 1: Create node (this is the critical part)
            self.logger.debug("Creating node in database...")
            new_id = await self.cog_db_manager.create_node(node)
            self.logger.debug(f"Node created in DB with ID {new_id}")
            
            node.node_id = str(new_id)
            self.nodes[node.node_id] = node
            self.logger.debug(f"Node {node.node_id} added to memory network")
            

            await self._form_node_connections(node, lock_acquired=True)
            
            return node.node_id

    async def _form_node_connections(self, node: CognitiveMemoryNode, lock_acquired: bool = False) -> None:
        """Form initial connections for node as a separate operation."""
        try:
            candidates = await self.get_active_nodes()
            if candidates:
                self.logger.debug(f"Forming initial connections for node {node.node_id}")
                connection_weights = await self.connection_manager.form_initial_connections(
                    node=node,
                    candidates=candidates,
                    max_candidates=self.config.min_active_nodes,
                    lock_acquired=lock_acquired
                )
                
                # Update the node's connections
                node.connections.update(connection_weights)
                node.last_connection_update = time.time()
                await self._persist_node(node, lock_acquired)
                
                # Update reciprocal connections
                for other_id, weight in connection_weights.items():
                    other_node = self.nodes.get(other_id)
                    if other_node:
                        other_node.connections[node.node_id] = weight
                        other_node.last_connection_update = time.time()
                        await self._persist_node(other_node, lock_acquired)
                
                self.logger.debug(f"Initial connections formed for node {node.node_id}")
        except Exception as e:
            self.logger.error(f"Failed to form initial connections for node {node.node_id}: {e}")

    async def remove_node(self, node_id: str) -> None:
        """
        Remove a CognitiveMemoryNode from the network (and database).
        Also cleans up connections referencing this node.
        """
        async with self.network_operation("remove_node"):
            if node_id not in self.nodes:
                raise NetworkError(f"Node {node_id} not found in CognitiveMemoryNetwork.")

            node = self.nodes.pop(node_id)
            # Clean up connections in other nodes.
            for other_node in self.nodes.values():
                if node_id in other_node.connections:
                    other_node.connections.pop(node_id)
                    await self._persist_node(other_node, lock_acquired=True)

            await self.cog_db_manager.delete_node(node)
            self.logger.debug(f"Node {node_id} removed from CognitiveMemoryNetwork.")

    async def update_node(self, node: CognitiveMemoryNode) -> None:
        """
        Update an existing CognitiveMemoryNode in the network (and database).
        Also triggers connection updates if needed.
        """
        async with self.network_operation("update_node"):
            if not node.node_id:
                raise NetworkError("Cannot update a node without an ID.")
            if node.node_id not in self.nodes:
                raise NetworkError(f"Node {node.node_id} not found in network.")

            current_time = time.time()
            node.last_accessed = current_time
            node.last_connection_update = node.last_connection_update or current_time
            if (current_time - node.last_connection_update) > self.config.activity_window:
                try:
                    # Add timeout to connection update
                    async with asyncio.timeout(5.0):  # 5 second timeout
                        await self.connection_manager.update_connections(node, lock_acquired=True)
                    node.last_connection_update = current_time
                except TimeoutError:
                    self.logger.error(f"Connection update timed out for node {node.node_id}")
                    # Still update the timestamp to prevent repeated timeout attempts
                    node.last_connection_update = current_time
                except Exception as e:
                    self.logger.error(f"Failed to update connections for node {node.node_id}: {e}")
                    # Still update timestamp to prevent retry storm
                    node.last_connection_update = current_time

            self.nodes[node.node_id] = node
            await self._persist_node(node, lock_acquired=True)
            self.logger.debug(f"Node {node.node_id} updated in CognitiveMemoryNetwork.")

    async def get_node(self, node_id: str) -> Optional[CognitiveMemoryNode]:
        """
        Retrieve a node from the network.
        Updates access time and triggers connection updates if necessary.
        """
        async with self.network_operation("get_node"):
            node = self.nodes.get(node_id)
            if node:
                current_time = time.time()
                node.last_accessed = current_time
                node.last_connection_update = node.last_connection_update or current_time
                
                if (current_time - node.last_connection_update) > self.config.activity_window:
                    try:
                        # Add timeout to connection update
                        async with asyncio.timeout(5.0):  # 5 second timeout
                            await self.connection_manager.update_connections(node, lock_acquired=True)
                        node.last_connection_update = current_time
                    except TimeoutError:
                        self.logger.error(f"Connection update timed out for node {node_id}")
                        # Still update timestamp to prevent repeated timeout attempts
                        node.last_connection_update = current_time
                    except Exception as e:
                        self.logger.error(f"Failed to update connections for node {node_id}: {e}")
                        # Still update timestamp to prevent retry storm
                        node.last_connection_update = current_time
                    
                    try:
                        await self._persist_node(node, lock_acquired=True)
                    except Exception as e:
                        self.logger.error(f"Failed to persist node {node_id}: {e}")
            
            return node

    # -------------------------------------------------------------------------
    # Persistence Hook
    # -------------------------------------------------------------------------
    async def _persist_node(self, node: CognitiveMemoryNode, lock_acquired: bool = False) -> None:
        if not lock_acquired:
            async with self.network_operation("_persist_node"):
                await self._persist_node_core(node)
        else:
            await self._persist_node_core(node)

    async def _persist_node_core(self, node: CognitiveMemoryNode) -> None:
        if not node.node_id:
            new_id = await self.cog_db_manager.create_node(node)
            node.node_id = str(new_id)
        else:
            await self.cog_db_manager.update_node(node)
        self.nodes[node.node_id] = node
        self.logger.debug(f"Persisted node {node.node_id} to database.")

    # -------------------------------------------------------------------------
    # Network Maintenance
    # -------------------------------------------------------------------------
    async def _maintain_nodes(self) -> None:
        """
        Perform basic maintenance on nodes:
         - Apply decay
         - Update timestamps
         - Validate state and connection health
        """
        active_nodes = []
        for node in self.nodes.values():
            if node.ghosted:
                continue
            node.decay(self.config.decay_rate)
            active_nodes.append(node)
        
        # Pass the list of active nodes to the connection manager.
        await self.connection_manager.maintain_connections(active_nodes)

    # -------------------------------------------------------------------------
    # Shutdown
    # -------------------------------------------------------------------------