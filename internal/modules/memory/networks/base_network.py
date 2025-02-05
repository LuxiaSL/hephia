"""
base_network.py

Abstract base network implementation providing core memory network functionality.
Handles async-safe operations, network traversal, and memory consolidation mechanics.
Specialized networks implement specific memory dynamics while inheriting core operations.

Key capabilities:
- Async-safe network operations
- Core node management interface
- Network traversal and access
- Natural memory consolidation
- Activity-based maintenance
"""

from abc import ABC, abstractmethod
from contextlib import asynccontextmanager
from typing import TypeVar, Generic, Dict, List, Set, Any, Optional, AsyncGenerator
from pathlib import Path
import time
from dataclasses import dataclass

from loggers.loggers import MemoryLogger
from .reentrant_lock import shared_reentrant_lock
from .connections.base_manager import BaseConnectionManager
from ..nodes.base_node import BaseMemoryNode


# Type variable for node specialization
T = TypeVar('T', bound=BaseMemoryNode)

@dataclass
class NetworkConfig:
    """Configuration for network behavior."""
    min_active_nodes: int = 5
    min_connection_strength: float = 0.2  # For traversal/retrieval
    decay_rate: float = 0.1  # Base decay rate for nodes
    enable_maintenance: bool = True  # For basic maintenance
    activity_window: float = 3600            # 1 hour window for activity tracking
    consolidation_ratio: float = 0.7         # Activity threshold for consolidation

class NetworkError(Exception):
    """Base exception for network operations."""
    pass

class BaseNetwork(Generic[T], ABC):
    """
    Abstract base class for memory networks.
    
    Provides core functionality for managing memory nodes while ensuring
    async-safe operations and consistent behaviors.
    """
    
    def __init__(
        self,
        db_manager: Any,  # Type depends on implementation
        metrics_orchestrator: Any,  # Type depends on implementation
        config: Optional[NetworkConfig] = None,
        working_dir: Optional[str] = None
    ):
        """
        Initialize network with required components.
        
        Args:
            db_manager: Database management interface.
            metrics_orchestrator: Metrics calculation system.
            config: Optional network configuration.
            working_dir: Optional working directory for file operations.
        """
        # Core components
        self.db = db_manager
        self.metrics = metrics_orchestrator
        self.config = config or NetworkConfig()

        self.connection_manager: Optional[BaseConnectionManager[T]] = None
        
        # Node storage
        self.nodes: Dict[str, T] = {}
        
        # Async-safe lock for network operations
        self._operation_count = 0
        
        # Cross-platform path handling
        self.working_dir = Path(working_dir) if working_dir else Path.cwd()
        
        # Logging
        self.logger = MemoryLogger()

    # -------------------------------------------------------------------------
    # Async-Safe Network Operation Context Manager
    # -------------------------------------------------------------------------
    @asynccontextmanager
    async def network_operation(self, operation_name: str = "unnamed") -> AsyncGenerator[None, None]:
        """
        Async context manager for network operations.
        
        Args:
            operation_name: Name of operation for logging.
        """
        async with shared_reentrant_lock:
            self._operation_count += 1
            self.logger.debug(f"Starting network operation: {operation_name}")
            try:
                yield
            finally:
                self.logger.debug(f"Completed network operation: {operation_name}")
                self._operation_count -= 1

    # -------------------------------------------------------------------------
    # Core Node Operations
    # -------------------------------------------------------------------------
    @abstractmethod
    async def add_node(self, node: T) -> str:
        """
        Add new node to network. Must be implemented by subclasses.
        
        Args:
            node: Node to add.
            
        Returns:
            str: ID of added node.
        """
        pass

    @abstractmethod
    async def remove_node(self, node_id: str) -> None:
        """
        Remove node from network. Must be implemented by subclasses.
        
        Args:
            node_id: ID of node to remove.
        """
        pass

    async def get_node(self, node_id: str) -> Optional[T]:
        """
        Retrieve node by ID with proper locking.
        
        Args:
            node_id: ID of node to retrieve.
            
        Returns:
            Optional[T]: Node if found.
        """
        async with self.network_operation("get_node"):
            node = self.nodes.get(str(node_id))
            if node:
                node.last_accessed = time.time()
            return node

    async def update_node(self, node: T) -> None:
        """
        Update existing node with proper locking.
        
        Args:
            node: Node to update.
        """
        if not node.node_id:
            raise NetworkError("Cannot update node without ID")
            
        async with self.network_operation("update_node"):
            if node.node_id not in self.nodes:
                raise NetworkError(f"Node {node.node_id} not found")
            self.nodes[node.node_id] = node
            await self._persist_node(node)

    # -------------------------------------------------------------------------
    # Node Access & Traversal
    # -------------------------------------------------------------------------
    async def get_active_nodes(self) -> List[T]:
        """Get all non-ghosted nodes."""
        async with self.network_operation("get_active_nodes"):
            return [n for n in self.nodes.values() if not n.ghosted]

    async def get_connected_nodes(
        self,
        node: T,
        min_weight: float = 0.0
    ) -> List[T]:
        """
        Get nodes connected to given node.
        
        Args:
            node: Source node.
            min_weight: Minimum connection weight.
            
        Returns:
            List[T]: Connected nodes.
        """
        async with self.network_operation("get_connected_nodes"):
            connected = []
            for node_id, weight in node.connections.items():
                if weight >= min_weight:
                    connected_node = self.nodes.get(node_id)
                    if connected_node and not connected_node.ghosted:
                        connected.append(connected_node)
            return connected

    async def traverse_network(
        self,
        start_node: T,
        max_depth: int = 3,
        min_weight: float = 0.0,
        visited: Optional[Set[str]] = None
    ) -> Dict[int, List[T]]:
        """
        Traverse network from start node, grouping by depth.
        
        Args:
            start_node: Node to start from.
            max_depth: Maximum traversal depth.
            min_weight: Minimum connection weight.
            visited: Set of visited node IDs.
            
        Returns:
            Dict mapping depths to lists of nodes.
        """
        async with self.network_operation("traverse_network"):
            try:
                if visited is None:
                    visited = set()
                    
                results: Dict[int, List[T]] = {}
                
                # Early exit conditions
                if not start_node or not start_node.node_id:
                    self.logger.error("Invalid start node for traversal")
                    return results
                    
                if start_node.node_id in visited or max_depth <= 0:
                    return results
                    
                visited.add(start_node.node_id)
                
                # Get direct connections with error handling
                try:
                    connections = await self._get_valid_connections(start_node, min_weight, visited)
                    if connections:
                        if not isinstance(connections, list):
                            self.logger.warning(f"Expected list from _get_valid_connections, got {type(connections)}")
                            connections = list(connections) if hasattr(connections, '__iter__') else []
                        results[1] = connections
                except Exception as e:
                    self.logger.error(f"Error getting valid connections: {e}")
                    return results
                
                # Recursively get deeper connections
                if max_depth > 1 and connections:
                    for node in connections:
                        try:
                            deeper = await self.traverse_network(node, max_depth - 1, min_weight, visited)
                            for depth, nodes in deeper.items():
                                if not isinstance(nodes, list):
                                    nodes = list(nodes) if hasattr(nodes, '__iter__') else []
                                current = results.get(depth + 1, [])
                                current.extend(nodes)
                                results[depth + 1] = current
                        except Exception as e:
                            self.logger.error(f"Error in recursive traversal for node {node.node_id}: {e}")
                            continue
                            
                return results
                
            except Exception as e:
                self.logger.error(f"Network traversal failed: {e}")
                return {}  # Return empty dict on error

    async def _get_valid_connections(
        self,
        node: T,
        min_weight: float,
        visited: Set[str]
    ) -> List[T]:
        """
        Get valid connections for a node applying connection management rules.
        
        Args:
            node: Source node
            min_weight: Minimum connection weight
            visited: Set of already visited node IDs
            
        Returns:
            List[T]: Valid connected nodes
        """
        if not self.connection_manager:
            raise NetworkError("Network requires a connection manager for traversal")
            
        connections: List[T] = []
        health_map = self.connection_manager.connection_health.get(node.node_id, {})
        
        # Use connections dict directly instead of get_connected_nodes
        for node_id, weight in node.connections.items():
            if node_id in visited:
                continue
                
            if weight < min_weight:
                continue
                
            connected = self.connection_manager._get_node_by_id(node_id)
            if connected and not connected.ghosted:
                # Check connection health if available
                health = health_map.get(node_id)
                if health and health.strength_history:
                    weight = health.strength_history[-1]
                if weight >= min_weight:
                    connections.append(connected)
                    
        return connections

    # -------------------------------------------------------------------------
    # Network Maintenance
    # -------------------------------------------------------------------------
    async def maintain_network(self) -> None:
        """
        Perform network maintenance operations.
        
        Process:
         1. Update activity tracking
         2. Check nodes for ghost transitions
         3. Handle pending consolidations if activity is low
         4. Update network state
        """
        if not self.config.enable_maintenance:
            return
        async with self.network_operation("maintain_network"):
            try:
                await self._maintain_nodes()
            except Exception as e:
                self.logger.error(f"Network maintenance failed: {e}")

    @abstractmethod
    async def _maintain_nodes(self) -> None:
        """
        Perform network-specific maintenance.
        """
        pass

    # -------------------------------------------------------------------------
    # State Management
    # -------------------------------------------------------------------------
    @abstractmethod
    async def _persist_node(self, node: T) -> None:
        """
        Persist node to storage. Must be implemented by subclasses.
        
        Args:
            node: Node to persist.
        """
        pass

    # -------------------------------------------------------------------------
    # Utility Methods
    # -------------------------------------------------------------------------
    def _get_platform_path(self, path: str) -> Path:
        """Get platform-appropriate path."""
        return (self.working_dir / path).resolve()

    def __len__(self) -> int:
        """Get number of nodes in network."""
        return len(self.nodes)

    def __contains__(self, node_id: str) -> bool:
        """Check if node exists in network."""
        return str(node_id) in self.nodes

    async def __aiter__(self) -> AsyncGenerator[T, None]:
        """Async iterator over nodes."""
        async with self.network_operation("iter_network"):
            for node in self.nodes.values():
                yield node
