"""
internal/modules/memory/networks/connections/base_manager.py

Implements core connection management functionality for memory networks.
Provides async-safe operations for establishing, maintaining, and updating
network connections with support for specialized network implementations.

Key capabilities:
- Async-safe connection operations using asyncio.Lock
- Configurable connection weights & thresholds
- Metric-based connection strength calculation
- Temporal connection dynamics
- Cross-platform compatibility
- Connection state preservation
"""

import traceback
import asyncio
import math
import time
from abc import ABC, abstractmethod
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Set, TypeVar, Generic, NamedTuple

from loggers.loggers import MemoryLogger
from .node_lock_manager import NodeLockManager, LockType
from .queue_manager import ConnectionUpdateQueue, UpdatePriority, UpdateReason, UpdateBatch
from ..reentrant_lock import shared_reentrant_lock
from ...metrics.orchestrator import MetricsConfiguration, RetrievalMetricsOrchestrator
from ...nodes.base_node import BaseMemoryNode

# Type variable for node specialization
T = TypeVar('T', bound=BaseMemoryNode)

class ConnectionError(Exception):
    """Base exception for connection management errors."""
    pass


class ConnectionOperation(Enum):
    """Types of connection operations."""
    FORMATION = auto()
    UPDATE = auto()
    PRUNE = auto()
    MERGE = auto()
    GHOST = auto()


@dataclass
class ConnectionThresholds:
    """Configuration thresholds for connection management."""
    min_initial_weight: float = 0.3      # Minimum weight for new connections
    min_maintain_weight: float = 0.25    # Minimum weight to maintain connection
    strong_connection: float = 0.8       # Threshold for "strong" connections
    merge_threshold: float = 0.85        # Weight suggesting nodes should merge
    temporal_window: float = 3600        # Time window for temporal bonuses (1 hour)
    max_connections: int = 50            # Soft limit on connections per node


@dataclass
class ConnectionWeights:
    """Weight configuration for connection scoring."""
    semantic_weight: float = 0.4         # Weight for semantic similarity
    emotional_weight: float = 0.2        # Weight for emotional resonance  
    state_weight: float = 0.2            # Weight for state similarity
    temporal_weight: float = 0.2         # Weight for temporal proximity

    def __post_init__(self):
        """Validate weights sum to 1.0"""
        total = (self.semantic_weight + self.emotional_weight +
                 self.state_weight + self.temporal_weight)
        if not math.isclose(total, 1.0, rel_tol=1e-9):
            raise ValueError(f"Weights must sum to 1.0, got {total}")


@dataclass
class ConnectionConfig:
    """Complete configuration for connection management."""
    thresholds: ConnectionThresholds = field(default_factory=ConnectionThresholds)
    weights: ConnectionWeights = field(default_factory=ConnectionWeights)
    metrics_config: Optional[MetricsConfiguration] = None
    enable_pruning: bool = True
    enable_merges: bool = True
    enable_temporal: bool = True


@dataclass
class ConnectionHealth:
    """Tracks health/status of a connection."""
    last_updated: float
    update_count: int = 0
    strength_history: List[float] = field(default_factory=list)
    max_history: int = 10  # Keep last N strength values

    def add_strength(self, value: float) -> None:
        """Add strength value to history with limiting."""
        self.strength_history.append(value)
        if len(self.strength_history) > self.max_history:
            self.strength_history.pop(0)

    @property
    def trend(self) -> float:
        """Calculate strength trend [-1, 1]."""
        if len(self.strength_history) < 2:
            return 0.0
        return self.strength_history[-1] - self.strength_history[0]


class MergeCandidate(NamedTuple):
    """Represents a potential merge target."""
    node_id: str
    weight: float
    health: ConnectionHealth


class BaseConnectionManager(Generic[T], ABC):
    """
    Abstract base class for connection management.

    Provides core functionality while allowing specialized implementations
    to define how connections are actually calculated and maintained.
    """

    def __init__(
        self,
        metrics_orchestrator: RetrievalMetricsOrchestrator,
        config: Optional[ConnectionConfig] = None
    ):
        """
        Initialize connection manager.

        Args:
            metrics_orchestrator: System for calculating similarity metrics.
            config: Optional configuration override.
        """
        self.metrics = metrics_orchestrator
        self.config = config or ConnectionConfig()

        self._operation_count = 0

        # Operation tracking (for example, pending operations per node)
        self.pending_operations: Dict[str, Set[ConnectionOperation]] = {}

        # Logging
        self.logger = MemoryLogger()

        # Health tracking for connections
        self.connection_health: Dict[str, Dict[str, ConnectionHealth]] = {}

        self.connection_queue: Optional[ConnectionUpdateQueue] = None
        self.queue_enabled = False

        self.node_lock_manager = NodeLockManager(
            default_timeout=30.0,
            deadlock_detection_timeout=5.0,
            enable_global_fallback=True
        )

    @asynccontextmanager
    async def connection_operation(self, operation_name: str = "unnamed"):
        """Async-safe context manager for connection operations using granular locking."""
        # For now, we'll determine affected nodes dynamically
        # This is a transitional implementation
        try:
            # Try to use granular locking for simple operations
            async with self.node_lock_manager.acquire_node_lock(
                node_id="temp_fallback",  # This will be improved in specific methods
                lock_type=LockType.WRITE,
                operation_name=operation_name
            ):
                self._operation_count += 1
                self.logger.debug(f"Starting connection operation: {operation_name}")
                try:
                    yield
                finally:
                    self.logger.debug(f"Completed connection operation: {operation_name}")
                    self._operation_count -= 1
        except Exception as e:
            # Fallback to global lock if granular locking fails
            self.logger.warning(f"Granular locking failed for {operation_name}, falling back to global lock: {e}")
            async with shared_reentrant_lock:
                self._operation_count += 1
                self.logger.debug(f"Starting connection operation (global fallback): {operation_name}")
                try:
                    yield
                finally:
                    self.logger.debug(f"Completed connection operation (global fallback): {operation_name}")
                    self._operation_count -= 1

    @asynccontextmanager
    async def connection_operation_for_nodes(
        self, 
        node_ids: Set[str], 
        operation_name: str = "unnamed",
        write_node_ids: Optional[Set[str]] = None
    ):
        """
        Granular locking context manager for operations on specific nodes.
        
        Args:
            node_ids: Set of node IDs that will be affected
            operation_name: Name for logging/debugging  
            write_node_ids: Set of node IDs that need write access (defaults to all)
        """
        if not node_ids:
            # No nodes specified, use global lock
            async with self.connection_operation(operation_name):
                yield
            return
        
        # Default to write access for all nodes if not specified
        if write_node_ids is None:
            write_node_ids = node_ids
        
        # Create lock specification
        node_specs = {}
        for node_id in node_ids:
            node_specs[str(node_id)] = LockType.WRITE if node_id in write_node_ids else LockType.READ
        
        try:
            if len(node_specs) == 1:
                # Single node - use simple lock
                node_id = next(iter(node_specs.keys()))
                lock_type = node_specs[node_id]
                async with self.node_lock_manager.acquire_node_lock(
                    node_id=node_id,
                    lock_type=lock_type,
                    operation_name=operation_name
                ):
                    self._operation_count += 1
                    self.logger.debug(f"Starting connection operation for node {node_id}: {operation_name}")
                    try:
                        yield
                    finally:
                        self.logger.debug(f"Completed connection operation for node {node_id}: {operation_name}")
                        self._operation_count -= 1
            else:
                # Multiple nodes - use multi-node lock
                async with self.node_lock_manager.acquire_multiple_node_locks(
                    node_specs=node_specs,
                    operation_name=operation_name
                ):
                    self._operation_count += 1
                    self.logger.debug(f"Starting connection operation for nodes {node_ids}: {operation_name}")
                    try:
                        yield
                    finally:
                        self.logger.debug(f"Completed connection operation for nodes {node_ids}: {operation_name}")
                        self._operation_count -= 1
                        
        except Exception as e:
            # Fallback to global lock if granular locking fails
            self.logger.warning(f"Granular locking failed for {operation_name} on nodes {node_ids}, falling back: {e}")
            async with self.connection_operation(operation_name + "_fallback"):
                yield
            
    async def form_initial_connections(
        self,
        node: T,
        candidates: List[T], 
        max_candidates: Optional[int] = None,
        lock_acquired: bool = False
    ) -> Dict[str, float]:
        """
        Form initial bidirectional connections for a new node.
        Uses granular locking for better concurrency.

        Args:
            node: Newly created node.
            candidates: Potential nodes to connect to.
            max_candidates: Optional limit on connections to form.
            lock_acquired: Whether the lock is already acquired by caller.

        Returns:
            Dict mapping node IDs to connection weights.
        """
        if lock_acquired:
            # Caller already has locks, proceed directly
            return await self._form_connections_internal(node, candidates, max_candidates)
        
        # Determine nodes that will be affected
        affected_node_ids = {str(node.node_id)}
        max_count = max_candidates or min(len(candidates), self.config.thresholds.max_connections)
        
        for other in candidates[:max_count]:
            if other and other.node_id and other.node_id != node.node_id:
                affected_node_ids.add(str(other.node_id))
        
        # Use granular locking for all affected nodes
        async with self.connection_operation_for_nodes(
            node_ids=affected_node_ids,
            operation_name="form_initial_connections",
            write_node_ids=affected_node_ids  # All nodes need write access
        ):
            return await self._form_connections_internal(node, candidates, max_candidates)
        
    async def _form_connections_internal(
        self,
        node: T,
        candidates: List[T], 
        max_candidates: Optional[int] = None
    ) -> Dict[str, float]:
        """
        Form initial bidirectional connections for a new node (internal implementation).
        Uses softmax scaling for natural connection distribution.

        Args:
            node: Newly created node.
            candidates: Potential nodes to connect to.
            max_candidates: Optional limit on connections to form.
            lock_acquired: Whether the lock is already acquired by caller.

        Returns:
            Dict mapping node IDs to connection weights.
        """
        try:
            # Validate inputs
            if not node:
                self.logger.error("Cannot form connections - invalid node")
                return {}
            
            if not node.raw_state:
                self.logger.error("Cannot form connections - node has no state")
                return {}
                
            max_count = max_candidates or self.config.thresholds.max_connections
            connection_scores = []

            # Calculate raw similarity scores (ONE TIME for each pair)
            connection_cache = {}  # Cache weights to ensure consistency
            for other in candidates[:max_count]:
                if not other or other.node_id == node.node_id:
                    continue
                    
                try:
                    # Calculate weight once and cache it
                    score = await self._calculate_connection_weight(
                        node,
                        other,
                        ConnectionOperation.FORMATION
                    )
                    if score >= self.config.thresholds.min_initial_weight:
                        connection_scores.append((other.node_id, score))
                        connection_cache[other.node_id] = score
                except Exception as e:
                    self.logger.error(f"Error calculating connection weight: {e}")
                    continue

            # Apply softmax scaling unless extremely similar
            connections = {}
            if connection_scores:
                scores = [score for _, score in connection_scores]
                max_score = max(scores)
                
                # If any are very strong, use raw scores
                if max_score > self.config.thresholds.strong_connection:
                    connections = {node_id: score for node_id, score in connection_scores}
                else:
                    # Apply softmax scaling
                    scaled = self._softmax(scores)
                    connections = {
                        node_id: scaled[i] 
                        for i, (node_id, _) in enumerate(connection_scores)
                    }

                # Form reciprocal connections with SAME weights
                for other_id, weight in connections.items():
                    other = self._get_node_by_id(other_id)
                    if other and not getattr(other, 'ghosted', False):
                        if self._can_accept_connection(other) or self._make_room_for_connection(other, weight):
                            # Update other node's connections
                            other.connections[node.node_id] = weight
                            other.last_connection_update = time.time()

                            # Update health tracking for BOTH directions
                            await self._update_connection_health_internal(node.node_id, other_id, weight)
                            await self._update_connection_health_internal(other_id, node.node_id, weight)
                        else:
                            self.logger.debug(f"Skipped reciprocal connection from {other_id} to {node.node_id}: target node at capacity")

            return connections

        except Exception as e:
            self.logger.error(f"Failed to form initial connections: {e}")
            return {}
    
    async def _update_connection_health_internal(
        self,
        node_id: str,
        other_id: str,
        weight: float
    ) -> bool:
        """
        Internal connection health update (assumes locks already held).
        """
        try:
            if str(node_id) not in self.connection_health:
                self.connection_health[str(node_id)] = {}

            if str(other_id) not in self.connection_health[str(node_id)]:
                self.connection_health[str(node_id)][str(other_id)] = ConnectionHealth(last_updated=time.time())

            health = self.connection_health[str(node_id)][str(other_id)]
            health.last_updated = time.time()
            health.update_count += 1
            health.add_strength(weight)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to update connection health: {e}")
            return False
            
    def _enforce_connection_limit(self, node: T, new_connections: Dict[str, float]) -> Dict[str, float]:
        """
        Enforce max_connections limit on a node by keeping only the strongest connections.
        
        Args:
            node: Node to enforce limit on
            new_connections: New connections to potentially add
            
        Returns:
            Dict of connections that respect the limit
        """
        max_connections = self.config.thresholds.max_connections
        
        # Combine existing and new connections
        all_connections = {**node.connections, **new_connections}
        
        # If under limit, return all
        if len(all_connections) <= max_connections:
            return new_connections
        
        # Sort by weight (descending) and take top max_connections
        sorted_connections = sorted(all_connections.items(), key=lambda x: x[1], reverse=True)
        top_connections = dict(sorted_connections[:max_connections])
        
        # Return only the new connections that made the cut
        filtered_new = {
            conn_id: weight for conn_id, weight in new_connections.items()
            if conn_id in top_connections
        }
        
        # Update node's connections to respect limit
        node.connections = {
            conn_id: weight for conn_id, weight in top_connections.items()
            if conn_id not in new_connections  # Keep existing that made the cut
        }
        
        if len(all_connections) > max_connections:
            removed_count = len(all_connections) - max_connections
            self.logger.debug(f"Enforced connection limit on node {node.node_id}: removed {removed_count} weakest connections")
        
        return filtered_new

    def _can_accept_connection(self, node: T) -> bool:
        """
        Check if a node can accept a new connection without exceeding max_connections.
        
        Args:
            node: Node to check
            
        Returns:
            True if node can accept more connections
        """
        return len(node.connections) < self.config.thresholds.max_connections

    def _make_room_for_connection(self, node: T, new_weight: float) -> bool:
        """
        Make room for a new connection by removing the weakest existing connection if needed.
        
        Args:
            node: Node to make room in
            new_weight: Weight of the new connection
            
        Returns:
            True if room was made (new connection is stronger than weakest existing)
        """
        if len(node.connections) < self.config.thresholds.max_connections:
            return True
        
        # Find weakest connection
        if not node.connections:
            return True
            
        weakest_id = min(node.connections, key=node.connections.get)
        weakest_weight = node.connections[weakest_id]
        
        # Only make room if new connection is stronger
        if new_weight > weakest_weight:
            self.logger.debug(f"Removing weakest connection {weakest_id} (weight: {weakest_weight:.3f}) to make room for stronger connection (weight: {new_weight:.3f})")
            node.connections.pop(weakest_id)
            return True
        
        return False
    
    async def _persist_node_through_network(self, node: T, lock_acquired: bool = False) -> None:
        """Persist node through the network's persistence mechanism."""
        pass  # Implemented by specialized managers
            
    @abstractmethod
    async def _calculate_connection_weight(
        self,
        node_a: T,
        node_b: T,
        operation: ConnectionOperation
    ) -> float:
        """
        Calculate connection weight between two nodes.
        Must be implemented by specialized managers.

        Args:
            node_a: First node.
            node_b: Second node.
            operation: Type of operation triggering calculation.

        Returns:
            float: Connection weight [0,1].
        """
        pass

    def _softmax(self, scores: List[float]) -> List[float]:
        """Calculate softmax of input scores."""
        if not scores:
            return []
        exp_scores = [math.exp(s) for s in scores]
        sum_exp = sum(exp_scores)
        return [e / sum_exp for e in exp_scores]

    async def _calculate_temporal_bonus(
        self,
        node_a: T,
        node_b: T
    ) -> float:
        """
        Calculate temporal proximity bonus for connection weight.
        Recent nodes get higher base connection weights.

        Args:
            node_a: First node.
            node_b: Second node.

        Returns:
            float: Temporal bonus factor [0,1].
        """
        if not self.config.enable_temporal:
            return 0.0

        try:
            time_diff = abs(node_a.timestamp - node_b.timestamp)
            window = self.config.thresholds.temporal_window
            # Exponential decay based on time difference.
            return math.exp(-time_diff / window)
        except Exception as e:
            self.logger.error(f"Failed to calculate temporal bonus: {e}")
            return 0.0

    async def prune_connections(
        self,
        node: T,
        force: bool = False
    ) -> Set[str]:
        """
        Remove weak connections from node.

        Args:
            node: Node to prune connections from.
            force: Force pruning even if disabled.

        Returns:
            Set of removed connection IDs.
        """
        if not force and not self.config.enable_pruning:
            return set()

        async with self.connection_operation("prune_connections"):
            try:
                to_remove = set()
                threshold = self.config.thresholds.min_maintain_weight

                for other_id, weight in list(node.connections.items()):
                    if weight < threshold:
                        to_remove.add(other_id)

                # Remove pruned connections
                for other_id in to_remove:
                    node.connections.pop(other_id, None)

                if to_remove:
                    self.logger.debug(f"Pruned {len(to_remove)} connections from node {node.node_id}")

                return to_remove

            except Exception as e:
                self.logger.error(f"Failed to prune connections: {e}")
                return set()

    async def update_connections(
        self,
        node: T,
        connection_filter: Optional[Dict[str, float]] = None,
        skip_reciprocal: bool = False,
        lock_acquired: bool = False
    ) -> List[str]:
        """
        Update connection weights for a node using granular locking.
        """
        if lock_acquired:
            # Caller already has locks, proceed directly
            return await self._update_connections_internal(node, connection_filter, skip_reciprocal)
        
        # Determine affected nodes
        affected_node_ids = {str(node.node_id)}
        connections_to_check = connection_filter or node.connections
        
        for other_id in connections_to_check.keys():
            if other_id and other_id != node.node_id:
                affected_node_ids.add(str(other_id))
        
        # Use granular locking
        async with self.connection_operation_for_nodes(
            node_ids=affected_node_ids,
            operation_name="update_connections",
            write_node_ids=affected_node_ids  # All nodes may need updates
        ):
            return await self._update_connections_internal(node, connection_filter, skip_reciprocal)

    async def _update_connections_internal(
        self,
        node: T,
        connection_filter: Optional[Dict[str, float]],
        skip_reciprocal: bool
    ) -> List[str]:
        """
        Internal implementation of connection updates with max_connections enforcement.
        
        Returns:
            List of node IDs that had their connections updated.
        """
        try:
            connections = connection_filter or node.connections.copy()
            updated: Dict[str, float] = {}

            for other_id, old_weight in connections.items():
                if not other_id or other_id == node.node_id:
                    continue

                other_node = self._get_node_by_id(other_id)
                if not other_node:
                    continue

                new_weight = await self._calculate_connection_weight(
                    node,
                    other_node,
                    ConnectionOperation.UPDATE
                )

                if new_weight >= self.config.thresholds.min_maintain_weight:
                    updated[other_id] = new_weight

                await self._update_connection_health_internal(node.node_id, other_id, new_weight)

            updated = self._enforce_connection_limit(node, updated)
            
            pruned = set(connections.keys()) - set(updated.keys())
            if pruned:
                await self._handle_pruned_connections(node, pruned)

            node.connections = updated

            updated_nodes = []
            if not skip_reciprocal:
                updated_nodes = await self._update_reciprocal_connections_safe(node, updated)

            return updated_nodes

        except Exception as e:
            self.logger.error(f"Failed to update connections: {e}")
            return []

    async def _update_reciprocal_connections_safe(
        self,
        node: T,
        new_weights: Dict[str, float]
    ) -> List[str]:
        """
        Update reciprocal connections while respecting max_connections limits.
        
        Args:
            node: Node whose connections were updated.
            new_weights: New connection weights.
            
        Returns:
            List of node IDs that had their connections updated.
        """
        try:
            updated_nodes = []
            for other_id, weight in new_weights.items():
                other = self._get_node_by_id(other_id)
                if other and not getattr(other, 'ghosted', False):
                    # Respect max_connections on reciprocal updates
                    if self._can_accept_connection(other) or self._make_room_for_connection(other, weight):
                        other.connections[str(node.node_id)] = weight
                        other.last_connection_update = time.time()
                        updated_nodes.append(other_id)
                    else:
                        self.logger.debug(f"Skipped reciprocal connection update {other_id} -> {node.node_id}: target at capacity")
            return updated_nodes
        except Exception as e:
            self.logger.error(f"Failed to update reciprocal connections: {e}")
            return []

    async def _update_connection_health(
        self,
        node_id: str,
        other_id: str,
        weight: float
    ) -> bool:
        """
        Update health tracking for a connection with granular locking.
        """
        # Use read locks since we're just updating health tracking, not the nodes themselves
        affected_node_ids = {str(node_id), str(other_id)}
        
        try:
            async with self.connection_operation_for_nodes(
                node_ids=affected_node_ids,
                operation_name="update_connection_health",
                write_node_ids=set()  # No write access needed for health tracking
            ):
                if str(node_id) not in self.connection_health:
                    self.connection_health[str(node_id)] = {}

                if str(other_id) not in self.connection_health[str(node_id)]:
                    self.connection_health[str(node_id)][str(other_id)] = ConnectionHealth(last_updated=time.time())

                health = self.connection_health[str(node_id)][str(other_id)]
                health.last_updated = time.time()
                health.update_count += 1
                health.add_strength(weight)
                
                return True
                
        except Exception as e:
            self.logger.error(f"Failed to update connection health: {e}")
            return False

    async def _handle_pruned_connections(
        self,
        node: T,
        pruned_ids: Set[str]
    ) -> None:
        """
        Handle connections that were pruned during update.
        Preserves connection data for potential future restoration.

        Args:
            node: Node that had connections pruned.
            pruned_ids: Set of pruned connection IDs.
        """
        try:
            for other_id in pruned_ids:
                health = self.connection_health.get(node.node_id, {}).get(other_id)
                if not health:
                    continue

                if any(s > self.config.thresholds.strong_connection for s in health.strength_history):
                    await self._preserve_ghost_connection(node, self._get_node_by_id(other_id), health)

                if node.node_id in self.connection_health:
                    self.connection_health[node.node_id].pop(other_id, None)

        except Exception as e:
            self.logger.error(f"Failed to handle pruned connections: {e}")

    async def _preserve_ghost_connection(
        self,
        node: T,
        other: T,
        health: ConnectionHealth
    ) -> None:
        """
        Preserve a pruned connection for potential future revival.

        Args:
            node: Node that had connection pruned.
            other: Other node in connection.
            health: Health history of connection.
        """
        try:
            if not hasattr(node, 'ghost_connections'):
                node.ghost_connections = []

            ghost_connection = {
                'node_id': other.node_id,
                'last_weight': health.strength_history[-1],
                'max_weight': max(health.strength_history),
                'timestamp': time.time(),
                'update_count': health.update_count,
                'state_snapshot': {
                    'timestamp': other.timestamp,
                    'strength': getattr(other, 'strength', None)
                }
            }
            node.ghost_connections.append(ghost_connection)

        except Exception as e:
            self.logger.error(f"Failed to preserve ghost connection: {e}")

    async def _update_reciprocal_connections(
        self,
        node: T,
        new_weights: Dict[str, float]
    ) -> List[str]:
        """
        Update reciprocal connections to maintain consistency.
        
        Args:
            node: Node whose connections were updated.
            new_weights: New connection weights.
            
        Returns:
            List of node IDs that had their connections updated.
        """
        try:
            updated_nodes = []
            for other_id, weight in new_weights.items():
                other = self._get_node_by_id(other_id)
                if other and not getattr(other, 'ghosted', False):
                    other.connections[str(node.node_id)] = weight
                    other.last_connection_update = time.time()
                    updated_nodes.append(other_id)
            return updated_nodes
        except Exception as e:
            self.logger.error(f"Failed to update reciprocal connections: {e}")
            return []

    @abstractmethod
    def _get_node_by_id(self, node_id: str) -> Optional[T]:
        """
        Get node by ID. Must be implemented by specialized managers
        to use their network's node lookup.
        """
        pass

    async def maintain_connections(self, nodes: List[T]) -> None:
        """
        Perform maintenance on network connections using efficient batching.
        """
        if not nodes:
            return
        
        # Group nodes into batches to avoid lock contention
        batch_size = min(10, len(nodes))  # Process up to 10 nodes at once
        
        for i in range(0, len(nodes), batch_size):
            batch = nodes[i:i + batch_size]
            
            # Get all affected node IDs for this batch
            affected_node_ids = set()
            for node in batch:
                if not getattr(node, 'ghosted', False):
                    affected_node_ids.add(str(node.node_id))
                    # Add connected nodes
                    for conn_id in node.connections.keys():
                        affected_node_ids.add(str(conn_id))
            
            if not affected_node_ids:
                continue
                
            # Process this batch with granular locking
            try:
                async with self.connection_operation_for_nodes(
                    node_ids=affected_node_ids,
                    operation_name=f"maintain_connections_batch_{i//batch_size}",
                    write_node_ids=affected_node_ids
                ):
                    for node in batch:
                        if not getattr(node, 'ghosted', False):
                            await self._update_connections_internal(
                                node, 
                                connection_filter=None,
                                skip_reciprocal=False
                            )
            except Exception as e:
                self.logger.error(f"Failed to maintain connections for batch {i//batch_size}: {e}")

    @abstractmethod
    async def transfer_connections(
        self,
        from_node: T,
        to_node: T,
        transfer_weight: float = 0.9
    ) -> None:
        """
        Transfer connections from one node to another with proper weight adjustment.

        Args:
            from_node: Source node losing connections.
            to_node: Target node receiving connections.
            transfer_weight: Dampening factor for transferred connections.
        """
        pass

    # ---
    # Helper: run a synchronous function with a timeout
    # ---
    async def _run_with_timeout(self, func, timeout_secs: float = 1.0):
        """
        Run a synchronous function in a separate thread with a timeout.

        Args:
            func: Function to run.
            timeout_secs: Timeout in seconds.

        Returns:
            The result of func() if completed in time; otherwise, None.
        """
        try:
            # Use asyncio.to_thread to avoid blocking the event loop.
            return await asyncio.wait_for(asyncio.to_thread(func), timeout=timeout_secs)
        except asyncio.TimeoutError:
            self.logger.warning(f"Function {getattr(func, '__name__', 'anonymous')} timed out after {timeout_secs} seconds")
            return None
        except Exception as e:
            self.logger.error(f"Error executing function {getattr(func, '__name__', 'anonymous')}: {e}")
            return None
        
    # ---
    # Queue management methods
    # ---
    async def initialize_queue(self, **queue_kwargs):
        """Initialize and start the connection update queue."""
        self.connection_queue = ConnectionUpdateQueue(**queue_kwargs)
        
        # Set the connection manager reference for batch processing
        self.connection_queue.set_connection_manager(self)
        
        await self.connection_queue.start()
        self.queue_enabled = True
        self.logger.info("Connection update queue initialized")

    async def shutdown_queue(self):
        """Shutdown the connection update queue."""
        if self.connection_queue:
            await self.connection_queue.stop()
            self.queue_enabled = False

    async def queue_connection_update(
        self,
        node: T,
        priority: UpdatePriority = UpdatePriority.NORMAL,
        reason: UpdateReason = UpdateReason.NODE_ACCESS,
        lock_acquired: bool = False,
        **kwargs
    ) -> bool:
        """Queue a connection update request."""
        if not self.queue_enabled or not self.connection_queue:
            # Fallback to immediate processing
            return await self._immediate_update_fallback(node, lock_acquired, **kwargs)
        
        return await self.connection_queue.enqueue_update(
            node=node,
            priority=priority,
            reason=reason,
            **kwargs
        )

    async def _immediate_update_fallback(self, node: T, lock_acquired: bool = False, **kwargs) -> List[str]:
        """Fallback to immediate processing when queue unavailable."""
        try:
            # Process immediately and return updated node IDs for persistence
            if lock_acquired:
                updated_node_ids = await self._update_connections_internal(
                    node, 
                    kwargs.get('connection_filter'), 
                    kwargs.get('skip_reciprocal', False)
                )
            else:
                updated_node_ids = await self.update_connections(node, lock_acquired=False, **kwargs)
            
            return updated_node_ids if isinstance(updated_node_ids, list) else []
            
        except Exception as e:
            self.logger.error(f"Immediate update fallback failed: {e}")
            return []
    
    async def process_queued_batch_with_locks(self, batch: UpdateBatch) -> List[Dict[str, Any]]:
        """
        Enhanced batch processing that coordinates queue and lock systems.
        """
        # Determine all affected nodes from the batch
        affected_node_ids = set()
        node_refs = {}  # Keep references to prevent garbage collection
        
        for request in batch.requests:
            node = request.get_node()
            if node:
                affected_node_ids.add(str(request.node_id))
                node_refs[str(request.node_id)] = node
                
                # Add connected nodes that might be updated
                for conn_id in node.connections.keys():
                    affected_node_ids.add(str(conn_id))
        
        if not affected_node_ids:
            return [{
                'request_id': req.request_id,
                'success': False,
                'error': 'No valid nodes in batch'
            } for req in batch.requests]
        
        # Process the entire batch under a single lock operation
        async with self.connection_operation_for_nodes(
            node_ids=affected_node_ids,
            operation_name=f"process_batch_{batch.batch_id}",
            write_node_ids=affected_node_ids
        ):
            # Pre-warm cache if this is a cognitive batch
            if hasattr(self, 'warm_connection_cache_for_batch'):
                valid_nodes = [node for node in node_refs.values() if node]
                await self.warm_connection_cache_for_batch(valid_nodes)
            
            # Process each request in the batch
            results = []
            for request in batch.requests:
                try:
                    node = request.get_node()
                    if node is None:
                        results.append({
                            'request_id': request.request_id,
                            'success': False,
                            'error': 'Node no longer available'
                        })
                        continue
                    
                    # Process connection updates (locks already held)
                    updated_node_ids = await self._update_connections_internal(
                        node=node,
                        connection_filter=request.connection_filter,
                        skip_reciprocal=request.skip_reciprocal
                    )
                    
                    # Handle persistence through network
                    await self._persist_node_through_network(node, lock_acquired=True)
                    
                    for other_id in updated_node_ids:
                        other_node = self._get_node_by_id(str(other_id))
                        if other_node:
                            await self._persist_node_through_network(other_node, lock_acquired=True)
                    
                    results.append({
                        'request_id': request.request_id,
                        'node_id': request.node_id,
                        'success': True,
                        'updated_nodes': updated_node_ids,
                        'processed_at': time.time()
                    })
                    
                except Exception as e:
                    results.append({
                        'request_id': request.request_id,
                        'success': False,
                        'error': str(e)
                    })
            
            return results
    
    # ---
    # Lock utilities
    # ---
    def cleanup_locks(self) -> Dict[str, int]:
        """Clean up unused locks and return statistics."""
        return {
            'locks_cleaned': self.node_lock_manager.cleanup_unused_locks(),
            'lock_stats': self.node_lock_manager.get_stats()
        }

    def get_lock_stats(self) -> Dict[str, Any]:
        """Get current lock manager statistics."""
        return self.node_lock_manager.get_stats()
    
    async def handle_lock_failure(self, operation_name: str, affected_nodes: Set[str], error: Exception):
        """
        Handle lock acquisition failures with appropriate fallbacks.
        """
        self.logger.error(f"Lock failure in {operation_name} for nodes {affected_nodes}: {error}")
        
        # Increment failure stats
        if hasattr(self, 'node_lock_manager'):
            stats = self.node_lock_manager.get_stats()
            stats['operation_failures'] = stats.get('operation_failures', 0) + 1
    
        return False
    
    def get_combined_stats(self) -> Dict[str, Any]:
        """
        Get combined statistics from both queue and lock managers.
        """
        stats = {
            'queue_stats': {},
            'lock_stats': {},
            'connection_health': {
                'total_connections': sum(len(health_dict) for health_dict in self.connection_health.values()),
                'nodes_with_connections': len(self.connection_health)
            }
        }
        
        if hasattr(self, 'connection_queue') and self.connection_queue:
            stats['queue_stats'] = self.connection_queue.get_queue_stats()
        
        if hasattr(self, 'node_lock_manager'):
            stats['lock_stats'] = self.node_lock_manager.get_stats()
        
        return stats