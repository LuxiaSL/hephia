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
from typing import Dict, List, Optional, Set, TypeVar, Generic, NamedTuple

from loggers.loggers import MemoryLogger
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
    min_initial_weight: float = 0.2      # Minimum weight for new connections
    min_maintain_weight: float = 0.15    # Minimum weight to maintain connection
    strong_connection: float = 0.7       # Threshold for "strong" connections
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

    @asynccontextmanager
    async def connection_operation(self, operation_name: str = "unnamed"):
        """Async-safe context manager for connection operations."""
        async with shared_reentrant_lock:
            self._operation_count += 1
            self.logger.debug(f"Starting connection operation: {operation_name}")
            try:
                yield
            finally:
                self.logger.debug(f"Completed connection operation: {operation_name}")
                self._operation_count -= 1

    async def form_initial_connections(
        self,
        node: T,
        candidates: List[T], 
        max_candidates: Optional[int] = None,
        lock_acquired: bool = False
    ) -> Dict[str, float]:
        """
        Form initial bidirectional connections for a new node.
        Uses softmax scaling for natural connection distribution.

        Args:
            node: Newly created node.
            candidates: Potential nodes to connect to.
            max_candidates: Optional limit on connections to form.
            lock_acquired: Whether the lock is already acquired by caller.

        Returns:
            Dict mapping node IDs to connection weights.
        """
        async def _form_connections():
            try:
                # Validate inputs
                if not node or not node.raw_state:
                    self.logger.error("Cannot form connections - invalid node")
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
                            # Update other node's connections
                            other.connections[node.node_id] = weight
                            other.last_connection_update = time.time()

                            # Update health tracking for BOTH directions
                            await self._update_connection_health(node.node_id, other_id, weight)
                            await self._update_connection_health(other_id, node.node_id, weight)

                return connections

            except Exception as e:
                self.logger.error(f"Failed to form initial connections: {e}")
                self.logger.debug(f"Exception traceback:\n{''.join(traceback.format_tb(e.__traceback__))}")
                return {}

        if lock_acquired:
            return await _form_connections()
        else:
            async with self.connection_operation("form_initial_connections"):
                return await _form_connections()
            
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
    ) -> None:
        """
        Update connection weights for a node.

        Args:
            node: Node to update connections for.
            connection_filter: Optional subset of connections to update.
            skip_reciprocal: Skip updating reciprocal connections.
        """
        if lock_acquired:
            await self._update_connections_internal(node, connection_filter, skip_reciprocal)
        else:
            async with self.connection_operation("update_connections"):
                await self._update_connections_internal(node, connection_filter, skip_reciprocal)

    async def _update_connections_internal(
        self,
        node: T,
        connection_filter: Optional[Dict[str, float]],
        skip_reciprocal: bool
    ) -> None:
        """Internal implementation of connection updates."""
        try:
            connections = connection_filter or node.connections
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

                await self._update_connection_health(node.node_id, other_id, new_weight)

            pruned = set(connections.keys()) - set(updated.keys())
            if pruned:
                await self._handle_pruned_connections(node, pruned)

            node.connections = updated

            if not skip_reciprocal:
                await self._update_reciprocal_connections(node, updated)

        except Exception as e:
            self.logger.error(f"Failed to update connections: {e}")

    async def _update_connection_health(
        self,
        node_id: str,
        other_id: str,
        weight: float
    ) -> None:
        """
        Update health tracking for a connection.

        Args:
            node_id: ID of first node.
            other_id: ID of second node.
            weight: New connection weight.
        """
        try:
            if node_id not in self.connection_health:
                self.connection_health[node_id] = {}

            if other_id not in self.connection_health[node_id]:
                self.connection_health[node_id][other_id] = ConnectionHealth(last_updated=time.time())

            health = self.connection_health[node_id][other_id]
            health.last_updated = time.time()
            health.update_count += 1
            health.add_strength(weight)

        except Exception as e:
            self.logger.error(f"Failed to update connection health: {e}")

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
    ) -> None:
        """
        Update reciprocal connections to maintain consistency.

        Args:
            node: Node whose connections were updated.
            new_weights: New connection weights.
        """
        try:
            for other_id, weight in new_weights.items():
                other = self._get_node_by_id(other_id)
                if other and not getattr(other, 'ghosted', False):
                    other.connections[node.node_id] = weight
        except Exception as e:
            self.logger.error(f"Failed to update reciprocal connections: {e}")

    @abstractmethod
    def _get_node_by_id(self, node_id: str) -> Optional[T]:
        """
        Get node by ID. Must be implemented by specialized managers
        to use their network's node lookup.
        """
        pass

    async def maintain_connections(self, nodes: List[T]) -> None:
        """
        Perform maintenance on network connections.

        Args:
            nodes: List of nodes to maintain.
        """
        async with self.connection_operation("maintain_connections"):
            try:
                for node in nodes:
                    if getattr(node, 'ghosted', False):
                        continue
                    await self.update_connections(node, lock_acquired=True)
            except Exception as e:
                self.logger.error(f"Failed to maintain connections: {e}")

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
