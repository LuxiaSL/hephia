"""
node_lock_manager.py

Node-level lock management system for memory networks.
Provides granular per-node locking with deadlock prevention,
timeout handling, and fallback mechanisms.

Key capabilities:
- Per-node read-write locks using aiorwlock
- Deadlock prevention through ordered lock acquisition
- Timeout-based lock acquisition with fallback
- Global lock escalation for complex operations
- Comprehensive error handling and cleanup
- Integration with existing queue and connection systems
"""

import asyncio
import time
import weakref
from aiorwlock import RWLock
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Set, Optional, Any, AsyncGenerator, Union

from loggers.loggers import MemoryLogger
from ..reentrant_lock import shared_reentrant_lock


class LockType(Enum):
    """Types of locks that can be acquired."""
    READ = auto()
    WRITE = auto()


class LockScope(Enum):
    """Scope of lock operations."""
    SINGLE_NODE = auto()
    MULTIPLE_NODES = auto()
    NETWORK_WIDE = auto()


@dataclass
class LockRequest:
    """Individual lock request for a node."""
    node_id: str
    lock_type: LockType
    timeout: float = 30.0
    acquired_at: Optional[float] = None
    request_id: str = field(default_factory=lambda: f"req_{int(time.time() * 1000) % 1000000}")


@dataclass
class LockOperation:
    """Complete lock operation involving multiple nodes."""
    operation_name: str
    requests: List[LockRequest]
    scope: LockScope
    started_at: float = field(default_factory=time.time)
    operation_id: str = field(default_factory=lambda: f"op_{int(time.time() * 1000) % 1000000}")
    
    @property
    def node_ids(self) -> Set[str]:
        """Get all node IDs involved in this operation."""
        return {req.node_id for req in self.requests}
    
    @property
    def write_node_ids(self) -> Set[str]:
        """Get node IDs that require write locks."""
        return {req.node_id for req in self.requests if req.lock_type == LockType.WRITE}


class NodeLockManager:
    """
    Manages per-node locks for memory network operations.
    
    Provides deadlock-free, timeout-safe locking with fallback mechanisms
    for both simple single-node and complex multi-node operations.
    """
    
    def __init__(
        self,
        default_timeout: float = 30.0,
        deadlock_detection_timeout: float = 5.0,
        max_concurrent_operations: int = 100,
        enable_global_fallback: bool = True
    ):
        """
        Initialize the node lock manager.
        
        Args:
            default_timeout: Default timeout for lock acquisition (seconds)
            deadlock_detection_timeout: How long to wait before assuming deadlock
            max_concurrent_operations: Maximum concurrent lock operations
            enable_global_fallback: Whether to fall back to global lock on deadlock
        """
        self.default_timeout = default_timeout
        self.deadlock_detection_timeout = deadlock_detection_timeout
        self.max_concurrent_operations = max_concurrent_operations
        self.enable_global_fallback = enable_global_fallback
        
        # Lock storage - using weak references to allow garbage collection
        self._node_locks: Dict[str, RWLock] = {}
        self._lock_refs: weakref.WeakValueDictionary = weakref.WeakValueDictionary()
        
        # Global fallback lock
        self._global_fallback_lock = asyncio.RWLock() if hasattr(asyncio, 'RWLock') else shared_reentrant_lock
        
        # Operation tracking
        self._active_operations: Dict[str, LockOperation] = {}
        self._operation_semaphore = asyncio.Semaphore(max_concurrent_operations)
        
        # Deadlock detection
        self._lock_waiters: Dict[str, Set[str]] = {}  # node_id -> set of operation_ids waiting
        self._operation_locks: Dict[str, Set[str]] = {}  # operation_id -> set of node_ids held
        
        # Statistics
        self._stats = {
            'locks_acquired': 0,
            'locks_released': 0,
            'deadlocks_detected': 0,
            'global_fallbacks': 0,
            'timeouts': 0,
            'operations_completed': 0
        }
        
        self.logger = MemoryLogger

    def _get_node_lock(self, node_id: str) -> RWLock:
        """Get or create a lock for the specified node."""
        if node_id not in self._node_locks:
            lock = RWLock()
            self._node_locks[node_id] = lock
            self._lock_refs[node_id] = lock  # Track for cleanup
        return self._node_locks[node_id]

    @asynccontextmanager
    async def acquire_node_lock(
        self,
        node_id: str,
        lock_type: LockType = LockType.WRITE,
        timeout: Optional[float] = None,
        operation_name: str = "single_node_op"
    ) -> AsyncGenerator[None, None]:
        """
        Acquire a lock for a single node.
        
        Args:
            node_id: ID of node to lock
            lock_type: READ or WRITE lock
            timeout: Lock acquisition timeout
            operation_name: Name for logging/debugging
        """
        timeout = timeout or self.default_timeout
        request = LockRequest(node_id=str(node_id), lock_type=lock_type, timeout=timeout)
        operation = LockOperation(
            operation_name=operation_name,
            requests=[request],
            scope=LockScope.SINGLE_NODE
        )
        
        async with self._acquire_locks_for_operation(operation):
            yield

    @asynccontextmanager
    async def acquire_multiple_node_locks(
        self,
        node_specs: Dict[str, LockType],
        timeout: Optional[float] = None,
        operation_name: str = "multi_node_op"
    ) -> AsyncGenerator[None, None]:
        """
        Acquire locks for multiple nodes with deadlock prevention.
        
        Args:
            node_specs: Dict mapping node_id -> LockType
            timeout: Lock acquisition timeout
            operation_name: Name for logging/debugging
        """
        timeout = timeout or self.default_timeout
        requests = [
            LockRequest(node_id=str(node_id), lock_type=lock_type, timeout=timeout)
            for node_id, lock_type in node_specs.items()
        ]
        
        operation = LockOperation(
            operation_name=operation_name,
            requests=requests,
            scope=LockScope.MULTIPLE_NODES
        )
        
        async with self._acquire_locks_for_operation(operation):
            yield

    @asynccontextmanager
    async def _acquire_locks_for_operation(self, operation: LockOperation) -> AsyncGenerator[None, None]:
        """
        Core lock acquisition logic with deadlock prevention and fallback.
        """
        async with self._operation_semaphore:
            self._active_operations[operation.operation_id] = operation
            
            try:
                # Attempt granular locking first
                success = await self._try_acquire_granular_locks(operation)
                
                if not success and self.enable_global_fallback:
                    # Fall back to global lock
                    self.logger.warning(f"Falling back to global lock for operation {operation.operation_name}")
                    self._stats['global_fallbacks'] += 1
                    
                    async with self._acquire_global_fallback_lock(operation):
                        yield
                elif success:
                    try:
                        yield
                    finally:
                        await self._release_operation_locks(operation)
                else:
                    raise TimeoutError(f"Failed to acquire locks for operation {operation.operation_name}")
                    
            finally:
                self._active_operations.pop(operation.operation_id, None)
                self._stats['operations_completed'] += 1

    async def _try_acquire_granular_locks(self, operation: LockOperation) -> bool:
        """
        Attempt to acquire all locks for an operation using granular locking.
        
        Returns:
            bool: True if all locks acquired successfully, False otherwise
        """
        try:
            # Sort requests by node_id for consistent ordering (deadlock prevention)
            sorted_requests = sorted(operation.requests, key=lambda r: r.node_id)
            
            # Track this operation as waiting for locks
            for request in sorted_requests:
                if request.node_id not in self._lock_waiters:
                    self._lock_waiters[request.node_id] = set()
                self._lock_waiters[request.node_id].add(operation.operation_id)
            
            # Acquire locks in order
            acquired_locks = []
            
            for request in sorted_requests:
                try:
                    lock = self._get_node_lock(request.node_id)
                    
                    # Check for potential deadlock before acquiring
                    if self._would_cause_deadlock(operation.operation_id, request.node_id):
                        self.logger.warning(f"Potential deadlock detected for operation {operation.operation_name}")
                        self._stats['deadlocks_detected'] += 1
                        
                        # Release any locks we've already acquired
                        await self._release_acquired_locks(acquired_locks)
                        return False
                    
                    # Acquire the lock with timeout
                    if request.lock_type == LockType.READ:
                        lock_coro = lock.reader_lock.acquire()
                    else:
                        lock_coro = lock.writer_lock.acquire()
                    
                    await asyncio.wait_for(lock_coro, timeout=request.timeout)
                    
                    request.acquired_at = time.time()
                    acquired_locks.append((lock, request))
                    self._stats['locks_acquired'] += 1
                    
                    # Update tracking
                    if operation.operation_id not in self._operation_locks:
                        self._operation_locks[operation.operation_id] = set()
                    self._operation_locks[operation.operation_id].add(request.node_id)
                    
                    # Remove from waiters
                    self._lock_waiters.get(request.node_id, set()).discard(operation.operation_id)
                    
                except asyncio.TimeoutError:
                    self.logger.error(f"Timeout acquiring lock for node {request.node_id}")
                    self._stats['timeouts'] += 1
                    
                    # Release any locks we've already acquired
                    await self._release_acquired_locks(acquired_locks)
                    return False
                    
                except Exception as e:
                    self.logger.error(f"Error acquiring lock for node {request.node_id}: {e}")
                    
                    # Release any locks we've already acquired
                    await self._release_acquired_locks(acquired_locks)
                    return False
            
            # Store acquired locks for later release
            operation._acquired_locks = acquired_locks
            return True
            
        except Exception as e:
            self.logger.error(f"Error in granular lock acquisition: {e}")
            return False

    def _would_cause_deadlock(self, operation_id: str, node_id: str) -> bool:
        """
        Simple deadlock detection based on waiting patterns.
        
        This is a conservative check - it may report false positives but
        should avoid actual deadlocks.
        """
        try:
            # If we're already holding locks and there are other operations waiting
            # for nodes we want, and we're waiting for nodes they hold, potential deadlock
            our_held_nodes = self._operation_locks.get(operation_id, set())
            other_waiters = self._lock_waiters.get(node_id, set()) - {operation_id}
            
            for other_op_id in other_waiters:
                other_held_nodes = self._operation_locks.get(other_op_id, set())
                
                # If they hold nodes we want and we hold nodes they might want
                if our_held_nodes & other_held_nodes:
                    return True
                    
                # Check for circular wait patterns
                for our_node in our_held_nodes:
                    other_waiters_for_our_node = self._lock_waiters.get(our_node, set())
                    if other_op_id in other_waiters_for_our_node:
                        return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Error in deadlock detection: {e}")
            return True  # Conservative: assume deadlock to be safe

    async def _release_acquired_locks(self, acquired_locks: List[tuple]) -> None:
        """Release a list of acquired locks."""
        for lock, request in reversed(acquired_locks):  # Release in reverse order
            try:
                if request.lock_type == LockType.READ:
                    lock.reader_lock.release()
                else:
                    lock.writer_lock.release()
                self._stats['locks_released'] += 1
            except Exception as e:
                self.logger.error(f"Error releasing lock for node {request.node_id}: {e}")

    async def _release_operation_locks(self, operation: LockOperation) -> None:
        """Release all locks held by an operation."""
        if hasattr(operation, '_acquired_locks'):
            await self._release_acquired_locks(operation._acquired_locks)
            
        # Clean up tracking
        self._operation_locks.pop(operation.operation_id, None)
        for request in operation.requests:
            waiters = self._lock_waiters.get(request.node_id, set())
            waiters.discard(operation.operation_id)
            if not waiters:
                self._lock_waiters.pop(request.node_id, None)

    @asynccontextmanager
    async def _acquire_global_fallback_lock(self, operation: LockOperation) -> AsyncGenerator[None, None]:
        """Acquire global fallback lock for complex operations."""
        if hasattr(self._global_fallback_lock, '__aenter__'):
            # It's an async context manager
            async with self._global_fallback_lock:
                yield
        else:
            # It's the old shared_reentrant_lock
            async with self._global_fallback_lock:
                yield

    def get_stats(self) -> Dict[str, Any]:
        """Get lock manager statistics."""
        return {
            **self._stats,
            'active_operations': len(self._active_operations),
            'node_locks_created': len(self._node_locks),
            'waiting_operations': sum(len(waiters) for waiters in self._lock_waiters.values())
        }

    def cleanup_unused_locks(self) -> int:
        """Clean up locks that are no longer referenced."""
        initial_count = len(self._node_locks)
        
        # Remove locks that are no longer in the weak reference dictionary
        expired_nodes = []
        for node_id in list(self._node_locks.keys()):
            if node_id not in self._lock_refs:
                expired_nodes.append(node_id)
        
        for node_id in expired_nodes:
            self._node_locks.pop(node_id, None)
            self._lock_waiters.pop(node_id, None)
        
        cleaned_count = initial_count - len(self._node_locks)
        if cleaned_count > 0:
            self.logger.debug(f"Cleaned up {cleaned_count} unused node locks")
        
        return cleaned_count

    async def force_release_operation(self, operation_id: str) -> bool:
        """Force release all locks held by a specific operation."""
        operation = self._active_operations.get(operation_id)
        if not operation:
            return False
            
        try:
            await self._release_operation_locks(operation)
            self._active_operations.pop(operation_id, None)
            return True
        except Exception as e:
            self.logger.error(f"Error force-releasing operation {operation_id}: {e}")
            return False