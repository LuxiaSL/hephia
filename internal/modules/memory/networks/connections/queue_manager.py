"""
queue_manager.py

Intelligent connection update queue system for memory networks.
Provides batched, concurrent, and future-proof connection processing
designed to handle dense networks over long-running periods.

Key capabilities:
- Intelligent batching by node pairs, timeframes, and operation types
- Background concurrent processing with semaphore control
- Memory-efficient queue management with overflow handling
- Comprehensive error handling and fallback mechanisms
- Detailed metrics and monitoring for performance optimization
- Clean shutdown and cancellation support
"""

import asyncio
import time
import hashlib
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Set, Optional, Any, Callable, Tuple, Union, TYPE_CHECKING
import weakref
from concurrent.futures import ThreadPoolExecutor

from loggers.loggers import MemoryLogger
from ...nodes.base_node import BaseMemoryNode

if TYPE_CHECKING:
    from .base_manager import BaseConnectionManager

class UpdatePriority(Enum):
    """Priority levels for connection updates."""
    CRITICAL = auto()    # Immediate node access or critical updates
    HIGH = auto()        # Formation of new nodes
    NORMAL = auto()      # Regular node access
    LOW = auto()         # Background maintenance
    BATCH = auto()       # Batched background processing


class UpdateReason(Enum):
    """Reasons for connection updates."""
    NODE_ACCESS = auto()     # Node was accessed
    NODE_FORMATION = auto()  # New node created
    NODE_UPDATE = auto()     # Node data changed
    MAINTENANCE = auto()     # Scheduled maintenance
    MANUAL = auto()          # Explicitly requested


@dataclass
class UpdateRequest:
    """Individual connection update request."""
    node_id: str
    node_ref: weakref.ref  # Weak reference to avoid memory leaks
    priority: UpdatePriority
    reason: UpdateReason
    timestamp: float
    connection_filter: Optional[Dict[str, float]] = None
    skip_reciprocal: bool = False
    lock_acquired: bool = False
    timeout: float = 30.0
    request_id: str = field(default_factory=lambda: hashlib.md5(str(time.time()).encode()).hexdigest()[:8])

    def get_node(self):
        """Safely get the referenced node."""
        return self.node_ref() if self.node_ref else None

    @property
    def age(self) -> float:
        """Age of this request in seconds."""
        return time.time() - self.timestamp


@dataclass
class UpdateBatch:
    """Batch of similar update requests for efficient processing."""
    batch_id: str
    requests: List[UpdateRequest]
    batch_type: str  # "node_pairs", "timeframe", "operation_type"
    created_timestamp: float
    estimated_processing_time: float = 1.0
    
    @property
    def size(self) -> int:
        return len(self.requests)
    
    @property
    def priority_score(self) -> float:
        """Calculate batch priority score for scheduling."""
        if not self.requests:
            return 0.0
        
        priority_weights = {
            UpdatePriority.CRITICAL: 1000,
            UpdatePriority.HIGH: 100,
            UpdatePriority.NORMAL: 10,
            UpdatePriority.LOW: 1,
            UpdatePriority.BATCH: 0.1
        }
        
        total_score = sum(priority_weights.get(req.priority, 1) for req in self.requests)
        age_bonus = min(100, max(req.age for req in self.requests) * 10)  # Age bonus
        
        return total_score + age_bonus


@dataclass
class QueueMetrics:
    """Metrics for monitoring queue performance."""
    total_requests: int = 0
    processed_requests: int = 0
    failed_requests: int = 0
    current_queue_size: int = 0
    average_batch_size: float = 0.0
    average_processing_time: float = 0.0
    batches_created: int = 0
    batches_processed: int = 0
    last_reset_time: float = field(default_factory=time.time)
    
    def reset(self):
        """Reset metrics for fresh measurement."""
        self.__init__()


class ConnectionUpdateQueue:
    """
    Intelligent queue system for batched connection updates.
    
    Designed for high-throughput, long-running memory networks with
    optimal batching, concurrency, and resource management.
    """
    
    def __init__(
        self,
        max_queue_size: int = 10000,
        max_batch_size: int = 50,
        batch_timeout: float = 2.0,
        max_concurrent_batches: int = 5,
        processing_interval: float = 0.1,
        enable_batching: bool = True
    ):
        """
        Initialize the connection update queue.
        
        Args:
            max_queue_size: Maximum number of queued requests
            max_batch_size: Maximum requests per batch
            batch_timeout: Maximum time to wait when building batches
            max_concurrent_batches: Maximum batches processing simultaneously
            processing_interval: How often to check for new batches
            enable_batching: Whether to enable intelligent batching
        """
        self.max_queue_size = max_queue_size
        self.max_batch_size = max_batch_size
        self.batch_timeout = batch_timeout
        self.max_concurrent_batches = max_concurrent_batches
        self.processing_interval = processing_interval
        self.enable_batching = enable_batching
        
        # Queue storage
        self._request_queue: deque = deque(maxlen=max_queue_size)
        self._priority_queues: Dict[UpdatePriority, deque] = {
            priority: deque() for priority in UpdatePriority
        }
        
        # Batch management
        self._pending_batches: List[UpdateBatch] = []
        self._processing_batches: Set[str] = set()
        self._batch_semaphore = asyncio.Semaphore(max_concurrent_batches)
        
        # State management
        self._running = False
        self._processor_task: Optional[asyncio.Task] = None
        self._shutdown_event = asyncio.Event()
        
        # Metrics and monitoring
        self.metrics = QueueMetrics()
        self._request_history: deque = deque(maxlen=1000)  # Keep last 1000 for analysis
        
        # Thread pool for CPU-bound operations
        self._thread_pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="connection_queue")
        
        # Deduplication tracking
        self._active_requests: Dict[str, float] = {}  # node_id -> timestamp
        self._recent_completions: Dict[str, float] = {}  # node_id -> completion_time
        self._completion_cleanup_interval = 300  # 5 minutes
        
        self.logger = MemoryLogger

    async def start(self):
        """Start the queue processing system."""
        if self._running:
            self.logger.warning("ConnectionUpdateQueue already running")
            return
            
        self._running = True
        self._shutdown_event.clear()
        
        # Start the main processing task
        self._processor_task = asyncio.create_task(self._processing_loop())
        
        self.logger.info("ConnectionUpdateQueue started")

    async def stop(self):
        """Stop the queue processing system gracefully."""
        if not self._running:
            return
            
        self.logger.info("Stopping ConnectionUpdateQueue...")
        self._running = False
        self._shutdown_event.set()
        
        # Wait for current processing to complete
        if self._processor_task:
            try:
                await asyncio.wait_for(self._processor_task, timeout=2.0)
            except asyncio.TimeoutError:
                self.logger.warning("Queue processor did not shut down gracefully, cancelling")
                self._processor_task.cancel()
                try:
                    await self._processor_task
                except asyncio.CancelledError:
                    pass
        
        # Shutdown thread pool
        self._thread_pool.shutdown(wait=True, cancel_futures=True)
        
        self.logger.info("ConnectionUpdateQueue stopped")

    async def enqueue_update(
        self,
        node: BaseMemoryNode,
        priority: UpdatePriority = UpdatePriority.NORMAL,
        reason: UpdateReason = UpdateReason.NODE_ACCESS,
        connection_filter: Optional[Dict[str, float]] = None,
        skip_reciprocal: bool = False,
        lock_acquired: bool = False,
        timeout: float = 30.0
    ) -> bool:
        """
        Enqueue a connection update request.
        
        Args:
            node: Node to update connections for
            priority: Priority level for processing
            reason: Reason for the update
            connection_filter: Optional subset of connections to update
            skip_reciprocal: Skip updating reciprocal connections
            timeout: Maximum time to spend on this update
            
        Returns:
            bool: True if successfully queued, False if rejected
        """
        if not self._running:
            self.logger.error("Cannot enqueue update - queue not running")
            return False
            
        if not node or not node.node_id:
            self.logger.error("Cannot enqueue update for invalid node")
            return False
        
        # Check for duplicate recent requests
        if self._is_duplicate_request(node.node_id, priority):
            self.logger.debug(f"Skipping duplicate request for node {node.node_id}")
            return True
        
        # Check queue capacity
        if len(self._request_queue) >= self.max_queue_size:
            self.logger.warning(f"Queue at capacity ({self.max_queue_size}), rejecting request")
            return False
        
        # Create update request
        request = UpdateRequest(
            node_id=str(node.node_id),
            node_ref=weakref.ref(node),
            priority=priority,
            reason=reason,
            timestamp=time.time(),
            connection_filter=connection_filter,
            skip_reciprocal=skip_reciprocal,
            lock_acquired=lock_acquired,
            timeout=timeout
        )
        
        # Add to appropriate queue
        self._priority_queues[priority].append(request)
        self._request_queue.append(request)
        self._active_requests[request.node_id] = request.timestamp
        
        self.metrics.total_requests += 1
        self.metrics.current_queue_size = len(self._request_queue)
        
        self.logger.debug(f"Enqueued {priority.name} update for node {node.node_id}")
        return True

    def _is_duplicate_request(self, node_id: str, priority: UpdatePriority) -> bool:
        """Check if this request is a duplicate of a recent one."""
        current_time = time.time()
        
        # Check if already actively processing
        if node_id in self._active_requests:
            last_request_time = self._active_requests[node_id]
            if current_time - last_request_time < 1.0:  # Within 1 second
                return True
        
        # Check recent completions (don't repeat too quickly)
        if node_id in self._recent_completions:
            completion_time = self._recent_completions[node_id]
            min_interval = 5.0 if priority in [UpdatePriority.LOW, UpdatePriority.BATCH] else 1.0
            if current_time - completion_time < min_interval:
                return True
                
        return False

    async def _processing_loop(self):
        """Main processing loop for the queue."""
        self.logger.info("Queue processing loop started")
        
        last_cleanup = time.time()
        
        try:
            while self._running:
                try:
                    # Process pending requests
                    await self._process_queue_cycle()
                    
                    # Periodic cleanup
                    current_time = time.time()
                    if current_time - last_cleanup > self._completion_cleanup_interval:
                        self._cleanup_completion_tracking()
                        last_cleanup = current_time
                    
                    # Wait before next cycle
                    await asyncio.sleep(self.processing_interval)
                    
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    self.logger.error(f"Error in processing loop: {e}")
                    await asyncio.sleep(1.0)  # Brief pause on error
                    
        except asyncio.CancelledError:
            pass
        finally:
            self.logger.info("Queue processing loop ended")

    async def _process_queue_cycle(self):
        """Process one cycle of queue operations."""
        if not self._request_queue:
            return
            
        # Create new batches if batching is enabled
        if self.enable_batching:
            await self._create_optimal_batches()
        else:
            await self._create_individual_batches()
        
        # Process pending batches
        await self._process_pending_batches()

    async def _create_optimal_batches(self):
        """Create optimally sized batches from pending requests."""
        if not self._request_queue:
            return
            
        # Group requests by optimization potential
        batch_candidates = await self._analyze_batching_opportunities()
        
        for batch_type, requests in batch_candidates.items():
            if len(requests) < 2:  # Need at least 2 for a batch
                # Convert to individual batches
                for request in requests:
                    await self._create_single_request_batch(request)
                continue
                
            # Create optimally sized batches
            while requests:
                batch_size = min(self.max_batch_size, len(requests))
                batch_requests = requests[:batch_size]
                requests = requests[batch_size:]
                
                batch = UpdateBatch(
                    batch_id=self._generate_batch_id(),
                    requests=batch_requests,
                    batch_type=batch_type,
                    created_timestamp=time.time(),
                    estimated_processing_time=self._estimate_batch_processing_time(batch_requests)
                )
                
                self._pending_batches.append(batch)
                self.metrics.batches_created += 1
                
                # Remove from queue
                for request in batch_requests:
                    try:
                        self._request_queue.remove(request)
                        self._priority_queues[request.priority].remove(request)
                    except ValueError:
                        pass  # Already removed

    async def _analyze_batching_opportunities(self) -> Dict[str, List[UpdateRequest]]:
        """Analyze current queue for optimal batching opportunities."""
        # Convert to list for analysis (preserving queue order)
        requests = list(self._request_queue)
        
        batching_groups = {
            "high_priority": [],
            "node_pairs": [],
            "recent_timeframe": [],
            "maintenance": []
        }
        
        current_time = time.time()
        
        for request in requests:
            # High priority requests get expedited processing
            if request.priority in [UpdatePriority.CRITICAL, UpdatePriority.HIGH]:
                batching_groups["high_priority"].append(request)
            # Recent requests can be batched for temporal efficiency
            elif request.age < 60.0:  # Within last minute
                batching_groups["recent_timeframe"].append(request)
            # Maintenance requests can be heavily batched
            elif request.reason == UpdateReason.MAINTENANCE:
                batching_groups["maintenance"].append(request)
            # Default to node pairs optimization
            else:
                batching_groups["node_pairs"].append(request)
        
        return batching_groups

    async def _create_individual_batches(self):
        """Create individual batches when batching is disabled."""
        processed_count = 0
        max_individual_batches = min(50, len(self._request_queue))  # Limit individual processing
        
        while self._request_queue and processed_count < max_individual_batches:
            request = self._request_queue.popleft()
            
            # Remove from priority queue as well
            try:
                self._priority_queues[request.priority].remove(request)
            except ValueError:
                pass
                
            await self._create_single_request_batch(request)
            processed_count += 1

    async def _create_single_request_batch(self, request: UpdateRequest):
        """Create a batch containing a single request."""
        batch = UpdateBatch(
            batch_id=self._generate_batch_id(),
            requests=[request],
            batch_type="individual",
            created_timestamp=time.time(),
            estimated_processing_time=1.0
        )
        
        self._pending_batches.append(batch)
        self.metrics.batches_created += 1

    async def _process_pending_batches(self):
        """Process all pending batches concurrently."""
        if not self._pending_batches:
            return
            
        # Sort batches by priority
        self._pending_batches.sort(key=lambda b: b.priority_score, reverse=True)
        
        # Process as many batches as semaphore allows
        processing_tasks = []
        batches_to_process = []
        
        while (self._pending_batches and 
               len(processing_tasks) < self.max_concurrent_batches):
            batch = self._pending_batches.pop(0)
            if batch.batch_id not in self._processing_batches:
                batches_to_process.append(batch)
                processing_tasks.append(
                    asyncio.create_task(self._process_batch_safe(batch))
                )
        
        if processing_tasks:
            await asyncio.gather(*processing_tasks, return_exceptions=True)

    async def _process_batch_safe(self, batch: UpdateBatch):
        """Safely process a batch with comprehensive error handling."""
        async with self._batch_semaphore:
            self._processing_batches.add(batch.batch_id)
            start_time = time.time()
            
            try:
                self.logger.debug(f"Processing batch {batch.batch_id} with {batch.size} requests")
                
                # Process the batch through the connection manager
                results = await self._process_batch_requests(batch)
                
                # Update metrics
                processing_time = time.time() - start_time
                self.metrics.batches_processed += 1
                self.metrics.processed_requests += len([r for r in results if r.get('success', False)])
                self.metrics.failed_requests += len([r for r in results if not r.get('success', False)])
                
                # Update average processing time
                if self.metrics.batches_processed > 0:
                    self.metrics.average_processing_time = (
                        (self.metrics.average_processing_time * (self.metrics.batches_processed - 1) + processing_time) /
                        self.metrics.batches_processed
                    )
                
                self.logger.debug(f"Completed batch {batch.batch_id} in {processing_time:.2f}s")
                
            except Exception as e:
                self.logger.error(f"Error processing batch {batch.batch_id}: {e}")
                self.metrics.failed_requests += len(batch.requests)
                
            finally:
                self._processing_batches.discard(batch.batch_id)
                
                # Mark requests as completed
                for request in batch.requests:
                    self._recent_completions[request.node_id] = time.time()
                    self._active_requests.pop(request.node_id, None)

    async def _process_batch_requests(self, batch: UpdateBatch) -> List[Dict[str, Any]]:
        """
        Process actual batch requests through the connection manager.
        """
        if not hasattr(self, '_connection_manager') or not self._connection_manager:
            self.logger.error("No connection manager set for queue processing")
            return [{'request_id': req.request_id, 'success': False, 'error': 'No connection manager'} 
                    for req in batch.requests]
        
        # Use the connection manager's batch processing method
        if hasattr(self._connection_manager, 'process_queued_batch_with_locks'):
            return await self._connection_manager.process_queued_batch_with_locks(batch)
        else:
            # Fallback error handling (should never happen with current implementation)
            self.logger.error("Connection manager missing batch processing method")
            return [{'request_id': req.request_id, 'success': False, 'error': 'Batch processing not supported'}
                    for req in batch.requests]

    def set_connection_manager(self, connection_manager: 'BaseConnectionManager'):
        """Set the connection manager reference for batch processing."""
        self._connection_manager = connection_manager

    def _generate_batch_id(self) -> str:
        """Generate unique batch ID."""
        return f"batch_{int(time.time() * 1000)%1000000}_{len(self._pending_batches)}"

    def _estimate_batch_processing_time(self, requests: List[UpdateRequest]) -> float:
        """Estimate processing time for a batch."""
        base_time = 0.5  # Base processing time
        complexity_factor = len(requests) * 0.1  # Each request adds complexity
        return base_time + complexity_factor

    def _cleanup_completion_tracking(self):
        """Clean up old completion tracking data."""
        current_time = time.time()
        cutoff_time = current_time - self._completion_cleanup_interval
        
        # Clean up recent completions
        expired_completions = [
            node_id for node_id, completion_time in self._recent_completions.items()
            if completion_time < cutoff_time
        ]
        for node_id in expired_completions:
            self._recent_completions.pop(node_id, None)
        
        # Clean up active requests that are too old
        expired_active = [
            node_id for node_id, request_time in self._active_requests.items()
            if request_time < cutoff_time
        ]
        for node_id in expired_active:
            self._active_requests.pop(node_id, None)

    def get_queue_stats(self) -> Dict[str, Any]:
        """Get current queue statistics."""
        return {
            'queue_size': len(self._request_queue),
            'pending_batches': len(self._pending_batches),
            'processing_batches': len(self._processing_batches),
            'total_requests': self.metrics.total_requests,
            'processed_requests': self.metrics.processed_requests,
            'failed_requests': self.metrics.failed_requests,
            'success_rate': (
                self.metrics.processed_requests / max(1, self.metrics.total_requests) * 100
            ),
            'average_batch_size': (
                self.metrics.processed_requests / max(1, self.metrics.batches_processed)
            ),
            'average_processing_time': self.metrics.average_processing_time,
            'is_running': self._running
        }

    async def __aenter__(self):
        """Async context manager entry."""
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.stop()