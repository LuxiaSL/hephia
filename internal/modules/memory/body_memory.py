"""
body_memory.py

Implements the BodyMemory system for managing raw emotional/physical experiences as an
interconnected network of memory nodes. Handles formation, storage, decay dynamics and
interaction patterns of body-based memory nodes.

Key capabilities:
- Creates & stores memory nodes capturing raw emotional/physical experiences
- Maintains connections between related body memory experiences 
- Manages memory strength/decay with ghosting mechanics for weak memories
- Resolves state conflicts via memory merges
- Provides similarity-based node retrieval and graph traversal

Code organization:
    1) Node Definition & Dataclass
    2) Core Memory Manager Setup
    3) Memory Formation & Connections
    4) State Similarity & Comparisons 
    5) Node Retrieval & Queries
    6) Ghost & Decay Handling
    7) Database Operations & Persistence
    8) Utility Functions & Calculations

The system uses SQLite for persistence and implements configurable decay/ghosting
behaviors controlled by thresholds. Memory nodes can be queried by time window,
state similarity, or connection traversal.
"""

from typing import Dict, List, Any, Optional, Set, Tuple, Union
from collections import defaultdict
from dataclasses import dataclass, field
import time
import json
import math
import sqlite3
from loggers.loggers import MemoryLogger
from contextlib import contextmanager
from event_dispatcher import Event, global_event_dispatcher

class MemorySystemError(Exception):
    """Base exception for memory system errors"""
    pass

class DatabaseError(MemorySystemError):
    """Database operation errors"""
    pass

class NodeNotFoundError(MemorySystemError):
    """Node lookup/access errors"""
    pass

# -------------------------------------------------------------------------
# 1) Node Definition & Dataclass
# -------------------------------------------------------------------------
@dataclass
class BodyMemoryNode:
    """
    A single body memory experience with complete state information.
    Contains both raw and processed state data for complete recall and interaction.
    """
    timestamp: float
    raw_state: Dict[str, Any]
    processed_state: Dict[str, Any]
    strength: float
    node_id: Optional[str] = None
    ghosted: bool = False
    parent_node_id: Optional[str] = None
    ghost_nodes: List[Dict] = field(default_factory=list)
    ghost_states: List[Dict] = field(default_factory=list)
    connections: Dict[str, float] = field(default_factory=dict)
    last_connection_update: Optional[float] = None
    
    def __post_init__(self):
        """Initialize optional fields if not provided"""
        if self.ghost_nodes is None:
            self.ghost_nodes = []
        if self.ghost_states is None:
            self.ghost_states = []
        if self.connections is None:
            self.connections = {}

    def to_dict(self) -> Dict[str, Any]:
        """Convert node to database-friendly format"""
        return {
            'timestamp': self.timestamp,
            'raw_state': json.dumps(self.raw_state),
            'processed_state': json.dumps(self.processed_state),
            'strength': self.strength,
            'ghosted': self.ghosted,
            'parent_node_id': self.parent_node_id,
            'ghost_nodes': json.dumps(self.ghost_nodes),
            'ghost_states': json.dumps(self.ghost_states),
            'connections': json.dumps(self.connections),
            'last_connection_update': self.last_connection_update
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BodyMemoryNode':
        """Create node from database format"""
        try:
            return cls(
                timestamp=data['timestamp'],
                raw_state=json.loads(data['raw_state']),
                processed_state=json.loads(data['processed_state']),
                strength=data['strength'],
                node_id=data.get('id'),
                ghosted=data.get('ghosted', False),
                parent_node_id=data.get('parent_node_id'),
                ghost_nodes=json.loads(data['ghost_nodes']),
                ghost_states=json.loads(data['ghost_states']),
                connections=json.loads(data['connections']),
                last_connection_update=data.get('last_connection_update')
            )
        except (json.JSONDecodeError, KeyError) as e:
            raise MemorySystemError(f"Failed to create node from data: {e}")

    def merge_into_parent(self, parent: 'BodyMemoryNode') -> None:
        """
        Convert this node into a ghost node of the parent.
        Redirects all connections to the parent node.
        """
        self.ghosted = True
        self.parent_node_id = parent.node_id
        
        # Record this node as a ghost in the parent
        parent.ghost_nodes.append({
            'node_id': self.node_id,
            'timestamp': self.timestamp,
            'raw_state': self.raw_state,
            'processed_state': self.processed_state,
            'strength': self.strength,
            'ghost_nodes': self.ghost_nodes  # Preserve any nested ghosts
        })
        
        # Transfer any remaining strength to parent
        parent.strength = min(1.0, parent.strength + (self.strength * 0.3))
        
        # Merge connections - connections to this node should now point to parent
        for conn_id, weight in self.connections.items():
            if conn_id in parent.connections:
                # If connection already exists, strengthen it
                parent.connections[conn_id] = min(1.0, parent.connections[conn_id] + (weight * 0.5))
            else:
                # Transfer connection with reduced weight
                parent.connections[conn_id] = weight * 0.85
        
        self.update_connections(parent)

    def decay(self, rate: float, min_strength: float, min_ghost_strength: float) -> str:
        """
        Apply multi-stage decay to node strength.
        
        Args:
            rate: Base decay rate
            min_strength: Threshold for initial ghosting
            min_ghost_strength: Threshold for ghost state conversion
        
        Returns:
            str: Current decay state ('active', 'ghost', 'final_prune')
        """
        # Apply base decay
        self.strength *= (1.0 - rate)
        
        if self.ghosted:
            # Ghosted nodes decay faster
            self.strength *= (1.0 - rate * 0.85)
            return 'final_prune' if self.strength < min_ghost_strength else 'ghost'
        
        return 'ghost' if self.strength < min_strength else 'active'


# -------------------------------------------------------------------------
# 2) Core Memory Manager Setup
# -------------------------------------------------------------------------
class BodyMemory:
    """
    Stores and manages unmediated bodily/emotional experiences.
    
    Maintains a network of body memory nodes that can interact, decay,
    and merge over time. Each node contains complete state information
    to enable rich interaction patterns and realistic recall.
    """
    
    def __init__(self, internal_context, db_path: str = 'data/memory.db'):
        self.nodes: List[BodyMemoryNode] = []
        self.context = internal_context
        self.db_path = db_path
        self.min_active_nodes = 5
        self._init_database()
        self._load_nodes()
        self.logger = MemoryLogger

        self.setup_event_listeners()

    @contextmanager
    def _db_connection(self):
        """Context manager for database connections"""
        conn = None
        try:
            conn = sqlite3.connect(self.db_path)
            yield conn
        except sqlite3.Error as e:
            raise DatabaseError(f"Database operation failed: {e}")
        finally:
            if conn:
                conn.close()

    def _init_database(self) -> None:
        """Initialize the body memory database"""
        with self._db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS body_memory_nodes (
                    id INTEGER PRIMARY KEY,
                    timestamp REAL,
                    raw_state TEXT,
                    processed_state TEXT,
                    strength REAL,
                    ghosted BOOLEAN,
                    parent_node_id INTEGER,
                    ghost_nodes TEXT,
                    ghost_states TEXT,
                    connections TEXT,
                    last_connection_update REAL,
                    FOREIGN KEY(parent_node_id) REFERENCES body_memory_nodes(id)
                )
            """)
            conn.commit()

    def _load_nodes(self) -> None:
        """Load all nodes from database on startup"""
        with self._db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM body_memory_nodes")
            column_names = [description[0] for description in cursor.description]
            
            for row in cursor.fetchall():
                data = dict(zip(column_names, row))
                try:
                    node = BodyMemoryNode.from_dict(data)
                    self.nodes.append(node)
                except MemorySystemError as e:
                    self.logger.log_error(f"Failed to load node: {e}")
                    continue

    def setup_event_listeners(self) -> None:
        global_event_dispatcher.add_listener("emotion:finished", self.process_memory_formation)
        global_event_dispatcher.add_listener("behavior:changed", self.process_memory_formation)
        global_event_dispatcher.add_listener("mood:changed", self.process_memory_formation)
        global_event_dispatcher.add_listener("need:changed", self.process_memory_formation)

    def process_memory_formation(self, event: Event):
        """
        Process incoming events to form body memory nodes.
        For alpha: emotion/behavior/mood events always form memories;
        need changes form memories if >5 difference.
        """
        try:
            event_type = event.event_type.split(':')[0]
            
            metadata = {
                'source_event': event.event_type,
                'timestamp': time.time()
            }

            # Extract relevant event data
            if event_type == 'emotion':
                # Extract emotion vector data from event
                emotion = event.data.get('emotion')
                if emotion:
                    metadata.update({
                        'emotion_type': emotion.name,
                        'emotion_intensity': emotion.intensity,
                        'emotion_valence': emotion.valence,
                        'emotion_arousal': emotion.arousal,
                        'emotion_source': emotion.source_type,
                        'emotion_source_data': emotion.source_data
                    })
                    self.form_memory(metadata)
                return
                
            elif event_type in ['behavior', 'mood']:
                # Save behavior/mood changes
                metadata.update({
                    'change_type': event_type,
                    'old_state': event.data.get('old_state'),
                    'new_state': event.data.get('new_state')
                })
                self.form_memory(metadata)
                return
                
            elif event_type == 'need':
                # Only form memory if significant need change
                old_val = event.data.get('old_value', 50)
                new_val = event.data.get('new_value', 50)
                if abs(new_val - old_val) >= 5:
                    metadata.update({
                        'need_type': event.data.get('need_type'),
                        'old_value': old_val,
                        'new_value': new_val,
                        'change_magnitude': abs(new_val - old_val)
                    })
                    self.form_memory(metadata)
                return
                    
        except Exception as e:
            self.logger.log_error(f"Failed to process memory formation event: {e}")

    # -------------------------------------------------------------------------
    # 3) Memory Formation & Connections
    # -------------------------------------------------------------------------
    def form_memory(self, metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Create a new memory node using internal context state and optional metadata.
        
        Args:
            metadata: Optional metadata about the memory formation event/trigger
                     Can include source event info or direct formation data

        Returns:
            str: ID of created node
        """
        try:
            # Get current state from internal context
            # don't store cognitive context in body memory
            memory_context = self.context.get_memory_context()
            if not memory_context:
                raise MemorySystemError("Failed to get memory context")
                
            # Extract states from context
            raw_state = memory_context.get('raw_state', {})
            processed_state = memory_context.get('processed_state', {})
            
            # Add metadata if provided
            if metadata:
                raw_state['formation_metadata'] = metadata
            
            # Calculate initial node strength (use cognitive context retrieval here)
            initial_strength = self._calculate_initial_strength(self.context.get_memory_context(get_cognitive=True))
            
            # Create node
            node = BodyMemoryNode(
                timestamp=time.time(),
                raw_state=raw_state,
                processed_state=processed_state,
                strength=initial_strength
            )
            

             # Add to memory and persist
            self.nodes.append(node)
            self._persist_node(node)

            # initialize connections
            self._form_initial_connections(node)
            
            # Dispatch creation event
            global_event_dispatcher.dispatch_event(Event(
                "memory:node_created",
                {
                    "node_type": "body",
                    "node": node,
                    "metadata": metadata
                }
            ))

            return str(node.node_id)

        except Exception as e:
            self.logger.log_error(f"Failed to form memory: {e}")
            raise MemorySystemError(f"Memory formation failed: {e}")

    def _form_initial_connections(self, node: BodyMemoryNode, max_candidates: int = 50) -> None:
        """
        Create initial connections between new node and existing nodes based on
        state similarity and temporal proximity. Uses softmax scaling for weights
        unless extremely similar.
        
        Args:
            node: New node to connect
            max_candidates: Maximum number of recent nodes to consider for initial analysis
        """
        if not node:
            raise ValueError("Invalid node provided")
                
        try:
            # Get potential candidates, prioritizing recent nodes but considering more than before
            candidates = sorted(
                [n for n in self.nodes if not n.ghosted and n.node_id != node.node_id],
                key=lambda x: x.timestamp,
                reverse=True
            )[:max_candidates]
            
            if not candidates:
                return
                
            # Calculate similarity scores for each candidate
            connection_scores = []
            
            for other in candidates:
                try:
                    # Calculate base similarity using existing method
                    similarity = self._calculate_connection_weight(node, other)
                    
                    # Calculate temporal bonus (exponential decay)
                    time_diff = abs(node.timestamp - other.timestamp)
                    temporal_bonus = math.exp(-time_diff / 3600)  # 1-hour decay
                    
                    # Combine scores with temporal emphasis for initial formation
                    final_score = (similarity * 0.7) + (temporal_bonus * 0.3)
                    
                    if final_score > 0.2:  # Minimum threshold
                        connection_scores.append((other, final_score))
                        
                except Exception as e:
                    self.logger.warning(
                        f"Failed to calculate connection for nodes {node.node_id} and {other.node_id}: {e}"
                    )
                    continue
            
            if not connection_scores:
                return
                
            # Sort by score
            connection_scores.sort(key=lambda x: x[1], reverse=True)
            
            # Apply softmax scaling unless extremely similar
            scores = [score for _, score in connection_scores]
            softmax_scores = self._softmax(scores)
            
            # Create connections
            for i, (other_node, raw_score) in enumerate(connection_scores):
                # If extremely similar (>0.9), keep raw score
                # Otherwise use softmax scaling
                weight = raw_score if raw_score > 0.9 else softmax_scores[i]
                
                # Create bidirectional connections
                node.connections[other_node.node_id] = weight
                other_node.connections[node.node_id] = weight
                self._update_node(other_node)
                    
        except Exception as e:
            self.logger.log_error("Failed to create initial connections")
            raise MemorySystemError(f"Initial connection creation failed: {e}")

    def _softmax(self, scores: List[float]) -> List[float]:
        """Calculate softmax of input scores."""
        if not scores:
            return []
            
        exp_scores = [math.exp(s) for s in scores]
        sum_exp = sum(exp_scores)
        return [e / sum_exp for e in exp_scores]

    # -------------------------------------------------------------------------
    # 4) State Similarity & Comparisons
    # -------------------------------------------------------------------------
    def find_similar_nodes(self, state: Dict[str, Any], threshold: float = 0.7, max_results: int = 10) -> List[BodyMemoryNode]:
        """
        Find nodes similar to given state based on our similarity metrics.
        
        Args:
            state: State to compare against (can be either raw state dict or full memory context)
            threshold: Minimum similarity score [0,1]
            max_results: Maximum number of results to return
            
        Returns:
            List[BodyMemoryNode]: Similar nodes sorted by similarity
        """
        if not state:
            raise ValueError("Invalid state provided")
            
        try:
            # Handle both raw state dicts and full memory contexts
            comparison_state = {}
            if 'raw_state' in state and 'processed_state' in state:
                comparison_state = state
            else:
                # Convert raw state dict to memory context format
                comparison_state = {
                    'raw_state': {
                        'emotional_state': state.get('emotional_state', []),
                        'needs': state.get('needs', {}),
                        'behavior': state.get('behavior', {}),
                        'mood': state.get('mood', {})
                    },
                    'processed_state': {
                        k: v for k, v in state.items()
                        if k not in ['emotional_state', 'needs', 'behavior', 'mood']
                    }
                }
            
            active_nodes = [n for n in self.nodes if not n.ghosted]
            similarities = []
            
            for node in active_nodes:
                try:
                    similarity = self._calculate_state_similarity(
                        comparison_state,
                        {'raw_state': node.raw_state, 'processed_state': node.processed_state}
                    )
                    if similarity >= threshold:
                        similarities.append((node, similarity))
                except Exception as e:
                    self.logger.warning(f"Failed to calculate similarity for node {node.node_id}: {e}")
                    continue
            
            similarities.sort(key=lambda x: x[1], reverse=True)
            return [node for node, _ in similarities[:max_results]]
            
        except Exception as e:
            self.logger.log_error(f"Failed to find similar nodes: {e}")
            raise MemorySystemError(f"Similarity search failed: {e}")

    def _calculate_state_similarity(self, state1: Dict[str, Dict], state2: Dict[str, Dict]) -> float:
        """
        Calculate overall similarity between two states.
        
        Args:
            state1, state2: States to compare, each with raw_state and processed_state
                
        Returns:
            float: Similarity score [0,1] 
        """
        similarity = 0.0
        components = 0
        
        raw1, raw2 = state1['raw_state'], state2['raw_state']
        proc1, proc2 = state1['processed_state'], state2['processed_state']
        
        # Most important: Emotional & Mood States
        # Emotional state comparison (from EmotionalProcessor)
        if ('emotional_state' in raw1 and 'emotional_state' in raw2 and 
            raw1['emotional_state'] and raw2['emotional_state']):
            emotional_sim = self._calculate_emotional_similarity(
                raw1['emotional_state'],
                raw2['emotional_state']
            )
            similarity += emotional_sim * 0.35
            components += 0.35

        # Mood comparison (from MoodSynthesizer)
        if ('mood' in raw1 and 'mood' in raw2 and
            isinstance(raw1['mood'], dict) and isinstance(raw2['mood'], dict)):
            mood_sim = self._calculate_mood_similarity(raw1['mood'], raw2['mood'])
            similarity += mood_sim * 0.25 
            components += 0.25
            
        # Behavior state (from BehaviorManager)
        if ('behavior' in raw1 and 'behavior' in raw2 and
            isinstance(raw1['behavior'], dict) and isinstance(raw2['behavior'], dict)):
            behavior1 = raw1['behavior'].get('name')
            behavior2 = raw2['behavior'].get('name')
            behavior_sim = 1.0 if behavior1 and behavior1 == behavior2 else 0.0
            similarity += behavior_sim * 0.2
            components += 0.2
            
        # Needs similarity (from NeedsManager) 
        if ('needs' in raw1 and 'needs' in raw2 and
            isinstance(raw1['needs'], dict) and isinstance(raw2['needs'], dict)):
            needs_sim = self._calculate_needs_similarity(
                {k: v.get('satisfaction', 0.5) for k,v in raw1['needs'].items()},
                {k: v.get('satisfaction', 0.5) for k,v in raw2['needs'].items()}
            )
            similarity += needs_sim * 0.2
            components += 0.2
        
        # Normalize by components present
        return similarity / components if components > 0 else 0.0

    # -------------------------------------------------------------------------
    # 5) Node Retrieval & Queries
    # -------------------------------------------------------------------------
    def traverse_connections(
        self, 
        start_node: BodyMemoryNode, 
        max_depth: int = 3,
        min_weight: float = 0.3,
        visited: Set[str] = None
    ) -> Dict[str, List[Tuple[BodyMemoryNode, float, int]]]:
        """
        Follow connection paths from a starting node, returning connected nodes
        grouped by their distance from the start.
        
        Args:
            start_node: Node to start traversal from
            max_depth: Maximum connection distance to traverse
            min_weight: Minimum connection weight to follow
            visited: Set of visited node IDs (for recursion)
            
        Returns:
            Dict mapping depths to lists of (node, connection_weight, path_length) tuples
        """
        if not start_node or not start_node.node_id:
            raise ValueError("Invalid start node provided")
            
        try:
            if visited is None:
                visited = set()
            
            results = defaultdict(list)
            if start_node.node_id in visited or max_depth <= 0:
                return results
                
            visited.add(start_node.node_id)
            
            direct_connections = [
                (nid, weight) for nid, weight in start_node.connections.items()
                if weight >= min_weight and nid not in visited
            ]
            
            for nid, weight in direct_connections:
                node = self._get_node_by_id(nid)
                if node and not node.ghosted:
                    results[1].append((node, weight, 1))
            
            if max_depth > 1:
                for node, weight, _ in results[1]:
                    try:
                        deeper = self.traverse_connections(
                            node, 
                            max_depth - 1,
                            min_weight,
                            visited
                        )
                        for depth, nodes in deeper.items():
                            results[depth + 1].extend(nodes)
                    except Exception as e:
                        self.logger.warning(f"Failed to traverse deeper for node {node.node_id}: {e}")
                        continue
            
            return results
            
        except Exception as e:
            self.logger.log_error(f"Failed to traverse connections from node {start_node.node_id}: {e}")
            raise MemorySystemError(f"Connection traversal failed: {e}")

    def query_by_time_window(
        self, 
        start_time: float,
        end_time: float,
        include_ghosted: bool = False
    ) -> List[BodyMemoryNode]:
        """
        Get nodes within a specific time window.
        
        Args:
            start_time: Start of time window
            end_time: End of time window
            include_ghosted: Whether to include ghosted nodes
            
        Returns:
            List[BodyMemoryNode]: Nodes within time window
        """
        if start_time > end_time:
            raise ValueError("Invalid time window")
            
        try:
            nodes = self.nodes if include_ghosted else [n for n in self.nodes if not n.ghosted]
            return [
                node for node in nodes
                if start_time <= node.timestamp <= end_time
            ]
        except Exception as e:
            self.logger.log_error(f"Failed to query time window: {e}")
            raise MemorySystemError(f"Time window query failed: {e}")

    def get_recent_memories(
        self,
        count: int = 5,
        include_ghosted: bool = False,
        with_metrics: bool = False,
        time_window: Optional[float] = None
    ) -> Union[List[BodyMemoryNode], Tuple[List[BodyMemoryNode], List[Dict[str, Any]]]]:
        """
        Retrieve the most recent body memory nodes, optionally within a time window.
        Can return detailed metrics about the recent memories' emotional/physical significance.
        
        Args:
            count: Number of recent memories to retrieve
            include_ghosted: Whether to include ghosted memories
            with_metrics: Return detailed metrics about each memory 
            time_window: Optional time window in seconds to limit search
            
        Returns:
            Either List[BodyMemoryNode] or Tuple of (nodes, metrics) if with_metrics=True
        """
        try:
            # Filter nodes based on criteria
            current_time = time.time()
            filtered_nodes = [
                n for n in self.nodes
                if (not n.ghosted or include_ghosted) and
                (not time_window or (current_time - n.timestamp) <= time_window)
            ]
            
            # Sort by timestamp
            recent_nodes = sorted(
                filtered_nodes,
                key=lambda x: x.timestamp,
                reverse=True
            )[:count]
            
            if with_metrics:
                # Get current state for metric calculation
                current_state = self.context.get_memory_context(get_cognitive=False)
                
                # Calculate metrics for each node
                node_metrics = []
                for node in recent_nodes:
                    metrics = {
                        'emotional_relevance': self._calculate_emotional_similarity(
                            current_state['raw_state'].get('emotional_vectors', []),
                            node.raw_state.get('emotional_vectors', [])
                        ),
                        'mood_alignment': self._calculate_mood_similarity(
                            current_state['raw_state'].get('mood', {}),
                            node.raw_state.get('mood', {})
                        ),
                        'behavior_match': 1.0 if (
                            current_state['raw_state'].get('behavior', {}).get('name') ==
                            node.raw_state.get('behavior', {}).get('name')
                        ) else 0.0,
                        'needs_similarity': self._calculate_needs_similarity(
                            current_state['raw_state'].get('needs', {}),
                            node.raw_state.get('needs', {})
                        ),
                        'strength': node.strength,
                        'age_factor': math.exp(-(current_time - node.timestamp) / 3600)  # 1-hour decay
                    }
                    
                    # Calculate overall significance
                    metrics['overall_significance'] = (
                        metrics['emotional_relevance'] * 0.35 +
                        metrics['mood_alignment'] * 0.25 +
                        metrics['behavior_match'] * 0.2 +
                        metrics['needs_similarity'] * 0.2
                    ) * metrics['strength'] * metrics['age_factor']
                    
                    node_metrics.append(metrics)
                    
                return recent_nodes, node_metrics
                
            return recent_nodes
            
        except Exception as e:
            self.logger.log_error(f"Failed to retrieve recent body memories: {e}")
            return [] if not with_metrics else ([], [])

    def _get_node_by_id(self, node_id: str) -> Optional[BodyMemoryNode]:
        """Safe node lookup by ID"""
        return next((n for n in self.nodes if n.node_id == node_id), None)

    # -------------------------------------------------------------------------
    # 6) Ghost & Decay Handling
    # -------------------------------------------------------------------------
    def update(self, decay_rate: float = 0.01) -> None:
        """
        Update memory nodes - apply decay and handle weak nodes.
        
        Args:
            decay_rate: Base rate of decay per update
        """
        try:
            active_nodes = [n for n in self.nodes if not n.ghosted]
            
            if len(active_nodes) <= self.min_active_nodes:
                return
                
            ghost_candidates = []
            state_conversion_candidates = []
            
            for node in self.nodes:
                try:
                    decay_state = node.decay(
                        rate=decay_rate,
                        min_strength=0.1,
                        min_ghost_strength=0.05
                    )
                    
                    if decay_state == 'ghost':
                        ghost_candidates.append(node)
                    elif decay_state == 'final_prune':
                        state_conversion_candidates.append(node)
                except Exception as e:
                    self.logger.warning(f"Failed to decay node {node.node_id}: {e}")
                    continue
            
            for node in ghost_candidates:
                try:
                    self._handle_weak_node(node)
                except Exception as e:
                    self.logger.warning(f"Failed to handle weak node {node.node_id}: {e}")
                    
            for node in state_conversion_candidates:
                try:
                    self._convert_to_ghost_state(node)
                except Exception as e:
                    self.logger.warning(f"Failed to convert node {node.node_id} to ghost state: {e}")
                
        except Exception as e:
            self.logger.log_error(f"Failed to update memory system: {e}")
            raise MemorySystemError(f"Memory system update failed: {e}")

    def _handle_weak_node(self, node: BodyMemoryNode) -> None:
        """
        Handle a node that has decayed below threshold.
        Either merge it into a connected node or convert to ghost state.
        """
        if not node or not node.node_id:
            raise ValueError("Invalid node provided")
            
        try:
            if node.ghosted:
                if node.strength < 0.05:
                    self._convert_to_ghost_state(node)
                return

            merge_candidates = [
                (other, weight) 
                for other in self.nodes 
                if not other.ghosted and other.node_id 
                for nid, weight in node.connections.items()
                if other.node_id == nid and weight > 0.7
            ]
            
            if merge_candidates:
                target, _ = max(merge_candidates, key=lambda x: x[1])
                node.merge_into_parent(target)
                self._update_node(target)
            else:
                self._convert_to_ghost_state(node)
            
        except Exception as e:
            self.logger.log_error(f"Failed to handle weak node {node.node_id}: {e}")
            raise MemorySystemError(f"Weak node handling failed: {e}")

    def _convert_to_ghost_state(self, node: BodyMemoryNode) -> None:
        """
        Convert a ghost node's data into a ghost state entry.
        After conversion, the node can be removed from active tracking.
        """
        if not node or not node.node_id:
            raise ValueError("Invalid node provided")
            
        try:
            if node.parent_node_id:
                parent = self._get_node_by_id(node.parent_node_id)
                if parent:
                    ghost_state = {
                        'timestamp': node.timestamp,
                        'raw_state': node.raw_state,
                        'processed_state': node.processed_state,
                        'strength': node.strength,
                        'derived_from_node': node.node_id
                    }
                    parent.ghost_states.append(ghost_state)
                    self._update_node(parent)
            
            self.nodes.remove(node)
            self._delete_node(node)
            
        except Exception as e:
            self.logger.log_error(f"Failed to convert node {node.node_id} to ghost state: {e}")
            raise MemorySystemError(f"Ghost state conversion failed: {e}")

    # -------------------------------------------------------------------------
    # 7) Database Operations & Persistence
    # -------------------------------------------------------------------------
    def _persist_node(self, node: BodyMemoryNode) -> None:
        """Save new node to database with error handling"""
        with self._db_connection() as conn:
            cursor = conn.cursor()
            data = node.to_dict()
            
            try:
                cursor.execute("""
                    INSERT INTO body_memory_nodes 
                    (timestamp, raw_state, processed_state, strength, ghosted,
                     parent_node_id, ghost_nodes, ghost_states, connections,
                     last_connection_update)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    data['timestamp'],
                    data['raw_state'],
                    data['processed_state'],
                    data['strength'],
                    data['ghosted'],
                    data['parent_node_id'],
                    data['ghost_nodes'],
                    data['ghost_states'],
                    data['connections'],
                    data['last_connection_update']
                ))
                
                node.node_id = cursor.lastrowid
                conn.commit()
            except sqlite3.Error as e:
                self.logger.log_error(f"Failed to persist node: {e}")
                raise DatabaseError(f"Failed to save node: {e}")

    def _update_node(self, node: BodyMemoryNode) -> None:
        """Update existing node in database"""
        if not node or not node.node_id:
            raise ValueError("Invalid node provided")
            
        with self._db_connection() as conn:
            cursor = conn.cursor()
            data = node.to_dict()
            
            try:
                cursor.execute("""
                    UPDATE body_memory_nodes 
                    SET timestamp=?, raw_state=?, processed_state=?, strength=?,
                        ghosted=?, parent_node_id=?, ghost_nodes=?, ghost_states=?,
                        connections=?, last_connection_update=?
                    WHERE id=?
                """, (
                    data['timestamp'],
                    data['raw_state'],
                    data['processed_state'],
                    data['strength'],
                    data['ghosted'],
                    data['parent_node_id'],
                    data['ghost_nodes'],
                    data['ghost_states'],
                    data['connections'],
                    data['last_connection_update'],
                    node.node_id
                ))
                
                if cursor.rowcount == 0:
                    raise NodeNotFoundError(f"Node {node.node_id} not found in database")
                    
                conn.commit()
            except sqlite3.Error as e:
                self.logger.log_error(f"Failed to update node {node.node_id}: {e}")
                raise DatabaseError(f"Failed to update node: {e}")

    def _delete_node(self, node: BodyMemoryNode) -> None:
        """Remove node from database"""
        if not node or not node.node_id:
            raise ValueError("Invalid node provided")
            
        with self._db_connection() as conn:
            cursor = conn.cursor()
            try:
                cursor.execute("DELETE FROM body_memory_nodes WHERE id=?", (node.node_id,))
                
                if cursor.rowcount == 0:
                    raise NodeNotFoundError(f"Node {node.node_id} not found in database")
                    
                conn.commit()
            except sqlite3.Error as e:
                self.logger.log_error(f"Failed to delete node {node.node_id}: {e}")
                raise DatabaseError(f"Failed to delete node: {e}")

    # -------------------------------------------------------------------------
    # 8) Utility Functions & Calculations
    # -------------------------------------------------------------------------
    def _calculate_initial_strength(self, state: Dict[str, Any]) -> float:
        """
        Calculate initial node strength considering multiple factors.
        """
        # Base strength calculation
        raw_state = state.get('raw_state', {})
        strength = self._get_core_strength(raw_state)
        
        # Apply contextual modifiers
        mood_mod = self._get_mood_modifier(raw_state)
        behavior_mod = self._get_behavior_modifier(raw_state)
        cognitive_mod = self._get_cognitive_modifier(state.get('processed_state', {}))
        
        # Combine modifiers - multiply to allow for significant dampening
        final_strength = strength * mood_mod * behavior_mod * cognitive_mod
        
        return min(1.0, max(0.1, final_strength))

    def _calculate_connection_weight(self, node1: BodyMemoryNode, node2: BodyMemoryNode) -> float:
        """
        Calculate connection weight between two nodes based on state similarity.
        
        Args:
            node1, node2: Nodes to compare
            
        Returns:
            float: Connection weight [0,1]
        """
        weight = 0.0
        
        # Emotional similarity
        if 'emotional_vectors' in node1.raw_state and 'emotional_vectors' in node2.raw_state:
            # Average similarity across emotional dimensions
            emotions1 = node1.raw_state['emotional_vectors']
            emotions2 = node2.raw_state['emotional_vectors']
            
            emotional_sim = self._calculate_emotional_similarity(emotions1, emotions2)
            weight += emotional_sim * 0.35  # Emotional similarity heavily weighted

        # Mood similarity
        if 'mood' in node1.raw_state and 'mood' in node2.raw_state:
            mood_sim = self._calculate_mood_similarity(
                node1.raw_state['mood'],
                node2.raw_state['mood']
            )
            weight += mood_sim * 0.25
        
        # Behavioral similarity
        if 'behavior' in node1.raw_state and 'behavior' in node2.raw_state:
            behavior_sim = 1.0 if node1.raw_state['behavior'] == node2.raw_state['behavior'] else 0.0
            weight += behavior_sim * 0.2
        
        # Need state similarity
        if 'needs' in node1.raw_state and 'needs' in node2.raw_state:
            needs_sim = self._calculate_needs_similarity(
                node1.raw_state['needs'],
                node2.raw_state['needs']
            )
            weight += needs_sim * 0.2
            
        return min(1.0, weight)

    def _calculate_emotional_similarity(self, emotions1: List[Dict], emotions2: List[Dict]) -> float:
        """Calculate similarity between emotional states"""
        if not emotions1 or not emotions2:
            return 0.0
            
        # Average vector similarities
        similarities = []
        for e1 in emotions1:
            for e2 in emotions2:
                # Compare valence, arousal, intensity
                sim = (
                    (1 - abs(e1.get('valence', 0) - e2.get('valence', 0)) * 0.4) +
                    (1 - abs(e1.get('arousal', 0) - e2.get('arousal', 0)) * 0.4) +
                    (1 - abs(e1.get('intensity', 0) - e2.get('intensity', 0)) * 0.2)
                )
                similarities.append(sim)
                
        return sum(similarities) / len(similarities) if similarities else 0.0

    def _calculate_needs_similarity(self, needs1: Dict, needs2: Dict) -> float:
        """Calculate similarity between need states"""
        if not needs1 or not needs2:
            return 0.0
            
        common_needs = set(needs1.keys()) & set(needs2.keys())
        if not common_needs:
            return 0.0
            
        similarities = []
        for need in common_needs:
            val1 = needs1[need].get('value', 50)
            val2 = needs2[need].get('value', 50)
            similarities.append(1 - abs(val1 - val2) / 100)
            
        return sum(similarities) / len(similarities)

    def _calculate_mood_similarity(self, mood1: Dict, mood2: Dict) -> float:
        """Calculate similarity between mood states"""
        if not mood1 or not mood2:
            return 0.0
            
        # Compare mood vectors similar to emotional vectors
        sim = (
            (1 - abs(mood1.get('valence', 0) - mood2.get('valence', 0)) * 0.5) +
            (1 - abs(mood1.get('arousal', 0) - mood2.get('arousal', 0)) * 0.5)
        )
        return sim

    def _get_core_strength(self, raw_state: Dict[str, Any]) -> float:
        """Calculate base strength from core state values"""
        strength = 0.5  # Base value
        
        # Emotional contribution
        if 'emotional_vectors' in raw_state:
            emotional_strength = 0
            for vector in raw_state['emotional_vectors']:
                # Strong emotions increase strength
                intensity = vector.get('intensity', 0)
                emotional_strength += (abs(vector.get('valence', 0)) * 0.3 + 
                                    abs(vector.get('arousal', 0)) * 0.3 + 
                                    intensity * 0.4)
            strength += min(0.4, emotional_strength / len(raw_state['emotional_vectors']))

        # Needs contribution
        if 'needs' in raw_state:
            need_strength = 0
            for need in raw_state['needs'].values():
                # Extreme needs (very high or low) increase strength
                value = need.get('value', 50)
                urgency = abs(value - 50) / 50
                need_strength += urgency * 0.2
            strength += min(0.3, need_strength)

        return strength

    def _get_mood_modifier(self, raw_state: Dict[str, Any]) -> float:
        """
        Calculate how current mood affects memory strength.
        Extreme moods can either heighten or dampen memory formation.
        """
        mood = raw_state.get('mood', {})
        if not mood:
            return 1.0

        # High arousal can increase memory strength
        arousal_factor = 1.0 + (abs(mood.get('arousal', 0)) * 0.3)
        
        # Extreme valence (positive or negative) can increase strength
        valence_factor = 1.0 + (abs(mood.get('valence', 0)) * 0.2)
        
        return min(1.5, arousal_factor * valence_factor)

    def _get_behavior_modifier(self, raw_state: Dict[str, Any]) -> float:
        """
        Calculate how current behavior affects memory strength.
        Some behaviors might reduce memory formation (sleep, idle)
        while others might enhance it (active engagement).
        """
        behavior = raw_state.get('behavior', {}).get('name', 'idle')
        
        behavior_mods = {
            'sleep': 0.2,    
            'idle': 1.0,     
            'walk': 1.3,     
            'chase': 0.9,    
            'rest': 1.1
        }
        
        return behavior_mods.get(behavior, 1.0)

    def _get_cognitive_modifier(self, processed_state: Dict[str, Any]) -> float:
        """
        Evaluate cognitive state influence on memory strength by analyzing
        the processed cognitive state summary for semantic density and emotional content.
        
        Uses processed_state['cognitive'] which should contain:
        - summary: Processed summary of recent cognitive interactions
        
        Returns:
            float: Cognitive state modifier [0.5-1.5]
        """
        try:
            summary = processed_state.get('cognitive', {})
            if not summary:
                return 1.0

            # Semantic analysis
            semantic_weight = 0.0
            
            # Text complexity analysis
            words = summary.split()
            if not words:
                return 1.0
                
            avg_word_length = sum(len(w) for w in words) / len(words)
            sentence_count = summary.count('.') + summary.count('!') + summary.count('?')
            avg_sentence_length = len(words) / sentence_count if sentence_count > 0 else 0
            
            # Calculate semantic density score
            semantic_score = min(1.0, (
                (len(words) / 50) * 0.3 +  # Length factor (normalized for summary)
                (avg_word_length / 8) * 0.3 +  # Complexity factor
                (avg_sentence_length / 15) * 0.4  # Structure factor
            ))
            semantic_weight = semantic_score
            
            # Emotional content analysis 
            emotional_weight = 0.0
            
            # Enhanced emotional indicator categories
            emotional_indicators = {
                'high': [
                    '!', '?!', 'very', 'extremely', 'critical', 'urgent',
                    'excited', 'thrilled', 'angry', 'overjoyed', 'devastated'
                ],
                'medium': [
                    'should', 'need', 'important', 'significant',
                    'happy', 'sad', 'concerned', 'interested', 'worried'
                ],
                'low': [
                    'maybe', 'perhaps', 'consider', 'slight',
                    'mild', 'calm', 'steady', 'stable'
                ]
            }
            
            # Check for emotional content density
            summary_lower = summary.lower()
            emotion_scores = []
            
            for level, indicators in emotional_indicators.items():
                count = sum(summary_lower.count(i) for i in indicators)
                normalized_count = count / len(words) if words else 0
                
                if level == 'high':
                    emotion_scores.append(normalized_count * 0.5)
                elif level == 'medium':
                    emotion_scores.append(normalized_count * 0.3)
                else:
                    emotion_scores.append(normalized_count * 0.2)
            
            emotional_weight = min(1.0, sum(emotion_scores))
            
            # Combine modifiers with adjusted weights
            # Give more weight to semantic density for summaries
            semantic_modifier = semantic_weight * 0.7
            emotional_modifier = emotional_weight * 0.3
            
            # Return final modifier [0.5-1.5]
            return 0.5 + min(1.0, semantic_modifier + emotional_modifier)
            
        except Exception as e:
            self.logger.warning(f"Failed to calculate cognitive modifier: {e}")
            return 1.0

    def update_connections(self, node: BodyMemoryNode) -> None:
        """
        Update node's connections based on current state similarities.
        Handles ghost connections and connection pruning.
        """
        if node.ghosted:
            return
                
        try:
            # Get candidates for connection updates
            candidates = [n for n in self.nodes if not n.ghosted and n.node_id != node.node_id]
            
            old_connections = node.connections.copy()
            new_connections = {}
            
            for other in candidates:
                weight = self._calculate_connection_weight(node, other)
                
                # Only keep strong enough connections
                if weight > 0.2:
                    new_connections[other.node_id] = weight
                    
                    # Update reciprocal connection
                    other.connections[node.node_id] = weight
                    self._update_node(other)
                    
                # Handle significantly weakened connections
                elif (other.node_id in old_connections and 
                    old_connections[other.node_id] > 0.7 and 
                    weight < 0.3):
                    self._preserve_ghost_connection(node, other, old_connections[other.node_id])
            
            # Update node's connections
            node.connections = new_connections
            node.last_connection_update = time.time()
            self._update_node(node)
                
        except Exception as e:
            self.logger.log_error(f"Failed to update connections for node {node.node_id}: {e}")
            raise MemorySystemError(f"Connection update failed: {e}")

    def _preserve_ghost_connection(
        self,
        node: BodyMemoryNode,
        other: BodyMemoryNode,
        original_weight: float
    ) -> None:
        """
        Preserve a weakened connection as a ghost connection.
        Helpful for potential future resonance or memory resurrection.
        """
        ghost_connection = {
            'node_id': other.node_id,
            'original_weight': original_weight,
            'timestamp': time.time(),
            'state_snapshot': {
                'emotional': other.raw_state.get('emotional_vectors', []),
                'behavior': other.raw_state.get('behavior', {}),
                'needs': other.raw_state.get('needs', {})
            }
        }
        
        if not hasattr(node, 'ghost_connections'):
            node.ghost_connections = []
        node.ghost_connections.append(ghost_connection)

    def update_all_connections(self, force: bool = False) -> None:
        """
        Periodic update of all node connections.
        Useful after significant system changes or by request.
        
        Args:
            force: Whether to force update all connections regardless of state
        """
        active_nodes = [n for n in self.nodes if not n.ghosted]
        
        for node in active_nodes:
            # Skip if connections were recently updated unless forced
            if not force and hasattr(node, 'last_connection_update'):
                if time.time() - node.last_connection_update < 3600:  # 1 hour
                    continue
            
            self.update_connections(node)
            node.last_connection_update = time.time()