"""
cognitive_memory.py

Implements the CognitiveMemory system for managing semantically interpreted experiences
as an associative network of memory nodes. Handles formation, storage, relationships,
decay dynamics and synthesis of cognitive memories.

Key capabilities:
- Creates & stores memory nodes from significant emotional/behavioral events
- Maintains associations between cognitive nodes and body memory nodes
- Manages memory strength/decay with ghosting mechanics for weak memories 
- Handles triggering memory echo effects and dampening
- Resolves conflicts and synthesizes new memories from merged experiences
- Provides traversal/querying of the memory network

Code organization:
    1) DB & Table Setup 
    2) Event Listening & Memory Formation
    3) Link & Merge Handling / DB Ops
    4) Node Retrieval & Queries
    5) Ghost & Decay Handling
    6) Echos
    7) Utility & Internals

The system uses SQLite for persistence, with memory_links tracking body-cognitive
relationships and synthesis_relations tracking memory merges/resurrections.
"""

import asyncio
import sqlite3
import time
import json
import math
from loggers.loggers import MemoryLogger
from typing import Dict, List, Any, Optional, Tuple, Union
from contextlib import contextmanager
from sentence_transformers import SentenceTransformer
from collections import defaultdict
from nltk import pos_tag, word_tokenize, ne_chunk
from nltk.tree import Tree
import numpy as np

from event_dispatcher import Event, global_event_dispatcher
from .cognitive_memory_node import CognitiveMemoryNode

class DatabaseError(Exception):
    """Database operation errors."""
    pass

class MemorySystemError(Exception):
    """Generic memory system error."""
    pass

class CognitiveMemory:
    """
    Manages semantically interpreted experiences and their relationships.
    
    Maintains a network of cognitive memory nodes that can interact with both
    other cognitive nodes and body memory nodes. Handles memory formation,
    synthesis, and retrieval based on both semantic and somatic context.
    """
    
    # -------------------------------------------------------------------------
    # 1) DB & Table Setup
    # -------------------------------------------------------------------------
    def __init__(self, internal_context, body_memory, db_path: str = 'data/memory.db',
                 ghost_threshold: float = 0.1,
                 final_prune_threshold: float = 0.05,
                 revive_threshold: float = 0.2):
        """
        Args:
            internal_context: Access to internal states for memory capturing
            body_memory: Reference to BodyMemory manager for retrieving body node IDs
            db_path: Where to store SQLite data
        """
        self.internal_context = internal_context
        self.body_memory = body_memory
        self.nodes: List[CognitiveMemoryNode] = []

        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

        self.db_path = db_path
        self.min_active_nodes = 5
        self.logger = MemoryLogger

        self._init_database()
        self._load_nodes_sync()

        # Ghost config
        self.ghost_threshold = ghost_threshold
        self.final_prune_threshold = final_prune_threshold
        self.revive_threshold = revive_threshold

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
        """Create tables for cognitive nodes & memory_links if not exist."""
        with self._db_connection() as conn:
            cursor = conn.cursor()
            # Nodes
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS cognitive_memory_nodes (
                    id INTEGER PRIMARY KEY,
                    timestamp REAL,
                    text_content TEXT,
                    embedding TEXT,
                    raw_state TEXT,
                    processed_state TEXT,
                    strength REAL,
                    ghosted BOOLEAN,
                    parent_node_id INTEGER,
                    ghost_nodes TEXT,
                    ghost_states TEXT,
                    connections TEXT,
                    semantic_context TEXT,
                    last_accessed REAL,
                    formation_source TEXT,
                    last_echo_time REAL,
                    echo_dampening REAL,
                    FOREIGN KEY(parent_node_id) REFERENCES cognitive_memory_nodes(id)
                )
            """)
            # memory_links: body <-> cognitive relationships
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS memory_links (
                    id INTEGER PRIMARY KEY,
                    cognitive_node_id INTEGER,
                    body_node_id INTEGER,
                    link_strength REAL,
                    link_type TEXT,
                    created_at REAL,
                    metadata TEXT,
                    UNIQUE(cognitive_node_id, body_node_id),
                    FOREIGN KEY(cognitive_node_id) REFERENCES cognitive_memory_nodes(id),
                    FOREIGN KEY(body_node_id) REFERENCES body_memory_nodes(id)
                )
            """)
            # Synthesis relations
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS synthesis_relations (
                    id INTEGER PRIMARY KEY,
                    synthesis_node_id INTEGER,
                    constituent_node_id INTEGER,
                    relationship_type TEXT,
                    metadata TEXT,
                    FOREIGN KEY(synthesis_node_id) REFERENCES cognitive_memory_nodes(id),
                    FOREIGN KEY(constituent_node_id) REFERENCES cognitive_memory_nodes(id)
                )
            """)
            conn.commit()

    # NOTE: For alpha, we do a sync load.
    def _load_nodes_sync(self):
        """Synchronous load of all cognitive nodes from DB into memory."""
        with self._db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM cognitive_memory_nodes")
            columns = [d[0] for d in cursor.description]
            
            for row in cursor.fetchall():
                data = dict(zip(columns, row))
                node = CognitiveMemoryNode.from_dict(data)
                self.nodes.append(node)
                
    # -------------------------------------------------------------------------
    # 2) Event Listening & Memory Formation
    # -------------------------------------------------------------------------
    def setup_event_listeners(self):
        """
        Use lambda-based async tasks to play nice with synch event dispatcher
        """
        global_event_dispatcher.add_listener(
            'cognitive:memory:content_generated',
            lambda event: asyncio.create_task(self.complete_memory_formation(event))
        )

        global_event_dispatcher.add_listener('cognitive:memory:conflict_resolved', self.on_conflict_resolved)

    async def request_memory_formation(self, event: Event):
        """
        First half of memory formation - requests content generation from ExoProcessor
        """
        try:
            # 1) Request content generation via event (state requested by formation process)
            global_event_dispatcher.dispatch_event(Event(
                "memory:content_requested",
                {
                    "event_type": event.event_type,
                    "event_data": event.data,
                    "original_event": event
                }
            ))

        except Exception as e:
            self.logger.log_error(f"Failed to request memory formation: {e}")

    async def complete_memory_formation(self, event: Event) -> Optional[str]:
        """
        Second half of memory formation - processes content and creates node
        """
        try:
            # Extract original context and generated content
            text_content = event.data.get('content')
            original_event = event.data.get('original_event')
            current_state = self.internal_context.get_memory_context(is_cognitive=True)
            
            if not all([text_content, original_event, current_state]):
                raise ValueError("Missing required data for memory formation")
            
            # figure out a way to attach the meta-data here in additional property maybe

            # Form the actual memory node (async)
            node_id = await self.form_memory(
                text_content=text_content,
                raw_state=current_state.raw_state,
                processed_state=current_state.processed_state,
                formation_source=original_event.event_type
            )

            # Log memory formation completion
            self.logger.log_memory_formation(
                memory_type='cognitive',
                memory_id=node_id,
                details={
                    'text_content': text_content,
                    'formation_source': original_event.event_type,
                    'state_snapshot': {
                        'raw': current_state.raw_state,
                        'processed': current_state.processed_state
                    }
                }
            )

            return node_id
            
        except Exception as e:
            self.logger.log_error(f"Failed to complete memory formation: {e}")
            return None

    # -------------------------------------------------------------------------
    # 3) Link & Merge Handling
    # -------------------------------------------------------------------------
    def add_body_link(self, cognitive_id: str, body_id: str, strength: float = 1.0) -> None:
        """
        Insert or update a body-cognitive link in the memory_links table.
        """
        with self._db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO memory_links (cognitive_node_id, body_node_id, link_strength, created_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(cognitive_node_id, body_node_id) DO UPDATE 
                SET link_strength = ?
            """, (int(cognitive_id), int(body_id), strength, time.time(), strength))
            conn.commit()

    def get_body_links(self, cognitive_id: str) -> List[Tuple[str, float]]:
        """Retrieve all body links for a given cognitive node from the DB."""
        with self._db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT body_node_id, link_strength
                FROM memory_links
                WHERE cognitive_node_id = ?
            """, (cognitive_id,))
            return [(str(r[0]), r[1]) for r in cursor.fetchall()]

    def remove_body_links(self, cognitive_id: str) -> None:
        """Remove all body links for a given cognitive node."""
        with self._db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                DELETE FROM memory_links
                WHERE cognitive_node_id = ?
            """, (cognitive_id,))
            conn.commit()

    def transfer_body_links(self, from_id: str, to_id: str, strength_modifier: float = 0.9) -> None:
        """
        Transfer or unify body links from one cognitive node to another (during merges).
        """
        links = self.get_body_links(from_id)
        for (body_id, old_strength) in links:
            new_strength = old_strength * strength_modifier
            self.add_body_link(to_id, body_id, new_strength)
        self.remove_body_links(from_id)

    def merge_nodes(self, from_node: CognitiveMemoryNode, to_node: CognitiveMemoryNode):
        """
        Manager-level merge to unify body links and handle node merges.
        1) Transfer body links in DB
        2) Use from_node.merge_into_parent(...) for cognitive merges
        3) Call update_connections
        """
        try:
            from_id = from_node.node_id
            to_id = to_node.node_id
            if not from_id or not to_id:
                return
            
            # Transfer DB-based body links
            self.transfer_body_links(from_id, to_id, 0.9)
            
            # Merge at node level (ghosting)
            from_node.merge_into_parent(to_node)
            
            # Update adjacency
            self.update_connections(to_node)
            
        except Exception as e:
            self.logger.log_error(f"Failed to merge nodes {from_node.node_id} -> {to_node.node_id}: {e}")

    # -------------------------------------------------------------------------
    # 4) Node Retrieval Metrics & Queries
    # -------------------------------------------------------------------------
    def _get_node_by_id(self, node_id: str) -> Optional[CognitiveMemoryNode]:
        """Fetch a node from in-memory list. For alpha use only—no DB query here."""
        return next((n for n in self.nodes if str(n.node_id) == str(node_id)), None)

    def find_nodes_by_text(self, substring: str) -> List[CognitiveMemoryNode]:
        """
        Example of a naive text search. Real approach would do embedding similarity.
        """
        return [
            n for n in self.nodes
            if (substring.lower() in n.text_content.lower())
            and not n.ghosted
        ]

    def retrieve_memories(
        self,
        query: str,
        given_state: Dict[str, Any],
        top_k: int = 10,
        return_details: bool = False
    ) -> Union[List[CognitiveMemoryNode], Tuple[List[CognitiveMemoryNode], List[Dict[str, Any]]]]:
        """
        Retrieves the top_k most relevant memories based on query and state.
        Uses centralized retrieval metrics for consistent analysis. 

        Args:
            query: The query string to retrieve relevant memories
            given_state: The given internal state for comparison
            top_k: Number of top memories to retrieve
            return_details: If True, returns detailed retrieval metrics alongside nodes
            
        Returns:
            Either List[CognitiveMemoryNode] or Tuple of (nodes, details) if return_details=True
        """
        try:
            # Generate query embedding once
            query_embedding = self._generate_embedding(query)
            
            # Calculate retrieval metrics for all active nodes
            retrieval_scores = []
            
            for node in (n for n in self.nodes if not n.ghosted):
                metrics = self.calculate_retrieval_metrics(
                    target_node=node,
                    comparison_state=given_state,
                    query_text=query,
                    query_embedding=query_embedding,
                    include_strength=True,
                    detailed_metrics=return_details
                )
                
                if return_details:
                    retrieval_scores.append((node, metrics['final_score'], metrics))
                else:
                    retrieval_scores.append((node, metrics))
            
            # Sort by final score
            retrieval_scores.sort(key=lambda x: x[1], reverse=True)
            
            # Trigger echos on top results
            top_results = retrieval_scores[:top_k]
            for node, similarity, *_ in top_results:
                asyncio.create_task(self.trigger_echo(
                        node,
                        similarity,
                        comparison_state=given_state,
                        query_text=query,
                        query_embedding=query_embedding
                    )
                )
            
            if return_details:
                # Return both nodes and their detailed metrics
                nodes = [n for n, _, _ in top_results]
                details = [d for _, _, d in top_results]
                return nodes, details
            else:
                # Return just the nodes
                return [n for n, _ in top_results]

        except Exception as e:
            self.logger.log_error(f"Failed to retrieve memories: {e}")
            return [] if not return_details else ([], [])

    def calculate_retrieval_metrics(
        self,
        target_node: CognitiveMemoryNode,
        comparison_state: Dict[str, Any] = None,
        query_text: str = None,
        query_embedding: List[float] = None,
        include_strength: bool = True,
        detailed_metrics: bool = False
    ) -> Union[float, Dict[str, Any]]:
        """
        Calculate comprehensive retrieval metrics for a single node, comparing against
        either current state or provided comparison state/query.
        
        Use cases:
        1. Memory retrieval scoring (default)
        2. Initial strength calculation (include_strength=False)
        3. Conflict detection (detailed_metrics=True)
        4. State synthesis analysis (detailed_metrics=True)
        
        Args:
            target_node: Node to analyze
            comparison_state: Optional explicit state to compare against
            query_text: Optional text query for semantic comparison
            query_embedding: Optional pre-computed embedding vector
            include_strength: Whether to factor node strength into final score
            detailed_metrics: Return detailed component scores vs single value
            
        Returns:
            Union[float, Dict[str, Any]]: Either final similarity score or detailed metrics
        """
        metrics = {}
        
        try:
            # Get comparison state if not provided
            if not comparison_state:
                comparison_state = self.internal_context.get_memory_context(is_cognitive=True)
            
            # 1. Semantic Analysis
            semantic_metrics = self._calculate_semantic_metrics(
                target_node,
                query_text,
                query_embedding
            )
            metrics['semantic'] = semantic_metrics
            
            # 2. Emotional Analysis
            emotional_metrics = self._calculate_emotional_metrics(
                target_node.raw_state,
                comparison_state.get('raw_state', {})
            )
            metrics['emotional'] = emotional_metrics
            
            # 3. State Analysis (broken down by component)
            state_metrics = self._calculate_state_metrics(
                target_node.raw_state,
                target_node.processed_state,
                comparison_state.get('raw_state', {}),
                comparison_state.get('processed_state', {})
            )
            metrics['state'] = state_metrics
            
            # 4. Temporal Analysis
            temporal_metrics = self._calculate_temporal_metrics(target_node)
            metrics['temporal'] = temporal_metrics
            
            # 5. Strength Component (optional)
            if include_strength:
                strength_metrics = self._calculate_strength_metrics(target_node)
                metrics['strength'] = strength_metrics

            # Return either detailed metrics or computed final score
            if detailed_metrics:
                return {
                    'final_score': self._compute_final_score(metrics),
                    'component_metrics': metrics,
                    'component_weights': self._get_component_weights()
                }
            else:
                return self._compute_final_score(metrics)
            
        except Exception as e:
            self.logger.log_error(f"Failed to calculate retrieval metrics: {e}")
            return 0.0 if not detailed_metrics else {'error': str(e)}

    def _calculate_semantic_metrics(
        self,
        node: CognitiveMemoryNode,
        query_text: Optional[str],
        query_embedding: Optional[List[float]]
    ) -> Dict[str, float]:
        """Calculate detailed semantic similarity metrics"""
        metrics = {}
        
        try:
            # Generate embedding if needed
            if query_embedding is None and query_text:
                query_embedding = self._generate_embedding(query_text)
                
            if query_embedding is not None:
                # Direct embedding similarity
                metrics['embedding_similarity'] = self._cosine_similarity(
                    query_embedding,
                    node.embedding
                )
                
            # Text content analysis if query provided
            if query_text:
                metrics['text_relevance'] = self._analyze_text_relevance(
                    query_text,
                    node.text_content
                )
                
            # Calculate semantic density
            metrics['semantic_density'] = self._calculate_semantic_density(
                node.text_content
            )
            
            return metrics
            
        except Exception as e:
            self.logger.log_error(f"Semantic metric calculation failed: {e}")
            return {'error': str(e)}
        
    def _calculate_semantic_density(self, text: str) -> float:
        """
        Calculate semantic density using multiple linguistic features:
        1. Embedding-based semantic richness (using existing model)
        2. Lexical diversity and information content
        3. Syntactic complexity
        4. Named entity density
        5. Content word ratio
        
        Returns:
            float: Normalized semantic density score [0-1]
        """
        try:
            
            # Ensure text is valid
            if not text or len(text.strip()) == 0:
                return 0.0
                
            # 1. Embedding-based semantic richness
            # Get sentence embeddings and calculate average cosine similarity
            # between segments to measure semantic cohesion
            sentences = [s.strip() for s in text.split('.') if s.strip()]
            if len(sentences) > 1:
                embeddings = [self.embedding_model.encode(s) for s in sentences]
                similarities = []
                for i in range(len(embeddings)):
                    for j in range(i+1, len(embeddings)):
                        sim = self._cosine_similarity(embeddings[i], embeddings[j])
                        similarities.append(sim)
                semantic_cohesion = np.mean(similarities) if similarities else 0.5
            else:
                semantic_cohesion = 0.5  # Neutral for single sentences
            
            # 2. Lexical diversity & information content
            tokens = word_tokenize(text.lower())
            unique_ratio = len(set(tokens)) / len(tokens) if tokens else 0
            
            # 3. Syntactic complexity 
            pos_tags = pos_tag(tokens)
            # Calculate ratio of complex structures (subordinating conjunctions, etc)
            complex_tags = {'IN', 'WDT', 'WP', 'WRB'}
            syntactic_complexity = len([t for _, t in pos_tags if t in complex_tags]) / len(pos_tags) if pos_tags else 0
            
            # 4. Named entity density
            try:
                ne_tree = ne_chunk(pos_tags)
                named_entities = len([subtree for subtree in ne_tree if type(subtree) == Tree])
                ne_density = named_entities / len(tokens) if tokens else 0
            except:
                ne_density = 0
            
            # 5. Content word ratio
            content_tags = {'NN', 'NNS', 'NNP', 'NNPS', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', 'JJ', 'JJR', 'JJS'}
            content_ratio = len([t for _, t in pos_tags if t in content_tags]) / len(pos_tags) if pos_tags else 0
            
            # Combine metrics with weighted average
            weights = {
                'semantic_cohesion': 0.3,
                'lexical_diversity': 0.2,
                'syntactic_complexity': 0.15,
                'ne_density': 0.15,
                'content_ratio': 0.2
            }
            
            density_score = (
                semantic_cohesion * weights['semantic_cohesion'] +
                unique_ratio * weights['lexical_diversity'] +
                syntactic_complexity * weights['syntactic_complexity'] +
                ne_density * weights['ne_density'] +
                content_ratio * weights['content_ratio']
            )
            
            # Normalize to [0-1] range
            return min(1.0, max(0.0, density_score))
            
        except Exception as e:
            self.logger.log_error(f"Enhanced semantic density calculation failed: {e}")
            # Fallback to simpler calculation if NLP tools fail
            try:
                words = text.split()
                meaningful_words = [w for w in words if w.isalpha()]
                return len(set(meaningful_words)) / len(meaningful_words) if meaningful_words else 0.0
            except:
                return 0.0

    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """
        Calculate the cosine similarity between two vectors.

        Args:
            vec1 (List[float]): First vector.
            vec2 (List[float]): Second vector.

        Returns:
            float: Cosine similarity score between 0 and 1.
        """
        try:
            dot_product = sum(a * b for a, b in zip(vec1, vec2))
            magnitude1 = math.sqrt(sum(a ** 2 for a in vec1))
            magnitude2 = math.sqrt(sum(b ** 2 for b in vec2))
            if magnitude1 == 0 or magnitude2 == 0:
                return 0.0
            return dot_product / (magnitude1 * magnitude2)
        except Exception as e:
            self.logger.log_error(f"Cosine similarity calculation failed: {e}")
            return 0.0

    def _analyze_text_relevance(self, query: str, text: str) -> float:
        """
        Analyze the relevance of the text to the query.

        Args:
            query (str): The query string.
            text (str): The text to analyze.

        Returns:
            float: Relevance score between 0 and 1.
        """
        try:
            # Simple keyword matching for relevance
            query_words = set(query.lower().split())
            text_words = set(text.lower().split())
            if not query_words:
                return 0.0
            relevant_words = query_words.intersection(text_words)
            relevance = len(relevant_words) / len(query_words)
            return relevance
        except Exception as e:
            self.logger.log_error(f"Text relevance analysis failed: {e}")
            return 0.0

    def _calculate_emotional_metrics(
        self,
        node_state: Dict[str, Any],
        comparison_state: Dict[str, Any]
    ) -> Dict[str, float]:
        """Calculate detailed emotional similarity and significance metrics"""
        metrics = {}
        
        try:
            if 'emotional_vectors' in node_state and 'emotional_vectors' in comparison_state:
                node_vectors = node_state['emotional_vectors']
                comp_vectors = comparison_state['emotional_vectors']
                
                # Vector similarity
                metrics['vector_similarity'] = self._calculate_emotional_similarity(
                    node_vectors,
                    comp_vectors
                )
                
                # Emotional intensity
                metrics['node_intensity'] = self._calculate_emotional_intensity(node_vectors)
                metrics['comparison_intensity'] = self._calculate_emotional_intensity(comp_vectors)
                
                # Emotional complexity (number of distinct emotions)
                metrics['emotional_complexity'] = len(node_vectors)
                
                # Valence analysis
                metrics['valence_shift'] = abs(
                    self._calculate_net_valence(node_vectors) -
                    self._calculate_net_valence(comp_vectors)
                )
            
            return metrics
            
        except Exception as e:
            self.logger.log_error(f"Emotional metric calculation failed: {e}")
            return {'error': str(e)}
        
    def _calculate_emotional_similarity(self, node_vectors: List[Dict[str, Any]], comp_vectors: List[Dict[str, Any]]) -> float:
        """
        Calculate similarity between two sets of emotional vectors.
        Compares the valence-arousal pairs between vectors.
        """
        try:
            if not node_vectors or not comp_vectors:
                return 0.0
            
            # Extract valence-arousal pairs for cosine similarity
            # EmotionalVector instances are serialized as dicts in state
            similarities = []
            for nv in node_vectors:
                n_vec = [nv.get('valence', 0.0), nv.get('arousal', 0.0)]
                for cv in comp_vectors:
                    c_vec = [cv.get('valence', 0.0), cv.get('arousal', 0.0)]
                    similarities.append(self._cosine_similarity(n_vec, c_vec))
                    
            return sum(similarities) / len(similarities) if similarities else 0.0
        except Exception as e:
            self.logger.log_error(f"Emotional similarity calculation failed: {e}")
            return 0.0

    def _calculate_emotional_intensity(self, vectors: List[Dict[str, Any]]) -> float:
        """
        Calculate emotional intensity based on vector intensities.
        Uses direct intensity values rather than computing magnitudes.
        """
        try:
            if not vectors:
                return 0.0
            # Use stored intensity values from EmotionalVectors
            intensities = [v.get('intensity', 0.0) for v in vectors]
            return sum(intensities) / len(intensities) if intensities else 0.0
        except Exception as e:
            self.logger.log_error(f"Emotional intensity calculation failed: {e}")
            return 0.0

    def _calculate_net_valence(self, vectors: List[Dict[str, Any]]) -> float:
        """
        Calculate net valence based on intensity-weighted emotional vectors.
        
        Emotional vectors in the system are represented as dicts with 'valence' and 
        'intensity' fields, serialized from EmotionalVector instances.
        """
        try:
            if not vectors:
                return 0.0
                
            # Weight valences by their intensities for net calculation
            weighted_valences = []
            total_intensity = 0.0
            
            for vector in vectors:
                valence = vector.get('valence', 0.0)
                intensity = vector.get('intensity', 0.0)
                
                weighted_valences.append(valence * intensity)
                total_intensity += intensity
            
            # Return intensity-weighted average
            if total_intensity > 0:
                return sum(weighted_valences) / total_intensity
            return 0.0
            
        except Exception as e:
            self.logger.log_error(f"Net valence calculation failed: {e}")
            return 0.0


    def _calculate_state_metrics(
        self,
        node_raw: Dict[str, Any], 
        node_processed: Dict[str, Any],
        comp_raw: Dict[str, Any],
        comp_processed: Dict[str, Any]
    ) -> Dict[str, Dict[str, float]]:
        """
        Calculate detailed state similarity metrics across all components.
        Uses BodyMemory's core state comparison logic where possible
        """
        metrics = {
            'needs': {},
            'behavior': {},
            'mood': {},
            'emotional': {}  
        }
        
        try:
            # Get base similarity metrics from BodyMemory
            base_similarity = self.body_memory._calculate_state_similarity(
                {'raw_state': node_raw, 'processed_state': node_processed},
                {'raw_state': comp_raw, 'processed_state': comp_processed}
            )

            # Map basic components from BodyMemory results
            if 'needs' in node_raw and 'needs' in comp_raw:
                try:
                    metrics['needs'] = {
                        'satisfaction_similarity': base_similarity,
                        'urgency_levels': self._calculate_need_urgency(node_raw['needs']),
                        'state_shifts': abs(
                            self.body_memory._calculate_needs_similarity(node_raw['needs'], comp_raw['needs']) - 1.0
                        )
                    }
                except Exception as inner_e:
                    self.logger.error(f"Debug - Failed in needs calculation: {str(inner_e)}")
                    self.logger.error(f"needs_similarity access attempted on: {type(base_similarity)}")
                    raise
                
            if 'behavior' in node_raw and 'behavior' in comp_raw:
                behavior_match = node_raw['behavior'].get('name') == comp_raw['behavior'].get('name')
                metrics['behavior'] = {
                    'matching': behavior_match,
                    'transition_significance': 1.0 - float(behavior_match)  # Simple transition metric
                }
                
            if 'mood' in node_raw and 'mood' in comp_raw:
                metrics['mood'] = {
                    'similarity': self.body_memory._calculate_mood_similarity(
                        node_raw['mood'],
                        comp_raw['mood']
                    ),
                    'intensity_delta': abs(
                        self._get_mood_intensity(node_raw['mood']) -
                        self._get_mood_intensity(comp_raw['mood'])
                    )
                }
                
            # Add emotional state metrics (counts here, just not weighted as strongly)
            if 'emotional_vectors' in node_raw and 'emotional_vectors' in comp_raw:
                emotional_similarity = self._calculate_emotional_similarity(
                    node_raw['emotional_vectors'],
                    comp_raw['emotional_vectors']
                )
                
                metrics['emotional'] = {
                    'vector_similarity': emotional_similarity,
                    'intensity_delta': abs(
                        self._calculate_emotional_intensity(node_raw['emotional_vectors']) -
                        self._calculate_emotional_intensity(comp_raw['emotional_vectors'])
                    ),
                    'valence_shift': abs(
                        self._calculate_net_valence(node_raw['emotional_vectors']) -
                        self._calculate_net_valence(comp_raw['emotional_vectors'])
                    )
                }
                
            # Add cognitive-specific metrics
            # at some point, include this here. for now, we can't quite manage it. need more detailed metrics or ways of accessing the llms state itself.
                
            return metrics
            
        except Exception as e:
            import traceback
            print("Debug - Full error in state metric calculation:")
            print(traceback.format_exc())
            self.logger.log_error(f"State metric calculation failed: {e}")
            return {'error': str(e)}
        
    def _calculate_need_urgency(self, needs: Dict[str, Dict[str, Any]]) -> float:
        """
        Calculate urgency level based on need satisfaction values.
        Uses the same satisfaction calculation as NeedsManager for consistency,
        accounting for different need types (e.g. stamina vs other needs).
        
        Args:
            needs: Dictionary of needs with their current values and states
            
        Returns:
            float: Overall urgency level [0,1] across all needs
        """
        try:
            if not needs:
                return 0.0
                
            urgency_sum = 0.0
            need_count = len(needs)
            
            for need_name, need_data in needs.items():
                # Get current value and account for stamina's inverse satisfaction
                current_value = need_data.get('current_value', 0)
                if need_name == 'stamina':
                    # Stamina urgency increases as value decreases
                    satisfaction = current_value / 100.0  # Assuming max value of 100
                else:
                    # Other needs urgency increases as value increases
                    satisfaction = 1.0 - (current_value / 100.0)
                    
                # Can also use pre-calculated satisfaction if available
                if 'satisfaction' in need_data:
                    satisfaction = need_data['satisfaction']
                    
                # Higher urgency = lower satisfaction
                urgency_sum += (1.0 - satisfaction)
            
            return urgency_sum / need_count if need_count > 0 else 0.0
            
        except Exception as e:
            self.logger.log_error(f"Need urgency calculation failed: {e}")
            return 0.0

    def _get_mood_intensity(self, mood: Dict[str, Any]) -> float:
        """Calculate overall mood intensity from valence/arousal."""
        if not mood:
            return 0.0
        return math.sqrt(
            mood.get('valence', 0) ** 2 + 
            mood.get('arousal', 0) ** 2
        ) / math.sqrt(2)  # Normalize to [0,1]


    def _calculate_temporal_metrics(self, node: CognitiveMemoryNode) -> Dict[str, float]:
        """Calculate temporal relevance metrics"""
        metrics = {}
        
        try:
            current_time = time.time()
            time_diff = current_time - node.timestamp
            
            # Basic time decay
            metrics['recency'] = math.exp(-time_diff / 3600)  # 1-hour decay
            
            # Access history
            if node.last_accessed:
                access_diff = current_time - node.last_accessed
                metrics['access_recency'] = math.exp(-access_diff / 3600)
            
            # Echo history
            if node.last_echo_time:
                echo_diff = current_time - node.last_echo_time
                metrics['echo_recency'] = math.exp(-echo_diff / 3600)
                metrics['echo_dampening'] = node.echo_dampening
                
            return metrics
            
        except Exception as e:
            self.logger.log_error(f"Temporal metric calculation failed: {e}")
            return {'error': str(e)}

    def _calculate_strength_metrics(self, node: CognitiveMemoryNode) -> Dict[str, float]:
        """Calculate strength-related metrics"""
        metrics = {}
        
        try:
            # Base strength
            metrics['current_strength'] = node.strength
            
            # Network position
            connected_strengths = [
                self._get_node_by_id(nid).strength
                for nid in node.connections.keys()
                if self._get_node_by_id(nid)
            ]
            
            if connected_strengths:
                metrics['relative_strength'] = (
                    node.strength / 
                    (sum(connected_strengths) / len(connected_strengths))
                )
            
            # Ghost state
            metrics['is_ghosted'] = float(node.ghosted)
            metrics['ghost_nodes_count'] = len(node.ghost_nodes)
            
            return metrics
            
        except Exception as e:
            self.logger.log_error(f"Strength metric calculation failed: {e}")
            return {'error': str(e)}

    def _get_component_weights(self) -> Dict[str, float]:
        """Get current weight configuration for components"""
        return {
            'semantic': 0.3,
            'emotional': 0.25,
            'state': 0.25,
            'temporal': 0.1,
            'strength': 0.1
        }

    def _compute_final_score(self, metrics: Dict[str, Any]) -> float:
        """
        Compute final similarity score by combining weighted component metrics.
        
        Components align with calculate_retrieval_metrics outputs:
        - Semantic: embedding similarity, text relevance, semantic density
        - Emotional: vector similarity, valence shifts, emotional complexity
        - State: needs, behavior, mood similarity metrics  
        - Temporal: recency and access/echo history
        - Strength (optional): current strength and network position
        """
        weights = self._get_component_weights()
        score = 0.0
        
        try:
            # Semantic Score 
            if 'semantic' in metrics:
                semantic_score = (
                    metrics['semantic'].get('embedding_similarity', 0) * 0.5 +
                    metrics['semantic'].get('text_relevance', 0) * 0.2 +
                    metrics['semantic'].get('semantic_density', 0) * 0.3
                )
                score += semantic_score * weights['semantic']
            
            # Emotional Score
            if 'emotional' in metrics:
                emotional_score = (
                    metrics['emotional'].get('vector_similarity', 0) * 0.4 +
                    metrics['emotional'].get('valence_shift', 0) * 0.3 +
                    (metrics['emotional'].get('emotional_complexity', 0) / 5.0) * 0.3  # Normalized by max 5 emotions
                )
                score += emotional_score * weights['emotional']
            
            # State Score (weighted average of components)
            if 'state' in metrics:
                state_scores = {}
                
                # Needs comparison
                if 'needs' in metrics['state']:
                    state_scores['needs'] = (
                        metrics['state']['needs'].get('satisfaction_similarity', 0) * 0.6 +
                        metrics['state']['needs'].get('urgency_levels', 0) * 0.4
                    )
                
                # Behavior matching
                if 'behavior' in metrics['state']:
                    state_scores['behavior'] = (
                        1.0 - metrics['state']['behavior'].get('transition_significance', 0)
                    )
                    
                # Mood comparison
                if 'mood' in metrics['state']:
                    state_scores['mood'] = (
                        metrics['state']['mood'].get('similarity', 0) * 0.7 +
                        (1.0 - metrics['state']['mood'].get('intensity_delta', 0)) * 0.3
                    )
                    
                # Emotional state 
                if 'emotional' in metrics['state']:
                    state_scores['emotional'] = (
                        metrics['state']['emotional'].get('vector_similarity', 0) * 0.6 +
                        (1.0 - metrics['state']['emotional'].get('valence_shift', 0)) * 0.4
                    )
                
                if state_scores:
                    # Weight the state components
                    component_weights = {
                        'needs': 0.3,
                        'behavior': 0.2, 
                        'mood': 0.25,
                        'emotional': 0.25
                    }
                    
                    state_score = sum(
                        score * component_weights.get(component, 1.0)
                        for component, score in state_scores.items()
                    ) / sum(
                        component_weights.get(component, 1.0) 
                        for component in state_scores.keys()
                    )
                    
                    score += state_score * weights['state']
            
            # Temporal Score
            if 'temporal' in metrics:
                temporal_score = (
                    metrics['temporal'].get('recency', 0) * 0.4 +
                    metrics['temporal'].get('access_recency', 0) * 0.3 +
                    metrics['temporal'].get('echo_recency', 0) * 0.3
                )
                # Apply echo dampening if present
                if 'echo_dampening' in metrics['temporal']:
                    temporal_score *= metrics['temporal']['echo_dampening']
                    
                score += temporal_score * weights['temporal']
            
            # Strength Score (when included)
            if 'strength' in metrics:
                strength_score = (
                    metrics['strength'].get('current_strength', 0) * 0.6 +
                    metrics['strength'].get('relative_strength', 1.0) * 0.3 +
                    (1.0 - metrics['strength'].get('ghost_nodes_count', 0) / 10.0) * 0.1  # Normalize by max 10 ghosts
                )
                score += strength_score * weights['strength']
            
            return min(1.0, max(0.0, score))
            
        except Exception as e:
            self.logger.log_error(f"Score computation failed: {e}")
            return 0.0

    def traverse_connections(
        self,
        start_node: CognitiveMemoryNode,
        include_body: bool = True,
        max_depth: int = 3,
        min_weight: float = 0.3
    ) -> Dict[int, List[Tuple[Any, float, int]]]:
        """
        Traverse both cognitive and body connections, returning nodes grouped by depth.
        
        Args:
            start_node: The node to start traversal from.
            include_body: Whether to include body memory connections in traversal.
            max_depth: Maximum recursion depth for traversal.
            min_weight: Minimum connection strength to include a node.

        Returns:
            Dict[int, List[Tuple[Any, float, int]]]: Nodes grouped by distance from start.
                Each entry includes the node, connection weight, and depth.
        """

        results = defaultdict(list)
        visited = set([start_node.node_id])
        
        # Get cognitive connections
        for node_id, weight in start_node.connections.items():
            if weight >= min_weight:
                node = self._get_node_by_id(node_id)
                if node and not node.ghosted:
                    results[1].append((node, weight, 1))
                    visited.add(node_id)
        
        # Add body connections if requested
        if include_body:
            body_links = self.get_body_links(start_node.node_id)
            for body_id, weight in body_links:
                if weight >= min_weight:
                    body_node = self.body_memory._get_node_by_id(body_id)
                    if body_node and not body_node.ghosted:
                        results[1].append((body_node, weight, 1))
                        visited.add(f"body_{body_id}")
        
        # Recurse for deeper connections if needed
        if max_depth > 1:
            for node, weight, _ in results[1]:
                deeper = self.traverse_connections(
                    node,
                    include_body,
                    max_depth - 1,
                    min_weight
                )
                for depth, nodes in deeper.items():
                    results[depth + 1].extend(nodes)
        
        return results

    def get_recent_memories(
        self,
        count: int = 5,
        include_ghosted: bool = False,
        with_metrics: bool = False,
        time_window: Optional[float] = None
    ) -> Union[List[CognitiveMemoryNode], Tuple[List[CognitiveMemoryNode], List[Dict[str, Any]]]]:
        """
        Retrieve the most recent cognitive memories, optionally within a time window.
        Can return detailed metrics about the recent memories' significance.
        
        Args:
            count: Number of recent memories to retrieve
            include_ghosted: Whether to include ghosted memories
            with_metrics: Return detailed metrics about each memory
            time_window: Optional time window in seconds to limit search
            
        Returns:
            Either List[CognitiveMemoryNode] or Tuple of (nodes, metrics) if with_metrics=True
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
                current_state = self.internal_context.get_memory_context(is_cognitive=True)
                
                # Calculate metrics for each node
                node_metrics = []
                for node in recent_nodes:
                    metrics = self.calculate_retrieval_metrics(
                        target_node=node,
                        comparison_state=current_state,
                        detailed_metrics=True
                    )
                    node_metrics.append(metrics)
                    
                return recent_nodes, node_metrics
                
            return recent_nodes
            
        except Exception as e:
            self.logger.log_error(f"Failed to retrieve recent memories: {e}")
            return [] if not with_metrics else ([], [])

    # -------------------------------------------------------------------------
    # 5) Ghost & Decay Handling
    # -------------------------------------------------------------------------
    def update(self, decay_rate: float = 0.01) -> None:
        """
        Periodic update to decay cognitive memory nodes and handle ghost transitions.
        - decay_rate: base rate of decay
        (set properties)
        - ghost_threshold: if strength < ghost_threshold, we ghost the node
        - final_prune_threshold: if ghosted and strength < final_prune_threshold, we finalize prune
        - revive_threshold: if ghosted and strength > revive_threshold, we 'resurrect' the node
        """
        # We'll accumulate some node references for actions:
        ghost_candidates = []
        prune_candidates = []
        revive_candidates = []

        for node in self.nodes:
            try:
                # 1) Decay node
                decay_state = node.decay(
                    rate=decay_rate,
                    min_strength=self.ghost_threshold,
                    min_ghost_strength=self.final_prune_threshold
                )
                
                # 2) If node is ghosted but above revive_threshold, we plan to resurrect it
                if node.ghosted and node.strength > self.revive_threshold:
                    revive_candidates.append(node)
                    continue
                
                # 3) If node is not ghosted, but the decay state is 'ghost', we ghost it
                if not node.ghosted and decay_state == 'ghost':
                    ghost_candidates.append(node)
                    continue
                
                # 4) If ghosted and decay_state is 'final_prune', we remove it
                if node.ghosted and decay_state == 'final_prune':
                    prune_candidates.append(node)

            except Exception as e:
                self.logger.warning(f"Failed to update node {node.node_id}: {e}")
                continue

        # 1) Ghost any newly weak nodes
        for node in ghost_candidates:
            self._handle_weak_node(node)

        # 2) Prune final-weak ghost nodes
        for node in prune_candidates:
            self._convert_to_ghost_state(node)  # or finalize removing from DB

        # 3) Resurrect ghost nodes that overcame the threshold
        for node in revive_candidates:
            self._revive_ghost_node(node)

        # Possibly after all updates, do a partial re-check or connection update
        # If you prefer: self.update_all_connections(force=False)

    def _revive_ghost_node(self, node: CognitiveMemoryNode) -> None:
        """
        Revive a ghosted node back to active status.
        If the node had a parent_node_id, we can optionally treat
        this as a 'synthesis' event. For now, we simply remove
        the ghosted flag and re-establish adjacency.
        """
        try:
            if not node.ghosted:
                return  # Already active, no need

            # treat revival as synthesis if node had a parent
            if node.parent_node_id:
                self._record_synthesis_resurrection(node)

            # Clear out ghost flags
            node.ghosted = False

            # Re-check adjacency so it's recognized in the active network
            self.update_connections(node)
            node.last_accessed = time.time()

            # Save changes
            self._update_node(node)

            self.logger.info(f"Node {node.node_id} revived from ghost. Strength={node.strength:.2f}")

        except Exception as e:
            self.logger.log_error(f"Failed to revive node {node.node_id}: {e}")

    def _record_synthesis_resurrection(self, node: CognitiveMemoryNode) -> None:
        """
        Mark in 'synthesis_relations' that node was 'reborn' from its parent.
        For now, we record a relationship_type like 'resurrection'.
        """
        parent_id = node.parent_node_id
        if not parent_id:
            return  # No parent, do nothing

        with self._db_connection() as conn:
            cursor = conn.cursor()
            # Insert a row capturing that this node is effectively re-synthesized
            cursor.execute("""
                INSERT INTO synthesis_relations
                (synthesis_node_id, constituent_node_id, relationship_type, metadata)
                VALUES (?, ?, ?, ?)
            """, (
                node.node_id,
                parent_id,
                'resurrection',
                json.dumps({"timestamp": time.time()})
            ))
            conn.commit()

    def _handle_weak_node(self, node: CognitiveMemoryNode) -> None:
        """
        Handle a node that has decayed below threshold for 'active'.
        Typically, we either:
        - Mark it as ghosted if not ghosted yet
        - If it's ghosted but still too high to prune, do nothing
        - If it's strongly linked to another node, we might do a partial merge (future conflict logic)
        - Or we do a direct ghost conversion
        """
        if not node or not node.node_id:
            raise ValueError("Invalid node provided to _handle_weak_node")

        try:
            if node.ghosted:
                # If it's already ghosted, we might keep an eye on final prune logic or do nothing
                # (Conflict merges or 'synthesis merges' could go here if we detect contradictory states)
                return
            else:
                # Mark node as ghost
                node.ghosted = True
                # You might also set parent_node_id if merging, but for alpha, we just ghost it
                self.logger.info(f"Node {node.node_id} ghosted due to weakness (strength={node.strength:.2f})")

                # Persist changes
                self._update_node(node)

        except Exception as e:
            self.logger.log_error(f"Failed to handle weak node {node.node_id}: {e}")
            raise MemorySystemError(f"Weak node handling failed: {e}")

    def _convert_to_ghost_state(self, node: CognitiveMemoryNode) -> None:
        """
        Permanently remove a ghosted node from active tracking.
        Handles all relationships and data preservation before removal.
        
        Args:
            node: Node to convert to ghost state
            
        Raises:
            ValueError: If invalid node provided
            MemorySystemError: If conversion fails
        """
        if not node or not node.node_id:
            raise ValueError("Invalid node provided to _convert_to_ghost_state")
            
        try:
            # 1. Ensure node is marked as ghosted
            if not node.ghosted:
                node.ghosted = True
                self._update_node(node)

            # 2. Handle synthesis relations
            if node.parent_node_id:
                # Transfer synthesis relations to parent
                self._handle_synthesis_relations(node.node_id, 'transfer')
            else:
                # Clear synthesis relations if no parent
                self._handle_synthesis_relations(node.node_id, 'clear')
                
            # 3. Handle memory links
            if node.parent_node_id:
                # Transfer to parent with reduced strength
                self._handle_memory_links(node.node_id, 'transfer', node.parent_node_id)
            else:
                # Clear links if no parent
                self._handle_memory_links(node.node_id, 'clear')

            # 4. Preserve state in parent if exists
            if node.parent_node_id:
                parent = self._get_node_by_id(node.parent_node_id)
                if parent:
                    ghost_state = {
                        'timestamp': node.timestamp,
                        'text_content': node.text_content,
                        'raw_state': node.raw_state,
                        'processed_state': node.processed_state,
                        'strength': node.strength,
                        'derived_from': node.node_id,
                        'synthesis_source': bool(node.formation_source == 'synthesis'),
                        'ghost_nodes': node.ghost_nodes,  # Preserve any nested ghosts
                        'connections': node.connections   # Preserve connection history
                    }
                    parent.ghost_states.append(ghost_state)
                    self._update_node(parent)

            # 5. Remove from active memory and DB
            if node in self.nodes:
                self.nodes.remove(node)
            self._delete_node(node)

            self.logger.info(f"Node {node.node_id} fully pruned / converted to ghost state.")
            
        except Exception as e:
            self.logger.log_error(f"Failed to convert node {node.node_id} to ghost state: {e}")
            raise MemorySystemError(f"Ghost state conversion failed: {e}")

    def _evaluate_ghost_state(self, node: CognitiveMemoryNode) -> None:
        """
        Evaluate the node's strength to decide if we ghost, revive, or prune.
        Called after an echo or whenever node strength changes.
        """
        if not node or not node.node_id:
            return

        # 1) If node is ghosted
        if node.ghosted:
            # Check final prune
            if node.strength < self.final_prune_threshold:
                self._convert_to_ghost_state(node)
                return
            # Check revive
            if node.strength > self.revive_threshold:
                self._revive_ghost_node(node)
                return
            # else remain ghosted

        # 2) If node is not ghosted
        else:
            if node.strength < self.ghost_threshold:
                # Ghost the node
                self._handle_weak_node(node)

    def _handle_synthesis_relations(self, node_id: str, action: str) -> None:
        """
        Manage synthesis relations during node state changes.
        
        Args:
            node_id: ID of node being modified
            action: One of 'clear', 'transfer'
        """
        with self._db_connection() as conn:
            cursor = conn.cursor()
            
            if action == 'clear':
                cursor.execute("""
                    DELETE FROM synthesis_relations
                    WHERE synthesis_node_id = ? OR constituent_node_id = ?
                """, (node_id, node_id))
                
            elif action == 'transfer':
                # Get parent node ID
                cursor.execute("""
                    SELECT parent_node_id FROM cognitive_memory_nodes 
                    WHERE id = ?
                """, (node_id,))
                parent_id = cursor.fetchone()
                
                if parent_id:
                    parent_id = parent_id[0]
                    # Transfer relations to parent
                    cursor.execute("""
                        UPDATE synthesis_relations
                        SET synthesis_node_id = ?
                        WHERE synthesis_node_id = ?
                    """, (parent_id, node_id))
                    
                    cursor.execute("""
                        UPDATE synthesis_relations
                        SET constituent_node_id = ?
                        WHERE constituent_node_id = ?
                    """, (parent_id, node_id))
                    
            conn.commit()

    def _handle_memory_links(self, node_id: str, action: str, target_id: Optional[str] = None) -> None:
        """
        Manage memory links during node state changes.
        
        Args:
            node_id: ID of cognitive node being modified
            action: One of 'clear', 'transfer'
            target_id: Optional target node ID for transfers
        """
        with self._db_connection() as conn:
            cursor = conn.cursor()
            
            if action == 'clear':
                cursor.execute("""
                    DELETE FROM memory_links
                    WHERE cognitive_node_id = ?
                """, (node_id,))
                
            elif action == 'transfer' and target_id:
                # Transfer links with reduced strength
                cursor.execute("""
                    UPDATE memory_links
                    SET cognitive_node_id = ?,
                        link_strength = link_strength * 0.8
                    WHERE cognitive_node_id = ?
                """, (target_id, node_id))
                
            conn.commit()

    # Additional Safety Checks
    def _validate_node_state(self, node: CognitiveMemoryNode) -> bool:
        """
        Validate node's database consistency.
        Returns True if node state is valid.
        """
        with self._db_connection() as conn:
            cursor = conn.cursor()
            
            # Check main node record
            cursor.execute("SELECT id FROM cognitive_memory_nodes WHERE id=?", (node.node_id,))
            if not cursor.fetchone():
                return False
                
            # Check synthesis relations
            cursor.execute("""
                SELECT COUNT(*) FROM synthesis_relations 
                WHERE synthesis_node_id=? OR constituent_node_id=?
            """, (node.node_id, node.node_id))
            has_synthesis = cursor.fetchone()[0] > 0
            
            # Check memory links
            cursor.execute("""
                SELECT COUNT(*) FROM memory_links 
                WHERE cognitive_node_id=?
            """, (node.node_id,))
            has_links = cursor.fetchone()[0] > 0
            
            # Validate against node state
            if node.ghosted:
                # Ghosted nodes shouldn't have direct relations
                return not (has_synthesis or has_links)
            
            return True

    def repair_node_state(self, node: CognitiveMemoryNode) -> None:
        """
        Attempt to repair inconsistent node state.
        """
        if not self._validate_node_state(node):
            self.logger.warning(f"Repairing inconsistent state for node {node.node_id}")
            
            if node.ghosted:
                # Clear any lingering relations
                self._handle_synthesis_relations(node.node_id, 'clear')
                self._handle_memory_links(node.node_id, 'clear')
            else:
                # Rebuild relations from node data
                self.update_connections(node)
                if node.parent_node_id:
                    self._record_synthesis_resurrection(node)


    # -------------------------------------------------------------------------
    # 6) Echos
    # -------------------------------------------------------------------------
    def evaluate_echo(
            self, 
            node: CognitiveMemoryNode,
            comparison_state: Optional[Dict[str, Any]] = None, 
            query_text: Optional[str] = None, 
            query_embedding: Optional[List[float]] = None
        ) -> Tuple[CognitiveMemoryNode, float, List[Dict]]:
        """
        Evaluate node & related ghosts for echo potential using comprehensive retrieval metrics.
        Returns tuple of (selected_node, echo_intensity, evaluation_details).
        """
        if not node:
            self.logger.warning("evaluate_echo called with None node")
            return None, 0.0, []
            
        evaluations = []
        try:    
            if comparison_state:
                current_state = comparison_state
            else:
                current_state = self.internal_context.get_memory_context(is_cognitive=True)

            if not current_state:
                self.logger.warning("Failed to get cognitive memory context")
                return None, 0.0, []
            
            # Evaluate parent node using full retrieval metrics
            print("Debug - Evaluating parent node:", node.node_id)
            parent_metrics = self.calculate_retrieval_metrics(
                target_node=node,
                comparison_state=current_state,
                include_strength=True,
                detailed_metrics=True,
                query_text=query_text,
                query_embedding=query_embedding
            )
            
            print("Debug - Parent metrics:", parent_metrics)
            
            if parent_metrics and 'final_score' in parent_metrics:
                try:
                    # Validate state metrics structure
                    state_metrics = parent_metrics['component_metrics']['state']
                    print("Debug - Parent state metrics structure:", state_metrics)
                    
                    # Calculate state alignment with validation
                    valid_components = [
                        comp for comp in state_metrics.values()
                        if isinstance(comp, dict) and len(comp) > 0
                    ]
                    print("Debug - Parent valid state components:", valid_components)
                    
                    if not valid_components:
                        self.logger.warning("No valid state components found for alignment calculation")
                        state_alignment = 0.0
                    else:
                        component_sums = []
                        for comp in valid_components:
                            numeric_values = [v for v in comp.values() if isinstance(v, (int, float))]
                            if numeric_values:  # Only calculate if we have valid numbers
                                component_sums.append(sum(numeric_values) / len(comp))
                        
                        state_alignment = (sum(component_sums) / len(valid_components)) if component_sums else 0.0
                    
                    print("Debug - Parent state alignment calculated:", state_alignment)
                    
                    # Use components to calculate echo potential
                    echo_components = {
                        'semantic_match': (
                            parent_metrics['component_metrics']['semantic'].get('embedding_similarity', 0) * 0.6 +
                            parent_metrics['component_metrics']['semantic'].get('cognitive_patterns', 0) * 0.4
                        ),
                        'emotional_resonance': (
                            parent_metrics['component_metrics']['emotional'].get('vector_similarity', 0) * 0.5 +
                            parent_metrics['component_metrics']['emotional'].get('valence_shift', 0) * 0.5
                        ),
                        'state_alignment': state_alignment
                    }
                    
                    print("Debug - Parent echo components calculated:", echo_components)
                    
                    echo_intensity = (
                        echo_components['semantic_match'] * 0.3 +
                        echo_components['emotional_resonance'] * 0.4 +
                        echo_components['state_alignment'] * 0.3
                    )
                    
                    evaluations.append({
                        "node": node,
                        "echo": {
                            "intensity": echo_intensity,
                            "components": echo_components
                        },
                        "relevance": parent_metrics['final_score'],
                        "metrics": parent_metrics,
                        "is_ghost": False
                    })
                except Exception as e:
                    self.logger.error(f"Failed to calculate parent echo components: {str(e)}")
                    print("Debug - Parent error details:", e.__class__.__name__, str(e))
                    return None, 0.0, []

            # Evaluate ghost nodes using same metrics system
            print(f"Debug - Evaluating {len(node.ghost_nodes)} ghost nodes")
            for ghost in node.ghost_nodes:
                try:
                    print(f"Debug - Processing ghost node: {ghost.get('node_id')}")
                    # Create temporary node structure for ghost evaluation
                    ghost_node = CognitiveMemoryNode(
                        node_id=ghost['node_id'],
                        text_content=ghost['text_content'],
                        embedding=ghost.get('embedding', self._generate_embedding(ghost['text_content'])),
                        raw_state=ghost['raw_state'],
                        processed_state=ghost['processed_state'],
                        timestamp=ghost['timestamp'],
                        strength=ghost['strength'],
                        ghosted=True,
                        last_accessed=None,
                        last_echo_time=None,
                        echo_dampening=1.0
                    )
                    
                    ghost_metrics = self.calculate_retrieval_metrics(
                        target_node=ghost_node,
                        comparison_state=current_state,
                        include_strength=True,
                        detailed_metrics=True,
                        query_text=query_text,
                        query_embedding=query_embedding
                    )
                    
                    print("Debug - Ghost metrics:", ghost_metrics)
                    
                    if ghost_metrics and 'final_score' in ghost_metrics:
                        # Validate ghost state metrics structure
                        ghost_state_metrics = ghost_metrics['component_metrics']['state']
                        print("Debug - Ghost state metrics structure:", ghost_state_metrics)
                        
                        # Calculate ghost state alignment with validation
                        valid_ghost_components = [
                            comp for comp in ghost_state_metrics.values()
                            if isinstance(comp, dict) and len(comp) > 0
                        ]
                        print("Debug - Ghost valid state components:", valid_ghost_components)
                        
                        if not valid_ghost_components:
                            self.logger.warning(f"No valid state components found for ghost {ghost['node_id']}")
                            ghost_state_alignment = 0.0
                        else:
                            ghost_component_sums = []
                            for comp in valid_ghost_components:
                                numeric_values = [v for v in comp.values() if isinstance(v, (int, float))]
                                if numeric_values:
                                    ghost_component_sums.append(sum(numeric_values) / len(comp))
                            
                            ghost_state_alignment = (sum(ghost_component_sums) / len(valid_ghost_components)) if ghost_component_sums else 0.0
                        
                        print("Debug - Ghost state alignment calculated:", ghost_state_alignment)
                        
                        # Calculate ghost echo components
                        ghost_echo_components = {
                            'semantic_match': (
                                ghost_metrics['component_metrics']['semantic'].get('embedding_similarity', 0) * 0.6 +
                                ghost_metrics['component_metrics']['semantic'].get('cognitive_patterns', 0) * 0.4
                            ),
                            'emotional_resonance': (
                                ghost_metrics['component_metrics']['emotional'].get('vector_similarity', 0) * 0.5 +
                                ghost_metrics['component_metrics']['emotional'].get('valence_shift', 0) * 0.5
                            ),
                            'state_alignment': ghost_state_alignment
                        }
                        
                        print("Debug - Ghost echo components calculated:", ghost_echo_components)
                        
                        ghost_echo_intensity = (
                            ghost_echo_components['semantic_match'] * 0.3 +
                            ghost_echo_components['emotional_resonance'] * 0.4 +
                            ghost_echo_components['state_alignment'] * 0.3
                        )
                        
                        # Apply ghost bonus multiplier
                        ghost_echo_intensity *= 1.25
                        
                        evaluations.append({
                            "node": ghost_node,
                            "echo": {
                                "intensity": ghost_echo_intensity,
                                "components": ghost_echo_components
                            },
                            "relevance": ghost_metrics['final_score'] * 1.25,  # Ghost relevance bonus
                            "metrics": ghost_metrics,
                            "is_ghost": True
                        })
                except Exception as e:
                    self.logger.error(f"Failed to calculate ghost echo components: {str(e)}")
                    print("Debug - Ghost error details:", e.__class__.__name__, str(e))
                    continue
            
            if not evaluations:
                self.logger.warning("No valid evaluations generated")
                return None, 0.0, []
                
            # Select strongest echo based on combined intensity and relevance
            selected = max(evaluations, key=lambda x: x["echo"]["intensity"] * x["relevance"])
            final_intensity = selected["echo"]["intensity"] * selected["relevance"]
            
            print("Debug - Final evaluation selected:", {
                "node_id": selected["node"].node_id,
                "intensity": final_intensity,
                "is_ghost": selected["is_ghost"]
            })
            
            return selected["node"], final_intensity, evaluations
            
        except Exception as e:
            self.logger.error(f"Unexpected error in evaluate_echo: {str(e)}")
            print("Debug - Critical error:", e.__class__.__name__, str(e))
            return None, 0.0, []

    async def trigger_echo(
            self,
            node: CognitiveMemoryNode, 
            intensity: float, 
            comparison_state: Optional[Dict[str, Any]] = None, 
            query_text: Optional[str] = None, 
            query_embedding: Optional[List[float]] = None
        ) -> None:
        """
        Orchestrate echo activation and handle strength adjustments based on 
        comprehensive evaluation metrics.
        """
        if not comparison_state:
            comparison_state = self.internal_context.get_memory_context(is_cognitive=True)
        if not query_text:
            query_text = comparison_state['processed_state']['cognitive']
        if not query_embedding:
            query_embedding = self._generate_embedding(query_text)
            
        selected_node, final_intensity, details = self.evaluate_echo(node, comparison_state, query_text, query_embedding)
        if not selected_node:
            return
            
        node.last_accessed = time.time()
        echo_window = 30  # seconds
        
        if isinstance(selected_node, CognitiveMemoryNode):
            # Calculate time-based dampening
            time_since_echo = time.time() - (selected_node.last_echo_time or 0)
            
            if time_since_echo > echo_window:
                selected_node.echo_dampening = 1.0
            else:
                # Progressive dampening with floor
                selected_node.echo_dampening = max(0.15, selected_node.echo_dampening * 0.85)
            
            # Apply dampening to final intensity
            final_intensity *= selected_node.echo_dampening
            
            # Calculate strength boost based on echo components
            selected_eval = next(e for e in details if e["node"] == selected_node)
            echo_components = selected_eval["echo"]["components"]
            
            strength_boost = (
                echo_components['semantic_match'] * 0.3 +
                echo_components['emotional_resonance'] * 0.5 +
                echo_components['state_alignment'] * 0.2
            ) * final_intensity * 0.2
            
            # Apply strength adjustments
            if selected_node != node:
                # Ghost node resurrection boost
                ghost_idx = next(i for i, g in enumerate(node.ghost_nodes) 
                            if g["node_id"] == selected_node.node_id)
                node.ghost_nodes[ghost_idx]["strength"] = min(
                    1.0, 
                    node.ghost_nodes[ghost_idx]["strength"] + (strength_boost * 1.5)
                )
            else:
                # Normal node strength boost
                node.strength = min(1.0, node.strength + strength_boost)
            
            # Trigger echo effect
            selected_node.activate_echo(final_intensity)
            selected_node.last_echo_time = time.time()
            
            # Handle body memory effects
            for body_id, link_strength in self.get_body_links(selected_node.node_id):
                body_node = self.body_memory._get_node_by_id(body_id)
                if body_node:
                    # Scale body boost by link strength and echo intensity
                    body_boost = strength_boost * link_strength * 0.8
                    body_node.strength = min(1.0, body_node.strength + body_boost)
            
            self.logger.info(
                f"Echo triggered for node {selected_node.node_id} off {node.node_id} "
                f"with intensity {final_intensity:.2f} (dampening={selected_node.echo_dampening:.2f})"
            )
            
            # Re-evaluate node state after echo
            if selected_node == node:
                self.jitter(node)

    def jitter(self, node: CognitiveMemoryNode):
        """Re-evaluate node state and connections after echo effect."""
        self._evaluate_ghost_state(node)
        self.update_connections(node)
        self.detect_conflicts(node)
        self._update_node(node)
        
    # -------------------------------------------------------------------------
    # 7) Utility & Internals
    # -------------------------------------------------------------------------
    def detect_conflicts(self, node: CognitiveMemoryNode) -> None:
        """
        Detect semantic/emotional/state conflicts with strongly connected neighbors
        using the retrieval metric system. Dispatches detailed conflict info to
        ExoProcessor for LLM synthesis.
        """
        for other_id, weight in node.connections.items():
            if weight < 0.7:  # High connection strength threshold
                continue
            other = self._get_node_by_id(other_id)
            if not other or other.ghosted:
                continue

            conflicts, metrics = self._is_conflicting(node, other)
            if conflicts:
                # We found conflicts. Dispatch event with detailed metrics
                global_event_dispatcher.dispatch_event(Event(
                    "memory:conflict_detected",
                    {
                        "node_a_id": node.node_id,
                        "node_b_id": other.node_id,
                        "conflicts": conflicts,
                        "similarity_metrics": metrics
                    }
                ))
                # Handle one conflict at a time for alpha
                return

    def _is_conflicting(self, nodeA: CognitiveMemoryNode, nodeB: CognitiveMemoryNode) -> Tuple[Optional[Dict], Optional[Dict]]:
        """
        Detailed conflict analysis using retrieval metrics system.
        Returns tuple of (conflicts dict, similarity metrics) if conflicts found,
        otherwise (None, None).
        
        Detects conflicts across:
        - Semantic/factual contradictions
        - Emotional valence mismatches  
        - State inconsistencies
        - Behavioral transitions
        """
        try:
            # Get full retrieval metrics comparing the nodes
            comparison = self.calculate_retrieval_metrics(
                target_node=nodeA,
                comparison_state={
                    'raw_state': nodeB.raw_state,
                    'processed_state': nodeB.processed_state
                },
                query_text=nodeB.text_content,
                query_embedding=nodeB.embedding,
                detailed_metrics=True
            )

            if not isinstance(comparison, dict):
                return None, None

            metrics = comparison['component_metrics']
            conflicts = {}

            # 1. Check semantic conflicts
            semantic = metrics.get('semantic', {})
            if semantic.get('embedding_similarity', 0) > 0.8:
                text_relevance = semantic.get('text_relevance', 0)
                if text_relevance < 0.4:  # Factual mismatch despite similarity
                    conflicts['semantic'] = {
                        'type': 'factual_mismatch',
                        'similarity': semantic.get('embedding_similarity'),
                        'relevance': text_relevance,
                        'patterns': semantic.get('cognitive_patterns', 0)
                    }

            # 2. Check emotional conflicts
            emotional = metrics.get('emotional', {})
            if emotional:
                valence_shift = emotional.get('valence_shift', 0)
                if valence_shift > 0.6:  # Major emotional mismatch
                    conflicts['emotional'] = {
                        'type': 'valence_conflict',
                        'shift': valence_shift,
                        'intensity_delta': abs(
                            emotional.get('node_intensity', 0) - 
                            emotional.get('comparison_intensity', 0)
                        )
                    }

            # 3. Check state conflicts
            state = metrics.get('state', {})
            if state:
                state_conflicts = {}
                
                # Need states
                needs = state.get('needs', {})
                if needs and needs.get('state_shifts', 0) > 0.7:
                    state_conflicts['needs'] = {
                        'shift': needs.get('state_shifts'),
                        'urgency_delta': needs.get('urgency_levels', 0)
                    }
                
                # Behavioral transitions
                behavior = state.get('behavior', {})
                if behavior and not behavior.get('matching', True):
                    state_conflicts['behavior'] = {
                        'transition': behavior.get('transition_significance', 0)
                    }
                
                # Mood mismatches
                mood = state.get('mood', {})
                if mood and mood.get('intensity_delta', 0) > 0.6:
                    state_conflicts['mood'] = {
                        'intensity_delta': mood.get('intensity_delta'),
                        'similarity': mood.get('similarity', 0)
                    }
                
                if state_conflicts:
                    conflicts['state'] = state_conflicts

            # Return results if we found conflicts
            if conflicts:
                return conflicts, comparison
                
            return None, None

        except Exception as e:
            self.logger.log_error(f"Failed to check conflicts: {e}")
            return None, None

    def on_conflict_resolved(self, event: Event):
        # received synthesis back, send it
        node_a_id = event.data["node_a_id"]
        node_b_id = event.data["node_b_id"]
        synthesis_text = event.data["synthesis_text"]

        self._finalize_conflict_merge(node_a_id, node_b_id, synthesis_text)

    async def _finalize_conflict_merge(self, node_a_id: str, node_b_id: str, synthesis_text: str):
        """
        Finalize a conflict merge by creating a synthesized node and properly handling the relationships.
        Uses the form_memory system to ensure consistent node creation and processing.
        """
        nodeA = self._get_node_by_id(node_a_id) 
        nodeB = self._get_node_by_id(node_b_id)
        if not nodeA or not nodeB:
            return

        try:
            # Create blended state for synthesis node using full state structure
            raw_state = self._blend_states(nodeA.raw_state, nodeB.raw_state)
            processed_state = self._blend_states(nodeA.processed_state, nodeB.processed_state)

            # Calculate contributed strength - nodes give up 30% each
            strength_from_a = nodeA.strength * 0.3
            strength_from_b = nodeB.strength * 0.3
            
            # Reduce child node strengths
            nodeA.strength -= strength_from_a
            nodeB.strength -= strength_from_b
            
            # Create synthesis node through form_memory
            synthesis_node_id = await self.form_memory(
                text_content=synthesis_text,
                raw_state=raw_state,
                processed_state=processed_state,
                formation_source="synthesis"
            )

            if not synthesis_node_id:
                raise MemorySystemError("Failed to create synthesis node")

            synthesis_node = self._get_node_by_id(synthesis_node_id)

            # Record synthesis relationships with conflict metrics
            with self._db_connection() as conn:
                cursor = conn.cursor()
                conflict_details = self._is_conflicting(nodeA, nodeB)
                
                for constituent_id in [node_a_id, node_b_id]:
                    cursor.execute("""
                        INSERT INTO synthesis_relations (
                            synthesis_node_id, constituent_node_id, 
                            relationship_type, metadata
                        )
                        VALUES (?, ?, ?, ?)
                    """, (
                        synthesis_node_id,
                        constituent_id,
                        'conflict_synthesis',
                        json.dumps({
                            "timestamp": time.time(),
                            "synthesis_type": "conflict_resolution",
                            "conflict_metrics": conflict_details[0] if conflict_details else None,
                            "similarity_metrics": conflict_details[1] if conflict_details else None
                        })
                    ))
                conn.commit()

            # Transfer and merge memory links with strength preservation
            self.transfer_body_links(node_a_id, synthesis_node_id, 0.9)
            self.transfer_body_links(node_b_id, synthesis_node_id, 0.9)

            # Merge old nodes into synthesis node
            nodeA.merge_into_parent(synthesis_node)
            nodeB.merge_into_parent(synthesis_node)

            # Calculate synthesis node strength using 1.5:1 energy gain ratio
            # Base strength comes from form_memory calculation
            synthesis_node.strength = min(1.0, (
                strength_from_a + 
                strength_from_b + 
                (strength_from_a + strength_from_b) * 0.5 + # 50% energy gain
                synthesis_node.strength * 0.2  # Keep some initial strength
            ))

            # Update and persist changes
            self._update_node(nodeA)  # Save reduced strengths
            self._update_node(nodeB)
            self.update_connections(synthesis_node)  # Establish network position
            self._update_node(synthesis_node)

            self.logger.info(
                f"Conflict synthesis complete: created node {synthesis_node_id} "
                f"from {node_a_id} and {node_b_id} with strength {synthesis_node.strength:.2f}"
            )

            # Dispatch synthesis completion event
            global_event_dispatcher.dispatch_event(Event(
                "memory:synthesis_complete",
                {
                    "synthesis_node_id": synthesis_node_id,
                    "constituent_nodes": [node_a_id, node_b_id],
                    "synthesis_type": "conflict_resolution",
                    "final_strength": synthesis_node.strength
                }
            ))

        except Exception as e:
            self.logger.log_error(f"Failed to finalize conflict merge: {e}")
            raise MemorySystemError(f"Conflict synthesis failed: {e}")

    def _blend_states(self, state_a: Dict, state_b: Dict) -> Dict:
        """
        Helper to blend two state dictionaries for synthesis node creation.
        Uses knowledge of state structure from memory system for proper merging.
        """
        blended = {}
        
        try:
            # Emotional vectors - average with potential intensity preservation
            if 'emotional_vectors' in state_a and 'emotional_vectors' in state_b:
                vectors_a = state_a['emotional_vectors']
                vectors_b = state_b['emotional_vectors']
                
                # Preserve strongest emotions while averaging others
                blended['emotional_vectors'] = {}
                all_emotions = set(vectors_a) | set(vectors_b)
                
                for emotion in all_emotions:
                    val_a = vectors_a.get(emotion, 0)
                    val_b = vectors_b.get(emotion, 0)
                    # Keep stronger emotion if significant difference
                    if abs(val_a - val_b) > 0.3:
                        blended['emotional_vectors'][emotion] = max(val_a, val_b)
                    else:
                        blended['emotional_vectors'][emotion] = (val_a + val_b) * 0.5

            # Needs - preserve highest urgency needs
            if 'needs' in state_a and 'needs' in state_b:
                needs_a = state_a['needs']
                needs_b = state_b['needs']
                
                blended['needs'] = {}
                all_needs = set(needs_a) | set(needs_b)
                
                for need in all_needs:
                    urgency_a = needs_a.get(need, {}).get('urgency', 0)
                    urgency_b = needs_b.get(need, {}).get('urgency', 0)
                    
                    # Keep complete need state of highest urgency
                    if urgency_a >= urgency_b:
                        blended['needs'][need] = needs_a[need]
                    else:
                        blended['needs'][need] = needs_b[need]

            # Behavioral state - take most recent unless significant conflict
            if 'behavior' in state_a and 'behavior' in state_b:
                if state_a.get('timestamp', 0) > state_b.get('timestamp', 0):
                    blended['behavior'] = state_a['behavior']
                else:
                    blended['behavior'] = state_b['behavior']

            # Cognitive state - merge attention and processing states
            if 'cognitive' in state_a and 'cognitive' in state_b:
                cog_a = state_a['cognitive']
                cog_b = state_b['cognitive']
                
                blended['cognitive'] = {
                    'attention': {
                        k: max(cog_a.get('attention', {}).get(k, 0),
                              cog_b.get('attention', {}).get(k, 0))
                        for k in set(cog_a.get('attention', {})) | 
                                 set(cog_b.get('attention', {}))
                    },
                    'processing_depth': max(
                        cog_a.get('processing_depth', 0),
                        cog_b.get('processing_depth', 0)
                    )
                }

            return blended

        except Exception as e:
            self.logger.log_error(f"Failed to blend states: {e}")
            return {}

    async def form_memory(
        self,
        text_content: str,
        embedding: Optional[List[float]] = None,
        raw_state: Optional[Dict[str, Any]] = None,
        processed_state: Optional[Dict[str, Any]] = None,
        body_node_id: Optional[str] = None,
        formation_source: Optional[str] = None
    ) -> str:
        """
        Create and persist a new cognitive memory node. Link to a body node if possible.
        If no body_node_id is provided, try to find or create one from body_memory.
        If states not provided, fetches current state from context.

        Args:
            text_content: LLM's semantic interpretation
            embedding: Vector representation for semantic search (optional)
            raw_state: Complete system state at formation (optional)
            processed_state: Processed state data (optional) 
            body_node_id: ID of existing body memory node to link to (optional)
            formation_source: Event/trigger that created this memory (optional but i want to make it not optional and get more data)

        Returns: CognitiveMemoryNode: The formed memory node id
        """
        try:
            # Fetch states from context if not provided
            if raw_state is None or processed_state is None:
                current_state = self.internal_context.get_memory_context(is_cognitive=True)
                raw_state = raw_state or current_state['raw_state']
                processed_state = processed_state or current_state['processed_state']

            # Attempt to find or create body node if none given
            if not body_node_id:
                body_node_id = self._find_recent_body_node()
                if not body_node_id:
                    # Create new body experience
                    body_node_id = self._create_body_experience()

            # if no embedding provided, generate one
            if not embedding:
                embedding = self._generate_embedding(text_content)

            # Calculate strength 
            strength = self._calculate_initial_strength(text_content, raw_state, processed_state, embedding)
            
            node = CognitiveMemoryNode(
                timestamp=time.time(),
                text_content=text_content,
                embedding=embedding,
                raw_state=raw_state,
                processed_state=processed_state,
                strength=strength,
                formation_source=formation_source
            )
            
            # Persist in DB
            self.nodes.append(node)
            self._persist_node(node)

            # Form initial connections
            self._form_initial_connections(node)

            # Evaluate for new conflicts
            self.detect_conflicts(node)
            
            # Insert body link if we have it
            if body_node_id:
                self.add_body_link(str(node.node_id), body_node_id, 1.0)
            
            # Dispatch event
            global_event_dispatcher.dispatch_event(Event(
                "memory:node_created",
                {"node_type": "cognitive", "node_id": node.node_id, "content": text_content}
            ))

            return str(node.node_id)  # Return the node ID
            
        except Exception as e:
            import traceback
            print("Debug - Full error in memory formation:")
            print(traceback.format_exc()) 
            self.logger.log_error(f"Failed to form cognitive memory: {e}")
            raise MemorySystemError(f"Memory formation failed: {e}")

    def _find_recent_body_node(self, within_sec: float = 5.0) -> Optional[str]:
        """
        Attempt to find a body node created in the last X seconds.
        Helps prevent double-creation from the same event.
        """
        current_time = time.time()

        recent = [
            n for n in self.body_memory.nodes 
            if (current_time - n.timestamp) <= within_sec and not n.ghosted
        ]
        if recent:
            return str(recent[0].node_id)
        return None

    def _create_body_experience(self) -> Optional[str]:
        """
        If no suitable body node was found, create a new BodyMemory node and return its ID.
        """
        try:
            # need to consider what metadata we want the body memory to keep on each formation
            # for now making one and returning is fine
            return self.body_memory.form_memory({'source': 'cognitive_formation'})
        except Exception as e:
            self.logger.log_error(f"Failed to create body experience: {e}")
            return None

    def _calculate_initial_strength(
        self,
        text_content: str,
        raw_state: Dict[str, Any],
        processed_state: Dict[str, Any],
        embedding: Optional[List[float]] = None
    ) -> float:
        """
        Calculate initial memory strength using retrieval metrics before node creation.
        Creates a temporary node structure to leverage retrieval analysis.
        
        Args:
            text_content: LLM's semantic interpretation
            raw_state: Complete system state at formation
            processed_state: Processed state data
            
        Returns:
            float: Initial strength value [0.1, 1.0]
        """
        try:
            # Generate embedding for semantic analysis
            if not embedding:
                embedding = self._generate_embedding(text_content)
            
            # Create temporary node structure
            temp_node = CognitiveMemoryNode(
                text_content=text_content,
                embedding=embedding,
                raw_state=raw_state,
                processed_state=processed_state,
                timestamp=time.time(),
                strength=1.0,  # Start with full strength
                ghosted=False,
                ghost_nodes=[],
                last_accessed=None,
                last_echo_time=None,
                echo_dampening=1.0
            )
            
            # Get recent nodes for comparative analysis
            recent_nodes = sorted(
                [n for n in self.nodes if not n.ghosted],
                key=lambda x: x.timestamp,
                reverse=True
            )[:5]
            
            # Calculate metrics against recent nodes
            significance_scores = []
            
            for node in recent_nodes:
                # Compare against each recent node
                metrics = self.calculate_retrieval_metrics(
                    target_node=temp_node,
                    comparison_state={
                        'raw_state': node.raw_state,
                        'processed_state': node.processed_state
                    },
                    include_strength=False,  # Ignore strength component
                    detailed_metrics=True    # Get detailed breakdown
                )
                
                if isinstance(metrics, dict) and 'component_metrics' in metrics:
                    significance_scores.append(metrics['component_metrics'])
            
            # If we have no comparison nodes, evaluate against current state
            if not significance_scores:
                current_state = self.internal_context.get_memory_context(is_cognitive=True)
                metrics = self.calculate_retrieval_metrics(
                    target_node=temp_node,
                    comparison_state=current_state,
                    include_strength=False,
                    detailed_metrics=True
                )
                if isinstance(metrics, dict) and 'component_metrics' in metrics:
                    significance_scores.append(metrics['component_metrics'])
            
            if not significance_scores:
                return 0.1  # Minimum baseline if we can't calculate
                
            # Analyze significance patterns
            strength_components = {
                'novelty': self._calculate_novelty_score(significance_scores),
                'emotional_impact': self._calculate_emotional_impact(significance_scores),
                'state_significance': self._calculate_state_significance(significance_scores)
            }
            
            # Calculate final strength
            return self._compute_initial_strength(strength_components)
            
        except Exception as e:
            self.logger.log_error(f"Failed to calculate initial strength: {e}")
            return 0.1

    def _calculate_novelty_score(self, significance_scores: List[Dict]) -> float:
        """
        Calculate novelty based on semantic and state differences.
        High differences indicate novel experiences worth remembering strongly.
        """
        semantic_similarities = []
        state_differences = []
        
        for metrics in significance_scores:
            # Get semantic similarities
            if 'semantic' in metrics:
                sem_metrics = metrics['semantic']
                if 'embedding_similarity' in sem_metrics:
                    semantic_similarities.append(sem_metrics['embedding_similarity'])
            
            # Get state differences
            if 'state' in metrics:
                state_metrics = metrics['state']
                # Average difference across state components
                diff_scores = []
                
                for component in ['needs', 'behavior', 'mood', 'cognitive']:
                    if component in state_metrics:
                        component_metrics = state_metrics[component]
                        if isinstance(component_metrics, dict):
                            diff_scores.append(
                                1.0 - sum(v for v in component_metrics.values() 
                                        if isinstance(v, (int, float))) / len(component_metrics)
                            )
                
                if diff_scores:
                    state_differences.append(sum(diff_scores) / len(diff_scores))
        
        # Calculate novelty:
        # - Low semantic similarity = more novel
        # - High state differences = more novel
        if semantic_similarities:
            semantic_novelty = 1.0 - (sum(semantic_similarities) / len(semantic_similarities))
        else:
            semantic_novelty = 0.5  # Neutral if no comparisons
            
        if state_differences:
            state_novelty = sum(state_differences) / len(state_differences)
        else:
            state_novelty = 0.5  # Neutral if no comparisons
            
        # Combine with emphasis on semantic novelty
        return (semantic_novelty * 0.6) + (state_novelty * 0.4)

    def _calculate_emotional_impact(self, significance_scores: List[Dict]) -> float:
        """
        Calculate emotional significance from emotional metrics.
        High emotional intensity or significant shifts indicate important memories.
        """
        emotional_scores = []
        
        for metrics in significance_scores:
            if 'emotional' in metrics:
                emotional_metrics = metrics['emotional']
                
                # Consider intensity
                intensity = emotional_metrics.get('node_intensity', 0)
                
                # Consider valence shifts
                valence_shift = emotional_metrics.get('valence_shift', 0)
                
                # Consider emotional complexity
                complexity = emotional_metrics.get('emotional_complexity', 0) / 5  # Normalize assuming max 5 emotions
                
                # Combine factors
                emotional_score = (
                    intensity * 0.4 +
                    valence_shift * 0.4 +
                    complexity * 0.2
                )
                emotional_scores.append(emotional_score)
        
        return max(emotional_scores) if emotional_scores else 0.5

    def _calculate_state_significance(self, significance_scores: List[Dict]) -> float:
        """
        Calculate overall state significance.
        Major state changes or extreme states indicate important memories.
        """
        state_scores = []
        
        for metrics in significance_scores:
            if 'state' in metrics:
                state_metrics = metrics['state']
                
                component_scores = []
                
                # Need state extremes/changes
                if 'needs' in state_metrics:
                    need_metrics = state_metrics['needs']
                    urgency = need_metrics.get('urgency_levels', 0)
                    shifts = need_metrics.get('state_shifts', 0)
                    component_scores.append((urgency * 0.6) + (shifts * 0.4))
                
                # Behavior transitions
                if 'behavior' in state_metrics:
                    behavior_metrics = state_metrics['behavior']
                    transition = behavior_metrics.get('transition_significance', 0)
                    component_scores.append(transition)
                
                # Mood shifts
                if 'mood' in state_metrics:
                    mood_metrics = state_metrics['mood']
                    intensity = mood_metrics.get('intensity_delta', 0)
                    component_scores.append(intensity)
                
                # Cognitive processing
                if 'cognitive' in state_metrics:
                    cognitive_metrics = state_metrics['cognitive']
                    depth = cognitive_metrics.get('processing_depth', 0)
                    component_scores.append(depth)
                
                if component_scores:
                    state_scores.append(sum(component_scores) / len(component_scores))
        
        return max(state_scores) if state_scores else 0.5

    def _compute_initial_strength(self, components: Dict[str, float]) -> float:
        """
        Compute final initial strength from component scores.
        
        Args:
            components: Dict with 'novelty', 'emotional_impact', and 'state_significance'
            
        Returns:
            float: Initial strength value [0.1, 1.0]
        """
        # Weight the components
        weighted_score = (
            components['novelty'] * 0.4 +                # Novel experiences
            components['emotional_impact'] * 0.35 +      # Emotional significance  
            components['state_significance'] * 0.25      # State changes
        )
        
        # Apply sigmoid scaling for more natural distribution
        scaled_strength = 2 / (1 + math.exp(-3 * (weighted_score - 0.5))) - 1
        
        # Ensure minimum strength
        return max(0.1, min(1.0, scaled_strength))

    def _form_initial_connections(self, node: CognitiveMemoryNode) -> None:
        """
        Form initial connections for a new node using retrieval metrics and temporal proximity.
        Applies softmax scaling for connection weights unless extremely similar.
        
        Args:
            node: Newly created node to connect to the network
        """
        try:
            # Get active nodes for potential connections, including recently ghosted nodes
            # that might be candidates for resurrection
            active_nodes = [
                n for n in self.nodes 
                if (not n.ghosted or 
                    (n.ghosted and n.strength > self.final_prune_threshold))
                and n.node_id != node.node_id
            ]
            
            if not active_nodes:
                return
                
            connection_scores = []
            
            for other_node in active_nodes:
                # Get detailed retrieval metrics
                metrics = self.calculate_retrieval_metrics(
                    target_node=node,
                    comparison_state={
                        'raw_state': other_node.raw_state,
                        'processed_state': other_node.processed_state
                    },
                    include_strength=False,
                    detailed_metrics=True
                )
                
                if isinstance(metrics, dict) and 'component_metrics' in metrics:
                    base_score = metrics['final_score']
                    
                    # Calculate temporal bonus (decaying exponential)
                    time_diff = abs(node.timestamp - other_node.timestamp)
                    temporal_bonus = math.exp(-time_diff / (3600 * 24))  # 1-day decay
                    
                    # Adjust score for ghosted nodes
                    if other_node.ghosted:
                        # Reduce connection strength to ghosted nodes
                        base_score *= 0.7
                        # But boost if it's a potential resurrection candidate
                        if other_node.strength > self.revive_threshold * 0.8:
                            base_score *= 1.2
                    
                    final_score = (base_score * 0.7) + (temporal_bonus * 0.3)
                    
                    if final_score > 0.2:  # Minimum threshold for connection
                        connection_scores.append((other_node, final_score, metrics))
            
            if not connection_scores:
                return
                
            # Sort by score
            connection_scores.sort(key=lambda x: x[1], reverse=True)
            
            # Apply softmax scaling unless extremely similar
            scores_for_softmax = [score for _, score, _ in connection_scores]
            softmax_scores = self._softmax(scores_for_softmax)
            
            # Create connections
            for i, (other_node, raw_score, metrics) in enumerate(connection_scores):
                # If extremely similar (>0.9), keep raw score
                # Otherwise use softmax scaling
                weight = raw_score if raw_score > 0.9 else softmax_scores[i]
                
                # Handle connection formation
                self._establish_connection(node, other_node, weight)

        except Exception as e:
            self.logger.log_error(f"Failed to form initial connections: {e}")
    
    def _softmax(self, scores: List[float]) -> List[float]:
        """Calculate softmax of input scores."""
        if not scores:
            return []
            
        exp_scores = [math.exp(s) for s in scores]
        sum_exp = sum(exp_scores)
        return [e / sum_exp for e in exp_scores]

    def update_connections(self, node: CognitiveMemoryNode) -> None:
        """
        Update a node's connections based on current retrieval metrics.
        Handles ghost connections and potential resurrections.
        
        Args:
            node: Node to update connections for
        """
        if node.ghosted:
            return
            
        try:
            # Get all potential connection candidates
            candidates = [
                n for n in self.nodes 
                if (not n.ghosted or 
                    (n.ghosted and n.strength > self.final_prune_threshold))
                and n.node_id != node.node_id
            ]
            
            old_connections = node.connections.copy()
            new_connections = {}
            
            # Track potential resurrections from ghost connections
            potential_resurrections = set()
            
            for other in candidates:
                metrics = self.calculate_retrieval_metrics(
                    target_node=node,
                    comparison_state={
                        'raw_state': other.raw_state,
                        'processed_state': other.processed_state
                    },
                    include_strength=False,
                    detailed_metrics=True
                )
                
                if isinstance(metrics, dict) and 'component_metrics' in metrics:
                    weight = metrics['final_score']
                    
                    # Check ghost connections for potential resurrection
                    if hasattr(node, 'ghost_connections'):
                        ghost_conn = next(
                            (g for g in node.ghost_connections 
                            if g['node_id'] == other.node_id),
                            None
                        )
                        if ghost_conn and weight > self.revive_threshold:
                            potential_resurrections.add(other.node_id)
                            weight *= 1.2  # Boost weight for resurrection
                    
                    # Only keep strong enough connections
                    if weight > 0.2:
                        new_connections[other.node_id] = weight
                        
                        # Update reciprocal connection
                        if not other.ghosted:
                            other.connections[node.node_id] = weight
                            self._update_node(other)
                    
                    # Handle significantly weakened connections
                    elif (other.node_id in old_connections and 
                        old_connections[other.node_id] > 0.7 and 
                        weight < 0.3):
                        self._preserve_ghost_connection(node, other, old_connections[other.node_id])
            
            # Update node's connections
            node.connections = new_connections
            
            # Handle any potential resurrections
            for other_id in potential_resurrections:
                other_node = self._get_node_by_id(other_id)
                if other_node and other_node.ghosted:
                    self._evaluate_ghost_state(other_node)
            
            self._update_node(node)
            
        except Exception as e:
            self.logger.log_error(f"Failed to update connections: {e}")

    def _establish_connection(
        self,
        node: CognitiveMemoryNode,
        other: CognitiveMemoryNode,
        weight: float
    ) -> None:
        """
        Establish a connection between two nodes, handling ghost states appropriately.
        """
        # Add connection to current node
        node.connections[other.node_id] = weight
        
        # Handle reciprocal connection based on ghost state
        if not other.ghosted:
            other.connections[node.node_id] = weight
            self._update_node(other)
        elif weight > self.revive_threshold:
            # If connection is strong enough, might trigger resurrection evaluation
            self._evaluate_ghost_state(other)

    def _preserve_ghost_connection(
        self,
        node: CognitiveMemoryNode,
        other: CognitiveMemoryNode,
        original_weight: float
    ) -> None:
        """
        Preserve a weakened connection as a ghost connection with additional context.
        """
        ghost_connection = {
            'node_id': other.node_id,
            'original_weight': original_weight,
            'timestamp': time.time(),
            'context': {
                'was_synthesis': bool(other.formation_source == 'synthesis'),
                'had_memory_links': bool(self.get_body_links(other.node_id))
            }
        }
        
        if not hasattr(node, 'ghost_connections'):
            node.ghost_connections = []
        node.ghost_connections.append(ghost_connection)

    def _persist_node(self, node: CognitiveMemoryNode) -> None:
        """
        Insert (if node.node_id is None) or update (if node.node_id is known) in DB.
        This is the 'create or update' method for a node.
        """
        data = node.to_dict()
        with self._db_connection() as conn:
            cursor = conn.cursor()

            if not node.node_id:
                # Insert new record
                cursor.execute("""
                    INSERT INTO cognitive_memory_nodes (
                        timestamp, text_content, embedding, raw_state, 
                        processed_state, strength, ghosted, parent_node_id,
                        ghost_nodes, ghost_states, connections, semantic_context,
                        last_accessed, formation_source, last_echo_time, echo_dampening
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    data['timestamp'],
                    data['text_content'],
                    data['embedding'],
                    data['raw_state'],
                    data['processed_state'],
                    data['strength'],
                    data['ghosted'],
                    data['parent_node_id'],
                    data['ghost_nodes'],
                    data['ghost_states'],
                    data['connections'],
                    data['semantic_context'],
                    data['last_accessed'],
                    data['formation_source'],
                    data['last_echo_time'],
                    data['echo_dampening']
                ))
                node.node_id = str(cursor.lastrowid)
                self.logger.debug(f"Inserted new cognitive node {node.node_id}")
            else:
                # Node already has an ID, so do an UPDATE
                cursor.execute("""
                    UPDATE cognitive_memory_nodes
                    SET timestamp=?,
                        text_content=?,
                        embedding=?,
                        raw_state=?,
                        processed_state=?,
                        strength=?,
                        ghosted=?,
                        parent_node_id=?,
                        ghost_nodes=?,
                        ghost_states=?,
                        connections=?,
                        semantic_context=?,
                        last_accessed=?,
                        formation_source=?,
                        last_echo_time=?,
                        echo_dampening=?
                    WHERE id=?
                """, (
                    data['timestamp'],
                    data['text_content'],
                    data['embedding'],
                    data['raw_state'],
                    data['processed_state'],
                    data['strength'],
                    data['ghosted'],
                    data['parent_node_id'],
                    data['ghost_nodes'],
                    data['ghost_states'],
                    data['connections'],
                    data['semantic_context'],
                    data['last_accessed'],
                    data['formation_source'],
                    data['last_echo_time'],
                    data['echo_dampening'],
                    node.node_id
                ))
                if cursor.rowcount == 0:
                    raise MemorySystemError(f"Node {node.node_id} not found in database for update")
                self.logger.debug(f"Updated cognitive node {node.node_id}")

            conn.commit()

    def _update_node(self, node: CognitiveMemoryNode) -> None:
        """
        Convenience method to quickly persist changes for an existing node.
        """
        if not node.node_id:
            # If no node_id yet, call _persist_node to create
            self._persist_node(node)
            return
        self._persist_node(node)  # We rely on the same method

    def _delete_node(self, node: CognitiveMemoryNode) -> None:
        """
        Remove the node from the database entirely.
        """
        if not node.node_id:
            self.logger.warning("Trying to delete a node with no ID; ignoring.")
            return

        with self._db_connection() as conn:
            cursor = conn.cursor()
            try:
                cursor.execute("DELETE FROM cognitive_memory_nodes WHERE id=?", (node.node_id,))
                if cursor.rowcount == 0:
                    raise MemorySystemError(f"Node {node.node_id} not found in database for deletion")
                conn.commit()
                self.logger.debug(f"Deleted cognitive node {node.node_id} from DB.")
            except sqlite3.Error as e:
                self.logger.log_error(f"Failed to delete node {node.node_id}: {e}")
                raise DatabaseError(f"Failed to delete node: {e}")

    def _generate_embedding(self, text_content: str) -> List[float]:
        """
        Generate embedding vector for semantic search using sentence transformers.
        Converts text content into a dense vector representation for similarity
        comparisons in memory retrieval and conflict detection.
        
        Args:
            text_content: Text content to embed
                
        Returns:
            List[float]: 384-dimensional embedding vector
        """
        try:
            # Normalize text
            cleaned_text = text_content.strip().lower()
            
            # If text is too long, take meaningful chunks
            # (BERT-like models typically have 512 token limit)
            max_chars = 512
            if len(cleaned_text) > max_chars:
                # Take first and last portions to capture key content
                split = max_chars // 2
                cleaned_text = cleaned_text[:split] + " ... " + cleaned_text[-split:]
                
            # Generate embedding - using all-MiniLM-L6-v2 model
            # This provides good balance of speed/quality and matches 384 dim
            embedding = self.embedding_model.encode(
                cleaned_text,
                convert_to_tensor=False,  # Return numpy array
                normalize_embeddings=True  # L2 normalize
            )
            
            return embedding.tolist()
            
        except Exception as e:
            self.logger.log_error(f"Failed to generate embedding: {e}")
            # Return zero vector as fallback
            return [0.0] * 384
            