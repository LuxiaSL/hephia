"""
\\metrics\\semantic.py

Implements semantic analysis calculations for cognitive memory retrieval.
Handles text similarity, embedding comparison, and semantic density analysis.

Key capabilities:
- Embedding-based similarity calculation
- Text relevance scoring
- Semantic density analysis
- Support for future cluster analysis
"""

from typing import Dict, List, Optional, Any
from abc import ABC, abstractmethod

import time
import numpy as np
import concurrent.futures as cf
import asyncio
import hashlib

# Attempt to import NLTK and its components; if unavailable, define fallbacks.
try:
    import nltk
    from nltk import pos_tag, word_tokenize, ne_chunk
    from nltk.tree import Tree
except ImportError:
    # Log a warning (you might want to integrate with your logger here)
    print("Warning: NLTK not available, using basic fallbacks.")
    nltk = None

    def word_tokenize(text: str) -> List[str]:
        # Fallback: very naive whitespace tokenization.
        return text.split()

    def pos_tag(tokens: List[str]) -> List[tuple]:
        # Fallback: assign a default tag.
        return [(token, "NN") for token in tokens]

    def ne_chunk(pos_tags: List[tuple]) -> Any:
        # Fallback: return an empty list indicating no named entities.
        return []

    Tree = list  # Dummy alias

# Even if NLTK is importable, ensure required resources are available.
def safe_word_tokenize(text: str) -> List[str]:
    try:
        tokens = word_tokenize(text)
        return tokens
    except LookupError:
        try:
            if nltk is not None:
                nltk.download("punkt")
                return word_tokenize(text)
        except Exception as e:
            print(f"Error downloading 'punkt': {e}")
        # Fallback to basic split.
        return text.split()
    except Exception as e:
        print(f"Unexpected error in word_tokenize: {e}")
        return text.split()

def safe_pos_tag(tokens: List[str]) -> List[tuple]:
    try:
        return pos_tag(tokens)
    except Exception as e:
        print(f"Error in pos_tag, using fallback: {e}")
        return [(token, "NN") for token in tokens]

def safe_ne_chunk(pos_tags: List[tuple]) -> Any:
    try:
        return ne_chunk(pos_tags)
    except Exception as e:
        print(f"Error in ne_chunk, returning empty list: {e}")
        return []

# Import internal modules
from internal.modules.memory.async_lru_cache import async_lru_cache
from internal.modules.memory.embedding_manager import EmbeddingManager
from loggers.loggers import MemoryLogger

class SemanticAnalysisError(Exception):
    """Base exception for semantic analysis failures."""
    pass

class BaseSemanticMetricsCalculator(ABC):
    """
    Abstract base class defining the interface for semantic metrics calculation.
    Allows for different semantic analysis implementations while maintaining
    consistent interface.
    """
    
    @abstractmethod
    async def calculate_metrics(
        self,
        text_content: str,
        embedding: List[float],
        query_text: Optional[str] = None,
        query_embedding: Optional[List[float]] = None,
        **kwargs
    ) -> Dict[str, float]:
        """Calculate semantic similarity metrics."""
        pass

class SemanticMetricsCalculator(BaseSemanticMetricsCalculator):
    """
    Calculates semantic similarity metrics between memory nodes or queries.
    Handles embedding comparison, text analysis, and semantic density calculation.
    """
    
    def __init__(self, embedding_manager: EmbeddingManager):
        """
        Initialize calculator with required dependencies.
        
        Args:
            embedding_manager: Manager for generating/comparing embeddings
        """
        self.embedding_manager = embedding_manager
        self.logger = MemoryLogger

    def _generate_text_cache_key(self, text: str, query: str = "") -> str:
        """Generate cache key for text-based calculations."""
        try:
            text_hash = hashlib.md5(text.encode('utf-8')).hexdigest()[:12]
            query_hash = hashlib.md5(query.encode('utf-8')).hexdigest()[:8] if query else "no_query"
            return f"text:{text_hash}:query:{query_hash}"
        except Exception:
            return f"fallback:{hash((text, query))}"
        
    async def calculate_metrics(
        self,
        text_content: str,
        embedding: List[float],
        query_text: Optional[str] = None,
        query_embedding: Optional[List[float]] = None,
        analyze_clusters: bool = False,
        **kwargs
    ) -> Dict[str, float]:
        """
        Pipelined semantic metrics calculation.
        """
        try:
            metrics = {}
            
            # Start the expensive semantic density calculation first
            # This runs in the background while we do other calculations
            density_task = self._calculate_semantic_density(text_content)
            
            # Start cluster analysis in parallel if requested (also expensive)
            cluster_task = None
            if analyze_clusters and 'cluster_nodes' in kwargs:
                cluster_task = asyncio.create_task(self._analyze_cluster_semantics_async(
                    embedding, kwargs['cluster_nodes']
                ))
            
            # Do quick calculations immediately
            if query_embedding is not None:
                metrics['embedding_similarity'] = self.embedding_manager.calculate_similarity(
                    embedding,
                    query_embedding
                )
            
            if query_text:
                metrics['text_relevance'] = self._calculate_text_relevance(
                    text_content,
                    query_text
                )
            
            # Wait for density calculation to complete
            metrics['semantic_density'] = await density_task
            
            # Get cluster results if requested
            if cluster_task:
                metrics['cluster_metrics'] = await cluster_task
            
            return metrics
            
        except Exception as e:
            self.logger.log_error(f"Semantic metrics calculation failed: {str(e)}")
            return self._get_fallback_metrics()
            
    @async_lru_cache(
        maxsize=1000, 
        ttl=3600, 
        key_func=lambda self, text, query: self._generate_text_cache_key(text, query)
    )
    async def _calculate_text_relevance_cached(self, text: str, query: str) -> float:
        """Cached version of text relevance calculation."""
        return self._calculate_text_relevance_internal(text, query)

    def _calculate_text_relevance(self, text: str, query: str) -> float:
        """
        Calculate text relevance using cached computation.
        """
        try:
            # Use cached version if in async context
            if asyncio.current_task() is not None:
                return asyncio.create_task(self._calculate_text_relevance_cached(text, query))
            else:
                return self._calculate_text_relevance_internal(text, query)
        except RuntimeError:
            return self._calculate_text_relevance_internal(text, query)

    def _calculate_text_relevance_internal(self, text: str, query: str) -> float:
        """
        Internal text relevance calculation (the actual computation).
        """
        try:
            if not text or not query:
                return 0.0
                
            text_lower = text.lower()
            query_lower = query.lower()
            
            # Calculate keyword overlap
            query_words = set(query_lower.split())
            text_words = set(text_lower.split())
            if not query_words:
                return 0.0
                
            # Direct keyword matching
            word_matches = query_words.intersection(text_words)
            keyword_score = len(word_matches) / len(query_words)

            # Add semantic patterns here in future
            # EXPANSION POINT: Enhanced semantic pattern matching
            
            return keyword_score
            
        except Exception as e:
            self.logger.log_error(f"Text relevance calculation failed: {str(e)}")
            return 0.0
            
    @async_lru_cache(maxsize=1000, ttl=7200)
    async def _calculate_semantic_density(self, text: str) -> float:
        """
        Optimized semantic density calculation with tiered processing and parallelization.
        """
        try:
            # Fast path for empty or very short text
            if not text or len(text.strip()) < 10:
                return 0.0
            
            # Medium texts get simplified calculation
            if len(text) < 100:
                return await self._calculate_simple_density(text)
            
            # For longer texts, use full parallelized approach
            return await self._calculate_full_density(text)
                
        except Exception as e:
            self.logger.log_error(f"Semantic density calculation failed: {str(e)}")
            # Fallback to simpler calculation
            try:
                words = text.split()
                meaningful_words = [w for w in words if w.isalpha()]
                return len(set(meaningful_words)) / len(meaningful_words) if meaningful_words else 0.0
            except Exception:
                return 0.0

    async def _calculate_simple_density(self, text: str) -> float:
        """Simplified density calculation for shorter texts."""
        tokens = text.lower().split()
        # Simple lexical diversity metric
        unique_ratio = len(set(tokens)) / len(tokens) if tokens else 0.0
        # Scale to appropriate range (0.3-0.7) for short texts
        return 0.3 + (unique_ratio * 0.4)

    async def _calculate_full_density(self, text: str) -> float:
        """Full parallel density calculation for longer texts."""
        # Split into sentences
        sentences = [s.strip() for s in text.split('.') if s.strip()]
        if not sentences:
            return 0.0
        
        # Start multiple async tasks in parallel
        tasks = {}
        
        # 1. Calculate semantic cohesion if multiple sentences
        tasks['cohesion'] = self._calculate_semantic_cohesion(sentences)
        
        # 2. Start NLP processing pipeline
        tasks['nlp'] = self._process_nlp_features(text)
        
        # Wait for all tasks to complete
        results = await asyncio.gather(
            tasks['cohesion'],
            tasks['nlp'],
            return_exceptions=True
        )
        
        # Extract results (with error handling)
        semantic_cohesion = 0.5
        nlp_features = {}
        
        if not isinstance(results[0], Exception):
            semantic_cohesion = results[0]
        
        if not isinstance(results[1], Exception):
            nlp_features = results[1]
        
        # Combine metrics with weights
        weights = {
            'semantic_cohesion': 0.3,
            'lexical_diversity': 0.2,
            'syntactic_complexity': 0.15,
            'ne_density': 0.15,
            'content_ratio': 0.2
        }
        
        # Extract features with defaults
        unique_ratio = nlp_features.get('lexical_diversity', 0.5)
        syntactic_complexity = nlp_features.get('syntactic_complexity', 0.3)
        ne_density = nlp_features.get('ne_density', 0.1)
        content_ratio = nlp_features.get('content_ratio', 0.5)
        
        # Calculate final score
        density_score = (
            semantic_cohesion * weights['semantic_cohesion'] +
            unique_ratio * weights['lexical_diversity'] +
            syntactic_complexity * weights['syntactic_complexity'] +
            ne_density * weights['ne_density'] +
            content_ratio * weights['content_ratio']
        )
        
        return min(1.0, max(0.0, density_score))
    
    async def _calculate_semantic_cohesion(self, sentences: List[str]) -> float:
        """Calculate semantic cohesion with parallel embedding generation."""
        if len(sentences) <= 1:
            return 0.5
        
        # Generate embeddings in parallel
        embedding_tasks = [
            self.embedding_manager.encode(s, normalize_embeddings=True)
            for s in sentences
        ]
        embeddings = await asyncio.gather(*embedding_tasks)
        
        # Calculate pairwise similarities
        similarities = []
        
        # For large numbers of sentences, we could parallelize this too
        # But for typical memory content, this is usually fast enough
        for i in range(len(embeddings)):
            for j in range(i+1, len(embeddings)):
                sim = self.embedding_manager.calculate_similarity(
                    embeddings[i],
                    embeddings[j]
                )
                similarities.append(sim)
        
        return np.mean(similarities) if similarities else 0.5

    async def _process_nlp_features(self, text: str) -> Dict[str, float]:
        """Process NLP features with parallel execution."""
        loop = asyncio.get_event_loop()
        
        # Start tokenization (required for other processes)
        tokens_future = loop.run_in_executor(None, safe_word_tokenize, text.lower())
        tokens = await tokens_future
        
        if not tokens:
            return {
                'lexical_diversity': 0.0,
                'syntactic_complexity': 0.0,
                'ne_density': 0.0,
                'content_ratio': 0.0
            }
        
        # Calculate lexical diversity immediately (fast)
        lexical_diversity = len(set(tokens)) / len(tokens)
        
        # Start POS tagging (required for other processes)
        pos_tags_future = loop.run_in_executor(None, safe_pos_tag, tokens)
        pos_tags = await pos_tags_future
        
        # Now run three parallel operations that all need pos_tags
        with cf.ThreadPoolExecutor() as executor:
            # Submit three tasks to thread pool
            complexity_future = loop.run_in_executor(
                executor,
                self._calculate_syntactic_complexity,
                pos_tags
            )
            
            entity_future = loop.run_in_executor(
                executor,
                self._calculate_named_entity_density,
                pos_tags,
                tokens
            )
            
            content_future = loop.run_in_executor(
                executor,
                self._calculate_content_ratio,
                pos_tags
            )
            
            # Wait for all to complete
            syntactic_complexity, ne_density, content_ratio = await asyncio.gather(
                complexity_future,
                entity_future,
                content_future
            )
        
        return {
            'lexical_diversity': lexical_diversity,
            'syntactic_complexity': syntactic_complexity,
            'ne_density': ne_density,
            'content_ratio': content_ratio
        }
    
    def _calculate_syntactic_complexity(self, pos_tags):
        """Calculate syntactic complexity (thread-safe)."""
        if not pos_tags:
            return 0.0
        
        complex_tags = {'IN', 'WDT', 'WP', 'WRB'}
        return len([t for _, t in pos_tags if t in complex_tags]) / len(pos_tags)

    def _calculate_named_entity_density(self, pos_tags, tokens):
        """Calculate named entity density (thread-safe)."""
        if not pos_tags or not tokens:
            return 0.0
        
        try:
            ne_tree = safe_ne_chunk(pos_tags)
            
            if isinstance(ne_tree, Tree):
                named_entities = len([
                    subtree for subtree in ne_tree 
                    if isinstance(subtree, Tree)
                ])
            else:
                named_entities = 0
                
            return named_entities / len(tokens)
        except Exception:
            return 0.0

    def _calculate_content_ratio(self, pos_tags):
        """Calculate content word ratio (thread-safe)."""
        if not pos_tags:
            return 0.0
        
        content_tags = {
            'NN', 'NNS', 'NNP', 'NNPS', 
            'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ',
            'JJ', 'JJR', 'JJS'
        }
        
        return len([t for _, t in pos_tags if t in content_tags]) / len(pos_tags)
    
    async def _analyze_cluster_semantics_async(
        self,
        center_embedding: List[float],
        cluster_nodes: List[Any]
    ) -> Dict[str, float]:
        """Parallelized cluster analysis."""
        if not cluster_nodes:
            return {}
        
        metrics = {}
        
        # Get all embeddings in parallel
        valid_nodes = [node for node in cluster_nodes if hasattr(node, 'embedding')]
        if not valid_nodes:
            return {}
        
        # Calculate similarities (could be parallelized for very large clusters)
        similarities = [
            self.embedding_manager.calculate_similarity(center_embedding, node.embedding)
            for node in valid_nodes
        ]
        
        if similarities:
            metrics['semantic_spread'] = np.std(similarities)
            metrics['semantic_cohesion'] = np.mean(similarities)
        
        return metrics
                
    def _analyze_cluster_semantics(
        self,
        center_embedding: List[float],
        cluster_nodes: List[Any]
    ) -> Dict[str, float]:
        """
        Analyze semantic properties of memory clusters.
        
        EXPANSION POINT: This method can be enhanced to provide deeper
        cluster analysis, pattern recognition, and emergent properties.
        
        Args:
            center_embedding: Embedding of the focal node
            cluster_nodes: List of related nodes to analyze
            
        Returns:
            Dict containing cluster semantic metrics
        """
        try:
            if not cluster_nodes:
                return {}
                
            metrics = {}
            
            # Calculate semantic spread
            similarities = []
            for node in cluster_nodes:
                if hasattr(node, 'embedding'):
                    sim = self.embedding_manager.calculate_similarity(
                        center_embedding,
                        node.embedding
                    )
                    similarities.append(sim)
                    
            if similarities:
                metrics['semantic_spread'] = np.std(similarities)
                metrics['semantic_cohesion'] = np.mean(similarities)
                
            # EXPANSION POINT: Add semantic field calculations
            # EXPANSION POINT: Add pattern recognition across cluster
            # EXPANSION POINT: Add theme extraction
            
            return metrics
            
        except Exception as e:
            self.logger.log_error(f"Cluster analysis failed: {str(e)}")
            return {}
            
    def _get_fallback_metrics(self) -> Dict[str, float]:
        """Provide safe fallback metrics if calculations fail."""
        return {
            'embedding_similarity': 0.0,
            'text_relevance': 0.0,
            'semantic_density': 0.0
        }
