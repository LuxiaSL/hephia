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
import numpy as np

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
    def calculate_metrics(
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
        
    def calculate_metrics(
        self,
        text_content: str,
        embedding: List[float],
        query_text: Optional[str] = None,
        query_embedding: Optional[List[float]] = None,
        analyze_clusters: bool = False,
        **kwargs
    ) -> Dict[str, float]:
        """
        Calculate comprehensive semantic metrics.
        
        Args:
            text_content: The text to analyze
            embedding: Pre-computed embedding for the text
            query_text: Optional query for direct text comparison
            query_embedding: Optional query embedding for similarity
            analyze_clusters: Whether to include cluster-based analysis
            
        Returns:
            Dict containing semantic metrics:
            - embedding_similarity: Direct embedding comparison
            - text_relevance: Keyword/content matching
            - semantic_density: Content richness score
            - cluster_metrics: (if requested) Cluster-based scores
            
        Raises:
            SemanticAnalysisError: If critical calculations fail
        """
        try:
            metrics = {}
            
            # Calculate embedding similarity if query provided
            if query_embedding is not None:
                metrics['embedding_similarity'] = self.embedding_manager.calculate_similarity(
                    embedding,
                    query_embedding
                )
            
            # Calculate text relevance if query text provided
            if query_text:
                metrics['text_relevance'] = self._calculate_text_relevance(
                    text_content,
                    query_text
                )
                
            # Always calculate semantic density
            metrics['semantic_density'] = self._calculate_semantic_density(text_content)
            
            # Optionally analyze cluster properties
            if analyze_clusters and 'cluster_nodes' in kwargs:
                metrics['cluster_metrics'] = self._analyze_cluster_semantics(
                    embedding,
                    kwargs['cluster_nodes']
                )
            
            return metrics
            
        except Exception as e:
            self.logger.log_error(f"Semantic metrics calculation failed: {str(e)}")
            # Provide fallback metrics
            return self._get_fallback_metrics()
            
    def _calculate_text_relevance(self, text: str, query: str) -> float:
        """
        Calculate text relevance using multiple methods.
        Combines keyword matching with semantic pattern recognition.
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
            if not text or len(text.strip()) == 0:
                return 0.0
                
            # Basic text preprocessing: split into sentences on period.
            sentences = [s.strip() for s in text.split('.') if s.strip()]
            if not sentences:
                return 0.0
                
            # Calculate semantic cohesion using embeddings (if more than one sentence)
            if len(sentences) > 1:
                embeddings = [
                    self.embedding_manager.encode(s, normalize_embeddings=True)
                    for s in sentences
                ]
                similarities = []
                for i in range(len(embeddings)):
                    for j in range(i+1, len(embeddings)):
                        sim = self.embedding_manager.calculate_similarity(
                            embeddings[i],
                            embeddings[j]
                        )
                        similarities.append(sim)
                semantic_cohesion = np.mean(similarities) if similarities else 0.5
            else:
                semantic_cohesion = 0.5
                
            # Lexical diversity analysis using safe tokenization.
            tokens = safe_word_tokenize(text.lower())
            if not tokens:
                return 0.0
                
            unique_ratio = len(set(tokens)) / len(tokens)
            
            # Syntactic complexity analysis using safe POS tagging.
            pos_tags = safe_pos_tag(tokens)
            complex_tags = {'IN', 'WDT', 'WP', 'WRB'}
            syntactic_complexity = len(
                [t for _, t in pos_tags if t in complex_tags]
            ) / len(pos_tags) if pos_tags else 0
            
            # Named entity density analysis using safe ne_chunk.
            try:
                ne_tree = safe_ne_chunk(pos_tags)
                # Count named entities if ne_tree is a Tree structure.
                if isinstance(ne_tree, Tree):
                    named_entities = len([
                        subtree for subtree in ne_tree 
                        if isinstance(subtree, Tree)
                    ])
                else:
                    named_entities = 0
                ne_density = named_entities / len(tokens)
            except Exception:
                ne_density = 0
            
            # Content word ratio (using a set of content tags)
            content_tags = {
                'NN', 'NNS', 'NNP', 'NNPS', 
                'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ',
                'JJ', 'JJR', 'JJS'
            }
            content_ratio = len(
                [t for _, t in pos_tags if t in content_tags]
            ) / len(pos_tags) if pos_tags else 0
            
            # Combine metrics with weights
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
            
            return min(1.0, max(0.0, density_score))
            
        except Exception as e:
            self.logger.log_error(f"Semantic density calculation failed: {str(e)}")
            # Fallback to simpler calculation
            try:
                words = text.split()
                meaningful_words = [w for w in words if w.isalpha()]
                return len(set(meaningful_words)) / len(meaningful_words) if meaningful_words else 0.0
            except Exception:
                return 0.0
                
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
