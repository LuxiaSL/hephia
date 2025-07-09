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

try:
    import spacy
    # Load the model once and reuse it. This is more efficient.
    # Note: Loading large models can take time and memory.
    NLP = spacy.load("en_core_web_sm") 
except ImportError:
    print("Warning: SpaCy not available. NE Density will be 0.")
    NLP = None
except OSError:
    print("Warning: SpaCy model 'en_core_web_sm' not found. NE Density will be 0.")
    NLP = None

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
            density_task = asyncio.create_task(self._calculate_semantic_density(text_content))

            # Start cluster analysis in parallel if requested (also expensive)
            cluster_task = None
            if analyze_clusters and 'cluster_nodes' in kwargs:
                cluster_task = asyncio.create_task(self._analyze_cluster_semantics_async(
                    embedding, kwargs['cluster_nodes']
                ))
            
            if query_embedding is not None:
                metrics['embedding_similarity'] = await self.embedding_manager.calculate_similarity_async(
                    embedding,
                    query_embedding
                )
            
            if query_text:
                metrics['text_relevance'] = await self._calculate_text_relevance_async(
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

    async def _calculate_text_relevance_async(self, text: str, query: str) -> float:
        """
        Async version of text relevance calculation with caching.
        Use this when calling from async contexts.
        """
        return await self._calculate_text_relevance_cached(text, query)

    def _calculate_text_relevance(self, text: str, query: str) -> float:
        """
        Synchronous text relevance calculation.
        Always returns a float, never a Task.
        """
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
        Optimized semantic density calculation
        """
        try:
            if not text or len(text.strip()) < 10:
                return 0.0
            
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
            'semantic_cohesion': 0.25,       # How connected are ideas
            'ne_density': 0.25,              # Specific entities mentioned
            'abstraction_level': 0.20,       # Abstract vs concrete reasoning
            'logical_complexity': 0.15,      # Logical structure complexity
            'conceptual_bridging': 0.10,     # Connecting disparate ideas
            'information_density': 0.05      # Factual information density
        }
        
        # Extract features with defaults
        ne_density = nlp_features.get('ne_density', 0.0)
        abstraction_level = nlp_features.get('abstraction_level', 0.5)
        logical_complexity = nlp_features.get('logical_complexity', 0.0)
        conceptual_bridging = nlp_features.get('conceptual_bridging', 0.0)
        information_density = nlp_features.get('information_density', 0.0)
        
        # Calculate final score
        density_score = (
            semantic_cohesion * weights['semantic_cohesion'] +
            ne_density * weights['ne_density'] +
            abstraction_level * weights['abstraction_level'] +
            logical_complexity * weights['logical_complexity'] +
            conceptual_bridging * weights['conceptual_bridging'] +
            information_density * weights['information_density']
        )

        raw_density = min(1.0, max(0.0, density_score))

        # Log raw density for analysis
        self.logger.debug(f"[DENSITY_DEBUG] Raw density before transformation: {raw_density:.3f}")
        # and log the density scores
        self.logger.debug(f"[DENSITY_DEBUG] Semantic Cohesion: {semantic_cohesion:.3f}, "
                    f"NE Density: {ne_density:.3f}, "
                    f"Abstraction Level: {abstraction_level:.3f}, "
                    f"Logical Complexity: {logical_complexity:.3f}, "
                    f"Conceptual Bridging: {conceptual_bridging:.3f}, "
                    f"Information Density: {information_density:.3f}")
        
        # Apply exponential transformation
        smushed_density = self._apply_exponential_density_transform(raw_density, text)

        self.logger.debug(f"[DENSITY_DEBUG] Density after transformation: {smushed_density:.3f}")

        return smushed_density
    
    def _apply_exponential_density_transform(self, raw_density: float, text: str) -> float:
        """
        Apply exponential transformation to enhance semantic density discrimination.
        """
        try:
            # Method 1: Power law transformation (more aggressive for high density)
            if raw_density < 0.35:
                enhanced = (raw_density / 0.35) ** 4.0 * 0.2  # Compress low end more
            elif raw_density < 0.45:
                enhanced = 0.2 + (raw_density - 0.35) * 5.0   # Steep middle
            else:
                enhanced = 0.7 + (1.0 - (0.5 ** ((raw_density - 0.45) * 15))) * 0.3
            
            # Ensure bounds
            enhanced = max(0.05, min(0.95, enhanced))
            
            # Log transformation details for analysis
            self.logger.debug(f"[DENSITY_TRANSFORM] Raw: {raw_density:.3f} → Enhanced: {enhanced:.3f}")
            
            return enhanced
            
        except Exception as e:
            self.logger.log_error(f"Density transformation failed: {e}")
            # Fallback to more aggressive sigmoid
            import math
            shifted = (raw_density - 0.35) * 16
            return 1 / (1 + math.exp(-shifted))

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
        """
        Process NLP features using enhanced semantic discriminators.
        """
        if not text or not NLP:
            return {
                'semantic_cohesion_base': 0.0,
                'ne_density': 0.0,
                'abstraction_level': 0.0,
                'logical_complexity': 0.0,
                'conceptual_bridging': 0.0,
                'information_density': 0.0
            }

        try:
            doc = NLP(text)
            
            if not doc:
                return {
                    'semantic_cohesion_base': 0.0,
                    'ne_density': 0.0,
                    'abstraction_level': 0.0,
                    'logical_complexity': 0.0,
                    'conceptual_bridging': 0.0,
                    'information_density': 0.0
                }

            # Calculate all discriminators from the single SpaCy doc
            ne_density = self._calculate_named_entity_density(doc)
            abstraction_level = self._calculate_abstraction_level(doc)
            logical_complexity = self._calculate_logical_complexity(doc)
            conceptual_bridging = self._calculate_conceptual_bridging(doc)
            information_density = self._calculate_information_density(doc)

            return {
                'semantic_cohesion_base': 0.5,  # Placeholder - calculated separately
                'ne_density': ne_density,
                'abstraction_level': abstraction_level,
                'logical_complexity': logical_complexity,
                'conceptual_bridging': conceptual_bridging,
                'information_density': information_density
            }
        except Exception as e:
            self.logger.log_error(f"Enhanced SpaCy processing failed: {e}")
            return {
                'semantic_cohesion_base': 0.5,
                'ne_density': 0.0,
                'abstraction_level': 0.5,
                'logical_complexity': 0.5,
                'conceptual_bridging': 0.5,
                'information_density': 0.5
            }
        
    def _calculate_abstraction_level(self, doc) -> float:
        """
        Calculate abstraction level using domain-specific patterns instead of word frequency.
        Optimized for AI agent memory formation content.
        """
        if not doc:
            return 0.5

        abstract_score = 0.0
        concrete_score = 0.0
        total_scored = 0

        for token in doc:
            if token.pos_ not in ['NOUN', 'VERB', 'ADJ'] or token.is_stop:
                continue

            lemma = token.lemma_.lower()
            scored = False

            # Concrete patterns - specific to agent memory domain
            if token.ent_type_ or token.like_num:
                concrete_score += 3.0
                scored = True
            elif lemma in {'execute', 'run', 'process', 'create', 'update', 'delete', 'fix', 'install', 'query', 'database', 'file', 'system', 'command', 'code', 'error', 'result'}:
                concrete_score += 2.0
                scored = True
            elif token.pos_ == 'VERB' and len(token.text) <= 5:
                concrete_score += 1.5  # Short action verbs
                scored = True

            # Abstract patterns - reasoning and conceptual words
            elif lemma in {'understanding', 'realization', 'insight', 'concept', 'approach', 'strategy', 'pattern', 'relationship', 'significance', 'complexity', 'abstraction'}:
                abstract_score += 2.5
                scored = True
            elif lemma in {'learn', 'understand', 'realize', 'recognize', 'develop', 'improve', 'analyze', 'evaluate', 'consider', 'determine'}:
                abstract_score += 2.0
                scored = True
            elif token.pos_ == 'ADJ' and len(token.text) >= 8:
                abstract_score += 1.5  # Long descriptive adjectives
                scored = True

            # Default patterns for unscored tokens
            elif not scored:
                if len(token.text) <= 4:
                    concrete_score += 1.0
                elif len(token.text) >= 10:
                    abstract_score += 1.0

            total_scored += 1

        if total_scored == 0:
            return 0.5

        # Calculate ratio
        total_points = abstract_score + concrete_score
        if total_points == 0:
            return 0.5

        abstraction_ratio = abstract_score / total_points
        
        # Map to realistic 0.2-0.8 range instead of 0.15-0.85
        final_score = 0.2 + (abstraction_ratio * 0.6)
        
        return min(0.95, max(0.05, final_score))
    
    def _calculate_logical_complexity(self, doc) -> float:
        """
        Calculate logical complexity using a hybrid of syntactic and semantic analysis.
        This combines robust grammatical structure analysis with domain-specific reasoning patterns.
        """
        if not doc or not len(doc):
            return 0.0
        
        sentence_count = len(list(doc.sents))
        if sentence_count == 0:
            return 0.0
        
        # 1. Analyze grammatical structure (dependency parsing)
        dependency_score = self._analyze_dependency_complexity(doc)
        
        # 2. Analyze simple structural markers (POS tags)
        pos_score = self._analyze_pos_complexity(doc)
        
        # 3. Analyze semantic and reasoning patterns
        semantic_score = self._analyze_semantic_complexity(doc)
        
        # Combine scores and normalize by sentence count
        total_complexity = dependency_score + pos_score + semantic_score
        complexity_per_sentence = total_complexity / sentence_count
        
        # Map to a 0.05-0.95 range. The divisor is tuned for the higher potential score.
        return max(0.05, min(0.95, complexity_per_sentence / 8.0))

    def _analyze_dependency_complexity(self, doc) -> float:
        """Analyze syntactic complexity using dependency parsing."""
        complexity = 0.0
        for token in doc:
            # Subordinate clauses indicate high complexity
            if token.dep_ in ['advcl', 'acl', 'ccomp', 'xcomp']:
                complexity += 1.5
            # Relative clauses are also complex
            elif token.dep_ == 'relcl':
                complexity += 1.8
            # Coordination adds moderate complexity
            elif token.dep_ == 'conj':
                complexity += 1.0
            # Conditional markers introduce logical branching
            elif token.dep_ == 'mark' and token.head.pos_ == 'VERB':
                complexity += 1.2
        return complexity

    def _analyze_pos_complexity(self, doc) -> float:
        """Analyze complexity using POS tag patterns as simple heuristics."""
        complexity = 0.0
        # Count conjunctions as a simple measure of clause combination
        pos_tags = [token.pos_ for token in doc]
        complexity += pos_tags.count('SCONJ') * 1.5  # Subordinating
        complexity += pos_tags.count('CCONJ') * 1.0  # Coordinating
        return complexity

    def _analyze_semantic_complexity(self, doc) -> float:
        """Analyze complexity using semantic and reasoning patterns."""
        complexity = 0.0
        
        # Define lemmas for different reasoning categories
        causal_verbs = {'cause', 'result', 'lead', 'make', 'force', 'imply', 'indicate'}
        realization_verbs = {'realize', 'notice', 'understand', 'recognize', 'discover', 'learn'}
        pattern_nouns = {'pattern', 'trend', 'similarity', 'connection', 'relationship', 'theme'}
        
        for token in doc:
            # Causal and reasoning verbs
            if token.pos_ == 'VERB' and token.lemma_ in causal_verbs:
                complexity += 1.8
                
            # Meta-cognitive / realization verbs
            elif token.pos_ == 'VERB' and token.lemma_ in realization_verbs:
                complexity += 1.5
                
            # Pattern recognition nouns
            elif token.pos_ == 'NOUN' and token.lemma_ in pattern_nouns:
                complexity += 2.0
                
            # Uncertainty reasoning (modal verbs)
            elif token.tag_ == 'MD':
                if token.lemma_ in ['must', 'will']:
                    complexity += 0.8  # Certainty
                elif token.lemma_ in ['might', 'could', 'may', 'should']:
                    complexity += 1.2  # Uncertainty/possibility is complex
                    
            # Comparative analysis
            elif token.pos_ == 'ADP' and token.text.lower() in {'than', 'like', 'unlike'}:
                complexity += 1.0
                
        return complexity
    
    def _calculate_conceptual_bridging(self, doc) -> float:
        """
        Measures conceptual bridging using a hybrid of syntactic and semantic analysis.
        """
        if not doc or not len(doc):
            return 0.0

        bridging_score = 0.0

        for token in doc:
            # Calculate bridging score from different linguistic facets
            bridging_score += self._get_syntactic_bridge_score(token) * 2.0
            bridging_score += self._get_semantic_bridge_score(token) * 2.0

        entity_count = len(doc.ents)
        if entity_count >= 2:
            bridging_score += min(entity_count * 0.8, 3.0)
        
        # Normalize by the number of tokens
        density = bridging_score / len(doc)
        
        # Scale the final score
        return max(0.05, min(0.95, density / 0.5))

    def _get_syntactic_bridge_score(self, token) -> float:
        """Analyzes grammatical markers that create bridges."""
        # Demonstrative determiners and pronouns pointing to concepts
        if token.pos_ in ['DET', 'PRON'] and token.lemma_ in {'this', 'that', 'these', 'those'}:
            return 1.5
        # Personal pronouns indicating a shared social context
        elif token.pos_ == 'PRON' and token.lemma_ in {'we', 'they', 'our', 'their'}:
            return 1.0
        # Subordinating conjunctions linking clauses
        elif token.pos_ == 'SCONJ' and token.lemma_ in {'as', 'like', 'while', 'since', 'because'}:
            return 1.2
            
        return 0.0

    def _get_semantic_bridge_score(self, token) -> float:
        """Analyzes word meanings that imply conceptual bridges."""
        # Verbs of comparison or relation
        if token.pos_ == 'VERB' and token.lemma_ in {'compare', 'relate', 'connect', 'remind', 'associate'}:
            return 2.0
            
        # Adjectives of comparison or relation
        elif token.pos_ == 'ADJ' and token.lemma_ in {'similar', 'different', 'related', 'typical'}:
            return 1.8
            
        # Adverbs implying temporal or logical sequence
        elif token.pos_ == 'ADV' and token.lemma_ in {'previously', 'earlier', 'consequently', 'therefore', 'however'}:
            return 1.5
            
        # Nouns that explicitly name patterns or relationships
        elif token.pos_ == 'NOUN' and token.lemma_ in {'pattern', 'trend', 'similarity', 'connection', 'relationship'}:
            return 2.5 # Highest score for explicit pattern recognition
            
        return 0.0
    
    def _calculate_information_density(self, doc) -> float:
        """
        Calculate information density using linguistic features rather than exact vocabulary.
        More robust and generalizable approach.
        """
        if not doc or not len(doc):
            return 0.0
        
        total_score = 0.0
        
        for token in doc:
            # Use separate scoring functions for different information types
            total_score += self._get_technical_info_score(token)
            total_score += self._get_social_info_score(token)
            total_score += self._get_factual_info_score(token)
        
        # Normalize by document length
        density = total_score / len(doc)
        
        # Map to 0.05-0.95 range
        return max(0.05, min(0.95, density / 2.0))

    def _get_technical_info_score(self, token) -> float:
        """Score technical information using linguistic features."""
        score = 0.0
        
        # More aggressive scoring for high-value information
        if token.like_num:
            score += 4.0
        elif token.ent_type_ in ['DATE', 'TIME', 'MONEY', 'PERCENT', 'QUANTITY']:
            score += 3.5
        elif token.ent_type_ in ['ORG', 'PRODUCT', 'EVENT']:
            score += 3.0 
        elif token.pos_ in ['NOUN', 'VERB'] and not token.is_stop:
            if len(token.text) >= 8:
                score += 2.0
            elif any(suffix in token.text.lower() for suffix in ['tion', 'ment', 'ness', 'ity', 'ize', 'ise']):
                score += 1.5
            elif '_' in token.text or (token.text.islower() and any(c.isupper() for c in token.text)):
                score += 2.0

        return score

    def _get_social_info_score(self, token) -> float:
        """Score social information using linguistic features."""
        score = 0.0
        
        if token.ent_type_ == 'PERSON':
            score += 2.5
        elif token.ent_type_ in ['GPE', 'LOC']:
            score += 1.5 
        elif token.pos_ == 'ADJ':
            if len(token.text) >= 6:
                score += 1.3
            elif token.tag_ in ['JJR', 'JJS']:
                score += 1.5
        elif token.pos_ == 'VERB' and token.lemma_ in {
            'say', 'tell', 'ask', 'reply', 'respond', 'talk', 'discuss', 
            'explain', 'suggest', 'recommend', 'help', 'support'
        }:
            score += 2.0
        elif token.pos_ == 'PRON' and token.text.lower() in ['i', 'you', 'we', 'they']:
            score += 0.8
        
        return score

    def _get_factual_info_score(self, token) -> float:
        """Score factual information using linguistic features."""
        score = 0.0
        
        # Proper nouns generally carry factual information
        if token.pos_ == 'PROPN' and not token.ent_type_:
            score += 1.6
        
        # Cardinal numbers (ONE, TWO, etc.) are factual
        elif token.pos_ == 'NUM':
            score += 1.8
        
        # Specific determiners that indicate concrete reference
        elif token.pos_ == 'DET' and token.text.lower() in ['this', 'that', 'these', 'those']:
            score += 0.8
        
        # Modal verbs can indicate uncertainty vs certainty
        elif token.pos_ == 'VERB' and token.tag_ == 'MD':
            if token.text.lower() in ['must', 'will', 'did']:
                score += 1.0  # Certainty
            else:
                score += 0.5  # Uncertainty
        
        return score
    
    def _calculate_named_entity_density(self, doc) -> float:
        """Calculate and amplify named entity density from the SpaCy doc."""
        if not doc:
            return 0.0
        
        # Extract tokens from the SpaCy doc
        tokens = [token for token in doc if not token.is_space]
        
        if not tokens:
            return 0.0
        
        raw_density = len(doc.ents) / len(tokens)
        
        # Amplify the raw score. We'll set a "high water mark" where a raw density
        # of 0.1 or greater gets a full score of 1.0. This makes the metric
        # much more sensitive to the presence of even a few entities.
        amplification_factor = 10.0  # (since 1.0 / 0.1 = 10)
        amplified_score = min(1.0, raw_density * amplification_factor)
        
        return amplified_score

    async def _analyze_cluster_semantics_async(
        self,
        center_embedding: List[float],
        cluster_nodes: List[Any]
    ) -> Dict[str, float]:
        """Parallelized cluster analysis."""
        if not cluster_nodes:
            return {}
        
        metrics = {}
        
        # Get all embeddings
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
