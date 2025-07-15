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
    print("Warning: SpaCy not available. NE Density will be 0. Please install 'spacy' and 'en_core_web_sm' model.")
    NLP = None
except OSError:
    print("Warning: SpaCy model 'en_core_web_sm' not found. NE Density will be 0. Please install 'spacy' and 'en_core_web_sm' model.")
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

            cohesion_task = None
            sentences = [s.strip() for s in text_content.split('.') if s.strip()]
            if len(sentences) >= 2:
                cohesion_task = asyncio.create_task(self._calculate_semantic_cohesion_cached(sentences))

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

            if cohesion_task:
                metrics['semantic_cohesion'] = await cohesion_task
            else:
                metrics['semantic_cohesion'] = 0.4
            
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
        Enhanced text relevance calculation optimized for direct matching.
        Ensures important entities and phrases aren't lost in semantic analysis.
        """
        try:
            if not text or not query:
                return 0.0
                
            text_lower = text.lower()
            query_lower = query.lower()
            
            # Base keyword overlap (preserve existing logic)
            query_words = set(query_lower.split())
            text_words = set(text_lower.split())
            if not query_words:
                return 0.0
                
            word_matches = query_words.intersection(text_words)
            keyword_score = len(word_matches) / len(query_words)
            
            # ENHANCEMENT 1: Named entity exact matching boost
            entity_boost = self._calculate_entity_relevance_boost(text, query)
            
            # ENHANCEMENT 2: Information density weighting for matched words
            weighted_score = self._calculate_weighted_word_relevance(text_lower, query_lower, word_matches)
            
            # ENHANCEMENT 3: Exact phrase matching bonus
            phrase_bonus = self._calculate_phrase_relevance_bonus(text_lower, query_lower)
            
            # Combine scores: baseline + enhancements
            final_score = (
                keyword_score * 0.5 +           # Base keyword overlap
                entity_boost * 0.25 +           # Entity matching boost
                weighted_score * 0.15 +         # Information density weighting  
                phrase_bonus * 0.10             # Exact phrase bonus
            )
            
            return min(1.0, final_score)
            
        except Exception as e:
            self.logger.log_error(f"Enhanced text relevance calculation failed: {e}")
            return 0.0
        
    def _calculate_entity_relevance_boost(self, text: str, query: str) -> float:
        """Calculate boost for exact entity matches (names, important terms)."""
        try:
            # Look for capitalized words (likely entities) - simple but effective
            query_entities = set()
            text_entities = set()
            
            for word in query.split():
                if word and word[0].isupper() and len(word) > 2:  # Capitalized, meaningful length
                    query_entities.add(word.lower())
                    
            for word in text.split():
                if word and word[0].isupper() and len(word) > 2:
                    text_entities.add(word.lower())
            
            if not query_entities:
                return 0.0
                
            entity_matches = query_entities.intersection(text_entities)
            return len(entity_matches) / len(query_entities)
            
        except Exception:
            return 0.0
        
    def _calculate_weighted_word_relevance(self, text_lower: str, query_lower: str, word_matches: set) -> float:
        """Weight matched words by their information value."""
        try:
            if not word_matches:
                return 0.0
                
            weighted_score = 0.0
            query_words = set(query_lower.split())
            
            for word in word_matches:
                weight = 1.0  # Base weight
                
                # Higher weight for longer words (more specific)
                if len(word) >= 6:
                    weight += 0.5
                elif len(word) >= 4:
                    weight += 0.2
                    
                # Higher weight for technical/specific terms
                if any(suffix in word for suffix in ['tion', 'ment', 'ness', 'ity']):
                    weight += 0.3
                    
                # Higher weight for numbers/dates
                if any(c.isdigit() for c in word):
                    weight += 0.4
                    
                weighted_score += weight
                
            return weighted_score / len(query_words)
            
        except Exception:
            return 0.0
        
    def _calculate_phrase_relevance_bonus(self, text_lower: str, query_lower: str) -> float:
        """Calculate bonus for exact phrase matches."""
        try:
            bonus = 0.0
            
            # Look for 2-word and 3-word phrases
            query_words = query_lower.split()
            
            for i in range(len(query_words) - 1):
                # 2-word phrases
                phrase_2 = f"{query_words[i]} {query_words[i+1]}"
                if phrase_2 in text_lower:
                    bonus += 0.3
                    
                # 3-word phrases
                if i < len(query_words) - 2:
                    phrase_3 = f"{query_words[i]} {query_words[i+1]} {query_words[i+2]}"
                    if phrase_3 in text_lower:
                        bonus += 0.5
                        
            return min(1.0, bonus)
            
        except Exception:
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

    async def _calculate_full_density(self, text: str) -> float:
        """Full parallel density calculation for longer texts."""
        # Split into sentences
        sentences = [s.strip() for s in text.split('.') if s.strip()]
        if not sentences:
            return 0.0
        
        # Start multiple async tasks in parallel
        tasks = {}
        
        # Start NLP processing pipeline
        tasks['nlp'] = self._process_nlp_features(text)
        
        # Wait for all tasks to complete
        results = await asyncio.gather(
            tasks['nlp'],
            return_exceptions=True
        )
        
        # Extract results (with error handling)
        nlp_features = {}
        
        if not isinstance(results[0], Exception):
            nlp_features = results[0]
        
        # OPTIMIZED: Component weights from Bayesian optimization (semantic_cohesion moved to orchestrator)
        weights = {
            'ne_density': 0.35697005328542325,
            'conceptual_surprise': 0.12266690258945108,
            'logical_complexity': 0.24064287712085727,
            'conceptual_bridging': 0.18685031356266707,
            'information_density': 0.0928698534416014
        }
        
        # Extract features with defaults
        ne_density = nlp_features.get('ne_density', 0.0)
        conceptual_surprise = nlp_features.get('conceptual_surprise', 0.5) 
        logical_complexity = nlp_features.get('logical_complexity', 0.0)
        conceptual_bridging = nlp_features.get('conceptual_bridging', 0.0)
        information_density = nlp_features.get('information_density', 0.0)
        
        # Calculate final score
        density_score = (
            ne_density * weights['ne_density'] +
            conceptual_surprise * weights['conceptual_surprise'] +
            logical_complexity * weights['logical_complexity'] +
            conceptual_bridging * weights['conceptual_bridging'] +
            information_density * weights['information_density']
        )

        raw_density = min(1.0, max(0.0, density_score))

        # Log raw density for analysis
        self.logger.debug(f"[DENSITY_DEBUG] Raw density before transformation: {raw_density:.3f}")
        # and log the density scores
        self.logger.debug(f"[DENSITY_DEBUG] NE Density: {ne_density:.3f}, "
                    f"Conceptual Surprise: {conceptual_surprise:.3f}, "
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
        OPTIMIZED: Bayesian-optimized transformation parameters for better LLM content discrimination.
        """
        try:
            # OPTIMIZED PARAMETERS: From Bayesian optimization
            low_threshold = 0.3691616082034308
            mid_threshold = 0.38941444011515397
            low_power = 2.9488588762066383
            low_scale = 0.3935247114507835
            mid_slope = 6.6566558364680715
            high_base = 0.48682834810794123
            high_scale = 9.877801387880073
            min_output = 0.08324797874360076
            max_output = 0.8723102125163544

            # Power law transformation with optimized parameters
            if raw_density < low_threshold:
                enhanced = (raw_density / low_threshold) ** low_power * low_scale
            elif raw_density < mid_threshold:
                enhanced = low_scale + (raw_density - low_threshold) * mid_slope
            else:
                enhanced = 0.7 + (1.0 - (high_base ** ((raw_density - mid_threshold) * high_scale))) * 0.3
            
            # Apply optimized output bounds
            enhanced = max(min_output, min(max_output, enhanced))
            
            # Log transformation details for analysis
            self.logger.debug(f"[DENSITY_TRANSFORM] Raw: {raw_density:.3f} → Enhanced: {enhanced:.3f}")
            
            return enhanced
            
        except Exception as e:
            self.logger.log_error(f"Density transformation failed: {e}")
            # Fallback to more aggressive sigmoid
            import math
            shifted = (raw_density - 0.35) * 16
            return 1 / (1 + math.exp(-shifted))

    @async_lru_cache(
        maxsize=1000, 
        ttl=3600, 
        key_func=lambda self, sentences: self._generate_text_cache_key('|||'.join(sentences), "cohesion")
    )
    async def _calculate_semantic_cohesion_cached(self, sentences: List[str]) -> float:
        """Cached version of semantic cohesion calculation."""
        return await self._calculate_semantic_cohesion_internal(sentences)

    async def _calculate_semantic_cohesion_internal(self, sentences: List[str]) -> float:
        """Calculate semantic cohesion with parallel embedding generation."""
        if len(sentences) < 2:
            return 0.4
        
        try:
            # Generate embeddings in parallel
            embedding_tasks = [
                self.embedding_manager.encode(s, normalize_embeddings=True)
                for s in sentences
            ]
            embeddings = await asyncio.gather(*embedding_tasks)
            
            # Calculate pairwise similarities
            similarities = []
            for i in range(len(embeddings)):
                for j in range(i+1, len(embeddings)):
                    try:
                        sim = self.embedding_manager.calculate_similarity(
                            embeddings[i],
                            embeddings[j]
                        )
                        similarities.append(sim)
                    except Exception as e:
                        self.logger.log_error(f"Similarity calculation failed: {e}")
                        continue
            
            return np.mean(similarities) if similarities else 0.4
            
        except Exception as e:
            self.logger.log_error(f"Semantic cohesion calculation failed: {e}")
            return 0.4

    async def _process_nlp_features(self, text: str) -> Dict[str, float]:
        """
        Process NLP features using enhanced semantic discriminators.
        """
        if not text or not NLP:
            return {
                'ne_density': 0.0,
                'conceptual_surprise': 0.0,
                'logical_complexity': 0.0,
                'conceptual_bridging': 0.0,
                'information_density': 0.0
            }

        try:
            doc = NLP(text)
            
            if not doc:
                return {
                    'ne_density': 0.0,
                    'conceptual_surprise': 0.0,
                    'logical_complexity': 0.0,
                    'conceptual_bridging': 0.0,
                    'information_density': 0.0
                }

            # Calculate all discriminators from the single SpaCy doc
            ne_density = self._calculate_named_entity_density(doc)
            conceptual_surprise = self._calculate_conceptual_surprise(doc)
            logical_complexity = self._calculate_logical_complexity(doc)
            conceptual_bridging = self._calculate_conceptual_bridging(doc)
            information_density = self._calculate_information_density(doc)

            return {
                'ne_density': ne_density,
                'conceptual_surprise': conceptual_surprise,
                'logical_complexity': logical_complexity,
                'conceptual_bridging': conceptual_bridging,
                'information_density': information_density
            }
        except Exception as e:
            self.logger.log_error(f"Enhanced SpaCy processing failed: {e}")
            return {
                'ne_density': 0.0,
                'conceptual_surprise': 0.5,
                'logical_complexity': 0.5,
                'conceptual_bridging': 0.5,
                'information_density': 0.5
            }

    def _calculate_conceptual_surprise(self, doc) -> float:
        """
        Calculate conceptual surprise - how linguistically unexpected are the patterns in this text?
        Measures deviation from typical language patterns across multiple axes.
        OPTIMIZED: Replaced abstraction heuristics with multi-axis surprise analysis.
        """
        if not doc:
            return 0.5
        
        # OPTIMIZED PARAMETERS: From Bayesian optimization
        syntactic_surprise_weight = 0.7199080112909861
        semantic_role_surprise_weight = 1.636863548712361
        discourse_surprise_weight = 1.5645776018689803
        concept_surprise_normalization = 2.9918127231770475

        # Component 1: Syntactic Surprise - unusual dependency structures and POS patterns
        syntactic_surprise = self._analyze_syntactic_surprise(doc) * syntactic_surprise_weight
        
        # Component 2: Semantic Role Surprise - unexpected agent-action-object combinations  
        semantic_role_surprise = self._analyze_semantic_role_surprise(doc) * semantic_role_surprise_weight
        
        # Component 3: Discourse Surprise - unusual information packaging and emphasis
        discourse_surprise = self._analyze_discourse_surprise(doc) * discourse_surprise_weight
        
        # Combine and normalize
        total_surprise = syntactic_surprise + semantic_role_surprise + discourse_surprise
        normalized = max(0.05, min(0.95, total_surprise / concept_surprise_normalization))
        
        return normalized
    
    def _analyze_syntactic_surprise(self, doc) -> float:
        """
        Analyze syntactic surprise - unusual dependency structures and POS patterns.
        Returns surprise score from 0 (expected) to 1 (highly unexpected).
        """
        if not doc:
            return 0.0
        
        surprise_score = 0.0
        total_tokens = len([token for token in doc if not token.is_space])
        
        if total_tokens == 0:
            return 0.0
        
        # Detect unusual dependency patterns
        rare_dependencies = {'advcl', 'acl:relcl', 'ccomp', 'xcomp', 'csubj', 'csubjpass'}
        complex_dependencies = 0
        
        for token in doc:
            # Count rare/complex dependency relations
            if token.dep_ in rare_dependencies:
                complex_dependencies += 1
            
            # Detect unusual POS sequences (simplified heuristic)
            if token.pos_ == 'ADV' and token.head.pos_ == 'ADJ':
                surprise_score += 0.1  # Adverb modifying adjective (somewhat unusual)
            elif token.pos_ == 'VERB' and len([child for child in token.children if child.pos_ == 'VERB']) >= 2:
                surprise_score += 0.2  # Verb with multiple verb children (unusual)
        
        # Normalize by text length
        dependency_surprise = min(1.0, complex_dependencies / max(1, total_tokens * 0.1))
        pattern_surprise = min(1.0, surprise_score / max(1, total_tokens * 0.05))
        
        return (dependency_surprise + pattern_surprise) / 2.0
    
    def _analyze_semantic_role_surprise(self, doc) -> float:
        """
        Analyze semantic role surprise - unexpected agent-action-object combinations.
        Returns surprise score from 0 (expected) to 1 (highly unexpected).
        """
        if not doc:
            return 0.0
        
        surprise_score = 0.0
        verb_count = 0
        
        for token in doc:
            if token.pos_ == 'VERB' and not token.is_stop:
                verb_count += 1
                
                # Find subjects and objects
                subjects = [child for child in token.children if child.dep_ in ['nsubj', 'nsubjpass']]
                objects = [child for child in token.children if child.dep_ in ['dobj', 'pobj', 'iobj']]
                
                # Detect unusual combinations 
                for subj in subjects:
                    if subj.ent_type_ == 'PERSON' and token.lemma_ in ['compile', 'execute', 'process']:
                        surprise_score += 0.1  # Person doing technical actions (slightly unusual)
                    elif subj.ent_type_ in ['ORG', 'PRODUCT'] and token.lemma_ in ['think', 'feel', 'believe']:
                        surprise_score += 0.3  # Organizations having emotions (more unusual)
                
                # Abstract subjects with concrete actions
                for subj in subjects:
                    if subj.pos_ == 'NOUN' and subj.lemma_ in ['idea', 'concept', 'thought'] and token.lemma_ in ['run', 'move', 'break']:
                        surprise_score += 0.2  # Abstract concepts doing physical actions
        
        return min(1.0, surprise_score / max(1, verb_count * 0.2)) if verb_count > 0 else 0.0
    
    def _analyze_discourse_surprise(self, doc) -> float:
        """
        Analyze discourse surprise - unusual information packaging and emphasis.
        Returns surprise score from 0 (expected) to 1 (highly unexpected).
        """
        if not doc:
            return 0.0
        
        surprise_score = 0.0
        sentence_count = len(list(doc.sents))
        
        if sentence_count == 0:
            return 0.0
        
        for sent in doc.sents:
            sent_tokens = [token for token in sent if not token.is_space]
            if not sent_tokens:
                continue
            
            # Detect unusual emphasis patterns
            caps_count = sum(1 for token in sent_tokens if token.text.isupper() and len(token.text) > 1)
            if caps_count > len(sent_tokens) * 0.3:  # More than 30% caps
                surprise_score += 0.3
            
            # Detect unusual sentence structures
            # Very short sentences with complex punctuation
            if len(sent_tokens) <= 3 and any(token.text in ['!', '?', '...'] for token in sent_tokens):
                surprise_score += 0.2
            
            # Very long sentences (potential run-ons)
            if len(sent_tokens) > 30:
                surprise_score += 0.2
            
            # Detect fronted elements
            if sent_tokens and sent_tokens[0].pos_ in ['ADV', 'SCONJ'] and sent_tokens[0].dep_ == 'advmod':
                surprise_score += 0.1  # Fronted adverbials
        
        return min(1.0, surprise_score / max(1, sentence_count * 0.3))
        
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
        
        # OPTIMIZED PARAMETERS: From Bayesian optimization
        dependency_weight = 1.284465024260733
        pos_weight = 1.878490677686282
        semantic_reasoning_weight = 0.5
        complexity_normalization = 19.014960800460248

        # 1. Analyze grammatical structure (dependency parsing)
        dependency_score = self._analyze_dependency_complexity(doc) * dependency_weight
        
        # 2. Analyze simple structural markers (POS tags)  
        pos_score = self._analyze_pos_complexity(doc) * pos_weight
        
        # 3. Analyze semantic and reasoning patterns
        semantic_score = self._analyze_semantic_complexity(doc) * semantic_reasoning_weight
        
        # Combine scores and normalize by sentence count
        total_complexity = dependency_score + pos_score + semantic_score
        complexity_per_sentence = total_complexity / sentence_count
        
        # OPTIMIZED: Enhanced normalization factor (was 8.0, now 19.015)
        return max(0.05, min(0.95, complexity_per_sentence / complexity_normalization))

    def _analyze_dependency_complexity(self, doc) -> float:
        """Analyze syntactic complexity using dependency parsing"""
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
        OPTIMIZED: Enhanced with Bayesian-optimized component weights and normalization.
        """
        if not doc or not len(doc):
            return 0.0

        # OPTIMIZED PARAMETERS: From Bayesian optimization
        syntactic_bridge_weight = 3.431218468386543
        semantic_bridge_weight = 2.404539956228189
        entity_bridge_weight = 1.5521114443634185
        bridge_normalization = 0.7189199492281488

        bridging_score = 0.0

        for token in doc:
            # Calculate bridging score from different linguistic facets
            bridging_score += self._get_syntactic_bridge_score(token) * syntactic_bridge_weight
            bridging_score += self._get_semantic_bridge_score(token) * semantic_bridge_weight

        entity_count = len(doc.ents)
        if entity_count >= 2:
            bridging_score += min(entity_count * entity_bridge_weight, 3.0)
        
        # Normalize by the number of tokens
        density = bridging_score / len(doc)
        
        # OPTIMIZED: Enhanced normalization factor
        return max(0.05, min(0.95, density / bridge_normalization))

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
        OPTIMIZED: Enhanced with Bayesian-optimized component weights and amplification.
        """
        if not doc or not len(doc):
            return 0.0
        
        # OPTIMIZED PARAMETERS: From Bayesian optimization
        technical_info_weight = 1.5739014204032535
        social_info_weight = 0.5
        factual_info_weight = 2.0
        info_density_amplification = 5.75187709484826

        total_score = 0.0
        
        for token in doc:
            # OPTIMIZED: Apply component-specific weights to different information types
            total_score += self._get_technical_info_score(token) * technical_info_weight
            total_score += self._get_social_info_score(token) * social_info_weight
            total_score += self._get_factual_info_score(token) * factual_info_weight
        
        # Normalize by document length
        raw_density = total_score / len(doc)
        
        # OPTIMIZED: Enhanced amplification factor prevents saturation in LLM content
        return max(0.05, min(0.95, raw_density / info_density_amplification))

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
        
        # OPTIMIZED: Bayesian optimization found 23.546 as optimal amplification
        amplification_factor = 23.5461056089212
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
