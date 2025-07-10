"""
semantic_calculator.py

Fully parameterized semantic calculator for neural optimization.
Every component, weight, and calculation method is configurable.
Optimized for performance with comprehensive error handling.
"""

import asyncio
import hashlib
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any, Union, Callable, Tuple
from abc import ABC, abstractmethod
import statistics
import numpy as np
from functools import lru_cache
import copy

# SpaCy import with graceful fallback
try:
    import spacy
    NLP = spacy.load("en_core_web_sm")
except (ImportError, OSError):
    print("Warning: SpaCy not available. NLP features will use fallbacks.")
    NLP = None

# Import embedding provider interface
try:
    from embedding_providers import EmbeddingProvider
except ImportError:
    # Fallback definition
    class EmbeddingProvider(ABC):
        @abstractmethod
        def encode(self, text: str) -> List[float]: pass
        @abstractmethod
        def calculate_similarity(self, vec1: List[float], vec2: List[float]) -> float: pass


@dataclass
class ComponentWeights:
    """Component weight configuration with validation."""
    topic_surprise: float = 0.25         # Replaces semantic_cohesion
    ne_density: float = 0.25  
    conceptual_surprise: float = 0.20     # Replaces abstraction_level
    logical_complexity: float = 0.15
    conceptual_bridging: float = 0.10
    information_density: float = 0.05
    
    def __post_init__(self):
        """Validate weights sum to 1.0 and are non-negative."""
        total = sum(asdict(self).values())
        if not (0.99 <= total <= 1.01):  # Allow small floating point errors
            raise ValueError(f"Component weights must sum to 1.0, got {total:.3f}")
        
        for name, weight in asdict(self).items():
            if weight < 0:
                raise ValueError(f"Weight {name} must be non-negative, got {weight}")
    
    def normalize(self) -> 'ComponentWeights':
        """Normalize weights to sum to 1.0."""
        total = sum(asdict(self).values())
        if total == 0:
            raise ValueError("Cannot normalize zero weights")
        
        factor = 1.0 / total
        return ComponentWeights(
            topic_surprise=self.topic_surprise * factor,
            ne_density=self.ne_density * factor,
            conceptual_surprise=self.conceptual_surprise * factor,
            logical_complexity=self.logical_complexity * factor,
            conceptual_bridging=self.conceptual_bridging * factor,
            information_density=self.information_density * factor
        )


@dataclass
class TransformationParams:
    """Exponential transformation parameters for optimization."""
    # Power law transformation parameters
    low_threshold: float = 0.35      # Below this: power law compression
    mid_threshold: float = 0.45      # Above this: exponential expansion
    low_power: float = 4.0           # Power for low-end compression  
    low_scale: float = 0.2           # Scale factor for low end
    mid_slope: float = 5.0           # Slope in middle region
    high_base: float = 0.5           # Base for exponential in high region
    high_scale: float = 15.0         # Scale for exponential steepness
    
    # Output bounds
    min_output: float = 0.05         # Minimum output value
    max_output: float = 0.95         # Maximum output value
    
    def __post_init__(self):
        """Validate transformation parameters."""
        if not (0.0 <= self.low_threshold <= 1.0):
            raise ValueError(f"low_threshold must be in [0,1], got {self.low_threshold}")
        if not (self.low_threshold <= self.mid_threshold <= 1.0):
            raise ValueError(f"mid_threshold must be >= low_threshold, got {self.mid_threshold}")
        if self.low_power <= 0:
            raise ValueError(f"low_power must be positive, got {self.low_power}")
        if not (0.0 <= self.min_output < self.max_output <= 1.0):
            raise ValueError(f"Invalid output bounds: [{self.min_output}, {self.max_output}]")


@dataclass
class DiscriminatorConfig:
    """Configuration for individual discriminator calculations."""
    # Named Entity Density
    ne_amplification_factor: float = 10.0    # How much to amplify NE density
    
    # Abstraction Level
    abstract_boost: float = 1.0              # Boost for abstract terms
    concrete_boost: float = 1.0              # Boost for concrete terms
    length_weight: float = 1.0               # Weight for word length heuristics
    
    # Logical Complexity  
    dependency_weight: float = 1.0           # Weight for dependency parsing
    pos_weight: float = 1.0                  # Weight for POS patterns
    semantic_reasoning_weight: float = 1.0   # Weight for reasoning patterns
    complexity_normalization: float = 8.0   # Normalization factor
    
    # Conceptual Bridging
    syntactic_bridge_weight: float = 2.0     # Weight for syntactic bridges
    semantic_bridge_weight: float = 2.0      # Weight for semantic bridges
    entity_bridge_weight: float = 0.8        # Weight for entity co-occurrence
    bridge_normalization: float = 0.5        # Normalization factor
    
    # Information Density
    technical_info_weight: float = 1.0       # Weight for technical information
    social_info_weight: float = 1.0          # Weight for social information
    factual_info_weight: float = 1.0         # Weight for factual information
    info_density_amplification: float = 3.0  # Amplification factor (like ne_density insight)
    info_normalization: float = 2.0          # Base normalization factor
    
    # Semantic Cohesion (embedding-based)
    cohesion_fallback_similarity: float = 0.5  # Fallback similarity value
    min_sentences_for_cohesion: int = 2      # Minimum sentences needed
    
    # Conceptual Surprise (replaces abstraction_level)
    syntactic_surprise_weight: float = 1.0      # Weight for syntactic pattern unusualness
    semantic_role_surprise_weight: float = 1.0  # Weight for semantic role unusualness
    discourse_surprise_weight: float = 1.0      # Weight for discourse pattern unusualness
    concept_surprise_normalization: float = 3.0 # Normalization factor for conceptual surprise
    
    # Topic Surprise (replaces semantic_cohesion) 
    topic_discontinuity_weight: float = 1.0     # Weight for topic jump detection
    density_surprise_weight: float = 1.0        # Weight for information density surprise
    structure_surprise_weight: float = 1.0      # Weight for narrative structure surprise
    topic_surprise_amplification: float = 8.0   # Amplification factor (like ne_density insight)
    topic_surprise_normalization: float = 3.0   # Base normalization factor
    min_sentences_for_topic_surprise: int = 2   # Minimum sentences needed for topic analysis


@dataclass
class CalculatorConfiguration:
    """Complete configuration for parameterized semantic calculator."""
    # Core component configuration
    component_weights: ComponentWeights = field(default_factory=ComponentWeights)
    transformation_params: TransformationParams = field(default_factory=TransformationParams)
    discriminator_config: DiscriminatorConfig = field(default_factory=DiscriminatorConfig)
    
    # Performance configuration
    enable_caching: bool = True
    cache_size: int = 1000
    enable_parallel_processing: bool = True
    max_workers: int = 4
    
    # Error handling configuration
    strict_validation: bool = True
    fallback_on_errors: bool = True
    log_calculation_details: bool = False
    
    # Alternative calculation methods
    cohesion_method: str = "embedding_similarity"  # "embedding_similarity", "word_overlap"
    ne_detection_method: str = "spacy"             # "spacy", "heuristic"
    complexity_method: str = "hybrid"              # "hybrid", "dependency_only", "semantic_only"
    
    def validate(self) -> None:
        """Validate entire configuration."""
        try:
            self.component_weights.__post_init__()
            self.transformation_params.__post_init__()
            
            if self.cache_size <= 0:
                raise ValueError(f"cache_size must be positive, got {self.cache_size}")
            if self.max_workers <= 0:
                raise ValueError(f"max_workers must be positive, got {self.max_workers}")
                
            valid_cohesion_methods = {"embedding_similarity", "word_overlap"}
            if self.cohesion_method not in valid_cohesion_methods:
                raise ValueError(f"cohesion_method must be one of {valid_cohesion_methods}")
                
            valid_ne_methods = {"spacy", "heuristic"}
            if self.ne_detection_method not in valid_ne_methods:
                raise ValueError(f"ne_detection_method must be one of {valid_ne_methods}")
                
            valid_complexity_methods = {"hybrid", "dependency_only", "semantic_only"}
            if self.complexity_method not in valid_complexity_methods:
                raise ValueError(f"complexity_method must be one of {valid_complexity_methods}")
                
        except Exception as e:
            if self.strict_validation:
                raise ValueError(f"Configuration validation failed: {e}")
            else:
                print(f"Warning: Configuration validation failed: {e}")
    
    def to_optimization_vector(self) -> np.ndarray:
        """Convert configuration to optimization vector for neural optimization."""
        weights = asdict(self.component_weights)
        transform = asdict(self.transformation_params)
        discriminator = asdict(self.discriminator_config)
        
        # Flatten all numerical parameters
        vector = []
        vector.extend(weights.values())
        vector.extend(transform.values())
        vector.extend(discriminator.values())
        
        return np.array(vector)
    
    @classmethod
    def from_optimization_vector(cls, vector: np.ndarray, base_config: Optional['CalculatorConfiguration'] = None) -> 'CalculatorConfiguration':
        """Create configuration from optimization vector."""
        if base_config is None:
            base_config = cls()
        
        config = copy.deepcopy(base_config)
        idx = 0
        
        # Reconstruct component weights (6 values)
        weights_dict = {}
        for name in asdict(config.component_weights).keys():
            weights_dict[name] = float(vector[idx])
            idx += 1
        config.component_weights = ComponentWeights(**weights_dict).normalize()
        
        # Reconstruct transformation params (9 values) 
        transform_dict = {}
        for name in asdict(config.transformation_params).keys():
            transform_dict[name] = float(vector[idx])
            idx += 1
        config.transformation_params = TransformationParams(**transform_dict)
        
        # Reconstruct discriminator config (remaining values)
        discriminator_dict = {}
        for name in asdict(config.discriminator_config).keys():
            if idx < len(vector):
                discriminator_dict[name] = float(vector[idx])
                idx += 1
        config.discriminator_config = DiscriminatorConfig(**discriminator_dict)
        
        return config


class ParameterizedSemanticCalculator:
    """
    Fully parameterized semantic calculator optimized for neural optimization.
    Every component is configurable and optimized for performance.
    """
    
    def __init__(self, config: CalculatorConfiguration, embedding_provider: Optional[EmbeddingProvider] = None):
        """
        Initialize with comprehensive configuration.
        
        Args:
            config: Complete calculator configuration
            embedding_provider: Optional embedding provider
        """
        self.config = config
        self.config.validate()
        self.embedding_provider = embedding_provider
        
        # Performance optimization: Caches
        if self.config.enable_caching:
            self._setup_caches()
        
        # Thread pool for parallel processing
        if self.config.enable_parallel_processing:
            self._executor = ThreadPoolExecutor(max_workers=self.config.max_workers)
        else:
            self._executor = None
        
        # Statistics tracking
        self._stats = {
            'calculations': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'errors': 0,
            'total_time': 0.0
        }
    
    def _setup_caches(self) -> None:
        """Setup LRU caches for performance."""
        # Text-based calculation cache
        self._density_cache = {}
        self._cohesion_cache = {}
        self._component_cache = {}
        
        # Cache limits
        self._max_cache_size = self.config.cache_size
    
    def _cache_key(self, text: str, method_name: str) -> str:
        """Generate cache key for text-based calculations."""
        text_hash = hashlib.md5(text.encode('utf-8')).hexdigest()[:12]
        config_hash = hashlib.md5(str(self.config.to_optimization_vector()).encode()).hexdigest()[:8]
        return f"{method_name}:{text_hash}:{config_hash}"
    
    def _get_cached_or_compute(self, cache_dict: Dict, cache_key: str, compute_func: Callable) -> Any:
        """Generic cache-or-compute pattern."""
        if not self.config.enable_caching:
            return compute_func()
        
        if cache_key in cache_dict:
            self._stats['cache_hits'] += 1
            return cache_dict[cache_key]
        
        # Compute and cache
        try:
            result = compute_func()
            
            # Manage cache size
            if len(cache_dict) >= self._max_cache_size:
                # Remove oldest entry (simple FIFO)
                oldest_key = next(iter(cache_dict))
                del cache_dict[oldest_key]
            
            cache_dict[cache_key] = result
            self._stats['cache_misses'] += 1
            return result
            
        except Exception as e:
            self._stats['errors'] += 1
            if self.config.fallback_on_errors:
                return self._get_fallback_result(cache_key, e)
            else:
                raise
    
    def _get_fallback_result(self, cache_key: str, error: Exception) -> Any:
        """Provide fallback results when calculations fail."""
        if self.config.log_calculation_details:
            print(f"Warning: Calculation failed for {cache_key}: {error}")
        
        # Return safe fallback based on cache key
        if "density" in cache_key:
            return self._get_zero_density_result()
        elif "cohesion" in cache_key:
            return self.config.discriminator_config.cohesion_fallback_similarity
        else:
            return 0.0
    
    def calculate_semantic_density(self, text_content: str) -> Dict[str, float]:
        """
        Calculate semantic density with full parameterization.
        Main entry point for density calculation.
        """
        start_time = time.time()
        self._stats['calculations'] += 1
        
        try:
            if not text_content or len(text_content.strip()) < 10:
                return self._get_zero_density_result()
            
            cache_key = self._cache_key(text_content, "semantic_density")
            
            def compute():
                return self._calculate_density_internal(text_content)
            
            result = self._get_cached_or_compute(self._density_cache, cache_key, compute)
            
            self._stats['total_time'] += time.time() - start_time
            return result
            
        except Exception as e:
            self._stats['errors'] += 1
            if self.config.fallback_on_errors:
                return self._get_zero_density_result()
            else:
                raise
    
    def _calculate_density_internal(self, text_content: str) -> Dict[str, float]:
        """Internal density calculation with parameterized components."""
        # Split into sentences for cohesion calculation
        sentences = [s.strip() for s in text_content.split('.') if s.strip()]
        
        # Calculate topic surprise (replaces semantic cohesion)
        if self.config.enable_parallel_processing and self._executor and len(sentences) > 1:
            topic_surprise = self._calculate_topic_surprise_parallel(sentences, text_content)
        else:
            topic_surprise = self._calculate_topic_surprise_sync(sentences, text_content)
        
        # Calculate other components (potentially in parallel)
        if self.config.enable_parallel_processing and self._executor:
            nlp_features = self._process_nlp_features_parallel(text_content)
        else:
            nlp_features = self._process_nlp_features_sync(text_content)
        
        # Extract component scores
        components = {
            'topic_surprise': topic_surprise,
            'ne_density': nlp_features.get('ne_density', 0.0),
            'conceptual_surprise': nlp_features.get('conceptual_surprise', 0.5),
            'logical_complexity': nlp_features.get('logical_complexity', 0.0),
            'conceptual_bridging': nlp_features.get('conceptual_bridging', 0.0),
            'information_density': nlp_features.get('information_density', 0.0)
        }
        
        # Apply parameterized weights
        weights = asdict(self.config.component_weights)
        raw_density = sum(components[name] * weights[name] for name in components.keys())
        raw_density = max(0.0, min(1.0, raw_density))
        
        # Apply parameterized transformation
        transformed_density = self._apply_parameterized_transformation(raw_density, text_content)
        
        return {
            'components': components,
            'raw_density': raw_density,
            'transformed_density': transformed_density,
            'weights': weights
        }
    
    def _calculate_cohesion_sync(self, sentences: List[str]) -> float:
        """Calculate semantic cohesion synchronously."""
        if len(sentences) < self.config.discriminator_config.min_sentences_for_cohesion:
            return self.config.discriminator_config.cohesion_fallback_similarity
        
        cache_key = self._cache_key('|||'.join(sentences), f"cohesion_{self.config.cohesion_method}")
        
        def compute():
            if self.config.cohesion_method == "embedding_similarity":
                return self._calculate_embedding_cohesion(sentences)
            elif self.config.cohesion_method == "word_overlap":
                return self._calculate_word_overlap_cohesion(sentences)
            else:
                raise ValueError(f"Unknown cohesion method: {self.config.cohesion_method}")
        
        return self._get_cached_or_compute(self._cohesion_cache, cache_key, compute)
    
    def _calculate_cohesion_parallel(self, sentences: List[str]) -> float:
        """Calculate semantic cohesion with parallel embedding generation.""" 
        if len(sentences) < self.config.discriminator_config.min_sentences_for_cohesion:
            return self.config.discriminator_config.cohesion_fallback_similarity
        
        if self.config.cohesion_method == "word_overlap":
            return self._calculate_word_overlap_cohesion(sentences)
        
        # For embedding similarity, we need the embedding provider
        if not self.embedding_provider:
            return self._calculate_word_overlap_cohesion(sentences)
        
        try:
            # Generate embeddings in parallel
            futures = []
            for sentence in sentences:
                future = self._executor.submit(self.embedding_provider.encode, sentence)
                futures.append(future)
            
            # Collect embeddings
            embeddings = []
            for future in as_completed(futures, timeout=30):
                try:
                    embedding = future.result()
                    embeddings.append(embedding)
                except Exception as e:
                    if self.config.log_calculation_details:
                        print(f"Warning: Embedding generation failed: {e}")
                    continue
            
            if len(embeddings) < 2:
                return self.config.discriminator_config.cohesion_fallback_similarity
            
            # Calculate pairwise similarities
            similarities = []
            for i in range(len(embeddings)):
                for j in range(i+1, len(embeddings)):
                    try:
                        sim = self.embedding_provider.calculate_similarity(embeddings[i], embeddings[j])
                        similarities.append(sim)
                    except Exception:
                        continue
            
            return statistics.mean(similarities) if similarities else self.config.discriminator_config.cohesion_fallback_similarity
            
        except Exception as e:
            if self.config.log_calculation_details:
                print(f"Warning: Parallel cohesion calculation failed: {e}")
            return self._calculate_word_overlap_cohesion(sentences)
    
    def _calculate_embedding_cohesion(self, sentences: List[str]) -> float:
        """Calculate cohesion using embedding similarity."""
        if not self.embedding_provider:
            return self._calculate_word_overlap_cohesion(sentences)
        
        try:
            # Generate embeddings
            embeddings = []
            for sentence in sentences:
                embedding = self.embedding_provider.encode(sentence)
                embeddings.append(embedding)
            
            if len(embeddings) < 2:
                return self.config.discriminator_config.cohesion_fallback_similarity
            
            # Calculate pairwise similarities
            similarities = []
            for i in range(len(embeddings)):
                for j in range(i+1, len(embeddings)):
                    sim = self.embedding_provider.calculate_similarity(embeddings[i], embeddings[j])
                    similarities.append(sim)
            
            return statistics.mean(similarities) if similarities else self.config.discriminator_config.cohesion_fallback_similarity
            
        except Exception as e:
            if self.config.log_calculation_details:
                print(f"Warning: Embedding cohesion failed: {e}")
            return self._calculate_word_overlap_cohesion(sentences)
    
    def _calculate_word_overlap_cohesion(self, sentences: List[str]) -> float:
        """Fallback cohesion calculation using word overlap."""
        similarities = []
        
        for i in range(len(sentences)):
            for j in range(i+1, len(sentences)):
                words_i = set(sentences[i].lower().split())
                words_j = set(sentences[j].lower().split())
                
                if not words_i or not words_j:
                    similarities.append(0.0)
                    continue
                
                intersection = len(words_i.intersection(words_j))
                union = len(words_i.union(words_j))
                similarity = intersection / union if union > 0 else 0.0
                similarities.append(similarity)
        
        return statistics.mean(similarities) if similarities else self.config.discriminator_config.cohesion_fallback_similarity
    
    def _calculate_topic_surprise_sync(self, sentences: List[str], full_text: str) -> float:
        """Calculate topic surprise synchronously - how unpredictable is the information flow?"""
        if len(sentences) < self.config.discriminator_config.min_sentences_for_topic_surprise:
            return 0.5  # Default moderate surprise for single sentences
        
        cache_key = self._cache_key(full_text, f"topic_surprise_{self.config.discriminator_config.topic_discontinuity_weight}")
        
        def compute():
            return self._calculate_topic_surprise_internal(sentences, full_text)
        
        return self._get_cached_or_compute(self._cohesion_cache, cache_key, compute)
    
    def _calculate_topic_surprise_parallel(self, sentences: List[str], full_text: str) -> float:
        """Calculate topic surprise with parallel processing where possible."""
        # For now, topic surprise calculation is primarily sequential due to dependencies
        # Future enhancement: parallelize sentence-level analysis
        return self._calculate_topic_surprise_sync(sentences, full_text)
    
    def _calculate_topic_surprise_internal(self, sentences: List[str], full_text: str) -> float:
        """
        Internal topic surprise calculation with parameterized components.
        Measures how unpredictable the information sequencing is.
        """
        config = self.config.discriminator_config
        
        # Component 1: Topic Discontinuity - sudden changes in semantic space without bridging
        topic_discontinuity = self._analyze_topic_discontinuity(sentences) * config.topic_discontinuity_weight
        
        # Component 2: Information Density Surprise - unusual packing of information 
        density_surprise = self._analyze_information_density_surprise(full_text) * config.density_surprise_weight
        
        # Component 3: Narrative Structure Surprise - unusual story/argument progression
        structure_surprise = self._analyze_narrative_structure_surprise(sentences) * config.structure_surprise_weight
        
        # Combine surprise signals - following ne_density pattern exactly
        raw_surprise = topic_discontinuity + density_surprise + structure_surprise
        
        # Apply amplification directly to raw signal (like ne_density)
        # Meaningful topic surprise levels: ~0.1-0.3 raw = significant disruption for memory
        # Start with conservative 3x amplification to test the pattern
        amplified_surprise = min(1.0, raw_surprise * config.topic_surprise_amplification)
        
        return amplified_surprise
    
    def _process_nlp_features_sync(self, text: str) -> Dict[str, float]:
        """Process NLP features synchronously with parameterized calculations."""
        if not text:
            return self._get_zero_nlp_features()
        
        cache_key = self._cache_key(text, f"nlp_features_{self.config.ne_detection_method}_{self.config.complexity_method}")
        
        def compute():
            return self._compute_nlp_features(text)
        
        return self._get_cached_or_compute(self._component_cache, cache_key, compute)
    
    def _process_nlp_features_parallel(self, text: str) -> Dict[str, float]:
        """Process NLP features with parallel computation where possible."""
        # For now, most NLP features require sequential processing due to SpaCy
        # But we can parallelize independent calculations
        return self._process_nlp_features_sync(text)
    
    def _compute_nlp_features(self, text: str) -> Dict[str, float]:
        """Compute NLP features with parameterized methods."""
        if not NLP and self.config.ne_detection_method == "spacy":
            # Fall back to heuristic methods
            return self._compute_heuristic_nlp_features(text)
        
        if not NLP:
            return self._compute_heuristic_nlp_features(text)
        
        try:
            doc = NLP(text)
            
            return {
                'ne_density': self._calculate_ne_density_parameterized(doc),
                'conceptual_surprise': self._calculate_conceptual_surprise_parameterized(doc),
                'logical_complexity': self._calculate_logical_complexity_parameterized(doc),
                'conceptual_bridging': self._calculate_conceptual_bridging_parameterized(doc),
                'information_density': self._calculate_information_density_parameterized(doc)
            }
            
        except Exception as e:
            if self.config.log_calculation_details:
                print(f"Warning: SpaCy processing failed: {e}")
            return self._compute_heuristic_nlp_features(text)
    
    def _compute_heuristic_nlp_features(self, text: str) -> Dict[str, float]:
        """Compute NLP features using heuristic methods when SpaCy unavailable."""
        words = text.split()
        
        return {
            'ne_density': self._heuristic_ne_density(words),
            'conceptual_surprise': self._heuristic_conceptual_surprise(words, text),
            'logical_complexity': self._heuristic_logical_complexity(text),
            'conceptual_bridging': self._heuristic_conceptual_bridging(words),
            'information_density': self._heuristic_information_density(words)
        }
    
    # Parameterized discriminator calculations
    def _calculate_ne_density_parameterized(self, doc) -> float:
        """Calculate NE density with parameterized amplification."""
        if not doc:
            return 0.0
        
        tokens = [token for token in doc if not token.is_space]
        if not tokens:
            return 0.0
        
        raw_density = len(doc.ents) / len(tokens)
        amplified = min(1.0, raw_density * self.config.discriminator_config.ne_amplification_factor)
        
        return amplified
    
    def _calculate_conceptual_surprise_parameterized(self, doc) -> float:
        """
        Calculate conceptual surprise - how linguistically unexpected are the patterns in this text?
        Measures deviation from typical language patterns across multiple axes.
        """
        if not doc:
            return 0.5
        
        config = self.config.discriminator_config
        
        # Component 1: Syntactic Surprise - unusual dependency structures and POS patterns
        syntactic_surprise = self._analyze_syntactic_surprise(doc) * config.syntactic_surprise_weight
        
        # Component 2: Semantic Role Surprise - unexpected agent-action-object combinations  
        semantic_role_surprise = self._analyze_semantic_role_surprise(doc) * config.semantic_role_surprise_weight
        
        # Component 3: Discourse Surprise - unusual information packaging and emphasis
        discourse_surprise = self._analyze_discourse_surprise(doc) * config.discourse_surprise_weight
        
        # Combine and normalize
        total_surprise = syntactic_surprise + semantic_role_surprise + discourse_surprise
        normalized = max(0.05, min(0.95, total_surprise / config.concept_surprise_normalization))
        
        return normalized
    
    def _calculate_logical_complexity_parameterized(self, doc) -> float:
        """Calculate logical complexity with parameterized method selection."""
        if not doc or not len(doc):
            return 0.0
        
        sentence_count = len(list(doc.sents))
        if sentence_count == 0:
            return 0.0
        
        config = self.config.discriminator_config
        
        total_complexity = 0.0
        
        if self.config.complexity_method in ["hybrid", "dependency_only"]:
            dependency_score = self._analyze_dependency_complexity_parameterized(doc)
            total_complexity += dependency_score * config.dependency_weight
        
        if self.config.complexity_method in ["hybrid"]:
            pos_score = self._analyze_pos_complexity_parameterized(doc)
            total_complexity += pos_score * config.pos_weight
        
        if self.config.complexity_method in ["hybrid", "semantic_only"]:
            semantic_score = self._analyze_semantic_complexity_parameterized(doc)
            total_complexity += semantic_score * config.semantic_reasoning_weight
        
        complexity_per_sentence = total_complexity / sentence_count
        normalized = max(0.05, min(0.95, complexity_per_sentence / config.complexity_normalization))
        
        return normalized
    
    def _analyze_dependency_complexity_parameterized(self, doc) -> float:
        """Analyze dependency complexity with parameterized weights."""
        complexity = 0.0
        for token in doc:
            if token.dep_ in ['advcl', 'acl', 'ccomp', 'xcomp']:
                complexity += 1.5
            elif token.dep_ == 'relcl':
                complexity += 1.8
            elif token.dep_ == 'conj':
                complexity += 1.0
            elif token.dep_ == 'mark' and token.head.pos_ == 'VERB':
                complexity += 1.2
        return complexity
    
    def _analyze_pos_complexity_parameterized(self, doc) -> float:
        """Analyze POS complexity with parameterized weights."""
        complexity = 0.0
        pos_tags = [token.pos_ for token in doc]
        complexity += pos_tags.count('SCONJ') * 1.5
        complexity += pos_tags.count('CCONJ') * 1.0
        return complexity
    
    def _analyze_semantic_complexity_parameterized(self, doc) -> float:
        """Analyze semantic complexity with parameterized weights."""
        complexity = 0.0
        
        causal_verbs = {'cause', 'result', 'lead', 'make', 'force', 'imply', 'indicate'}
        realization_verbs = {'realize', 'notice', 'understand', 'recognize', 'discover', 'learn'}
        pattern_nouns = {'pattern', 'trend', 'similarity', 'connection', 'relationship', 'theme'}
        
        for token in doc:
            if token.pos_ == 'VERB' and token.lemma_ in causal_verbs:
                complexity += 1.8
            elif token.pos_ == 'VERB' and token.lemma_ in realization_verbs:
                complexity += 1.5
            elif token.pos_ == 'NOUN' and token.lemma_ in pattern_nouns:
                complexity += 2.0
            elif token.tag_ == 'MD':
                if token.lemma_ in ['must', 'will']:
                    complexity += 0.8
                elif token.lemma_ in ['might', 'could', 'may', 'should']:
                    complexity += 1.2
            elif token.pos_ == 'ADP' and token.text.lower() in {'than', 'like', 'unlike'}:
                complexity += 1.0
        
        return complexity
    
    def _calculate_conceptual_bridging_parameterized(self, doc) -> float:
        """Calculate conceptual bridging with parameterized weights."""
        if not doc or not len(doc):
            return 0.0
        
        config = self.config.discriminator_config
        bridging_score = 0.0
        
        for token in doc:
            bridging_score += self._get_syntactic_bridge_score_parameterized(token) * config.syntactic_bridge_weight
            bridging_score += self._get_semantic_bridge_score_parameterized(token) * config.semantic_bridge_weight
        
        entity_count = len(doc.ents)
        if entity_count >= 2:
            bridging_score += min(entity_count * config.entity_bridge_weight, 3.0)
        
        density = bridging_score / len(doc)
        return max(0.05, min(0.95, density / config.bridge_normalization))
    
    def _get_syntactic_bridge_score_parameterized(self, token) -> float:
        """Get syntactic bridge score with parameterized weights."""
        if token.pos_ in ['DET', 'PRON'] and token.lemma_ in {'this', 'that', 'these', 'those'}:
            return 1.5
        elif token.pos_ == 'PRON' and token.lemma_ in {'we', 'they', 'our', 'their'}:
            return 1.0
        elif token.pos_ == 'SCONJ' and token.lemma_ in {'as', 'like', 'while', 'since', 'because'}:
            return 1.2
        return 0.0
    
    def _get_semantic_bridge_score_parameterized(self, token) -> float:
        """Get semantic bridge score with parameterized weights."""
        if token.pos_ == 'VERB' and token.lemma_ in {'compare', 'relate', 'connect', 'remind', 'associate'}:
            return 2.0
        elif token.pos_ == 'ADJ' and token.lemma_ in {'similar', 'different', 'related', 'typical'}:
            return 1.8
        elif token.pos_ == 'ADV' and token.lemma_ in {'previously', 'earlier', 'consequently', 'therefore', 'however'}:
            return 1.5
        elif token.pos_ == 'NOUN' and token.lemma_ in {'pattern', 'trend', 'similarity', 'connection', 'relationship'}:
            return 2.5
        return 0.0
    
    def _calculate_information_density_parameterized(self, doc) -> float:
        """Calculate information density with parameterized weights."""
        if not doc or not len(doc):
            return 0.0
        
        config = self.config.discriminator_config
        total_score = 0.0
        
        for token in doc:
            total_score += self._get_technical_info_score_parameterized(token) * config.technical_info_weight
            total_score += self._get_social_info_score_parameterized(token) * config.social_info_weight
            total_score += self._get_factual_info_score_parameterized(token) * config.factual_info_weight
        
        # Apply amplification directly to raw signal (following ne_density pattern exactly)
        # Meaningful info density: ~0.2-0.5 raw score per token = high information content  
        # Start with conservative 2x amplification to test the pattern
        raw_density = total_score / len(doc)
        amplified_density = min(1.0, raw_density * config.info_density_amplification)
        
        return amplified_density
    
    def _get_technical_info_score_parameterized(self, token) -> float:
        """Get technical info score with parameterized weights."""
        score = 0.0
        
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
    
    def _get_social_info_score_parameterized(self, token) -> float:
        """Get social info score with parameterized weights."""
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
    
    def _get_factual_info_score_parameterized(self, token) -> float:
        """Get factual info score with parameterized weights."""
        score = 0.0
        
        if token.pos_ == 'PROPN' and not token.ent_type_:
            score += 1.6
        elif token.pos_ == 'NUM':
            score += 1.8
        elif token.pos_ == 'DET' and token.text.lower() in ['this', 'that', 'these', 'those']:
            score += 0.8
        elif token.pos_ == 'VERB' and token.tag_ == 'MD':
            if token.text.lower() in ['must', 'will', 'did']:
                score += 1.0
            else:
                score += 0.5
        
        return score
    
    # Surprise-based component helper methods
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
                
                # Detect unusual combinations (simplified heuristics)
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
            
            # Detect fronted elements (simplified)
            if sent_tokens and sent_tokens[0].pos_ in ['ADV', 'SCONJ'] and sent_tokens[0].dep_ == 'advmod':
                surprise_score += 0.1  # Fronted adverbials
        
        return min(1.0, surprise_score / max(1, sentence_count * 0.3))
    
    def _analyze_topic_discontinuity(self, sentences: List[str]) -> float:
        """
        Analyze topic discontinuity - sudden changes in semantic space without bridging.
        Returns discontinuity score from 0 (smooth flow) to 1 (highly discontinuous).
        """
        if len(sentences) < 2:
            return 0.0
        
        discontinuity_score = 0.0
        
        for i in range(len(sentences) - 1):
            curr_words = set(sentences[i].lower().split())
            next_words = set(sentences[i + 1].lower().split())
            
            # Remove stop words (simplified)
            stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'must', 'shall', 'can'}
            curr_content = curr_words - stop_words
            next_content = next_words - stop_words
            
            if not curr_content or not next_content:
                continue
            
            # Calculate word overlap
            overlap = len(curr_content.intersection(next_content))
            total_unique = len(curr_content.union(next_content))
            
            if total_unique > 0:
                overlap_ratio = overlap / total_unique
                # Low overlap = high discontinuity
                discontinuity_score += (1.0 - overlap_ratio)
        
        return min(1.0, discontinuity_score / max(1, len(sentences) - 1))
    
    def _analyze_information_density_surprise(self, text: str) -> float:
        """
        Analyze information density surprise - unusual packing of information.
        Returns surprise score from 0 (expected density) to 1 (highly unexpected density).
        """
        if not text:
            return 0.0
        
        words = text.split()
        if not words:
            return 0.0
        
        # Calculate various density metrics
        numbers = sum(1 for word in words if any(c.isdigit() for c in word))
        capitals = sum(1 for word in words if word and word[0].isupper())
        long_words = sum(1 for word in words if len(word) >= 8)
        punctuation = sum(1 for char in text if char in '.,;:!?()[]{}')
        
        # Detect unusual density patterns
        surprise_score = 0.0
        
        # Very high number density (more than 20% numbers)
        number_density = numbers / len(words)
        if number_density > 0.2:
            surprise_score += min(0.4, number_density)
        
        # Very high capitalization (more than 30% capitals, excluding start of sentences)
        capital_density = capitals / len(words)
        if capital_density > 0.3:
            surprise_score += min(0.3, capital_density - 0.3)
        
        # Unusual punctuation density
        punct_density = punctuation / len(text)
        if punct_density > 0.1:  # More than 10% punctuation
            surprise_score += min(0.3, punct_density)
        
        # Very sparse information (mostly short common words)
        if long_words == 0 and len(words) > 5:
            surprise_score += 0.2
        
        return min(1.0, surprise_score)
    
    def _analyze_narrative_structure_surprise(self, sentences: List[str]) -> float:
        """
        Analyze narrative structure surprise - unusual story/argument progression.
        Returns surprise score from 0 (expected structure) to 1 (highly unexpected structure).
        """
        if len(sentences) < 2:
            return 0.0
        
        surprise_score = 0.0
        
        # Detect temporal inconsistencies (simplified)
        time_markers = {
            'past': ['yesterday', 'before', 'earlier', 'previously', 'then', 'after'],
            'present': ['now', 'currently', 'today', 'this'],
            'future': ['tomorrow', 'later', 'next', 'will', 'going to', 'plan to']
        }
        
        sentence_times = []
        for sentence in sentences:
            sentence_lower = sentence.lower()
            time_score = {'past': 0, 'present': 0, 'future': 0}
            
            for time_type, markers in time_markers.items():
                for marker in markers:
                    if marker in sentence_lower:
                        time_score[time_type] += 1
            
            # Determine dominant time frame
            if sum(time_score.values()) > 0:
                dominant_time = max(time_score.items(), key=lambda x: x[1])[0]
                sentence_times.append(dominant_time)
            else:
                sentence_times.append('neutral')
        
        # Detect non-linear time progression
        if len(sentence_times) >= 3:
            for i in range(len(sentence_times) - 2):
                # Look for past->future->past or future->past->future patterns
                if sentence_times[i] == 'past' and sentence_times[i+1] == 'future' and sentence_times[i+2] == 'past':
                    surprise_score += 0.3
                elif sentence_times[i] == 'future' and sentence_times[i+1] == 'past' and sentence_times[i+2] == 'future':
                    surprise_score += 0.3
        
        # Detect abrupt topic changes (simplified by sentence length variation)
        lengths = [len(sentence.split()) for sentence in sentences]
        if len(lengths) >= 3:
            length_variance = statistics.stdev(lengths) if len(lengths) > 1 else 0
            avg_length = statistics.mean(lengths)
            if avg_length > 0 and length_variance / avg_length > 1.0:  # High variation in sentence length
                surprise_score += 0.2
        
        return min(1.0, surprise_score)
    
    # Heuristic methods for fallback
    def _heuristic_ne_density(self, words: List[str]) -> float:
        """Heuristic NE density calculation.""" 
        if not words:
            return 0.0
        
        # Simple heuristics for named entities
        capitalized = sum(1 for word in words if word and word[0].isupper())
        return min(1.0, (capitalized / len(words)) * self.config.discriminator_config.ne_amplification_factor)
    
    def _heuristic_conceptual_surprise(self, words: List[str], text: str) -> float:
        """
        Heuristic conceptual surprise calculation for when SpaCy is unavailable.
        Simplified pattern-based surprise detection.
        """
        if not words:
            return 0.5
        
        surprise_score = 0.0
        
        # Pattern 1: Unusual word combinations (simplified)
        unusual_pairs = {
            ('thought', 'ran'), ('idea', 'jumped'), ('concept', 'fell'),  # Abstract + physical verbs
            ('organization', 'felt'), ('company', 'believed'),  # Org + emotion verbs
            ('system', 'hoped'), ('process', 'dreamed')  # Technical + mental verbs
        }
        
        word_list = [w.lower() for w in words]
        for i in range(len(word_list) - 1):
            if (word_list[i], word_list[i+1]) in unusual_pairs:
                surprise_score += 0.3
        
        # Pattern 2: Unusual punctuation patterns
        unusual_punct_count = text.count('!!!') + text.count('???') + text.count('...')
        if unusual_punct_count > 0:
            surprise_score += min(0.2, unusual_punct_count * 0.1)
        
        # Pattern 3: Mixed case patterns (simplified)
        mixed_case_words = sum(1 for word in words if any(c.isupper() for c in word) and any(c.islower() for c in word))
        if mixed_case_words > 0:
            surprise_score += min(0.2, mixed_case_words / len(words))
        
        # Pattern 4: Very short or very long words in unusual contexts
        very_short = sum(1 for word in words if len(word) <= 2 and word.isalpha())
        very_long = sum(1 for word in words if len(word) >= 15)
        unusual_length_ratio = (very_short + very_long) / len(words)
        if unusual_length_ratio > 0.1:
            surprise_score += min(0.3, unusual_length_ratio)
        
        return min(0.95, max(0.05, surprise_score))
    
    def _heuristic_logical_complexity(self, text: str) -> float:
        """Heuristic logical complexity calculation."""
        complexity_markers = ['because', 'therefore', 'however', 'although', 'while', 'since', 'if', 'then']
        marker_count = sum(1 for marker in complexity_markers if marker in text.lower())
        
        sentence_count = len([s for s in text.split('.') if s.strip()])
        if sentence_count == 0:
            return 0.0
        
        return min(0.95, (marker_count / sentence_count) / 2.0)
    
    def _heuristic_conceptual_bridging(self, words: List[str]) -> float:
        """Heuristic conceptual bridging calculation."""
        bridge_words = {'this', 'that', 'these', 'those', 'similar', 'different', 'like', 'unlike', 'connect', 'relate'}
        bridge_count = sum(1 for word in words if word.lower() in bridge_words)
        
        if not words:
            return 0.0
        
        return min(0.95, (bridge_count / len(words)) * 10.0)
    
    def _heuristic_information_density(self, words: List[str]) -> float:
        """Heuristic information density calculation."""
        if not words:
            return 0.0
        
        # Simple heuristics
        long_words = sum(1 for word in words if len(word) >= 8)
        numbers = sum(1 for word in words if any(c.isdigit() for c in word))
        
        density = (long_words + numbers * 2) / len(words)
        return min(0.95, density)
    
    def _apply_parameterized_transformation(self, raw_density: float, text: str) -> float:
        """Apply parameterized exponential transformation."""
        try:
            params = self.config.transformation_params
            
            if raw_density < params.low_threshold:
                # Power law compression for low values
                enhanced = (raw_density / params.low_threshold) ** params.low_power * params.low_scale
            elif raw_density < params.mid_threshold:
                # Linear middle region
                enhanced = params.low_scale + (raw_density - params.low_threshold) * params.mid_slope
            else:
                # Exponential expansion for high values
                enhanced = 0.7 + (1.0 - (params.high_base ** ((raw_density - params.mid_threshold) * params.high_scale))) * 0.3
            
            # Apply output bounds
            enhanced = max(params.min_output, min(params.max_output, enhanced))
            
            return enhanced
            
        except Exception as e:
            if self.config.log_calculation_details:
                print(f"Warning: Parameterized transformation failed: {e}")
            
            # Fallback to simple sigmoid
            import math
            shifted = (raw_density - 0.35) * 16
            return 1 / (1 + math.exp(-shifted))
    
    def _get_zero_density_result(self) -> Dict[str, float]:
        """Return zero density result structure."""
        return {
            'components': {
                'topic_surprise': 0.0,
                'ne_density': 0.0,
                'conceptual_surprise': 0.0,
                'logical_complexity': 0.0,
                'conceptual_bridging': 0.0,
                'information_density': 0.0
            },
            'raw_density': 0.0,
            'transformed_density': 0.0,
            'weights': asdict(self.config.component_weights)
        }
    
    def _get_zero_nlp_features(self) -> Dict[str, float]:
        """Return zero NLP features."""
        return {
            'ne_density': 0.0,
            'conceptual_surprise': 0.5,
            'logical_complexity': 0.0,
            'conceptual_bridging': 0.0,
            'information_density': 0.0
        }
    
    def calculate_similarity_metrics(
        self,
        text_content: str,
        embedding: List[float],
        query_text: Optional[str] = None,
        query_embedding: Optional[List[float]] = None
    ) -> Dict[str, float]:
        """Calculate similarity metrics using configured providers."""
        metrics = {}
        
        if query_embedding is not None and self.embedding_provider:
            metrics['embedding_similarity'] = self.embedding_provider.calculate_similarity(
                embedding, query_embedding
            )
        
        if query_text:
            metrics['text_relevance'] = self._calculate_text_relevance_internal(
                text_content, query_text
            )
        
        return metrics
    
    def _calculate_text_relevance_internal(self, text: str, query: str) -> float:
        """Calculate text relevance using keyword overlap."""
        try:
            if not text or not query:
                return 0.0
                
            text_lower = text.lower()
            query_lower = query.lower()
            
            query_words = set(query_lower.split())
            text_words = set(text_lower.split())
            if not query_words:
                return 0.0
                
            word_matches = query_words.intersection(text_words)
            keyword_score = len(word_matches) / len(query_words)
            
            return keyword_score
            
        except Exception as e:
            if self.config.log_calculation_details:
                print(f"Text relevance calculation failed: {e}")
            return 0.0
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get performance and usage statistics."""
        total_requests = self._stats['cache_hits'] + self._stats['cache_misses']
        cache_hit_rate = self._stats['cache_hits'] / total_requests if total_requests > 0 else 0.0
        avg_time = self._stats['total_time'] / self._stats['calculations'] if self._stats['calculations'] > 0 else 0.0
        
        return {
            **self._stats,
            'cache_hit_rate': cache_hit_rate,
            'average_calculation_time': avg_time,
            'cache_sizes': {
                'density_cache': len(self._density_cache) if hasattr(self, '_density_cache') else 0,
                'cohesion_cache': len(self._cohesion_cache) if hasattr(self, '_cohesion_cache') else 0,
                'component_cache': len(self._component_cache) if hasattr(self, '_component_cache') else 0
            }
        }
    
    def clear_caches(self) -> None:
        """Clear all caches to free memory."""
        if hasattr(self, '_density_cache'):
            self._density_cache.clear()
        if hasattr(self, '_cohesion_cache'):
            self._cohesion_cache.clear()
        if hasattr(self, '_component_cache'):
            self._component_cache.clear()
    
    def __del__(self):
        """Cleanup resources."""
        if hasattr(self, '_executor') and self._executor:
            self._executor.shutdown(wait=False)


# Factory functions for common configurations
def create_baseline_config() -> CalculatorConfiguration:
    """Create baseline configuration matching original implementation."""
    return CalculatorConfiguration()

def create_optimized_config() -> CalculatorConfiguration:
    """Create configuration optimized for performance."""
    return CalculatorConfiguration(
        enable_parallel_processing=True,
        max_workers=6,
        cache_size=2000,
        cohesion_method="word_overlap",  # Faster than embeddings
        complexity_method="semantic_only"  # Skip expensive dependency parsing
    )

def create_comprehensive_config() -> CalculatorConfiguration:
    """Create configuration for maximum analytical depth."""
    return CalculatorConfiguration(
        enable_parallel_processing=True,
        max_workers=4,
        cache_size=5000,
        cohesion_method="embedding_similarity",
        complexity_method="hybrid",
        log_calculation_details=True
    )

def create_config_from_vector(vector: np.ndarray) -> CalculatorConfiguration:
    """Create configuration from optimization vector for neural optimization."""
    return CalculatorConfiguration.from_optimization_vector(vector)