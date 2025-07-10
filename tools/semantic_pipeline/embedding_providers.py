"""
testing/embedding_providers.py

Modular embedding providers for easy model swapping during testing.
Replicates exact logic from embedding_manager.py but with clean interfaces.
"""

from typing import List, Optional, Dict, Any
from abc import ABC, abstractmethod
import hashlib
import numpy as np

# Optional imports with graceful fallbacks
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("Warning: PyTorch not available - local models disabled")

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    print("Warning: sentence-transformers not available - local models disabled")


class EmbeddingProvider(ABC):
    """Abstract interface for embedding providers - enables easy model swapping."""
    
    @abstractmethod
    def encode(self, text: str) -> List[float]:
        """Encode text to embedding vector."""
        pass
    
    @abstractmethod
    def calculate_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between vectors."""
        pass
    
    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return human-readable model identifier."""
        pass
    
    @property
    @abstractmethod
    def embedding_dimension(self) -> int:
        """Return embedding vector dimension."""
        pass
    
    def preprocess_text(self, text: str, max_length: int = 512) -> str:
        """
        EXACT preprocessing from embedding_manager.py.
        Handles truncation and normalization.
        """
        if not text:
            return ""
            
        try:
            text = text.strip().lower()
            
            # EXACT logic from original: handle long texts by taking meaningful chunks
            if len(text) > max_length:
                # Try to split on sentence boundaries first
                sentences = text.split('.')
                if len(sentences) > 1:
                    mid = len(sentences) // 2
                    text = '. '.join(sentences[:mid//2] + 
                                   ['...'] + 
                                   sentences[-mid//2:])
                else:
                    # If no sentences, fall back to character split
                    split = max_length // 2
                    text = text[:split] + " ... " + text[-split:]
                    
            return text
            
        except Exception as e:
            print(f"Text preprocessing failed: {e}")
            return text  # Return original text if processing fails


class StellaEmbeddingProvider(EmbeddingProvider):
    """
    EXACT replication of Stella 400M embedding logic from embedding_manager.py.
    Supports GPU acceleration and xformers optimization.
    """
    
    def __init__(self, device: Optional[str] = None, use_optimization: bool = True):
        """
        Initialize Stella 400M with EXACT configuration from original.
        
        Args:
            device: Force specific device ('cuda'/'cpu') or auto-detect
            use_optimization: Enable xformers optimization if available
        """
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            raise ImportError("sentence-transformers required for StellaEmbeddingProvider")
        
        self._model = None
        self._device = device
        self._use_optimization = use_optimization
        self._embedding_dim = 1024  # Stella 400M dimension
        
        # Lazy load model on first use (EXACT pattern from original)
        print("Stella 400M configured - will load on first use")
    
    @property
    def model_name(self) -> str:
        return "stella_en_400M_v5"
    
    @property
    def embedding_dimension(self) -> int:
        return self._embedding_dim
    
    @property
    def sentence_transformer(self) -> Optional[Any]:
        """
        EXACT lazy loading logic from embedding_manager.py.
        """
        if self._model is None:
            # EXACT GPU detection from original
            if self._device:
                gpu_available = self._device == 'cuda' and TORCH_AVAILABLE and torch.cuda.is_available()
            else:
                gpu_available = TORCH_AVAILABLE and torch.cuda.is_available()
            
            fast_kernel = False
            
            if gpu_available and self._use_optimization:
                try:
                    import xformers  # EXACT import pattern from original
                    fast_kernel = True
                except ImportError:
                    print("xformers not found; falling back to standard attention.")
            
            device_str = "cuda" if gpu_available else "cpu"
            print(f"Loading Stella 400M on {device_str.upper()} "
                  f"{'with xformers' if fast_kernel else 'without xformers'}")
            
            # EXACT config from original
            config_kwargs = {
                "use_memory_efficient_attention": fast_kernel,
                "unpad_inputs": fast_kernel,
                "attn_implementation": "eager",
            }
            
            try:
                self._model = SentenceTransformer(
                    "dunzhang/stella_en_400M_v5",  # EXACT model ID from original
                    device=device_str,
                    trust_remote_code=True,
                    config_kwargs=config_kwargs
                )
                print("Loaded Stella 400M successfully.")
            except Exception as e:
                print(f"Stella init failed: {e}")
                raise
        
        return self._model
    
    def encode(self, text: str) -> List[float]:
        """
        EXACT encoding logic from embedding_manager.py.
        """
        if not text:
            return [0.0] * self._embedding_dim
        
        try:
            # EXACT preprocessing from original
            preprocessed_text = self.preprocess_text(text)
            
            # EXACT encoding parameters from original
            result = self.sentence_transformer.encode(
                preprocessed_text,
                convert_to_tensor=False,
                normalize_embeddings=True,
                show_progress_bar=False
            )
            
            return [float(x) for x in result]
            
        except Exception as e:
            print(f"Stella encoding failed: {e}")
            return [0.0] * self._embedding_dim
    
    def calculate_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """EXACT similarity calculation from embedding_manager.py."""
        try:
            if not vec1 or not vec2:
                return 0.0
                
            if len(vec1) != len(vec2):
                print(f"Vector length mismatch: {len(vec1)} vs {len(vec2)}")
                return 0.0
                
            dot_product = sum(a * b for a, b in zip(vec1, vec2))
            norm1 = sum(a * a for a in vec1) ** 0.5
            norm2 = sum(b * b for b in vec2) ** 0.5
            
            if norm1 <= 0 or norm2 <= 0:
                return 0.0
                
            return dot_product / (norm1 * norm2)
            
        except Exception as e:
            print(f"Similarity calculation failed: {e}")
            return 0.0


class OpenAIEmbeddingProvider(EmbeddingProvider):
    """
    OpenAI embedding provider for comparison testing.
    Requires API access - configure with your API manager setup.
    """
    
    def __init__(self, api_client=None, model: str = "text-embedding-3-small"):
        """
        Initialize OpenAI provider.
        
        Args:
            api_client: Your API client instance (or None for placeholder)
            model: OpenAI embedding model to use
        """
        self.api_client = api_client
        self.model = model
        self._embedding_dim = 1536 if "3-small" in model else 3072  # OpenAI dimensions
        
        if not api_client:
            print("Warning: No API client provided - OpenAI embeddings will return zeros")
    
    @property
    def model_name(self) -> str:
        return f"openai_{self.model}"
    
    @property
    def embedding_dimension(self) -> int:
        return self._embedding_dim
    
    def encode(self, text: str) -> List[float]:
        """Generate OpenAI embedding."""
        if not text:
            return [0.0] * self._embedding_dim
        
        if not self.api_client:
            # Placeholder for testing without API access
            print("Warning: No API client - returning zero vector")
            return [0.0] * self._embedding_dim
        
        try:
            # EXACT API call pattern from your embedding_manager.py
            preprocessed_text = self.preprocess_text(text)
            
            # Sync version - would need async version for production
            response = self.api_client.embeddings.create(
                input=preprocessed_text,
                model=self.model
            )
            
            return response.data[0].embedding
            
        except Exception as e:
            print(f"OpenAI embedding failed: {e}")
            return [0.0] * self._embedding_dim
    
    def calculate_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """EXACT similarity calculation (same as Stella)."""
        try:
            if not vec1 or not vec2:
                return 0.0
                
            if len(vec1) != len(vec2):
                print(f"Vector length mismatch: {len(vec1)} vs {len(vec2)}")
                return 0.0
                
            dot_product = sum(a * b for a, b in zip(vec1, vec2))
            norm1 = sum(a * a for a in vec1) ** 0.5
            norm2 = sum(b * b for b in vec2) ** 0.5
            
            if norm1 <= 0 or norm2 <= 0:
                return 0.0
                
            return dot_product / (norm1 * norm2)
            
        except Exception as e:
            print(f"Similarity calculation failed: {e}")
            return 0.0


class GenericSentenceTransformerProvider(EmbeddingProvider):
    """
    Generic provider for any sentence-transformers model.
    Perfect for testing different local models quickly.
    """
    
    def __init__(self, model_name: str, device: Optional[str] = None):
        """
        Initialize with any sentence-transformers model.
        
        Args:
            model_name: HuggingFace model identifier
            device: Force specific device or auto-detect
        """
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            raise ImportError("sentence-transformers required for GenericSentenceTransformerProvider")
        
        self.model_id = model_name
        self._model = None
        self._device = device
        self._embedding_dim = None  # Will be detected on first load
    
    @property
    def model_name(self) -> str:
        return f"generic_{self.model_id.replace('/', '_')}"
    
    @property
    def embedding_dimension(self) -> int:
        if self._embedding_dim is None:
            # Load model to detect dimension
            _ = self.sentence_transformer
        return self._embedding_dim or 384  # Common default
    
    @property
    def sentence_transformer(self) -> Optional[Any]:
        """Lazy load any sentence transformer model."""
        if self._model is None:
            # Simple device detection
            if self._device:
                device_str = self._device
            else:
                device_str = "cuda" if TORCH_AVAILABLE and torch.cuda.is_available() else "cpu"
            
            print(f"Loading {self.model_id} on {device_str.upper()}")
            
            try:
                self._model = SentenceTransformer(self.model_id, device=device_str)
                
                # Detect embedding dimension
                test_embedding = self._model.encode("test", convert_to_tensor=False)
                self._embedding_dim = len(test_embedding)
                
                print(f"Loaded {self.model_id} successfully (dim: {self._embedding_dim})")
            except Exception as e:
                print(f"Failed to load {self.model_id}: {e}")
                raise
        
        return self._model
    
    def encode(self, text: str) -> List[float]:
        """Encode using the generic model."""
        if not text:
            return [0.0] * self.embedding_dimension
        
        try:
            preprocessed_text = self.preprocess_text(text)
            
            result = self.sentence_transformer.encode(
                preprocessed_text,
                convert_to_tensor=False,
                normalize_embeddings=True,
                show_progress_bar=False
            )
            
            return [float(x) for x in result]
            
        except Exception as e:
            print(f"Generic encoding failed for {self.model_id}: {e}")
            return [0.0] * self.embedding_dimension
    
    def calculate_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Standard cosine similarity."""
        try:
            if not vec1 or not vec2:
                return 0.0
                
            if len(vec1) != len(vec2):
                print(f"Vector length mismatch: {len(vec1)} vs {len(vec2)}")
                return 0.0
                
            dot_product = sum(a * b for a, b in zip(vec1, vec2))
            norm1 = sum(a * a for a in vec1) ** 0.5
            norm2 = sum(b * b for b in vec2) ** 0.5
            
            if norm1 <= 0 or norm2 <= 0:
                return 0.0
                
            return dot_product / (norm1 * norm2)
            
        except Exception as e:
            print(f"Similarity calculation failed: {e}")
            return 0.0


class DeterministicTestProvider(EmbeddingProvider):
    """
    Deterministic embedding provider for consistent testing.
    Useful for regression testing and debugging.
    """
    
    def __init__(self, dimension: int = 384, seed: int = 42):
        """
        Initialize deterministic provider.
        
        Args:
            dimension: Embedding vector dimension
            seed: Random seed for consistency
        """
        self.dimension = dimension
        self.seed = seed
        self._cache = {}  # Cache for deterministic results
    
    @property
    def model_name(self) -> str:
        return f"deterministic_test_dim{self.dimension}_seed{self.seed}"
    
    @property
    def embedding_dimension(self) -> int:
        return self.dimension
    
    def encode(self, text: str) -> List[float]:
        """Generate deterministic embedding based on text hash."""
        if not text:
            return [0.0] * self.dimension
        
        # Use cache for consistency
        if text in self._cache:
            return self._cache[text]
        
        # Generate deterministic vector based on text content
        text_hash = hashlib.md5(text.encode()).hexdigest()
        
        # Use hash to seed random generation
        import random
        random.seed(int(text_hash[:8], 16) + self.seed)
        
        # Generate normalized random vector
        vector = [random.gauss(0, 1) for _ in range(self.dimension)]
        
        # Normalize to unit length
        norm = sum(x * x for x in vector) ** 0.5
        if norm > 0:
            vector = [x / norm for x in vector]
        
        # Add text-specific bias for more realistic similarities
        text_words = set(text.lower().split())
        for i, word in enumerate(list(text_words)[:self.dimension]):
            word_hash = hash(word) % self.dimension
            vector[word_hash] += 0.1 * (1.0 / (i + 1))  # Diminishing influence
        
        # Re-normalize
        norm = sum(x * x for x in vector) ** 0.5
        if norm > 0:
            vector = [x / norm for x in vector]
        
        self._cache[text] = vector
        return vector
    
    def calculate_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Standard cosine similarity."""
        try:
            if not vec1 or not vec2:
                return 0.0
                
            if len(vec1) != len(vec2):
                return 0.0
                
            dot_product = sum(a * b for a, b in zip(vec1, vec2))
            norm1 = sum(a * a for a in vec1) ** 0.5
            norm2 = sum(b * b for b in vec2) ** 0.5
            
            if norm1 <= 0 or norm2 <= 0:
                return 0.0
                
            return dot_product / (norm1 * norm2)
            
        except Exception as e:
            return 0.0


# Convenience factory functions for easy model swapping
def create_stella_provider(device: Optional[str] = None) -> EmbeddingProvider:
    """Create Stella 400M provider (your current production model)."""
    return StellaEmbeddingProvider(device=device)

def create_openai_provider(api_client=None) -> EmbeddingProvider:
    """Create OpenAI provider for comparison."""
    return OpenAIEmbeddingProvider(api_client=api_client)

def create_generic_provider(model_name: str, device: Optional[str] = None) -> EmbeddingProvider:
    """Create provider for any sentence-transformers model."""
    return GenericSentenceTransformerProvider(model_name, device=device)

def create_test_provider(dimension: int = 384) -> EmbeddingProvider:
    """Create deterministic provider for testing."""
    return DeterministicTestProvider(dimension=dimension)


# Common model configurations for easy testing
COMMON_MODELS = {
    "stella_400m": "dunzhang/stella_en_400M_v5",          # Your current production
    "all_mpnet": "sentence-transformers/all-mpnet-base-v2",  # Strong general model
    "all_minilm": "sentence-transformers/all-MiniLM-L6-v2",  # Fast lightweight
    "e5_large": "intfloat/e5-large-v2",                   # Strong multilingual
    "bge_large": "BAAI/bge-large-en-v1.5",               # SOTA retrieval
    "instructor": "hkunlp/instructor-large",              # Instruction-tuned
}

def create_provider_by_alias(alias: str, device: Optional[str] = None) -> EmbeddingProvider:
    """Create provider by common alias for easy experimentation."""
    if alias == "stella":
        return create_stella_provider(device)
    elif alias == "openai":
        return create_openai_provider()
    elif alias == "test":
        return create_test_provider()
    elif alias in COMMON_MODELS:
        return create_generic_provider(COMMON_MODELS[alias], device)
    else:
        # Assume it's a HuggingFace model ID
        return create_generic_provider(alias, device)