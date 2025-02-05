# embeddings/embedding_manager.py
from typing import List, Optional, Any
from functools import lru_cache

from loggers.loggers import MemoryLogger
from config import Config

from api_clients import APIManager

class EmbeddingManager:
    """
    Manages text embedding generation with fallback strategies.
    Supports both local sentence-transformers and API-based embeddings.
    """
    
    def __init__(self, api_manager: APIManager=None):
        self.api_manager = api_manager
        self.use_local = Config.get_use_local_embedding()
        self.model = None
        self.max_retries = 3
        self.retry_delay = 1
        self._sentence_transformer = None  # For lazy loading
        self._embedding_cache = {} 
        self._cache_size = 1000
        
        if self.use_local:
            # Don't load immediately - wait for first use
            MemoryLogger.info("Local embedding configured - will load on first use")
        else:
            MemoryLogger.info("Using API embedding")

    @property
    def sentence_transformer(self) -> Optional[Any]:
        """Lazy load the sentence transformer model."""
        if self._sentence_transformer is None and self.use_local:
            try:
                # Lazy import
                from sentence_transformers import SentenceTransformer
                self._sentence_transformer = SentenceTransformer('all-MiniLM-L6-v2')
                MemoryLogger.info("Successfully loaded sentence-transformers")
            except ImportError as e:
                MemoryLogger.warning(
                    f"Could not import sentence-transformers: {e}. "
                    "Will use API fallback for embeddings."
                )
                self.use_local = False
            except Exception as e:
                MemoryLogger.warning(
                    f"Failed to initialize sentence-transformers: {e}. "
                    "Will use API fallback for embeddings."
                )
                self.use_local = False
        return self._sentence_transformer
            
    def preprocess_text(self, text: str, max_length: int = 512) -> str:
        """
        Preprocess text for embedding generation.
        Handles truncation and normalization.
        """
        if not text:
            return ""
            
        try:
            text = text.strip().lower()
            
            # Handle long texts by taking meaningful chunks
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
            MemoryLogger.error(f"Text preprocessing failed: {str(e)}")
            return text  # Return original text if processing fails
        
    def encode(
        self,
        text: str,
        convert_to_tensor: bool = False,
        normalize_embeddings: bool = True
    ) -> List[float]:
        """
        Generate embedding vector for text.
        Falls back gracefully through multiple options.
        
        Args:
            text: Text to embed
            convert_to_tensor: Whether to return PyTorch tensor
            normalize_embeddings: Whether to L2-normalize embeddings
            
        Returns:
            List[float]: 384-dimensional embedding vector
        """
        return list(self._cached_internal_encode(text, convert_to_tensor, normalize_embeddings))

    @lru_cache(maxsize=1000)
    def _cached_internal_encode(
        self,
        text: str,
        convert_to_tensor: bool = False,
        normalize_embeddings: bool = True
    ) -> tuple:
        """Cached internal implementation that returns tuple for hashability."""
        if not text:
            return tuple([0.0] * 384)
            
        try:
            text = self.preprocess_text(text)
            
            # Try local model first if configured
            if self.use_local:
                try:
                    if self.sentence_transformer:  # This triggers lazy load
                        result = self.sentence_transformer.encode(
                            text,
                            convert_to_tensor=convert_to_tensor,
                            normalize_embeddings=normalize_embeddings,
                            show_progress_bar=False
                        )
                        return tuple(float(x) for x in result)
                except Exception as e:
                    MemoryLogger.warning(
                        f"Local embedding failed, falling back to API: {e}"
                    )
                    self.use_local = False  # Disable local for future calls
            
            # Try API if available
            if self.api_manager:
                try:
                    result = self._get_api_embedding(text)
                    return tuple(float(x) for x in result)
                except Exception as e:
                    MemoryLogger.error(f"API embedding failed: {str(e)}")
                    
            # If all else fails, return zero vector
            MemoryLogger.warning(
                "All embedding methods failed - returning zero vector"
            )
            return tuple([0.0] * 384)
                
        except Exception as e:
            MemoryLogger.error(f"Embedding generation failed: {str(e)}")
            return tuple([0.0] * 384)

    def _get_api_embedding(self, text: str) -> List[float]:
        """Get embedding via API with retry logic."""
        if not self.api_manager:
            raise ValueError("No API manager available")
            
        try:
            client = self.api_manager.get_client('openai')
            
            MemoryLogger.debug(f"Starting API embedding request. Text length: {len(text)}")
            
            # Use async context manager from existing API client
            response = client._make_request(
                "embeddings",
                payload={
                    "input": text,
                    "model": "text-embedding-3-small"
                }
            )
            
            return response['data'][0]['embedding']
            
        except Exception as e:
            MemoryLogger.error(f"API embedding failed: {str(e)}")
            raise

    def calculate_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """
        Calculate cosine similarity between two vectors.
        Handles edge cases gracefully.
        """
        try:
            if not vec1 or not vec2:
                return 0.0
                
            if len(vec1) != len(vec2):
                MemoryLogger.warning(
                    f"Vector length mismatch: {len(vec1)} vs {len(vec2)}"
                )
                return 0.0
                
            dot_product = sum(a * b for a, b in zip(vec1, vec2))
            norm1 = sum(a * a for a in vec1) ** 0.5
            norm2 = sum(b * b for b in vec2) ** 0.5
            
            if norm1 <= 0 or norm2 <= 0:
                return 0.0
                
            return dot_product / (norm1 * norm2)
            
        except Exception as e:
            MemoryLogger.error(f"Similarity calculation failed: {str(e)}")
            return 0.0