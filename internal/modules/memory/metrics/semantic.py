"""
metrics/semantic.py

Implements semantic analysis calculations for cognitive memory retrieval.
Handles embedding comparison, text relevance scoring, and information richness.

Key capabilities:
- Embedding-based similarity calculation
- Text relevance scoring with entity/phrase matching
- Information richness heuristic (replaces SpaCy NLP density pipeline)
- Semantic cohesion via pairwise sentence embeddings
"""

from typing import Dict, List, Optional, Any
from abc import ABC, abstractmethod

import asyncio
import hashlib

import numpy as np

from internal.modules.memory.async_lru_cache import async_lru_cache
from internal.modules.memory.embedding_manager import EmbeddingManager
from loggers.loggers import MemoryLogger


class SemanticAnalysisError(Exception):
    """Base exception for semantic analysis failures."""
    pass


class BaseSemanticMetricsCalculator(ABC):
    """
    Abstract base class defining the interface for semantic metrics calculation.
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
    Handles embedding comparison, text analysis, and information richness.
    """

    def __init__(self, embedding_manager: EmbeddingManager):
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
        **kwargs
    ) -> Dict[str, float]:
        """
        Pipelined semantic metrics calculation.

        Computes embedding similarity, text relevance, information richness
        (replaces SpaCy-based semantic density), and semantic cohesion.
        """
        try:
            metrics: Dict[str, float] = {}

            # Start information richness (cheap, no NLP dependency)
            density_value = self._calculate_information_richness(text_content)

            # Start cohesion if we have multiple sentences
            cohesion_task = None
            sentences = [s.strip() for s in text_content.split('.') if s.strip()]
            if len(sentences) >= 2:
                cohesion_task = asyncio.create_task(
                    self._calculate_semantic_cohesion_cached(sentences)
                )

            # Embedding similarity (core retrieval signal)
            if query_embedding is not None:
                metrics['embedding_similarity'] = await self.embedding_manager.calculate_similarity_async(
                    embedding,
                    query_embedding
                )

            # Text relevance with entity/phrase matching
            if query_text:
                metrics['text_relevance'] = await self._calculate_text_relevance_async(
                    text_content,
                    query_text
                )

            # Information richness (replaces semantic_density)
            metrics['semantic_density'] = density_value

            # Semantic cohesion
            if cohesion_task:
                metrics['semantic_cohesion'] = await cohesion_task
            else:
                metrics['semantic_cohesion'] = 0.4

            return metrics

        except Exception as e:
            self.logger.log_error(f"Semantic metrics calculation failed: {str(e)}")
            return self._get_fallback_metrics()

    # ---- Text Relevance ----

    @async_lru_cache(
        maxsize=1000,
        ttl=3600,
        key_func=lambda self, text, query: self._generate_text_cache_key(text, query)
    )
    async def _calculate_text_relevance_cached(self, text: str, query: str) -> float:
        """Cached version of text relevance calculation."""
        return self._calculate_text_relevance_internal(text, query)

    async def _calculate_text_relevance_async(self, text: str, query: str) -> float:
        """Async wrapper with caching."""
        return await self._calculate_text_relevance_cached(text, query)

    def _calculate_text_relevance(self, text: str, query: str) -> float:
        """Synchronous text relevance calculation."""
        return self._calculate_text_relevance_internal(text, query)

    def _calculate_text_relevance_internal(self, text: str, query: str) -> float:
        """
        Enhanced text relevance calculation with entity matching,
        weighted word importance, and phrase matching.
        """
        try:
            if not text or not query:
                return 0.0

            text_lower = text.lower()
            query_lower = query.lower()

            # Base keyword overlap
            query_words = set(query_lower.split())
            text_words = set(text_lower.split())
            if not query_words:
                return 0.0

            word_matches = query_words.intersection(text_words)
            keyword_score = len(word_matches) / len(query_words)

            # Entity matching boost
            entity_boost = self._calculate_entity_relevance_boost(text, query)

            # Information density weighting for matched words
            weighted_score = self._calculate_weighted_word_relevance(
                text_lower, query_lower, word_matches
            )

            # Exact phrase matching bonus
            phrase_bonus = self._calculate_phrase_relevance_bonus(text_lower, query_lower)

            final_score = (
                keyword_score * 0.5 +
                entity_boost * 0.25 +
                weighted_score * 0.15 +
                phrase_bonus * 0.10
            )

            return min(1.0, final_score)

        except Exception as e:
            self.logger.log_error(f"Text relevance calculation failed: {e}")
            return 0.0

    def _calculate_entity_relevance_boost(self, text: str, query: str) -> float:
        """Calculate boost for exact entity matches (capitalized words)."""
        try:
            query_entities = set()
            text_entities = set()

            for word in query.split():
                if word and word[0].isupper() and len(word) > 2:
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

    def _calculate_weighted_word_relevance(
        self, text_lower: str, query_lower: str, word_matches: set
    ) -> float:
        """Weight matched words by their information value."""
        try:
            if not word_matches:
                return 0.0

            weighted_score = 0.0
            query_words = set(query_lower.split())

            for word in word_matches:
                weight = 1.0

                # Longer words are more specific
                if len(word) >= 6:
                    weight += 0.5
                elif len(word) >= 4:
                    weight += 0.2

                # Technical/specific suffixes
                if any(suffix in word for suffix in ['tion', 'ment', 'ness', 'ity']):
                    weight += 0.3

                # Numbers/dates carry factual weight
                if any(c.isdigit() for c in word):
                    weight += 0.4

                weighted_score += weight

            return weighted_score / len(query_words)

        except Exception:
            return 0.0

    def _calculate_phrase_relevance_bonus(self, text_lower: str, query_lower: str) -> float:
        """Calculate bonus for exact 2-word and 3-word phrase matches."""
        try:
            bonus = 0.0
            query_words = query_lower.split()

            for i in range(len(query_words) - 1):
                phrase_2 = f"{query_words[i]} {query_words[i+1]}"
                if phrase_2 in text_lower:
                    bonus += 0.3

                if i < len(query_words) - 2:
                    phrase_3 = f"{query_words[i]} {query_words[i+1]} {query_words[i+2]}"
                    if phrase_3 in text_lower:
                        bonus += 0.5

            return min(1.0, bonus)

        except Exception:
            return 0.0

    # ---- Information Richness (replaces SpaCy semantic density) ----

    def _calculate_information_richness(self, text: str) -> float:
        """
        Simple information richness heuristic replacing the SpaCy NLP density pipeline.

        Computes: unique_words / total_words * entity_boost * length_factor

        Produces values in roughly the same 0-1 range as the old semantic_density
        so downstream quality multipliers continue to work without retuning.
        """
        try:
            if not text or len(text.strip()) < 10:
                return 0.0

            words = text.split()
            if not words:
                return 0.0

            # Filter to meaningful words (alphabetic, len > 1)
            meaningful = [w.lower() for w in words if w.isalpha() and len(w) > 1]
            if not meaningful:
                return 0.0

            total = len(meaningful)
            unique = len(set(meaningful))

            # Base: lexical diversity
            diversity = unique / total

            # Entity boost: capitalized words indicate named entities / specificity
            capitalized = sum(
                1 for w in words
                if w and w[0].isupper() and len(w) > 2 and w.isalpha()
            )
            entity_factor = 1.0 + min(0.4, capitalized / max(1, len(words)) * 2.0)

            # Length factor: longer texts tend to be richer, with diminishing returns
            length_factor = min(1.0, len(words) / 30.0)  # saturates at 30 words

            # Technical term boost: words with suffixes indicating domain specificity
            technical_suffixes = ('tion', 'ment', 'ness', 'ity', 'ize', 'ise', 'ous', 'ive')
            technical_count = sum(
                1 for w in meaningful
                if any(w.endswith(s) for s in technical_suffixes)
            )
            technical_factor = 1.0 + min(0.3, technical_count / max(1, total) * 2.0)

            raw = diversity * entity_factor * length_factor * technical_factor

            # Clamp to match old density output range (~0.08 to ~0.87)
            return max(0.08, min(0.87, raw))

        except Exception as e:
            self.logger.log_error(f"Information richness calculation failed: {e}")
            return 0.0

    # ---- Semantic Cohesion ----

    @async_lru_cache(
        maxsize=1000,
        ttl=3600,
        key_func=lambda self, sentences: self._generate_text_cache_key(
            '|||'.join(sentences), "cohesion"
        )
    )
    async def _calculate_semantic_cohesion_cached(self, sentences: List[str]) -> float:
        """Cached version of semantic cohesion calculation."""
        return await self._calculate_semantic_cohesion_internal(sentences)

    async def _calculate_semantic_cohesion_internal(self, sentences: List[str]) -> float:
        """Calculate semantic cohesion via pairwise sentence embedding similarity."""
        if len(sentences) < 2:
            return 0.4

        try:
            embedding_tasks = [
                self.embedding_manager.encode(s, normalize_embeddings=True)
                for s in sentences
            ]
            embeddings = await asyncio.gather(*embedding_tasks)

            similarities = []
            for i in range(len(embeddings)):
                for j in range(i + 1, len(embeddings)):
                    try:
                        sim = self.embedding_manager.calculate_similarity(
                            embeddings[i],
                            embeddings[j]
                        )
                        similarities.append(sim)
                    except Exception as e:
                        self.logger.log_error(f"Similarity calculation failed: {e}")
                        continue

            return float(np.mean(similarities)) if similarities else 0.4

        except Exception as e:
            self.logger.log_error(f"Semantic cohesion calculation failed: {e}")
            return 0.4

    # ---- Fallback ----

    def _get_fallback_metrics(self) -> Dict[str, float]:
        """Provide safe fallback metrics if calculations fail."""
        return {
            'embedding_similarity': 0.0,
            'text_relevance': 0.0,
            'semantic_density': 0.0,
            'semantic_cohesion': 0.4
        }
