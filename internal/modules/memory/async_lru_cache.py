import asyncio
import functools
import time
from collections import OrderedDict
from typing import Any, Callable, Coroutine, Tuple, Optional, Dict
from dataclasses import dataclass


@dataclass
class CacheEntry:
    """Cache entry with TTL support."""
    future: asyncio.Future
    created_at: float
    ttl: Optional[float] = None
    
    def is_expired(self) -> bool:
        """Check if cache entry has expired."""
        if self.ttl is None:
            return False
        return time.time() - self.created_at > self.ttl


class AsyncLRUCacheWithTTL:
    """
    Thread-safe async LRU cache with TTL support.
    Supports both time-based expiration and LRU eviction.
    """
    
    def __init__(self, maxsize: int = 128, ttl: Optional[float] = None):
        self.maxsize = maxsize
        self.ttl = ttl
        self.cache: OrderedDict[Any, CacheEntry] = OrderedDict()
        self.lock = asyncio.Lock()
        self.hits = 0
        self.misses = 0
    
    async def get(self, key: Any) -> Optional[Any]:
        """Get value from cache if exists and not expired."""
        async with self.lock:
            if key not in self.cache:
                self.misses += 1
                return None
                
            entry = self.cache[key]
            
            # Check if expired
            if entry.is_expired():
                del self.cache[key]
                self.misses += 1
                return None
            
            # Move to end (mark as recently used)
            self.cache.move_to_end(key)
            self.hits += 1
            
            try:
                return await entry.future
            except Exception:
                # Remove failed entries
                if key in self.cache:
                    del self.cache[key]
                raise
    
    async def put(self, key: Any, future: asyncio.Future, ttl: Optional[float] = None) -> None:
        """Put value in cache with optional TTL override."""
        async with self.lock:
            # Use instance TTL if no override provided
            effective_ttl = ttl if ttl is not None else self.ttl
            
            entry = CacheEntry(
                future=future,
                created_at=time.time(),
                ttl=effective_ttl
            )
            
            self.cache[key] = entry
            
            # Evict LRU if over capacity
            if len(self.cache) > self.maxsize:
                self.cache.popitem(last=False)
    
    async def evict_expired(self) -> int:
        """Manually evict expired entries. Returns number evicted."""
        evicted = 0
        async with self.lock:
            expired_keys = [
                key for key, entry in self.cache.items()
                if entry.is_expired()
            ]
            for key in expired_keys:
                del self.cache[key]
                evicted += 1
        return evicted
    
    async def clear(self) -> None:
        """Clear all cache entries."""
        async with self.lock:
            self.cache.clear()
            self.hits = 0
            self.misses = 0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_requests = self.hits + self.misses
        hit_rate = self.hits / total_requests if total_requests > 0 else 0.0
        
        return {
            'hits': self.hits,
            'misses': self.misses,
            'hit_rate': hit_rate,
            'size': len(self.cache),
            'max_size': self.maxsize
        }


def async_lru_cache(maxsize: int = 128, ttl: Optional[float] = None, key_func: Optional[Callable] = None):
    """
    Enhanced async LRU cache decorator with TTL support.
    
    Args:
        maxsize: Maximum number of entries to cache
        ttl: Time-to-live in seconds (None for no expiration)
        key_func: Optional custom function to generate cache keys
    """
    def decorator(func: Callable) -> Callable:
        cache = AsyncLRUCacheWithTTL(maxsize=maxsize, ttl=ttl)

        @functools.wraps(func)
        async def wrapper(*args, **kwargs) -> Any:
            # Create hashable key from args and kwargs
            try:
                if key_func:
                    # Use custom key function if provided
                    if asyncio.iscoroutinefunction(key_func):
                        key = await key_func(*args, **kwargs)
                    else:
                        key = key_func(*args, **kwargs)
                else:
                    # Use automatic key generation
                    key = (args, frozenset(kwargs.items()))
            except TypeError:
                # Handle unhashable arguments by converting to strings
                try:
                    if key_func:
                        # Custom key function failed, fall back to string conversion
                        key = f"custom_failed:{hash(str(args))}"
                    else:
                        str_args = tuple(str(arg) for arg in args)
                        str_kwargs = frozenset((k, str(v)) for k, v in kwargs.items())
                        key = (str_args, str_kwargs)
                except Exception:
                    # If all else fails, don't cache this call
                    return await func(*args, **kwargs)
            
            # Try to get from cache
            try:
                result = await cache.get(key)
                if result is not None:
                    return result
            except Exception:
                # Cache error - proceed without caching
                pass
            
            # Cache miss - execute function
            try:
                future = asyncio.ensure_future(func(*args, **kwargs))
                await cache.put(key, future)
                return await future
            except Exception as e:
                # Function execution failed
                # Try to remove failed entry from cache
                try:
                    async with cache.lock:
                        if key in cache.cache:
                            del cache.cache[key]
                except Exception:
                    pass
                raise e

        # Expose cache for monitoring/debugging
        wrapper.cache = cache
        wrapper.cache_info = cache.get_stats
        wrapper.cache_clear = cache.clear
        
        return wrapper

    return decorator


# Backwards compatibility - keep original function name
def async_lru_cache_legacy(maxsize: int = 128):
    """Legacy async_lru_cache without TTL for backwards compatibility."""
    return async_lru_cache(maxsize=maxsize, ttl=None)