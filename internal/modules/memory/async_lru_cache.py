import asyncio
import functools
from collections import OrderedDict
from typing import Any, Callable, Coroutine, Tuple

def async_lru_cache(maxsize: int = 128):
    """
    An async LRU cache decorator for caching the final result of an async function.
    
    This decorator:
      - Uses an OrderedDict to keep track of call order.
      - Uses an asyncio.Lock to ensure thread safety.
      - Caches the final result (not the coroutine) so that the heavy computation
        is only performed once per unique set of arguments.
      - Evicts the least-recently-used entry when the cache exceeds the maxsize.
    """
    def decorator(func: Callable) -> Callable:
        cache: "OrderedDict[Any, asyncio.Future]" = OrderedDict()
        lock = asyncio.Lock()

        @functools.wraps(func)
        async def wrapper(*args, **kwargs) -> Any:
            # Create a hashable key from args and kwargs.
            # Here we assume that all arguments are hashable.
            key = (args, frozenset(kwargs.items()))
            async with lock:
                if key in cache:
                    # Cache hit: move key to end to mark it as recently used.
                    future = cache.pop(key)
                    cache[key] = future
                else:
                    # Cache miss: schedule the async function and store its future.
                    future = asyncio.ensure_future(func(*args, **kwargs))
                    cache[key] = future
                    # Evict the oldest item if cache size exceeds maxsize.
                    if len(cache) > maxsize:
                        cache.popitem(last=False)
            try:
                # Await the future (which might be newly created or cached).
                result = await future
                return result
            except Exception:
                # If the function call fails, remove the entry from the cache.
                async with lock:
                    if key in cache and cache[key] is future:
                        del cache[key]
                raise

        return wrapper

    return decorator
