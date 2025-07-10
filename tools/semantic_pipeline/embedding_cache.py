
"""
Global embedding provider cache to prevent model reloading.
"""

_GLOBAL_PROVIDER_CACHE = {}

def get_cached_provider(provider_alias: str):
    """Get cached provider or create if not exists."""
    if provider_alias not in _GLOBAL_PROVIDER_CACHE:
        from embedding_providers import create_provider_by_alias
        print(f"🔄 Loading {provider_alias} embedding model (first time only)...")
        _GLOBAL_PROVIDER_CACHE[provider_alias] = create_provider_by_alias(provider_alias)
        print(f"✅ {provider_alias} loaded and cached")
    return _GLOBAL_PROVIDER_CACHE[provider_alias]

def clear_cache():
    """Clear the global cache."""
    global _GLOBAL_PROVIDER_CACHE
    _GLOBAL_PROVIDER_CACHE.clear()
