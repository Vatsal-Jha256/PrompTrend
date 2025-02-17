# core/cache.py
import redis
from typing import Optional, Any, Dict
import json
import os
from functools import wraps

class RedisCache:
    def __init__(self):
        self.redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
        self.redis_client = redis.from_url(self.redis_url)
        self.default_ttl = 3600  # 1 hour default TTL

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set a value in cache with optional TTL"""
        try:
            serialized_value = json.dumps(value)
            return self.redis_client.set(
                key,
                serialized_value,
                ex=ttl or self.default_ttl
            )
        except Exception as e:
            print(f"Cache set error: {str(e)}")
            return False

    def get(self, key: str) -> Optional[Any]:
        """Get a value from cache"""
        try:
            value = self.redis_client.get(key)
            return json.loads(value) if value else None
        except Exception as e:
            print(f"Cache get error: {str(e)}")
            return None

    def delete(self, key: str) -> bool:
        """Delete a value from cache"""
        try:
            return bool(self.redis_client.delete(key))
        except Exception as e:
            print(f"Cache delete error: {str(e)}")
            return False

# Cache decorator
def cached(ttl: Optional[int] = None):
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            cache = RedisCache()
            
            # Create cache key from function name and arguments
            cache_key = f"{func.__name__}:{str(args)}:{str(kwargs)}"
            
            # Try to get from cache
            cached_result = cache.get(cache_key)
            if cached_result is not None:
                return cached_result
            
            # If not in cache, execute function and cache result
            result = await func(*args, **kwargs)
            cache.set(cache_key, result, ttl)
            return result
            
        return wrapper
    return decorator