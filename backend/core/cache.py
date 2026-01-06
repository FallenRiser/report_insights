"""
Caching Utilities

In-memory caching for analysis results with TTL support.
"""

import functools
import hashlib
import json
import time
from collections import OrderedDict
from threading import Lock
from typing import Any, Callable, Optional, TypeVar

from config import get_settings


T = TypeVar("T")


class TTLCache:
    """Thread-safe LRU cache with TTL support."""
    
    def __init__(self, maxsize: int = 128, ttl_seconds: int = 3600):
        self.maxsize = maxsize
        self.ttl_seconds = ttl_seconds
        self._cache: OrderedDict[str, tuple[Any, float]] = OrderedDict()
        self._lock = Lock()
    
    def _make_key(self, *args, **kwargs) -> str:
        """Create cache key from arguments."""
        key_parts = [str(arg) for arg in args]
        key_parts.extend(f"{k}={v}" for k, v in sorted(kwargs.items()))
        key_string = "|".join(key_parts)
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache if not expired."""
        with self._lock:
            if key not in self._cache:
                return None
            
            value, timestamp = self._cache[key]
            if time.time() - timestamp > self.ttl_seconds:
                # Expired
                del self._cache[key]
                return None
            
            # Move to end (most recently accessed)
            self._cache.move_to_end(key)
            return value
    
    def set(self, key: str, value: Any) -> None:
        """Set value in cache."""
        with self._lock:
            # Remove oldest if at capacity
            while len(self._cache) >= self.maxsize:
                self._cache.popitem(last=False)
            
            self._cache[key] = (value, time.time())
    
    def delete(self, key: str) -> bool:
        """Delete key from cache."""
        with self._lock:
            if key in self._cache:
                del self._cache[key]
                return True
            return False
    
    def clear(self) -> None:
        """Clear all cache entries."""
        with self._lock:
            self._cache.clear()
    
    def __contains__(self, key: str) -> bool:
        """Check if key exists and is not expired."""
        return self.get(key) is not None


class SessionStore:
    """Session metadata storage."""
    
    def __init__(self):
        self._sessions: dict[str, dict[str, Any]] = {}
        self._lock = Lock()
        settings = get_settings()
        self.ttl_seconds = settings.session_ttl_hours * 3600
    
    def create(self, session_id: str, metadata: dict[str, Any]) -> None:
        """Create new session."""
        with self._lock:
            self._sessions[session_id] = {
                **metadata,
                "created_at": time.time(),
            }
    
    def get(self, session_id: str) -> Optional[dict[str, Any]]:
        """Get session metadata."""
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                return None
            
            # Check expiration
            if time.time() - session["created_at"] > self.ttl_seconds:
                del self._sessions[session_id]
                return None
            
            return session
    
    def update(self, session_id: str, updates: dict[str, Any]) -> bool:
        """Update session metadata."""
        with self._lock:
            if session_id not in self._sessions:
                return False
            self._sessions[session_id].update(updates)
            return True
    
    def delete(self, session_id: str) -> bool:
        """Delete session."""
        with self._lock:
            if session_id in self._sessions:
                del self._sessions[session_id]
                return True
            return False
    
    def list_sessions(self) -> list[str]:
        """List all active session IDs."""
        with self._lock:
            now = time.time()
            # Clean expired and return active
            active = []
            expired = []
            for sid, session in self._sessions.items():
                if now - session["created_at"] > self.ttl_seconds:
                    expired.append(sid)
                else:
                    active.append(sid)
            
            for sid in expired:
                del self._sessions[sid]
            
            return active


def cached(cache: TTLCache):
    """Decorator for caching function results."""
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> T:
            # Skip first arg if it's 'self'
            cache_args = args[1:] if args and hasattr(args[0], "__class__") else args
            key = cache._make_key(func.__name__, *cache_args, **kwargs)
            
            result = cache.get(key)
            if result is not None:
                return result
            
            result = func(*args, **kwargs)
            cache.set(key, result)
            return result
        
        return wrapper
    return decorator


# Global instances
settings = get_settings()
analysis_cache = TTLCache(
    maxsize=settings.cache.max_size,
    ttl_seconds=settings.cache.ttl_seconds
)
session_store = SessionStore()
