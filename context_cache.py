from datetime import datetime, timedelta, timezone
from typing import Any

# Shared in-memory cache for user context
# Each entry is a dictionary with 'context' and 'timestamp' keys
context_cache: dict[str, dict[str, Any]] = {}

# Time-to-live for each cache entry (adjust as needed)
context_ttl = timedelta(minutes=5)

def is_cache_valid(user_id: str) -> bool:
    """Check if a user's cached context is still valid based on TTL."""
    if user_id not in context_cache:
        return False
    entry = context_cache[user_id]
    timestamp = entry.get("timestamp")
    if not timestamp:
        return False
    now = datetime.now(timezone.utc)
    return now - timestamp < context_ttl

def get_cached_context_entry(user_id: str) -> dict | None:
    """Return cached context for a user if it exists and is still fresh."""
    if is_cache_valid(user_id):
        return context_cache[user_id]["context"]
    return None

def store_context(user_id: str, context: dict):
    """Store a new context entry in the cache for a given user."""
    context_cache[user_id] = {
        "context": context,
        "timestamp": datetime.now(timezone.utc)
    }
