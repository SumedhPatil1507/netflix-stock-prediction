"""
Redis Caching Layer for Lightning-Fast Feature Lookup.
Provides caching of computed feature DataFrames and single-row lookups.
Falls back to a local memory/JSON file cache if Redis is unavailable.
"""
from __future__ import annotations
import os
import json
import logging
import pandas as pd

logger = logging.getLogger(__name__)

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

# Local cache fallback
_local_cache: dict[str, str] = {}
_fallback_file = os.path.join("outputs", "features_redis_fallback.json")

def get_redis_client():
    """Get Redis client, returns None if redis package is missing or connection fails."""
    try:
        import redis
        client = redis.from_url(REDIS_URL, decode_responses=True)
        # Ping to test connection
        client.ping()
        return client
    except Exception as e:
        logger.debug(f"Redis not available ({e}). Using local in-memory fallback cache.")
        return None

def cache_features(df: pd.DataFrame, ticker: str) -> None:
    """Serialize and cache feature DataFrame to Redis (and latest row specifically)."""
    if df.empty:
        return
        
    ticker = ticker.upper()
    
    # Standardize data types to float to prevent serialization errors
    df_serialized = df.copy()
    for col in df_serialized.columns:
        df_serialized[col] = pd.to_numeric(df_serialized[col], errors="coerce")
        
    # Convert entire DataFrame to JSON
    df_json = df_serialized.to_json(date_format="iso")
    
    # Extract latest row as dict
    latest_row = df_serialized.iloc[-1].to_dict()
    # Handle timestamp index
    latest_time = str(df_serialized.index[-1])
    latest_row["_timestamp"] = latest_time
    latest_row_json = json.dumps(latest_row)
    
    client = get_redis_client()
    if client:
        try:
            # Set keys with 2 hours TTL (7200 seconds)
            client.set(f"features:{ticker}:df", df_json, ex=7200)
            client.set(f"features:{ticker}:latest_row", latest_row_json, ex=7200)
            logger.info(f"Successfully cached features for {ticker} in Redis.")
            return
        except Exception as e:
            logger.warning(f"Failed to save features to Redis ({e}). Saving to local fallback.")
            
    # Fallback to local memory & disk
    _local_cache[f"features:{ticker}:df"] = df_json
    _local_cache[f"features:{ticker}:latest_row"] = latest_row_json
    
    try:
        os.makedirs("outputs", exist_ok=True)
        with open(_fallback_file, "w") as f:
            json.dump(_local_cache, f)
    except Exception as fe:
        logger.warning(f"Could not write to local fallback cache file ({fe}).")

def get_cached_features(ticker: str) -> pd.DataFrame | None:
    """Retrieve full feature DataFrame from Redis or local cache."""
    ticker = ticker.upper()
    client = get_redis_client()
    df_json = None
    
    if client:
        try:
            df_json = client.get(f"features:{ticker}:df")
        except Exception as e:
            logger.warning(f"Redis get failed ({e}), checking local fallback.")
            
    if not df_json:
        # Check local cache
        df_json = _local_cache.get(f"features:{ticker}:df")
        if not df_json and os.path.exists(_fallback_file):
            try:
                with open(_fallback_file) as f:
                    fallback_data = json.load(f)
                    df_json = fallback_data.get(f"features:{ticker}:df")
            except Exception:
                pass
                
    if not df_json:
        return None
        
    try:
        df = pd.read_json(df_json)
        # Ensure correct index sorting
        df = df.sort_index()
        return df
    except Exception as e:
        logger.error(f"Error deserializing cached features: {e}")
        return None

def get_latest_features_row(ticker: str) -> dict | None:
    """Retrieve only the latest feature row (very fast lookup)."""
    ticker = ticker.upper()
    client = get_redis_client()
    row_json = None
    
    if client:
        try:
            row_json = client.get(f"features:{ticker}:latest_row")
        except Exception as e:
            logger.warning(f"Redis get latest failed ({e}), checking local fallback.")
            
    if not row_json:
        row_json = _local_cache.get(f"features:{ticker}:latest_row")
        if not row_json and os.path.exists(_fallback_file):
            try:
                with open(_fallback_file) as f:
                    fallback_data = json.load(f)
                    row_json = fallback_data.get(f"features:{ticker}:latest_row")
            except Exception:
                pass
                
    if not row_json:
        return None
        
    try:
        return json.loads(row_json)
    except Exception as e:
        logger.error(f"Error parsing latest cached row: {e}")
        return None
