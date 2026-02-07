"""
Redis caching module for stock simulator.
Uses redis-py for caching stock data and alpha calculations.
"""
import json
import pickle
from datetime import datetime
from typing import Optional, Any, Dict
import redis

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DatabaseConfig, DATA_DIR

import loguru
logger = loguru.logger


class RedisCache:
    """Redis cache manager"""
    
    def __init__(
        self,
        host: str = DatabaseConfig.REDIS_HOST,
        port: int = DatabaseConfig.REDIS_PORT,
        db: int = DatabaseConfig.REDIS_DB,
        decode_responses: bool = True
    ):
        """
        Initialize Redis connection.
        
        Args:
            host: Redis host
            port: Redis port
            db: Redis database number
            decode_responses: Whether to decode responses to strings
        """
        self.host = host
        self.port = port
        self.db = db
        self.decode_responses = decode_responses
        
        try:
            self.client = redis.Redis(
                host=host,
                port=port,
                db=db,
                decode_responses=decode_responses,
                socket_timeout=5,
                socket_connect_timeout=5
            )
            # Test connection
            self.client.ping()
            logger.info(f"Redis connected: {host}:{port}/{db}")
        except redis.ConnectionError as e:
            logger.error(f"Failed to connect to Redis: {e}")
            raise
    
    def _make_key(self, prefix: str, *args) -> str:
        """Generate cache key with prefix"""
        parts = [str(arg).replace(' ', '_') for arg in args]
        return f"stock_sim:{prefix}:{':'.join(parts)}"
    
    def get_cached_stock_data(self, stock_code: str, date: str = None) -> Optional[Dict]:
        """
        Get cached stock data.
        
        Args:
            stock_code: Stock symbol
            date: Optional date filter
        
        Returns:
            Dict with data or None if not cached
        """
        try:
            if date:
                key = self._make_key("stock", stock_code, date)
            else:
                key = self._make_key("stock", stock_code, "*")
            
            # Use pattern matching for multiple keys
            if "*" in key:
                keys = self.client.keys(key)
                if not keys:
                    return None
                
                # Get the most recent key
                key = sorted(keys)[-1]
                data = self.client.get(key)
            else:
                data = self.client.get(key)
            
            if data:
                logger.debug(f"Cache hit for {key}")
                return json.loads(data)
            
            logger.debug(f"Cache miss for {key}")
            return None
            
        except Exception as e:
            logger.warning(f"Cache read error: {e}")
            return None
    
    def set_cached_stock_data(
        self, 
        stock_code: str, 
        data: Dict, 
        ttl: int = 3600
    ) -> bool:
        """
        Cache stock data.
        
        Args:
            stock_code: Stock symbol
            data: Data to cache
            ttl: Time to live in seconds (default 1 hour)
        
        Returns:
            True if successful
        """
        try:
            # Get the date from data
            date = data.get('date', datetime.now().strftime('%Y-%m-%d'))
            key = self._make_key("stock", stock_code, date)
            
            # Serialize data
            serialized = json.dumps(data, default=str)
            
            self.client.setex(key, ttl, serialized)
            logger.debug(f"Cached stock data: {key} (TTL={ttl}s)")
            return True
            
        except Exception as e:
            logger.warning(f"Cache write error: {e}")
            return False
    
    def set_cached_alpha(
        self, 
        stock_code: str, 
        date: str, 
        alpha_values: Dict[str, float], 
        ttl: int = 3600
    ) -> bool:
        """
        Cache alpha calculation results.
        
        Args:
            stock_code: Stock symbol
            date: Date
            alpha_values: Dict of {alpha_name: value}
            ttl: Time to live in seconds
        
        Returns:
            True if successful
        """
        try:
            key = self._make_key("alpha", stock_code, date)
            serialized = json.dumps(alpha_values, default=str)
            self.client.setex(key, ttl, serialized)
            logger.debug(f"Cached alpha values: {key} (TTL={ttl}s)")
            return True
            
        except Exception as e:
            logger.warning(f"Cache write error for alpha: {e}")
            return False
    
    def get_cached_alpha(
        self, 
        stock_code: str, 
        date: str
    ) -> Optional[Dict[str, float]]:
        """
        Get cached alpha values.
        
        Args:
            stock_code: Stock symbol
            date: Date
        
        Returns:
            Dict of {alpha_name: value} or None
        """
        try:
            key = self._make_key("alpha", stock_code, date)
            data = self.client.get(key)
            
            if data:
                logger.debug(f"Cache hit for alpha: {key}")
                return json.loads(data)
            
            logger.debug(f"Cache miss for alpha: {key}")
            return None
            
        except Exception as e:
            logger.warning(f"Cache read error for alpha: {e}")
            return None
    
    def invalidate_stock_cache(self, stock_code: str) -> int:
        """
        Invalidate all cached data for a stock.
        
        Args:
            stock_code: Stock symbol
        
        Returns:
            Number of keys deleted
        """
        try:
            pattern = self._make_key("stock", stock_code, "*")
            keys = self.client.keys(pattern)
            
            if keys:
                deleted = self.client.delete(*keys)
                logger.info(f"Invalidated {deleted} cache entries for {stock_code}")
                return deleted
            
            return 0
            
        except Exception as e:
            logger.warning(f"Cache invalidation error: {e}")
            return 0
    
    def invalidate_alpha_cache(self, stock_code: str = None, date: str = None) -> int:
        """
        Invalidate cached alpha values.
        
        Args:
            stock_code: Optional stock code filter
            date: Optional date filter
        
        Returns:
            Number of keys deleted
        """
        try:
            if stock_code and date:
                pattern = self._make_key("alpha", stock_code, date)
            elif stock_code:
                pattern = self._make_key("alpha", stock_code, "*")
            else:
                pattern = self._make_key("alpha", "*", "*")
            
            keys = self.client.keys(pattern)
            
            if keys:
                deleted = self.client.delete(*keys)
                logger.info(f"Invalidated {deleted} alpha cache entries")
                return deleted
            
            return 0
            
        except Exception as e:
            logger.warning(f"Alpha cache invalidation error: {e}")
            return 0
    
    def cache_market_data(self, market_data: Dict, ttl: int = 3600) -> bool:
        """
        Cache market-wide data (indexes, etc.).
        
        Args:
            market_data: Dict of market data
            ttl: Time to live in seconds
        
        Returns:
            True if successful
        """
        try:
            key = self._make_key("market", "general", datetime.now().strftime('%Y%m%d'))
            serialized = json.dumps(market_data, default=str)
            self.client.setex(key, ttl, serialized)
            return True
            
        except Exception as e:
            logger.warning(f"Market cache error: {e}")
            return False
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        try:
            info = self.client.info("stats")
            memory = self.client.info("memory")
            
            return {
                'connected': True,
                'key_count': len(self.client.keys("stock_sim:*")),
                'hits': info.get('keyspace_hits', 0),
                'misses': info.get('keyspace_misses', 0),
                'used_memory': memory.get('used_memory_human', 'N/A'),
                'uptime': info.get('uptime_in_seconds', 0)
            }
            
        except Exception as e:
            return {
                'connected': False,
                'error': str(e)
            }
    
    def close(self):
        """Close Redis connection"""
        try:
            self.client.close()
            logger.info("Redis connection closed")
        except Exception as e:
            logger.warning(f"Error closing Redis: {e}")


# Global cache instance
_cache: Optional[RedisCache] = None


def get_cache() -> RedisCache:
    """Get or create global cache instance"""
    global _cache
    if _cache is None:
        _cache = RedisCache()
    return _cache


if __name__ == "__main__":
    # Test Redis connection
    cache = RedisCache()
    
    # Test basic operations
    test_data = {
        'stock_code': '000001',
        'date': '2024-01-01',
        'close': 10.5,
        'volume': 1000000
    }
    
    cache.set_cached_stock_data('000001', test_data)
    result = cache.get_cached_stock_data('000001', '2024-01-01')
    print(f"Test result: {result}")
    
    # Show stats
    print(f"Cache stats: {cache.get_cache_stats()}")
