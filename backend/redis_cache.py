#这个redis_cache.py文件用于管理Redis连接，确保在应用关闭时关闭Redis连接,缓存用户画像，
#对话的缓存由langchain提供的AsyncRedisSaver自动管理
import os
import logging
from dotenv import load_dotenv
from redis.asyncio import Redis
import asyncio

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

load_dotenv(override=True)

REDIS_URI = os.getenv("Redis_URI")

_global_redis_client = None
_redis_lock = asyncio.Lock()

async def get_redis_client():
    global _global_redis_client
    if _global_redis_client is None:
        async with _redis_lock:
            if _global_redis_client is None:  # ← 补上这行
                _global_redis_client = Redis.from_url(
                    REDIS_URI,
                    decode_responses=True,
                    socket_connect_timeout=5,
                    socket_timeout=5,
                    max_connections=20,
                )
                logger.info("Redis client initialized (global singleton)")
    return _global_redis_client



async def close_redis_client():
    """关闭 Redis 连接（应用关闭时调用）"""
    global _global_redis_client
    if _global_redis_client:
        await _global_redis_client.close()
        logger.info("Redis connection closed")
        _global_redis_client = None