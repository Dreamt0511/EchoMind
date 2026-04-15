import redis
import json
import sys
import os

# 将项目根目录添加到 Python 路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# 加载环境变量 - 添加这部分
from dotenv import load_dotenv
# 查找 .env 文件（可能在项目根目录）
env_path = os.path.join(project_root, '.env')
load_dotenv(env_path)

from redis_cache import get_redis_client
import logging
import asyncio

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def invalidate_user_profile_cache(user_id: int) -> bool:
    """使指定用户的画像缓存失效"""
    try:
        redis_client = await get_redis_client()
        cache_key = f"user_profile:{user_id}"
        await redis_client.delete(cache_key)
        logger.info(f"用户画像缓存已失效 user_id={user_id}")
        return True
    except Exception as e:
        logger.error(f"删除用户画像缓存失败: {e}")
        return False

# 连接 Redis
r = redis.Redis(
    host='localhost',
    port=6379,
    db=0,
    decode_responses=True
)

user_id = 1

# 遍历并删除所有包含 user_id 的 key
print(f"开始删除包含 {user_id} 的所有 key...\n")
deleted_count = 0
for key in r.scan_iter(match=f"*{user_id}*"):
    print(f"删除 key: {key}")
    r.delete(key)
    deleted_count += 1

print(f"\n✅ 同步删除完成！共删除 {deleted_count} 个 key")

# 运行异步删除
asyncio.run(invalidate_user_profile_cache(user_id))