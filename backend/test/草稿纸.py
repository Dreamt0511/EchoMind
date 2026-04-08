import redis

# 连接方式完全一样
redis_client = redis.Redis(
    host='localhost',
    port=6379,
    decode_responses=True
)

# 测试连接
print(redis_client.ping())  # 输出: True