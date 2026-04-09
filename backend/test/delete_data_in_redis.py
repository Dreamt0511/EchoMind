import redis
import json

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
for key in r.scan_iter(match=f"*{user_id}*"):
    print(f"删除 key: {key}")
    r.delete(key)  # 这一行就是删除命令

print("\n✅ 删除完成！")