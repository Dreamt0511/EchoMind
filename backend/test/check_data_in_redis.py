import redis
import json

# 连接 Redis（根据你的 Redis_URI 调整）
r = redis.Redis(
    host='localhost',
    port=6379,
    db=0,
    decode_responses=True  # 自动将响应解码为字符串，方便查看
)

# 查找所有 Key
user_id = 1
for key in r.scan_iter(match=f"*{user_id}*"):
    key_type = r.type(key)
    print(f"\nKey: {key} (Type: {key_type})")
    
    # 根据类型获取值
    if key_type == 'ReJSON-RL':
        # 对于 JSON 类型，获取并美化打印
        value = r.json().get(key)
        print(json.dumps(value, indent=2, ensure_ascii=False))
    elif key_type == 'string':
        value = r.get(key)
        print(f"Value: {value}")
    elif key_type == 'hash':
        value = r.hgetall(key)
        print(f"Value: {value}")
    # ... 可以按需添加对其他类型的处理