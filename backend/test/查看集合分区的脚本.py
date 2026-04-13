import asyncio
from pymilvus import AsyncMilvusClient
import os
import dotenv

# 加载环境变量
dotenv.load_dotenv()

Token = os.getenv("Token")

# ===================== 你的 Milvus 配置 =====================
MILVUS_URI = "https://in03-bf51824a0cbc1a5.serverless.ali-cn-hangzhou.cloud.zilliz.com.cn"
MILVUS_TOKEN = Token
# ==========================================================

async def check_partitions():
    client = AsyncMilvusClient(uri=MILVUS_URI, token=MILVUS_TOKEN)

    print("\n===== 所有集合 =====")
    collections = await client.list_collections()
    for c in collections:
        print(f"- {c}")

    print("\n===== 每个集合的分区 =====")
    for coll in collections:
        partitions = await client.list_partitions(coll)
        print(f"\n【集合】{coll}")
        print(f"  分区列表：{partitions}")

        # 直接判断你要的分区在不在
        if "summary_memory" in partitions:
            print(f"  ✅ summary_memory 存在")
        else:
            print(f"  ❌ summary_memory 不存在！！！")

    await client.close()

if __name__ == "__main__":
    asyncio.run(check_partitions())