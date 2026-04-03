import asyncpg
import asyncio

async def test():
    try:
        conn = await asyncpg.connect(
            user='dreamt',
            password='0511',
            database='postgres',
            host='localhost',
            port=5432
        )
        print("连接成功！")
        await conn.close()
    except Exception as e:
        print(f"连接失败: {e}")

asyncio.run(test())