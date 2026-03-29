import asyncpg
import logging
import asyncio
from dotenv import load_dotenv
import os

load_dotenv()

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def ensure_database_exists(dsn: str) -> None:
    """确保数据库存在，不存在则自动创建"""
    # 解析 DSN 获取数据库名
    base_dsn = dsn.split('?')[0]
    db_name = base_dsn.split('/')[-1]
    
    # 连接到 postgres 系统数据库
    postgres_dsn = base_dsn.rsplit('/', 1)[0] + '/postgres'
    if '?' in dsn:
        postgres_dsn += '?' + dsn.split('?')[1]
    
    conn = await asyncpg.connect(postgres_dsn)
    try:
        # 检查数据库是否存在
        exists = await conn.fetchval(
            "SELECT 1 FROM pg_database WHERE datname = $1", db_name
        )
        
        if not exists:
            logger.info(f"Database '{db_name}' does not exist, creating...")
            await conn.execute(f"CREATE DATABASE {db_name}")
            logger.info(f"✓ Database '{db_name}' created")
        else:
            logger.info(f"Database '{db_name}' already exists")
    finally:
        await conn.close()


if __name__ == "__main__":
    dsn = os.getenv("DATABASE_URL")
    if dsn:
        asyncio.run(ensure_database_exists(dsn))
    else:
        logger.error("DATABASE_URL not found in environment variables")