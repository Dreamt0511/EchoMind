import asyncpg
import json
import logging
from typing import List, Optional, Dict, Any
from dataclasses import dataclass, field
from datetime import datetime
from dotenv import load_dotenv
import os
import asyncio

load_dotenv()

logger = logging.getLogger(__name__)


async def ensure_database_exists(dsn: str) -> None:
    """确保数据库存在，不存在则自动创建"""
    # 移除可能的查询参数
    clean_dsn = dsn.split('?')[0]
    
    try:
        conn = await asyncpg.connect(clean_dsn)
        await conn.close()
        logger.info(f"Database already exists")
        return True
    except asyncpg.InvalidCatalogNameError as e:
        # 数据库不存在，创建它
        db_name = str(e).split('"')[1]
        # 连接到 postgres 数据库
        default_dsn = clean_dsn.rsplit('/', 1)[0] + '/postgres'
        
        try:
            conn = await asyncpg.connect(default_dsn)
            # 使用引号包裹数据库名，避免 SQL 注入和大小写问题
            await conn.execute(f'CREATE DATABASE "{db_name}"')
            logger.info(f"Created database: {db_name}")
            await conn.close()
            return True
        except Exception as create_error:
            logger.error(f"Failed to create database: {create_error}")
            raise
    except Exception as e:
        logger.error(f"Failed to connect to database: {e}")
        raise

@dataclass
class ParentDocument:
    """父文档数据类"""
    parent_id: str
    text: str
    knowledge_base_id: Optional[str] = None
    file_name: Optional[str] = None      # 新增
    file_hash: Optional[str] = None      # 新增
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            'parent_id': self.parent_id,
            'knowledge_base_id': self.knowledge_base_id,
            'text': self.text,
            'file_name': self.file_name,
            'file_hash': self.file_hash,
            'created_at': self.created_at,
            'updated_at': self.updated_at
        }


class PostgreSQLParentClient:
    """PostgreSQL 客户端，用于存储父块（单例模式）"""

    _instance = None
    _singleton_initialized = False

    def __new__(cls, dsn: Optional[str] = None, min_size: int = 5, max_size: int = 20,
                auto_create_db: bool = True):
        """单例模式"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, dsn: Optional[str] = None, min_size: int = 5, max_size: int = 20,
                 auto_create_db: bool = True):
        """
        初始化 PostgreSQL 客户端

        Args:
            dsn: 数据库连接字符串，如果不提供则从环境变量 DATABASE_URL 读取
            min_size: 连接池最小连接数
            max_size: 连接池最大连接数
            auto_create_db: 是否自动创建数据库
        """
        if self._singleton_initialized:
            return

        self.dsn = dsn or os.getenv("DATABASE_URL")
        if not self.dsn:
            raise ValueError(
                "DSN must be provided or set in DATABASE_URL environment variable")

        self.min_size = min_size
        self.max_size = max_size
        self.auto_create_db = auto_create_db
        self.pool: Optional[asyncpg.Pool] = None
        self._pool_initialized = False
        self._pool_lock = asyncio.Lock()

        self._singleton_initialized = True

    async def init_pool(self) -> None:
        """初始化连接池并创建表（只执行一次）"""
        if self._pool_initialized:
            return

        async with self._pool_lock:
            if self._pool_initialized:
                return

            try:
                if self.auto_create_db:
                    await ensure_database_exists(self.dsn)

                self.pool = await asyncpg.create_pool(
                    self.dsn,
                    min_size=self.min_size,
                    max_size=self.max_size,
                    command_timeout=60
                )

                async with self.pool.acquire() as conn:
                    await conn.execute("""
                        -- 1. 用户表
                        CREATE TABLE IF NOT EXISTS users (
                            user_id SERIAL PRIMARY KEY,
                            username VARCHAR(50) UNIQUE NOT NULL,
                            created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
                        );

                        -- 2. 文件信息表
                        CREATE TABLE IF NOT EXISTS file_metadata (
                            file_hash VARCHAR(64) PRIMARY KEY,
                            file_name VARCHAR(255) NOT NULL,
                            knowledge_base_id VARCHAR(128) NOT NULL, 
                            uploaded_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
                        );               
                        
                        -- 3. 父块表，存储父块文本和关联的文件信息
                        CREATE TABLE IF NOT EXISTS parent_chunks (
                            parent_id VARCHAR(128) PRIMARY KEY,
                            knowledge_base_id VARCHAR(128) NOT NULL,
                            text TEXT NOT NULL,
                            file_name VARCHAR(255) NOT NULL,
                            file_hash VARCHAR(64) NOT NULL,
                            created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
                            updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
                        );
                    """)

                    await conn.execute("""
                        -- 4. 知识库-用户关联表
                        CREATE TABLE IF NOT EXISTS user_knowledge_bases (
                            user_id INTEGER REFERENCES users(user_id) ON DELETE CASCADE,
                            knowledge_base_id VARCHAR(128) NOT NULL,
                            created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
                            PRIMARY KEY (user_id, knowledge_base_id)
                        );

                        -- 索引
                        CREATE INDEX IF NOT EXISTS idx_file_kb ON file_metadata(knowledge_base_id);
                        CREATE INDEX IF NOT EXISTS idx_parent_kb ON parent_chunks(knowledge_base_id);
                        CREATE INDEX IF NOT EXISTS idx_parent_file_hash ON parent_chunks(file_hash);
                    """)

                self._pool_initialized = True
                logger.info("PostgreSQL client initialized successfully")

            except Exception as e:
                logger.error(f"Failed to initialize PostgreSQL client: {e}")
                raise

    async def get_parents(self, parent_ids: List[str]) -> List[ParentDocument]:
        """批量获取父块"""
        if not self.pool:
            raise RuntimeError("Connection pool not initialized")

        if not parent_ids:
            return []

        try:
            async with self.pool.acquire() as conn:
                results = await conn.fetch("""
                    SELECT parent_id, knowledge_base_id, text, file_name, file_hash, created_at, updated_at
                    FROM parent_chunks
                    WHERE parent_id = ANY($1::text[])
                """, parent_ids)

                return [
                    ParentDocument(
                        parent_id=r['parent_id'],
                        knowledge_base_id=r['knowledge_base_id'],
                        text=r['text'],
                        file_name=r['file_name'],
                        file_hash=r['file_hash'],
                        created_at=r['created_at'],
                        updated_at=r['updated_at']
                    ) for r in results
                ]

        except Exception as e:
            logger.error(f"Error getting parents: {e}")
            raise

    async def delete_knowledge_base(self, knowledge_base_id: str) -> int:
        """删除整个知识库的父块"""
        if not self.pool:
            raise RuntimeError("Connection pool not initialized")

        try:
            async with self.pool.acquire() as conn:
                result = await conn.execute("""
                    DELETE FROM parent_chunks WHERE knowledge_base_id = $1
                """, knowledge_base_id)

                return int(result.split()[-1]) if result.startswith("DELETE ") else 0

        except Exception as e:
            logger.error(
                f"Error deleting knowledge base {knowledge_base_id}: {e}")
            raise

    async def add_parent_chunk_batch(self, parents_data: List[Dict[str, Any]]) -> int:
        """
        批量添加父块
        parents_data 格式: [{"parent_id": "...", "knowledge_base_id": "...", "text": "...", "file_name": "...", "file_hash": "..."}]
        """
        if not self.pool:
            raise RuntimeError("Connection pool not initialized")

        try:
            async with self.pool.acquire() as conn:
                async with conn.transaction():
                    for data in parents_data:
                        await conn.execute(
                        """
                        INSERT INTO parent_chunks (parent_id, knowledge_base_id, text, file_name, file_hash)
                        VALUES ($1, $2, $3, $4, $5)
                        ON CONFLICT (parent_id) DO UPDATE 
                        SET text = EXCLUDED.text, 
                        file_name = EXCLUDED.file_name, 
                        file_hash = EXCLUDED.file_hash,
                        updated_at = CURRENT_TIMESTAMP
                        """, 
                        data["parent_id"], data["knowledge_base_id"], data["text"], data["file_name"], data["file_hash"])
            return len(parents_data)
        except Exception as e:
            logger.error(f"Error batch adding parents: {e}")
            raise

    async def get_user_knowledge_bases(self, user_id: int) -> List[str]:
        """获取用户的所有知识库名称"""
        if not self.pool:
            raise RuntimeError("Connection pool not initialized")

        async with self.pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT knowledge_base_id 
                FROM user_knowledge_bases 
                WHERE user_id = $1
            """, user_id)
            return [row['knowledge_base_id'] for row in rows]

    async def get_knowledge_base_files(self, knowledge_base_id: str) -> List[Dict]:
        """获取知识库中的所有文件（从 file_metadata 表读取）"""
        if not self.pool:
            raise RuntimeError("Connection pool not initialized")

        async with self.pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT file_hash, file_name, uploaded_at
                FROM file_metadata
                WHERE knowledge_base_id = $1
                ORDER BY uploaded_at DESC
            """, knowledge_base_id)
            return [dict(row) for row in rows]

    async def add_file_metadata(self, file_hash: str, file_name: str, knowledge_base_id: str):
        """添加文件元数据"""
        if not self.pool:
            raise RuntimeError("Connection pool not initialized")

        async with self.pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO file_metadata (file_hash, file_name, knowledge_base_id)
                VALUES ($1, $2, $3)
                ON CONFLICT (file_hash) DO UPDATE 
                SET file_name = EXCLUDED.file_name, knowledge_base_id = EXCLUDED.knowledge_base_id
            """, file_hash, file_name, knowledge_base_id)

    async def delete_file_by_hash(self, file_hash: str, knowledge_base_id: str) -> int:
        """
        删除文件：同时删除 file_metadata 和 parent_chunks 中的记录
        返回删除的 parent_chunks 数量
        """
        if not self.pool:
            raise RuntimeError("Connection pool not initialized")

        async with self.pool.acquire() as conn:
            async with conn.transaction():
                # 1. 删除 parent_chunks 中的记录（直接使用 file_hash 字段）
                result = await conn.execute("""
                    DELETE FROM parent_chunks
                    WHERE knowledge_base_id = $1
                    AND file_hash = $2
                """, knowledge_base_id, file_hash)

                deleted_count = int(
                    result.split()[-1]) if result.startswith("DELETE ") else 0

                # 2. 删除 file_metadata 中的记录
                await conn.execute("""
                    DELETE FROM file_metadata
                    WHERE file_hash = $1 AND knowledge_base_id = $2
                """, file_hash, knowledge_base_id)

                logger.info(f"Deleted {deleted_count} parent documents for file_hash: {file_hash}")
                return deleted_count

    async def create_knowledge_base_for_user(self, user_id: int, knowledge_base_id: str):
        """为用户创建知识库"""
        if not self.pool:
            raise RuntimeError("Connection pool not initialized")

        async with self.pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO user_knowledge_bases (user_id, knowledge_base_id)
                VALUES ($1, $2)
                ON CONFLICT (user_id, knowledge_base_id) DO NOTHING
            """, user_id, knowledge_base_id)

    async def close(self):
        """关闭连接池"""
        if self.pool:
            await self.pool.close()
            logger.info("PostgreSQL connection pool closed")


_global_postgresql_client = None


async def get_postgresql_client():
    """获取全局 PostgreSQL 客户端实例"""
    global _global_postgresql_client
    if _global_postgresql_client is None:
        _global_postgresql_client = PostgreSQLParentClient()
        await _global_postgresql_client.init_pool()
    return _global_postgresql_client


async def main():
    """查看数据库所有内容"""
    client = await get_postgresql_client()

    await client.init_pool()  # 确保连接池已初始化

    async with client.pool.acquire() as conn:
        records = await conn.fetch("""
            SELECT 
                parent_id,
                knowledge_base_id,
                text,
                file_name,
                file_hash,
                created_at,
                updated_at
            FROM parent_chunks
            ORDER BY created_at DESC
        """)

        print("=" * 80)
        print(f"数据库内容（共 {len(records)} 条记录）:")
        
        for idx, record in enumerate(records, 1):
            print(f"\n记录 {idx}:")
            print(f"  parent_id: {record['parent_id']}")
            print(f"  knowledge_base_id: {record['knowledge_base_id']}")
            print(f"  text: {record['text'][:100]}..." if len(record['text']) > 100 else f"  text: {record['text']}")
            print(f"  file_name: {record['file_name']}")
            print(f"  file_hash: {record['file_hash']}")
            print(f"  created_at: {record['created_at']}")
            print(f"  updated_at: {record['updated_at']}")
            print("-" * 80)

    await client.close()


if __name__ == "__main__":
    asyncio.run(main())