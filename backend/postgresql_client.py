import asyncpg
import json
import logging
from typing import List, Optional, Dict, Any
from dataclasses import dataclass, field
from datetime import datetime
from dotenv import load_dotenv
import os

load_dotenv()

logger = logging.getLogger(__name__)


async def ensure_database_exists(dsn: str) -> None:
    """确保数据库存在，不存在则自动创建"""
    try:
        await asyncpg.connect(dsn)
    except asyncpg.InvalidCatalogNameError as e:
        db_name = str(e).split('"')[1]
        default_dsn = dsn.rsplit('/', 1)[0] + '/postgres'
        conn = await asyncpg.connect(default_dsn)
        try:
            await conn.execute(f"CREATE DATABASE {db_name}")
            logger.info(f"Created database: {db_name}")
        finally:
            await conn.close()


@dataclass
class ParentDocument:
    """父文档数据类"""
    parent_id: str
    text: str
    knowledge_base_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            'parent_id': self.parent_id,
            'knowledge_base_id': self.knowledge_base_id,
            'text': self.text,
            'metadata': self.metadata,
            'created_at': self.created_at,
            'updated_at': self.updated_at
        }


class PostgreSQLParentClient:
    """PostgreSQL 客户端，用于存储父块"""

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
        self.dsn = dsn or os.getenv("DATABASE_URL")
        if not self.dsn:
            raise ValueError("DSN must be provided or set in DATABASE_URL environment variable")
        
        self.min_size = min_size
        self.max_size = max_size
        self.auto_create_db = auto_create_db
        self.pool: Optional[asyncpg.Pool] = None

    async def __aenter__(self):
        await self.init_pool()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()

    async def init_pool(self) -> None:
        """初始化连接池并创建表"""
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
                    CREATE TABLE IF NOT EXISTS parent_documents (
                        parent_id VARCHAR(128) PRIMARY KEY,
                        knowledge_base_id VARCHAR(128) NOT NULL,
                        text TEXT NOT NULL,
                        metadata JSONB DEFAULT '{}'::jsonb,
                        created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
                    );
                    
                    -- 单列索引：加速按知识库查询
                    CREATE INDEX IF NOT EXISTS idx_parent_kb 
                    ON parent_documents(knowledge_base_id);
                    
                    -- 单列索引：加速按时间查询
                    CREATE INDEX IF NOT EXISTS idx_parent_created 
                    ON parent_documents(created_at);
                    
                    -- 复合索引：加速按知识库+时间排序查询
                    CREATE INDEX IF NOT EXISTS idx_parent_kb_created 
                    ON parent_documents(knowledge_base_id, created_at DESC);
                    
                    -- GIN索引：加速JSONB字段内部查询
                    CREATE INDEX IF NOT EXISTS idx_parent_metadata_gin 
                    ON parent_documents USING gin(metadata);
                """)

            logger.info("PostgreSQL client initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize PostgreSQL client: {e}")
            raise

    async def add_parent(self, parent_id: str, knowledge_base_id: str,
                         text: str, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """添加或更新父块"""
        if not self.pool:
            raise RuntimeError("Connection pool not initialized")

        try:
            async with self.pool.acquire() as conn:
                await conn.execute("""
                    INSERT INTO parent_documents (parent_id, knowledge_base_id, text, metadata)
                    VALUES ($1, $2, $3, $4)
                    ON CONFLICT (parent_id) 
                    DO UPDATE SET 
                        knowledge_base_id = EXCLUDED.knowledge_base_id,
                        text = EXCLUDED.text,
                        metadata = EXCLUDED.metadata,
                        updated_at = CURRENT_TIMESTAMP
                """, parent_id, knowledge_base_id, text, json.dumps(metadata or {}))

                return True

        except Exception as e:
            logger.error(f"Error adding parent {parent_id}: {e}")
            raise

    async def get_parent(self, parent_id: str) -> Optional[ParentDocument]:
        """获取单个父块"""
        results = await self.get_parents([parent_id])
        return results[0] if results else None

    async def get_parents(self, parent_ids: List[str]) -> List[ParentDocument]:
        """批量获取父块"""
        if not self.pool:
            raise RuntimeError("Connection pool not initialized")

        if not parent_ids:
            return []

        try:
            async with self.pool.acquire() as conn:
                results = await conn.fetch("""
                    SELECT parent_id, knowledge_base_id, text, metadata, created_at, updated_at
                    FROM parent_documents
                    WHERE parent_id = ANY($1::text[])
                """, parent_ids)

                return [
                    ParentDocument(
                        parent_id=r['parent_id'],
                        knowledge_base_id=r['knowledge_base_id'],
                        text=r['text'],
                        metadata=r['metadata'] or {},
                        created_at=r['created_at'],
                        updated_at=r['updated_at']
                    ) for r in results
                ]

        except Exception as e:
            logger.error(f"Error getting parents: {e}")
            raise

    async def get_parents_by_knowledge_base(
        self,
        knowledge_base_id: str,
        limit: int = 100,
        offset: int = 0
    ) -> List[ParentDocument]:
        """获取知识库下的所有父块"""
        if not self.pool:
            raise RuntimeError("Connection pool not initialized")

        try:
            async with self.pool.acquire() as conn:
                results = await conn.fetch("""
                    SELECT parent_id, knowledge_base_id, text, metadata, created_at, updated_at
                    FROM parent_documents
                    WHERE knowledge_base_id = $1
                    ORDER BY created_at DESC
                    LIMIT $2 OFFSET $3
                """, knowledge_base_id, limit, offset)

                return [
                    ParentDocument(
                        parent_id=r['parent_id'],
                        knowledge_base_id=r['knowledge_base_id'],
                        text=r['text'],
                        metadata=r['metadata'] or {},
                        created_at=r['created_at'],
                        updated_at=r['updated_at']
                    ) for r in results
                ]

        except Exception as e:
            logger.error(f"Error getting parents by knowledge base: {e}")
            raise

    async def delete_parent(self, parent_id: str) -> bool:
        """删除父块"""
        if not self.pool:
            raise RuntimeError("Connection pool not initialized")

        try:
            async with self.pool.acquire() as conn:
                result = await conn.execute("""
                    DELETE FROM parent_documents WHERE parent_id = $1
                """, parent_id)

                return result == "DELETE 1"

        except Exception as e:
            logger.error(f"Error deleting parent {parent_id}: {e}")
            raise

    async def delete_knowledge_base(self, knowledge_base_id: str) -> int:
        """删除整个知识库的父块"""
        if not self.pool:
            raise RuntimeError("Connection pool not initialized")

        try:
            async with self.pool.acquire() as conn:
                result = await conn.execute("""
                    DELETE FROM parent_documents WHERE knowledge_base_id = $1
                """, knowledge_base_id)

                return int(result.split()[-1]) if result.startswith("DELETE ") else 0

        except Exception as e:
            logger.error(f"Error deleting knowledge base {knowledge_base_id}: {e}")
            raise

    async def close(self):
        """关闭连接池"""
        if self.pool:
            await self.pool.close()
            logger.info("PostgreSQL connection pool closed")

import asyncio
async def main():
    """查看数据库所有内容（完整文本）"""
    async with PostgreSQLParentClient(
        dsn=os.getenv("DATABASE_URL"),
        auto_create_db=True
    ) as client:
        
        # #删除记录
        # await client.delete_parent("6a43c051-717b-4bbe-9797-5f13b18aabc7")

        async with client.pool.acquire() as conn:
            records = await conn.fetch("""
                SELECT 
                    parent_id,
                    knowledge_base_id,
                    text,
                    metadata,
                    created_at
                FROM parent_documents
                ORDER BY created_at DESC
            """)
            
            print(f"数据库内容（共 {len(records)} 条记录）:")
            print("=" * 80)
            
            for idx, record in enumerate(records, 1):
                print(f"\n记录 {idx}:")
                print(f"  parent_id: {record['parent_id']}")
                print(f"  knowledge_base_id: {record['knowledge_base_id']}")
                print(f"  text: {record['text']}")
                print(f"  metadata: {record['metadata']}")
                print(f"  created_at: {record['created_at']}")
                print("=" * 80)

if __name__ == "__main__":
    asyncio.run(main())