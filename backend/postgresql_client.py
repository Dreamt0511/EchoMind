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
            raise ValueError(
                "DSN must be provided or set in DATABASE_URL environment variable")

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
                #::text[]：把参数强制转换为文本数组

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
            logger.error(
                f"Error deleting knowledge base {knowledge_base_id}: {e}")
            raise

    async def delete_all_file(self, knowledge_base_id: str, file_hash: str) -> int:
        """
        删除指定知识库中指定文件的所有父块（通过 metadata 中的 file_hash）
        Args:
            file_hash: 文件的哈希值
            knowledge_base_id: 知识库ID
        Returns:
            删除的记录数量
        """
        if not self.pool:
            raise RuntimeError("Connection pool not initialized")

        try:
            async with self.pool.acquire() as conn:
                # 同时使用 knowledge_base_id 和 file_hash 条件删除
                result = await conn.execute("""
                    DELETE FROM parent_documents
                    WHERE knowledge_base_id = $1
                    AND metadata->>'file_hash' = $2
                """, knowledge_base_id, file_hash)

                # 解析删除的行数
                deleted_count = int(
                    result.split()[-1]) if result.startswith("DELETE ") else 0

                if deleted_count > 0:
                    logger.info(
                        f"Deleted {deleted_count} parent documents in the knowledge_base: {knowledge_base_id}")
                else:
                    logger.warning(
                        f"No parent documents foundin knowledge_base: {knowledge_base_id}")

                return deleted_count

        except Exception as e:
            logger.error(
                f"Error deleting file with hash {file_hash} from knowledge_base {knowledge_base_id}: {e}")
            raise

    async def add_parents_batch(self, parents_data: List[Dict[str, Any]]) -> int:
        """批量添加父块"""
        if not self.pool:
            raise RuntimeError("Connection pool not initialized")
        
        #transaction开启数据库事务,如果任何操作失败，所有更改都会回滚保证数据一致性（要么全部成功，要么全部失败）
        try:
            async with self.pool.acquire() as conn:
                async with conn.transaction():
                    for data in parents_data:
                        await conn.execute("""
                            INSERT INTO parent_documents (parent_id, knowledge_base_id, text, metadata)
                            VALUES ($1, $2, $3, $4)
                            ON CONFLICT (parent_id) DO UPDATE 
                            SET text = EXCLUDED.text, metadata = EXCLUDED.metadata, updated_at = CURRENT_TIMESTAMP
                        """, data["parent_id"], data["knowledge_base_id"], data["text"], 
                        json.dumps(data["metadata"] or {}))
            return len(parents_data)
        except Exception as e:
            logger.error(f"Error batch adding parents: {e}")
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
            
            print("=" * 80)
            
            for idx, record in enumerate(records, 1):
                print(f"\n记录 {idx}:")
                print(f"  parent_id: {record['parent_id']}")
                print(f"  knowledge_base_id: {record['knowledge_base_id']}")
                print(f"  text: {record['text']}")
                print(f"  metadata: {record['metadata']}")
                print(f"  created_at: {record['created_at']}")
                print("=" * 80)
            print(f"数据库内容（共 {len(records)} 条记录）:")
if __name__ == "__main__":
    asyncio.run(main())