import asyncpg
import logging
from typing import List, Optional, Dict, Any, Set
from dataclasses import dataclass, field
from datetime import datetime
from dotenv import load_dotenv
import os
import asyncio

load_dotenv()

logger = logging.getLogger(__name__)


async def ensure_database_exists(dsn: str) -> None:
    """确保数据库存在，不存在则自动创建"""
    
    # 直接使用参数，不依赖 DSN 解析
    user = "dreamt"
    password = "0511"
    host = "localhost"
    port = 5432
    db_name = "echomind_db"
    
    try:
        conn = await asyncpg.connect(
            user=user, password=password, host=host, port=port, database=db_name
        )
        await conn.close()
        logger.info(f"Database '{db_name}' already exists")
    except Exception as e:
        logger.error(f"Error occurred while checking database: {e}")
        logger.info(f"Database '{db_name}' does not exist. Creating...")
        
        conn = await asyncpg.connect(
            user=user, password=password, host=host, port=port, database="postgres"
        )
        await conn.execute(f'CREATE DATABASE "{db_name}"')
        await conn.close()
        logger.info(f"Created database: {db_name}")

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

                        -- 2. 知识库表（关联用户，支持级联删除）
                        CREATE TABLE IF NOT EXISTS knowledge_bases (
                            knowledge_base_id VARCHAR(128) NOT NULL,
                            user_id INTEGER NOT NULL,
                            created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
                            PRIMARY KEY (knowledge_base_id, user_id),
                            FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE
                        );

                        -- 3. 文件信息表（关联知识库和用户，支持级联删除）
                        CREATE TABLE IF NOT EXISTS file_metadata (
                            file_hash VARCHAR(64) NOT NULL,
                            file_name VARCHAR(255) NOT NULL,
                            knowledge_base_id VARCHAR(128) NOT NULL,
                            user_id INTEGER NOT NULL,
                            uploaded_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
                            PRIMARY KEY (file_hash, knowledge_base_id, user_id),
                            FOREIGN KEY (knowledge_base_id, user_id) REFERENCES knowledge_bases(knowledge_base_id, user_id) ON DELETE CASCADE
                        );               
                        
                        -- 4. 父块表（关联文件，支持级联删除）
                        CREATE TABLE IF NOT EXISTS parent_chunks (
                            parent_id VARCHAR(128) NOT NULL,
                            knowledge_base_id VARCHAR(128) NOT NULL,
                            user_id INTEGER NOT NULL,
                            text TEXT NOT NULL,
                            file_name VARCHAR(255) NOT NULL,
                            file_hash VARCHAR(64) NOT NULL,
                            created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
                            updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
                            PRIMARY KEY (parent_id, knowledge_base_id, user_id),
                            FOREIGN KEY (file_hash, knowledge_base_id, user_id) REFERENCES file_metadata(file_hash, knowledge_base_id, user_id) ON DELETE CASCADE
                        );
                        
                        -- 5. 块哈希表（关联文件，支持级联删除）
                        CREATE TABLE IF NOT EXISTS chunk_hashes (
                            chunk_hash VARCHAR(64) NOT NULL,
                            file_hash VARCHAR(64) NOT NULL,
                            knowledge_base_id VARCHAR(128) NOT NULL,
                            user_id INTEGER NOT NULL,
                            created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
                            PRIMARY KEY (chunk_hash, file_hash, knowledge_base_id, user_id),
                            FOREIGN KEY (file_hash, knowledge_base_id, user_id) REFERENCES file_metadata(file_hash, knowledge_base_id, user_id) ON DELETE CASCADE
                        );
                    """)
                    #为5个表创建6个索引，加速多表关联查询和过滤操作
                    await conn.execute("""
                        -- 索引优化
                        CREATE INDEX IF NOT EXISTS idx_file_kb_user ON file_metadata(knowledge_base_id, user_id);-- 加速查询：查询某知识库下某用户的所有文件
                        CREATE INDEX IF NOT EXISTS idx_parent_kb_user ON parent_chunks(knowledge_base_id, user_id);-- 加速查询：查询某用户在某知识库中的所有父块
                        CREATE INDEX IF NOT EXISTS idx_parent_file_hash ON parent_chunks(file_hash);-- 加速查询：根据文件哈希查找所有父块
                        CREATE INDEX IF NOT EXISTS idx_chunk_hash ON chunk_hashes(chunk_hash);-- 加速查询：快速查找特定哈希值的块（用于去重）
                        CREATE INDEX IF NOT EXISTS idx_chunk_file_hash ON chunk_hashes(file_hash, knowledge_base_id, user_id);-- 复合索引：加速多条件查询，场景：查找某文件在特定知识库中的所有块哈希
                        CREATE INDEX IF NOT EXISTS idx_kb_user ON knowledge_bases(user_id);-- 加速查询：查询某用户的所有知识库
                    """)

                self._pool_initialized = True
                logger.info("PostgreSQL client initialized successfully")

            except Exception as e:
                logger.error(f"Failed to initialize PostgreSQL client: {e}")
                raise

    async def close(self):
        """关闭连接池"""
        if self.pool:
            await self.pool.close()
            logger.info("PostgreSQL connection pool closed")

    # ============ 知识库管理 ============
    async def create_knowledge_base(self, knowledge_base_id: str, user_id: int) -> Dict[str, any]:
        """为用户创建知识库
        Returns:
            Dict[str, any]: {
                "success": bool,
                "message": str,
                "knowledge_base_id": str (可选)
            }
        """
        if not self.pool:
            raise RuntimeError("Connection pool not initialized")
        
        try:
            async with self.pool.acquire() as conn:
                async with conn.transaction():
                    # 先检查知识库是否已存在（可选，但保持风格一致）
                    exists = await conn.fetchval("""
                        SELECT knowledge_base_id FROM knowledge_bases 
                        WHERE knowledge_base_id = $1 AND user_id = $2
                    """, knowledge_base_id, user_id)
                    
                    if exists:
                        logger.warning(f"知识库 {knowledge_base_id} 已存在（用户 {user_id}）")
                        return {
                            "success": False,
                            "message": f"知识库 {knowledge_base_id} 已存在",
                            "knowledge_base_id": knowledge_base_id
                        }
                    
                    # 插入新知识库
                    await conn.execute("""
                        INSERT INTO knowledge_bases (knowledge_base_id, user_id)
                        VALUES ($1, $2)
                    """, knowledge_base_id, user_id)
                    
                    logger.info(f"为用户 {user_id} 创建知识库: {knowledge_base_id}")
                    
                    return {
                        "success": True,
                        "message": f"知识库 {knowledge_base_id} 创建成功",
                        "knowledge_base_id": knowledge_base_id
                    }
                    
        except Exception as e:
            logger.error(f"创建知识库 {knowledge_base_id} 失败: {e}")
            raise
    
    async def delete_knowledge_base(self, knowledge_base_id: str, user_id: int) -> Dict[str, any]:
        """
        删除整个知识库（级联删除会自动处理 file_metadata、parent_chunks、chunk_hashes）
        返回删除的统计信息
        """
        if not self.pool:
            raise RuntimeError("Connection pool not initialized")
        
        try:
            async with self.pool.acquire() as conn:
                async with conn.transaction():
                    # 先检查知识库是否存在且属于该用户
                    exists = await conn.fetchval("""
                        SELECT knowledge_base_id FROM knowledge_bases 
                        WHERE knowledge_base_id = $1 AND user_id = $2
                    """, knowledge_base_id, user_id)
                    
                    if not exists:
                        logger.warning(f"知识库 {knowledge_base_id} 不存在或不属于用户 {user_id}")
                        return {
                            "success": False,
                            "message": "知识库不存在或无权删除",
                            "files_deleted": 0
                        }
                    
                    # 获取要删除的文件数量（用于日志）
                    file_count = await conn.fetchval("""
                        SELECT COUNT(*) FROM file_metadata 
                        WHERE knowledge_base_id = $1 AND user_id = $2
                    """, knowledge_base_id, user_id)
                    
                    # 删除知识库（由于外键级联，会自动删除所有关联数据）
                    await conn.execute("""
                        DELETE FROM knowledge_bases 
                        WHERE knowledge_base_id = $1 AND user_id = $2
                    """, knowledge_base_id, user_id)
                    
                    logger.info(f"成功删除知识库 {knowledge_base_id}（用户 {user_id}），以及其中包含的{file_count} 个文件")
                    
                    return {
                        "success": True,
                        "message": f"知识库删除成功，共删除 {file_count} 个文件",
                        "files_deleted": file_count
                    }
                    
        except Exception as e:
            logger.error(f"删除知识库 {knowledge_base_id} 失败: {e}")
            raise
    
    async def get_user_knowledge_bases(self, user_id: int) -> Dict[str, any]:
        """获取用户的所有知识库ID
        Returns:
            Dict[str, any]: {
                "success": bool,
                "message": str,
                "knowledge_bases": List[str] (可选),
                "count": int (可选)
            }
        """
        if not self.pool:
            raise RuntimeError("Connection pool not initialized")
        
        try:
            async with self.pool.acquire() as conn:
                async with conn.transaction():
                    rows = await conn.fetch("""
                        SELECT knowledge_base_id 
                        FROM knowledge_bases 
                        WHERE user_id = $1
                        ORDER BY created_at DESC
                    """, user_id)
                    
                    knowledge_bases = [row['knowledge_base_id'] for row in rows]
                    count = len(knowledge_bases)
                    
                    logger.info(f"获取用户 {user_id} 的知识库列表，共 {count} 个")
                    
                    return {
                        "success": True,
                        "message": f"成功获取 {count} 个知识库",
                        "knowledge_bases": knowledge_bases,
                        "count": count
                    }
                    
        except Exception as e:
            logger.error(f"获取用户 {user_id} 的知识库列表失败: {e}")
            raise

    # ============ 文件哈希管理 ============
    async def is_file_duplicate(self, file_hash: str, knowledge_base_id: str, user_id: int) -> bool:
        """检查文件哈希在指定用户的知识库中是否已存在"""
        if not self.pool:
            raise RuntimeError("Connection pool not initialized")
        
        try:
            async with self.pool.acquire() as conn:
                result = await conn.fetchval("""
                    SELECT 1 FROM file_metadata 
                    WHERE file_hash = $1 AND knowledge_base_id = $2 AND user_id = $3
                    LIMIT 1
                """, file_hash, knowledge_base_id, user_id)
                return result is not None
        except Exception as e:
            logger.error(f"Error checking file duplicate: {e}")
            raise

    async def add_file_metadata(self, file_hash: str, file_name: str, knowledge_base_id: str, user_id: int):
        """添加文件元数据"""
        if not self.pool:
            raise RuntimeError("Connection pool not initialized")
        
        try:
            async with self.pool.acquire() as conn:
                async with conn.transaction():
                    # 1. 先确保用户存在
                    await conn.execute("""
                        INSERT INTO users (user_id, username)
                        VALUES ($1, $2)
                        ON CONFLICT (user_id) DO NOTHING
                    """, user_id, f"user_{user_id}")
                    
                    # 2. 再确保知识库存在
                    await conn.execute("""
                        INSERT INTO knowledge_bases (knowledge_base_id, user_id)
                        VALUES ($1, $2)
                        ON CONFLICT (knowledge_base_id, user_id) DO NOTHING
                    """, knowledge_base_id, user_id)
                    
                    # 3. 最后添加文件元数据
                    await conn.execute("""
                        INSERT INTO file_metadata (file_hash, file_name, knowledge_base_id, user_id)
                        VALUES ($1, $2, $3, $4)
                        ON CONFLICT (file_hash, knowledge_base_id, user_id) DO UPDATE 
                        SET file_name = EXCLUDED.file_name
                    """, file_hash, file_name, knowledge_base_id, user_id)
                    
                    logger.info(f"添加文件元数据: {file_hash[:16]}... 到知识库 {knowledge_base_id}（用户 {user_id}）")
        except Exception as e:
            logger.error(f"Error adding file metadata: {e}")
            raise

    async def delete_file(self, file_hash: str, knowledge_base_id: str, user_id: int) -> Dict[str, int]:
        """
        删除单个文件（级联删除会自动处理 parent_chunks 和 chunk_hashes）
        返回删除的统计信息
        """
        if not self.pool:
            raise RuntimeError("Connection pool not initialized")
        
        try:
            async with self.pool.acquire() as conn:
                async with conn.transaction():
                    # 获取要删除的父块数量（用于统计）
                    parent_count = await conn.fetchval("""
                        SELECT COUNT(*) FROM parent_chunks 
                        WHERE file_hash = $1 AND knowledge_base_id = $2 AND user_id = $3
                    """, file_hash, knowledge_base_id, user_id)
                    
                    # 删除文件元数据（由于外键级联，会自动删除关联的父块和块哈希）
                    result = await conn.execute("""
                        DELETE FROM file_metadata 
                        WHERE file_hash = $1 AND knowledge_base_id = $2 AND user_id = $3
                    """, file_hash, knowledge_base_id, user_id)
                    
                    deleted_file_count = int(result.split()[-1]) if result.startswith("DELETE ") else 0
                    
                    logger.info(f"删除文件 {file_hash[:16]}（知识库 {knowledge_base_id}，用户 {user_id}），关联的 {parent_count} 个父块将被级联删除")
                    
                    return {
                        "files_deleted": deleted_file_count,
                        "parent_chunks_deleted": parent_count
                    }
        except Exception as e:
            logger.error(f"Error deleting file: {e}")
            raise

    async def get_knowledge_base_files(self, knowledge_base_id: str, user_id: int) -> List[Dict]:
        """获取知识库中的所有文件"""
        if not self.pool:
            raise RuntimeError("Connection pool not initialized")
        
        try:
            async with self.pool.acquire() as conn:
                rows = await conn.fetch("""
                    SELECT file_hash, file_name, uploaded_at
                    FROM file_metadata
                    WHERE knowledge_base_id = $1 AND user_id = $2
                    ORDER BY uploaded_at DESC
                """, knowledge_base_id, user_id)
                return [dict(row) for row in rows]
        except Exception as e:
            logger.error(f"Error getting knowledge base files: {e}")
            raise

    # ============ 父块管理 ============
    async def add_parent_chunk_batch(self, parents_data: List[Dict[str, any]], user_id: int) -> int:
        """
        批量添加父块
        parents_data 格式: [{"parent_id": "...", "knowledge_base_id": "...", "text": "...", "file_name": "...", "file_hash": "..."}]
        """
        if not self.pool:
            raise RuntimeError("Connection pool not initialized")
        
        if not parents_data:
            return 0
        
        try:
            async with self.pool.acquire() as conn:
                async with conn.transaction():
                    for data in parents_data:
                        await conn.execute("""
                            INSERT INTO parent_chunks (parent_id, knowledge_base_id, user_id, text, file_name, file_hash)
                            VALUES ($1, $2, $3, $4, $5, $6)
                            ON CONFLICT (parent_id, knowledge_base_id, user_id) DO UPDATE 
                            SET text = EXCLUDED.text, 
                                file_name = EXCLUDED.file_name, 
                                file_hash = EXCLUDED.file_hash,
                                updated_at = CURRENT_TIMESTAMP
                        """, 
                        data["parent_id"], data["knowledge_base_id"], user_id, 
                        data["text"], data["file_name"], data["file_hash"])
            
            logger.info(f"批量添加 {len(parents_data)} 个父块（用户 {user_id}）")
            return len(parents_data)
        except Exception as e:
            logger.error(f"Error batch adding parents: {e}")
            raise
    
    async def get_parents(self, parent_ids: List[str], knowledge_base_id: str, user_id: int) -> List[str]:
        """批量获取父块"""
        if not self.pool:
            raise RuntimeError("Connection pool not initialized")
        
        if not parent_ids:
            return []
        
        try:
            async with self.pool.acquire() as conn:
                #默认知识库的情况下查询所有知识库的父块文本，其他知识库名就限制在知识库里查找
                if knowledge_base_id == "默认知识库":
                    rows = await conn.fetch("""
                    SELECT text
                    FROM parent_chunks
                    WHERE parent_id = ANY($1::text[]) 
                    AND user_id = $2
                """, parent_ids, user_id)
                else:
                    rows = await conn.fetch("""
                    SELECT text
                    FROM parent_chunks
                    WHERE parent_id = ANY($1::text[]) 
                    AND knowledge_base_id = $2 
                    AND user_id = $3
                """, parent_ids, knowledge_base_id, user_id)

            # 提取所有文本内容
            texts = [row["text"] for row in rows]
            return texts
        except Exception as e:
            logger.error(f"Error getting parents: {e}")
            raise

    
    #文档块去重
    async def batch_check_chunk_duplicates(self, chunk_hashes: List[str], file_hash: str, knowledge_base_id: str, user_id: int) -> Set[str]:
        """
        批量检查块哈希是否已被当前用户的其他文件使用过
        注意：排除当前文件自身的块哈希
        
        Args:
            chunk_hashes: 哈希值列表
            file_hash: 当前文件的哈希值
            knowledge_base_id: 知识库ID
            user_id: 用户ID
            
        Returns:
            已被其他文件使用过的哈希值集合
        """
        if not self.pool:
            raise RuntimeError("Connection pool not initialized")
        
        if not chunk_hashes:
            return set()
        
        try:
            async with self.pool.acquire() as conn:
                # 查找被同一用户同一知识库中其他文件使用过的块哈希
                rows = await conn.fetch("""
                    SELECT DISTINCT chunk_hash FROM chunk_hashes 
                    WHERE chunk_hash = ANY($1::text[])
                    AND file_hash != $2
                    AND knowledge_base_id = $3
                    AND user_id = $4
                """, chunk_hashes, file_hash, knowledge_base_id, user_id)
                
                return {row['chunk_hash'] for row in rows}
        except Exception as e:
            logger.error(f"Error batch checking chunk duplicates: {e}")
            raise

    async def batch_add_chunk_hashes(self, chunk_hashes: List[str], file_hash: str, knowledge_base_id: str, user_id: int):
        """
        批量添加块哈希，关联到指定文件
        
        Args:
            chunk_hashes: 哈希值列表
            file_hash: 文件哈希值
            knowledge_base_id: 知识库ID
            user_id: 用户ID
        """
        if not self.pool:
            raise RuntimeError("Connection pool not initialized")
        
        if not chunk_hashes:
            return
        
        try:
            async with self.pool.acquire() as conn:
                # 构建批量插入参数
                values = [(chunk_hash, file_hash, knowledge_base_id, user_id) for chunk_hash in chunk_hashes]
                await conn.executemany("""
                    INSERT INTO chunk_hashes (chunk_hash, file_hash, knowledge_base_id, user_id)
                    VALUES ($1, $2, $3, $4)
                    ON CONFLICT (chunk_hash, file_hash, knowledge_base_id, user_id) DO NOTHING
                """, values)
                logger.info(f"批量添加 {len(chunk_hashes)} 个块哈希，关联文件 {file_hash[:16]}（用户 {user_id}）")
        except Exception as e:
            logger.error(f"Error batch adding chunk hashes: {e}")
            raise


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