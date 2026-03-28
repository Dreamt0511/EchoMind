import asyncpg
import json
from typing import List, Optional

class PostgreSQLParentClient:
    """PostgreSQL 客户端，用于存储父块"""
    
    def __init__(self, dsn: str):
        self.dsn = dsn
        self.pool = None
    
    async def init_pool(self):
        """初始化连接池"""
        self.pool = await asyncpg.create_pool(self.dsn, min_size=5, max_size=20)
        
        # 创建表
        async with self.pool.acquire() as conn:
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS parent_documents (
                    parent_id VARCHAR(128) PRIMARY KEY,
                    knowledge_base_id VARCHAR(128) NOT NULL,
                    text TEXT NOT NULL,
                    metadata JSONB,
                    total_chunks INTEGER DEFAULT 0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                
                CREATE INDEX IF NOT EXISTS idx_parent_kb 
                ON parent_documents(knowledge_base_id);
                
                CREATE INDEX IF NOT EXISTS idx_parent_created 
                ON parent_documents(created_at);
            """)
    
    async def add_parent(self, parent_id: str, knowledge_base_id: str, 
                         text: str, metadata: dict = None, total_chunks: int = 0):
        """添加父块"""
        async with self.pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO parent_documents (parent_id, knowledge_base_id, text, metadata, total_chunks)
                VALUES ($1, $2, $3, $4, $5)
                ON CONFLICT (parent_id) 
                DO UPDATE SET 
                    text = EXCLUDED.text,
                    metadata = EXCLUDED.metadata,
                    total_chunks = EXCLUDED.total_chunks,
                    updated_at = CURRENT_TIMESTAMP
            """, parent_id, knowledge_base_id, text, json.dumps(metadata or {}), total_chunks)
    
    async def get_parents(self, parent_ids: List[str]) -> List[dict]:
        """批量获取父块"""
        if not parent_ids:
            return []
        
        async with self.pool.acquire() as conn:
            results = await conn.fetch("""
                SELECT parent_id, text, metadata, total_chunks, created_at
                FROM parent_documents
                WHERE parent_id = ANY($1::text[])
            """, parent_ids)
            
            return [dict(r) for r in results]
    
    async def delete_parent(self, parent_id: str):
        """删除父块"""
        async with self.pool.acquire() as conn:
            await conn.execute("""
                DELETE FROM parent_documents WHERE parent_id = $1
            """, parent_id)
    
    async def delete_knowledge_base(self, knowledge_base_id: str):
        """删除整个知识库的父块"""
        async with self.pool.acquire() as conn:
            await conn.execute("""
                DELETE FROM parent_documents WHERE knowledge_base_id = $1
            """, knowledge_base_id)
    
    async def close(self):
        """关闭连接池"""
        if self.pool:
            await self.pool.close()