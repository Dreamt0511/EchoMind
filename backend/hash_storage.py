# hash_storage.py
from typing import Set, List
import logging
from postgresql_client import get_postgresql_client

logger = logging.getLogger(__name__)


class HashStorage:
    """
    基于 PostgreSQL 的哈希存储
    文件哈希存储在 file_metadata 表中
    块哈希存储在 chunk_hashes 表中，关联文件哈希
    """
    
    def __init__(self):
        # 延迟初始化，避免循环依赖
        self._pg_client = None
    
    async def _get_pg_client(self):
        """延迟获取 PostgreSQL 客户端"""
        if self._pg_client is None:
            self._pg_client = await get_postgresql_client()
        return self._pg_client
    
    # ============ 文件哈希管理 ============
    
    async def is_file_duplicate(self, file_hash: str, knowledge_base_id: str, user_id: int) -> bool:
        """检查文件哈希是否已存在"""
        client = await self._get_pg_client()
        return await client.is_file_duplicate(file_hash, knowledge_base_id, user_id)
    
    async def add_file_hash(self, file_hash: str, file_name: str, knowledge_base_id: str, user_id: int):
        """添加文件哈希值（同时添加文件元数据）"""
        client = await self._get_pg_client()
        await client.add_file_metadata(file_hash, file_name, knowledge_base_id, user_id)
    
    async def delete_file(self, file_hash: str, knowledge_base_id: str, user_id: int):
        """删除文件（级联删除会自动处理块哈希）"""
        client = await self._get_pg_client()
        return await client.delete_file(file_hash, knowledge_base_id, user_id)
    
    # ============ 块哈希管理 ============
    
    async def batch_check_duplicates(self, chunk_hashes: List[str], file_hash: str, knowledge_base_id: str, user_id: int) -> Set[str]:
        """
        批量检查分块哈希是否已被同一用户的其他文件使用
        
        Args:
            chunk_hashes: 哈希值列表
            file_hash: 当前文件的哈希值（用于排除自身）
            knowledge_base_id: 知识库ID
            user_id: 用户ID
            
        Returns:
            已被其他文件使用过的哈希值集合
        """
        if not chunk_hashes:
            return set()
        
        client = await self._get_pg_client()
        return await client.batch_check_chunk_duplicates(chunk_hashes, file_hash, knowledge_base_id, user_id)
    
    async def batch_add_chunk_hashes(self, chunk_hashes: List[str], file_hash: str, knowledge_base_id: str, user_id: int):
        """
        批量添加分块哈希，关联到指定文件
        
        Args:
            chunk_hashes: 哈希值列表
            file_hash: 文件哈希值
            knowledge_base_id: 知识库ID
            user_id: 用户ID
        """
        if not chunk_hashes:
            return
        
        client = await self._get_pg_client()
        await client.batch_add_chunk_hashes(chunk_hashes, file_hash, knowledge_base_id, user_id)