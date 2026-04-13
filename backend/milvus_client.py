import os
import asyncio
from dotenv import load_dotenv
from pymilvus import AsyncMilvusClient
from langchain_community.embeddings import DashScopeEmbeddings
import logging
from memory_manager import MemoryManager
from knowledeg_base_manager import KnowledgeBaseManager



logger = logging.getLogger(__name__)

# 加载环境变量
load_dotenv()

_global_milvus_client = None


async def get_milvus_client():
    """获取全局 Milvus 客户端实例"""
    global _global_milvus_client
    if _global_milvus_client is None:
        _global_milvus_client = AsyncMilvusClientWrapper()
        await _global_milvus_client.ensure_collection()
    return _global_milvus_client


class AsyncMilvusClientWrapper:
    """Milvus 客户端包装器 - 只负责向量数据库连接和集合初始化"""

    _instance = None
    _singleton_initialized = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if self._singleton_initialized:
            return

        self.knowledge_base_collection = os.getenv("knowledge_base_collection")
        self.memory_collection = os.getenv("memory_collection")

        self.embeddings = DashScopeEmbeddings(
            model=os.getenv("EMBEDDING_MODEL"),
            dashscope_api_key=os.getenv("DASHSCOPE_API_KEY"),
        )
        self.client = AsyncMilvusClient(
            uri=os.getenv("Milvus_url"),
            token=os.getenv("Token"),
        )
        self.dense_dim = int(os.getenv("dense_dimension", "1024"))
        self._collection_initialized = False
        self._collection_lock = asyncio.Lock()

        self._singleton_initialized = True


        self.memory_manager = MemoryManager(
            client=self.client,
            embeddings=self.embeddings,
            collection_name=self.memory_collection,
            dense_dim=self.dense_dim,
        )

        self.knowledge_base_manager = KnowledgeBaseManager(
            client=self.client,
            embeddings=self.embeddings,
            collection_name=self.knowledge_base_collection,
            dense_dim=self.dense_dim,
        )



    async def ensure_collection(self):
        """确保集合已初始化（只在启动时调用一次）"""
        if self._collection_initialized:
            return

        async with self._collection_lock:
            if self._collection_initialized:
                return
            await self._init_knowledge_base_collection()
            await self._init_memory_collection()
            self._collection_initialized = True

    async def _init_memory_collection(self):
        """初始化对话记忆集合"""
        await self.memory_manager.init_collection()

    async def _init_knowledge_base_collection(self):
        """初始化知识库集合"""
        await self.knowledge_base_manager.init_collection()

    async def close(self):
        """关闭客户端连接"""
        if self.client:
            await self.client.close()
            logger.info("Milvus 客户端已关闭")

    # ==================== 记忆管理方法代理 ====================
    async def hybrid_retrieval_memories(self, query: str, user_id: int, summary_k: int,
                                        semantic_k: int, episodic_k: int, procedural_k: int):
        """混合检索记忆 - 代理到 memory_manager"""
        return await self.memory_manager.hybrid_retrieval_memories(
            query, user_id, summary_k, semantic_k, episodic_k, procedural_k
        )

    async def resolve_conflicts(self, filtered_memory: dict, user_id: int) -> dict:
        """检测并过滤重复记忆 - 代理到 memory_manager"""
        return await self.memory_manager.resolve_conflicts(filtered_memory, user_id)

    async def add_memories_batch(self, user_id: int, thread_id: str, memory_dict: dict,
                                  summary_id: str = None, **kwargs) -> bool:
        """批量添加记忆 - 代理到 memory_manager"""
        return await self.memory_manager.add_memories_batch(
            user_id, thread_id, memory_dict, summary_id, **kwargs
        )

    # ==================== 知识库管理方法代理 ====================

    async def hybrid_retrieval_knowledge_base(self, query: str, knowledge_base_id: str,
                                              top_k: int, user_id: int):
        """混合检索知识库 - 代理到 knowledge_base_manager"""
        return await self.knowledge_base_manager.hybrid_retrieval_knowledge_base(
            query, knowledge_base_id, top_k, user_id
        )

    async def add_chunks_batch(self, knowledge_base_id: str, chunks, user_id: int) -> bool:
        """批量添加知识块 - 代理到 knowledge_base_manager"""
        return await self.knowledge_base_manager.add_chunks_batch(
            knowledge_base_id, chunks, user_id
        )

    async def delete_knowledge_file_chunks(self, knowledge_base_id: str, user_id: int) -> bool:
        """删除知识库文件块 - 代理到 knowledge_base_manager"""
        return await self.knowledge_base_manager.delete_knowledge_file_chunks(
            knowledge_base_id, user_id
        )
    
    async def delete_file_chunks(
        self, knowledge_base_id: str, file_hash: str, user_id: int
    ):
        """删除知识库文件 - 代理到 knowledge_base_manager"""
        return await self.knowledge_base_manager.delete_file_chunks(
            knowledge_base_id = knowledge_base_id, file_hash=file_hash, user_id = user_id
        )

