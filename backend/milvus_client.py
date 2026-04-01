import os
import asyncio
from dotenv import load_dotenv
from typing import List, Optional, Tuple
from langchain_core.documents import Document
from langchain_community.embeddings import DashScopeEmbeddings
from pymilvus import AsyncMilvusClient, DataType, Function, FunctionType
from pymilvus import AnnSearchRequest, RRFRanker
from hash_storage import HashStorage
import logging
logger = logging.getLogger(__name__)

# 加载环境变量
load_dotenv()


class AsyncMilvusClientWrapper:
    _instance = None
    _singleton_initialized = False
    
    def __new__(cls, hash_storage: Optional["HashStorage"] = None):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self, hash_storage: Optional["HashStorage"] = None):
        # 检查单例是否已初始化
        if self._singleton_initialized:
            return
            
        self.collection_name = os.getenv("collection_name")
        self.embeddings = DashScopeEmbeddings(
            model=os.getenv("EMBEDDING_MODEL"),
            dashscope_api_key=os.getenv("DASHSCOPE_API_KEY")
        )
        self.client = AsyncMilvusClient(
            uri=os.getenv("Milvus_url"),
            token=os.getenv("Token")
        )
        self.dense_dim = int(os.getenv("dense_dimension", "1024"))
        self.hash_storage = hash_storage
        self._collection_initialized = False
        self._collection_lock = asyncio.Lock()
        
        self._singleton_initialized = True

    async def ensure_collection(self):
        """确保集合已初始化（只在启动时调用一次）"""
        if self._collection_initialized:
            return
        
        async with self._collection_lock:
            if self._collection_initialized:
                return
            await self._init_collection()
            self._collection_initialized = True

    async def _init_collection(self):
        """异步初始化集合，配置 BM25 内置函数"""
        # 检查集合是否已存在
        if await self.client.has_collection(self.collection_name):
            logger.info(f"集合 {self.collection_name} 已存在")
            return

        # 创建 Schema
        schema = self.client.create_schema(enable_dynamic_field=True)

        # 添加字段
        schema.add_field(
            field_name="id",
            datatype=DataType.INT64,
            is_primary=True,
            auto_id=True
        )

        schema.add_field(
            field_name="chunk_id",
            datatype=DataType.VARCHAR,
            max_length=128,
            is_primary=False
        )

        schema.add_field(
            field_name="parent_id",
            datatype=DataType.VARCHAR,
            max_length=128
        )

        schema.add_field(
            field_name="knowledge_base_id",
            datatype=DataType.VARCHAR,
            max_length=128
        )

        schema.add_field(
            field_name="chunk_index",
            datatype=DataType.INT32
        )

        schema.add_field(
            field_name="text",
            datatype=DataType.VARCHAR,
            max_length=65535,
            enable_analyzer=True,
            analyzer_params={"type": "chinese"}
        )

        schema.add_field(
            field_name="metadata",
            datatype=DataType.JSON
        )

        schema.add_field(
            field_name="dense_vector",
            datatype=DataType.FLOAT_VECTOR,
            dim=self.dense_dim
        )

        schema.add_field(
            field_name="sparse_bm25",
            datatype=DataType.SPARSE_FLOAT_VECTOR
        )

        # 添加 BM25 内置函数
        bm25_function = Function(
            name="bm25_func",
            function_type=FunctionType.BM25,
            input_field_names=["text"],
            output_field_names=["sparse_bm25"]
        )
        schema.add_function(bm25_function)

        # 配置索引
        index_params = self.client.prepare_index_params()

        index_params.add_index(
            field_name="dense_vector",
            index_name="dense_index",
            index_type="HNSW",
            metric_type="COSINE",
            params={"M": 16, "efConstruction": 200}
        )

        index_params.add_index(
            field_name="sparse_bm25",
            index_name="sparse_bm25_index",
            index_type="SPARSE_INVERTED_INDEX",
            metric_type="BM25"
        )

        # 异步创建集合
        await self.client.create_collection(
            collection_name=self.collection_name,
            schema=schema,
            index_params=index_params
        )

        logger.info(f"集合 {self.collection_name} 创建成功")

    async def add_chunks_batch(self, knowledge_base_id: str, chunks_with_ids: List[Tuple[str, Document]]):
        """批量添加已切好的子块到 Milvus"""
        # 移除 _ensure_initialized 调用，因为启动时已经初始化
        if not chunks_with_ids:
            return

        # 准备数据
        texts = []
        chunks_info = []

        for chunk_id, chunk in chunks_with_ids:
            if "parent_id" not in chunk.metadata:
                raise ValueError(f"子块缺少 parent_id 元数据")

            chunk_index = chunk.metadata.get("chunk_index", 0)
            texts.append(chunk.page_content)
            chunks_info.append({
                "chunk_id": chunk_id,
                "parent_id": chunk.metadata["parent_id"],
                "knowledge_base_id": knowledge_base_id,
                "chunk_index": chunk_index,
                "chunk": chunk
            })
        
        # 使用异步批量接口向量化文本
        dense_vectors = await self.embeddings.aembed_documents(texts)

        # 准备插入数据
        data = []
        for i, chunk_info in enumerate(chunks_info):
            chunk = chunk_info["chunk"]

            stored_metadata = {
                k: v for k, v in chunk.metadata.items()
                if k not in ["parent_id", "chunk_index"]
            }

            data.append({
                "chunk_id": chunk_info["chunk_id"],
                "parent_id": chunk_info["parent_id"],
                "knowledge_base_id": chunk_info["knowledge_base_id"],
                "chunk_index": chunk_info["chunk_index"],
                "text": chunk.page_content,
                "metadata": stored_metadata,
                "dense_vector": dense_vectors[i]
            })

        # 异步批量插入
        result = await self.client.insert(
            collection_name=self.collection_name,
            data=data
        )

        logger.info(f"已添加 {len(data)} 个子块到 Milvus（知识库: {knowledge_base_id}）")
        return result

    async def hybrid_retrieval(self, query: str, knowledge_base_id: Optional[str] = None, top_k: int = 10):
        """混合检索，返回去重后的父块ID列表，混合检索失败时自动降级为稠密向量检索（即语义搜索）"""
        # 构建过滤表达式
        if knowledge_base_id:
            filter_expr = f'knowledge_base_id == "{knowledge_base_id}"'
        else:
            filter_expr = None

        # 生成稠密向量，带重试机制
        max_retries = 3
        for attempt in range(max_retries):
            try:
                dense_vector = await self.embeddings.aembed_query(query)
                break  # 成功则跳出循环
            except Exception as e:
                logger.error(f"Embedding API 调用失败 (尝试 {attempt + 1}/{max_retries}): {e}")
                if attempt == max_retries - 1:  # 最后一次尝试失败
                    logger.error(f"Embedding API 最终失败，返回空结果")
                    return []  # 返回空列表，避免程序崩溃
                else:
                    logger.warning(f"等待 1 秒后重试...")
                    await asyncio.sleep(1)
                    continue

        # 创建搜索请求
        dense_req = AnnSearchRequest(
            data=[dense_vector],
            anns_field="dense_vector",
            param={"metric_type": "COSINE", "params": {"ef": 64}},
            limit=top_k * 7,  # 语义召回70条
            expr=filter_expr
        )

        sparse_req = AnnSearchRequest(
            data=[query],
            anns_field="sparse_bm25",
            param={"metric_type": "BM25"},
            limit=top_k * 3,  # bm25召回30条
            expr=filter_expr
        )
        # 异步混合检索
        try:
            results = await self.client.hybrid_search(
                collection_name=self.collection_name,
                reqs=[dense_req, sparse_req],
                ranker=RRFRanker(),
                limit=top_k * 3,
                output_fields=["parent_id"]
            )
            """
            results的结构如下：
            data: [[
            {'id': 465176452476909908, 'distance': 0.032258063554763794,'entity': {'parent_id': 'aed97ca4-aae4-41a8-9fbc-a912ec87c7fa'}},
            {'id': 465176452476909873, 'distance': 0.03154495730996132, 'entity': {'parent_id': '7d8c5d53-1b74-4bbf-ae37-fcadf6c9b67d'}}, 
            {'id': 465176452476909847, 'distance': 0.016393441706895828, 'entity': {'parent_id': '9becd140-9a4d-43b9-b0ab-7152cc19eacd'}}]],
            {'cost': 6}
            """
            # 返回去重后的父块ID列表
            return list(set([hit["entity"]["parent_id"] for hit in results[0]]))
        except Exception as e:
            logger.error(f"混合检索失败: {e},降级为稠密向量检索")
            
            results = await self.client.search(
                collection_name=self.collection_name,
                data=[dense_vector],
                anns_field="dense_vector",
                param={"metric_type": "COSINE", "params": {"ef": 64}},
                limit=top_k,
                filter=filter_expr,
                output_fields=["parent_id"]
            )
            # 处理结果
            parent_ids = set()

            if results and len(results) > 0:
                for hit in results[0]:
                    entity = hit["entity"]
                    parent_id = entity.get("parent_id")
                    parent_ids.add(parent_id)
            return list(parent_ids)

        

    async def delete_file_by_hash(self, knowledge_base_id: str, file_hash: str):
        """根据文件哈希删除整个文件的所有子块"""
        # 移除 _ensure_initialized 调用，因为启动时已经初始化

        # 构造过滤条件
        filter_expr = f'knowledge_base_id == "{knowledge_base_id}" and metadata["file_hash"] == "{file_hash}"'

        # 异步查询要删除的数据
        results = await self.client.query(
            collection_name=self.collection_name,
            filter=filter_expr,
            output_fields=["chunk_id", "metadata"]
        )

        if results:
            # 收集所有子块哈希
            chunk_hashes = []
            for result in results:
                metadata = result.get("metadata", {})
                chunk_hash = metadata.get("child_chunk_hash")
                if chunk_hash:
                    chunk_hashes.append(chunk_hash)

            # 异步删除数据
            await self.client.delete(
                collection_name=self.collection_name,
                filter=filter_expr
            )

            # 批量删除块哈希
            if chunk_hashes and self.hash_storage:
                await self.hash_storage.remove_chunk_hashes_batch(chunk_hashes)
                logger.info(f"已删除 {len(chunk_hashes)} 个块哈希记录")

            # 从哈希存储中移除文件哈希
            if self.hash_storage:
                await self.hash_storage.remove_file_hash(file_hash)

            logger.info(f"已删除文件哈希 {file_hash} 的 {len(results)} 个子块")
            return len(results)
        else:
            logger.warning(f"未找到文件哈希 {file_hash} 的数据")
            return 0

    async def close(self):
        """关闭客户端连接"""
        if self.client:
            await self.client.close()
            logger.info("Milvus 客户端已关闭")


_global_milvus_client = None

async def get_milvus_client(hash_storage: Optional["HashStorage"] = None):
    """获取全局 Milvus 客户端实例"""
    global _global_milvus_client
    if _global_milvus_client is None:
        _global_milvus_client = AsyncMilvusClientWrapper(hash_storage)
        # 启动时立即初始化集合，而不是等到第一次请求
        await _global_milvus_client.ensure_collection()
    return _global_milvus_client