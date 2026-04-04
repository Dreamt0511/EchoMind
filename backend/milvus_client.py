import os
import asyncio
from dotenv import load_dotenv
from typing import List, Optional, Tuple
from langchain_core.documents import Document
from langchain_community.embeddings import DashScopeEmbeddings
from pymilvus import AsyncMilvusClient, DataType, Function, FunctionType
from pymilvus import AnnSearchRequest, RRFRanker
import logging
logger = logging.getLogger(__name__)

# 加载环境变量
load_dotenv()


class AsyncMilvusClientWrapper:
    _instance = None
    _singleton_initialized = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        # 检查单例是否已初始化
        if self._singleton_initialized:
            return

        self.collection_name = os.getenv("collection_name")
        self.embeddings = DashScopeEmbeddings(
            model=os.getenv("EMBEDDING_MODEL"),
            dashscope_api_key=os.getenv("DASHSCOPE_API_KEY")
        )
        self.client = AsyncMilvusClient(
            #uri=os.getenv("Milvus_url"),
            uri="https://in03-bf51824a0cbc1a5.serverless.ali-cn-hangzhou.cloud.zilliz.com.cn",
            token="83929fadec379ce1d9acd2d3b1707f515fc2677410f020b564e4b6d2c4157c5311c12a77e10a3485b147f127c21e4dab19926475",
        )
        self.dense_dim = int(os.getenv("dense_dimension", "1024"))
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

        schema.add_field(
            field_name="user_id",
            datatype=DataType.INT64
        )
        schema.add_field(
            field_name="file_hash",
            datatype=DataType.VARCHAR,
            max_length=64
        )
        # 主键字段
        schema.add_field(
            field_name="id",
            datatype=DataType.INT64,
            is_primary=True,
            auto_id=True
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

    async def add_chunks_batch(self, knowledge_base_id: str, chunks: List[Document], user_id: int):
        if not chunks:
            return

        # 一次遍历准备文本和基础数据（向量占位）
        texts = []
        data = []

        for chunk in chunks:
            if "parent_id" not in chunk.metadata:
                raise ValueError(f"子块缺少 parent_id 元数据")

            texts.append(chunk.page_content)

            # 提取元数据（过滤掉已单独存储的字段）
            stored_metadata = {
                k: v for k, v in chunk.metadata.items()
                if k not in ["parent_id", "file_hash","knowledge_base_id", "user_id"]
            }

            data.append({
                "parent_id": chunk.metadata["parent_id"],
                "knowledge_base_id": knowledge_base_id,
                "text": chunk.page_content,
                "metadata": stored_metadata,
                "dense_vector": None,  # 占位
                "user_id": user_id,
                "file_hash": chunk.metadata.get("file_hash", "")
            })

        # 批量向量化
        dense_vectors = await self.embeddings.aembed_documents(texts)

        # 填充向量
        for i in range(len(data)):
            data[i]["dense_vector"] = dense_vectors[i]

        result = await self.client.insert(
            collection_name=self.collection_name,
            data=data
        )

        logger.info(f"已添加 {len(data)} 个子块到 Milvus 知识库:【{knowledge_base_id}】")
        return result

    async def hybrid_retrieval_knowledge_base(self, query: str, knowledge_base_id: str, user_id: int, top_k: int = 10):
        """混合检索，返回去重后的父块ID列表，混合检索失败时自动降级为稠密向量检索（即语义搜索）"""

        # 构建过滤表达式
        if knowledge_base_id == "默认知识库":
            filter_expr = f'user_id == {user_id}'   # 不添加过滤条件，搜索该用户的全部知识库
        else:
            # 构建过滤表达式，限制在指定知识库内搜索
            filter_expr = f'knowledge_base_id == "{knowledge_base_id}" and user_id == {user_id}'

        # 生成稠密向量，带重试机制
        max_retries = 3
        for attempt in range(max_retries):
            try:
                dense_vector = await self.embeddings.aembed_query(query)
                break  # 成功则跳出循环
            except Exception as e:
                logger.error(
                    f"Embedding API 调用失败 (尝试 {attempt + 1}/{max_retries}): {e}")
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

    async def delete_knowledge_file_chunks(self, knowledge_base_id: str, user_id: int) -> int:
        """
        删除知识库中的所有文件的所有子块
        返回删除的子块数量
        """
        try:
            # 根据 knowledge_base_id 和 user_id 删除所有子块
            filter_condition = f'knowledge_base_id == "{knowledge_base_id}" and user_id == {user_id}'
            result = await self.client.delete(
                collection_name=self.collection_name,
                filter=filter_condition
                )
            print(f"删除知识库的结果: {result}")
            # 注意：Milvus 的 delete 操作返回的 result 中包含 delete_count
            deleted_count = result["delete_count"] if hasattr(result, 'delete_count') else 0
            logger.info(f"从 Milvus 删除知识库 {knowledge_base_id}（用户 {user_id}）的所有文件子块")
            return deleted_count
        except Exception as e:
            logger.error(f"从 Milvus 删除知识库 {knowledge_base_id} 的所有文件子块失败: {e}")
            raise

    async def delete_flie_chunks(self, knowledge_base_id: str, file_hash: str, user_id: int) -> int:
        """删除指定文件的所有子块（只负责 Milvus 数据删除）"""
        try:
            # 构建删除表达式，包含 user_id 确保隔离
            filter = f'knowledge_base_id == "{knowledge_base_id}" and file_hash == "{file_hash}" and user_id == {user_id}'

            # 执行删除
            result = await self.client.delete(
                collection_name=self.collection_name,
                filter=filter
            )
            logger.info(f"从 Milvus 删除文件 {file_hash[:16]} 的子块，影响数量: {result['delete_count']}")
            deleted_count = result["delete_count"] if hasattr(result, 'delete_count') else 0
            return deleted_count

        except Exception as e:
            logger.error(f"从 Milvus 删除文件失败: {e}")
            raise

    async def close(self):
        """关闭客户端连接"""
        if self.client:
            await self.client.close()
            logger.info("Milvus 客户端已关闭")


_global_milvus_client = None


async def get_milvus_client():
    """获取全局 Milvus 客户端实例"""
    global _global_milvus_client
    if _global_milvus_client is None:
        _global_milvus_client = AsyncMilvusClientWrapper()
        # 启动时立即初始化集合，而不是等到第一次请求
        await _global_milvus_client.ensure_collection()
    return _global_milvus_client
