import os
import asyncio
from dotenv import load_dotenv
from typing import List, Dict, Any
from langchain_core.documents import Document
from langchain_community.embeddings import DashScopeEmbeddings
from pymilvus import AsyncMilvusClient, DataType, Function, FunctionType
from pymilvus import AnnSearchRequest, RRFRanker
import logging

logger = logging.getLogger(__name__)

# 加载环境变量
load_dotenv()


class KnowledgeBaseManager:
    """知识库管理器 - 负责知识库的初始化和数据操作"""

    def __init__(self, client: AsyncMilvusClient, embeddings: DashScopeEmbeddings,
                 collection_name: str, dense_dim: int):
        self.client = client
        self.embeddings = embeddings
        self.collection_name = collection_name
        self.dense_dim = dense_dim

    async def init_collection(self):
        """初始化知识库集合，配置 BM25 内置函数"""
        # 检查集合是否已存在
        if await self.client.has_collection(self.collection_name):
            logger.info(f"集合 {self.collection_name} 已存在")
            # 加载集合
            await self.client.load_collection(self.collection_name)
            return

        # 创建 Schema
        schema = self.client.create_schema(enable_dynamic_field=True)

        schema.add_field(
            field_name="user_id", datatype=DataType.INT64, is_partition_key=True
        )  # 将 user_id 作为分区密钥
        schema.add_field(
            field_name="file_hash", datatype=DataType.VARCHAR, max_length=64
        )
        # 主键字段
        schema.add_field(
            field_name="id", datatype=DataType.INT64, is_primary=True, auto_id=True
        )
        schema.add_field(
            field_name="parent_id", datatype=DataType.VARCHAR, max_length=128
        )

        schema.add_field(
            field_name="knowledge_base_id", datatype=DataType.VARCHAR, max_length=128
        )

        schema.add_field(
            field_name="text",
            datatype=DataType.VARCHAR,
            max_length=65535,
            enable_analyzer=True,  # 启用分词
            analyzer_params={"type": "chinese"},
        )

        schema.add_field(field_name="metadata", datatype=DataType.JSON)

        schema.add_field(
            field_name="dense_vector",
            datatype=DataType.FLOAT_VECTOR,
            dim=self.dense_dim,
        )

        schema.add_field(
            field_name="sparse_bm25", datatype=DataType.SPARSE_FLOAT_VECTOR
        )

        # 添加 BM25 内置函数
        bm25_function = Function(
            name="bm25_func",
            function_type=FunctionType.BM25,
            input_field_names=["text"],
            output_field_names=["sparse_bm25"],
        )
        schema.add_function(bm25_function)

        # 配置索引
        index_params = self.client.prepare_index_params()

        index_params.add_index(
            field_name="dense_vector",
            index_name="dense_index",
            index_type="HNSW",
            metric_type="COSINE",
            params={"M": 16, "efConstruction": 200},
        )

        index_params.add_index(
            field_name="sparse_bm25",
            index_name="sparse_bm25_index",
            index_type="SPARSE_INVERTED_INDEX",
            metric_type="BM25",
        )

        index_params.add_index(
            field_name="user_id",
            index_name="user_id_idx",
            index_type="STL_SORT",
        )

        index_params.add_index(
            field_name="knowledge_base_id",
            index_name="kb_id_idx",
            index_type="STL_SORT",
        )

        index_params.add_index(
            field_name="file_hash",
            index_name="file_hash_idx",
            index_type="STL_SORT",
        )

        # 异步创建集合
        await self.client.create_collection(
            collection_name=self.collection_name,
            schema=schema,
            index_params=index_params,
            properties={"partitionkey.isolation": True},
        )
        # 创建集合后，加载集合
        await self.client.load_collection(self.collection_name)

        logger.info(f"集合 {self.collection_name} 创建成功")

    async def add_chunks_batch(
        self, knowledge_base_id: str, chunks: List[Document], user_id: int
    ):
        """批量添加文档块到知识库"""
        if not chunks:
            return

        texts = []
        data = []

        for chunk in chunks:
            if "parent_id" not in chunk.metadata:
                raise ValueError(f"子块缺少 parent_id 元数据")

            texts.append(chunk.page_content)

            stored_metadata = {
                k: v
                for k, v in chunk.metadata.items()
                if k not in ["parent_id", "file_hash", "knowledge_base_id", "user_id"]
            }

            data.append(
                {
                    "parent_id": chunk.metadata["parent_id"],
                    "knowledge_base_id": knowledge_base_id,
                    "text": chunk.page_content,
                    "metadata": stored_metadata,
                    "dense_vector": None,
                    "user_id": user_id,
                    "file_hash": chunk.metadata.get("file_hash", ""),
                }
            )

        # 批量向量化
        dense_vectors = await self.embeddings.aembed_documents(texts)

        # 填充向量
        for i in range(len(data)):
            data[i]["dense_vector"] = dense_vectors[i]

        result = await self.client.insert(
            collection_name=self.collection_name, data=data
        )

        logger.info(
            f"已添加 {len(data)} 个子块到 Milvus 知识库：【{knowledge_base_id}】"
        )
        return result

    async def get_dense_vector(self, query: str) -> List[float]:
        """生成稠密向量，带重试机制"""
        max_retries = 3
        dense_vector = []

        for attempt in range(max_retries):
            try:
                dense_vector = await self.embeddings.aembed_query(query)
                break
            except Exception as e:
                logger.error(
                    f"Embedding API 调用失败 (尝试 {attempt + 1}/{max_retries}): {e}"
                )
                if attempt == max_retries - 1:
                    logger.error(f"Embedding API 最终失败，返回空结果")
                    return []
                else:
                    logger.warning(f"等待 1 秒后重试...")
                    await asyncio.sleep(1)
                    continue

        return dense_vector

    async def hybrid_retrieval_knowledge_base(
        self, query: str, knowledge_base_id: str, user_id: int, top_k: int = 5
    ) -> List[str]:
        """混合检索，返回去重后的父块 ID 列表，混合检索失败时自动降级为稠密向量检索"""
        
        # 构建过滤表达式
        if knowledge_base_id == "默认知识库":
            filter_expr = f"user_id == {user_id}"
        else:
            filter_expr = f'knowledge_base_id == "{knowledge_base_id}" and user_id == {user_id}'
        # 生成稠密向量，带重试机制
        dense_vector = await self.get_dense_vector(query)
        if not dense_vector:
            return []

        # 创建搜索请求
        dense_req = AnnSearchRequest(
            data=[dense_vector],
            anns_field="dense_vector",
            param={"metric_type": "COSINE", "params": {"ef": 64}},
            limit=top_k * 7,
            expr=filter_expr,
        )

        sparse_req = AnnSearchRequest(
            data=[query],
            anns_field="sparse_bm25",
            param={"metric_type": "BM25"},
            limit=top_k * 3,
            expr=filter_expr,
        )

        # 异步混合检索
        try:
            results = await self.client.hybrid_search(
                collection_name=self.collection_name,
                reqs=[dense_req, sparse_req],
                ranker=RRFRanker(),
                limit=top_k * 3,
                output_fields=["parent_id"],
            )
            # 返回去重后的父块 ID 列表
            return list(set([hit["entity"]["parent_id"] for hit in results[0]]))
        except Exception as e:
            logger.error(f"混合检索失败：{e},降级为稠密向量检索")

            results = await self.client.search(
                collection_name=self.collection_name,
                data=[dense_vector],
                anns_field="dense_vector",
                search_params={"metric_type": "COSINE", "params": {"ef": 64}},
                limit=top_k,
                filter=filter_expr,
                output_fields=["parent_id"],
            )

            parent_ids = set()
            if results and len(results) > 0:
                for hit in results[0]:
                    entity = hit["entity"]
                    parent_id = entity.get("parent_id")
                    parent_ids.add(parent_id)
            return list(parent_ids)

    async def delete_knowledge_file_chunks(
        self, knowledge_base_id: str, user_id: int
    ) -> int:
        """
        删除知识库中的所有文件的所有子块
        返回删除的子块数量
        """
        try:
            filter_condition = (
                f'knowledge_base_id == "{knowledge_base_id}" and user_id == {user_id}'
            )
            result = await self.client.delete(
                collection_name=self.collection_name, filter=filter_condition
            )
            print(f"删除知识库的结果：{result}")
            deleted_count = (
                result["delete_count"] if hasattr(result, "delete_count") else 0
            )
            logger.info(
                f"从 Milvus 删除知识库 {knowledge_base_id}（用户 {user_id}）的所有文件子块"
            )
            return deleted_count
        except Exception as e:
            logger.error(
                f"从 Milvus 删除知识库 {knowledge_base_id} 的所有文件子块失败：{e}"
            )
            raise
            
    async def delete_file_chunks(
        self, knowledge_base_id: str, file_hash: str, user_id: int
    ) -> int:
        """删除指定文件的所有子块（只负责 Milvus 数据删除）"""
        try:
            filter = f'knowledge_base_id == "{knowledge_base_id}" and file_hash == "{file_hash}" and user_id == {user_id}'

            result = await self.client.delete(
                collection_name=self.collection_name, filter=filter
            )
            logger.info(
                f"从 Milvus 删除文件 {file_hash[:16]} 的子块，影响数量：{result['delete_count']}"
            )
            deleted_count = (
                result["delete_count"] if hasattr(result, "delete_count") else 0
            )
            return deleted_count

        except Exception as e:
            logger.error(f"从 Milvus 删除文件失败：{e}")
            raise
