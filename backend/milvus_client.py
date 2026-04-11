import os
import asyncio
from dotenv import load_dotenv
from typing import List, Optional, Tuple, Dict, Any
from langchain_core.documents import Document
from langchain_community.embeddings import DashScopeEmbeddings
from pymilvus import AsyncMilvusClient, DataType, Function, FunctionType
from pymilvus import AnnSearchRequest, RRFRanker
import logging
import time


logger = logging.getLogger(__name__)

# 加载环境变量
load_dotenv()

_global_milvus_client = None


async def get_milvus_client():
    """获取全局 Milvus 客户端实例"""
    global _global_milvus_client
    if _global_milvus_client is None:
        _global_milvus_client = AsyncMilvusClientWrapper()
        # 启动时立即初始化集合，而不是等到第一次请求
        await _global_milvus_client.ensure_collection()
    return _global_milvus_client


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

        self.knowledge_base_collection = os.getenv("knowledge_base_collection")
        self.memory_collection = os.getenv("memory_collection")

        self.embeddings = DashScopeEmbeddings(
            model=os.getenv("EMBEDDING_MODEL"),
            dashscope_api_key=os.getenv("DASHSCOPE_API_KEY"),
        )
        self.client = AsyncMilvusClient(
            # uri=os.getenv("Milvus_url"),
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
            await self._init_konwledge_base_collection()
            await self._init_memory_collection()
            self._collection_initialized = True

    async def _init_memory_collection(self):
        """初始化对话记忆集合（独立于知识库）"""
        collection_name = self.memory_collection

        if await self.client.has_collection(collection_name):
            logger.info(f"对话记忆集合 {collection_name} 已存在")
            # 加载集合
            await self.client.load_collection(collection_name)
            return

        # 创建 Schema（专为对话记忆设计）
        schema = self.client.create_schema(
            enable_dynamic_field=True
        )  # 记忆建议开，方便扩展

        """
        id
        user_id
        thread_id
        memory_type
        content
        summary_id
        importance
        created_at
        last_access_at
        vector#稠密向量
        sparse_vector#bm25向量
        """
        # 主键
        schema.add_field(
            field_name="id",
            datatype=DataType.VARCHAR,
            max_length=50,
            is_primary=True,
            auto_id=True,
        )

        # 用户标识
        schema.add_field(
            field_name="user_id", datatype=DataType.INT64, is_partition_key=True
        )  # 将user_id作为分区密钥

        # 会话标识
        schema.add_field(
            field_name="thread_id", datatype=DataType.VARCHAR, max_length=100
        )

        # 记忆类型: preference, fact, topic, task, repetitive
        schema.add_field(
            field_name="memory_type", datatype=DataType.VARCHAR, max_length=30
        )

        # 记忆内容
        schema.add_field(
            field_name="content",
            datatype=DataType.VARCHAR,
            max_length=65535,
            enable_analyzer=True,  # 启用分词
            analyzer_params={"type": "chinese"},
        )

        # 关联的 PostgreSQL summary_id(可选，其他两种类型默认为0)
        schema.add_field(
            field_name="summary_id",
            datatype=DataType.VARCHAR,
            max_length=50,
            nullable=True,  # 非摘要记忆允许为空
        )

        # 重要性评分（0-1之间，1表示重要性最高）
        schema.add_field(field_name="importance", datatype=DataType.FLOAT)

        # 时间戳
        schema.add_field(field_name="created_at", datatype=DataType.INT64)

        schema.add_field(
            field_name="last_access_at",
            datatype=DataType.INT64,
            default_value=0,  # 最后访问时间，LRU 淘汰
        )

        # 向量字段
        schema.add_field(
            field_name="vector", datatype=DataType.FLOAT_VECTOR, dim=self.dense_dim
        )

        # BM25 稀疏向量（记忆必须关键词检索）
        schema.add_field(
            field_name="sparse_vector", datatype=DataType.SPARSE_FLOAT_VECTOR
        )

        # 配置 BM25 内置函数
        bm25_function = Function(
            name="bm25_func",
            function_type=FunctionType.BM25,
            input_field_names=["content"],  # 对 content 字段进行分词
            output_field_names=["sparse_vector"],
        )
        schema.add_function(bm25_function)

        # 配置索引
        index_params = self.client.prepare_index_params()
        # 向量索引
        index_params.add_index(
            field_name="vector",
            index_name="vector_index",
            index_type="HNSW",
            metric_type="COSINE",
            params={"M": 16, "efConstruction": 200},
        )

        # BM25 索引
        index_params.add_index(
            field_name="sparse_vector",
            index_name="sparse_index",
            index_type="SPARSE_INVERTED_INDEX",
            metric_type="BM25",
        )

        # 记忆集合需要添加的标量索引
        # 高优先级（必须）
        index_params.add_index(field_name="user_id", index_type="STL_SORT")  # 数据隔离
        index_params.add_index(
            field_name="thread_id", index_type="STL_SORT"
        )  # 会话过滤
        index_params.add_index(
            field_name="memory_type", index_type="STL_SORT"
        )  # 类型过滤

        # 中优先级（强烈建议）
        index_params.add_index(
            field_name="importance", index_type="STL_SORT"
        )  # 重要性排序
        index_params.add_index(
            field_name="last_access_at", index_type="STL_SORT"
        )  # LRU淘汰
        index_params.add_index(
            field_name="created_at", index_type="STL_SORT"
        )  # 时间过滤
        # 创建集合
        await self.client.create_collection(
            collection_name=collection_name,
            schema=schema,
            index_params=index_params,
            properties={"partitionkey.isolation": True},  # 启用分区密钥隔离功能
        )
        # 创建集合后，加载集合
        await self.client.load_collection(collection_name)

    async def _init_konwledge_base_collection(self):
        """异步初始化集合，配置 BM25 内置函数"""
        # 检查集合是否已存在
        if await self.client.has_collection(self.knowledge_base_collection):
            logger.info(f"集合 {self.knowledge_base_collection} 已存在")
            # 加载集合
            await self.client.load_collection(self.knowledge_base_collection)
            return

        # 创建 Schema
        schema = self.client.create_schema(enable_dynamic_field=True)

        schema.add_field(
            field_name="user_id", datatype=DataType.INT64, is_partition_key=True
        )  # 将user_id作为分区密钥
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

        # 在 _init_konwledge_base_collection 方法中，添加索引配置
        index_params.add_index(
            field_name="user_id",
            index_name="user_id_idx",
            index_type="STL_SORT",  # 整型字段，适合排序和等值查询
        )

        index_params.add_index(
            field_name="knowledge_base_id",
            index_name="kb_id_idx",
            index_type="STL_SORT",  # 字符串字段，几乎每次查询都用到
        )

        index_params.add_index(
            field_name="file_hash",
            index_name="file_hash_idx",
            index_type="STL_SORT",  # 用于删除文件和去重
        )

        # 异步创建集合
        await self.client.create_collection(
            collection_name=self.knowledge_base_collection,
            schema=schema,
            index_params=index_params,
            properties={"partitionkey.isolation": True},  # 启用分区密钥隔离功能
        )
        # 创建集合后，加载集合
        await self.client.load_collection(self.knowledge_base_collection)

        logger.info(f"集合 {self.knowledge_base_collection} 创建成功")

    async def add_chunks_batch(
        self, knowledge_base_id: str, chunks: List[Document], user_id: int
    ):
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
                    "dense_vector": None,  # 占位
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
            collection_name=self.knowledge_base_collection, data=data
        )

        logger.info(
            f"已添加 {len(data)} 个子块到 Milvus 知识库:【{knowledge_base_id}】"
        )
        return result

    async def get_dense_vector(
        self,
        query: str,
    ) -> List[float]:
        """生成稠密向量，带重试机制"""
        max_retries = 3
        for attempt in range(max_retries):
            try:
                dense_vector = await self.embeddings.aembed_query(query)
                break  # 成功则跳出循环
            except Exception as e:
                logger.error(
                    f"Embedding API 调用失败 (尝试 {attempt + 1}/{max_retries}): {e}"
                )
                if attempt == max_retries - 1:  # 最后一次尝试失败
                    logger.error(f"Embedding API 最终失败，返回空结果")
                    return []  # 返回空列表，避免程序崩溃
                else:
                    logger.warning(f"等待 1 秒后重试...")
                    await asyncio.sleep(1)
                    continue
        return dense_vector

    async def hybrid_retrieval_memories(
        self,
        query: str,
        user_id: int,
        summary_k: int = 1,
        semantic_k: int = 3,
        episodic_k: int = 2,
        procedural_k: int = 2,
    ):
        """混合检索，每种记忆类型独立检索、去重，返回召回的相似记忆内容"""

        # 定义各类型记忆的参数配置
        memory_configs = {
            "summary": {
                "k": summary_k,
                "filter": f"user_id == {user_id} and memory_type == 'summary'",
            },
            "semantic": {
                "k": semantic_k,
                "filter": f"user_id == {user_id} and memory_type == 'semantic'",
            },
            "episodic": {
                "k": episodic_k,
                "filter": f"user_id == {user_id} and memory_type == 'episodic'",
            },
            "procedural": {
                "k": procedural_k,
                "filter": f"user_id == {user_id} and memory_type == 'procedural'",
            },
        }

        # 生成问题稠密向量，带重试机制
        dense_vector = await self.get_dense_vector(query)
        if not dense_vector:
            return {}

        # 为每种记忆类型创建混合检索请求
        async def search_memory_type(memory_type: str, config: dict):
            """对单个记忆类型进行混合检索、去重"""
            k = config["k"]
            filter_expr = config["filter"]

            try:
                # 创建搜索请求
                dense_req = AnnSearchRequest(
                    data=[dense_vector],
                    anns_field="vector",
                    param={"metric_type": "COSINE", "params": {"ef": 64}},
                    limit=k * 6,  # 语义召回：k值的6倍
                    expr=filter_expr,
                )

                sparse_req = AnnSearchRequest(
                    data=[query],
                    anns_field="sparse_vector",
                    param={"metric_type": "BM25"},
                    limit=k * 4,  # BM25召回：k值的4倍
                    expr=filter_expr,
                )

                # 执行混合检索
                results = await self.client.hybrid_search(
                    collection_name=self.memory_collection,
                    reqs=[dense_req, sparse_req],
                    ranker=RRFRanker(),
                    limit=k * 3,  # RRF返回：k值的3倍
                    output_fields=[
                        "id",
                        "memory_type",
                        "content",
                        "summary_id",
                        "importance",
                        "last_access_at",
                    ],
                )

                # 处理结果：去重（基于id）
                memories = []
                seen_ids = set()

                if results and len(results) > 0:
                    for hit in results[0]:
                        entity = hit["entity"]
                        memory_id = entity.get("id")

                        # 基于id去重
                        if memory_id and memory_id not in seen_ids:
                            seen_ids.add(memory_id)
                            memories.append(
                                {
                                    "memory_type": entity.get("memory_type"),
                                    "content": entity.get("content"),
                                    "summary_id": entity.get("summary_id"),
                                    "importance": entity.get("importance"),
                                    "last_access_at": entity.get("last_access_at"),
                                    "score": hit["distance"],  # RRF分数
                                }
                            )

                # 取前2k条返回
                return {memory_type: memories[: k * 2]}

            except Exception as e:
                logger.error(
                    f"记忆类型 {memory_type} 混合检索失败: {e}, 降级为稠密向量检索"
                )

                # 降级为稠密向量检索
                try:
                    results = await self.client.search(
                        collection_name=self.memory_collection,
                        data=[dense_vector],
                        anns_field="vector",
                        search_params={"metric_type": "COSINE", "params": {"ef": 64}},
                        limit=k * 6,
                        filter=filter_expr,
                        output_fields=[
                            "id",
                            "memory_type",
                            "content",
                            "summary_id",
                            "importance",
                            "last_access_at",
                        ],
                    )

                    # 处理结果
                    memories = []
                    if results and len(results) > 0:
                        for hit in results[0]:
                            entity = hit["entity"]
                            memories.append(
                                {
                                    "memory_type": entity.get("memory_type"),
                                    "content": entity.get("content"),
                                    "summary_id": entity.get("summary_id"),
                                    "importance": entity.get("importance"),
                                    "last_access_at": entity.get("last_access_at"),
                                    "score": hit["distance"],
                                }
                            )

                    # 取前2k条返回
                    return {memory_type: memories[: k * 2]}

                except Exception as search_e:
                    logger.error(f"记忆类型 {memory_type} 稠密检索也失败: {search_e}")
                    return {memory_type: []}

        # 并行执行所有记忆类型的检索
        tasks = [
            search_memory_type(mem_type, config)
            for mem_type, config in memory_configs.items()
        ]

        all_results = await asyncio.gather(*tasks, return_exceptions=True)

        # 合并结果，按记忆类型组织
        final_results = {
            "summary": [],
            "semantic": [],
            "episodic": [],
            "procedural": [],
        }

        for result in all_results:
            if isinstance(result, Exception):
                logger.error(f"并行检索任务失败: {result}")
                continue

            if result and isinstance(result, dict):
                # 合并到最终结果
                for mem_type, memories in result.items():
                    if mem_type in final_results:
                        final_results[mem_type].extend(memories)

        top_k_memories = self.get_the_top_k_memories(
            memory_dict=final_results,
            memory_configs=memory_configs,
        )

        return top_k_memories
        """
        返回的结构类似下面这样
        {
        "summary": [
            {
                "memory_type": "summary",
                "content": "用户喜欢喝美式咖啡，不加糖",
                "summary_id": "sum_001",或者None
                "last_access_at": 1734019200.0,  # 时间戳
            },
        ],
        "semantic": [],
        "episodic": [],
        "procedural": [],
        }
        """

    def get_the_top_k_memories(
        self,
        memory_dict: Dict[str, List[Dict]],
        alpha: float = 0.45,  # 语义相关性权重
        beta: float = 0.25,  # 时间新鲜度权重
        gamma: float = 0.3,  # 重要性权重
        memory_configs: Dict[str, Dict] = None,  # 记忆类型配置
        type_weights: Dict[str, float] = None,  # 类型权重
    ) -> Dict[str, List[Dict]]:
        """
        从记忆字典中提取每个记忆类型的前k条记录

        参数塑造的AI效果：
        这组参数塑造了一个专注但灵活的学习者：
        它能牢牢记住核心人格（gamma=0.3），精准理解当前问题（alpha=0.45），
        同时具备不错的记忆能力以跟上对话节奏（beta=0.25）。
        """
        # 默认类型权重（基于记忆的长期价值）
        if type_weights is None:
            type_weights = {
                "summary": 0.7,  # 摘要记忆：信息已压缩，降权
                "semantic": 1.3,  # 语义记忆：核心知识，提权
                "episodic": 1.0,  # 情景记忆：标准权重
                "procedural": 1.2,  # 程序记忆：技能习惯，较高权重
            }

        current_time = time.time()
        DECAY_RATE = 0.995  # 时间衰减率

        result = {}

        for mem_type, memories in memory_dict.items():
            if not memories:
                result[mem_type] = []
                continue

            # 获取当前类型的权重
            type_weight = type_weights.get(mem_type, 1.0)

            scored_memories = []

            for mem in memories:
                """
                每个mem都是一个dict,包含以下字段：
                "memory_type": entity.get("memory_type"),
                "content": entity.get("content"),
                "summary_id": entity.get("summary_id"),
                "importance": entity.get("importance"),
                "last_access_at": entity.get("last_access_at"),
                "score": hit["distance"],
                """
                # 1. 语义相关性（直接从score字段提取）
                semantic_score = mem.get("score", 0.5)

                # 2. 时间新鲜度（基于最后访问时间）
                last_access = mem.get("last_access_at", current_time)
                hours_passed = (current_time - last_access) / 3600
                recency_score = DECAY_RATE**hours_passed

                # 3. 重要性评分
                importance_score = mem.get("importance", 0.5)

                # 4. 综合评分（乘以类型权重）
                final_score = (
                    alpha * semantic_score
                    + beta * recency_score
                    + gamma * importance_score
                ) * type_weight  # 关键：类型权重作为乘数

                scored_memories.append(
                    (mem, final_score)
                )  # [(记忆内容,综合评分),(记忆内容,综合评分),...]

            # 按综合评分排序，取Top K
            scored_memories.sort(key=lambda x: x[1], reverse=True)  # x[1]是综合评分
            # 每种类型记忆取出scored_memories中前top_k_per_type条mem
            top_k = memory_configs[mem_type]["k"]

            # 改造：只保留 memory_type, content, last_access_at 三个字段
            result[mem_type] = [
                {
                    "memory_type": mem.get("memory_type"),
                    "content": mem.get("content"),
                    "last_access_at": mem.get("last_access_at"),
                    "summary_id": mem.get("summary_id"),
                }
                for mem, _ in scored_memories[:top_k]
            ]

        return result

    async def hybrid_retrieval_knowledge_base(
        self, query: str, knowledge_base_id: str, user_id: int, top_k: int = 5
    ):
        """混合检索，返回去重后的父块ID列表，混合检索失败时自动降级为稠密向量检索（即语义搜索）"""

        # 构建过滤表达式
        if knowledge_base_id == "默认知识库":
            filter_expr = (
                f"user_id == {user_id}"  # 不添加过滤条件，搜索该用户的全部知识库
            )
        else:
            # 构建过滤表达式，限制在指定知识库内搜索
            filter_expr = (
                f'knowledge_base_id == "{knowledge_base_id}" and user_id == {user_id}'
            )

        # 生成稠密向量，带重试机制
        dense_vector = await self.get_dense_vector(query)
        if not dense_vector:
            return []

        # 创建搜索请求
        dense_req = AnnSearchRequest(
            data=[dense_vector],
            anns_field="dense_vector",
            param={"metric_type": "COSINE", "params": {"ef": 64}},
            limit=top_k * 7,  # 语义召回70条
            expr=filter_expr,
        )

        sparse_req = AnnSearchRequest(
            data=[query],
            anns_field="sparse_bm25",
            param={"metric_type": "BM25"},
            limit=top_k * 3,  # bm25召回30条
            expr=filter_expr,
        )
        # 异步混合检索
        try:
            results = await self.client.hybrid_search(
                collection_name=self.knowledge_base_collection,
                reqs=[dense_req, sparse_req],
                ranker=RRFRanker(),
                limit=top_k * 3,
                output_fields=["parent_id"],
            )
            """
            混合搜索results的结构如下：
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
                collection_name=self.knowledge_base_collection,
                data=[dense_vector],
                anns_field="dense_vector",
                search_params={"metric_type": "COSINE", "params": {"ef": 64}},
                limit=top_k,
                filter=filter_expr,
                output_fields=["parent_id"],
            )
            # 处理结果
            parent_ids = set()
            """
            稠密向量检索results的结构如下：
            data: [[{'id': 465201081576042423, 'distance': 0.7446649074554443, 'entity': {'parent_id': 'a85ecfe0-5958-4bc2-a516-a89a6e0b21be'}},
            {'id': 465201081576042424, 'distance': 0.6389918923377991, 'entity': {'parent_id': 'a85ecfe0-5958-4bc2-a516-a89a6e0b21be'}},
            {'id': 465201081576042425, 'distance': 0.452325701713562, 'entity': {'parent_id': 'a85ecfe0-5958-4bc2-a516-a89a6e0b21be'}}, 
            {'id': 465201081576042426, 'distance': 0.44345560669898987, 'entity': {'parent_id': 'a85ecfe0-5958-4bc2-a516-a89a6e0b21be'}},
            {'id': 465201081576042427, 'distance': 0.433427631855011, 'entity': {'parent_id': 'a85ecfe0-5958-4bc2-a516-a89a6e0b21be'}}]],
            {'cost': 6}
            """
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
            # 根据 knowledge_base_id 和 user_id 删除所有子块
            filter_condition = (
                f'knowledge_base_id == "{knowledge_base_id}" and user_id == {user_id}'
            )
            result = await self.client.delete(
                collection_name=self.knowledge_base_collection, filter=filter_condition
            )
            print(f"删除知识库的结果: {result}")
            # 注意：Milvus 的 delete 操作返回的 result 中包含 delete_count
            deleted_count = (
                result["delete_count"] if hasattr(result, "delete_count") else 0
            )
            logger.info(
                f"从 Milvus 删除知识库 {knowledge_base_id}（用户 {user_id}）的所有文件子块"
            )
            return deleted_count
        except Exception as e:
            logger.error(
                f"从 Milvus 删除知识库 {knowledge_base_id} 的所有文件子块失败: {e}"
            )
            raise

    async def delete_flie_chunks(
        self, knowledge_base_id: str, file_hash: str, user_id: int
    ) -> int:
        """删除指定文件的所有子块（只负责 Milvus 数据删除）"""
        try:
            # 构建删除表达式，包含 user_id 确保隔离
            filter = f'knowledge_base_id == "{knowledge_base_id}" and file_hash == "{file_hash}" and user_id == {user_id}'

            # 执行删除
            result = await self.client.delete(
                collection_name=self.knowledge_base_collection, filter=filter
            )
            logger.info(
                f"从 Milvus 删除文件 {file_hash[:16]} 的子块，影响数量: {result['delete_count']}"
            )
            deleted_count = (
                result["delete_count"] if hasattr(result, "delete_count") else 0
            )
            return deleted_count

        except Exception as e:
            logger.error(f"从 Milvus 删除文件失败: {e}")
            raise

    async def close(self):
        """关闭客户端连接"""
        if self.client:
            await self.client.close()
            logger.info("Milvus 客户端已关闭")

    # 下面是记忆相关的函数
    async def add_memories_batch(
        self,
        user_id: int,
        thread_id: str,
        memory_dict: Dict[str, Any],
        summary_id: Optional[str] = None,
        **kwargs,
    ) -> bool:
        """
        批量添加记忆到 Milvus 集合

        Args:
            user_id: 用户ID
            thread_id: 会话ID
            memory_dict: 记忆字典，格式为：
                {
                    "summary": {"content": str, "importance_score": float},
                    "semantic_memory": [{"content": str, "importance_score": float}, ...],
                    "episodic_memory": [{"content": str, "importance_score": float}, ...],
                    "procedural_memory": [{"content": str, "importance_score": float}, ...],
                }
            summary_id: 摘要ID（可选，不提供则默认为 "0"）
            **kwargs: 其他可选参数 (如 created_at, last_access_at 等)

        Returns:
            bool: 是否添加成功
        """
        await self.ensure_collection()

        created_at = kwargs.get("created_at", int(time.time()))
        last_access_at = kwargs.get("last_access_at", int(time.time()))

        texts_to_embed = []  # 用于存储所有需要嵌入的文本
        records_to_insert = []  # 用于存储所有需要插入的记录

        # 处理 summary
        summary = memory_dict.get("summary", {})
        if summary and summary.get("content", "").strip():
            content = summary["content"].strip()
            texts_to_embed.append(content)
            records_to_insert.append(
                {
                    "user_id": user_id,
                    "thread_id": thread_id,
                    "memory_type": "summary",
                    "content": content,
                    "summary_id": summary_id,
                    "importance": float(summary.get("importance_score", 0.5)),
                    "created_at": created_at,
                    "last_access_at": last_access_at,
                }
            )

        # 处理三类记忆
        type_mapping = {
            "semantic_memory": "semantic",
            "episodic_memory": "episodic",
            "procedural_memory": "procedural",
        }

        for key, memory_type in type_mapping.items():
            items = memory_dict.get(key, [])
            if not isinstance(items, list):
                continue

            for item in items:  # 遍历该类型记忆的每个记忆项
                if isinstance(item, dict):
                    content = item.get("content", "").strip()
                    if content:
                        texts_to_embed.append(content)
                        records_to_insert.append(
                            {
                                "user_id": user_id,
                                "thread_id": thread_id,
                                "memory_type": memory_type,
                                "content": content,
                                "summary_id": None,  # 非摘要记忆的summary_id为空
                                "importance": float(item.get("importance_score", 0.5)),
                                "created_at": created_at,
                                "last_access_at": last_access_at,
                            }
                        )

        if not texts_to_embed:
            logger.warning("没有有效内容需要插入")
            return False

        # 批量向量化
        vectors = await self.embeddings.aembed_documents(texts_to_embed)

        # 添加向量
        for record, vector in zip(records_to_insert, vectors):
            record["vector"] = vector

        # 一次性批量插入
        try:
            await self.client.insert(
                collection_name=self.memory_collection,
                data=records_to_insert,
            )
            logger.info(
                f"批量添加记忆完成 - 总计: {len(records_to_insert)} 条, "
                f"用户: {user_id}, 会话: {thread_id}, summary_id: {summary_id}"
            )
            return True
        except Exception as e:
            logger.error(f"批量插入记忆失败: {e}")
            raise


async def example_hybrid_retrieval_memories():
    """示例：混合检索记忆"""

    # 1. 获取 Milvus 客户端
    milvus_client = await get_milvus_client()

    # 2. 调用混合检索
    query = "我是谁"
    user_id = 1  # 用户ID

    results = await milvus_client.hybrid_retrieval_memories(
        query=query,
        user_id=user_id,
        summary_k=2,  # 摘要记忆取1条
        semantic_k=3,  # 语义记忆取3条
        episodic_k=2,  # 情景记忆取2条
        procedural_k=3,  # 程序记忆取2条
    )

    # 3. 打印结果
    print("=" * 50)
    print(f"查询: {query}")
    print(f"用户ID: {user_id}")
    print("=" * 50)

    for mem_type, memories in results.items():
        print(f"\n【{mem_type.upper()}】类型记忆 (共{len(memories)}条):")
        for i, mem in enumerate(memories, 1):
            print(f"  {i}. 内容: {mem.get('content', 'N/A')[:50]}...")
            print(f"     评分: {mem.get('score', 'N/A'):.4f}")
            print(f"     最后访问: {mem.get('last_access_at', 'N/A')}")
            if mem.get("summary_id"):
                print(f"     摘要ID: {mem['summary_id']}")
            print()


# 运行
if __name__ == "__main__":
    asyncio.run(example_hybrid_retrieval_memories())
