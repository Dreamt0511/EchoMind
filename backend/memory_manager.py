import os
import asyncio
import time
from dotenv import load_dotenv
from typing import List, Optional, Dict, Any
from langchain_community.embeddings import DashScopeEmbeddings
from pymilvus import AsyncMilvusClient, DataType, Function, FunctionType
from pymilvus import AnnSearchRequest, RRFRanker
import logging

logger = logging.getLogger(__name__)

# 加载环境变量
load_dotenv()


class MemoryManager:
    """记忆管理器 - 负责记忆的初始化、存储和检索"""

    def __init__(
        self,
        client: AsyncMilvusClient,
        embeddings: DashScopeEmbeddings,
        collection_name: str,
        dense_dim: int,
    ):
        self.client = client
        self.embeddings = embeddings
        self.collection_name = collection_name
        self.dense_dim = dense_dim

    async def init_collection(self):
        """初始化对话记忆集合"""
        if await self.client.has_collection(self.collection_name):
            logger.info(f"对话记忆集合 {self.collection_name} 已存在")
            await self.client.load_collection(self.collection_name)
            return

        # 创建 Schema
        schema = self.client.create_schema(enable_dynamic_field=True)

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
        )

        # 会话标识
        schema.add_field(
            field_name="thread_id", datatype=DataType.VARCHAR, max_length=100
        )

        # 记忆类型
        schema.add_field(
            field_name="memory_type", datatype=DataType.VARCHAR, max_length=30
        )

        # 记忆内容
        schema.add_field(
            field_name="content",
            datatype=DataType.VARCHAR,
            max_length=65535,
            enable_analyzer=True,
            analyzer_params={"type": "chinese"},
        )

        # 关联的 summary_id
        schema.add_field(
            field_name="summary_id",
            datatype=DataType.VARCHAR,
            max_length=50,
            nullable=True,
        )

        # 重要性评分
        schema.add_field(field_name="importance", datatype=DataType.FLOAT)

        # 时间戳
        schema.add_field(field_name="created_at", datatype=DataType.INT64)
        schema.add_field(
            field_name="last_access_at",
            datatype=DataType.INT64,
            default_value=0,
        )

        # 向量字段
        schema.add_field(
            field_name="vector", datatype=DataType.FLOAT_VECTOR, dim=self.dense_dim
        )

        # BM25 稀疏向量
        schema.add_field(
            field_name="sparse_vector", datatype=DataType.SPARSE_FLOAT_VECTOR
        )

        # 配置 BM25 内置函数
        bm25_function = Function(
            name="bm25_func",
            function_type=FunctionType.BM25,
            input_field_names=["content"],
            output_field_names=["sparse_vector"],
        )
        schema.add_function(bm25_function)

        # 配置索引
        index_params = self.client.prepare_index_params()

        index_params.add_index(
            field_name="vector",
            index_name="vector_index",
            index_type="HNSW",
            metric_type="COSINE",
            params={"M": 16, "efConstruction": 200},
        )

        index_params.add_index(
            field_name="sparse_vector",
            index_name="sparse_index",
            index_type="SPARSE_INVERTED_INDEX",
            metric_type="BM25",
        )

        index_params.add_index(field_name="user_id", index_type="STL_SORT")
        index_params.add_index(field_name="thread_id", index_type="STL_SORT")
        index_params.add_index(field_name="memory_type", index_type="STL_SORT")
        index_params.add_index(field_name="importance", index_type="STL_SORT")
        index_params.add_index(field_name="last_access_at", index_type="STL_SORT")
        index_params.add_index(field_name="created_at", index_type="STL_SORT")

        # 创建集合
        await self.client.create_collection(
            collection_name=self.collection_name,
            schema=schema,
            index_params=index_params,
            properties={"partitionkey.isolation": True},
        )
        await self.client.load_collection(self.collection_name)

    async def get_dense_vector(self, query: str) -> List[float]:
        """生成稠密向量，带重试机制"""
        max_retries = 3
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

    async def hybrid_retrieval_memories(
        self,
        query: str,
        user_id: int,
        summary_k: int,
        semantic_k: int,
        episodic_k: int,
        procedural_k: int,
    ) -> Dict[str, List[Dict]]:
        """混合检索，每种记忆类型并行检索、去重，返回召回的相似记忆内容"""

        memory_configs = {
            key: {"k": k, "filter": f"user_id == {user_id} and memory_type == '{key}'"}
            for key, k in {
                "summary": summary_k,
                "semantic": semantic_k,
                "episodic": episodic_k,
                "procedural": procedural_k,
            }.items()
            if k
        }

        dense_vector = await self.get_dense_vector(query)
        if not dense_vector:
            return {}

        async def search_memory_type(memory_type: str, config: dict):
            k = config["k"]
            filter_expr = config["filter"]

            try:
                dense_req = AnnSearchRequest(
                    data=[dense_vector],
                    anns_field="vector",
                    param={"metric_type": "COSINE", "params": {"ef": 64}},
                    limit=k * 6,
                    expr=filter_expr,
                )

                sparse_req = AnnSearchRequest(
                    data=[query],
                    anns_field="sparse_vector",
                    param={"metric_type": "BM25"},
                    limit=k * 4,
                    expr=filter_expr,
                )

                results = await self.client.hybrid_search(
                    collection_name=self.collection_name,
                    reqs=[dense_req, sparse_req],
                    ranker=RRFRanker(),
                    limit=k * 3,
                    output_fields=[
                        "id",
                        "memory_type",
                        "content",
                        "summary_id",
                        "importance",
                        "last_access_at",
                    ],
                )

                memories = []
                seen_ids = set()

                if results and len(results) > 0:
                    for hit in results[0]:
                        entity = hit["entity"]
                        memory_id = entity.get("id")

                        if memory_id and memory_id not in seen_ids:
                            seen_ids.add(memory_id)
                            memories.append(
                                {
                                    "id": memory_id,
                                    "memory_type": entity.get("memory_type"),
                                    "content": entity.get("content"),
                                    "summary_id": entity.get("summary_id"),
                                    "importance": entity.get("importance"),
                                    "last_access_at": entity.get("last_access_at"),
                                    "score": hit["distance"],
                                }
                            )

                return {memory_type: memories[: k * 2]}

            except Exception as e:
                logger.error(
                    f"记忆类型 {memory_type} 混合检索失败：{e}, 降级为稠密向量检索"
                )

                try:
                    results = await self.client.search(
                        collection_name=self.collection_name,
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

                    memories = []
                    if results and len(results) > 0:
                        for hit in results[0]:
                            entity = hit["entity"]
                            memories.append(
                                {
                                    "id": entity.get("id"),
                                    "memory_type": entity.get("memory_type"),
                                    "content": entity.get("content"),
                                    "summary_id": entity.get("summary_id"),
                                    "importance": entity.get("importance"),
                                    "last_access_at": entity.get("last_access_at"),
                                    "score": hit["distance"],
                                }
                            )

                    return {memory_type: memories[: k * 2]}

                except Exception as search_e:
                    logger.error(f"记忆类型 {memory_type} 稠密检索也失败：{search_e}")
                    return {memory_type: []}

        tasks = [
            search_memory_type(mem_type, config)
            for mem_type, config in memory_configs.items()
        ]

        all_results = await asyncio.gather(*tasks, return_exceptions=True)

        final_results = {
            "summary": [],
            "semantic": [],
            "episodic": [],
            "procedural": [],
        }

        for result in all_results:
            if isinstance(result, Exception):
                logger.error(f"并行检索任务失败：{result}")
                continue

            if result and isinstance(result, dict):
                for mem_type, memories in result.items():
                    if mem_type in final_results:
                        final_results[mem_type].extend(memories)

        top_k_memories = self.get_the_top_k_memories(
            memory_dict=final_results,
            memory_configs=memory_configs,
        )

        await self.update_memory_last_access_time(top_k_memories)

        return top_k_memories
        """
        返回的结构类似下面这样
        {
        "summary": 
            {
                "id": "mem_001",#这里的id是记忆的id，方便后续更新记忆的last_access_at时使用
                "memory_type": "summary",
                "content": "用户喜欢喝美式咖啡，不加糖",
                "summary_id": "sum_001",或者None
                "last_access_at": 1734019200.0,  # 时间戳
            },
        "semantic": [],
        "episodic": [],
        "procedural": [],
        }
        """

    async def update_memory_last_access_time(
        self, top_k_memories: Dict[str, List[Dict]]
    ):
        """更新记忆的最后访问时间"""
        current_timestamp = int(time.time())
        update_data = []

        for memories in top_k_memories.values():
            for mem in memories:
                if "id" in mem:
                    update_data.append(
                        {"id": mem["id"], "last_access_at": current_timestamp}
                    )

        if update_data:
            res = await self.client.upsert(
                collection_name=self.collection_name,
                data=update_data,
                partial_update=True,
            )
            print(f"更新记忆访问时间结果：{res}")

    def get_the_top_k_memories(
        self,
        memory_dict: Dict[str, List[Dict]],
        alpha: float = 0.45,
        beta: float = 0.25,
        gamma: float = 0.3,
        memory_configs: Dict[str, Dict] = None,
        type_weights: Dict[str, float] = None,
    ) -> Dict[str, List[Dict]]:
        """从记忆字典中提取每个记忆类型的前 k 条记录"""
        if type_weights is None:
            type_weights = {
                "summary": 0.7,
                "semantic": 1.3,
                "episodic": 1.0,
                "procedural": 1.2,
            }

        current_time = time.time()
        DECAY_RATE = 0.995

        result = {}

        for mem_type, memories in memory_dict.items():
            if not memories:
                result[mem_type] = []
                continue

            type_weight = type_weights.get(mem_type, 1.0)
            scored_memories = []

            for mem in memories:
                semantic_score = mem.get("score", 0.5)

                last_access = mem.get("last_access_at", current_time)
                hours_passed = (current_time - last_access) / 3600
                recency_score = DECAY_RATE**hours_passed

                importance_score = mem.get("importance", 0.5)

                final_score = (
                    alpha * semantic_score
                    + beta * recency_score
                    + gamma * importance_score
                ) * type_weight

                scored_memories.append((mem, final_score))

            scored_memories.sort(key=lambda x: x[1], reverse=True)
            top_k = memory_configs[mem_type]["k"]

            result[mem_type] = [
                {
                    "id": mem.get("id"),
                    "memory_type": mem.get("memory_type"),
                    "content": mem.get("content"),
                    "last_access_at": mem.get("last_access_at"),
                    "summary_id": mem.get("summary_id"),
                }
                for mem, _ in scored_memories[:top_k]
            ]

        return result

    async def resolve_conflicts(self, filtered_memory: dict, user_id: int) -> dict:
        """检测并删除重复记忆"""
        items = []

        for key in ["semantic_memory", "episodic_memory", "procedural_memory"]:
            for idx, mem in enumerate(filtered_memory.get(key, [])):
                items.append((key, idx, mem["content"], key.replace("_memory", "")))

        if filtered_memory.get("summary"):
            items.append(
                ("summary", None, filtered_memory["summary"]["content"], "summary")
            )

        if not items:
            return filtered_memory

        vectors = await self.embeddings.aembed_documents([item[2] for item in items])

        from collections import defaultdict

        type_to_items = defaultdict(list)
        type_to_vectors = defaultdict(list)

        for item, vec in zip(items, vectors):
            mem_type = item[3]
            type_to_items[mem_type].append(item)
            type_to_vectors[mem_type].append(vec)

        async def search_type(mem_type, type_items, type_vectors):
            filter_expr = f"user_id == {user_id} and memory_type == '{mem_type}'"
            results = await self.client.search(
                collection_name=self.collection_name,
                data=type_vectors,
                anns_field="vector",
                search_params={"metric_type": "COSINE", "params": {"ef": 64}},
                limit=1,
                filter=filter_expr,
                output_fields=["distance"],
            )
            return [
                results and results[i] and results[i][0]["distance"] >= 0.9
                for i in range(len(type_items))
            ]

        tasks = [
            search_type(t, type_to_items[t], type_to_vectors[t]) for t in type_to_items
        ]
        all_results = await asyncio.gather(*tasks)

        is_similar_list = []
        for results in all_results:
            is_similar_list.extend(results)

        to_remove = {
            "semantic_memory": [],
            "episodic_memory": [],
            "procedural_memory": [],
        }
        clear_summary = False

        for (key, idx, _, _), is_similar in zip(items, is_similar_list):
            if not is_similar:
                continue
            if key == "summary":
                clear_summary = True
            elif idx is not None:
                to_remove[key].append(idx)

        for key, indices in to_remove.items():
            if indices:
                memories = filtered_memory[key]
                for idx in sorted(indices, reverse=True):
                    if idx < len(memories):
                        del memories[idx]

        if clear_summary:
            filtered_memory["summary"] = {}

        return filtered_memory

    async def add_memories_batch(
        self,
        user_id: int,
        thread_id: str,
        memory_dict: Dict[str, Any],
        summary_id: Optional[str] = None,
        **kwargs,
    ) -> bool:
        """批量添加记忆到 Milvus 集合"""
        created_at = kwargs.get("created_at", int(time.time()))
        last_access_at = kwargs.get("last_access_at", int(time.time()))

        texts_to_embed = []
        records_to_insert = []

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

        type_mapping = {
            "semantic_memory": "semantic",
            "episodic_memory": "episodic",
            "procedural_memory": "procedural",
        }

        for key, memory_type in type_mapping.items():
            items = memory_dict.get(key, [])
            if not isinstance(items, list):
                continue

            for item in items:
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
                                "summary_id": None,
                                "importance": float(item.get("importance_score", 0.5)),
                                "created_at": created_at,
                                "last_access_at": last_access_at,
                            }
                        )

        if not texts_to_embed:
            logger.warning("没有有效内容需要插入")
            return False

        vectors = await self.embeddings.aembed_documents(texts_to_embed)

        for record, vector in zip(records_to_insert, vectors):
            record["vector"] = vector

        try:
            await self.client.insert(
                collection_name=self.collection_name,
                data=records_to_insert,
            )
            logger.info(
                f"批量添加记忆完成 - 总计：{len(records_to_insert)} 条，"
                f"用户：{user_id}, 会话：{thread_id}, summary_id: {summary_id}"
            )
            return True
        except Exception as e:
            logger.error(f"批量插入记忆失败：{e}")
            raise


async def test_update_access_time():
    """测试：更新记忆访问时间"""
    from milvus_client import get_milvus_client  # 根据实际导入路径调整

    client = AsyncMilvusClient(
        uri=os.getenv("Milvus_url"),
        token=os.getenv("Token"),
    )
    embeddings = DashScopeEmbeddings(
        model=os.getenv("EMBEDDING_MODEL"),
        dashscope_api_key=os.getenv("DASHSCOPE_API_KEY"),
    )

    memory_manager = MemoryManager(
        client=client,
        embeddings=embeddings,
        collection_name=os.getenv("memory_collection"),  # 你的集合名
        dense_dim=int(os.getenv("dense_dimension", "1024")),  # 你的向量维度
    )

    test_memory = {
        "semantic": [
            {
                "id": "465201085538812576",  # 这里的id是记忆的id
                "memory_type": "semantic",
                "content": "用户喜欢喝美式咖啡",
                "summary_id": None,
                "last_access_at": time.time(),
            },
        ]
    }

    await memory_manager.update_memory_last_access_time(test_memory)


if __name__ == "__main__":
    print("开始更新记忆访问时间")
    asyncio.run(test_update_access_time())
