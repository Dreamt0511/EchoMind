import os
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
    def __init__(self, hash_storage: Optional["HashStorage"] = None):
        """
        初始化异步 Milvus 客户端
        :param hash_storage: HashStorage 实例，可选参数，不传则不使用哈希存储
        """
        self.collection_name = os.getenv("collection_name")
        self.embeddings = DashScopeEmbeddings(
            model=os.getenv("EMBEDDING_MODEL"),
            dashscope_api_key=os.getenv("DASHSCOPE_API_KEY")
        )
        
        # 创建异步客户端
        self.client = AsyncMilvusClient(
            uri=os.getenv("Milvus_url"),
            token=os.getenv("Token")
        )
        
        self.dense_dim = int(os.getenv("dense_dimension", "1024"))
        
        #hash_storage 可选参数
        self.hash_storage = hash_storage  # 可以是 None 或 HashStorage 实例
        self._initialized = False

    async def _ensure_initialized(self):
        """确保集合已初始化"""
        if not self._initialized:
            await self._init_collection()
            self._initialized = True

    async def _init_collection(self):
        """异步初始化集合，配置 BM25 内置函数"""
        
        # 检查集合是否已存在
        if await self.client.has_collection(self.collection_name):
            logger.info(f"集合 {self.collection_name} 已存在")
            return
        
        # 创建 Schema（同步方法，直接调用）
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
        """
        批量添加已切好的子块到 Milvus（使用异步批量向量化，支持分批）
        """
        await self._ensure_initialized()
        
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
        ## 使用异步批量接口向量化文本
        dense_vectors = await self.embeddings.aembed_documents(texts)
        """ 
        # 分批处理（如果 API 有批次大小限制）
        batch_size = 20  # 根据实际情况调整
        dense_vectors = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            # 使用异步批量接口
            batch_vectors = await self.embeddings.aembed_documents(batch_texts)
            dense_vectors.extend(batch_vectors)

        """
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
        """
        混合检索，返回去重后的父块ID列表
        Args:
            query: 查询文本
            knowledge_base_id: 知识库ID，为None时检索所有知识库
            top_k: 返回结果数量
        Returns:
            去重后的父块ID列表
        """
        await self._ensure_initialized()
        
        # 构建过滤表达式
        if knowledge_base_id:
            filter_expr = f'knowledge_base_id == "{knowledge_base_id}"'
        else:
            filter_expr = None
        
        # 生成稠密向量
        dense_vector = self.embeddings.embed_query(query)
        
        # 创建搜索请求
        dense_req = AnnSearchRequest(
            data=[dense_vector],
            anns_field="dense_vector",
            param={"metric_type": "COSINE", "params": {"ef": 64}},
            limit=top_k * 10,#语义检索100个
            expr=filter_expr
        )
        
        sparse_req = AnnSearchRequest(
            data=[query],
            anns_field="sparse_bm25",
            param={"metric_type": "BM25"},
            limit=top_k * 3,#BM25检索30个
            expr=filter_expr
        )
        
        # 异步混合检索
        results = await self.client.hybrid_search(
            collection_name=self.collection_name,
            reqs=[dense_req, sparse_req],
            ranker=RRFRanker(),
            limit=top_k * 3,#取前30个融合的结果
            output_fields=["parent_id"]
        )
        
        # 返回去重后的父块ID列表
        return list(set([hit["entity"]["parent_id"] for hit in results[0]]))

    async def search_dense_only(self, query: str, knowledge_base_id: Optional[str] = None, top_k: int = 30):
        """仅使用稠密向量检索（语义）"""
        await self._ensure_initialized()
        
        # 生成查询向量
        query_vector = self.embeddings.embed_query(query)
        
        # 构建过滤表达式
        filter_expr = None
        if knowledge_base_id:
            filter_expr = f'knowledge_base_id == "{knowledge_base_id}"'
        
        # 异步搜索
        results = await self.client.search(
            collection_name=self.collection_name,
            data=[query_vector],
            anns_field="dense_vector",
            param={"metric_type": "COSINE", "params": {"ef": 64}},
            limit=top_k,
            filter=filter_expr,
            output_fields=["parent_id", "chunk_id", "chunk_index", "text"]
        )
        
        # 处理结果
        parent_ids = set()
        matched_chunks = []
        
        if results and len(results) > 0:
            for hit in results[0]:
                entity = hit["entity"]
                parent_id = entity.get("parent_id")
                parent_ids.add(parent_id)
                matched_chunks.append({
                    "parent_id": parent_id,
                    "chunk_id": entity.get("chunk_id"),
                    "text": entity.get("text"),
                    "score": hit["score"],
                    "chunk_index": entity.get("chunk_index", -1)
                })
        
        return {
            "parent_ids": list(parent_ids),
            "matched_chunks": matched_chunks
        }

    async def delete_file_by_hash(self, knowledge_base_id: str, file_hash: str):
        """根据文件哈希删除整个文件的所有子块"""
        await self._ensure_initialized()
        
        # 构造过滤条件
        filter_expr = f'knowledge_base_id == "{knowledge_base_id}" and metadata["file_hash"] == "{file_hash}"'
        
        # 异步查询要删除的数据（需要获取 child_chunk_hash）
        results = await self.client.query(
            collection_name=self.collection_name,
            filter=filter_expr,
            output_fields=["chunk_id", "metadata"]  # 添加 metadata 字段
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
            if chunk_hashes:
                await self.hash_storage.remove_chunk_hashes_batch(chunk_hashes)
                logger.info(f"已删除 {len(chunk_hashes)} 个块哈希记录")
            
            # 从哈希存储中移除文件哈希
            await self.hash_storage.remove_file_hash(file_hash)
            
            logger.info(f"已删除文件哈希 {file_hash} 的 {len(results)} 个子块")
            return len(results)
        else:
            logger.warning(f"未找到文件哈希 {file_hash} 的数据")
            return 0

    async def close(self):
        """关闭客户端连接"""
        await self.client.close()
        logger.info("Milvus 客户端已关闭")
    
    async def __aenter__(self):
        await self._ensure_initialized()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()

