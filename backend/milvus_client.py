import os
from langchain_milvus import Milvus
from dotenv import load_dotenv
from typing import List, Optional, Tuple
from langchain_core.documents import Document
from langchain_community.embeddings import DashScopeEmbeddings
from pymilvus import MilvusClient as PyMilvusClient, DataType, Function, FunctionType
from pymilvus import MilvusClient as PyMilvusClient, AnnSearchRequest, RRFRanker

import logging
logger = logging.getLogger(__name__)

# 加载环境变量
load_dotenv()


class MilvusClient:
    def __init__(self):
        """初始化 Milvus 客户端"""
        self.collection_name = os.getenv("collection_name")
        self.embeddings = DashScopeEmbeddings(
            model=os.getenv("EMBEDDING_MODEL"),
            dashscope_api_key=os.getenv("DASHSCOPE_API_KEY")
        )

        """
        本地文件存储{"uri": "./milvus_demo.db"},  
        Milvus 服务器{"uri": "http://localhost:19530"}
        Zilliz Cloud{"uri": "your_zilliz_uri", "token": "your_api_key"}
        """
        self.client = PyMilvusClient(uri=os.getenv("Milvus_url"),
                                     token=os.getenv("Token")
                                     )

        self.dense_dim = int(
            os.getenv("dense_dimension", "1024"))  # 添加默认值并转换为整数

        self._init_collection()

    # 初始化集合
    def _init_collection(self):
        """初始化集合，配置 BM25 内置函数"""

        # 检查集合是否已创建
        if self.client.has_collection(self.collection_name):
            return

        # 1. 创建 Schema
        schema = self.client.create_schema(enable_dynamic_field=True)

        # 2. 添加字段
        schema.add_field(
            field_name="id",
            datatype=DataType.INT64,
            is_primary=True,
            auto_id=True
        )

        # 子块唯一标识（UUID，用于关联 PostgreSQL 中的父块）
        schema.add_field(
            field_name="chunk_id",
            datatype=DataType.VARCHAR,
            max_length=128,
            is_primary=False
        )

        # 父块 ID（关联到 PostgreSQL 中的父文档）
        schema.add_field(
            field_name="parent_id",
            datatype=DataType.VARCHAR,
            max_length=128
        )

        # 知识库ID
        schema.add_field(
            field_name="knowledge_base_id",
            datatype=DataType.VARCHAR,
            max_length=128
        )

        # 子块索引位置
        schema.add_field(
            field_name="chunk_index",
            datatype=DataType.INT32
        )

        # 文本字段 - 启用分析器以支持 BM25
        schema.add_field(
            field_name="text",
            datatype=DataType.VARCHAR,
            max_length=65535,
            enable_analyzer=True,
            analyzer_params={"type": "chinese"}
        )

        # 元数据字段（存储子块的额外信息）
        schema.add_field(
            field_name="metadata",
            datatype=DataType.JSON
        )

        # 稠密向量字段（语义检索）
        schema.add_field(
            field_name="dense_vector",
            datatype=DataType.FLOAT_VECTOR,
            dim=self.dense_dim
        )

        # 稀疏向量字段（BM25 全文检索）
        schema.add_field(
            field_name="sparse_bm25",
            datatype=DataType.SPARSE_FLOAT_VECTOR
        )

        # 3. 添加 BM25 内置函数
        bm25_function = Function(
            name="bm25_func",
            function_type=FunctionType.BM25,
            input_field_names=["text"],
            output_field_names=["sparse_bm25"]
        )
        schema.add_function(bm25_function)

        # 4. 配置索引
        index_params = self.client.prepare_index_params()

        # 稠密向量索引
        index_params.add_index(
            field_name="dense_vector",
            index_name="dense_index",
            index_type="HNSW",
            metric_type="COSINE",
            params={"M": 16, "efConstruction": 200}
        )

        # 稀疏向量索引（BM25）
        index_params.add_index(
            field_name="sparse_bm25",
            index_name="sparse_bm25_index",
            index_type="SPARSE_INVERTED_INDEX",
            metric_type="BM25"
        )

        # 5. 创建集合
        self.client.create_collection(
            collection_name=self.collection_name,
            schema=schema,
            index_params=index_params
        )

        logger.info(f"集合{self.collection_name}创建成功")

    def add_chunks_batch(self, knowledge_base_id: str, chunks_with_ids: List[Tuple[str, Document]]):
        """
        批量添加已切好的子块到 Milvus
        Args:
            knowledge_base_id: 知识库ID
            chunks_with_ids: [(chunk_id, Document), ...] 列表
        """
        if not chunks_with_ids:
            return

        # 准备插入数据
        data = []
        for chunk_id, chunk in chunks_with_ids:
            # 验证必要的元数据
            if "parent_id" not in chunk.metadata:
                raise ValueError(
                    f"子块缺少 parent_id 元数据: {chunk.page_content[:50]}...")

            chunk_index = chunk.metadata.get("chunk_index", 0)

            # 生成稠密向量
            dense_vector = self.embeddings.embed_query(chunk.page_content)

            # 准备存储的元数据（不包含 parent_id 和 chunk_index）
            stored_metadata = {
                k: v for k, v in chunk.metadata.items()
                if k not in ["parent_id", "chunk_index"]
            }

            data.append({
                "chunk_id": chunk_id,
                "parent_id": chunk.metadata["parent_id"],
                "knowledge_base_id": knowledge_base_id,
                "chunk_index": chunk_index,
                "text": chunk.page_content,
                "metadata": stored_metadata,
                "dense_vector": dense_vector
                # sparse_bm25 由 Milvus 自动生成
            })

        # 批量插入到 Milvus
        result = self.client.insert(
            collection_name=self.collection_name,
            data=data
        )

        logger.info(f"已添加 {len(data)} 个子块到 Milvus（知识库: {knowledge_base_id}）")
        return result

    def hybrid_search(self, query: str, knowledge_base_id: Optional[str] = None,
                      top_k: int = 5, return_parent_ids: bool = True):
        """
        混合检索，返回匹配的子块或父块ID列表

        Args:
            query: 查询文本
            knowledge_base_id: 知识库ID（可选）
            top_k: 返回结果数量
            return_parent_ids: 是否返回去重后的父块ID列表（True）还是返回子块详情（False）

        Returns:
            如果 return_parent_ids=True: 返回去重后的父块ID列表
            如果 return_parent_ids=False: 返回子块 Document 列表
        """

        # 构建过滤表达式
        filter_parts = []
        if knowledge_base_id:
            filter_parts.append(f'knowledge_base_id == "{knowledge_base_id}"')

        filter_expr = " and ".join(filter_parts) if filter_parts else None

        # 1. 稠密向量检索（语义）
        dense_vector = self.embeddings.embed_query(query)
        dense_req = AnnSearchRequest(
            data=[dense_vector],
            anns_field="dense_vector",
            param={"metric_type": "COSINE", "params": {"ef": 64}},
            limit=top_k * 2,  # 多召回一些用于融合
            expr=filter_expr
        )

        # 2. 稀疏向量检索（BM25 全文）
        sparse_req = AnnSearchRequest(
            data=[query],
            anns_field="sparse_bm25",
            param={"metric_type": "BM25"},
            limit=top_k * 2,
            expr=filter_expr
        )

        # 3. 混合检索
        results = self.client.hybrid_search(
            collection_name=self.collection_name,
            reqs=[dense_req, sparse_req],
            ranker=RRFRanker(),
            limit=top_k,
            output_fields=["chunk_id", "parent_id", "text",
                           "metadata", "chunk_index", "knowledge_base_id"]
        )

        if return_parent_ids:
            # 返回去重后的父块ID列表（用于从 PostgreSQL 查询父块）
            parent_ids = list(set([hit["entity"]["parent_id"]
                              for hit in results[0]]))
            # 同时返回匹配的子块信息，便于后续高亮显示
            matched_chunks = [
                {
                    "parent_id": hit["entity"]["parent_id"],
                    "chunk_id": hit["entity"]["chunk_id"],
                    "text": hit["entity"]["text"],
                    "score": hit.get("score", 0),
                    "chunk_index": hit["entity"].get("chunk_index", -1)
                }
                for hit in results[0]
            ]
            return {
                "parent_ids": parent_ids,
                "matched_chunks": matched_chunks
            }
        else:
            # 返回子块详情
            docs = []
            for hit in results[0]:
                doc = Document(
                    page_content=hit["entity"]["text"],
                    metadata={
                        **hit["entity"].get("metadata", {}),
                        "chunk_id": hit["entity"]["chunk_id"],
                        "parent_id": hit["entity"]["parent_id"],
                        "chunk_index": hit["entity"].get("chunk_index", -1),
                        "score": hit.get("score", 0)
                    }
                )
                docs.append(doc)
            return docs

    def search_dense_only(self, query: str, knowledge_base_id: Optional[str] = None,
                          top_k: int = 5, return_parent_ids: bool = True):
        """仅使用稠密向量检索（语义）"""
        vectorstore = Milvus(
            embedding_function=self.embeddings,
            collection_name=self.collection_name,
            connection_args={
                "uri": os.getenv("Milvus_url"),
                "token": os.getenv("Token")
            }
        )

        if knowledge_base_id:
            filter_expr = f'knowledge_base_id == "{knowledge_base_id}"'
            results = vectorstore.similarity_search_with_score(
                query, filter=filter_expr, k=top_k)
        else:
            results = vectorstore.similarity_search_with_score(query, k=top_k)

        if return_parent_ids:
            parent_ids = list(
                set([doc.metadata.get("parent_id") for doc, _ in results]))
            matched_chunks = [
                {
                    "parent_id": doc.metadata.get("parent_id"),
                    "chunk_id": doc.metadata.get("chunk_id"),
                    "text": doc.page_content,
                    "score": score,
                    "chunk_index": doc.metadata.get("chunk_index", -1)
                }
                for doc, score in results
            ]
            return {
                "parent_ids": parent_ids,
                "matched_chunks": matched_chunks
            }
        else:
            return [doc for doc, _ in results]

    def delete_file_by_hash(self, knowledge_base_id: str, file_hash: str):
        """根据文件哈希删除整个文件的所有子块"""
        
        # 构造过滤条件
        filter_expr = f'knowledge_base_id == "{knowledge_base_id}" and metadata["file_hash"] == "{file_hash}"'
        
        # 先查询有多少数据要删除
        results = self.client.query(
            collection_name=self.collection_name,
            filter=filter_expr,
            output_fields=["chunk_id"]
        )
        
        if results:
            # 删除数据
            self.client.delete(
                collection_name=self.collection_name,
                filter=filter_expr
            )
            
            # 从哈希存储中移除文件哈希
            self.hash_storage.files_hash.discard(file_hash)
            self.hash_storage._save_hashes(
                self.hash_storage.files_hash_path, 
                self.hash_storage.files_hash
            )
            
            logger.info(f"已删除文件哈希 {file_hash} 的 {len(results)} 个子块")
            return {"success": True, "deleted_chunks": len(results)}
        else:
            logger.warning(f"未找到文件哈希 {file_hash} 的数据")
            return {"success": False, "message": "文件不存在"}