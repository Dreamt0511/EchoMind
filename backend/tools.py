from langchain_core.tools import tool
import asyncio
from milvus_client import AsyncMilvusClientWrapper
from documents_process import rerank_documents
from postgresql_client import PostgreSQLParentClient

@tool("search_knowledge_base")
async def search_knowledge_base(query: str,knowledge_base_id: str, top_k: int = 10) -> str:
    """
    Hybrid retrieval from the private knowledge base (dense + sparse vectors) to obtain relevant documents, professional knowledge, business rules, and internal materials.
    Use when:
    - The question involves domain-specific knowledge, internal docs, or uncertain facts
    - You need accurate information to avoid hallucinations
    Do NOT use for general common sense, casual chat, or easily inferred answers.
    Returns reranked, highly relevant document snippets.
    """
    async with AsyncMilvusClientWrapper() as milvus_client:
        parent_chunkId_list = await milvus_client.hybrid_retrieval(
            query, knowledge_base_id, top_k)
        
        # 从 PostgreSQL获取父块
        async with PostgreSQLParentClient() as postgresql_client:
            parent_documents = await postgresql_client.get_parents(parent_chunkId_list)
            text_list = [doc.text for doc in parent_documents]
            rerank_result = await rerank_documents(query, text_list, top_k)
            related_documents = []
            if not rerank_result:
                #重排序失败的情况下降级取RRF融合后的前10个片段
                related_documents = parent_documents[:top_k]
            else:
                for item in rerank_result['output']['results']:
                    related_documents.append(item['document']['text'])
                related_documents = related_documents[:top_k]
        
        return related_documents