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
   