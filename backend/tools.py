from langchain_core.tools import tool
import asyncio
from documents_process import rerank_documents
from postgresql_client import get_postgresql_client
from schemas import RerankDocumentItem
from milvus_client import get_milvus_client
import logging
import sys

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s-%(name)s-%(levelname)s-%(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


@tool("search_knowledge_base")
async def search_knowledge_base(query: str, knowledge_base_id: str = "多页数", top_k: int = 5) -> str:
    """
    **功能**：从私有知识库检索相关文档、专业知识、业务规则和内部资料
    - **适用场景**：
    - 问题涉及领域专业知识、技术细节或特定业务规则
    - 需要引用内部文档、官方资料或准确事实
    - 对不确定的信息进行验证，避免产生幻觉
    - 用户明确要求查询知识库或资料
    - **不适用场景**：
    - 日常闲聊、问候等无需知识支撑的场景
    - 通用常识、基础推理即可解答的问题
    - 基于已有记忆就能回答的个性化问题
    """
    logger.info(f"{'---'*20}开始混合检索hybrid_retrieval{'---'*20}")

    # 使用全局 Milvus 客户端
    from api import hash_storage  # 避免循环导入问题
    milvus_client = await get_milvus_client(hash_storage)
    parent_chunkId_list = await milvus_client.hybrid_retrieval(query, knowledge_base_id, top_k)

    logger.info(
        f"{'---'*20}开始从PostgreSQL获取父块get_parents，一共需要检索出{len(parent_chunkId_list)}个父块{'---'*20}")

    # 使用全局 PostgreSQL 客户端
    postgresql_client = await get_postgresql_client()
    parent_documents = await postgresql_client.get_parents(parent_chunkId_list)

    # 提取父块文本列表
    text_list = [doc.text for doc in parent_documents]

    # 重排序父块
    logger.info(f"{'---'*20}开始重排序父块rerank_documents{'---'*20}")
    rerank_result = await rerank_documents(query, text_list, top_k)

    related_documents = []
    if not rerank_result:
        # 重排序失败：返回字符串列表
        related_documents = text_list[:top_k]
    else:
        # 重排序成功：返回 RerankDocumentItem 对象列表
        for item in rerank_result['output']['results']:
            related_document = RerankDocumentItem(
                text=item['document']['text'],
                relevance_score=item['relevance_score']
            )
            related_documents.append(related_document)
        related_documents = related_documents[:top_k]

    logger.info(f"{'---'*20}检索完成{'---'*20}")
    return related_documents


"""
rerank_result的结构示例：
{'output': {'results': [{'document': {'text': '20 世纪80 年代末'}, 'index': 0, 
'relevance_score': 0.886919463597282}]}, 'usage': {'total_tokens': 1224}, 
'request_id': '2252323b-a3ba-4ef5-a203-e305b64249e1'}  
"""
