import asyncio
from documents_process import rerank_documents
from postgresql_client import get_postgresql_client
from schemas import RerankDocumentItem
from milvus_client import get_milvus_client
import logging
import sys
from langgraph.config import get_stream_writer
import time
from schemas import ContextSchema
from langchain.tools import tool, ToolRuntime


# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s-%(name)s-%(levelname)s-%(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


@tool("search_knowledge_base")
async def search_knowledge_base(query: str, # 只需要 query，其他从 runtime 获取
                                runtime: ToolRuntime[ContextSchema]) -> str:
    """
    **功能**：从私有知识库检索相关文档、专业知识、业务规则和内部资料
    - **适用场景**：
    - 问题涉及领域专业知识、技术细节或特定业务规则
    - 需要引用内部文档、官方资料或准确事实
    - 对不确定的信息进行验证，避免产生幻觉
    - 用户明确要求查询知识库或资料
    Args:
        query: 搜索查询词
    """
    # 从强类型 context 获取参数
    user_id = runtime.context.user_id
    knowledge_base_id = runtime.context.knowledge_base_id
    top_k = runtime.context.top_k
    
    print(f"当前用户ID: {user_id}\n查询问题: {query}")
    writer = get_stream_writer()
    writer(f"🔍 正在检索知识库【{knowledge_base_id}】...")
    logger.info(f"{'---'*20}开始混合检索hybrid_retrieval_knowledge_base{'---'*20}")
    start_time = time.time()


    # 使用全局 Milvus 客户端
    milvus_client = await get_milvus_client()
    parent_chunkId_list = await milvus_client.hybrid_retrieval_knowledge_base(query=query, knowledge_base_id=knowledge_base_id, top_k=top_k,user_id=user_id)
    
    logger.info(f"{'---'*20}开始从PostgreSQL获取父块get_parents，一共需要检索出{len(parent_chunkId_list)}个父块{'---'*20}")

    if not parent_chunkId_list:
        writer("⚠️ 未检索到相关文档，请尝试更换关键词或稍后再试。")
        logger.info(f"{'---'*20}未检索到相关文档{'---'*20}")
        return []

    # 使用全局 PostgreSQL 客户端
    postgresql_client = await get_postgresql_client()
    #或取父块文本列表
    parent_documents = await postgresql_client.get_parents(
        knowledge_base_id=knowledge_base_id,
        parent_ids=parent_chunkId_list,
        user_id=user_id,
    )

    # 重排序父块
    logger.info(f"{'---'*20}开始重排序父块rerank_documents{'---'*20}")
    rerank_result = await rerank_documents(query, parent_documents, top_k)

    related_documents = []
    if not rerank_result:
        # 重排序失败：返回字符串列表
        related_documents = parent_documents[:top_k]
    else:
        # 重排序成功：返回 RerankDocumentItem 对象列表
        for item in rerank_result['output']['results']:
            related_document = RerankDocumentItem(
                text=item['document']['text'],
                relevance_score=item['relevance_score']
            )
            related_documents.append(related_document)
        related_documents = related_documents[:top_k]

        print("重排序后的最相关文档的相关性得分：", related_documents[0].relevance_score if related_documents else "无相关文档")

    end_time = time.time()
    writer(f"✓ 检索完成，本次检索耗时{end_time - start_time:.2f}秒")
    logger.info(f"{'---'*20}检索完成{'---'*20}")
    return related_documents


"""
rerank_result的结构示例：
{'output': {'results': [{'document': {'text': '20 世纪80 年代末'}, 'index': 0, 
'relevance_score': 0.886919463597282}]}, 'usage': {'total_tokens': 1224}, 
'request_id': '2252323b-a3ba-4ef5-a203-e305b64249e1'}  
"""
