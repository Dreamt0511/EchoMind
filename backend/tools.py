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
import config


# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s-%(name)s-%(levelname)s-%(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


@tool("get_memory")
async def get_memory(
    query: str,
    semantic_k: int,
    episodic_k: int,
    procedural_k: int,
    runtime: ToolRuntime[ContextSchema],
) -> str:
    """
    检索用户长期记忆，用于个性化回答和历史连贯性。

    【调用时机】满足以下任一条件即调用：
    - 用户提到之前、刚才、上次等回忆性词汇
    - 重复提问或用户表现困惑
    - 询问操作步骤、个人偏好、习惯设置
    - 任何需要记住历史对话的场景

    【参数决策】（k值 0-5，0=不需要，3-5=核心依赖）
    - semantic_k（事实/概念）：事实问答3-5，操作问题2-3，历史回忆1-2
    - episodic_k（历史/对话）：重复提问/回忆3-5，事实问答2-3，操作问题1-2
    - procedural_k（步骤/方法）：操作问题3-5，事实问答2-3，历史回忆1-2

    Args:
        query: 用户原始问题（如需了解记忆详情可做适当改写）
        semantic_k: 语义记忆召回数量
        episodic_k: 情节记忆召回数量
        procedural_k: 程序记忆召回数量
    """
    logger.info(f"{'---'*20}开始检索记忆hybrid_retrieval_memories{'---'*20}")
    logger.info(f"ai希望检索的各种记忆数量: 语义记忆{semantic_k}, 情节记忆{episodic_k}, 程序记忆{procedural_k}")
    user_id = runtime.context.user_id
    writer = get_stream_writer()
    writer(f"🔍 正在检索记忆库...")
    start_time = time.time()
    milvus_client = await get_milvus_client()
    memories_dict = await milvus_client.hybrid_retrieval_memories(
        query,
        user_id=user_id,
        summary_k=1,
        semantic_k=semantic_k,
        episodic_k=episodic_k,
        procedural_k=procedural_k,
    )
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
    count = 1
    for memory in memories_dict.values():
        if isinstance(memory, list):
            count += len(memory)
        
    writer(f"🔍 检索到{count}条记忆，耗时{time.time() - start_time:.2f}秒")
    now = int(time.time())
    memories_text = config.MEMORY_USAGE_PROMPT.format(
        memories_dict=memories_dict,
        current_timestamp=now,
    )
    return memories_text


@tool("search_knowledge_base")
async def search_knowledge_base(
    query: str, runtime: ToolRuntime[ContextSchema]  # 只需要 query，其他从 runtime 获取
) -> str:
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
    parent_chunkId_list = await milvus_client.hybrid_retrieval_knowledge_base(
        query=query, knowledge_base_id=knowledge_base_id, top_k=top_k, user_id=user_id
    )

    logger.info(
        f"{'---'*20}开始从PostgreSQL获取父块get_parents，一共需要检索出{len(parent_chunkId_list)}个父块{'---'*20}"
    )

    if not parent_chunkId_list:
        writer("⚠️ 未检索到相关文档，请尝试更换关键词或稍后再试。")
        logger.info(f"{'---'*20}未检索到相关文档{'---'*20}")
        return "未检索到相关文档"

    # 使用全局 PostgreSQL 客户端
    postgresql_client = await get_postgresql_client()
    # 或取父块文本列表
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
        for item in rerank_result["output"]["results"]:
            related_document = RerankDocumentItem(
                text=item["document"]["text"], relevance_score=item["relevance_score"]
            )
            related_documents.append(related_document)
        related_documents = related_documents[:top_k]

        print(
            "重排序后的最相关文档的相关性得分：",
            related_documents[0].relevance_score if related_documents else "无相关文档",
        )

    end_time = time.time()
    writer(f"✓ 检索完成，本次检索耗时{end_time - start_time:.2f}秒")
    logger.info(f"{'---'*20}检索完成{'---'*20}")

    # 确保 related_documents 转换为字符串
    if isinstance(related_documents, list):
        if not related_documents:
            return "未检索到相关文档"
        # 将列表中的每个文档转换为字符串
        doc_strings = []
        for doc in related_documents:
            if hasattr(doc, 'text'):
                doc_strings.append(doc.text)
            else:
                doc_strings.append(str(doc))
        documents_text = "\n---\n".join(doc_strings)
    else:
        documents_text = str(related_documents)
    
    related_documents_result = f"检索到的相关文档如下：\n{documents_text}"
    return related_documents_result

"""rerank_result的结构示例：
{'output': {'results': [{'document': {'text': '20 世纪80 年代末'}, 'index': 0, 
'relevance_score': 0.886919463597282}]}, 'usage': {'total_tokens': 1224}, 
'request_id': '2252323b-a3ba-4ef5-a203-e305b64249e1'}  
"""