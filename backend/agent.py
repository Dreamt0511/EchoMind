import os
import asyncio
import logging
import sys
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
import config
from typing import AsyncGenerator
from tools import search_knowledge_base
import json
from schemas import ContextSchema
from langchain.agents.middleware import ToolCallLimitMiddleware,dynamic_prompt,ModelRequest
from fastapi import BackgroundTasks
from postgresql_client import get_postgresql_client

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s-%(name)s-%(levelname)s-%(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

load_dotenv(override=True)


# 使用 OpenAI 兼容接口的方式更通用
model = ChatOpenAI(
    model=os.getenv("AGENT_BASE_MODEL"),
    openai_api_key=os.getenv("DASHSCOPE_API_KEY"),
    openai_api_base=os.getenv("BASE_URL"),
    temperature=0.7,
    streaming=True,
)


"""=============中间件============="""
# 单次提问最多调用2次search_knowledge_base工具
search_knowledge_limit = ToolCallLimitMiddleware(
    tool_name="search_knowledge_base",
    run_limit=2,  # 最多调用两次
)

#动态系统系统提示词切换，根据知识库id切换系统提示词
@dynamic_prompt
def dynamic_prompt(request: ModelRequest) -> str:
    knowledge_base_id = request.runtime.context.knowledge_base_id
    if knowledge_base_id == "默认知识库":
        return config.SYSTEM_PROMPT_DEFAULT
    else:
        return config.SYSTEM_PROMPT_SPECIFIC

# 创建agent
agent = create_agent(
    model=model, 
    tools=[search_knowledge_base],
    middleware=[search_knowledge_limit,dynamic_prompt],
    context_schema=ContextSchema,
)


async def stream_agent_response(
    user_message: str, 
    knowledge_base_id: str,
    user_id: int,
    background_tasks: BackgroundTasks = None,
) -> AsyncGenerator[str, None]:
    ai_message = ""
    try:
        async for chunk in agent.astream(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": user_message
                    }
                ]
            },
            stream_mode=["messages", "custom"],#使用messages模式逐token输出实现打字效果，custom模式输出工具调用信息展示agent的思考过程
            context=ContextSchema(  # 传递强类型 context
                user_id=user_id,
                knowledge_base_id=knowledge_base_id,
                top_k=5
            )
        ):
            stream_mode, chunk_data = chunk

            # ---------------- 回答内容 ----------------
            if stream_mode == "messages":
                message_chunk, metadata = chunk_data
                node = metadata.get("langgraph_node", "")
                if (
                    node == "model"
                    and hasattr(message_chunk, "content")
                    and message_chunk.content
                ):
                    ai_message += message_chunk.content#累加ai回复
                    yield json.dumps(
                        {"type": "answer", "content": message_chunk.content},
                        ensure_ascii=False,
                    ) + "\n"

            # ---------------- 工具状态（检索提示） ----------------
            elif stream_mode == "custom":
                custom_info = chunk_data
                logger.info(f"【工具信息】: {custom_info}")
                yield json.dumps(
                    {"type": "status", "content": custom_info}, ensure_ascii=False
                ) + "\n"

        #流式响应结束或，添加后台任务保存对话
        # 流式结束后，用 background_tasks 保存
        if background_tasks:
            background_tasks.add_task(
                save_conversation_messages,
                user_id=user_id,
                thread_id=f"{user_id}",
                user_message=user_message,
                ai_message=ai_message
            )

    except Exception as e:
        logger.error(f"错误: {e}")
        yield json.dumps(
            {"type": "error", "content": str(e)}, ensure_ascii=False
        ) + "\n"

# 保存对话到数据库
async def save_conversation_messages(
    user_id: int, 
    thread_id: str, 
    user_message: str, 
    ai_message: str
):
    """后台任务：保存对话到数据库"""
    try:
        postgresql_client = await get_postgresql_client()
        
        # 保存用户消息
        await postgresql_client.add_conversation_message(
            user_id=user_id,
            thread_id=thread_id,
            role="human",
            content=user_message
        )
        
        # 保存 AI 回复
        await postgresql_client.add_conversation_message(
            user_id=user_id,
            thread_id=thread_id,
            role="ai",
            content=ai_message
        )
        
        logger.info(f"对话已保存: user_id={user_id}, thread_id={thread_id}")
        
    except Exception as e:
        logger.error(f"保存对话失败: {e}")
