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
from langchain.agents.middleware import ToolCallLimitMiddleware

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

# 配置工具调用次数

# 单次提问最多调用2次search_knowledge_base工具
search_knowledge_limit = ToolCallLimitMiddleware(
    tool_name="search_knowledge_base",
    run_limit=2,  # 最多调用两次
)


# 创建agent
agent = create_agent(
    tools=[search_knowledge_base],
    model=model, 
    system_prompt=config.SYSTEM_PROMPT,
    middleware=[search_knowledge_limit],
    context_schema=ContextSchema,
)


async def stream_agent_response(
    user_message: str, 
    knowledge_base_id: str,
    user_id: int
) -> AsyncGenerator[str, None]:
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
            stream_mode=["messages", "custom"],
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

    except Exception as e:
        logger.error(f"错误: {e}")
        yield json.dumps(
            {"type": "error", "content": str(e)}, ensure_ascii=False
        ) + "\n"


async def main():
    """
    主异步函数
    """
    user_message = "检索知识库，回答应该怎样对待马克思主义？"

    logger.info(f"{"=" * 50}开始处理用户请求{"=" * 50}")

    # 执行异步流式响应
    await stream_agent_response(user_message, knowledge_base_id="默认知识库")

    print("\n" + "=" * 50)
    logger.info("Agent 响应处理完成")


if __name__ == "__main__":
    # 运行异步主函数
    asyncio.run(main())
