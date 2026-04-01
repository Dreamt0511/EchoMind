import os
import asyncio
import logging
import sys
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
import config
from tools import search_knowledge_base

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s-%(name)s-%(levelname)s-%(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

load_dotenv(override=True)

# 使用 OpenAI 兼容接口的方式更通用
model = ChatOpenAI(
    model=os.getenv("AGENT_BASE_MODEL"),
    openai_api_key=os.getenv("DASHSCOPE_API_KEY"),
    openai_api_base=os.getenv("BASE_URL"),
    temperature=0.7,
)

# 创建 agent（同步创建即可）
agent = create_agent(
    tools=[search_knowledge_base],  
    model=model,
    system_prompt=config.SYSTEM_PROMPT
)

async def stream_agent_response(user_message: str):
    """
    异步流式处理 agent 响应
    
    Args:
        user_message: 用户输入的消息
    """
    try:
        # 使用 astream 进行异步流式处理
        async for chunk in agent.astream(
            {"messages": [{"role": "user", "content": user_message}]}
        ):
            # 处理不同类型的 chunk
            if isinstance(chunk, dict):
                # 如果是消息类型
                if "messages" in chunk:
                    for msg in chunk["messages"]:
                        if hasattr(msg, 'content') and msg.content:
                            print(msg.content, end='', flush=True)
                # 如果是工具调用结果
                elif "tools" in chunk:
                    for tool_result in chunk["tools"]:
                        if hasattr(tool_result, 'content') and tool_result.content:
                            print(f"\n[工具结果]: {tool_result.content}", flush=True)
                # 其他类型的响应
                else:
                    print(chunk, end='', flush=True)
            else:
                print(chunk, end='', flush=True)
                
    except Exception as e:
        logger.error(f"流式处理出错: {e}", exc_info=True)
        raise

async def main():
    """
    主异步函数
    """
    user_message = "你是谁？"
    
    logger.info("开始处理用户请求")
    logger.info("=" * 50)
    
    # 执行异步流式响应
    await stream_agent_response(user_message)
    
    print("\n" + "=" * 50)
    logger.info("Agent 响应处理完成")
if __name__ == "__main__":
    # 运行异步主函数
    asyncio.run(main())

