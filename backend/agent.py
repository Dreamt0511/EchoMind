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
    streaming=True
)

# 创建 agent（同步创建即可）
agent = create_agent(
    tools=[search_knowledge_base],  
    model=model,
    system_prompt=config.SYSTEM_PROMPT
)

async def stream_agent_response(user_message: str) -> AsyncGenerator[str, None]:
    """
    异步流式处理 agent 响应，返回生成器供 StreamingResponse 使用
    
    Args:
        user_message: 用户输入的消息
        
    Yields:
        str: 流式输出的文本块
    """
    try:
        # 使用 astream 进行异步流式处理
        async for chunk in agent.astream(
            {"messages": [{"role": "user", "content": user_message}]},
            stream_mode="messages"
        ):
            message_chunk, metadata = chunk
            
            # 获取当前节点类型
            node = metadata.get('langgraph_node', '')
            
            # 如果是模型节点，输出 AI 响应的文本内容
            if node == 'model' and hasattr(message_chunk, 'content') and message_chunk.content:
                # 直接输出文本内容
                print(message_chunk.content)
                yield message_chunk.content
            
            # 如果是工具节点，输出工具调用信息（可选，可以注释掉）
            elif node == 'tools' and hasattr(message_chunk, 'name'):
                # 可以选择是否输出工具调用信息给前端
                # 如果需要，可以以特殊格式输出，如 SSE 格式
                tool_info = {
                    "type": "tool_call",
                    "name": message_chunk.name,
                    "status": "completed"
                }
                # 使用 SSE 格式输出工具调用信息
                yield f"data: {json.dumps(tool_info)}\n\n"
                
                # 可选：输出文档数量信息
                if hasattr(message_chunk, 'content'):
                    try:
                        content = json.loads(message_chunk.content) if isinstance(message_chunk.content, str) else message_chunk.content
                        if isinstance(content, list):
                            doc_info = {
                                "type": "tool_result",
                                "doc_count": len(content)
                            }
                            yield f"data: {json.dumps(doc_info)}\n\n"
                    except:
                        pass
           
    except Exception as e:
        logger.error(f"处理 agent 响应时发生错误: {e}")
        yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"

async def main():
    """
    主异步函数
    """
    user_message = "检索知识库，回答应该怎样对待马克思主义？"
    
    logger.info(f"{"=" * 50}开始处理用户请求{"=" * 50}")
    
    # 执行异步流式响应
    await stream_agent_response(user_message)
    
    print("\n" + "=" * 50)
    logger.info("Agent 响应处理完成")
if __name__ == "__main__":
    # 运行异步主函数
    asyncio.run(main())

