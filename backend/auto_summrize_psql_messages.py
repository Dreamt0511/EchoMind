"""
定时自动压缩psql中的历史对话，压缩函数提取的是langchain中的SummarizationMiddleware，
压缩时检查对话是否超过4000个token，超过则压缩，否则不压缩。
"""

import asyncio
import logging
import uuid
from typing import List, Dict, Any
from datetime import datetime

from langchain_core.messages import HumanMessage
from langchain_core.language_models import BaseChatModel
from langchain_core.messages.utils import count_tokens_approximately
from langchain_core.messages import HumanMessage
from postgresql_client import get_postgresql_client
from config import DEFAULT_SUMMARY_PROMPT

logger = logging.getLogger(__name__)

# Token 阈值
TOKEN_THRESHOLD = 4000


async def compress_messages(
    conversation_text: str,
    model: BaseChatModel,
) -> str:
    """
    压缩对话文本为摘要

    Args:
        conversation_text: 格式化的对话文本（已经是 "role: content" 格式）
        model: 用于生成摘要的 LLM 模型

    Returns:
        压缩后的摘要
    """
    if not conversation_text:
        return ""

    # 调用模型生成摘要
    response = await model.ainvoke(
        [
            HumanMessage(
                content=DEFAULT_SUMMARY_PROMPT.format(
                    conversation_text=conversation_text
                )
            )
        ]
    )
    summary = response.content.strip()

    logger.info(f"压缩完成: 生成摘要 ({len(summary)} 字符)")

    return f"Previous conversation summary:\n{summary}"
   


async def get_unsunmarized_conversations(user_id: int, thread_id: str) -> List[Dict[str, Any]]:
    """获取指定用户和会话中未摘要的对话消息"""
    
    try:
        pg_client = await get_postgresql_client()
        
        if not pg_client.pool:
            logger.error("数据库连接池未初始化，无法获取未摘要消息")
            return []
        
        async with pg_client.pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT id, role, content, created_at, thread_id
                FROM raw_conversations
                WHERE user_id = $1 
                AND thread_id = $2
                AND (summary_id IS NULL OR summary_id = '')
                ORDER BY created_at ASC
            """, user_id, thread_id)
            
            messages = [dict(row) for row in rows]
            logger.info(f"获取到用户 {user_id} 会话 {thread_id} 的 {len(messages)} 条未摘要消息")
            return messages
            
    except Exception as e:
        logger.error(f"获取未摘要消息失败 user_id={user_id}, thread_id={thread_id}: {e}")
        return []


async def update_messages_with_summary_id(message_ids: List[str], summary_id: str) -> bool:
    """更新消息的 summary_id（后台任务，不抛出异常）"""
    try:
        pg_client = await get_postgresql_client()
        
        if not pg_client.pool:
            logger.error("数据库连接池未初始化，无法更新 summary_id")
            return False
        
        async with pg_client.pool.acquire() as conn:
            async with conn.transaction():
                await conn.execute("""
                    UPDATE raw_conversations
                    SET summary_id = $1
                    WHERE id = ANY($2::text[])
                """, summary_id, message_ids)
                
                logger.info(f"成功更新 {len(message_ids)} 条消息的 summary_id 为 {summary_id}")
                return True
                
    except Exception as e:
        logger.error(f"更新消息的 summary_id 失败: {e}")
        return False


async def compress_and_summarize_conversation(
    messages: List[Dict[str, Any]],
    model: BaseChatModel,
    user_id: int,
    thread_id: str
) -> Dict[str, Any]:
    """压缩单个会话的消息并生成摘要，更新数据库（psql中加上summary_id字段,摘要插入到milvus中）"""
    if not messages:
        return {"success": False, "reason": "没有消息需要压缩"}
    
    # 1. 格式化为对话文本
    formatted = []
    for msg in messages:
        role = msg.get("role", "").lower()
        content = msg.get("content", "")
        formatted.append(f"{role}: {content}")
    conversation_text = "\n\n".join(formatted)
    
    # 2. 计算 token 数,count_tokens_approximately 需要传入消息列表
    total_tokens = count_tokens_approximately([HumanMessage(content=conversation_text)])
    
    logger.info(f"会话 {thread_id} 共 {len(messages)} 条消息，总 token 数: {total_tokens}")
    
    # 3. 检查是否超过阈值
    if total_tokens <= TOKEN_THRESHOLD:
        return {
            "success": False,
            "reason": f"token 数未超过阈值 ({total_tokens} <= {TOKEN_THRESHOLD})",
            "token_count": total_tokens,
            "message_count": len(messages)
        }
    
    # 4. 调用压缩函数生成摘要
    try:
        summary = await compress_messages(conversation_text, model)
        
        # 5. 生成摘要ID
        summary_id = str(uuid.uuid4())
        
        # 6. 更新数据库
        message_ids = [msg["id"] for msg in messages]
        success = await update_messages_with_summary_id(message_ids, summary_id)
        
        if success:
            logger.info(f"成功压缩会话 {thread_id}，生成摘要 {summary_id}")
            return {
                "success": True,
                "summary_id": summary_id,
                "summary": summary,
                "token_count": total_tokens,
                "message_count": len(messages)
            }
        else:
            return {
                "success": False,
                "reason": "数据库更新失败",
                "token_count": total_tokens,
                "message_count": len(messages)
            }
    except Exception as e:
        logger.error(f"压缩会话 {thread_id} 失败: {e}")
        return {
            "success": False,
            "reason": str(e),
            "token_count": total_tokens,
            "message_count": len(messages)
        }


async def process_all_users_conversations(
    model: BaseChatModel,
) -> Dict[str, Any]:
    """处理所有用户的未摘要对话（后台任务，不抛出异常）"""
    
    try:
        pg_client = await get_postgresql_client()
        
        if not pg_client.pool:
            logger.error("数据库连接池未初始化，无法执行压缩任务")
            return {
                "total_conversations_processed": 0,
                "compressed_count": 0,
                "skipped_count": 0,
                "failed_count": 0,
                "error": "数据库连接池未初始化",
                "details": []
            }
        
        results = {
            "total_conversations_processed": 0,
            "compressed_count": 0,
            "skipped_count": 0,
            "failed_count": 0,
            "details": []
        }
        
        # 获取所有未摘要的会话
        async with pg_client.pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT DISTINCT user_id, thread_id
                FROM raw_conversations
                WHERE summary_id IS NULL
                ORDER BY user_id, thread_id
            """)
        
        # 处理每个会话，也要捕获单个会话的异常
        for row in rows:
            user_id = row["user_id"]
            thread_id = row["thread_id"]
            
            try:
                # 获取未摘要的会话消息
                messages = await get_unsunmarized_conversations(user_id, thread_id)
                if messages:
                    # 压缩会话消息
                    result = await compress_and_summarize_conversation(
                        messages, model, user_id, thread_id
                    )
                    results["total_conversations_processed"] += 1
                    
                    if result.get("success"):
                        results["compressed_count"] += 1
                    elif "未超过阈值" in result.get("reason", ""):
                        results["skipped_count"] += 1
                    else:
                        results["failed_count"] += 1
                    
                    results["details"].append({
                        "user_id": user_id,
                        "thread_id": thread_id,
                        **result
                    })
            except Exception as e:
                # 单个会话处理失败，记录错误但继续处理其他会话
                logger.error(f"处理会话失败 user_id={user_id}, thread_id={thread_id}: {e}", exc_info=True)
                results["failed_count"] += 1
                results["details"].append({
                    "user_id": user_id,
                    "thread_id": thread_id,
                    "success": False,
                    "reason": f"处理异常: {str(e)}"
                })
        
        # 输出汇总信息
        logger.info("=" * 60)
        logger.info("压缩任务完成汇总:")
        logger.info(f"  处理会话数: {results['total_conversations_processed']}")
        logger.info(f"  成功压缩数: {results['compressed_count']}")
        logger.info(f"  跳过（未超阈值）: {results['skipped_count']}")
        logger.info(f"  失败数: {results['failed_count']}")
        logger.info("=" * 60)
        
        return results
        
    except Exception as e:
        # 捕获整个任务的致命错误
        logger.error(f"压缩任务执行失败: {e}", exc_info=True)
        return {
            "total_conversations_processed": 0,
            "compressed_count": 0,
            "skipped_count": 0,
            "failed_count": 0,
            "error": str(e),
            "details": []
        }


async def run_compression_task(model: BaseChatModel, user_id: int = None):
    """定时任务入口函数"""
    logger.info("开始执行对话压缩任务...")
    start_time = datetime.now()
    
    try:
        pg_client = await get_postgresql_client()
        await pg_client.init_pool()
        
        results = await process_all_users_conversations(model, user_id)
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        logger.info(f"压缩任务完成，耗时: {duration:.2f} 秒")
        
        return results
    except Exception as e:
        logger.error(f"压缩任务执行失败: {e}")
        return


# ============ 使用示例 ============
if __name__ == "__main__":
    import sys
    sys.path.append(".")
    
    from langchain_openai import ChatOpenAI
    
    async def main():
        model = ChatOpenAI(
            model="gpt-3.5-turbo",
            temperature=0,
            max_tokens=1000
        )
        
        results = await run_compression_task(model)
        
        print("\n详细结果:")
        for detail in results["details"]:
            if detail.get("success"):
                print(f"  ✅ 用户 {detail['user_id']}, 会话 {detail['thread_id'][:20]}...: "
                      f"压缩 {detail['message_count']} 条消息 ({detail['token_count']} tokens)")
            elif "未超过阈值" in detail.get("reason", ""):
                print(f"  ⏭️  用户 {detail['user_id']}, 会话 {detail['thread_id'][:20]}...: "
                      f"跳过 ({detail['token_count']} tokens)")
            else:
                print(f"  ❌ 用户 {detail['user_id']}, 会话 {detail['thread_id'][:20]}...: "
                      f"失败 - {detail.get('reason', '未知错误')}")
    
    asyncio.run(main())