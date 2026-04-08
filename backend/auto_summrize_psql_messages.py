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


def format_conversation_from_db(messages: List[Dict[str, Any]]) -> str:
    """
    将数据库查询的消息列表格式化为对话文本
    数据库中的每条消息已经有 role 字段，直接拼接成 "role: content" 格式
    
    Args:
        messages: 数据库查询的消息列表，每条包含 role 和 content
    
    Returns:
        格式化的对话文本
    """
    formatted = []
    for msg in messages:
        role = msg.get("role", "").lower()
        content = msg.get("content", "")
        formatted.append(f"{role}: {content}")
    return "\n\n".join(formatted)


def count_text_tokens(text: str) -> int:
    """
    估算文本的 token 数量
    使用 LangChain 的 count_tokens_approximately，需要包装成消息列表
    
    Args:
        text: 要计算 token 的文本
    
    Returns:
        估算的 token 数量
    """
    if not text:
        return 0
    # count_tokens_approximately 需要传入消息列表
    from langchain_core.messages import HumanMessage
    return count_tokens_approximately([HumanMessage(content=text)])


async def get_unsunmarized_conversations(user_id: int, thread_id: str = None) -> List[Dict[str, Any]]:
    """获取指定用户和会话中未摘要的对话消息"""
    pg_client = await get_postgresql_client()
    
    if not pg_client.pool:
        raise RuntimeError("数据库连接池未初始化")
    
    async with pg_client.pool.acquire() as conn:
        if thread_id:
            rows = await conn.fetch("""
                SELECT id, role, content, created_at, thread_id
                FROM raw_conversations
                WHERE user_id = $1 
                  AND thread_id = $2
                  AND (summary_id IS NULL OR summary_id = '')
                ORDER BY created_at ASC
            """, user_id, thread_id)
        else:
            rows = await conn.fetch("""
                SELECT id, role, content, created_at, thread_id
                FROM raw_conversations
                WHERE user_id = $1 
                  AND (summary_id IS NULL OR summary_id = '')
                ORDER BY thread_id, created_at ASC
            """, user_id)
        
        messages = [dict(row) for row in rows]
        logger.info(f"获取到用户 {user_id} 的 {len(messages)} 条未摘要消息")
        return messages


async def update_messages_with_summary(message_ids: List[str], summary_id: str) -> bool:
    """更新消息的 summary_id"""
    pg_client = await get_postgresql_client()
    
    if not pg_client.pool:
        raise RuntimeError("数据库连接池未初始化")
    
    try:
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
    """压缩单个会话的消息并生成摘要"""
    if not messages:
        return {"success": False, "reason": "没有消息需要压缩"}
    
    # 1. 格式化为对话文本
    conversation_text = format_conversation_from_db(messages)
    
    # 2. 计算 token 数
    total_tokens = count_text_tokens(conversation_text)
    
    logger.info(f"会话 {thread_id} 共 {len(messages)} 条消息，总 token 数: {total_tokens}")
    
    # 3. 检查是否超过阈值
    if total_tokens <= TOKEN_THRESHOLD:
        logger.info(f"会话 {thread_id} token 数 ({total_tokens}) 未超过阈值 {TOKEN_THRESHOLD}，跳过压缩")
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
        success = await update_messages_with_summary(message_ids, summary_id)
        
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
    user_id: int = None,
    thread_id: str = None
) -> Dict[str, Any]:
    """处理所有用户的未摘要对话"""
    pg_client = await get_postgresql_client()
    
    if not pg_client.pool:
        raise RuntimeError("数据库连接池未初始化")
    
    results = {
        "total_conversations_processed": 0,
        "compressed_count": 0,
        "skipped_count": 0,
        "failed_count": 0,
        "details": []
    }
    
    if user_id and thread_id:
        # 处理指定用户的指定会话
        messages = await get_unsunmarized_conversations(user_id, thread_id)
        if messages:
            result = await compress_and_summarize_conversation(
                messages, model, user_id, thread_id
            )
            results["total_conversations_processed"] = 1
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
    
    elif user_id:
        # 处理指定用户的所有会话
        async with pg_client.pool.acquire() as conn:
            threads = await conn.fetch("""
                SELECT DISTINCT thread_id 
                FROM raw_conversations 
                WHERE user_id = $1 
                  AND (summary_id IS NULL OR summary_id = '')
                ORDER BY thread_id
            """, user_id)
        
        for thread in threads:
            thread_id_val = thread["thread_id"]
            messages = await get_unsunmarized_conversations(user_id, thread_id_val)
            if messages:
                result = await compress_and_summarize_conversation(
                    messages, model, user_id, thread_id_val
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
                    "thread_id": thread_id_val,
                    **result
                })
    
    else:
        # 处理所有用户的所有未摘要会话
        async with pg_client.pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT DISTINCT user_id, thread_id
                FROM raw_conversations
                WHERE summary_id IS NULL OR summary_id = ''
                ORDER BY user_id, thread_id
            """)
        
        for row in rows:
            uid = row["user_id"]
            tid = row["thread_id"]
            messages = await get_unsunmarized_conversations(uid, tid)
            if messages:
                result = await compress_and_summarize_conversation(
                    messages, model, uid, tid
                )
                results["total_conversations_processed"] += 1
                if result.get("success"):
                    results["compressed_count"] += 1
                elif "未超过阈值" in result.get("reason", ""):
                    results["skipped_count"] += 1
                else:
                    results["failed_count"] += 1
                results["details"].append({
                    "user_id": uid,
                    "thread_id": tid,
                    **result
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
        raise


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