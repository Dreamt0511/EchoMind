"""
定时自动压缩psql中的历史对话，压缩函数提取的是langchain中的SummarizationMiddleware，
压缩时检查对话是否超过4000个token，超过则压缩，否则不压缩。
新增逻辑：大模型识别后半部分语义差异大且不完整的消息，仅压缩前半部分相关内容，返回过滤的消息ID列表
"""
import asyncio
import logging
import uuid
import json
from typing import List, Dict, Any
from datetime import datetime
import config
from langchain_core.messages import HumanMessage
from langchain_core.language_models import BaseChatModel
from langchain_core.messages.utils import count_tokens_approximately
from postgresql_client import get_postgresql_client
from langchain_openai import ChatOpenAI
from milvus_client import get_milvus_client
import os
logger = logging.getLogger(__name__)

# Token 阈值,debug下设置成600，生产下设置成2000
TOKEN_THRESHOLD = 2000 #中间件压缩的是4000token
MEMORY_EXTRACT_PROMPT = config.MEMORY_EXTRACT_PROMPT

async def extract_memories(
    messages: List[Dict[str, Any]],
    model: BaseChatModel,
) -> Dict[str, Any]:
    
    """
    从对话中提取多种类型的记忆

    Args:
        messages: 原始消息列表（包含id/role/content/created_at）
        model: 用于提取记忆的 LLM 模型

    Returns:
        dict: {
            "summary": {"content": str, "importance_score": float},
            "semantic_memory": [{"content": str, "importance_score": float}, ...],
            "episodic_memory": [{"content": str, "importance_score": float}, ...],
            "procedural_memory": [{"content": str, "importance_score": float}, ...],
            "user_profile": str,
            "filtered_message_ids": list
        }
    """
    if not messages:
        return {
            "summary": {"content": "", "importance_score": 0.0},
            "semantic_memory": [],
            "episodic_memory": [],
            "procedural_memory": [],
            "user_profile": "",
            "filtered_message_ids": [],
        }

    # 格式化对话文本，每个消息占一行，格式为：id | role | content | time
    lines = []
    for msg in messages:
        lines.append(f"{msg.get('id', '')} | {msg.get('role', '')} | {msg.get('content', '')} | {msg.get('created_at', 'N/A')}")
    conversation_text = "\n\n".join(lines)

    # 调用模型
    response = await model.ainvoke([
        HumanMessage(content=MEMORY_EXTRACT_PROMPT.format(conversation_text=conversation_text))
    ])
    
    content = response.content.strip()
    if content.startswith("```json"):
        content = content[7:]
    elif content.startswith("```"):
        content = content[3:]
    if content.endswith("```"):
        content = content[:-3]
    
    result = json.loads(content.strip())

    # 解析结果
    def safe_score(v):
        try:
            return round(max(0.0, min(1.0, float(v))), 1)#将importance_score安全地限制在 0.0 到 1.0 之间，并保留 1 位小数
        except:
            return 0.5
    
    # 解析记忆项，确保 content 字段存在且非空字符串
    def parse_items(items):
        if not isinstance(items, list):
            return []
        return [
            {"content": item["content"].strip(), "importance_score": safe_score(item.get("importance_score", 0.5))}
            for item in items if isinstance(item, dict) and item.get("content", "").strip()#确保 content 字段存在且非空字符串
        ]

    summary_raw = result.get("summary", {})
    
    return {
        "summary": {
            "content": summary_raw.get("content", "").strip(),
            "importance_score": safe_score(summary_raw.get("importance_score", 0.5))
        },
        "semantic_memory": parse_items(result.get("semantic_memory")),
        "episodic_memory": parse_items(result.get("episodic_memory")),
        "procedural_memory": parse_items(result.get("procedural_memory")),
        "user_profile": result.get("user_profile", "").strip(),
        "filtered_message_ids": result.get("filtered_message_ids", []),
    }


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
        logger.error(f"获取未摘要消息失败 user_id={user_id}, thread_id={thread_id}: {e}", exc_info=True)
        return []


#这个才是存入记忆的关键
async def extract_and_append_memory(
    messages: List[Dict[str, Any]],
    model: BaseChatModel,
    user_id: int,
    thread_id: str
) -> Dict[str, Any]:
    """压缩单个会话的消息并生成摘要"""
    if not messages:
        return {"success": False, "reason": "没有消息需要压缩"}
    
    # 1. 计算 token 数
    conversation_text = "\n\n".join([f"{m.get('role')}: {m.get('content')}" for m in messages])
    total_tokens = count_tokens_approximately([HumanMessage(content=conversation_text)])
    
    if total_tokens <= TOKEN_THRESHOLD:
        return {
            "success": False,
            "reason": f"token 数未超过阈值 ({total_tokens} <= {TOKEN_THRESHOLD})",
            "token_count": total_tokens,
            "message_count": len(messages)
        }
    
    # 2. 提取记忆
    try:
        extract_result = await extract_memories(messages, model)
    except Exception as e:
        logger.error(f"记忆提取失败: {e}", exc_info=True)
        return {
            "success": False,
            "reason": f"记忆提取失败: {str(e)}",
            "token_count": total_tokens,
            "message_count": len(messages)
        }
    
    # 3. 存储记忆
    filtered_message_ids = extract_result.get("filtered_message_ids", [])
    try:
        return await _store_memories(
            messages=messages,
            extract_result=extract_result,
            user_id=user_id,
            thread_id=thread_id,
            total_tokens=total_tokens,
            filtered_message_ids=filtered_message_ids
        )
    except Exception as e:
        logger.error(f"存储记忆失败: {e}", exc_info=True)
        return {
            "success": False,
            "reason": f"存储记忆失败: {str(e)}",
            "filtered_message_ids": filtered_message_ids,
            "token_count": total_tokens,
            "message_count": len(messages)
        }


async def _store_memories(
    messages: List[Dict[str, Any]],
    extract_result: Dict[str, Any],
    user_id: int,
    thread_id: str,
    total_tokens: int,
    filtered_message_ids: List[str]
) -> Dict[str, Any]:
    """存储记忆到 PostgreSQL 和 Milvus，任何一项失败都不提交数据"""
    
    new_user_profile = extract_result.get("user_profile", "")
    summary_id = str(uuid.uuid4())
    
    # 确定需要更新的消息ID
    all_message_ids = [str(msg["id"]) for msg in messages]
    update_message_ids = [mid for mid in all_message_ids if mid not in filtered_message_ids]
    
    if not update_message_ids:
        logger.warning(f"会话 {thread_id} 所有消息都被过滤")
        return {
            "success": False,
            "reason": "所有消息都被过滤",
            "token_count": total_tokens,
            "message_count": len(messages)
        }
    
    # 准备数据
    milvus_client = await get_milvus_client()
    pg_client = await get_postgresql_client()
    keep_keys = ["summary", "semantic_memory", "episodic_memory", "procedural_memory"]
    filtered_memory = {k: v for k, v in extract_result.items() if k in keep_keys}
    
    # 查询旧画像（用于回滚）
    old_user_profile = await pg_client.get_user_profile(user_id)
    
    try:
        # 1. 更新 summary_id
        summary_success = await pg_client.update_messages_with_summary_id(update_message_ids, summary_id)
        if not summary_success:
            logger.error(f"summary_id 更新失败，终止操作")
            return {
                "success": False,
                "reason": "summary_id 更新失败",
                "filtered_message_ids": filtered_message_ids,
                "token_count": total_tokens,
                "message_count": len(messages)
            }
        
        # 2. 更新用户画像
        if new_user_profile:
            profile_success = await pg_client.update_user_profile(user_id, new_user_profile)
            if not profile_success:
                logger.error(f"用户画像更新失败，回滚 summary_id")
                await pg_client.update_messages_with_summary_id(update_message_ids, None)
                return {
                    "success": False,
                    "reason": "用户画像更新失败，已回滚",
                    "filtered_message_ids": filtered_message_ids,
                    "token_count": total_tokens,
                    "message_count": len(messages)
                }
        
        # 3. 更新 Milvus
        milvus_success = await milvus_client.add_memories_batch(
            user_id=user_id,
            thread_id=thread_id,
            memory_dict=filtered_memory,
            summary_id=summary_id,
        )
        
        if not milvus_success:
            logger.error(f"Milvus插入失败，回滚PostgreSQL")
            # 回滚 summary_id
            await pg_client.update_messages_with_summary_id(update_message_ids, None)
            # 回滚用户画像
            if old_user_profile is not None:
                await pg_client.update_user_profile(user_id, old_user_profile)
            return {
                "success": False,
                "reason": "Milvus插入失败，已回滚PostgreSQL",
                "filtered_message_ids": filtered_message_ids,
                "token_count": total_tokens,
                "message_count": len(messages)
            }
        
        # 全部成功
        logger.info(f"成功压缩会话 {thread_id}，摘要 {summary_id}")
        return {
            "success": True,
            "summary_id": summary_id,
            "user_profile": new_user_profile,
            "filtered_message_ids": filtered_message_ids,
            "token_count": total_tokens,
            "message_count": len(messages),
            "updated_message_count": len(update_message_ids)
        }
        
    except Exception as e:
        logger.error(f"存储记忆异常: {e}")
        
        # 尝试回滚
        try:
            await pg_client.update_messages_with_summary_id(update_message_ids, None)
            if old_user_profile is not None:
                await pg_client.update_user_profile(user_id, old_user_profile)
        except Exception as rollback_error:
            logger.error(f"回滚失败: {rollback_error}")
            
        return {
            "success": False,
            "reason": f"存储记忆失败: {str(e)}",
            "filtered_message_ids": filtered_message_ids,
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
        
        # 处理每个会话，捕获单个会话的异常
        for row in rows:
            user_id = row["user_id"]
            thread_id = row["thread_id"]
            
            try:
                # 获取未摘要的会话消息
                messages = await get_unsunmarized_conversations(user_id, thread_id)
                if messages:
                    # 压缩会话消息
                    result = await extract_and_append_memory(
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
                        "thread_id": thread_id,** result
                    })
            except Exception as e:
                # 单个会话处理失败，记录错误但继续处理其他会话
                logger.error(f"处理会话失败 user_id={user_id}, thread_id={thread_id}: {e}", exc_info=True)
                results["failed_count"] += 1
                results["details"].append({
                    "user_id": user_id,
                    "thread_id": thread_id,
                    "success": False,
                    "reason": f"处理异常: {str(e)}",
                    "filtered_message_ids": []
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


async def run_compression_task(model: BaseChatModel):
    """定时任务入口函数（兼容user_id参数，后台任务不抛出错误）"""
    logger.info("开始执行记忆存储任务...")
    start_time = datetime.now()
    
    try:
        pg_client = await get_postgresql_client()
        await pg_client.init_pool()
        
        # 兼容原参数（实际未使用user_id，保持接口一致）
        results = await process_all_users_conversations(model)
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        logger.info(f"记忆存储任务完成，耗时: {duration:.2f} 秒")
        
        return results
    except Exception as e:
        logger.error(f"记忆存储任务执行失败: {e}", exc_info=True)
        return {
            "total_conversations_processed": 0,
            "compressed_count": 0,
            "skipped_count": 0,
            "failed_count": 0,
            "error": str(e),
            "details": []
        }
        


# ============ 使用示例 ============
if __name__ == "__main__":
    import sys
    sys.path.append(".")
    
    async def main():
        summarize_model = ChatOpenAI(
        model=os.getenv("SUMMARIZATION_MODEL", "qwen-turbo"),
        openai_api_key=os.getenv("DASHSCOPE_API_KEY"),
        openai_api_base=os.getenv("BASE_URL"),
        temperature=0.5#总结模型温度，控制总结对话的随机性，0-1之间，0越确定，1越随机
        )
        
        results = await run_compression_task(summarize_model)
        print(results)
        print("\n压缩结果:")
        for detail in results["details"]:
            if detail.get("success"):
                filter_count = len(detail.get("filtered_message_ids", []))
                print(f"  ✅ 用户 {detail['user_id']}, 会话 {detail['thread_id'][:20]}...: "
                      f"压缩 {detail['message_count']} 条消息 (过滤 {filter_count} 条) | {detail['token_count']} tokens")
            elif "未超过阈值" in detail.get("reason", ""):
                print(f"  ⏭️  用户 {detail['user_id']}, 会话 {detail['thread_id'][:20]}...: "
                      f"跳过 ({detail['token_count']} tokens)")
    
    asyncio.run(main())