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
TOKEN_THRESHOLD = 500 #由于中间件压缩的是4000token但是这里压缩的只提取了human提问和ai回答，所以阈值设为2000
MEMORY_EXTRACT_PROMPT = config.MEMORY_EXTRACT_PROMPT

async def extract_memories(
    messages: List[Dict[str, Any]],
    model: BaseChatModel,
) -> Dict[str, Any]:
    """
    从对话中提取多种类型的记忆（摘要、语义记忆、情景记忆、程序记忆、用户画像），
    并返回过滤的消息ID列表和重要性评分

    Args:
        messages: 原始消息列表（包含id/role/content）
        model: 用于提取记忆的 LLM 模型

    Returns:
        dict: 包含summary（摘要文本）、semantic_memory（语义记忆列表）、
              episodic_memory（情景记忆列表）、procedural_memory（程序记忆列表）、
              user_profile（用户画像）、filtered_message_ids（过滤的消息ID列表）、
              importance_score（重要性评分0-1）
    """
    if not messages:
        return {
            "summary": "",
            "semantic_memory": [],
            "episodic_memory": [],
            "procedural_memory": [],
            "user_profile": "",
            "filtered_message_ids": [],
            "importance_score": 0.0
        }

    # 格式化对话文本（包含message_id）
    formatted_lines = []
    for msg in messages:
        msg_id = msg.get("id", "")
        role = msg.get("role", "").lower()
        content = msg.get("content", "")
        formatted_lines.append(f"{msg_id} | {role} | {content}")
    conversation_text = "\n\n".join(formatted_lines)

    try:
        # 调用模型生成记忆提取结果
        response = await model.ainvoke(
            [
                HumanMessage(
                    content=MEMORY_EXTRACT_PROMPT.format(
                        conversation_text=conversation_text
                    )
                )
            ]
        )
        response_content = response.content.strip()

        # 清理可能的Markdown代码块标记
        if response_content.startswith("```json"):
            response_content = response_content[7:]
        if response_content.startswith("```"):
            response_content = response_content[3:]
        if response_content.endswith("```"):
            response_content = response_content[:-3]
        response_content = response_content.strip()

        # 解析JSON响应
        result = json.loads(response_content)

        # 提取各字段，提供默认值
        summary = result.get("summary", "").strip()
        semantic_memory = result.get("semantic_memory", [])
        episodic_memory = result.get("episodic_memory", [])
        procedural_memory = result.get("procedural_memory", [])
        user_profile = result.get("user_profile", "").strip()
        filtered_message_ids = result.get("filtered_message_ids", [])
        importance_score = result.get("importance_score", 0.5)

        # 确保列表类型
        if not isinstance(semantic_memory, list):
            semantic_memory = []
        if not isinstance(episodic_memory, list):
            episodic_memory = []
        if not isinstance(procedural_memory, list):
            procedural_memory = []
        if not isinstance(filtered_message_ids, list):
            filtered_message_ids = []

        # 确保字符串类型
        if not isinstance(user_profile, str):
            user_profile = ""

        # 确保重要性评分在有效范围内
        try:
            importance_score = float(importance_score)
            importance_score = max(0.0, min(1.0, importance_score))
        except (ValueError, TypeError):
            logger.warning(f"无效的重要性评分: {importance_score}，使用默认值0.5")
            importance_score = 0.5

        # 验证过滤ID的合法性（必须是后半部分、字符串、存在于原始消息中）
        total_count = len(messages)
        half_index = total_count // 2
        valid_filtered_ids = []
        all_message_ids = [str(msg["id"]) for msg in messages]

        for msg_id in filtered_message_ids:
            str_msg_id = str(msg_id)
            # 检查ID是否存在 + 是否在后半部分
            if str_msg_id in all_message_ids:
                msg_index = all_message_ids.index(str_msg_id)
                if msg_index >= half_index:
                    valid_filtered_ids.append(str_msg_id)

        logger.info(
            f"记忆提取完成: 摘要 ({len(summary)} 字符)，"
            f"语义记忆 {len(semantic_memory)} 条，"
            f"情景记忆 {len(episodic_memory)} 条，"
            f"程序记忆 {len(procedural_memory)} 条，"
            f"用户画像 ({len(user_profile)} 字符)，"
            f"过滤 {len(valid_filtered_ids)} 条消息，"
            f"重要性评分: {importance_score:.2f}"
        )

        return {
            "summary": summary,
            "semantic_memory": semantic_memory,
            "episodic_memory": episodic_memory,
            "procedural_memory": procedural_memory,
            "user_profile": user_profile,
            "filtered_message_ids": valid_filtered_ids,
            "importance_score": importance_score
        }

    except json.JSONDecodeError as e:
        logger.error(f"解析大模型JSON响应失败: {e}, 响应内容: {response_content}")
        # 降级处理：返回基础摘要，空记忆列表，默认重要性0.3
        fallback_formatted = []
        for msg in messages:
            fallback_formatted.append(f"{msg.get('role', '').lower()}: {msg.get('content', '')}")
        fallback_summary = f"Previous conversation summary:\n{' '.join(fallback_formatted)[:2000]}"
        return {
            "summary": fallback_summary,
            "semantic_memory": [],
            "episodic_memory": [],
            "procedural_memory": [],
            "user_profile": "",
            "filtered_message_ids": [],
            "importance_score": 0.3
        }
    except Exception as e:
        logger.error(f"记忆提取失败: {e}", exc_info=True)
        return {
            "summary": "",
            "semantic_memory": [],
            "episodic_memory": [],
            "procedural_memory": [],
            "user_profile": "",
            "filtered_message_ids": [],
            "importance_score": 0.0
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


async def update_messages_with_summary_id(message_ids: List[str], summary_id: str) -> bool:
    """更新被压缩的消息的 summary_id（后台任务，不抛出异常）"""
    try:
        pg_client = await get_postgresql_client()
        
        if not pg_client.pool:
            logger.error("数据库连接池未初始化，无法更新 summary_id")
            raise Exception("数据库连接池未初始化")
        
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
        logger.error(f"更新消息的 summary_id 失败: {e}", exc_info=True)
        raise Exception(e)


#这个才是存入记忆的关键
async def compress_and_summarize_conversation(
    messages: List[Dict[str, Any]],
    model: BaseChatModel,
    user_id: int,
    thread_id: str
) -> Dict[str, Any]:
    """压缩单个会话的消息并生成摘要，更新数据库（psql中加上summary_id字段）"""
    if not messages:
        return {"success": False, "reason": "没有消息需要压缩"}
    
    # 1. 计算 token 数（基于原始消息内容）
    conversation_text = "\n\n".join([f"{m.get('role')}: {m.get('content')}" for m in messages])
    total_tokens = count_tokens_approximately([HumanMessage(content=conversation_text)])
    
    logger.info(f"会话 {thread_id} 共 {len(messages)} 条消息，总 token 数: {total_tokens}")
    
    # 2. 检查是否超过阈值
    if total_tokens <= TOKEN_THRESHOLD:
        return {
            "success": False,
            "reason": f"token 数未超过阈值 ({total_tokens} <= {TOKEN_THRESHOLD})",
            "token_count": total_tokens,
            "message_count": len(messages)
        }
    
    # 3. 提取记忆，
    try:
        extract_result = await extract_memories(messages, model)

        summary = extract_result["summary"]
        semantic_memory = extract_result["semantic_memory"]
        episodic_memory = extract_result["episodic_memory"]
        procedural_memory = extract_result["procedural_memory"]
        user_profile = extract_result["user_profile"]
        filtered_message_ids = extract_result["filtered_message_ids"]
        importance_score = extract_result["importance_score"]

        # 4. 生成摘要ID
        summary_id = str(uuid.uuid4())
        
        # 5. 确定需要更新summary_id的消息（排除过滤的消息）
        all_message_ids = [str(msg["id"]) for msg in messages]#所有消息id
        update_message_ids = [mid for mid in all_message_ids if mid not in filtered_message_ids]#需要更新的summary_id的消息id
        
        if not update_message_ids:
            logger.warning(f"会话 {thread_id} 所有消息都被过滤，无需更新数据库")
        try:#下面这个还要考虑原子性，避免只有一个数据库的数据更新成功，理论上要2个数据库都执行成功才行，不应该出现只有一个数据库执行成功的情况，因此还得做一些特殊处理
            # 6. 更新数据库（psql中加上summary_id字段）
            summary_success = await update_messages_with_summary_id(update_message_ids, summary_id)

            # 7. 更新Milvus数据库
            milvus_client = await get_milvus_client()
            milvus_success = await milvus_client.add_memory(
                memory_type="summary",
                user_id=user_id,
                thread_id=thread_id,
                content=summary,
                summary_id=summary_id,
                importance=importance_score,
            )
            if summary_success and milvus_success:
                logger.info(f"成功压缩会话 {thread_id}，生成摘要 {summary_id}，过滤 {len(filtered_message_ids)} 条消息")

                return {
                    "success": True,
                    "summary_id": summary_id,
                    "summary": summary,
                    "semantic_memory": semantic_memory,
                    "episodic_memory": episodic_memory,
                    "procedural_memory": procedural_memory,
                    "user_profile": user_profile,
                    "filtered_message_ids": filtered_message_ids,
                    "importance_score": importance_score,
                    "token_count": total_tokens,
                    "message_count": len(messages),
                    "updated_message_count": len(update_message_ids)
                }
        except Exception as e:
            return {
                "success": False,
                "reason": f"数据库更新失败:{str(e)}",
                "filtered_message_ids": filtered_message_ids,
                "token_count": total_tokens,
                "message_count": len(messages)
            }
    except Exception as e:
        logger.error(f"压缩会话 {thread_id} 失败: {e}", exc_info=True)
        return {
            "success": False,
            "reason": str(e),
            "filtered_message_ids": [],
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
        logger.info(results)
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
    logger.info("开始执行对话压缩任务...")
    start_time = datetime.now()
    
    try:
        pg_client = await get_postgresql_client()
        await pg_client.init_pool()
        
        # 兼容原参数（实际未使用user_id，保持接口一致）
        results = await process_all_users_conversations(model)
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        logger.info(f"压缩任务完成，耗时: {duration:.2f} 秒")
        
        return results
    except Exception as e:
        logger.error(f"压缩任务执行失败: {e}", exc_info=True)
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