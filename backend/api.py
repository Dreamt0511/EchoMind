from documents_process import TempDocumentProcessor, DocumentProcessor
from hash_storage import HashStorage
from fastapi import (
    APIRouter,
    UploadFile,
    File,
    HTTPException,
    BackgroundTasks,
    Request,
    status,
)
from typing import List, Dict, Optional
from postgresql_client import get_postgresql_client
from milvus_client import get_milvus_client
import logging
import sys
import asyncio
import hashlib
import uuid
import aiofiles
import os
from fastapi.responses import StreamingResponse
from agent import stream_agent_response
from schemas import (
    DocumentUploadResponse,
    DocumentDeleteResponse,
    CreateKnowledgeBaseResponse,
)


# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s-%(name)s-%(levelname)s-%(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

router = APIRouter()

# 配置常量
MAX_FILE_SIZE = 200 * 1024 * 1024  # 200mb
CHUNK_SIZE = 1024 * 1024  # 1mb
MAX_CONCURRENT_UPLOADS = 10


# 全局实例
hash_storage = HashStorage()  # 异步版本


upload_semaphore = asyncio.Semaphore(MAX_CONCURRENT_UPLOADS)


@router.post("/document_upload", response_model=DocumentUploadResponse)
async def file_upload(
    request: Request,
    file: UploadFile = File(...),
    knowledge_base_id: str = File(...),
    user_id: int = File(...),  # 新增 user_id 参数
    background_tasks: BackgroundTasks = None,
):
    """上传文档并进行embedding,支持文件去重"""

    # 初始化变量，避免作用域问题
    temp_file_path = None

    async with upload_semaphore:
        try:
            filename = file.filename
            logger.info(f"文件名: {filename}")
            file_lower = filename.lower()

            # 文件类型验证
            if not file_lower.endswith((".pdf", ".docx", ".doc")):
                raise HTTPException(status_code=400, detail="仅支持PDF和Word文档")

            # 修复：检查文件大小，处理 file.size 为 None 的情况
            if file.size is not None and file.size > MAX_FILE_SIZE:
                raise HTTPException(
                    status_code=413,
                    detail=f"文件过大，最大支持 {MAX_FILE_SIZE // (1024*1024)}MB",
                )

            # 检查客户端连接状态
            if await request.is_disconnected():
                raise HTTPException(status_code=400, detail="客户端已断开连接")

            # 创建哈希对象和临时文件处理对象
            sha256 = hashlib.sha256()
            temp_processor = TempDocumentProcessor()

            temp_file_path = temp_processor.temp_dir / f"{uuid.uuid4()}_{filename}"

            # 流式计算哈希，写入文件
            try:
                async with aiofiles.open(temp_file_path, "wb") as temp_file:
                    total_size = 0
                    while True:
                        # 检查客户端连接状态
                        if await request.is_disconnected():
                            raise HTTPException(
                                status_code=400, detail="客户端已断开连接"
                            )

                        chunk = await file.read(CHUNK_SIZE)
                        if not chunk:
                            break

                        total_size += len(chunk)
                        # 实时检查文件大小
                        if total_size > MAX_FILE_SIZE:
                            raise HTTPException(
                                status_code=413,
                                detail=f"文件过大，最大支持 {MAX_FILE_SIZE // (1024*1024)}MB",
                            )

                        sha256.update(chunk)
                        await temp_file.write(chunk)

                    file_hash = sha256.hexdigest()
                    logger.info(
                        f"文件上传完成: {filename}, 哈希: {file_hash[:16]}, 大小: {total_size} bytes"
                    )

            except Exception as e:
                # 异常时清理临时文件
                if temp_file_path and temp_file_path.exists():
                    try:
                        # ✅ 改为异步删除
                        await temp_processor.delete_temp_file(temp_file_path)
                    except:
                        pass
                raise

            # 再次检查客户端连接状态
            if await request.is_disconnected():
                if temp_file_path and temp_file_path.exists():
                    try:
                        # 异步删除
                        await temp_processor.delete_temp_file(temp_file_path)
                    except:
                        pass
                raise HTTPException(status_code=400, detail="客户端已断开连接")

            # 异步调用检查文件是否重复（传入 user_id）
            if await hash_storage.is_file_duplicate(
                file_hash, knowledge_base_id, user_id
            ):
                logger.info(f"文件已存在，跳过处理: {filename}")
                if temp_file_path and temp_file_path.exists():
                    try:
                        # 异步删除
                        await temp_processor.delete_temp_file(temp_file_path)
                    except:
                        pass

                return DocumentUploadResponse(
                    filename=filename,
                    message="文件已存在，无需重复上传",
                    file_hash=file_hash,
                    knowledge_base_id=knowledge_base_id,
                    is_duplicate=True,
                )

            # 确保 background_tasks 不为 None
            if background_tasks is None:
                background_tasks = BackgroundTasks()

            # DocumentProcessor 需要改为异步版本
            document_instance = DocumentProcessor(hash_storage)

            # 将文档处理任务添加到后台（处理完成后会自动清理临时文件，传入 user_id）
            # process_document 需要是异步函数
            background_tasks.add_task(
                document_instance.process_document,
                temp_file_path=temp_file_path,
                filename=filename,
                file_hash=file_hash,
                knowledge_base_id=knowledge_base_id,
                user_id=user_id,  # 新增
            )

            # 立即返回
            return DocumentUploadResponse(
                filename=filename,
                message="文档已上传，正在后台处理中",
                file_hash=file_hash,
                knowledge_base_id=knowledge_base_id,
                is_duplicate=False,
            )

        except HTTPException:
            raise
        except asyncio.CancelledError:
            logger.warning(
                f"请求被取消: {filename if 'filename' in locals() else 'unknown'}"
            )
            raise HTTPException(status_code=499, detail="客户端已取消请求")
        except Exception as e:
            logger.error(f"文件上传错误: {str(e)}", exc_info=True)
            # 清理临时文件
            if temp_file_path and temp_file_path.exists():
                try:
                    await temp_processor.delete_temp_file(temp_file_path)
                except:
                    pass
            raise HTTPException(status_code=500, detail=f"文件处理失败: {str(e)}")


@router.post("/knowledge-bases")
async def create_knowledge_base(
    knowledge_base_id: str,
    user_id: int
):
    """创建知识库"""
    postgresql_client = await get_postgresql_client()
    result = await postgresql_client.create_knowledge_base(knowledge_base_id, user_id)
    
    if not result["success"]:
        if "已存在" in result["message"]:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=result["message"]
            )
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=result["message"]
            )
    
    return {
        "status": "success",
        "message": result["message"],
        "knowledge_base_id": result["knowledge_base_id"]
    }
    
@router.delete("/knowledge-bases/{knowledge_base_id}")
async def delete_knowledge_base(
    knowledge_base_id: str,
    user_id: int
):
    """删除知识库"""
    postgresql_client = await get_postgresql_client()
    result = await postgresql_client.delete_knowledge_base(knowledge_base_id, user_id)
    
    if not result["success"]:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=result["message"]
        )
    
    return {
        "status": "success",
        "message": result["message"],
        "files_deleted": result["files_deleted"]
    }

@router.get("/knowledge-bases")
async def get_user_knowledge_bases(
    user_id: int
):
    """获取用户的所有知识库"""
    postgresql_client = await get_postgresql_client()
    result = await postgresql_client.get_user_knowledge_bases(user_id)
    
    if not result["success"]:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=result["message"]
        )
    
    return {
        "status": "success",
        "message": result["message"],
        "knowledge_bases": result["knowledge_bases"],
        "count": result["count"]
    }

@router.delete("/documents", response_model=DocumentDeleteResponse)
async def delete_file(
    file_hash: str, knowledge_base_id: str, user_id: int
):  # 新增 user_id
    deleted_parent_count = 0
    deleted_child_count = 0

    try:
        # 删除 PostgreSQL中的父块 - 使用全局客户端（传入 user_id）
        postgresql_client = await get_postgresql_client()
        deleted_parent_count = await postgresql_client.delete_file_by_hash(
            knowledge_base_id=knowledge_base_id,
            file_hash=file_hash,
            user_id=user_id,  # 新增
        )

        # 删除 Milvus中的子块 - 使用全局客户端（传入 user_id）
        milvus_client = await get_milvus_client()
        deleted_child_count = await milvus_client.delete_file_by_hash(
            knowledge_base_id=knowledge_base_id,
            file_hash=file_hash,
            user_id=user_id,  # 新增
        )

        # 块哈希的删除由文件删除时自动触发（ON DELETE CASCADE），无需手动处理

    except Exception as e:
        logger.error(f"删除失败: {e}")
        raise HTTPException(status_code=500, detail=f"删除失败: {str(e)}")

    return DocumentDeleteResponse(
        message=f"成功删除 {deleted_parent_count} 个父块和 {deleted_child_count} 个子块",
        knowledge_base_id=knowledge_base_id,
    )


@router.get("/chat_with_agent/stream")
async def chat_with_agent(
    query: str, knowledge_base_id: str, user_id: int, top_k: int = 5
):  # 新增 user_id
    print("当前知识库ID:", knowledge_base_id)
    """
    流式返回 agent 响应
    """
    return StreamingResponse(
        stream_agent_response(
            user_message=query, knowledge_base_id=knowledge_base_id, user_id=user_id
        ),
        media_type="text/plain; charset=utf-8",
    )
