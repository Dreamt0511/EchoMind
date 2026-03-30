from documents_process import TempDocumentProcessor, DocumentProcessor, rerank_documents
from hash_storage import HashStorage
from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks, Request
from typing import List, Dict, Optional
from schemas import DocumentUploadResponse, DocumentDeleteResponse, DocumentRetrievalResponse
from postgresql_client import PostgreSQLParentClient
from milvus_client import AsyncMilvusClientWrapper
import logging
import sys
import asyncio
import hashlib
import uuid
import aiofiles
import os

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s-%(name)s-%(levelname)s-%(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
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
    background_tasks: BackgroundTasks = None
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
                    status_code=413, detail=f"文件过大，最大支持 {MAX_FILE_SIZE // (1024*1024)}MB")

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
                                status_code=400, detail="客户端已断开连接")

                        chunk = await file.read(CHUNK_SIZE)
                        if not chunk:
                            break

                        total_size += len(chunk)
                        # 实时检查文件大小
                        if total_size > MAX_FILE_SIZE:
                            raise HTTPException(
                                status_code=413, detail=f"文件过大，最大支持 {MAX_FILE_SIZE // (1024*1024)}MB")

                        sha256.update(chunk)
                        await temp_file.write(chunk)

                    file_hash = sha256.hexdigest()
                    logger.info(
                        f"文件上传完成: {filename}, 哈希: {file_hash[:16]}, 大小: {total_size} bytes")

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
                        #异步删除
                        await temp_processor.delete_temp_file(temp_file_path)
                    except:
                        pass
                raise HTTPException(status_code=400, detail="客户端已断开连接")

            #异步调用检查文件是否重复
            if await hash_storage.is_file_duplicate(file_hash):
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

            # 将文档处理任务添加到后台（处理完成后会自动清理临时文件）
            #process_document 需要是异步函数
            background_tasks.add_task(
                document_instance.process_document,
                temp_file_path=temp_file_path,
                filename=filename,
                file_hash=file_hash,
                knowledge_base_id=knowledge_base_id
            )

            # 立即返回
            return DocumentUploadResponse(
                filename=filename,
                message="文档已上传，正在后台处理中",
                file_hash=file_hash,
                knowledge_base_id=knowledge_base_id,
                is_duplicate=False
            )

        except HTTPException:
            raise
        except asyncio.CancelledError:
            logger.warning(
                f"请求被取消: {filename if 'filename' in locals() else 'unknown'}")
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


@router.delete("/documents", response_model=DocumentDeleteResponse)
async def delete_document(file_hash: str, knowledge_base_id: str):
    deleted_parent_count = 0
    deleted_child_count = 0

    try:
        # 删除 PostgreSQL中的父块
        async with PostgreSQLParentClient() as postgresql_client:
            deleted_parent_count = await postgresql_client.delete_all_file(
                knowledge_base_id, file_hash
            )

        # 删除 Milvus中的子块
        async with AsyncMilvusClientWrapper(hash_storage=hash_storage) as milvus_client:
            deleted_child_count = await milvus_client.delete_file_by_hash(
                knowledge_base_id, file_hash
            )
        
    except Exception as e:
        logger.error(f"删除失败: {e}")
        raise HTTPException(status_code=500, detail=f"删除失败: {str(e)}")

    return DocumentDeleteResponse(
        message=f"成功删除 {deleted_parent_count} 个父块和 {deleted_child_count} 个子块",
        knowledge_base_id=knowledge_base_id,
    )


@router.get("/documents/retrieval", response_model=DocumentRetrievalResponse)
async def retrieval_document(query: str, knowledge_base_id: Optional[str] = None, top_k: int = 10):
    """检索文档"""
    async with AsyncMilvusClientWrapper(hash_storage=hash_storage) as milvus_client:
        parent_chunkId_list = await milvus_client.hybrid_retrieval(
            query, knowledge_base_id, top_k)

        # 从 PostgreSQL获取父块
        async with PostgreSQLParentClient() as postgresql_client:
            parent_documents = await postgresql_client.get_parents(parent_chunkId_list)
            # 提取父块文本列表
            text_list = [doc.text for doc in parent_documents]
            # 重排序父块
            rerank_result = await rerank_documents(query, text_list, top_k)

            related_documents = []
            if not rerank_result:
                # 重排序失败的情况下降级取RRF融合后的前10个片段
                related_documents = parent_documents[:top_k]
            else:
                for item in rerank_result['output']['results']:
                    related_documents.append(item['document']['text'])


                related_documents = related_documents[:top_k]

    return DocumentRetrievalResponse(parent_documents=related_documents)

"""
rerank_result的结构示例：
{'output': {'results': [{'document': {'text': '20 世纪80 年代末'}, 'index': 0, 
'relevance_score': 0.886919463597282}]}, 'usage': {'total_tokens': 1224}, 
'request_id': '2252323b-a3ba-4ef5-a203-e305b64249e1'}  
"""


