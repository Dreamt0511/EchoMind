from documents_process import TempDocumentProcessor, DocumentProcessor, HashStorage
from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks, Request
from typing import List, Dict
from schemas import DocumentUploadResponse
import logging
import sys
import asyncio
import hashlib
import uuid
import aiofiles

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
CHUNK_SIZE = 256 * 1024  # 256kb
MAX_CONCURRENT_UPLOADS = 10

# 全局实例
hash_storage = HashStorage()  # 修复：改为正确的变量名
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

            # 修复：正确调用 uuid.uuid4()
            temp_file_path = temp_processor.temp_dir / \
                f"{uuid.uuid4()}_{filename}"

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
                        temp_file_path.unlink()
                    except:
                        pass
                raise

            # 再次检查客户端连接状态
            if await request.is_disconnected():
                if temp_file_path and temp_file_path.exists():
                    try:
                        temp_file_path.unlink()
                    except:
                        pass
                raise HTTPException(status_code=400, detail="客户端已断开连接")

            # 检查文件是否重复
            if hash_storage.is_file_duplicate(file_hash):
                logger.info(f"文件已存在，跳过处理: {filename}")
                if temp_file_path and temp_file_path.exists():
                    try:
                        temp_file_path.unlink()
                    except:
                        pass

                return DocumentUploadResponse(
                    filename=filename,
                    message="文件已存在，无需重复上传",
                    file_hash=file_hash,
                    knowledge_base_id=knowledge_base_id
                    is_duplicate=True
                )

            # 确保 background_tasks 不为 None
            if background_tasks is None:
                background_tasks = BackgroundTasks()

            document_instance = DocumentProcessor(hash_storage)

            # 将文档处理任务添加到后台（处理完成后会自动清理临时文件）
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
                knowledge_base_id=knowledge_base_id
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
                    temp_file_path.unlink()
                except:
                    pass
            raise HTTPException(status_code=500, detail=f"文件处理失败: {str(e)}")
