from documents_process import TempDocumentProcessor, DocumentProcessor
from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks
from typing import List, Dict
from schemas import DocumentResponse
import logging

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/process_document", response_model=DocumentResponse)
async def file_upload(
    file: UploadFile = File(...), background_tasks: BackgroundTasks = None
):
    """上传文档并进行embedding（使用临时文件，处理完自动删除）"""
    try:
        filename = file.filename
        file_lower = filename.lower()
        #文件类型验证
        if not file_lower.endswith((".pdf",".docx",".doc")):
            raise HTTPException(status_code=400,detail="仅支持PDF和Word文档")

        content = await file.read()
        
        max_size = 200 * 1024 * 1024 # 200MB
        if len(content) > max_size:
            raise HTTPException(status_code=400, detail=f"文件过大，最大支持 {max_size // (1024*1024)}MB")
        
        temp_processor = TempDocumentProcessor
        #创建临时文件
        temp_file_path =  temp_processor.create_temp_file(filename=filename,content=content)
        
        #将文档处理任务添加到后台
        background_tasks.add_task(
            DocumentProcessor.process_document,
            temp_file_path = temp_file_path,
            filename = filename
        )
        
        #立即返回
        return DocumentResponse(
            filename = filename,
            message = f"文档已上传，正在后台处理中。"
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.info(f"文件上传错误，原因{str(e)}")
    