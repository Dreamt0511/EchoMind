from pydantic import BaseModel
from typing import Optional

class DocumentUploadResponse(BaseModel):
    "文件上传响应模型"
    filename: str
    message: str
    file_hash: str
    knowledge_base_id: str#删除指定知识库的文件时可以通过知识库ID和文件哈希来删除
    is_duplicate: bool