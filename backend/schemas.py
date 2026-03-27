from pydantic import BaseModel
from typing import Optional

class DocumentUploadResponse(BaseModel):
    "文件上传响应模型"
    filename: str
    message: str