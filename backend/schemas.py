from pydantic import BaseModel
from typing import Optional, List, Dict,Union
from dataclasses import dataclass
from datetime import datetime


class DocumentUploadResponse(BaseModel):
    "文件上传响应模型"
    filename: str
    message: str
    file_hash: str
    knowledge_base_id: str#删除指定知识库的文件时可以通过知识库ID和文件哈希来删除
    is_duplicate: bool

class FileItem(BaseModel):
    """文件信息"""
    file_hash: str
    file_name: str
    uploaded_at: datetime

class GetKnowledgeBaseFilesResponse(BaseModel):
    """获取知识库文件列表响应模型"""
    status: str
    message: str
    count: int
    files: List[FileItem]

class DocumentDeleteResponse(BaseModel):
    "文件删除响应模型"
    message: str
    status: str
    knowledge_base_id : str
    parent_chunks_deleted : int
    child_chunks_deleted : int


class RerankDocumentItem(BaseModel):
    "文档重新排序项模型"
    text: str
    relevance_score: Optional[float] = None

class CreateKnowledgeBaseResponse(BaseModel):
    "创建知识库响应模型"
    knowledge_base_id: str
    status: str
    message: str

class DeleteKnowledgeBaseResponse(BaseModel):
    "删除知识库响应模型"
    knowledge_base_id: str
    status: str
    message: str
    child_chunks_deleted : int

class GetKnowledgeBaseResponse(BaseModel):
    status: str
    message: str
    count: int
    knowledge_bases: List[str]#知识库ID列表

#上下文模型
@dataclass
class ContextSchema:
    """知识库检索上下文模型"""
    user_id: int
    knowledge_base_id: str
    top_k: int = 5

