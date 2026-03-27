from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import config
from typing import List, Dict
import os
from pathlib import Path
import uuid
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class TempDocumentProcessor:
    """临时文档处理器：负责从上传的文件中提取文本并保存到临时文件"""

    def __init__(self):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        self.temp_dir = current_dir / "temp" / "doc_processing"
        self.temp_dir.mkdir(parents=True, exist_ok=True)
    
    def create_temp_file(self,filename:str,content:bytes)->Path:
        uuid_str = str(uuid.uuid4())
        temp_file_path = self.temp_dir / f"{uuid_str}_{filename}"
        with open(temp_file_path, "wb") as f:
            f.write(content)
        return temp_file_path
    
    def delete_temp_file(self, temp_file_path: Path):
        try:
            temp_file_path.unlink()
            logger.info(f"临时文件 {temp_file_path} 已删除")
        except Exception as e:
            logger.error(f"删除临时文件 {temp_file_path} 时出错: {e}")  
    

class DocumentProcessor:
    """文档加载和分片服务"""

    def __init__(self):
        self.parent_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.parent_chunk_size,
            chunk_overlap=config.parent_chunk_overlap,
            add_start_index=True,
            separators=config.separators,
        )
        self.child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.child_chunk_size,
            chunk_overlap=config.child_chunk_overlap,
            add_start_index=True,
            separators=config.separators,
        )
        # 从环境变量获取临时文件目录
        self.temp_dir = os.getenv("TEMP_DIR", "/tmp")
        
    def process_document(self, temp_file_path: Path, filename: str) -> List[Dict]:
        """
        处理文档：提取文本、分块
        :param temp_file_path: 临时文件路径
        :param filename: 文件名
        :return: 分块列表
        """
       