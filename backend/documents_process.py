from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import config
from typing import List, Dict,Set
import os
from pathlib import Path
import uuid
import logging
from pathlib import Path
logger = logging.getLogger(__name__)

class HashStorage:
    """哈希存储管理器,只存储文件哈希和分块哈希"""
    
    def __init__(self):
        #创建哈希目录
        current_dir = Path(__file__).parent
        hash_dir = current_dir / "hash_storage"
        hash_dir.mkdir(parents=True,exist_ok=True)
        #创建文件和分块哈希记录文件
        self.files_hash_path = hash_dir / "files_hash.txt"
        self.chunks_hash_path = hash_dir / "chunks_hash.txt"
        
        #加载现有哈希值
        self.files_hash = self._load_hashes(self.files_hash_path)
        self.chunks_hash = self._load_hashes(self.chunks_hash_path)
        
    #哈希文件加载与保存============
    def _load_hashes(self,file_path: Path) -> Set[str]:
        """从文本文件加载哈希集合"""
        hashes = set()
        if file_path.exists():
            try:
                with open(file_path,"r",encoding="utf-8") as f:
                    for line in f:
                        hash_value = line.strip()
                        if hash_value:
                            hashes.add(hash_value)
            except Exception as e:
                logger.error(f"加载哈希文件{Path(file_path).name}失败,原因{str(e)}")
        return hashes       
            
    
    def _save_hashes(self,file_path: Path,hashes:Set[str]):
        """保存哈希值到文件"""
        try:
            #采用清空写入而不是追加是因为要保存chunks_hash的时候就是全部写入所有哈希
            with open(file_path,"w",encoding="utf-8") as f:
                for hash_value in hashes:
                    f.write(f"{hash_value}\n")
        except Exception as e:
            logger.error(f"保存哈希值失败，原因{str(e)}")
            
    #文件哈希管理============
    def is_file_duplicate(self,file_hash:str)->bool:
        """检查文件哈希是否已存在"""
        return file_hash in self.files_hash
    
    def add_file_hash(self,file_hash:str):
        """添加文件哈希值"""
        if not self.is_file_duplicate(file_hash):
            self.files_hash.add(file_hash)
            self._save_hashes(self.files_hash_path,self.files_hash)
            logger.info(f"添加文件哈希: {file_hash[:16]}...")

    #分块哈希管理
    def is_chunk_duplicate(self,chunk_hash:str)->bool:
        """检查分块哈希是否已存在"""
        return chunk_hash in self.chunks_hash
    
    def add_chunk_hash(self,chunk_hash:str):
        if not self.is_chunk_duplicate(chunk_hash):
            self.chunks_hash.add(chunk_hash)
            self._save_hashes(self.chunks_hash_path,self.chunks_hash)


    
class TempDocumentProcessor:
    """临时文档处理器：负责从上传的文件中提取文本并保存到临时文件"""
    def __init__(self):
        current_dir = Path(__file__).parent
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

    def __init__(self,hash_storage: HashStorage = None):
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
        
    def process_document(self, temp_file_path: Path, filename: str,file_hash :str) -> List[Dict]:
        """
        处理文档：提取文本、分块
        :param temp_file_path: 临时文件路径
        :param filename: 文件名
        :return: 分块列表
        """
        pass
       