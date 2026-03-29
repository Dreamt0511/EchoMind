from langchain_community.document_loaders import Docx2txtLoader, PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from milvus_client import AsyncMilvusClientWrapper
import config
from typing import List, Dict, Set
import os
from pathlib import Path
import uuid
import hashlib
import logging
from pathlib import Path
from postgresql_client import PostgreSQLParentClient
logger = logging.getLogger(__name__)
import asyncio
import requests
import json
from http import HTTPStatus

# 从独立文件导入 HashStorage（唯一新增代码）
from hash_storage import HashStorage


class TempDocumentProcessor:
    """临时文档处理器：负责从上传的文件中提取文本并保存到临时文件"""

    def __init__(self):
        current_dir = Path(__file__).parent
        self.temp_dir = current_dir / "temp" / "doc_processing"
        self.temp_dir.mkdir(parents=True, exist_ok=True)

    def delete_temp_file(self, temp_file_path: Path):
        try:
            temp_file_path.unlink()
            logger.info(f"临时文件 {temp_file_path} 已删除")
        except Exception as e:
            logger.error(f"删除临时文件 {temp_file_path} 时出错: {e}")


class DocumentProcessor:
    """文档加载和分片服务"""

    def __init__(self, hash_storage: HashStorage = None):
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
        # 确保哈希存储可用
        self.hash_storage = hash_storage
        self.milvus_client = AsyncMilvusClientWrapper(hash_storage)

    async def process_document(self,
                         temp_file_path: Path,
                         filename: str,
                         file_hash: str,
                         knowledge_base_id: str
                         ) -> List[Dict]:
        """
        处理文档：提取文本、分块
        :param temp_file_path: 临时文件路径
        :param filename: 文件名
        :param knowledge_base_id 知识库id
        :return: 分块列表
        """
        logger.info(f"{'='*20} 开始进行文本分块 {filename} {'='*20}")

        loader_mapping = {
            ".pdf": PyMuPDFLoader,
            ".doc": Docx2txtLoader,
            ".docx": Docx2txtLoader
        }
        # 获取对应的加载器类
        loader_class = None
        file_lower = filename.lower()
        for ends, loader in loader_mapping.items():
            if file_lower.endswith(ends):
                loader_class = loader
                break

        if not loader_class:
            raise ValueError(f"不支持的文件类型: {filename}")

        # 加载文档
        loader = loader_class(temp_file_path)
        documents = loader.load()

        if not documents:
            logger.warning(f"文件 {filename} 加载后无内容")
            return

        # 分割成父块
        parent_chunks = self.parent_splitter.split_documents(documents)

        all_chunks_with_ids = []

        # 准备批量插入postgresql数据
        parent_data_list = []

        for parent_index, parent_chunk in enumerate(parent_chunks):
            # 为每个父块生成id
            parent_chunk_id = str(uuid.uuid4())

            #添加父块元数据
            parent_chunk.metadata.update({
                "file_name": filename,
                "file_hash": file_hash,
            })

            #记录当前父块数据
            parent_data_list.append({
                "parent_id": parent_chunk_id,
                "knowledge_base_id": knowledge_base_id,
                "text": parent_chunk.page_content,
                "metadata": parent_chunk.metadata
            })
            
            # 再次分割成子块
            child_chunks = self.child_splitter.split_documents([parent_chunk])

            for child_index, child_chunk in enumerate(child_chunks):
                child_chunk_id = str(uuid.uuid4())
                # 生成子块的哈希对比是否和记录的哈希重复
                child_chunk_hash = hashlib.sha256(
                    child_chunk.page_content.encode()).hexdigest()

                if self.hash_storage.is_chunk_duplicate(child_chunk_hash):
                    logger.debug(f"子块已存在，跳过: {child_chunk_hash[:16]}...")
                    continue

                # 添加子块元数据
                child_chunk.metadata.update({
                    "parent_id": parent_chunk_id,  # 父块ID
                    "parent_index": parent_index,
                    "chunk_index": child_index,
                    "file_name": filename,
                    "child_chunk_hash":child_chunk_hash,#子块哈希用于删除hashStorge中的记录
                    "file_hash": file_hash,#文件哈希，用于定位文件删除关联子块
                    "knowledge_base_id": knowledge_base_id
                })

                all_chunks_with_ids.append((child_chunk_id, child_chunk))

                # 记录子块哈希(存到本地文件)
                self.hash_storage.add_chunk_hash(child_chunk_hash)

        # 批量插入子块到milvus数据库
        if all_chunks_with_ids:
            await self.milvus_client.add_chunks_batch(
                knowledge_base_id=knowledge_base_id,
                chunks_with_ids=all_chunks_with_ids
            )

        # 批量插入 PostgreSQL（一次连接）
        async with PostgreSQLParentClient(...) as postgresql_client:
            await postgresql_client.add_parents_batch(parent_data_list)


        logger.info(
            f"完成处理: 文件={filename}, 父块数={len(parent_chunks)}, 子块数={len(all_chunks_with_ids)}")

        # 记录文件哈希
        self.hash_storage.add_file_hash(file_hash)

        #清理临时文件
        tempProcessor = TempDocumentProcessor()
        tempProcessor.delete_temp_file(temp_file_path)


async def rerank_documents(query: str, documents: list, top_n=5):
    """
    一个通用的重排序API调用模板
    """
    rerank_model = os.getenv('RERANK_MODEL', 'qwen3-rerank')
    api_key = os.getenv('DASHSCOPE_API_KEY')
    if not api_key:
        print("错误: 请先在环境变量中设置 DASHSCOPE_API_KEY")
        return None
        
    # 官方文档中的API地址和模型名称
    url = os.getenv('RERANK_URL')
    
    # 请求头
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    # 请求体
    payload = payload = {
        "model": rerank_model,
        "input": {  # 必须包在 input 里面！
            "query": query,
            "documents": documents
        },
        "parameters": {  # 必须包在 parameters 里面！
            "top_n": top_n,
            "return_documents": True
        }
    }
    # 发送请求
    try:
        response = requests.post(url, headers=headers, json=payload)
        if response.status_code == HTTPStatus.OK:
            return response.json()
        else:
            logger.error(f"API 请求失败: {response.status_code} - {response.text}")
            return False
    except Exception as e:
        logger.error(f"请求发生异常: {e}")
        raise False