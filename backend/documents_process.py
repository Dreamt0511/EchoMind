from hash_storage import HashStorage
import aiofiles.os
import aiofiles
from http import HTTPStatus
import json
import httpx
import asyncio
from langchain_community.document_loaders import Docx2txtLoader, PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from milvus_client import get_milvus_client
import config
from typing import List, Dict, Set
import os
from urllib.parse import unquote
from pathlib import Path
import uuid
import hashlib
import logging
from postgresql_client import get_postgresql_client

logger = logging.getLogger(__name__)


class TempDocumentProcessor:
    """临时文档处理器：负责从上传的文件中提取文本并保存到临时文件"""

    def __init__(self):
        current_dir = Path(__file__).parent
        self.temp_dir = current_dir / "temp" / "doc_processing"
        self.temp_dir.mkdir(parents=True, exist_ok=True)

    async def delete_temp_file(self, temp_file_path: Path):
        """异步删除临时文件"""
        try:
            await aiofiles.os.remove(temp_file_path)
            logger.info(f"临时文件 {temp_file_path} 已删除")
        except Exception as e:
            logger.error(f"删除临时文件 {temp_file_path} 时出错: {e}")


class DocumentProcessor:
    """文档加载和分片服务"""

    def __init__(self, hash_storage: HashStorage = None):
        self.parent_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.parent_chunk_size,
            chunk_overlap=config.parent_chunk_overlap,
            add_start_index=False,
            separators=config.separators,
        )
        self.child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.child_chunk_size,
            chunk_overlap=config.child_chunk_overlap,
            add_start_index=False,
            separators=config.separators,
        )
        self.temp_dir = os.getenv("TEMP_DIR", "/tmp")
        self.hash_storage = hash_storage
        # 修改：不再直接创建实例，延迟初始化
        self._milvus_client = None

    async def _get_milvus_client(self):
        """延迟获取 Milvus 客户端"""
        if self._milvus_client is None:
            self._milvus_client = await get_milvus_client()
        return self._milvus_client

    async def process_document(
        self,
        temp_file_path: Path,
        filename: str,
        file_hash: str,
        knowledge_base_id: str,
        user_id: int,
    ) -> List[Dict]:
        # 解码 URL 编码的文件名
        filename = unquote(filename)
        logger.info(f"{'='*20} 开始进行文本分块 {filename} {'='*20}")

        loader_mapping = {
            ".pdf": PyMuPDFLoader,
            ".doc": Docx2txtLoader,
            ".docx": Docx2txtLoader,
        }

        loader_class = None
        file_lower = filename.lower()
        for ends, loader in loader_mapping.items():
            if file_lower.endswith(ends):
                loader_class = loader
                break

        if not loader_class:
            raise ValueError(f"不支持的文件类型: {filename}")

        loop = asyncio.get_event_loop()
        loader = loader_class(temp_file_path)
        
        #父子块文档元数据增强
        def process_all_sync():
            """在一个线程中完成所有同步操作"""
            documents = loader.load()
            if not documents:
                return None, None

            parent_chunks = self.parent_splitter.split_documents(documents)
            if not parent_chunks:
                return None, None

            parent_data_list = []
            all_child_items = []  

            for parent_index, parent_chunk in enumerate(parent_chunks):
                parent_id = str(uuid.uuid4())

                parent_data_list.append(
                    {
                        "parent_id": parent_id,
                        "knowledge_base_id": knowledge_base_id,
                        "text": parent_chunk.page_content,
                        "file_name": filename,
                        "file_hash": file_hash,
                    }
                )

                child_chunks = self.child_splitter.split_documents([parent_chunk])

                for child_chunk in child_chunks:
                    child_hash = hashlib.sha256(
                        child_chunk.page_content.encode()
                    ).hexdigest()

                    child_chunk.metadata.update(
                        {
                            "parent_id": parent_id,
                            "parent_index": parent_index,
                            "file_name": filename,
                            "child_chunk_hash": child_hash,
                            "file_hash": file_hash,
                            "knowledge_base_id": knowledge_base_id,
                            "user_id": user_id,
                        }
                    )

                    all_child_items.append(child_chunk)

            return parent_data_list, all_child_items

        parent_data_list, all_child_items = await loop.run_in_executor(
            None, process_all_sync
        )

        logger.info(
            f"==========文本分块完成: 文件={filename}, 父块数={len(parent_data_list)}, 子块数={len(all_child_items)}=========="
        )

        if not parent_data_list:
            logger.warning(f"文件 {filename} 加载或分割后无内容")
            return

        # ========== 1. 先添加文件元数据（必须在所有外键引用之前） ==========
        await self.hash_storage.add_file_hash(
            file_hash, filename, knowledge_base_id, user_id
        )
        logger.info(f"==========添加文件元数据完成==========")

        # ========== 2. 批量检查所有哈希（排除当前文件的已有块哈希） ==========
        all_hashes = [
            hashlib.sha256(chunk.page_content.encode()).hexdigest()
            for chunk in all_child_items
        ]
        existing_hashes = await self.hash_storage.batch_check_duplicates(
            all_hashes, file_hash, knowledge_base_id, user_id
        )

        # ========== 3. 过滤出新子块（未被其他文件使用过的） ==========
        new_child_items = []
        new_hashes = []

        for child_chunk in all_child_items:
            child_hash = hashlib.sha256(child_chunk.page_content.encode()).hexdigest()
            if child_hash in existing_hashes:
                logger.debug(f"子块已被其他文件使用，跳过: {child_hash[:16]}...")
                continue

            new_child_items.append(child_chunk)
            new_hashes.append(child_hash)

        # ========== 4-6. 并行执行独立的数据库操作 ==========
        # 注意：批量添加新哈希依赖文件元数据（已存在），可以与 Milvus 和 PostgreSQL 并行
        tasks = []
        
        # 任务1：批量添加新哈希（如果有）
        if new_hashes:
            tasks.append(
                self.hash_storage.batch_add_chunk_hashes(
                    new_hashes, file_hash, knowledge_base_id, user_id
                )
            )
        
        # 任务2：批量插入 Milvus 数据库（只插入新子块）
        if new_child_items:
            milvus_client = await self._get_milvus_client()
            tasks.append(
                milvus_client.add_chunks_batch(
                    knowledge_base_id=knowledge_base_id,
                    chunks=new_child_items,
                    user_id=user_id,
                )
            )
        
        # 任务3：批量插入 PostgreSQL 数据库（父块）
        postgresql_client = await get_postgresql_client()
        tasks.append(
            postgresql_client.add_parent_chunk_batch(parent_data_list, user_id)
        )
        
        # 并行执行所有数据库操作
        if tasks:
            await asyncio.gather(*tasks)
        
        if new_hashes:
            logger.info(f"==========批量添加 {len(new_hashes)} 个新哈希==========")
        
        if new_child_items:
            logger.info(f"==========批量插入 {len(new_child_items)} 个子块到milvus完成==========")
        
        logger.info(f"==========批量插入 {len(parent_data_list)} 个父块到postgresql完成==========")

        # 清理临时文件
        tempProcessor = TempDocumentProcessor()
        await tempProcessor.delete_temp_file(temp_file_path)

        return [
            {
                "filename": filename,
                "parent_chunks": len(parent_data_list),
                "child_chunks": len(all_child_items),
                "new_chunks": len(new_child_items),
            }
        ]


async def rerank_documents(query: str, documents: list, top_n=5):
    """
    一个通用的重排序API调用模板，带重试机制
    """
    # 优化：限制输入文档数量
    max_input = min(top_n * 3, 30)  # 最多30个文档
    if len(documents) > max_input:
        logger.info(f"重排序输入文档数从 {len(documents)} 截取到 {max_input}")
        documents = documents[:max_input]

    rerank_model = os.getenv("RERANK_MODEL", "qwen3-rerank")
    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        logger.error("错误: 请先在环境变量中设置 DASHSCOPE_API_KEY")
        return None

    # 官方文档中的API地址和模型名称
    url = os.getenv("RERANK_URL")

    # 请求头
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    # 请求体
    payload = {
        "model": rerank_model,
        "input": {"query": query, "documents": documents},
        "parameters": {"top_n": top_n, "return_documents": True},
    }

    # 重试逻辑：最多3次，间隔1秒
    max_retries = 3

    for attempt in range(max_retries):
        try:
            async with httpx.AsyncClient(timeout=3.0) as client:  # 3秒超时
                response = await client.post(url, headers=headers, json=payload)

            if response.status_code == HTTPStatus.OK:
                return response.json()
            else:
                logger.error(
                    f"API 请求失败 (尝试 {attempt + 1}/{max_retries}): {response.status_code}"
                )
                if attempt == max_retries - 1:
                    logger.error(f"重排序 API 最终失败，返回 None")
                    return None
                else:
                    logger.warning(f"等待 1 秒后重试...")
                    await asyncio.sleep(1)
                    continue

        except httpx.TimeoutException as e:
            logger.error(f"重排序 API 超时 (尝试 {attempt + 1}/{max_retries}): {e}")
            if attempt == max_retries - 1:
                logger.warning("重排序 API 最终超时，使用原始结果")
                return None
            else:
                logger.warning(f"等待 1 秒后重试...")
                await asyncio.sleep(1)
                continue

        except Exception as e:
            logger.error(f"重排序请求发生异常 (尝试 {attempt + 1}/{max_retries}): {e}")
            if attempt == max_retries - 1:
                logger.error(f"重排序 API 最终失败，返回 None")
                return None
            else:
                logger.warning(f"等待 1 秒后重试...")
                await asyncio.sleep(1)
                continue

    return None  # 返回 None 触发降级
