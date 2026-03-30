from pathlib import Path
import logging
import aiofiles
import asyncio
from typing import Set, List

logger = logging.getLogger(__name__)


class HashStorage:
    def __init__(self):
        # 创建哈希目录
        current_dir = Path(__file__).parent
        self.hash_dir = current_dir / "hash_storage"
        self.hash_dir.mkdir(parents=True, exist_ok=True)

        self.files_hash_path = self.hash_dir / "files_hash.txt"
        self.chunks_hash_path = self.hash_dir / "chunks_hash.txt"

        # 加载现有哈希值
        self.files_hash = self._load_hashes(self.files_hash_path)
        self.chunks_hash = self._load_hashes(self.chunks_hash_path)

        self._files_lock = asyncio.Lock()
        self._chunks_lock = asyncio.Lock()

    # 哈希文件加载与保存============
    def _load_hashes(self, file_path: Path) -> Set[str]:
        """从文本文件加载哈希集合"""
        hashes = set()
        if file_path.exists():
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    for line in f:
                        hash_value = line.strip()
                        if hash_value:
                            hashes.add(hash_value)
            except Exception as e:
                logger.error(f"加载哈希文件{Path(file_path).name}失败,原因{str(e)}")
        return hashes

    async def _save_hashes(self, file_path: Path, hashes: Set[str]):
        """异步保存哈希值到文件"""
        try:
            # 采用清空写入而不是追加是因为要保存chunks_hash的时候就是全部写入所有哈希
            async with aiofiles.open(file_path, "w", encoding="utf-8") as f:
                for hash_value in hashes:
                    await f.write(f"{hash_value}\n")
        except Exception as e:
            logger.error(f"保存哈希值失败，原因{str(e)}")

    # 文件哈希管理============
    async def is_file_duplicate(self, file_hash: str) -> bool:
        """异步检查文件哈希是否已存在"""
        async with self._files_lock:
            return file_hash in self.files_hash

    async def add_file_hash(self, file_hash: str):
        """异步添加文件哈希值"""
        async with self._files_lock:
            if file_hash not in self.files_hash:
                self.files_hash.add(file_hash)
                await self._save_hashes(self.files_hash_path, self.files_hash)
                logger.info(f"添加文件哈希: {file_hash[:16]}...")

    async def remove_file_hash(self, file_hash: str):
        """异步删除文件哈希值"""
        async with self._files_lock:
            if file_hash in self.files_hash:
                self.files_hash.discard(file_hash)
                await self._save_hashes(self.files_hash_path, self.files_hash)
                logger.info(f"删除文件哈希: {file_hash[:16]}...")

    # 分块哈希管理============
    async def remove_chunk_hashes_batch(self, chunk_hashes: List[str]):
        """异步批量删除块哈希"""
        async with self._chunks_lock:
            removed_count = 0
            for chunk_hash in chunk_hashes:
                if chunk_hash in self.chunks_hash:
                    self.chunks_hash.discard(chunk_hash)
                    removed_count += 1

            if removed_count > 0:
                await self._save_hashes(self.chunks_hash_path, self.chunks_hash)
                logger.info(f"批量删除 {removed_count} 个块哈希")

    async def is_chunk_duplicate(self, chunk_hash: str) -> bool:
        """异步检查分块哈希是否已存在"""
        async with self._chunks_lock:
            return chunk_hash in self.chunks_hash

    async def add_chunk_hash(self, chunk_hash: str):
        """异步添加分块哈希"""
        async with self._chunks_lock:
            if chunk_hash not in self.chunks_hash:
                self.chunks_hash.add(chunk_hash)
                await self._save_hashes(self.chunks_hash_path, self.chunks_hash)

    # ============ 新增批量方法 ============
    async def batch_check_duplicates(self, chunk_hashes: List[str]) -> Set[str]:
        """批量检查分块哈希是否已存在
        
        Args:
            chunk_hashes: 哈希值列表
            
        Returns:
            已存在的哈希值集合
        """
        if not chunk_hashes:
            return set()
            
        async with self._chunks_lock:
            existing = set()
            for chunk_hash in chunk_hashes:
                if chunk_hash in self.chunks_hash:
                    existing.add(chunk_hash)
            return existing

    async def batch_add_chunk_hashes(self, chunk_hashes: List[str]):
        """批量添加分块哈希
        
        Args:
            chunk_hashes: 哈希值列表
        """
        if not chunk_hashes:
            return
            
        async with self._chunks_lock:
            # 找出新的哈希
            new_hashes = []
            for chunk_hash in chunk_hashes:
                if chunk_hash not in self.chunks_hash:
                    new_hashes.append(chunk_hash)
            
            if new_hashes:
                # 批量添加到集合
                for chunk_hash in new_hashes:
                    self.chunks_hash.add(chunk_hash)
                # 一次性保存所有哈希
                await self._save_hashes(self.chunks_hash_path, self.chunks_hash)
