from pathlib import Path
import logging
from typing import Set, List

logger = logging.getLogger(__name__)


class HashStorage:
    """哈希存储管理器,只存储文件哈希和分块哈希"""

    def __init__(self):
        # 创建哈希目录
        current_dir = Path(__file__).parent
        hash_dir = current_dir / "hash_storage"
        hash_dir.mkdir(parents=True, exist_ok=True)
        # 创建文件和分块哈希记录文件
        self.files_hash_path = hash_dir / "files_hash.txt"
        self.chunks_hash_path = hash_dir / "chunks_hash.txt"

        # 加载现有哈希值
        self.files_hash = self._load_hashes(self.files_hash_path)
        self.chunks_hash = self._load_hashes(self.chunks_hash_path)

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

    def _save_hashes(self, file_path: Path, hashes: Set[str]):
        """保存哈希值到文件"""
        try:
            # 采用清空写入而不是追加是因为要保存chunks_hash的时候就是全部写入所有哈希
            with open(file_path, "w", encoding="utf-8") as f:
                for hash_value in hashes:
                    f.write(f"{hash_value}\n")
        except Exception as e:
            logger.error(f"保存哈希值失败，原因{str(e)}")

    # 文件哈希管理============
    def is_file_duplicate(self, file_hash: str) -> bool:
        """检查文件哈希是否已存在"""
        return file_hash in self.files_hash

    def add_file_hash(self, file_hash: str):
        """添加文件哈希值"""
        if not self.is_file_duplicate(file_hash):
            self.files_hash.add(file_hash)
            self._save_hashes(self.files_hash_path, self.files_hash)
            logger.info(f"添加文件哈希: {file_hash[:16]}...")

    def remove_file_hash(self, file_hash: str):
        """删除文件哈希值"""
        if file_hash in self.files_hash:
            self.files_hash.discard(file_hash)
            self._save_hashes(self.files_hash_path, self.files_hash)
            logger.info(f"删除文件哈希: {file_hash[:16]}...")

    def remove_chunk_hashes_batch(self, chunk_hashes: List[str]):
        """批量删除块哈希"""
        removed_count = 0
        for chunk_hash in chunk_hashes:
            if chunk_hash in self.chunks_hash:
                self.chunks_hash.discard(chunk_hash)
                removed_count += 1
        
        if removed_count > 0:
            self._save_hashes(self.chunks_hash_path, self.chunks_hash)
            logger.info(f"批量删除 {removed_count} 个块哈希")

    # 分块哈希管理
    def is_chunk_duplicate(self, chunk_hash: str) -> bool:
        """检查分块哈希是否已存在"""
        return chunk_hash in self.chunks_hash

    def add_chunk_hash(self, chunk_hash: str):
        if not self.is_chunk_duplicate(chunk_hash):
            self.chunks_hash.add(chunk_hash)
            self._save_hashes(self.chunks_hash_path, self.chunks_hash)