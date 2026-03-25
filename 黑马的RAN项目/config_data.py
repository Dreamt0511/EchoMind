import os

md5_path = "./md5.txt"
filesName_path = ".filesName.txt"

# Chorma
collection_name = "RAG_Project"
persist_directory = "./chroma_db"

# splitter
chunk_size = 100  # 每段最大字符数
chunk_overlap = 20  # 每段之间重叠的字符数
separators = [
    "\n\n",  # 1. 优先按段落分割（两个换行）
    "\n",  # 2. 其次按行分割
    "。", "！", "？",  # 3. 再按句子分割
    "；", "，", "、",  # 4. 再按短语分割
    " ",  # 5. 再按空格
    ""  # 6. 最后按字符
]
max_split_char_num = 100  # 文本分割的阈值

# rag
similarity_threshold = 3  # 检索返回匹配的文档数量

embedding_model_name = "text-embedding-v4"
chat_model_name = "qwen-max"
api_key = os.getenv("DASHSCOPE_API_KEY")

# chat_history
history_path = "./chat_histories"  # 历史会话存储目录
session_config = {
    "configurable": {
        "session_id": "Dreamt"
    }
}
