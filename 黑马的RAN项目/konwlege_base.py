""""
知识库基础服务
"""
import config_data as config
import hashlib
from  langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import DashScopeEmbeddings
import os

#检查md5是否已存在，从而判断知识库是否重复
def check_md5(md5_str:str)->bool:
    """检查是否已经处理过md5制"""
    try:
        with open(config.md5_path,'r',encoding='utf-8') as f:
            exists = md5_str in (line.strip() for line in f)
            return exists
    except FileNotFoundError:
        #文件不存在，创建空文档然后关闭
        open(config.md5_path,'w',encoding='utf-8').close()
        return False

#保存md5值
def save_md5(md5_str):
    #需要使用a追加写入，而不是w，w会清空原内容
    with open(config.md5_path,'a',encoding='utf-8') as f:
        f.write(md5_str+'\n')

#记录文件名，写入文件
def save_fileName(fileName):
    with open(config.filesName_path,'a',encoding='utf-8') as f:
        f.write(fileName+'\n')

#输入字符串获得md5字符串
def get_md5_str(input_str):
    md5_obj = hashlib.md5()#获得md5对象
    # 2. 输入数据（需要字节格式）input_str.encode('utf-8') 将字符串转为字节
    md5_obj.update(input_str.encode('utf-8')) # 需要编码为字节,因为哈希算法只能处理字节数据，不能直接处理字符串
    md5_value = md5_obj.hexdigest()#获取16进制字符串

    return md5_value

class KnowledgeBaseService():
    def __init__(self):
        #如果不存在数据库文件则创建，存在则跳过
        os.makedirs(config.persist_directory,exist_ok=True)

        #分割器
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size= config.chunk_size,
            chunk_overlap= config.chunk_overlap,
            separators= config.separators
        )
        self.chroma = Chroma(
            embedding_function= DashScopeEmbeddings(model=config.embedding_model_name),
            collection_name= config.collection_name,
            persist_directory= config.persist_directory
        )

    def upload_to_chroma_by_str(self,data:str,filename):
        md5_value = get_md5_str(data)
        if check_md5(md5_value):
            return "[跳过]内容已在数据库中"

        #切分文件内容，获得碎片数组
        knowledge_chunks:list[str] = []
        if len(data) > config.max_split_char_num:#如果大于规定的最小片段长度
            knowledge_chunks = self.splitter.split_text(data)
        else:
            knowledge_chunks = [data]

        #存入向量数据库
        self.chroma.add_texts(
            texts=knowledge_chunks
        )

        #记录Md5值（方便判断知识库是否已存入数据库）
        save_md5(md5_value)
        #记录文件名(方便显示已存入的知识库名)
        save_fileName(filename)

        return "[成功]内容已存入向量数据库"



if __name__ == "__main__":
   service = KnowledgeBaseService()
   res = service.upload_to_chroma_by_str("Dreamt is an AI engineer","t")
   print(res)






