from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

#与InMemoryVectorStore的区别在这
vector_store = Chroma(
    collection_name= "test",
    embedding_function=DashScopeEmbeddings(),
    persist_directory= "./chroma__db"
)

#注释掉存入数据的逻辑（如果已有了向量数据库的数据）
"""
loader = TextLoader(
    file_path="./data/分割测试文档.txt",
    encoding="utf-8")

documents = loader.load()

#对加载的documents进行分割
splitter = RecursiveCharacterTextSplitter(
    chunk_size= 300,
    chunk_overlap= 50,
    separators=[
        "\n\n",  # 1. 优先按段落分割（两个换行）
        "\n",  # 2. 其次按行分割
        "。", "！", "？",  # 3. 再按句子分割
        "；", "，", "、",  # 4. 再按短语分割
        " ",  # 5. 再按空格
        ""  # 6. 最后按字符
    ]
)
split_docs = splitter.split_documents(documents=documents)
print(f"\n一共分割成了{len(split_docs)}段")

#把分割后的内容向量化存入
vector_store.add_documents(
    documents= split_docs,#被添加的文档，类型为list[Document]
    ids= [f"id{i}" for i in range(1,len(split_docs)+1)]#给添加的文档提供id,list[str
)
"""

#检索相似的片段
#带相似度的搜素
q = "AI 时代会不会淘汰程序员？"
res = vector_store.similarity_search_with_score(
    query= q,
    k = 5#返回5条最相似的片段
)
print(f"开始检索与问题最相似的5条片段\n问题：{q}")
for i,similar_doc in enumerate(res,start=1):
    print("=" * 100)
    print(f"【第{i}条相似信息相似度为{similar_doc[1]}】\n内容为:\n{similar_doc[0].page_content}\n")



