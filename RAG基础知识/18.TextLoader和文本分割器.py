from operator import length_hint

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

loader = TextLoader(file_path="./data/分割测试文档.txt",
                    encoding= "utf-8")

docs = loader.load()

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

split_docs = splitter.split_documents(documents=docs)
for i,doc in enumerate(split_docs):
    print(f"\n第{i+1}段数据","="*100)
    print(doc)
print(f"\n一共分割成了{len(split_docs)}段")
