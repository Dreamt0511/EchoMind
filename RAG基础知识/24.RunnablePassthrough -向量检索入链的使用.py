from langchain_core.documents import Document
from langchain_core.runnables import RunnablePassthrough
from langchain_chroma import Chroma
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.chat_models import ChatTongyi
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os


#创建模型
model = ChatTongyi(
    model = 'qwen-max',
    api_key= os.getenv("DASHSCOPE_API_KEY")
)

vector_store = Chroma(
    collection_name= "test",
    persist_directory= "./chroma__db",
    embedding_function= DashScopeEmbeddings()
)
#创建检索器
retriever = vector_store.as_retriever(search_kwargs = {"k":5})

prompt = ChatPromptTemplate.from_messages(
    [
        ("system","以我提供的资料为主，根据资料内容回答用户问题，简洁为主，如果参考资料不符合用户提问，请回复：未检索到相关资料，参考资料：{context}"),
        ("user", "用户提问：{question}")
    ]
)
def print_prompt(prompt):
    print("="*100,"参考原文","="*100)
    print(prompt)
    print("=" * 100, "参考原文", "=" * 100)
    return prompt

#定义检索器返回的文档格式化函数
def format_docs(docs:list[Document])-> str:
    """将文档列表格式化为字符串"""
    return "\n".join(doc.page_content for doc in docs)

chain = {"question":RunnablePassthrough(),"context":retriever| format_docs} | prompt | print_prompt | model | StrOutputParser()

result = chain.invoke("大模型应用开发的路线是什么")#这里的输入就不用dict了
print("\n")
print(result)
