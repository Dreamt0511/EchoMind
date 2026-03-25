from langchain_community.chat_models import ChatTongyi
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os
import asyncio
import time

start_time = time.time()
# 获取向量对象
vector_store = Chroma(
    collection_name="test",
    embedding_function=DashScopeEmbeddings(),
    persist_directory="./chroma__db"
)

# 构建提示词模板
prompt = ChatPromptTemplate.from_messages(
    [
        ("system",
         "以我提供的资料为主，根据资料内容回答用户问题，简洁为主，如果参考资料不符合用户提问，请回复：未检索到相关资料，参考资料：{context}"),
        ("user", "用户提问：{question}")
    ]
)

# 创建模型
model = ChatTongyi(
    model='qwen-max',
    api_key=os.getenv("DASHSCOPE_API_KEY")
)

str_parser = StrOutputParser()
q = "大模型应用开发的路线是什么"

# 构建链
chain = prompt | model | str_parser


async def main():
    # 构建检索出来的参考信息列表
    reference_list = []

    # 从向量数据库中检索相关信息（similarity_search本身不支持异步，需要在线程中执行）
    loop = asyncio.get_event_loop()
    res = await loop.run_in_executor(
        None,  # 使用默认线程池
        lambda: vector_store.similarity_search(q, 5)
    )

    # 添加相关信息进列表
    for doc in res:
        reference_list.append(doc.page_content)

    reference_context = ','.join(reference_list)

    # 把问题和参考一起发给ai进行提问（使用异步流式）
    print(f"参考原文：\n{reference_context}")
    print("\n", "=" * 100)
    print("AI回复如下")

    # 使用异步流式处理
    async for chunk in chain.astream(input={"question": q, "context": reference_context}):
        print(chunk, end="", flush=True)
    print()  # 最后加一个换行
    consume_seconds = time.time() - start_time
    print(f"\n系统执行时长为{consume_seconds}")


# 运行异步主函数
if __name__ == "__main__":
    asyncio.run(main())
