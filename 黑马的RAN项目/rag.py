import os
from langchain_core.documents import Document
from vector_stores import VectorStoreService
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.chat_models.tongyi import ChatTongyi
import config_data as config
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import FileChatMessageHistory
from langchain_core.runnables import RunnableLambda, RunnableParallel
from langchain_core.prompts import MessagesPlaceholder


class RagService:

    def __init__(self):
        self.vector_service = VectorStoreService(
            embedding=DashScopeEmbeddings(model=config.embedding_model_name)
        )

        self.prompt_template = ChatPromptTemplate.from_messages(
            [
                ("system", "以我提供的参考资料为主，简介和专业地回答用户问题,如果，参考资料{context}"),
                MessagesPlaceholder(variable_name="history"),
                ("user", "请回答用户问题，{input}")
            ]
        )

        self.chat_model = ChatTongyi(
            model=config.chat_model_name,
            api_key=config.api_key
        )

        self.chain = self.get_chain()

    def get_chain(self):
        # 创建存储历史对话的目录
        os.makedirs(config.history_path, exist_ok=True)  # exist_ok=True：如果目标目录已经存在，不会抛出异常

        retriever = self.vector_service.get_retriever()

        # 打印查看retriever接收到的内容
        def print_input_to_prompt(value):
            print("-" * 100)
            print(f"接收到的提示词内容类型为：{type(value)}\n内容为：{value}\n")
            return value

        def format_document(docs: list[Document]):
            return '\n'.join(doc.page_content for doc in docs)

        def get_history(session_id):
            file_path = config.history_path + f"/{session_id}.json"
            chat_history = FileChatMessageHistory(file_path=file_path)
            return chat_history

        # 提取相关内容，转换成下一个提示词模板的输入（类型为字典类型，需要question,context,history三个参数键）
        def format_prompt(value):
            new_value = {}
            input = value["input"]
            new_value["input"] = input["input"]
            new_value["history"] = input["history"]
            new_value["context"] = value["context"]

            return new_value

        chain = (
                RunnableParallel(  # 显式代替普通字典
                    input=RunnablePassthrough(),
                    context=(RunnableLambda(lambda x: x["input"])  # 提取出input进行向量检索
                             | retriever
                             | RunnableLambda(print_input_to_prompt)
                             | RunnableLambda(format_document)
                             )
                )
                # | RunnableLambda(print_input_to_prompt) #打印上一个的输出，方便转换成下一个的输入
                | RunnableLambda(format_prompt)  # 提取相关内容，转换成下一个的输入
                | self.prompt_template
                | self.chat_model
                | StrOutputParser()
        )

        conversation_chain = RunnableWithMessageHistory(
            chain,
            get_session_history=get_history,
            input_messages_key="input",
            history_messages_key="history"
        )

        return conversation_chain


if __name__ == "__main__":
    session_config = config.session_config
    res = RagService().chain.stream({"input": "我问过你哪些问题"}, session_config)
    for chunk in res:
        print(chunk, end="", flush=True)

"""
【完整数据流转过程】

原始输入：周杰伦简介
    ↓
{"input": "周杰伦简介", "history": [...]}
    ↓ [RunnableParallel]
{
    'input': {'input': '周杰伦简介（RunnablePassthrough()占位，原样保留输入）', 'history': [...]}, #这里靠RunnablePassthrough()原样保留输入
    'context': '周杰伦是一位歌手...\n他的作品包括...'
}
    ↓ [print_input_to_prompt]  # 打印调试信息
{
    'input': {'input': '周杰伦简介', 'history': [...]},
    'context': '周杰伦是一位歌手...\n他的作品包括...'
}
    ↓ [format_prompt] 提取input、history、context作为下一个 提示词模板 的输入
{
    'input': '周杰伦简介',
    'history': [...],
    'context': '周杰伦是一位歌手...\n他的作品包括...'
}
    ↓ [prompt_template] 填充内容到提示词模板
[
    ("system", "以我提供的参考资料为主...{context}"),
    MessagesPlaceholder(history),
    ("user", "请回答用户问题，{input}")
]
    ↓ [chat_model]
AIMessage(content="周杰伦，1979年1月18日出生于台湾省...")
    ↓ [StrOutputParser]
"周杰伦，1979年1月18日出生于台湾省..."

"""
