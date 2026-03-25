from langchain_community.chat_models.tongyi import ChatTongyi
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import FileChatMessageHistory
import os
"""
消息过长可以采用剪枝，参考链接：https://reference.langchain.com/python/langchain-core/messages/utils/trim_messages
只需要提供 get_history 函数返回一个支持 add_message 的历史对象
RunnableWithMessageHistory 内部会自动调用 add_user_message 和 add_ai_message
FileChatMessageHistory 的 add_message 方法负责写入文件

此外长期记忆应该存到数据库，在实际生产环境都需要持久化的存储数据库。
langchain 提供了很多基于其他存储系统的扩展依赖，例如 redis、kafka、MongoDB 等。
from langchain_community.chat_message_histories import RedisChatMessageHistory 就是使用redis存储
"""

# 创建存储历史对话的目录
os.makedirs("./chat_histories", exist_ok=True)

model = ChatTongyi(model="qwen-max")
str_parser = StrOutputParser()

prompt = ChatPromptTemplate.from_messages(
    [
        ("system","你需要根据对话历史回应用户问题"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human","用户当前输入：{input},请给出回应")
    ]
)

def print_prompt(prompt):
    print("\n","="*50,"\n",prompt.to_string(),"\n","="*50)
    return prompt


base_chain = prompt | print_prompt | model | str_parser

# ===== 只需要修改这里！=====
def get_history(sessionId: str):
    """返回文件存储的历史记录"""
    # 每个会话一个文件，存在 ./chat_histories/ 目录下
    file_path = f"./chat_histories/{sessionId}.json"
    return FileChatMessageHistory(file_path)
# =========================

chain_with_history = RunnableWithMessageHistory(
    base_chain,
    get_session_history= get_history,#获取历史会话的函数
    input_messages_key= "input",#声明用户输入消息在模板中的占位符
    history_messages_key= "chat_history",#声明历史消息再模板中的占位符
    #history_factory_config用于自定义传递给 get_session_history 函数的参数。
    #默认只传递一个 session_id。通过此参数，可以定义多个参数（如 user_id 和 conversation_id）来更精细地管理历史记录。
)

if __name__ == "__main__":
    session_config = {"configurable":{"session_id":"Dreamt"}}


    #有存储的长期对话历史的话就可以删除前期的几个补充说明了
    """
    print(chain_with_history.invoke({"input":"我家有一只猫"},session_config))
    print(chain_with_history.invoke({"input": "我家还有2只兔子"},session_config))
    print(chain_with_history.invoke({"input":"还有一条大狗狗"},session_config))
    """

    #传入session_id查看存储的长期历史对话
    history = get_history("Dreamt")
    history_messages = history.messages
    print(f"读取到的历史对话如下：\n")
    for i,msg in enumerate(history_messages):
        print(f"第{i+1}条：{msg.content}")
    print(f"\n读取到{len(history_messages)}条历史消息")

    #在下面添加提问
    #print(chain_with_history.invoke({"input":"你是谁"},session_config))



