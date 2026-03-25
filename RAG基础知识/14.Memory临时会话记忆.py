from langchain_community.chat_models.tongyi import ChatTongyi
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
#from langchain_community.chat_message_histories import FileChatMessageHistory长期会话记忆库

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


base_chain = prompt | model | str_parser

store = {}
def get_history(sessionId):
    if sessionId not in store:
        store[sessionId] = InMemoryChatMessageHistory()
    print(f"\n\nstore存储的消息长度为\n{len(store[sessionId].messages)}\n消息内容\n{store[sessionId]}\n")
    return store[sessionId]

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

    print(chain_with_history.invoke({"input":"我家有一只猫"},session_config))
    print(chain_with_history.invoke({"input": "我家还有2只兔子"},session_config))
    print(chain_with_history.invoke({"input":"还有一条大狗狗"},session_config))
    print(chain_with_history.invoke({"input":"请问一共有几只宠物"},session_config))


