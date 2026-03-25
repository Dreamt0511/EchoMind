from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder
from langchain_community.llms.tongyi import Tongyi

chat_prompt_template = ChatPromptTemplate(
    [
        ("system","你是一个边塞诗人，可以作诗,作诗即可，不要说多余的话"),
        MessagesPlaceholder("history"),
        ("human","请在再一首诗")
    ]
)

history_data = [
    ("human","你来写一首词"),
    ("ai","床前明月光，疑是地上霜，举头望明月，低头思故乡")
]

model = Tongyi(model = "qwen-max")

chain = chat_prompt_template | model
print(type(chain))

"""
#invoke执行
res = chain.invoke({"history":history_data})
print(res)
print("下面是流式输出")
"""

#stream执行，流式输出
for chunk in chain.stream({"history":history_data}):
    print(chunk,end="",flush=True)