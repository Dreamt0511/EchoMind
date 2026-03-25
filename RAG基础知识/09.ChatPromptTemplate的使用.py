from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder
from langchain_community.llms.tongyi import Tongyi

chat_prompt_template = ChatPromptTemplate(
    [
        ("system","你是一个宋代的豪放派代表词人，可以写词，风格参考辛弃疾(你只需要创作出结果即可，不需要做额外的说明)"),
        MessagesPlaceholder("history"),
        ("human","请再写出一首词")
    ]
)

history_data = [
    ("human","你来写一首词"),
    ("ai","八百里分麾下制，五十弦翻塞外声")
]
prompt_text = chat_prompt_template.invoke({"history": history_data}).to_string()
print(prompt_text)
model = Tongyi(model="qwen-max")
res = model.stream(input=prompt_text)
for chunk in res:
    print(chunk,end = "",flush=True)
