from langchain_core.prompts import PromptTemplate
from langchain_community.llms.tongyi import Tongyi

"""这样写的作用是方便后期加入到链条chain里"""

prompt_temp = PromptTemplate.from_template(
    "我的邻居姓{lastname},刚生了{gender}孩，请你帮他取一个名字，简要回答。"
)

# print(prompt_temp)
"""调用.format方法注入信息"""
prompt_text = prompt_temp.format_prompt(lastname = "张",gender = "男")
# print(prompt_text)

model = Tongyi(model="qwen-plus",)

# print(model.invoke(prompt_text))
response = model.stream(prompt_text)
for chunk in response:
    print(chunk,end="",flush=True)
