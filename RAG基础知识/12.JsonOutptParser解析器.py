from langchain_core.output_parsers import JsonOutputParser,StrOutputParser
from  langchain_community.chat_models.tongyi import ChatTongyi
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import AIMessage

first_template = PromptTemplate.from_template(
    "我的邻居姓{lastname},刚生了一个{gender}孩子，请起名，并封装到json格式返回给我"
    "要求key是name,value就是你起的名字，请严格按照要求，只输出封装后的json数据，不用额外输出其他说明内容"
    "简要回答，不要太罗嗦"
)

second_template = PromptTemplate.from_template(
    "姓名{name}，请解释这个名字的含义"
)

str_parser = StrOutputParser()
json_Parser = JsonOutputParser()
model = ChatTongyi(model= "qwen-max")

chain = first_template | model | json_Parser | second_template | model | str_parser

print(model.invoke(first_template.format_prompt(lastname="张",gender="男")).content)
print("\n\n=======================开始解析名字含义=============================")
res = chain.stream({"lastname":"张","gender":"男"})
for chunk in res:
    print(chunk,end="",flush=True)
