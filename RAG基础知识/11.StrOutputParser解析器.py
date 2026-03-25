# 导入StrOutputParser解析器
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_models import ChatTongyi
import os
from langchain_core.prompts import ChatPromptTemplate

model = ChatTongyi(
        model="qwen-max",
        api_key=os.getenv("DASHSCOPE_API_KEY")
)

#查看AI回答的消息类型
#print(type(model.invoke("你好")))
#输出结果为<class 'langchain_core.messages.ai.AIMessage'>即AIMessage，不能直接传递给model，因为chatModel不支持AIMessage类型的input,传递前需要经过StrOutputParser解析器解析
res = model.invoke("你是谁")
print(f"提问你是谁的回复内容\n{res.content}")
try:
    #尝试传递给model，会报错
    print(model.invoke(res))#这里如果写的是res.content的话就不会报错，因为res.content的类型为str
except ValueError as e:
    print(f"\n\n出现错误，原因{e}")
    print("开始使用StrOutputParser解析器解析res后再次询问")
    parser = StrOutputParser()
    res = parser.invoke(res)
    #print(f"解析后的res内容为{res}")
    print(f"解析后的res类型为{type(res)}")
    print("再次把res(解析过后的)传递给model")
    print(model.invoke(res).content)


"""下面把解析器接入链中"""
print("\n\n===========下面把解析器接入链中==============")
#创建提示词模板
test_prompt_text = ChatPromptTemplate.from_template("用{style}的风格回答问题{question}")
#创建链
parser = StrOutputParser()
chain = test_prompt_text | model | parser
answer = chain.invoke({"style":"幽默","question":"你和deepseek谁厉害"})
print(answer)
print(f"回复的类型为{type(answer)}")