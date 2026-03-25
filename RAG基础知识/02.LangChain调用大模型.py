from langchain_community.llms.tongyi import Tongyi

model = Tongyi(model="qwen-plus")

res = model.invoke(input="你知识库最后一次更新是几号")

print(res)