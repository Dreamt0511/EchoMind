from langchain_community.llms.tongyi import Tongyi

model = Tongyi(model="qwen-plus")

res = model.stream(input="你是谁")
for chunk in res:
    print(chunk,end="",flush=True)