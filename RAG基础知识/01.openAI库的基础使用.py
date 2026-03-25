from openai import OpenAI

#获取client对象
client = OpenAI(
    base_url= "https://dashscope.aliyuncs.com/compatible-mode/v1"
)

response = client.chat.completions.create(
    model = "qwen3-max",
    messages = [
        {"role":"system","content":"你是一名编程专家，简单回答，不要太死板"},
        {"role":"assistant","content":"好的我是一名编程专家，话不多，你要问什么？"},
        {"role":"user","content":"你是谁，会什么"}
    ],
    stream= True
)

for chunk in response:
    print(chunk.choices[0].delta.content, end="", flush=False)