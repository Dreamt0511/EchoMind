import requests
import json
from http import HTTPStatus
import os
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

def call_rerank_api(query, documents, top_n=5):
    """
    一个通用的重排序API调用模板
    """
    # 阿里云百炼的配置 (这里以百炼为例)
    # 注意：请使用你自己账号的API-KEY
    api_key = os.getenv('DASHSCOPE_API_KEY') # 建议从环境变量获取
    if not api_key:
        print("错误: 请先在环境变量中设置 DASHSCOPE_API_KEY")
        return None
        
    # 官方文档中的API地址和模型名称[citation:2]
    url = "https://dashscope.aliyuncs.com/api/v1/services/rerank/text-rerank/text-rerank"
    
    # 请求头
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    # 请求体
    payload = payload = {
        "model": os.getenv('RERANK_MODEL', 'qwen3-rerank'),
        "input": {  # 必须包在 input 里面！
            "query": query,
            "documents": documents
        },
        "parameters": {  # 必须包在 parameters 里面！
            "top_n": top_n,
            "return_documents": True
        }
    }
    
    # 发送请求
    try:
        response = requests.post(url, headers=headers, json=payload)
        if response.status_code == HTTPStatus.OK:
            return response.json()
        else:
            print(f"API 请求失败: {response.status_code} - {response.text}")
            return None
    except Exception as e:
        print(f"请求发生异常: {e}")
        return None


if __name__ == '__main__':
    query = "什么是文本排序模型"
    documents = [
        "文本排序模型广泛用于搜索引擎和推荐系统中，它们根据文本相关性对候选文本进行排序",
        "量子计算是计算科学的一个前沿领域",
        "预训练语言模型的发展给文本排序模型带来了新的进展"
    ]
    
    # 调用函数
    result = call_rerank_api(query, documents, top_n=2)
    
    # 打印结果
    if result:
        print(json.dumps(result, indent=2, ensure_ascii=False))
        # 你可以从这里解析出排序后的结果，例如：
        # for item in result['output']['results']:
        #     print(f"文档: {item['document']['text']}")
        #     print(f"相关性分数: {item['relevance_score']}\n")