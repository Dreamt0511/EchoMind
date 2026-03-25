from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader(
    file_path= "./data/实验5-基于深度学习的人脸识别技术.pdf",
    mode= "page",#读取模式，默认为page即每页读取为一个Document,若使用single则所有页读取为一个
    #password = ""#密码（有就写）
)

docs = loader.lazy_load()#太多了使用懒加载
for i,doc in enumerate(docs,start=1):
    print(f"\n读取到第{i}页的内容","="*200)
    print(doc)