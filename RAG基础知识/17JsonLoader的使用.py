from langchain_community.document_loaders import JSONLoader
import json

loader = JSONLoader(
    file_path= "./data/stu.json",#文件路径
    jq_schema= ".",#jq schema语法,.代表抽取全部内容
    text_content= False,# 抽取的是否是字符串，默认True
    json_lines= False##是否是JsonLines文件(每一行都是JSON的文件,
)

document = loader.load()
print(type(document[0].page_content))
content_dict = json.loads(document[0].page_content)
print(content_dict)