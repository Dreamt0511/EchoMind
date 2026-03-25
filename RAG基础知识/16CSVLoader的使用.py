from langchain_community.document_loaders import CSVLoader

loader = CSVLoader(
    file_path="./data/stu.csv",
    csv_args={
        "delimiter": ",",
        "quotechar": "'",
        # 如果数据原本有表头，就不要下面的代码，如果没有可以使用
        "fieldnames": ["name", "age", "gender", "爱好"]
    },
    encoding="utf-8"
)

# 批量加载 .load() -> [Document, Document, ...]
# documents = loader.load()
#
# for document in documents:
#    print(type(document), document)

# 懒加载 .lazy_load() 迭代器[Document]
for document in loader.lazy_load():
    print(document)