from langchain_community.document_loaders import PyMuPDFLoader,Docx2txtLoader

loader = PyMuPDFLoader("./data/数据库原理简答题总结.pdf")

docs  = loader.load()

docs2 = Docx2txtLoader("./data/2023212010-张飞龙-第4次作业 .docx").load()
print(docs2)