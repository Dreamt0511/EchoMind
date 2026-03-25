from langchain_community.embeddings import DashScopeEmbeddings

embed = DashScopeEmbeddings()

strList = ["你好","雷浩阿","hello","早上好"]

# for s in strList:
#     print(embed.embed_query(s))
print(embed.embed_documents(strList))