from pymilvus import MilvusClient

client = MilvusClient(
    uri= "https://in03-aa3b3c45f217386.serverless.aws-eu-central-1.cloud.zilliz.com",
    token = "db_aa3b3c45f217386:Zfl123456"
   )

databases = client.list_databases()
print("数据库列表:", databases)

#获取集合列表
collections = client.list_collections()
print("集合列表:", collections)

collection = collections[0]

#获取分区列表
partitions = client.list_partitions(collection_name=collection)
print("分区列表:", partitions)