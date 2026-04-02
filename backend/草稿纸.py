from pymilvus import MilvusClient, MilvusException
try:
   uri = 'http://localhost:19530'
   client = MilvusClient(uri=uri)
   if client.has_collection(collection_name="demo_collection"):
       client.drop_collection(collection_name="demo_collection")
   client.create_collection(
       collection_name="demo_collection",
       dimension=768,
       
   )
   print(client.list_collections())
   print("连接成功")
except MilvusException as e:
   print(e)