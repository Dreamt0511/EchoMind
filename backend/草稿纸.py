from minio import Minio
import os
import io

# 初始化客户端
client = Minio(
    "localhost:9000",
    access_key="Dreamt",
    secret_key="zfl123456",
    secure=False
)

# 上传文件 - 直接存储
with open("./data/习概小组ppt文案.docx", "rb") as file:
    data = file.read()
    
client.put_object(
    bucket_name="echomind-uploads",
    object_name="my-first-file.docx",  # 文件名
    data=io.BytesIO(data),
    length=len(data),
    content_type="application/docx"
)

print("✅ 文件已存储到 MinIO！")