from langchain_chroma import Chroma
import config_data as config

"""返回向量检索器，方便加入链"""


class VectorStoreService():

    def __init__(self, embedding):
        self.vector_store = Chroma(
            collection_name=config.collection_name,
            persist_directory=config.persist_directory,
            embedding_function=embedding
        )

        # 关键：创建索引以提升检索速度
        try:
            # 获取底层集合
            collection = self.vector_store._collection
            # 创建IVF_FLAT索引（如果没有的话）
            if len(collection.list_indices()) == 0:
                collection.create_index(
                    index_type="IVF_FLAT",
                    metric_type="IP",
                    params={"nlist": 100}
                )
        except:
            pass  # 索引创建失败也不影响使用

    def get_retriever(self):
        retriever = self.vector_store.as_retriever(search_kwargs={"k": config.similarity_threshold})
        return retriever