# 项目目录结构

| 文件名 | 描述 |
|--------|------|
| `main.py` | FastAPI 主入口 |
| `api.py` | API 路由定义 |
| `agent.py` | Agent 核心逻辑 |
| `tools.py` | Agent 工具集 |
| `config.py` | 配置管理 |
| `schemas.py` | Pydantic 数据模型 |
| `documents_process.py` | 文档处理逻辑 |
| `milvus_client.py` | Milvus 连接 |
| `postgresql_client.py` | PostgreSQL 连接 |


1.两层降级策略：混合检索失败时自动降级为稠密向量检索（即语义搜索），重排序失败时自动降级为RRF融合排序后的结果。重排序内部带重试机制，最多重试3次，每次间隔1秒。

2.动态系统提示词切换：根据用户请求中的知识库id，切换系统提示词。“默认知识库”使用默认系统提示词（SYSTEM_PROMPT_DEFAULT），其他知识库使用特定系统提示词（SYSTEM_PROMPT_SPECIFIC）。选择“默认知识库”时回答结合知识库知识和通用知识进行组织。选择其他指定知识库时回答只限定于该知识库的内容。

3.知识库检索范围动态切换：根据用户选择的知识库id，切换知识库检索范围。“默认知识库”检索范围为所有知识库相关文档，其他指定知识库检索范围为该知识库相关文档。



