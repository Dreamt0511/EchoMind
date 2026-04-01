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


两层降级策略：混合检索失败时自动降级为稠密向量检索（即语义搜索），重排序失败时自动降级为RRF融合排序后的结果。重排序内部带重试机制，最多重试3次，每次间隔1秒。

