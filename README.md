# EchoMind - 个性化AI问答助手

<div align="center">

**念念不忘，必有回响**

[![GitHub](https://img.shields.io/badge/GitHub-EchoMind-blue?logo=GitHub)](https://github.com/Dreamt0511/EchoMind)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.135%2B-009688?logo=FastAPI)](https://fastapi.tiangolo.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.56%2B-FF4B4B?logo=Streamlit)](https://streamlit.io)
[![LangChain](https://img.shields.io/badge/LangChain-1.2%2B-1C3C3C?logo=LangChain)](https://langchain.com)
[![LangGraph](https://img.shields.io/badge/LangGraph-1.1%2B-1C3C3C?logo=LangGraph)](https://langchain-ai.github.io/langgraph/)
[![Milvus](https://img.shields.io/badge/pymilvus-2.6%2B-00A4B4?logo=Milvus)](https://milvus.io)
[![Redis](https://img.shields.io/badge/redis-7.4%2B-DC382D?logo=Redis)](https://redis.io)
[![DashScope](https://img.shields.io/badge/DashScope-1.25%2B-FF6A00?logo=AlibabaCloud)](https://dashscope.aliyun.com)
[![Docker](https://img.shields.io/badge/Docker-Supported-2496ED?logo=Docker)](https://docker.com)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**EchoMind** 是一个基于 FastAPI 构建的个性化 AI 问答助手，具备长期记忆能力和知识库集成功能。系统通过结合向量数据库检索、记忆管理和 LLM 处理，为用户提供智能化的个性化回答。


</div>

## 📋 目录

- [🎯 项目概述](#-项目概述)
- [✨ 核心特性](#-核心特性)
- [🏗️ 系统架构](#️-系统架构)
- [🚀 快速开始](#-快速开始)
- [📖 API 文档](#-api-文档)
- [🔧 配置说明](#-配置说明)
- [📁 项目结构](#-项目结构)
- [🤝 贡献指南](#-贡献指南)
- [📄 许可证](#-许可证)

## 🎯 项目概述

EchoMind 是一个全功能的个性化 AI 问答系统，主要特点包括：

- **长期记忆系统**：自动提取和存储用户对话记忆，支持个性化回答
- **知识库管理**：支持多知识库文档上传、检索和管理
- **混合检索**：结合稠密向量检索和稀疏检索，提供准确的文档匹配
- **实时对话**：流式响应，提供打字机效果的交互体验
- **智能记忆**：自动识别和过滤重复记忆，保持记忆库的高效性

## ✨ 核心特性

### 🧠 记忆系统
- **多类型记忆**：支持摘要记忆、语义记忆、情节记忆、程序记忆和用户画像
- **自动提取**：后台任务每3分钟自动检查并提取未压缩的对话
- **智能过滤**：相似度高于0.9的记忆自动标记为重复并过滤
- **记忆优化**：按主题合并碎片化记忆，确保信息完整性

### 📚 知识库管理
- **多知识库支持**：用户可创建多个知识库，分别管理不同领域的文档
- **文档处理**：支持PDF、Word文档上传，自动分块和向量化
- **混合检索**：结合稠密检索（语义搜索）和稀疏检索（关键词匹配）
- **重排序机制**：使用外部API进行相关性重排序，提供最优结果

### 🔍 检索系统
- **两层降级策略**：
  - 混合检索失败时自动降级为稠密向量检索
  - 重排序失败时降级为RRF融合排序结果
- **动态提示词**：根据选择的知识库动态切换系统提示词
- **检索范围控制**："默认知识库"检索所有知识，指定知识库限定检索范围

### 💬 对话系统
- **流式响应**：实时逐token输出，提供打字机效果
- **工具调用展示**：显示AI的思考过程和工具调用状态
- **会话管理**：Redis缓存会话历史，保证对话连贯性
- **实时保存**：对话内容实时异步写入PostgreSQL数据库

## 🏗️ 系统架构

### 整体架构

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Streamlit     │    │    FastAPI      │    │     Redis       │
│   Frontend      │◄──►│    Backend      │◄──►│   (Session &   │
│                 │    │                 │    │   Checkpoint)  │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                              │
                              ▼
                    ┌─────────────────┐
                    │   PostgreSQL    │
                    │   (Metadata &   │
                    │   Conversations)│
                    └─────────────────┘
                              │
                              ▼
                    ┌─────────────────┐
                    │     Milvus      │
                    │  (Vector Store  │
                    │   & Memories)   │
                    └─────────────────┘
```

### 核心组件

#### 1. Agent 系统 (`agent.py`)
- 基于 LangChain 构建的智能代理
- 支持工具调用和流式响应生成
- 集成 Redis 检查点机制，实现会话状态管理
- 动态提示词切换，根据知识库选择调整回答策略

#### 2. 工具框架 (`tools.py`)
- `search_knowledge_base`: 知识库混合检索工具
- `get_memory`: 多类型记忆检索工具，模型根据当前对话内容自主决策各种类型记忆需要检索的数量
- `get_raw_conversation_by_summary_id`: 原始对话获取工具，当需要查询原始对话内容时输入summary_id获取摘要对应的原始对话内容

#### 3. 记忆管理 (`auto_store_memory_from_psql.py`)
- 后台任务自动提取对话记忆
- 支持4种记忆类型：摘要、语义、情节、程序，多轮对话自动提取用户画像。
- 记忆冲突检测和重复过滤
- 用户画像智能融合更新

#### 4. 知识库管理 (`knowledeg_base_manager.py`)
- 采用父子分块策略，（父块1000字符，子块200字符）
- 向量化存储和混合检索，RRF融合排序
- 文件整体去重（SHA-256哈希检测），子块内容去重，避免重复存储。

#### 5. 数据处理管道 (`documents_process.py`)
- 支持PDF、Word文档处理
- 文本分块和嵌入生成
- 重排序文档相关性评分

## 🚀 快速开始

### 环境要求

- Python 3.12+
- PostgreSQL 14+
- Redis 6+ (本项目为docker-compose部署)
- Milvus 2.4+ (Milvus 云服务)
- DashScope API Key

### 安装步骤

1. **克隆项目**
```bash
git clone https://github.com/Dreamt0511/EchoMind.git
cd EchoMind
```

2. **创建虚拟环境**
```bash
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# 或
.venv\Scripts\activate     # Windows
```

3. **安装依赖**
```bash
pip install -r requirements.txt
```

4. **配置环境变量**
创建 `.env` 文件并配置以下参数：

```env
# 大模型配置
DASHSCOPE_API_KEY=your_dashscope_api_key
BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
AGENT_BASE_MODEL=qwen-plus
SUMMARIZATION_MODEL=qwen-turbo

# 向量数据库配置
Milvus_url=your_milvus_url
Token=your_milvus_token
knowledge_base_collection=knowledge_base_collection
memory_collection=memory_collection

# Redis配置
Redis_URI=redis://localhost:6379/0

# 嵌入模型配置
EMBEDDING_MODEL=text-embedding-v4
dense_dimension=1024

# 数据库配置
DATABASE_URL=postgresql://user:pass@localhost:5432/echomind_db
user=your_db_user
password=your_db_password
host=localhost
port=5432
db_name=echomind_db
```

5. **启动后端服务**
```bash
cd backend
python main.py
或
uvicorn main:app --reload
```

6. **启动前端界面**
```bash
cd frontend
streamlit run app.py
```

### 访问应用

- **API 文档**: http://localhost:8000/docs
- **Web 界面**: http://localhost:8501

## 📖 API 文档

### 核心端点

#### 对话接口
```http
GET /chat_with_agent/stream
Parameters:
- query: 用户消息
- knowledge_base_id: 知识库ID
- user_id: 用户ID
Returns: 流式文本响应
```

#### 知识库管理
```http
POST /knowledge-bases                    # 创建知识库
DELETE /knowledge-bases/{id}            # 删除知识库
GET /knowledge-bases                    # 获取用户知识库列表
GET /knowledge-bases/{id}/files         # 获取知识库文件列表
```

#### 文档管理
```http
POST /document_upload                    # 上传文档
DELETE /knowledge-bases/{id}/documents/{hash}  # 删除文档
```

## 🔧 配置说明

### 记忆提取配置
```python
# 记忆类型重要性评分
SEMANTIC_MEMORY_SCORE = 0.8    # 语义记忆
EPISODIC_MEMORY_SCORE = 0.7    # 情节记忆  
PROCEDURAL_MEMORY_SCORE = 0.6  # 程序记忆
USER_PROFILE_SCORE = 0.9       # 用户画像
```

### 文档分块配置
```python
# config.py
parent_chunk_size = 1000        # 父块大小
parent_chunk_overlap = 200      # 父块重叠
child_chunk_size = 200          # 子块大小
child_chunk_overlap = 50        # 子块重叠
```

### Redis TTL配置
```python
TTL_CONFIG = {
    "default_ttl": 60,          # 60分钟后过期
    "refresh_on_read": True     # 读取时刷新TTL
}
```

## 📁 项目结构

```
EchoMind/
├── backend/
|   |── tests/                   # 测试文件夹
│   ├── agent.py                 # langchian智能代理
│   ├── api.py                   # FastAPI路由定义
│   ├── main.py                  # 应用入口
│   ├── tools.py                 # 工具函数集合
│   ├── config.py                # 系统配置
│   ├── schemas.py               # Pydantic数据模型
│   ├── documents_process.py     # 文档处理管道
│   ├── auto_store_memory_from_psql.py  # 记忆提取后台任务
│   ├── milvus_client.py         # Milvus客户端
│   ├── postgresql_client.py     # PostgreSQL客户端
│   ├── redis_cache.py           # Redis缓存管理
│   └── hash_storage.py          # 文件哈希管理
├── frontend/
│   ├── app.py                   # Streamlit前端应用
│   └── style.css                # 前端样式
├── requirements.txt             # Python依赖
├── .env                         # 环境变量配置
└── README.md                    # 项目文档
```

## 🔄 数据流

### 对话流程
```
用户输入 → FastAPI接收 → Agent处理 → 工具调用 → 
  ├→ 记忆检索 (Milvus + PostgreSQL)
  ├→ 知识库搜索 (Milvus + PostgreSQL + 重排序)
  └→ LLM响应生成 → 流式返回
```

### 记忆提取流程
```
对话存储 → PostgreSQL (summary_id=NULL) → 后台任务检查 → 
Token数量超过阈值 → 记忆提取 → 冲突检测 → 
用户画像融合 → 存储到Milvus和PostgreSQL
```

### 文档处理流程
```
文档上传 → 文件去重检查 → 临时存储 → 后台处理 → 
文档解析 → 文本分块 → 向量化 → 
存储到Milvus(子块) + PostgreSQL(父块)
```

## 🚀 性能优化

### 已知性能考虑
1. **多数据库往返**：每次请求涉及PostgreSQL、Redis、Milvus多次查询
2. **内存检索**：涉及3种不同类型记忆的并行搜索
3. **知识库搜索**：包含向量搜索、PostgreSQL查询和重排序
4. **Redis连接**：当前为每个请求创建新连接，建议实现连接池
5. **耗时分析**：当前数据检索时间约为2-3秒，主要耗时在API请求上，问题嵌入1秒左右，重排序0.8-1.2秒。为降低重复查询延迟，已在提示词中规定相关性问题利用上下文回答。

### 优化策略
- 优化文件上传处理方式，可考虑使用文件存储（如S3、O3、Google Cloud Storage等）
- 实现Redis连接池
- 缓存用户画像减少PostgreSQL查询（已实现）
- 批量数据库操作（已实现）
- 延迟加载非关键记忆
- 监控生产环境中的时间日志

## 🤝 贡献指南

我们欢迎所有形式的贡献！请阅读我们的[贡献指南](CONTRIBUTING.md)了解如何参与项目开发。欢迎提交Pull Request或Issue。

### 开发流程
1. Fork 项目
2. 创建功能分支 (`git checkout -b feature/amazing-feature`)
3. 提交更改 (`git commit -m 'Add amazing feature'`)
4. 推送到分支 (`git push origin feature/amazing-feature`)
5. 创建 Pull Request

### 代码规范
- 遵循 PEP 8 代码规范
- 为新功能添加适当的测试
- 更新相关文档
- 保持代码注释的清晰和完整

## 📄 许可证

本项目采用 MIT 许可证 - 查看 [LICENSE](LICENSE) 文件了解详情。

## 📞 联系方式

- **项目维护者**: Dreamt
- **邮箱**: [mochenge@163.com](mailto:your-email@example.com)
- **GitHub**: [@Dreamt0511](https://github.com/Dreamt0511)

## 🙏 致谢

感谢以下开源项目和框架的支持：
- [SuperMew](https://github.com/icey1287/SuperMew)
- [FastAPI](https://fastapi.tiangolo.com/)
- [LangChain](https://python.langchain.com/)
- [LangGraph](https://github.com/langchain-ai/langgraph)
- [Milvus](https://milvus.io/)
- [Streamlit](https://streamlit.io/)
- [DashScope](https://dashscope.aliyun.com/)

---

<div align="center">

**[🔝 返回顶部](#echomind---个性化ai问答助手)**

Made with ❤️ by Dreamt | EchoMind

</div>