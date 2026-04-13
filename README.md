# EchoMind - 智能个性化记忆与知识库助手

<div align="center">
  <img src="https://img.shields.io/badge/Python-3.12%2B-blue" alt="Python Version">
  <img src="https://img.shields.io/badge/FastAPI-0.104%2B-green" alt="FastAPI">
  <img src="https://img.shields.io/badge/Milvus-2.4%2B-orange" alt="Milvus">
  <img src="https://img.shields.io/badge/LangChain-0.1%2B-red" alt="LangChain">
  <img src="https://img.shields.io/badge/License-MIT-purple" alt="License">
</div>

EchoMind是一个基于人工智能的智能助手系统，具备**长期记忆能力**和**知识库检索功能**。通过先进的向量数据库技术和自然语言处理，为用户提供个性化、精准的信息服务。

## 🎯 核心功能

### 1. 🧠 长期记忆系统
- **语义记忆**：存储事实性知识、概念定义、用户偏好等稳定信息
- **情景记忆**：记录用户与AI的互动历史、特定事件和对话过程
- **程序记忆**：保存可复用的工作流程、方法技巧和策略模板
- **记忆冲突检测**：自动识别并合并重复或冲突的记忆条目
- **动态访问时间更新**：智能追踪记忆使用频率，优化检索排序

### 2. 📚 知识库管理
- **多文档支持**：支持PDF、Word文档的上传和处理
- **智能分块**：采用父-子块架构，支持多层次文档切分
- **去重机制**：基于哈希算法的文件去重，避免重复存储
- **混合检索**：结合稠密向量（HNSW）和稀疏向量（BM25）检索
- **知识库隔离**：支持多用户、多知识库的权限隔离

### 3. 🤖 智能Agent
- **个性化响应**：基于用户画像和记忆历史提供定制化回答
- **工具调用能力**：支持记忆检索和知识库搜索工具
- **查询改写**：自动优化用户查询，提升检索准确性
- **流式响应**：支持实时流式输出，提升用户体验
- **多轮对话**：保持上下文连贯，支持复杂对话场景

### 4. 🔄 自动化任务
- **定时记忆提取**：每3分钟自动从对话中提取和存储记忆
- **用户画像构建**：动态提取和更新用户特征信息
- **记忆压缩**：将长对话压缩为结构化记忆条目
- **冲突解决**：自动检测和解决记忆冲突

## 🏗️ 系统架构

### 技术栈
- **后端框架**：FastAPI + Python 3.12+
- **向量数据库**：Milvus 2.4+（支持HNSW和BM25索引）
- **关系数据库**：PostgreSQL（存储元数据和文件信息）
- **AI框架**：LangChain + LangChain-OpenAI
- **嵌入模型**：DashScope Embeddings（BAAI/bge-small-zh-v1.5）
- **API服务**：Uvicorn ASGI服务器

### 数据架构
```
数据流向：用户对话 → PostgreSQL（原始消息）→ 定时任务 → 
记忆提取 → Milvus（语义/情景/程序记忆）→ 检索召回 → 
Agent回答 → 用户
```

### 核心组件
1. **MemoryManager**：记忆管理模块，负责记忆的CRUD和检索
2. **KnowledgeBaseManager**：知识库管理模块，处理文档上传和检索
3. **Agent系统**：基于LangChain的智能助手，集成记忆和知识库工具
4. **PostgreSQL Client**：关系型数据持久化，存储对话历史和文件元数据
5. **Milvus Client**：向量相似度检索，支持混合搜索和重排序

## 🚀 快速开始

### 环境要求
```bash
Python 3.12+
Milvus 2.4+
PostgreSQL 14+
```

### 安装步骤

1. **克隆项目**
```bash
git clone https://github.com/Dreamt0511/EchoMind.git
cd EchoMind/backend
```

2. **创建虚拟环境**
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或
venv\Scripts\activate     # Windows
```

3. **安装依赖**
```bash
pip install -r requirements.txt
```

4. **配置环境变量**
```bash
cp .env.example .env
# 编辑 .env 文件，配置以下关键参数：
# - Milvus连接信息
# - PostgreSQL数据库连接
# - DashScope API Key
# - 模型配置
```

5. **启动服务**
```bash
python main.py
# 服务将在 http://localhost:8000 启动
```

### API文档
启动服务后，访问 http://localhost:8000/docs 查看完整的API文档。

## 📖 使用指南

### 1. 文档上传
```http
POST /document_upload
Content-Type: multipart/form-data

file: <your-file.pdf>
knowledge_base_id: my_knowledge_base
user_id: 123
```

### 2. 知识库管理
```http
# 创建知识库
POST /knowledge-bases
knowledge_base_id: my_knowledge_base
user_id: 123

# 获取知识库列表
GET /knowledge-bases?user_id=123

# 删除知识库
DELETE /knowledge-bases/{knowledge_base_id}?user_id=123
```

### 3. 智能对话
```http
GET /chat_with_agent/stream?query=你好&knowledge_base_id=default&user_id=123&top_k=5
```

### 4. 记忆检索
系统自动在后台处理记忆提取，无需手动调用。

## ⚙️ 配置说明

### 环境变量配置

#### 数据库配置
```env
# Milvus配置
Milvus_url=http://localhost:19530
Token=your_milvus_token
memory_collection=conversation_memory
knowledge_base_collection=knowledge_base
dense_dimension=768

# PostgreSQL配置
POSTGRES_URL=postgresql://user:password@localhost:5432/echomind
```

#### AI模型配置
```env
# DashScope API配置
DASHSCOPE_API_KEY=your_api_key
EMBEDDING_MODEL=BAAI/bge-small-zh-v1.5
SUMMARIZATION_MODEL=qwen-turbo
BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
```

#### 服务配置
```env
HOST=0.0.0.0
PORT=8000
MAX_FILE_SIZE=209715200  # 200MB
```

### 文档切片配置
```python
# config.py 中的默认配置
parent_chunk_size = 1000  # 父块大小
parent_chunk_overlap = 200  # 父块重叠
child_chunk_size = 200     # 子块大小
child_chunk_overlap = 50   # 子块重叠
```

## 🧠 记忆系统详解

### 记忆类型

#### 1. 语义记忆（Semantic Memory）
存储稳定的知识、事实和概念，如技术定义、用户偏好等。

**特征：**
- 长期有效，不随时间变化
- 按主题合并，避免碎片化
- 包含丰富的元信息（定义、应用场景、关键词等）

#### 2. 情景记忆（Episodic Memory）
记录具体的事件和对话历史。

**特征：**
- 按时间顺序存储
- 包含精确的时间戳
- 记录用户行为和AI响应
- 支持事件链合并

#### 3. 程序记忆（Procedural Memory）
保存可复用的方法和流程。

**特征：**
- 完整的操作流程
- 包含触发条件、执行步骤、注意事项
- 强调可复用性和可执行性

### 记忆提取流程

1. **对话分析**：解析用户与AI的完整对话
2. **信息分类**：将内容分为摘要、语义、情景、程序四类
3. **重要性评分**：基于内容重要性和未来使用可能性评分
4. **冲突检测**：识别并合并重复或冲突的记忆
5. **向量化存储**：生成嵌入向量并存储到Milvus

### 检索机制

#### 混合检索策略
- **稠密向量检索**：基于语义相似度，使用HNSW索引
- **稀疏向量检索**：基于关键词匹配，使用BM25算法
- **重排序**：使用RRFRanker融合两种检索结果
- **时间衰减**：考虑记忆的新鲜度和访问频率

#### 检索参数
```python
# 每种记忆类型的检索数量配置
summary_k: int = 0-10    # 摘要记忆
semantic_k: int = 0-10   # 语义记忆
episodic_k: int = 0-10   # 情景记忆
procedural_k: int = 0-10 # 程序记忆
```

## 📚 知识库系统

### 文档处理流程

1. **文件上传**：支持PDF、DOCX格式，最大200MB
2. **哈希计算**：SHA256哈希，用于去重检测
3. **文档解析**：提取文本内容，保留结构信息
4. **智能分块**：
   - 父块：保持文档结构完整性
   - 子块：细粒度检索单元
5. **向量化**：生成稠密和稀疏向量
6. **存储**：Milvus向量数据库 + PostgreSQL元数据

### 检索优化

#### 混合检索
```python
# 同时使用稠密和稀疏检索
dense_req = AnnSearchRequest(
    data=[dense_vector],
    anns_field="dense_vector",
    param={"metric_type": "COSINE", "params": {"ef": 64}},
    limit=top_k * 7
)

sparse_req = AnnSearchRequest(
    data=[query],
    anns_field="sparse_bm25",
    param={"metric_type": "BM25"},
    limit=top_k * 3
)
```

#### 结果重排序
使用RRFRanker融合两种检索结果，平衡语义相似度和关键词匹配。

## 🤖 Agent系统

### 系统提示词设计

#### 默认模式（SYSTEM_PROMPT_DEFAULT）
- 记忆优先：优先使用记忆库信息
- 常识直接回答：通用知识基于训练数据
- 专有知识检索：必须调用工具获取
- 融合回答：结合记忆和检索结果

#### 指定知识库模式（SYSTEM_PROMPT_SPECIFIC）
- 绝对依赖检索：所有信息必须来自工具
- 记忆优先于知识库：先检索记忆再检索知识库
- 零发散原则：严格基于工具返回内容

### 查询改写规则

1. **去口语化**：去除无意义词汇
2. **去情绪化**：忽略情绪表达，提取核心诉求
3. **实体明确化**：将模糊指代转化为具体关键词
4. **陈述式转换**：将反问转为陈述式查询
5. **保留术语**：严格保留业务和技术术语

### 决策流程
```
用户输入 → 意图判断 → 检索记忆 → 判断是否需要知识库 → 
执行检索 → 生成回答 → 返回用户
```

## 🔧 高级特性

### 1. 记忆冲突检测
使用向量相似度检测重复记忆，相似度≥0.9时自动合并。

### 2. 用户画像动态更新
```python
# 用户画像格式
用户名为[姓名]，[身份背景]，[技术能力]，[兴趣领域]，[当前目标]，[行为偏好]
```

### 3. 时间衰减机制
记忆评分考虑访问时间，近期访问的记忆获得更高权重。

### 4. 错误处理和降级
- Embedding API失败时自动重试
- 混合检索失败时降级为稠密向量检索
- 服务异常时优雅降级，不影响主要功能

## 📊 性能优化

### 1. 异步处理
- 文档上传：流式处理，实时计算哈希
- 记忆提取：后台定时任务，不影响实时对话
- 批量操作：支持批量插入和检索

### 2. 缓存策略
- 禁用静态资源缓存，确保前端更新立即生效
- 记忆访问时间缓存，减少数据库访问

### 3. 并发控制
- 信号量控制并发上传数量（默认10）
- 异步任务队列，避免阻塞主线程

## 🔒 安全特性

### 1. 数据安全
- 文件哈希去重，避免重复存储
- 用户ID隔离，确保数据隐私
- 支持客户端断开检测，及时清理临时文件

### 2. API安全
- CORS配置，支持跨域访问
- 文件大小限制，防止恶意上传
- 异常处理，避免信息泄露

## 🧪 测试

### 运行测试
```bash
# 运行示例测试
python test/查看集合分区的脚本.py
python test/check_data_in_redis.py

# 手动测试API
curl -X POST http://localhost:8000/document_upload \
  -F "file=@test.pdf" \
  -F "knowledge_base_id=test" \
  -F "user_id=1"
```

## 📈 监控与日志

### 日志配置
```python
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s-%(name)s-%(levelname)s-%(message)s'
)
```

### 关键指标
- 文件上传成功率
- 记忆提取准确率
- 检索响应时间
- 系统资源使用

## 🚀 部署建议

### 生产环境配置
1. **使用Gunicorn**：提高并发处理能力
2. **配置Nginx**：反向代理和负载均衡
3. **监控告警**：集成Prometheus + Grafana
4. **数据备份**：定期备份PostgreSQL和Milvus数据

### Docker部署
```dockerfile
FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["python", "main.py"]
```

## 🤝 贡献指南

1. Fork项目
2. 创建特性分支
3. 提交代码
4. 创建Pull Request

## 📄 许可证

MIT License

## 📞 联系方式

- 项目地址：https://github.com/Dreamt0511/EchoMind
- 问题反馈：https://github.com/Dreamt0511/EchoMind/issues
- 邮箱：contact@echomind.ai

## 🙏 致谢

- LangChain社区：提供强大的AI框架
- Milvus团队：提供高性能向量数据库
- FastAPI团队：提供现代化的API框架
- 所有贡献者：感谢你们的支持和贡献

---

**EchoMind** - 让AI拥有记忆，让知识触手可及 🚀