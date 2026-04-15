# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

EchoMind is a personalized AI assistant backend built with FastAPI, featuring long-term memory capabilities and knowledge base integration. The system provides intelligent responses by combining vector database retrieval, memory management, and LLM processing.

## Architecture Overview

### Core Components

1. **Agent System** (`agent.py`)
   - LangGraph-based agent with tool calling capabilities
   - Streaming response generation with Redis checkpointing
   - Dynamic prompt switching based on knowledge base selection
   - Memory and knowledge base tool integration

2. **Tool Framework** (`tools.py`)
   - `search_knowledge_base`: Hybrid retrieval from Milvus + PostgreSQL with reranking
   - `get_memory`: Multi-type memory retrieval (semantic, episodic, procedural)
   - `get_raw_conversation_by_summary_id`: Conversation history access

3. **Memory Management**
   - **Short-term**: Redis-based conversation history with TTL
   - **Long-term**: Milvus vector storage with PostgreSQL metadata
   - **Memory Types**: Summary, semantic, episodic, procedural memories
   - **Auto-extraction**: Background task extracts memories from conversation history

4. **Knowledge Base System**
   - **Storage**: Milvus for vector embeddings, PostgreSQL for document chunks
   - **Retrieval**: Hybrid search (dense + sparse) with RRF fusion
   - **Reranking**: External API-based relevance scoring with fallback

5. **Data Pipeline** (`documents_process.py`)
   - Document chunking with parent-child relationships
   - Embedding generation and vector storage
   - File deduplication using SHA-256 hashing

### Data Flow

```
User Request → FastAPI Endpoint → Agent Processing → Tool Calls →
  ├→ Memory Retrieval (Milvus + PostgreSQL)
  ├→ Knowledge Base Search (Milvus + PostgreSQL + Rerank)
  └→ LLM Response Generation → Streaming Response
```

## Key Technical Stack

- **Framework**: FastAPI with async/await
- **Agent**: LangGraph with custom middleware
- **Vector DB**: Milvus (serverless) with hybrid search
- **Metadata DB**: PostgreSQL for document chunks and conversation history
- **Cache**: Redis for session management and checkpoints
- **Embeddings**: DashScope text-embedding-v4
- **LLM**: Qwen models via DashScope API
- **Reranking**: GTE-Rerank-v2 via DashScope

## Configuration Management

### Environment Variables (`.env`)
```env
# LLM Configuration
DASHSCOPE_API_KEY=your_key
BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
AGENT_BASE_MODEL=qwen3.6-plus
SUMMARIZATION_MODEL=qwen3-max-2025-09-23

# Vector Database
Milvus_url=your_milvus_url
Token=your_milvus_token
knowledge_base_collection=knowledge_base_collection
memory_collection=memory_collection

# Traditional Databases
DATABASE_URL=postgresql://user:pass@localhost:5432/db
Redis_URI=redis://localhost:6379/0
```

### Key Configuration Files
- `config.py`: System prompts, memory extraction rules, chunking parameters
- Chunking: Parent (1000 chars) + Child (200 chars) with overlap
- Memory extraction uses detailed prompts for 6 memory types

## Development Commands

### Setup and Dependencies
```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# or
.venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt

# Environment setup
cp .env.example .env  # if exists
# Edit .env with your API keys and database URLs
```

### Running the Application
```bash
# Development server
cd backend
python main.py

# Or with uvicorn directly
uvicorn main:app --reload --host 0.0.0.0 --port 8000

# API will be available at http://localhost:8000
# API docs at http://localhost:8000/docs
```

### Testing and Validation
```bash
# Test individual components
python -c "from agent import get_or_create_agent; print('Agent import successful')"

# Test database connections
python -c "from postgresql_client import get_postgresql_client; import asyncio; asyncio.run(get_postgresql_client())"

# Test API endpoints (after starting server)
curl "http://localhost:8000/chat_with_agent/stream?query=Hello&knowledge_base_id=默认知识库&user_id=1"
```

### Database Operations
```bash
# Check PostgreSQL connection
psql postgresql://dreamt:0511@localhost:5432/echomind_db

# Check Redis connection
redis-cli -p 6379

# Milvus operations (via Python)
python -c "from milvus_client import get_milvus_client; import asyncio; asyncio.run(get_milvus_client())"
```

## Performance Considerations

### Known Bottlenecks
1. **Multiple database round-trips** per request (PostgreSQL, Redis, Milvus)
2. **Memory retrieval** involves 3 different memory type searches
3. **Knowledge base search** includes vector search + PostgreSQL lookup + reranking
4. **Redis connection** created per request instead of connection pooling

### Optimization Strategies
- Implement Redis connection pooling
- Cache user profiles to reduce PostgreSQL queries
- Batch database operations where possible
- Consider lazy loading for memory retrieval
- Monitor with timing logs in production

## API Endpoints

### Core Chat Interface
```
GET /chat_with_agent/stream
Parameters:
- query: User message
- knowledge_base_id: Knowledge base selection
- user_id: User identifier
Returns: Streaming text response
```

### Knowledge Base Management
```
POST /knowledge-bases                    # Create knowledge base
DELETE /knowledge-bases/{id}            # Delete knowledge base
GET /knowledge-bases                    # List user's knowledge bases
GET /knowledge-bases/{id}/files         # List files in knowledge base
```

### Document Management
```
POST /document_upload                    # Upload and process documents
DELETE /knowledge-bases/{id}/documents/{hash}  # Delete specific file
```

## Memory System Architecture

### Memory Types and Storage
- **Summary Memory**: Conversation summaries with importance scores
- **Semantic Memory**: Stable facts and user preferences
- **Episodic Memory**: Specific events and interactions (timestamped)
- **Procedural Memory**: Reusable workflows and methods
- **User Profile**: Dynamic user characteristics and preferences

### Memory Lifecycle
1. Conversations stored in PostgreSQL with `summary_id = NULL`
2. Background task runs every 3 minutes to check for uncompressed conversations
3. Memory extraction using lightweight LLM model
4. Similarity check against existing memories (duplicate filtering at 0.9+ similarity)
5. Vector embedding and storage in Milvus with PostgreSQL metadata

## Troubleshooting

### Common Issues
1. **Database Connection Failures**: Check PostgreSQL/Redis/Milvus service status and credentials
2. **API Key Issues**: Verify DashScope API key and model availability
3. **Memory System Errors**: Check background task logs for extraction failures
4. **Performance Issues**: Monitor database query times and network latency

### Debug Mode
```python
# Enable detailed logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Check specific component
logger = logging.getLogger('agent')  # or 'tools', 'memory_manager', etc.
logger.setLevel(logging.DEBUG)
```

## Project-Specific Notes

- **Memory-first Design**: System prioritizes personalization through comprehensive memory management
- **Multi-tier Retrieval**: Combines vector similarity, keyword matching, and reranking
- **Dynamic Prompting**: System behavior adapts based on selected knowledge base
- **Background Processing**: Memory extraction and conversation storage happen asynchronously
- **Streaming Responses**: Real-time response generation with tool call visibility
- **File Deduplication**: SHA-256 based duplicate detection prevents redundant processing