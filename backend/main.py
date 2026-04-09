import api
import uvicorn
import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from milvus_client import get_milvus_client
from postgresql_client import get_postgresql_client
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s-%(name)s-%(levelname)s-%(message)s'
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    # 启动时初始化
    logger.info("=" * 50)
    logger.info("应用启动，开始初始化连接...")
    logger.info("=" * 50)
    
    # 初始化标志
    milvus_initialized = False
    postgresql_initialized = False
    
    try:
        # 初始化 PostgreSQL 客户端,保存到app.state
        app.state.postgresql_client = await get_postgresql_client()
        postgresql_initialized = True
        logger.info("【PostgreSQL】 客户端初始化完成")
        
        # 初始化 Milvus 客户端,保存到app.state
        try:
            app.state.milvus_client = await get_milvus_client()
            milvus_initialized = True
            logger.info("【Milvus】 客户端初始化完成")
        except Exception as e:
            logger.error(f"Milvus 客户端初始化失败: {e}")
            # Milvus 初始化失败不影响应用启动，但记录错误
            app.state.milvus_client = None
        
        logger.info("=" * 50)
        logger.info("所有连接初始化完成，应用已就绪")
        logger.info("=" * 50)
        
    except Exception as e:
        logger.error(f"初始化失败: {e}")
        raise
    
    yield  # 应用运行期间
    
    # 关闭时清理
    logger.info("=" * 50)
    logger.info("应用关闭，清理连接...")
    logger.info("=" * 50)
    
    try:
        # 关闭 Milvus 连接
        if milvus_initialized and hasattr(app.state, 'milvus_client') and app.state.milvus_client:
            await app.state.milvus_client.close()
            logger.info("✓ Milvus 连接已关闭")
        
        # 关闭 PostgreSQL 连接
        if postgresql_initialized and hasattr(app.state, 'postgresql_client') and app.state.postgresql_client:
            await app.state.postgresql_client.close()
            logger.info("✓ PostgreSQL 连接已关闭")
            
    except Exception as e:
        logger.error(f"清理连接时出错: {e}")
    
    logger.info("应用已关闭")


# 创建应用并传入 lifespan
app = FastAPI(title="EchoMind-个性化问答助手", lifespan=lifespan)

# 将项目中定义的所有 API 端点注册到应用中
app.include_router(api.router)

# 解决前端跨域访问问题，让浏览器可以正常调用后端 API。
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 允许所有来源
    allow_credentials=True,  # 允许携带凭证
    allow_methods=["*"],  # 允许所有 HTTP 方法
    allow_headers=["*"],  # 允许所有请求头
)


# 在开发环境中禁用静态资源缓存，确保前端代码修改后能立即生效。
@app.middleware("http")
async def _no_cache(request, call_next):
    response = await call_next(request)
    path = request.url.path or ""
    if path == "/" or path.endswith((".html", ".js", ".css")):
        response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
    return response


if __name__ == "__main__":
    uvicorn.run(
        app,
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", 8000)),
    )