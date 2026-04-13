# app.py - 修复异步加载问题版本
import streamlit as st
import pandas as pd
from datetime import datetime
import os
import aiohttp
import asyncio
import nest_asyncio
import hashlib
import time
import json
from typing import Dict, List, Optional
from concurrent.futures import ThreadPoolExecutor
from threading import Thread

nest_asyncio.apply()

# 用于线程间传递错误信息
_pending_upload_error = None

# 用于线程间传递创建结果
_create_success = None
_create_exists = None
_create_error = None

# 页面配置
st.set_page_config(
    page_title="EchoMind - 个性化AI问答助手",
    layout="wide",
    initial_sidebar_state="auto"
)

# ==========================================
# 加载CSS（已分离到 style.css）
# ==========================================
def load_css():
    """从外部文件加载CSS样式"""
    css_content = ""
    css_file = "style.css"
    if os.path.exists(css_file):
        with open(css_file, 'r', encoding='utf-8') as f:
            css_content = f.read()
    return f"<style>{css_content}</style>"

def add_sidebar_collapse_listener():
        """监听侧边栏折叠，隐藏机器人图标"""
        return """
        <script>
        function toggleRobotVisibility() {
            const sidebar = document.querySelector('[data-testid="stSidebar"]');
            const robot = document.querySelector('.robot-icon-container');
            if (!sidebar || !robot) return;
            
            // 检查侧边栏是否折叠（宽度小于50px表示折叠）
            const isCollapsed = sidebar.offsetWidth < 50;
            robot.style.display = isCollapsed ? 'none' : '';
        }
        
        // 初始检查
        setTimeout(toggleRobotVisibility, 100);
        
        // 监听窗口大小变化和侧边栏变化
        const observer = new MutationObserver(toggleRobotVisibility);
        const sidebar = document.querySelector('[data-testid="stSidebar"]');
        if (sidebar) {
            observer.observe(sidebar, { attributes: true, childList: true, subtree: true });
        }
        window.addEventListener('resize', toggleRobotVisibility);
        </script>
        """
st.markdown(load_css(), unsafe_allow_html=True)
st.markdown(add_sidebar_collapse_listener(), unsafe_allow_html=True)


# 后端API地址
API_BASE_URL = "http://localhost:8000"
USER_ID = 1

# ==========================================
# 初始化会话状态（修复版）
# ==========================================
def init_session_state():
    """初始化所有session state，避免KeyError"""
    defaults = {
        'chat_history': [],
        'knowledge_bases': {},    # 空字典，与加载完成后的格式一致
        'knowledge_base_files_info': {},
        'selected_kb': None,
        'pending_delete': None,
        'streaming': False,
        'current_response': "",
        'is_streaming': False,
        'initialized': False,
        'loading': True,
        'operation_in_progress': False,
        # 异步加载相关状态
        'data_loading': True,
        'kb_loading_status': {},
        'pending_operations': [],
        'last_refresh_time': 0,
        'optimistic_files': {},
        'pending_upload': None,
        'needs_rerun': False,
        'files_loaded': False,
        'load_error': None,  # 加载错误信息
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

init_session_state()

# 延迟执行 rerun，确保在脚本执行完成后
if st.session_state.get('needs_rerun'):
    st.session_state.needs_rerun = False
    st.rerun()

# ==========================================
# 骨架屏渲染函数
# ==========================================

def render_sidebar_skeleton():
    """渲染侧边栏骨架屏"""
    st.markdown("""
    <div style="padding: 20px 0;">
        <div class="skeleton skeleton-title"></div>
        <div class="skeleton skeleton-text" style="width: 80%;"></div>
        <div style="margin: 20px 0;">
            <div class="skeleton skeleton-card"></div>
            <div class="skeleton skeleton-card"></div>
        </div>
        <div style="margin: 20px 0;">
            <div class="skeleton skeleton-title" style="width: 40%;"></div>
            <div class="kb-skeleton-item"></div>
            <div class="kb-skeleton-item"></div>
            <div class="kb-skeleton-item"></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

def render_chat_skeleton():
    """渲染聊天区域骨架屏"""
    st.markdown("""
    <div style="padding: 40px 20px; max-width: 800px; margin: 0 auto;">
        <div style="display: flex; margin: 20px 0; align-items: flex-start;">
            <div class="skeleton skeleton-circle" style="margin-right: 12px; flex-shrink: 0;"></div>
            <div style="flex: 1;">
                <div class="skeleton skeleton-text" style="width: 90%;"></div>
                <div class="skeleton skeleton-text" style="width: 70%;"></div>
            </div>
        </div>
        <div style="display: flex; margin: 20px 0; align-items: flex-start; justify-content: flex-end;">
            <div style="flex: 1; margin-right: 12px;">
                <div class="skeleton skeleton-text" style="width: 80%; margin-left: auto;"></div>
                <div class="skeleton skeleton-text" style="width: 50%; margin-left: auto;"></div>
            </div>
            <div class="skeleton skeleton-circle" style="flex-shrink: 0;"></div>
        </div>
        <div style="display: flex; margin: 20px 0; align-items: flex-start;">
            <div class="skeleton skeleton-circle" style="margin-right: 12px; flex-shrink: 0;"></div>
            <div style="flex: 1;">
                <div class="skeleton skeleton-text" style="width: 95%;"></div>
                <div class="skeleton skeleton-text" style="width: 60%;"></div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ==========================================
# 异步数据加载函数（修复 session state 访问问题）
# ==========================================

async def fetch_files_async(session: aiohttp.ClientSession, kb_id: str, user_id: int) -> Dict:
    """异步获取单个知识库的文件列表"""
    try:
        url = f"{API_BASE_URL}/knowledge-bases/{kb_id}/files"
        params = {"user_id": user_id}

        async with session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=10)) as response:
            if response.status == 200:
                data = await response.json()

                if data.get("status") == "success":
                    files = data.get("files", [])
                elif "files" in data:
                    files = data.get("files", [])
                else:
                    files = []

                file_list = []
                file_info_dict = {}

                for f in files:
                    if isinstance(f, dict):
                        file_name = f.get("file_name", "")
                        file_hash = f.get("file_hash", "")
                        if file_name:
                            # 存储原始文件名（编码后的），显示时再解码
                            file_list.append(file_name)
                            file_info_dict[file_name] = file_hash
                    else:
                        file_str = str(f)
                        file_list.append(file_str)
                        file_info_dict[file_str] = ""

                return {
                    "kb_id": kb_id,
                    "files": file_list,
                    "files_info": file_info_dict
                }

            return {"kb_id": kb_id, "files": [], "files_info": {}}
    except Exception as e:
        print(f"异步获取知识库 {kb_id} 文件失败: {e}")
        return {"kb_id": kb_id, "files": [], "files_info": {}}

async def load_all_data_async():
    """
    异步加载所有数据：先加载知识库列表，再并发加载文件
    修复：不直接访问 st.session_state，而是通过返回值更新
    """
    result = {
        'knowledge_bases': {},
        'kb_loading_status': {},
        'knowledge_base_files_info': {},
        'selected_kb': None,
        'error': None
    }
    
    try:
        timeout = aiohttp.ClientTimeout(total=10)

        async with aiohttp.ClientSession(timeout=timeout) as session:
            # 1. 异步获取知识库列表
            url = f"{API_BASE_URL}/knowledge-bases"
            params = {"user_id": USER_ID}

            try:
                async with session.get(url, params=params) as response:
                    if response.status != 200:
                        result['error'] = f"获取知识库列表失败: HTTP {response.status}"
                        return result

                    data = await response.json()
            except asyncio.TimeoutError:
                result['error'] = "获取知识库列表超时"
                return result

            # 解析知识库列表
            knowledge_bases = []
            if isinstance(data, dict):
                if "knowledge_bases" in data:
                    knowledge_bases = data.get("knowledge_bases", [])
                elif "data" in data:
                    knowledge_bases = data.get("data", [])
            elif isinstance(data, list):
                knowledge_bases = data

            if not knowledge_bases:
                knowledge_bases = ["默认知识库"]

            # 2. 初始化知识库结构
            new_knowledge_bases = {}
            new_kb_loading_status = {}

            for kb in knowledge_bases:
                if isinstance(kb, dict):
                    kb_id = kb.get("knowledge_base_id") or kb.get("name") or kb.get("id")
                    if not kb_id:
                        continue
                else:
                    kb_id = str(kb)

                if kb_id:
                    new_knowledge_bases[kb_id] = []
                    new_kb_loading_status[kb_id] = True

            if "默认知识库" not in new_knowledge_bases:
                new_knowledge_bases["默认知识库"] = []
                new_kb_loading_status["默认知识库"] = True

            result['knowledge_bases'] = new_knowledge_bases
            result['kb_loading_status'] = new_kb_loading_status

            # 设置默认选中的知识库
            if new_knowledge_bases:
                result['selected_kb'] = "默认知识库" if "默认知识库" in new_knowledge_bases else list(new_knowledge_bases.keys())[0]

            # 3. 并发异步加载所有知识库的文件
            tasks = [
                fetch_files_async(session, kb_id, USER_ID)
                for kb_id in new_knowledge_bases.keys()
            ]

            results = await asyncio.gather(*tasks, return_exceptions=True)

            # 更新文件信息
            for res in results:
                if isinstance(res, Exception):
                    print(f"加载文件时发生错误: {res}")
                    continue

                kb_id = res.get("kb_id")
                if kb_id and kb_id in result['knowledge_bases']:
                    result['knowledge_bases'][kb_id] = res["files"]
                    result['knowledge_base_files_info'][kb_id] = res["files_info"]
                    result['kb_loading_status'][kb_id] = False

        return result

    except Exception as e:
        print(f"异步加载数据失败: {e}")
        result['error'] = str(e)
        return result

def start_async_load():
    """启动异步加载（在后台线程中运行，通过 st.rerun 触发主线程更新）"""
    def run_async():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(load_all_data_async())
        loop.close()
        
        # 在主线程中更新 session state（通过 st.rerun 触发）
        # 使用 st.toast 需要在主线程中执行，这里只能标记需要更新
        # 将结果保存到临时变量，通过 rerun 后的主线程检查
        if hasattr(st, '_async_load_result'):
            st._async_load_result = result
        else:
            st._async_load_result = result
        
        # 触发 rerun（需要在主线程中执行，这里只是标记）
        # 注意：Thread 中不能直接调用 st.rerun()
        # 改用设置标志的方式
        
    thread = Thread(target=run_async, daemon=True)
    thread.start()

def apply_async_load_result():
    """在主线程中应用异步加载结果"""
    if hasattr(st, '_async_load_result') and st._async_load_result is not None:
        result = st._async_load_result
        
        if result.get('error'):
            st.session_state.load_error = result['error']
        else:
            if result.get('knowledge_bases'):
                st.session_state.knowledge_bases = result['knowledge_bases']
            if result.get('knowledge_base_files_info'):
                st.session_state.knowledge_base_files_info = result['knowledge_base_files_info']
            if result.get('kb_loading_status'):
                st.session_state.kb_loading_status = result['kb_loading_status']
            if result.get('selected_kb') and st.session_state.selected_kb is None:
                st.session_state.selected_kb = result['selected_kb']
        
        st.session_state.data_loading = False
        st.session_state.files_loaded = True
        st.session_state.last_refresh_time = time.time()
        
        # 清除结果标记
        st._async_load_result = None
        return True
    
    return False

async def refresh_single_knowledge_base_async(knowledge_base_id: str) -> Dict:
    """异步刷新单个知识库"""
    try:
        timeout = aiohttp.ClientTimeout(total=10)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            result = await fetch_files_async(session, knowledge_base_id, USER_ID)
            return result
    except Exception as e:
        print(f"异步刷新知识库 {knowledge_base_id} 失败: {e}")
        return {"kb_id": knowledge_base_id, "files": [], "files_info": {}, "error": str(e)}

def start_refresh_single_kb(kb_id: str):
    """启动单个知识库的异步刷新"""
    def run_async():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(refresh_single_knowledge_base_async(kb_id))
        loop.close()
        
        # 保存结果
        if hasattr(st, '_single_refresh_result'):
            st._single_refresh_result = result
        else:
            st._single_refresh_result = result
        
        # 立即标记需要刷新
        if hasattr(st, '_needs_immediate_refresh'):
            st._needs_immediate_refresh = True
        
    thread = Thread(target=run_async, daemon=True)
    thread.start()

def apply_single_refresh_result():
    """应用单个知识库刷新结果"""
    if hasattr(st, '_single_refresh_result') and st._single_refresh_result is not None:
        result = st._single_refresh_result
        kb_id = result.get("kb_id")
        
        if kb_id and kb_id in st.session_state.knowledge_bases:
            st.session_state.knowledge_bases[kb_id] = result["files"]
            st.session_state.knowledge_base_files_info[kb_id] = result["files_info"]
            st.session_state.kb_loading_status[kb_id] = False
            
            # 清除该知识库的乐观更新状态
            if kb_id in st.session_state.optimistic_files:
                del st.session_state.optimistic_files[kb_id]
            
            st.session_state.needs_rerun = True
        
        st._single_refresh_result = None
        return True
    
    return False

# ==========================================
# API 调用函数
# ==========================================

async def upload_document_async(knowledge_base_id, file_name, file_content, file_type):
    """异步上传文档"""
    try:
        url = f"{API_BASE_URL}/document_upload"

        data = aiohttp.FormData()
        data.add_field('file', file_content, filename=file_name, content_type=file_type)
        data.add_field('knowledge_base_id', knowledge_base_id)
        data.add_field('user_id', str(USER_ID))

        timeout = aiohttp.ClientTimeout(total=60)

        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(url, data=data) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    return {"status": "error", "message": f"HTTP {response.status}"}
    except Exception as e:
        return {"status": "error", "message": str(e)}

def upload_document_api_optimized(knowledge_base_id, file):
    """优化的上传：乐观更新 + 后台异步同步"""
    file_name = file.name

    # 1. 乐观更新 - 在主线程中完成
    if knowledge_base_id not in st.session_state.optimistic_files:
        st.session_state.optimistic_files[knowledge_base_id] = []

    optimistic_entry = {
        "name": file_name,
        "status": "uploading",
        "timestamp": time.time()
    }
    st.session_state.optimistic_files[knowledge_base_id].append(optimistic_entry)

    if file_name not in st.session_state.knowledge_bases.get(knowledge_base_id, []):
        st.session_state.knowledge_bases[knowledge_base_id].append(file_name)

    st.session_state.needs_rerun = True

    # 2. 后台异步上传 - 线程中不访问 st.session_state
    def run_upload(kb_id, f_name, f_content, f_type):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(
            upload_document_async(kb_id, f_name, f_content, f_type)
        )
        loop.close()

        # 上传完成后刷新知识库（使用线程安全的标记方式）
        start_refresh_single_kb(kb_id)
        
        # 标记上传结果，用于显示提示
        if result.get("status") == "error":
            # 保存错误信息，让主线程处理
            import streamlit as st_runtime
            # 使用模块级变量传递（不推荐但可行）
            global _pending_upload_error
            _pending_upload_error = (kb_id, f_name, result.get('message', '未知错误'))

    # 预先读取文件内容，避免在线程中访问 Streamlit 的 file 对象
    file.seek(0)
    file_content = file.read()
    
    thread = Thread(target=run_upload, args=(knowledge_base_id, file_name, file_content, file.type), daemon=True)
    thread.start()
    return True

async def delete_file_async(knowledge_base_id: str, file_hash: str) -> Dict:
    """异步删除文件"""
    try:
        url = f"{API_BASE_URL}/knowledge-bases/{knowledge_base_id}/documents/{file_hash}"
        params = {"user_id": USER_ID}

        timeout = aiohttp.ClientTimeout(total=10)

        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.delete(url, params=params) as response:
                if response.status == 200:
                    return await response.json()
                elif response.status == 400:
                    return {"status": "error", "message": "文件不存在或无权删除"}
                else:
                    return {"status": "error", "message": f"HTTP {response.status}"}
    except Exception as e:
        return {"status": "error", "message": str(e)}

def delete_file_api_optimized(knowledge_base_id, file_name, file_hash):
    """优化的删除：乐观更新 + 后台异步确认"""
    if not file_hash:
        st.error(f"❌ 无法获取文件 {file_name} 的哈希值")
        return False

    # 1. 乐观更新
    if file_name in st.session_state.knowledge_bases.get(knowledge_base_id, []):
        st.session_state.knowledge_bases[knowledge_base_id].remove(file_name)

    if knowledge_base_id in st.session_state.knowledge_base_files_info:
        if file_name in st.session_state.knowledge_base_files_info[knowledge_base_id]:
            del st.session_state.knowledge_base_files_info[knowledge_base_id][file_name]

    st.session_state.needs_rerun = True

    # 2. 后台异步删除
    def run_delete():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(delete_file_async(knowledge_base_id, file_hash))
        loop.close()

        if result.get("status") == "success":
            start_refresh_single_kb(knowledge_base_id)
        else:
            # 记录错误，主线程中显示
            if hasattr(st, '_delete_error'):
                st._delete_error = (knowledge_base_id, file_name, result.get('message', '未知错误'))

    thread = Thread(target=run_delete, daemon=True)
    thread.start()
    return True

async def create_knowledge_base_async(knowledge_base_id: str) -> Dict:
    """异步创建知识库"""
    try:
        url = f"{API_BASE_URL}/knowledge-bases"
        params = {
            "knowledge_base_id": knowledge_base_id,
            "user_id": USER_ID
        }

        timeout = aiohttp.ClientTimeout(total=5)

        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(url, params=params) as response:
                if response.status == 200:
                    return await response.json()
                elif response.status == 409:
                    return {"status": "exists", "message": "知识库已存在"}
                else:
                    return {"status": "error", "message": f"HTTP {response.status}"}
    except Exception as e:
        return {"status": "error", "message": str(e)}

def create_knowledge_base_api_optimized(knowledge_base_id):
    """优化的创建：立即显示 + 后台异步验证"""
    if knowledge_base_id == "默认知识库":
        st.warning("⚠️ '默认知识库' 是系统保留名称，请使用其他名称")
        return False

    # 检查是否已存在
    if knowledge_base_id in st.session_state.knowledge_bases:
        st.warning(f"⚠️ 知识库 '{knowledge_base_id}' 已存在")
        return False

    # 乐观更新
    st.session_state.knowledge_bases[knowledge_base_id] = []
    st.session_state.kb_loading_status[knowledge_base_id] = False
    st.session_state.needs_rerun = True

    # 后台异步创建
    def run_create(kb_id):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(create_knowledge_base_async(kb_id))
        loop.close()

        if result.get("status") == "success":
            # 创建成功，刷新
            global _create_success
            _create_success = kb_id
        elif result.get("status") == "exists":
            # 知识库已存在，需要回滚
            global _create_exists
            _create_exists = kb_id
        else:
            global _create_error
            _create_error = kb_id

    thread = Thread(target=run_create, args=(knowledge_base_id,), daemon=True)
    thread.start()
    return True

async def delete_knowledge_base_async(knowledge_base_id: str) -> Dict:
    """异步删除知识库"""
    try:
        url = f"{API_BASE_URL}/knowledge-bases/{knowledge_base_id}"
        params = {"user_id": USER_ID}

        timeout = aiohttp.ClientTimeout(total=5)

        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.delete(url, params=params) as response:
                if response.status == 200:
                    return await response.json()
                elif response.status == 404:
                    return {"status": "success", "message": "知识库已删除"}
                else:
                    return {"status": "error", "message": f"HTTP {response.status}"}
    except Exception as e:
        return {"status": "error", "message": str(e)}

def delete_knowledge_base_api_optimized(knowledge_base_id):
    """优化的删除：立即移除 + 后台异步确认"""
    if knowledge_base_id == "默认知识库":
        st.warning("⚠️ 默认知识库不可删除")
        return False

    # 备份
    files_backup = st.session_state.knowledge_bases.get(knowledge_base_id, []).copy()
    info_backup = st.session_state.knowledge_base_files_info.get(knowledge_base_id, {}).copy()

    # 乐观更新
    if knowledge_base_id in st.session_state.knowledge_bases:
        del st.session_state.knowledge_bases[knowledge_base_id]
    if knowledge_base_id in st.session_state.knowledge_base_files_info:
        del st.session_state.knowledge_base_files_info[knowledge_base_id]

    if st.session_state.selected_kb == knowledge_base_id:
        remaining = list(st.session_state.knowledge_bases.keys())
        st.session_state.selected_kb = remaining[0] if remaining else None

    st.session_state.needs_rerun = True

    # 后台异步删除
    def run_delete():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(delete_knowledge_base_async(knowledge_base_id))
        loop.close()

        if result.get("status") != "success":
            # 回滚标记
            if hasattr(st, '_delete_kb_error'):
                st._delete_kb_error = (knowledge_base_id, files_backup, info_backup, result.get('message', '未知错误'))

    thread = Thread(target=run_delete, daemon=True)
    thread.start()
    return True

# ==========================================
# 辅助函数
# ==========================================

def rollback_optimistic_update(knowledge_base_id, file_name):
    """回滚乐观更新"""
    if knowledge_base_id in st.session_state.optimistic_files:
        st.session_state.optimistic_files[knowledge_base_id] = [
            f for f in st.session_state.optimistic_files[knowledge_base_id] 
            if f["name"] != file_name
        ]

    if file_name in st.session_state.knowledge_bases.get(knowledge_base_id, []):
        st.session_state.knowledge_bases[knowledge_base_id].remove(file_name)

    st.session_state.needs_rerun = True

def handle_send(input_text=None):
    query = input_text or st.session_state.get("user_query_input")
    if not query or not query.strip():
        return

    st.session_state.chat_history.append({
        "role": "user",
        "content": query.strip(),
        "timestamp": datetime.now().strftime("%H:%M")
    })

    st.session_state.is_streaming = True
    st.session_state.current_response = ""
    st.rerun()

def clear_chat_history():
    st.session_state.chat_history = []
    st.toast("✅ 聊天历史已清空")

def request_delete(delete_type, kb_name, file_name=None, file_hash=None):
    if delete_type == 'kb' and kb_name == "默认知识库":
        st.warning("⚠️ 默认知识库不可删除")
        return
    st.session_state.pending_delete = {
        'type': delete_type, 
        'kb_name': kb_name, 
        'file_name': file_name,
        'file_hash': file_hash
    }

def confirm_delete():
    if st.session_state.pending_delete:
        info = st.session_state.pending_delete
        try:
            success = False
            if info['type'] == 'file':
                success = delete_file_api_optimized(
                    knowledge_base_id=info['kb_name'],
                    file_name=info['file_name'],
                    file_hash=info['file_hash']
                )
            elif info['type'] == 'kb':
                success = delete_knowledge_base_api_optimized(info['kb_name'])

            if success:
                st.session_state.pending_delete = None
                st.rerun()
                return
            else:
                st.error("❌ 删除失败")
        except Exception as e:
            st.error(f"❌ 异常：{e}")
        finally:
            st.session_state.pending_delete = None
            st.rerun()

def cancel_delete():
    st.session_state.pending_delete = None
    st.rerun()

def render_message(role, content, timestamp):
    if role == "user":
        avatar = "👤"
        container_class = "user-message"
        bubble_class = "user-bubble"
        timestamp_class = "user-timestamp"
    else:
        avatar = "🤖"
        container_class = "assistant-message"
        bubble_class = "assistant-bubble"
        timestamp_class = "assistant-timestamp"

    from markdown import markdown
    try:
        rendered = markdown(content, extensions=['fenced_code', 'tables'])
    except:
        rendered = content

    return f"""
    <div class="message-container {container_class} fade-in">
        <div class="message-avatar {role}-avatar">{avatar}</div>
        <div class="message-bubble {bubble_class}">
            <div class="message-content">{rendered}</div>
            <div class="message-timestamp {timestamp_class}">
                <span>{'你' if role == 'user' else 'EchoMind'}</span>
                <span>•</span>
                <span>{timestamp}</span>
            </div>
        </div>
    </div>
    """

def render_loading_indicator():
    """渲染加载动画"""
    return """
    <div style="display: flex; align-items: center; gap: 8px; padding: 8px 16px; 
                background: rgba(102,126,234,0.1); border-radius: 20px; width: fit-content;">
        <div class="loading-pulse"></div>
        <div class="loading-pulse"></div>
        <div class="loading-pulse"></div>
        <span style="color: #667eea; font-size: 14px; margin-left: 8px;">加载中...</span>
    </div>
    """

# ==========================================
# 处理异步回调结果
# ==========================================

def process_async_results():
    """处理所有异步操作的结果"""
    rerun_needed = False
    
    # 1. 应用初始数据加载结果
    if apply_async_load_result():
        rerun_needed = True
    
    # 2. 应用单知识库刷新结果
    if apply_single_refresh_result():
        rerun_needed = True
    
    # 3. 处理上传成功
    if hasattr(st, '_upload_success') and st._upload_success is not None:
        kb_id, file_name = st._upload_success
        from urllib.parse import unquote
        display_name = unquote(file_name)
        st.toast(f"✅ 文件 '{display_name}' 上传成功！")
        st._upload_success = None
        rerun_needed = True
    
   # 4. 处理上传错误
    if hasattr(st, '_upload_error') and st._upload_error is not None:
        kb_id, file_name, msg = st._upload_error
        rollback_optimistic_update(kb_id, file_name)
        st.error(f"❌ 上传失败: {msg}")
        st._upload_error = None
        rerun_needed = True
    
    # 4.1 处理全局上传错误
    global _pending_upload_error
    if _pending_upload_error is not None:
        kb_id, file_name, msg = _pending_upload_error
        rollback_optimistic_update(kb_id, file_name)
        st.error(f"❌ 上传失败: {msg}")
        _pending_upload_error = None
        rerun_needed = True
    
    # 4. 处理删除错误
    if hasattr(st, '_delete_error') and st._delete_error is not None:
        kb_id, file_name, msg = st._delete_error
        st.error(f"❌ 删除失败: {msg}")
        st._delete_error = None
        rerun_needed = True
    
    # 5. 处理创建成功
    global _create_success
    if _create_success is not None:
        kb_id = _create_success
        st.toast(f"✅ 知识库 '{kb_id}' 创建成功")
        st.session_state.selected_kb = kb_id
        _create_success = None
        rerun_needed = True
    
    # 6. 处理知识库已存在
    global _create_exists
    if _create_exists is not None:
        kb_id = _create_exists
        # 回滚：删除乐观创建的知识库
        if kb_id in st.session_state.knowledge_bases:
            del st.session_state.knowledge_bases[kb_id]
        st.warning(f"⚠️ 知识库 '{kb_id}' 已存在")
        _create_exists = None
        rerun_needed = True
    
    # 7. 处理创建错误
    global _create_error
    if _create_error is not None:
        kb_id = _create_error
        # 回滚：删除乐观创建的知识库
        if kb_id in st.session_state.knowledge_bases:
            del st.session_state.knowledge_bases[kb_id]
        st.error(f"❌ 创建知识库 '{kb_id}' 失败")
        _create_error = None
        rerun_needed = True
    
    # 8. 处理删除知识库错误
    if hasattr(st, '_delete_kb_error') and st._delete_kb_error is not None:
        kb_id, files_backup, info_backup, msg = st._delete_kb_error
        # 回滚
        st.session_state.knowledge_bases[kb_id] = files_backup
        st.session_state.knowledge_base_files_info[kb_id] = info_backup
        st.error(f"❌ 删除知识库失败: {msg}")
        st._delete_kb_error = None
        rerun_needed = True
    
    if rerun_needed:
        st.rerun()

# ==========================================
# 初始化（异步加载策略 - 修复版）
# ==========================================

# 处理异步回调结果
process_async_results()
# 添加强制刷新检查
if hasattr(st, '_needs_immediate_refresh') and st._needs_immediate_refresh:
    st._needs_immediate_refresh = False
    st.rerun()
# 首次加载：启动后台异步加载，立即显示骨架屏
if not st.session_state.initialized:
    st.session_state.initialized = True
    start_async_load()
    st.rerun()

# 自动刷新机制：数据加载中时自动轮询刷新
if st.session_state.get('data_loading') and not st.session_state.get('files_loaded'):
    # 显示加载提示
    loading_placeholder = st.sidebar.empty()
    with loading_placeholder.container():
        st.caption("🔄 正在加载知识库数据...")
    # 短暂延迟后自动刷新
    import time
    time.sleep(0.3)
    st.rerun()

# 数据加载完成后触发刷新
if st.session_state.get('needs_rerun'):
    st.session_state.needs_rerun = False
    st.rerun()

# ==========================================
# 侧边栏 UI（支持骨架屏和异步加载）
# ==========================================
with st.sidebar:
    st.markdown("""
    <div class="main-header">
        <div class="header-container">
            <div class="title-section">
                <h1>EchoMind</h1>
                <p class="subtitle">个性化智能问答助手</p>
            </div>
            <div class="robot-icon-container">
                <div class="robot-3d">
                    <div class="robot-sphere">
                        <div class="robot-face">
                            <div class="robot-visors">
                                <div class="visor left-visor"><div class="visor-glow"></div></div>
                                <div class="visor right-visor"><div class="visor-glow"></div></div>
                            </div>
                            <div class="robot-mouth">
                                <div class="mouth-line"></div>
                                <div class="mouth-dot"></div>
                            </div>
                        </div>
                        <div class="robot-ring">
                            <div class="ring-particle"></div>
                            <div class="ring-particle"></div>
                            <div class="ring-particle"></div>
                            <div class="ring-particle"></div>
                            <div class="ring-particle"></div>
                            <div class="ring-particle"></div>
                        </div>
                    </div>
                    <div class="robot-energy">
                        <div class="energy-wave"></div>
                        <div class="energy-wave"></div>
                        <div class="energy-wave"></div>
                    </div>
                </div>
                <div class="ai-status">
                    <span class="status-dot"></span>
                    <span class="status-text">Active</span>
                </div>
                <div class="robot-tooltip-external">
                    <div class="tooltip-title">
                        <span class="memory-icon">🧠</span>
                        你好，我是 EchoMind
                    </div>
                    <div class="tooltip-divider"></div>
                    <div class="tooltip-content">
                        我具备<span style="color:#ffd700; font-weight:600;">长期记忆能力</span>，后台自动整理记忆<br>我能够记住我们过往的所有交互历史哦
                    </div>
                    <div class="tooltip-tag">知识库管理</div>
                    <div class="tooltip-tag">长期记忆</div>
                    <div class="tooltip-tag">个性化体验</div>
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # 统计信息（显示当前状态）
    total_kbs = len(st.session_state.knowledge_bases) if st.session_state.knowledge_bases else 0
    total_files = sum(
        len(files) for files in (st.session_state.knowledge_bases or {}).values()
    )

    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"<div class='info-card'>📚 知识库: {total_kbs}</div>", unsafe_allow_html=True)
    with c2: 
        st.markdown(f"<div class='info-card'>📄 总文件: {total_files}</div>", unsafe_allow_html=True)

    # 创建新知识库（始终可用）- 修复空 label 警告
    with st.expander("➕ 创建新知识库", expanded=False):
        new_kb_name = st.text_input("知识库名称", placeholder="输入知识库名称", key="sidebar_new_kb_name", label_visibility="collapsed")
        if st.button("创建", use_container_width=True):
            if new_kb_name:
                create_knowledge_base_api_optimized(new_kb_name)
            else:
                st.warning("⚠️ 请输入知识库名称")

    # 删除确认区域
    if st.session_state.pending_delete:
        info = st.session_state.pending_delete
        msg = f"确认删除 {info['kb_name']}" + (f" 中的 {info['file_name']}？" if info['file_name'] else "？")
        st.markdown(f"<div class='confirm-delete-area'>⚠️ {msg}</div>", unsafe_allow_html=True)
        co1, co2 = st.columns(2)
        with co1:
            if st.button("✅ 确认", key="confirm_del_btn", use_container_width=True):
                confirm_delete()
        with co2: 
            if st.button("❌ 取消", key="cancel_del_btn", use_container_width=True):
                cancel_delete()

    st.divider()
    st.markdown("### 知识库管理")

    # 知识库列表渲染（支持加载中和加载完成两种状态）
    if len(st.session_state.knowledge_bases) == 0:
        # 显示骨架屏
        if st.session_state.data_loading:
            render_sidebar_skeleton()
        else:
            st.info("暂无知识库，请创建新知识库")
    else:
        # 渲染知识库列表
        kb_items = list((st.session_state.knowledge_bases or {}).items())

        for kb_name, files in kb_items:
            is_default = (kb_name == "默认知识库")
            is_loading = st.session_state.kb_loading_status.get(kb_name, False)

            is_expanded = (kb_name == st.session_state.selected_kb)

            with st.expander(f"📁 {kb_name} ({len(files)}{' ⏳' if is_loading else ''})", expanded=is_expanded):
                if is_loading and len(files) == 0:
                    st.markdown(render_loading_indicator(), unsafe_allow_html=True)
                    st.caption("正在加载文件列表...")

                # 在文件顶部添加导入
                from urllib.parse import unquote

                # 找到显示文件列表的代码
                if files:
                    for i, file_name in enumerate(files):
                        # 解码文件名用于显示
                        display_name = unquote(file_name)
                        
                        is_optimistic = any(
                            f["name"] == file_name and f["status"] == "uploading"
                            for f in st.session_state.optimistic_files.get(kb_name, [])
                        )

                        file_hash = st.session_state.knowledge_base_files_info.get(kb_name, {}).get(file_name)

                        f1, f2 = st.columns([6, 1])
                        with f1:
                            status_icon = "⏳ " if is_optimistic else "📄 "
                            st.markdown(
                                f"<div class='file-item-optimized'>{status_icon}{display_name}</div>", 
                                unsafe_allow_html=True
                            )
                        with f2:
                            if not is_optimistic:
                                st.button(
                                    "🗑️", 
                                    key=f"del_f_{kb_name}_{i}", 
                                    help="删除文件",
                                    on_click=request_delete, 
                                    args=('file', kb_name, file_name, file_hash)  # 注意：这里传原始文件名（编码后的）
                                )
                elif not is_loading:
                    st.caption("📭 暂无文件")

                # 文件上传
                upload_key = f"uploader_{kb_name}"
                uploaded_file = st.file_uploader(
                    f"上传文件到 {kb_name}", 
                    type=['pdf', 'docx', 'doc'], 
                    key=upload_key,
                    label_visibility="collapsed"
                )

                if uploaded_file is not None:
                    col_upload_btn, _ = st.columns([1, 3])
                    with col_upload_btn:
                        if st.button("📤 上传", key=f"upload_btn_{kb_name}"):
                            if uploaded_file.name in st.session_state.knowledge_bases.get(kb_name, []):
                                st.warning("⚠️ 文件已存在")
                            else:
                                success = upload_document_api_optimized(kb_name, uploaded_file)
                                if success:
                                    st.success(f"文件 {uploaded_file.name} 上传中...")
                                    # 刷新文件列表
                                    st.rerun()

                if not is_default:
                    st.button(
                        f"🗑️ 删除知识库", 
                        key=f"del_kb_{kb_name}", 
                        use_container_width=True,
                        on_click=request_delete, 
                        args=('kb', kb_name)
                    )

    st.divider()
    foot_c1, foot_c2 = st.columns([1, 1])
    with foot_c1:
        st.markdown("<div style='text-align: left; color: #7F8C8D; font-size:0.8rem'>© Dreamt · EchoMind</div>", unsafe_allow_html=True)
    with foot_c2:
        st.markdown("""
        <div style='text-align: right; font-size:0.8rem'>
            <a href='https://github.com/Dreamt0511/EchoMind' target='_blank'>
                <img src='https://img.shields.io/badge/GitHub-View_on_GitHub-blue?logo=GitHub'>
            </a>
        </div>
        """, unsafe_allow_html=True)
    st.divider()
    st.markdown("<div style='text-align: center; color: #BDC3C7; font-size:0.7rem'>念念不忘，必有回响</div>", unsafe_allow_html=True)

# ==========================================
# 主界面（支持骨架屏）
# ==========================================
st.divider()

# 正常渲染主界面
if st.session_state.knowledge_bases and len(st.session_state.knowledge_bases) > 0:
    kb_list = list(st.session_state.knowledge_bases.keys())
    col_sel1, col_sel2 = st.columns([1, 1])

    with col_sel1:
        current_options = [kb for kb in kb_list if kb in st.session_state.knowledge_bases]
        current_index = 0
        if st.session_state.selected_kb in current_options:
            current_index = current_options.index(st.session_state.selected_kb)
        elif current_options:
            st.session_state.selected_kb = current_options[0]

        selected_kb = st.selectbox(
            "当前对话知识库：",
            options=current_options,
            index=current_index,
            key="kb_selector"
        )
        st.session_state.selected_kb = selected_kb

    with col_sel2:
        file_count = len(st.session_state.knowledge_bases.get(selected_kb, []))
        st.markdown(f"""
            <div style='line-height:40px; text-align:center; background:rgba(255,255,255,0.5); border-radius:10px; font-size:0.9rem; margin-top:28px;'>
            📁 {file_count} 文件
            </div>
        """, unsafe_allow_html=True)
else:
    if st.session_state.data_loading:
        render_chat_skeleton()
    else:
        if st.session_state.data_loading:
            render_chat_skeleton()
        else:
            st.info("👋 欢迎使用 EchoMind！请在左侧创建知识库开始使用。")
# 聊天容器
chat_container = st.container()

with chat_container:
    if not st.session_state.chat_history:
        if st.session_state.knowledge_bases:
            st.info("选择默认知识库时：AI将搜索所有知识库内容进行发散回答。选择指定知识库时：仅按库内内容作答。")
    else:
        for chat in st.session_state.chat_history:
            st.markdown(render_message(chat["role"], chat["content"], chat["timestamp"]), unsafe_allow_html=True)

# ==========================================
# 流式处理（保持原有逻辑）
# ==========================================
if st.session_state.is_streaming and st.session_state.knowledge_bases:
    last_user_msg = st.session_state.chat_history[-1]["content"]
    selected_kb = st.session_state.selected_kb
    accumulated_answer = ""
    has_received_response = False

    with chat_container:
        thinking_placeholder = st.empty()
        status_placeholder = st.empty()
        answer_placeholder = st.empty()

        thinking_placeholder.markdown(
            """
            <div style='display:flex; align-items:center; gap:12px; margin:4px 0 8px; padding:8px 16px; background: linear-gradient(135deg, rgba(102,126,234,0.1) 0%, rgba(118,75,162,0.1) 100%); border-radius:20px; width:fit-content; backdrop-filter: blur(5px);'>
                <div class="spinner">
                    <div class="spinner-ring"></div>
                    <div class="spinner-ring"></div>
                    <div class="spinner-ring"></div>
                </div>
                <div>
                    <div style='color:#667eea; font-size:14px; font-weight:500;'>EchoMind</div>
                    <div style='color:#9ca3af; font-size:12px;' class="thinking-phrase">正在思考你的问题</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    try:
        import requests
        url = f"{API_BASE_URL}/chat_with_agent/stream"
        params = {
            "query": last_user_msg, 
            "knowledge_base_id": selected_kb,
            "user_id": USER_ID,
            "top_k": 5
        }

        with requests.get(url, params=params, stream=True, timeout=120) as response:
            if response.status_code != 200:
                accumulated_answer = f"❌ 后端错误：{response.status_code}"
                has_received_response = True
                thinking_placeholder.empty()
            else:
                for line in response.iter_lines(decode_unicode=True):
                    if not line:
                        continue

                    if not has_received_response:
                        has_received_response = True
                        thinking_placeholder.empty()

                    try:
                        data = json.loads(line.strip())
                        msg_type = data.get("type")
                        content = data.get("content", "")

                        if msg_type == "status":
                            status_placeholder.markdown(
                                f"<div style='color:#9ca3af; font-size:13px; margin:4px 0 8px; padding-left:4px;'>{content}</div>",
                                unsafe_allow_html=True
                            )
                        elif msg_type == "answer":
                            accumulated_answer += content
                            answer_placeholder.markdown(
                                render_message("assistant", accumulated_answer, datetime.now().strftime("%H:%M")),
                                unsafe_allow_html=True
                            )
                        elif msg_type == "error":
                            accumulated_answer = f"❌ {content}"
                            answer_placeholder.markdown(
                                render_message("assistant", accumulated_answer, datetime.now().strftime("%H:%M")),
                                unsafe_allow_html=True
                            )
                    except:
                        accumulated_answer += line
                        answer_placeholder.markdown(
                            render_message("assistant", accumulated_answer, datetime.now().strftime("%H:%M")),
                            unsafe_allow_html=True
                        )

    except Exception as e:
        accumulated_answer = f"❌ 错误：{str(e)}"
        answer_placeholder.markdown(
            render_message("assistant", accumulated_answer, datetime.now().strftime("%H:%M")),
            unsafe_allow_html=True
        )
        if not has_received_response:
            thinking_placeholder.empty()

    thinking_placeholder.empty()
    status_placeholder.empty()

    st.session_state.chat_history.append({
        "role": "assistant",
        "content": accumulated_answer,
        "timestamp": datetime.now().strftime("%H:%M")
    })

    st.session_state.is_streaming = False
    st.rerun()

# ==========================================
# 输入框区域
# ==========================================
if st.session_state.knowledge_bases:
    st.markdown('<div class="input-container">', unsafe_allow_html=True)
    st.chat_input("给EchoMind发送消息...", key="user_query_input", on_submit=handle_send)
    if st.session_state.chat_history:
        st.button(
            "🗑️ 清除历史",
            on_click=clear_chat_history,
            key="clear_history_btn",
            use_container_width=False,
            help="清除所有聊天历史记录",
            type="primary"
        )
    st.markdown('</div>', unsafe_allow_html=True)

    