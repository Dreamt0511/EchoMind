import streamlit as st
import pandas as pd
from datetime import datetime
import os
import requests
import httpx
import nest_asyncio
import asyncio
import hashlib
import time
import json

nest_asyncio.apply()

# 页面配置
st.set_page_config(
    page_title="EchoMind - 个性化AI问答助手",
    layout="wide",
    initial_sidebar_state="auto"
)

# 加载CSS
def load_css():
    css_file = "style.css"
    if os.path.exists(css_file):
        with open(css_file, 'r', encoding='utf-8') as f:
            css_content = f.read()
        st.markdown(f"<style>{css_content}</style>", unsafe_allow_html=True)
        return True
    else:
        st.warning(f"⚠️ 未找到 {css_file} 文件，使用默认样式")
        return False

CUSTOM_CSS = load_css()

# 后端API地址
API_BASE_URL = "http://localhost:8000"
USER_ID = 1  # 固定用户ID

# 初始化会话
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'knowledge_bases' not in st.session_state:
    st.session_state.knowledge_bases = {}
if 'selected_kb' not in st.session_state:
    st.session_state.selected_kb = None
if 'pending_delete' not in st.session_state:
    st.session_state.pending_delete = None
if 'streaming' not in st.session_state:
    st.session_state.streaming = False
if 'current_response' not in st.session_state:
    st.session_state.current_response = ""
if 'is_streaming' not in st.session_state:
    st.session_state.is_streaming = False
if 'initialized' not in st.session_state:
    st.session_state.initialized = False
# 在 session_state 中存储文件信息（包含哈希）
if 'knowledge_base_files_info' not in st.session_state:
    st.session_state.knowledge_base_files_info = {}


# ==========================================
# 后端API调用函数
# ==========================================

def load_user_knowledge_bases():
    """从后端加载用户的所有知识库"""
    try:
        url = f"{API_BASE_URL}/knowledge-bases"
        params = {"user_id": USER_ID}
        response = requests.get(url, params=params)
        
        if response.status_code == 200:
            data = response.json()
            
            # 打印调试信息
            print(f"加载知识库返回数据: {data}")
            
            # 兼容不同的返回格式
            knowledge_bases = []
            
            # 检查是否是 GetKnowledgeBaseResponse 格式
            if isinstance(data, dict):
                # 如果有 status 字段且为 success，或者直接有 knowledge_bases 字段
                if "knowledge_bases" in data:
                    knowledge_bases = data.get("knowledge_bases", [])
                elif data.get("status") == "success" and "knowledge_bases" in data:
                    knowledge_bases = data.get("knowledge_bases", [])
                else:
                    # 可能是直接返回的列表或其他格式
                    knowledge_bases = []
            elif isinstance(data, list):
                knowledge_bases = data
            
            print(f"解析后的知识库列表: {knowledge_bases}")
            
            # 转换为前端格式
            kb_dict = {}
            for kb in knowledge_bases:
                if isinstance(kb, dict):
                    kb_id = kb.get("knowledge_base_id")
                    if kb_id:
                        kb_dict[kb_id] = []
                elif isinstance(kb, str):
                    kb_dict[kb] = []
            
            st.session_state.knowledge_bases = kb_dict
            
            # 初始化文件信息存储
            if 'knowledge_base_files_info' not in st.session_state:
                st.session_state.knowledge_base_files_info = {}
            
            # 为每个知识库加载文件列表（会自动填充 files_info）
            for kb_id in kb_dict:
                load_knowledge_base_files(kb_id)
            
            # 确保默认知识库存在（仅前端显示）
            if "默认知识库" not in st.session_state.knowledge_bases:
                st.session_state.knowledge_bases["默认知识库"] = []
            
            # 设置默认选中的知识库
            if st.session_state.knowledge_bases and st.session_state.selected_kb is None:
                st.session_state.selected_kb = "默认知识库"
            
            return True
        else:
            st.error(f"加载知识库失败: {response.status_code}")
            return False
    except Exception as e:
        st.error(f"加载知识库异常: {e}")
        import traceback
        st.error(traceback.format_exc())
        return False

def load_knowledge_base_files(knowledge_base_id):
    """加载指定知识库的文件列表"""
    try:
        url = f"{API_BASE_URL}/knowledge-bases/{knowledge_base_id}/files"
        params = {"user_id": USER_ID}
        response = requests.get(url, params=params)
        
        if response.status_code == 200:
            data = response.json()
            
            if data.get("status") == "success":
                files = data.get("files", [])
                
                # 保存文件名列表和文件哈希信息
                file_names = []
                if knowledge_base_id not in st.session_state.knowledge_base_files_info:
                    st.session_state.knowledge_base_files_info[knowledge_base_id] = {}
                
                for file in files:
                    file_name = file.get("file_name")
                    file_hash = file.get("file_hash")
                    if file_name:
                        file_names.append(file_name)
                        if file_hash:
                            st.session_state.knowledge_base_files_info[knowledge_base_id][file_name] = file_hash
                
                st.session_state.knowledge_bases[knowledge_base_id] = file_names
                return True
        return False
    except Exception as e:
        st.error(f"加载文件列表异常: {e}")
        return False
    
def create_knowledge_base_api(knowledge_base_id, auto_rerun=True):
    """调用后端创建知识库"""
    # 前端阻止创建名为"默认知识库"的知识库
    if knowledge_base_id == "默认知识库":
        st.warning("⚠️ '默认知识库' 是系统保留名称，请使用其他名称")
        return False
    
    try:
        url = f"{API_BASE_URL}/knowledge-bases"
        params = {
            "knowledge_base_id": knowledge_base_id,
            "user_id": USER_ID
        }
        response = requests.post(url, params=params)
        
        if response.status_code == 200:
            data = response.json()
            if data.get("status") == "success":
                st.toast(f"✅ 知识库 '{knowledge_base_id}' 创建成功")
                # 创建成功后重新加载知识库列表
                load_user_knowledge_bases()
                # 确保默认知识库仍然存在
                if "默认知识库" not in st.session_state.knowledge_bases:
                    st.session_state.knowledge_bases["默认知识库"] = []
                if auto_rerun:
                    st.rerun()
                return True
            else:
                st.error(f"❌ 创建失败: {data.get('message', '未知错误')}")
                return False
        elif response.status_code == 409:
            st.warning(f"⚠️ 知识库 '{knowledge_base_id}' 已存在")
            return False
        else:
            st.error(f"❌ 创建失败: HTTP {response.status_code}")
            return False
    except Exception as e:
        st.error(f"❌ 创建知识库异常: {e}")
        return False

def delete_knowledge_base_api(knowledge_base_id):
    """调用后端删除知识库"""
    # 前端阻止删除默认知识库
    if knowledge_base_id == "默认知识库":
        st.warning("⚠️ 默认知识库不可删除")
        return False
    
    try:
        url = f"{API_BASE_URL}/knowledge-bases/{knowledge_base_id}"
        params = {"user_id": USER_ID}
        response = requests.delete(url, params=params)
        
        if response.status_code == 200:
            data = response.json()
            if data.get("status") == "success":
                st.toast(f"✅ {data.get('message', f'知识库 {knowledge_base_id} 删除成功')}")
                # 删除成功后重新加载知识库列表
                load_user_knowledge_bases()
                # 确保默认知识库仍然存在
                if "默认知识库" not in st.session_state.knowledge_bases:
                    st.session_state.knowledge_bases["默认知识库"] = []
                st.rerun()
                return True
            else:
                st.error(f"❌ 删除失败: {data.get('message', '未知错误')}")
                return False
        elif response.status_code == 404:
            st.warning(f"⚠️ 知识库 '{knowledge_base_id}' 不存在")
            # 重新加载知识库列表
            load_user_knowledge_bases()
            return False
        else:
            st.error(f"❌ 删除失败: HTTP {response.status_code}")
            return False
    except Exception as e:
        st.error(f"❌ 删除知识库异常: {e}")
        return False

def upload_document_api(knowledge_base_id, file):
    """调用后端上传文档"""
    try:
        url = f"{API_BASE_URL}/document_upload"
        files = {"file": (file.name, file, file.type)}
        data = {
            "knowledge_base_id": knowledge_base_id,
            "user_id": USER_ID
        }
        response = requests.post(url, files=files, data=data)
        
        if response.status_code == 200:
            res = response.json()
            if res.get("is_duplicate"):
                st.toast("✅ 文件已存在，无需重复上传")
            else:
                st.toast(f"✅ 文件 '{file.name}' 上传成功，正在后台处理中")
            # 上传成功后重新加载文件列表
            load_knowledge_base_files(knowledge_base_id)
            st.rerun()
            return True
        else:
            st.error(f"❌ 上传失败: {response.status_code}")
            return False
    except Exception as e:
        st.error(f"❌ 上传异常: {e}")
        return False

def delete_file_api(knowledge_base_id, file_name, file_hash):
    """调用后端删除文件"""
    if not file_hash:
        st.error(f"❌ 无法获取文件 {file_name} 的哈希值")
        return False
    
    try:
        url = f"{API_BASE_URL}/knowledge-bases/{knowledge_base_id}/documents/{file_hash}"
        params = {"user_id": USER_ID}
        response = requests.delete(url, params=params)
        
        if response.status_code == 200:
            data = response.json()
            if data.get("status") == "success":
                st.toast(f"✅ {data.get('message', f'文件 {file_name} 删除成功')}")
                # 删除成功后重新加载文件列表
                load_knowledge_base_files(knowledge_base_id)
                st.rerun()
                return True
            else:
                st.error(f"❌ 删除失败: {data.get('message', '未知错误')}")
                return False
        elif response.status_code == 400:
            st.error(f"❌ 删除失败: 文件不存在或无权删除")
            return False
        else:
            st.error(f"❌ 删除失败: HTTP {response.status_code}")
            return False
    except Exception as e:
        st.error(f"❌ 删除文件异常: {e}")
        return False

def compute_file_hash(file_name):
    """计算文件名的哈希值（与后端保持一致）"""
    hash_obj = hashlib.sha256()
    hash_obj.update(file_name.encode('utf-8'))
    return hash_obj.hexdigest()

# ==========================================
# 核心流式输出函数
# ==========================================

def handle_send(input_text=None):
    query = input_text or st.session_state.get("user_query_input")
    if not query or not query.strip():
        return

    # 加入用户消息
    st.session_state.chat_history.append({
        "role": "user",
        "content": query.strip(),
        "timestamp": datetime.now().strftime("%H:%M")
    })

    # 标记开始流式输出
    st.session_state.is_streaming = True
    st.session_state.current_response = ""

    # 立即刷新界面显示用户消息
    st.rerun()

def clear_chat_history():
    st.session_state.chat_history = []
    st.toast("✅ 聊天历史已清空")
    st.rerun()

# 删除确认
def request_delete(delete_type, kb_name, file_name=None, file_hash=None):
    # 阻止删除默认知识库
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
            if info['type'] == 'file':
                # 调用删除文件接口，传入真实的哈希值
                delete_file_api(
                    knowledge_base_id=info['kb_name'],
                    file_name=info['file_name'],
                    file_hash=info['file_hash']
                )
            elif info['type'] == 'kb':
                # 调用删除知识库接口
                delete_knowledge_base_api(info['kb_name'])
        except Exception as e:
            st.error(f"❌ 异常：{e}")
        finally:
            st.session_state.pending_delete = None

def cancel_delete():
    st.session_state.pending_delete = None

# ==========================================
# 消息渲染函数
# ==========================================

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

    # 保留原始 Markdown，不转义、不破坏格式
    from markdown import markdown
    try:
        rendered = markdown(content, extensions=['fenced_code', 'tables'])
    except:
        rendered = content

    return f"""
    <div class="message-container {container_class}">
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

# ==========================================
# 初始化：从后端加载数据
# ==========================================
# 在初始化部分，加载后端数据后，确保默认知识库存在
if not st.session_state.initialized:
    with st.spinner("正在加载知识库..."):
        # 先尝试从后端加载
        load_success = load_user_knowledge_bases()
        
        # 如果后端加载成功，但知识库为空（可能是后端没有数据），添加默认知识库
        if load_success:
            if not st.session_state.knowledge_bases or "默认知识库" not in st.session_state.knowledge_bases:
                st.session_state.knowledge_bases["默认知识库"] = []
        else:
            # 如果加载失败，创建默认知识库
            st.session_state.knowledge_bases = {"默认知识库": []}
        
        st.session_state.initialized = True

# ==========================================
# 侧边栏
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
            </div>
        </div>
    </div>
""", unsafe_allow_html=True)

    total_kbs = len(st.session_state.knowledge_bases)
    total_files = sum(len(files) for files in st.session_state.knowledge_bases.values())
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(
            f"<div class='info-card'>📚 知识库: {total_kbs}</div>", unsafe_allow_html=True)
    with c2: 
        st.markdown(
            f"<div class='info-card'>📄 总文件: {total_files}</div>", unsafe_allow_html=True)

    with st.expander("➕ 创建新知识库", expanded=False):
        new_kb_name = st.text_input(
            "", placeholder="输入知识库名称", key="sidebar_new_kb_name", label_visibility="collapsed")
        if st.button("创建", use_container_width=True):
            if new_kb_name:
                create_knowledge_base_api(new_kb_name)
            else:
                st.warning("⚠️ 请输入知识库名称")

    if st.session_state.pending_delete:
        info = st.session_state.pending_delete
        msg = f"确认删除 {info['kb_name']}" + \
            (f" 中的 {info['file_name']}？" if info['file_name'] else "？")
        st.markdown(
            f"<div class='confirm-delete-area'>⚠️ {msg}</div>", unsafe_allow_html=True)
        co1, co2 = st.columns(2)
        with co1:
            st.button("✅ 确认", on_click=confirm_delete, use_container_width=True)
        with co2: 
            st.button("❌ 取消", on_click=cancel_delete, use_container_width=True)

    st.divider()
    st.markdown("### 知识库管理")

    if not st.session_state.knowledge_bases:
        st.info("📭 暂无知识库，请创建")
    else:
       for kb_name, files in st.session_state.knowledge_bases.items():
            is_default = (kb_name == "默认知识库")
            with st.expander(f"📁 {kb_name} ({len(files)})", expanded=(kb_name == st.session_state.selected_kb)):

                if files:
                    for i, file_name in enumerate(files):
                        # 从保存的信息中获取真实的文件哈希
                        file_hash = st.session_state.knowledge_base_files_info.get(kb_name, {}).get(file_name)
                        
                        f1, f2 = st.columns([6, 1])
                        with f1:
                            st.markdown(
                                f"<div class='file-item'>📄 {file_name}</div>", unsafe_allow_html=True)
                        with f2: 
                            st.button("🗑️", key=f"del_f_{kb_name}_{i}", help="删除文件", 
                                    on_click=request_delete, args=('file', kb_name, file_name, file_hash))
                else:
                    st.caption("📭 暂无文件")

                uploaded_file = st.file_uploader("", type=['pdf', 'docx', 'doc'], 
                                                key=f"uploader_{kb_name}", label_visibility="collapsed")
                if uploaded_file:
                    if uploaded_file.name in st.session_state.knowledge_bases.get(kb_name, []):
                        st.warning("⚠️ 文件已存在")
                    else:
                        upload_document_api(kb_name, uploaded_file)

                # 只有非默认知识库才显示删除按钮
                if not is_default:
                    st.button(f"🗑️ 删除知识库", key=f"del_kb_{kb_name}", use_container_width=True, 
                            on_click=request_delete, args=('kb', kb_name))

    st.divider()
    foot_c1, foot_c2 = st.columns([1, 1])
    with foot_c1:
        st.markdown(
            "<div style='text-align: left; color: #7F8C8D; font-size:0.8rem'>© Dreamt · EchoMind</div>", unsafe_allow_html=True)
    with foot_c2:
        st.markdown("""
    <div style='text-align: right; font-size:0.8rem'>
        <a href='https://github.com/Dreamt0511/EchoMind' target='_blank'>
            <img src='https://img.shields.io/badge/GitHub-View_on_GitHub-blue?logo=GitHub'>
        </a>
    </div>
    """, unsafe_allow_html=True)
    st.divider()
    st.markdown("<div style='text-align: center; color: #BDC3C7; font-size:0.7rem'>念念不忘，必有回响</div>",
                unsafe_allow_html=True)

# ==========================================
# 主界面
# ==========================================
st.divider()

if st.session_state.knowledge_bases:
    kb_list = list(st.session_state.knowledge_bases.keys())
    col_sel1, col_sel2 = st.columns([1, 1])

    with col_sel1:
        selected_kb = st.selectbox(
            "当前对话知识库：",
            options=kb_list,
            index=kb_list.index(
                st.session_state.selected_kb) if st.session_state.selected_kb in kb_list else 0
        )
        st.session_state.selected_kb = selected_kb

    with col_sel2:
        st.markdown(f"""
            <div style='line-height:40px; text-align:center; background:rgba(255,255,255,0.5); border-radius:10px; font-size:0.9rem; margin-top:28px;'>
            📁 {len(st.session_state.knowledge_bases.get(selected_kb, []))} 文件
            </div>
        """, unsafe_allow_html=True)
else:
    st.info("👋 欢迎使用 EchoMind！请在左侧创建知识库开始使用。")

# 聊天容器
chat_container = st.container()

# 渲染现有聊天历史
with chat_container:
    if not st.session_state.chat_history:
        if st.session_state.knowledge_bases:
            st.info("你好！我是 EchoMind AI 助手。选择默认知识库时，我将融合知识库检索与自身模型能力，进行综合发散回答；选择指定知识库时，则严格限定于该知识库内容作答，不引入外部或模型自身知识")
    else:
        for chat in st.session_state.chat_history:
            st.markdown(
                render_message(
                    chat["role"], chat["content"], chat["timestamp"]),
                unsafe_allow_html=True
            )

# ==========================================
# 流式处理
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
                                render_message(
                                    "assistant", accumulated_answer, datetime.now().strftime("%H:%M")),
                                unsafe_allow_html=True
                            )
                        elif msg_type == "error":
                            accumulated_answer = f"❌ {content}"
                            answer_placeholder.markdown(
                                render_message(
                                    "assistant", accumulated_answer, datetime.now().strftime("%H:%M")),
                                unsafe_allow_html=True
                            )
                    except:
                        accumulated_answer += line
                        answer_placeholder.markdown(
                            render_message(
                                "assistant", accumulated_answer, datetime.now().strftime("%H:%M")),
                            unsafe_allow_html=True
                        )

    except Exception as e:
        accumulated_answer = f"❌ 错误：{str(e)}"
        answer_placeholder.markdown(
            render_message("assistant", accumulated_answer,
                           datetime.now().strftime("%H:%M")),
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