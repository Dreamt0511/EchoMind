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

# 初始化会话
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'knowledge_bases' not in st.session_state:
    st.session_state.knowledge_bases = {"默认知识库": []}
if 'selected_kb' not in st.session_state:
    st.session_state.selected_kb = '默认知识库'
if 'pending_delete' not in st.session_state:
    st.session_state.pending_delete = None
if 'streaming' not in st.session_state:
    st.session_state.streaming = False
if 'current_response' not in st.session_state:
    st.session_state.current_response = ""
# 新增：标记是否正在流式输出，避免重复渲染
if 'is_streaming' not in st.session_state:
    st.session_state.is_streaming = False

# ==========================================
# 核心优化：真正流式输出 + 实时更新（已修复重复问题）
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

# 新增：清除历史对话函数
def clear_chat_history():
    st.session_state.chat_history = []
    st.toast("✅ 聊天历史已清空")
    st.rerun()

def rerender_chat_history():
    """完全重新渲染聊天历史（用于需要完全刷新的场景）"""
    with chat_container:
        # 清空容器
        chat_container.empty()
        # 重新渲染所有历史消息
        for chat in st.session_state.chat_history:
            st.markdown(
                render_message(
                    chat["role"], chat["content"], chat["timestamp"]),
                unsafe_allow_html=True
            )

# 删除确认


def request_delete(delete_type, kb_name, file_name=None):
    st.session_state.pending_delete = {
        'type': delete_type, 'kb_name': kb_name, 'file_name': file_name}


def confirm_delete():
    if st.session_state.pending_delete:
        info = st.session_state.pending_delete
        try:
            if info['type'] == 'file':
                kb = info['kb_name']
                fname = info['file_name']

                # 真实哈希
                hash_obj = hashlib.sha256()
                hash_obj.update(fname.encode('utf-8'))
                real_hash = hash_obj.hexdigest()

                url = "http://localhost:8000/documents"
                resp = requests.delete(url, params={
                    "file_hash": real_hash,
                    "knowledge_base_id": kb,
                    "user_id": 1
                })

                if resp.status_code == 200:
                    if kb in st.session_state.knowledge_bases and fname in st.session_state.knowledge_bases[kb]:
                        st.session_state.knowledge_bases[kb].remove(fname)
                    st.toast("✅ 删除成功")
                else:
                    st.error(f"❌ 删除失败：{resp.status_code}")

            elif info['type'] == 'kb':
                kb = info['kb_name']
                if kb != "默认知识库" and kb in st.session_state.knowledge_bases:
                    del st.session_state.knowledge_bases[kb]
                    if st.session_state.selected_kb == kb:
                        st.session_state.selected_kb = "默认知识库"
                    st.toast("✅ 知识库已删除")
        except Exception as e:
            st.error(f"❌ 异常：{e}")
        finally:
            st.session_state.pending_delete = None


def cancel_delete():
    st.session_state.pending_delete = None

# ====================== ✅ 消息渲染（已修复 Markdown） ======================


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

    # 关键修复：保留原始 Markdown，不转义、不破坏格式
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
    total_files = sum(len(files)
                      for files in st.session_state.knowledge_bases.values())
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(
            f"<div class='info-card'>📚 知识库: {total_kbs}</div>", unsafe_allow_html=True)
    with c2: st.markdown(
        f"<div class='info-card'>📄 总文件: {total_files}</div>", unsafe_allow_html=True)

    with st.expander("➕ 创建新知识库", expanded=False):
        new_kb_name = st.text_input(
            "", placeholder="输入名称", key="sidebar_new_kb_name", label_visibility="collapsed")
        if st.button("创建", use_container_width=True):
            if new_kb_name and new_kb_name not in st.session_state.knowledge_bases:
                st.session_state.knowledge_bases[new_kb_name] = []
                st.success(f"✅ '{new_kb_name}' 创建成功")
                st.rerun()
            elif new_kb_name:
                st.warning("⚠️ 已存在")

    if st.session_state.pending_delete:
        info = st.session_state.pending_delete
        msg = f"确认删除 {info['kb_name']}" + \
            (f"中的 {info['file_name']}？" if info['file_name'] else "？")
        st.markdown(
            f"<div class='confirm-delete-area'>⚠️ {msg}</div>", unsafe_allow_html=True)
        co1, co2 = st.columns(2)
        with co1:
            st.button("✅ 确认", on_click=confirm_delete,
                      use_container_width=True)
        with co2: st.button("❌ 取消", on_click=cancel_delete,
                            use_container_width=True)

    st.divider()
    st.markdown("### 知识库管理")

    for kb_name, files in st.session_state.knowledge_bases.items():
        is_default = (kb_name == '默认知识库')
        with st.expander(f"📁 {kb_name} ({len(files)})", expanded=(kb_name == st.session_state.selected_kb)):

            if files:
                for i, file_name in enumerate(files):
                    f1, f2 = st.columns([6, 1])
                    with f1:
                        st.markdown(
                            f"<div class='file-item'>📄 {file_name}</div>", unsafe_allow_html=True)
                    with f2: st.button("🗑️", key=f"del_f_{kb_name}_{i}", help="删除文件", on_click=request_delete, args=(
                        'file', kb_name, file_name))
            else:
                st.caption("📭 暂无文件")

            uploaded_file = st.file_uploader("", type=[
                                             'pdf', 'docx', 'doc'], key=f"uploader_{kb_name}", label_visibility="collapsed")
            if uploaded_file:
                if uploaded_file.name in st.session_state.knowledge_bases[kb_name]:
                    st.warning("⚠️ 文件已存在")
                    continue

                try:
                    url = "http://localhost:8000/document_upload"
                    files = {"file": (uploaded_file.name,
                                      uploaded_file, uploaded_file.type)}
                    data = {"knowledge_base_id": kb_name,"user_id": 1}
                    resp = requests.post(url, files=files, data=data)

                    if resp.status_code == 200:
                        res = resp.json()
                        if res.get("is_duplicate"):
                            st.toast("✅ 文件已存在")
                        else:
                            st.session_state.knowledge_bases[kb_name].append(
                                uploaded_file.name)
                            st.toast("✅ 上传成功")
                            st.rerun()
                    else:
                        st.error(f"❌ 上传失败：{resp.status_code}")
                except Exception as e:
                    st.error(f"❌ 异常：{e}")

            if not is_default:
                st.button(f"🗑️ 删除知识库", key=f"del_kb_{kb_name}", use_container_width=True, on_click=request_delete, args=(
                    'kb', kb_name))

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

# 聊天容器
chat_container = st.container()

# 渲染现有聊天历史
with chat_container:
    if not st.session_state.chat_history:
        st.info("👋 你好！我是 EchoMind AI 助手。选择默认知识库时，我将检索全部知识库内容为你提供回复。")
    else:
        for chat in st.session_state.chat_history:
            st.markdown(
                render_message(
                    chat["role"], chat["content"], chat["timestamp"]),
                unsafe_allow_html=True
            )

# ====================== ✅ 核心：已升级流式处理（添加思考中效果） ======================
if st.session_state.is_streaming:
    last_user_msg = st.session_state.chat_history[-1]["content"]
    selected_kb = st.session_state.selected_kb
    accumulated_answer = ""

    # 标记是否已经收到任何响应（用于控制思考中状态的显示）
    has_received_response = False

    with chat_container:
        # 思考中状态占位（灰色小字，带动态效果）
        thinking_placeholder = st.empty()
        # 工具状态占位（灰字）
        status_placeholder = st.empty()
        # AI 回答占位
        answer_placeholder = st.empty()

        # 立即显示思考中效果        
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
        url = "http://localhost:8000/chat_with_agent/stream"
        params = {"query": last_user_msg, "knowledge_base_id": selected_kb,"user_id": 1}

        with requests.get(url, params=params, stream=True, timeout=120) as response:
            if response.status_code != 200:
                accumulated_answer = f"❌ 后端错误：{response.status_code}"
                has_received_response = True
                thinking_placeholder.empty()  # 立即清除思考中状态
            else:
                for line in response.iter_lines(decode_unicode=True):
                    if not line:
                        continue

                    # 收到第一条响应时，清除思考中状态
                    if not has_received_response:
                        has_received_response = True
                        thinking_placeholder.empty()

                    try:
                        # 解析 JSON 结构
                        data = json.loads(line.strip())
                        msg_type = data.get("type")
                        content = data.get("content", "")

                        # 工具状态 → 灰色小字
                        if msg_type == "status":
                            status_placeholder.markdown(
                                f"<div style='color:#9ca3af; font-size:13px; margin:4px 0 8px; padding-left:4px;'>{content}</div>",
                                unsafe_allow_html=True
                            )
                        # 正式回答 → 正常渲染
                        elif msg_type == "answer":
                            accumulated_answer += content
                            answer_placeholder.markdown(
                                render_message(
                                    "assistant", accumulated_answer, datetime.now().strftime("%H:%M")),
                                unsafe_allow_html=True
                            )
                        # 错误
                        elif msg_type == "error":
                            accumulated_answer = f"❌ {content}"
                            answer_placeholder.markdown(
                                render_message(
                                    "assistant", accumulated_answer, datetime.now().strftime("%H:%M")),
                                unsafe_allow_html=True
                            )
                    except:
                        # 兼容兜底
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
        # 出错时也要清除思考中状态
        if not has_received_response:
            thinking_placeholder.empty()

    # 确保思考中状态已被清除（双重保险）
    thinking_placeholder.empty()
    # 工具状态提示在回答完成后清除
    status_placeholder.empty()

    # 保存最终回答
    st.session_state.chat_history.append({
        "role": "assistant",
        "content": accumulated_answer,
        "timestamp": datetime.now().strftime("%H:%M")
    })

    st.session_state.is_streaming = False
    st.rerun()

# ====================== ✅ 输入框区域：添加清除历史按钮 ======================
# 创建输入框容器（固定底部）
st.markdown('<div class="input-container">', unsafe_allow_html=True)
# 输入框
st.chat_input("给EchoMind发送消息...", key="user_query_input", on_submit=handle_send)
# 清除历史按钮，有历史记录时显示，否则不显示
if st.session_state.chat_history:
    st.button(
        "🗑️ 清除历史",
        on_click=clear_chat_history,
        key="clear_history_btn",
        use_container_width=False,
        help="清除所有聊天历史记录",
        args=(),
        kwargs={},
        type="primary"
    )
st.markdown('</div>', unsafe_allow_html=True)