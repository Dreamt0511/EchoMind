import streamlit as st
import pandas as pd
from datetime import datetime
import os

# ==========================================
# 1. 页面配置 & 响应式 CSS
# ==========================================
st.set_page_config(
    page_title="EchoMind - 个性化AI问答助手",
    layout="wide",
    initial_sidebar_state="auto"
)

# 2. 加载外部 CSS 文件
# ==========================================
def load_css():
    """加载外部 CSS 文件"""
    css_file = "style.css"
    if os.path.exists(css_file):
        with open(css_file, 'r', encoding='utf-8') as f:
            css_content = f.read()
        st.markdown(f"<style>{css_content}</style>", unsafe_allow_html=True)
        return True
    else:
        st.warning(f"⚠️ 未找到 {css_file} 文件，使用默认样式")
        return False

# 加载 CSS
CUSTOM_CSS =load_css()

# ==========================================
# 2. 初始化 Session State
# ==========================================
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'knowledge_bases' not in st.session_state:
    st.session_state.knowledge_bases = {"默认知识库": ["产品手册.pdf", "FAQ.txt"]}
if 'selected_kb' not in st.session_state:
    st.session_state.selected_kb = '默认知识库'
if 'pending_delete' not in st.session_state:
    st.session_state.pending_delete = None

# ==========================================
# 3. 辅助函数 & 回调
# ==========================================
def handle_send(input_text=None):
    """处理发送消息逻辑"""
    query = input_text or st.session_state.get("user_query_input")
    
    if query and query.strip():
        # 1. 添加用户消息
        st.session_state.chat_history.append({
            "role": "user",
            "content": query.strip(),
            "timestamp": datetime.now().strftime("%H:%M")
        })
        
        # 2. 模拟 AI 回答
        selected_kb = st.session_state.selected_kb
        files_in_kb = st.session_state.knowledge_bases.get(selected_kb, [])
        file_info = f" (参考了{len(files_in_kb)}个文件)" if files_in_kb else " (当前知识库为空)"
        
        ai_response = f"这是关于 '{query.strip()}' 在知识库 '{selected_kb}'{file_info} 中的回答。"
        
        st.session_state.chat_history.append({
            "role": "assistant",
            "content": ai_response,
            "timestamp": datetime.now().strftime("%H:%M")
        })

def request_delete(delete_type, kb_name, file_name=None):
    st.session_state.pending_delete = {'type': delete_type, 'kb_name': kb_name, 'file_name': file_name}

def confirm_delete():
    if st.session_state.pending_delete:
        delete_info = st.session_state.pending_delete
        try:
            if delete_info['type'] == 'file':
                kb_name = delete_info['kb_name']
                file_name = delete_info['file_name']
                if kb_name in st.session_state.knowledge_bases and file_name in st.session_state.knowledge_bases[kb_name]:
                    st.session_state.knowledge_bases[kb_name].remove(file_name)
                    st.toast(f"✅ 文件 '{file_name}' 已删除")
            elif delete_info['type'] == 'kb':
                kb_name = delete_info['kb_name']
                if kb_name in st.session_state.knowledge_bases and kb_name != "默认知识库":
                    del st.session_state.knowledge_bases[kb_name]
                    if st.session_state.selected_kb == kb_name:
                        st.session_state.selected_kb = "默认知识库"
                    st.toast(f"✅ 知识库 '{kb_name}' 已删除")
        except:
            st.error("❌ 删除失败")
        finally:
            st.session_state.pending_delete = None

def cancel_delete():
    st.session_state.pending_delete = None

def render_message(role, content, timestamp):
    """渲染单条消息"""
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
    
    return f"""
    <div class="message-container {container_class}">
        <div class="message-avatar {role}-avatar">
            {avatar}
        </div>
        <div class="message-bubble {bubble_class}">
            <div class="message-content">{content}</div>
            <div class="message-timestamp {timestamp_class}">
                <span>{role if role == 'user' else 'AI'}</span>
                <span>•</span>
                <span>{timestamp}</span>
            </div>
        </div>
    </div>
    """

# ==========================================
# 4. 侧边栏 UI
# ==========================================
with st.sidebar:
    st.markdown("<h2 style='text-align: center; color: #2C3E50;'>EchoMind</h2>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #7F8C8D; margin-top:-10px;'>个性化智能问答助手</p>", unsafe_allow_html=True)
    st.divider()
    
    # 统计卡片
    total_kbs = len(st.session_state.knowledge_bases)
    total_files = sum(len(files) for files in st.session_state.knowledge_bases.values())
    c1, c2 = st.columns(2)
    with c1: st.markdown(f"<div class='info-card'>📚 知识库: {total_kbs}</div>", unsafe_allow_html=True)
    with c2: st.markdown(f"<div class='info-card'>📄 总文件: {total_files}</div>", unsafe_allow_html=True)

    # 创建知识库
    with st.expander("➕ 创建新知识库", expanded=False):
        new_kb_name = st.text_input("", placeholder="输入名称", key="sidebar_new_kb_name", label_visibility="collapsed")
        if st.button("创建", use_container_width=True):
            if new_kb_name and new_kb_name not in st.session_state.knowledge_bases:
                st.session_state.knowledge_bases[new_kb_name] = []
                st.success(f"✅ '{new_kb_name}' 创建成功")
                st.rerun()
            elif new_kb_name: st.warning("⚠️ 已存在")

    # 删除确认区域
    if st.session_state.pending_delete:
        info = st.session_state.pending_delete
        msg = f"确认删除{info['kb_name']}" + (f"中的文件 {info['file_name']}？" if info['file_name'] else "？此操作不可撤销！")
        st.markdown(f"<div class='confirm-delete-area'>⚠️ {msg}</div>", unsafe_allow_html=True)
        co1, co2 = st.columns(2)
        with co1: st.button("✅ 确认", on_click=confirm_delete, use_container_width=True, key="conf_del_btn")
        with co2: st.button("❌ 取消", on_click=cancel_delete, use_container_width=True, key="canc_del_btn")

    st.divider()

    # 知识库列表管理
    st.markdown("### 📚 知识库管理")
    for kb_name, files in st.session_state.knowledge_bases.items():
        is_default = (kb_name == '默认知识库')
        with st.expander(f"📁 {kb_name} ({len(files)})", expanded=(kb_name == st.session_state.selected_kb)):
            # 文件列表
            if files:
                for i, file_name in enumerate(files):
                    f1, f2 = st.columns([6, 1])
                    with f1: st.markdown(f"<div class='file-item'>📄 {file_name}</div>", unsafe_allow_html=True)
                    with f2: st.button("🗑️", key=f"del_f_{kb_name}_{i}", help="删除文件", on_click=request_delete, args=('file', kb_name, file_name))
            else:
                st.caption("📭 暂无文件")

            # 上传文件
            uploaded_file = st.file_uploader("", type=['txt', 'pdf', 'docx', 'md', 'csv'], key=f"uploader_{kb_name}", label_visibility="collapsed")
            if uploaded_file:
                if uploaded_file.name not in st.session_state.knowledge_bases[kb_name]:
                    st.session_state.knowledge_bases[kb_name].append(uploaded_file.name)
                    st.toast(f"✅ {uploaded_file.name} 上传成功")
                    st.rerun()
                else: st.warning("⚠️ 文件已存在")

            # 删除知识库
            if not is_default:
                st.button(f"🗑️ 删除知识库", key=f"del_kb_{kb_name}", use_container_width=True, on_click=request_delete, args=('kb', kb_name))
    
    # ----- D. 底部页脚 -----
    st.divider()
    foot_c1, foot_c2 = st.columns([1, 1])
    with foot_c1:
        st.markdown("<div style='text-align: left; color: #7F8C8D; font-size:0.8rem'>© Dreamt · EchoMind</div>", unsafe_allow_html=True)
    with foot_c2:
        st.markdown("""
    <div style='text-align: right; font-size:0.8rem'>
        <a href='https://github.com/Dreamt0511/EchoMind' target='_blank'>
            <img src='https://img.shields.io/badge/GitHub-View_on_GitHub-blue?logo=GitHub' alt='GitHub'>
        </a>
    </div>
    """, unsafe_allow_html=True)
# ==========================================
# 5. 主界面 UI
# ==========================================
# 标题区域
# 标题区域 - 带3D高级动态机器人图标
st.markdown("""
    <style>
    .main-header {
        position: relative;
        top: 35px;  /* 向下偏移30px */
    }
    .header-container {
        height: 170px;  /* 设置容器高度 */
        display: flex;
        align-items: center;  /* 垂直居中内容 */
    }
    </style>
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
                                <div class="visor left-visor">
                                    <div class="visor-glow"></div>
                                </div>
                                <div class="visor right-visor">
                                    <div class="visor-glow"></div>
                                </div>
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

st.divider()

# ----- A. 知识库选择 -----
kb_list = list(st.session_state.knowledge_bases.keys())
col_sel1, col_sel2 = st.columns([1, 1])
with col_sel1:
    selected_kb = st.selectbox(
        label= "当前对话知识库：",
        options=kb_list,
        index=kb_list.index(st.session_state.selected_kb) if st.session_state.selected_kb in kb_list else 0,
        key="kb_selector"
        )
    st.session_state.selected_kb = selected_kb
with col_sel2:
    st.markdown(f"<div style='line-height:40px; text-align:center; background:rgba(255,255,255,0.5); border-radius:10px; font-size:0.9rem;'>📁 {len(st.session_state.knowledge_bases.get(selected_kb, []))} 文件</div>", unsafe_allow_html=True)

st.divider()

# ----- B. 聊天记录区域 -----
chat_container = st.container()

with chat_container:
    if not st.session_state.chat_history:
        st.info("👋 你好！我是 EchoMind AI 助手。请在下方输入问题，或者在左侧管理知识库。")
    else:
        # 关键修复：逐条渲染，避免 HTML 累积导致转义问题
        for chat in st.session_state.chat_history:
            message_html = render_message(chat["role"], chat["content"], chat["timestamp"])
            st.markdown(message_html, unsafe_allow_html=True)

# ----- C. 底部功能区域 -----
# 功能按钮
feat_col1, feat_col2, feat_col3 = st.columns([2, 2, 3])
with feat_col1:
    st.markdown('<div class="action-btn">', unsafe_allow_html=True)
    if st.button("🗑️ 清空", use_container_width=True, key="clear_chat"):
        st.session_state.chat_history = []
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)
with feat_col2:
    st.markdown('<div class="action-btn">', unsafe_allow_html=True)
    if st.session_state.chat_history:
        df = pd.DataFrame(st.session_state.chat_history)
        st.download_button("📥 导出", df.to_csv(index=False), f"chat_{datetime.now().strftime('%Y%m%d')}.csv", mime='text/csv', use_container_width=True)
    else:
        st.button("📥 导出", disabled=True, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# 聊天输入框
st.chat_input("请输入问题...", key="user_query_input", on_submit=handle_send)
