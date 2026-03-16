import streamlit as st
import pandas as pd
from datetime import datetime
import os

# 页面配置
st.set_page_config(
    page_title="个性化问答助手",
    layout="wide"
)
# 创建三列，中间列放图片，两侧列占空
left_col, center_col, right_col = st.columns([1.5, 2, 1])
with center_col:
    st.image("images/EchoMind.png")

# 自定义CSS样式
st.markdown("""
<style>
    /* 主标题样式 */
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 2rem;
        padding: 1rem;
        background: linear-gradient(90deg, #f8f9fa 0%, #e9ecef 100%);
        border-radius: 10px;
    }

    /* 卡片样式 */
    .stCard {
        background-color: #ffffff;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
    }

    /* 上传区域样式 */
    .upload-area {
        border: 2px dashed #1E88E5;
        border-radius: 10px;
        padding: 2rem;
        text-align: center;
        background-color: #f8f9fa;
    }

    /* 按钮样式 */
    .stButton > button {
        width: 100%;
        border-radius: 5px;
        background-color: #1E88E5;
        color: white;
        font-weight: bold;
    }

    /* 侧边栏样式 */
    .css-1d391kg {
        background-color: #f8f9fa;
    }

    /* 信息提示框 */
    .info-box {
        background-color: #e3f2fd;
        padding: 0.5rem;
        border-radius: 5px;
        border-left: 3px solid #1E88E5;
        margin: 0.5rem 0;
    }

    /* 文件列表样式 */
    .file-item {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 0.5rem;
        margin: 0.2rem 0;
        background-color: #f8f9fa;
        border-radius: 5px;
    }

    /* 删除按钮样式 */
    button[key*="del_file"], button[key*="del_kb"] {
        background-color: #dc3545 !important;
    }

    button[key*="del_file"]:hover, button[key*="del_kb"]:hover {
        background-color: #c82333 !important;
    }

    /* 确认区域样式 */
    .confirm-delete-area {
        background-color: #fff3cd;
        border: 1px solid #ffeeba;
        border-radius: 5px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# 初始化session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'knowledge_bases' not in st.session_state:
    st.session_state.knowledge_bases = {"默认知识库": []}
if 'user_info' not in st.session_state:
    st.session_state.user_info = {
        'name': '访客',
        'email': '',
        'preferences': {}
    }
if 'selected_kb' not in st.session_state:
    st.session_state.selected_kb = '默认知识库'
if 'pending_delete' not in st.session_state:
    st.session_state.pending_delete = None


# 定义发送消息的函数
def send_message():
    user_input = st.session_state.user_question
    if user_input and user_input.strip():
        selected_kb = st.session_state.selected_kb
        st.session_state.chat_history.append({
            "role": "user",
            "content": user_input,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })
        with st.spinner("正在思考中..."):
            files_in_kb = st.session_state.knowledge_bases.get(selected_kb, [])
            file_info = f" (参考了{len(files_in_kb)}个文件)" if files_in_kb else " (当前知识库为空)"
            ai_response = f"这是关于 '{user_input}' 在知识库 '{selected_kb}'{file_info} 中的回答。"
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": ai_response,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            })
        st.session_state.user_question = ""


# 删除相关函数
def request_delete(delete_type, kb_name, file_name=None):
    st.session_state.pending_delete = {'type': delete_type, 'kb_name': kb_name, 'file_name': file_name}


def confirm_delete():
    """确认删除"""
    if st.session_state.pending_delete:
        delete_info = st.session_state.pending_delete

        if delete_info['type'] == 'file':
            kb_name = delete_info['kb_name']
            file_name = delete_info['file_name']
            if kb_name in st.session_state.knowledge_bases:
                if file_name in st.session_state.knowledge_bases[kb_name]:
                    st.session_state.knowledge_bases[kb_name].remove(file_name)

        elif delete_info['type'] == 'kb':
            kb_name = delete_info['kb_name']
            if kb_name in st.session_state.knowledge_bases and kb_name != "默认知识库":
                del st.session_state.knowledge_bases[kb_name]
                if st.session_state.selected_kb == kb_name:
                    st.session_state.selected_kb = "默认知识库"

        # 清除待删除状态
        st.session_state.pending_delete = None


def cancel_delete():
    st.session_state.pending_delete = None


# ===================== 完美修复版：自动滑到底部 + 可手动查看历史消息 =====================
chat_html = """
<div style="height: 580px; overflow-y: auto; border: 1px solid #e0e0e0; 
            border-radius: 10px; padding: 20px; 
            background-color: #f9f9f9;" id="chat-container">
"""
if st.session_state.chat_history:
    for chat in st.session_state.chat_history:
        if chat["role"] == "user":
            chat_html += f"""
            <div style="background-color: #DCF8C6; padding:10px; border-radius:8px; 
                        margin:8px 0; max-width:80%; margin-left:auto;">
                <strong></strong> {chat['content']}<br>
                <small style="color:#666;">{chat['timestamp']}</small>
            </div>"""
        else:
            chat_html += f"""
            <div style="background-color: white; padding:10px; border-radius:8px; 
                        margin:8px 0; max-width:80%; margin-right:auto; border:1px solid #eee;">
                <strong>助手:</strong> {chat['content']}<br>
                <small style="color:#666;">{chat['timestamp']}</small>
            </div>"""
chat_html += """
</div>
<script>
    // 核心修复：等页面完全渲染后再平滑滚动到底部
    window.addEventListener('load', function() {
        const chatContainer = document.getElementById('chat-container');
        // 强制滚动到最底部
        chatContainer.scrollTop = chatContainer.scrollHeight;
    });
</script>
"""

import streamlit.components.v1 as components

# 保持开启滚动，高度匹配内部容器
components.html(chat_html, height=500, scrolling=True)

# 知识库选择
kb_list = list(st.session_state.knowledge_bases.keys())
if kb_list:
    selected_kb = st.selectbox("💬选择知识库进行提问",
                               options=kb_list,
                               index=kb_list.index(st.session_state.selected_kb),
                               key="kb_selector",
                               width= 300
                               )
    if selected_kb != st.session_state.selected_kb:
        st.session_state.selected_kb = selected_kb

# 固定底部输入区域
st.markdown("""
<style>
.fixed-input-area {position: fixed; bottom: 0; left: 0; right: 0; background: white; padding:20px; box-shadow:0 -2px 10px rgba(0,0,0,0.1); z-index:999}
</style>
<div class="fixed-input-area">
""", unsafe_allow_html=True)

input_col, button_col = st.columns([5, 1])
with input_col:
    st.text_input("输入问题", key="user_question", placeholder="请输入问题", label_visibility="collapsed",
                  on_change=send_message)
with button_col:
    st.button("📤 发送", use_container_width=True, on_click=send_message)

col1, col2 = st.columns(2)
with col1:
    if st.button("🗑️ 清空对话", use_container_width=True):
        st.session_state.chat_history = []
with col2:
    if st.button("📥 导出历史", use_container_width=True):
        if st.session_state.chat_history:
            df = pd.DataFrame(st.session_state.chat_history)
            st.download_button("下载CSV", df.to_csv(index=False),
                               f"chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
        else:
            st.warning("无历史")
st.markdown("</div>", unsafe_allow_html=True)

# 底部信息
st.markdown("---")
c1, c2, c3, c4 = st.columns([3, 1, 3, 3])
with c2:
    st.markdown(
        "[![GitHub](https://img.shields.io/badge/GitHub-View_on_GitHub-blue?logo=GitHub)](https://github.com/Dreamt0511/EchoMind)")
with c3:
    st.markdown("<div style='text-align:center;color:#666'>个性化问答助手-EchoMind | 作者 Dreamt</div>",
                unsafe_allow_html=True)

# 侧边栏
with st.sidebar:
    st.markdown("<h1 style='text-align:center'>EchoMind</h1>", unsafe_allow_html=True)
    st.markdown("<h1 style='text-align: center;'>个性化问答助手</h1>", unsafe_allow_html=True)
    st.markdown("### 知识库管理")
    total_files = sum(len(files) for files in st.session_state.knowledge_bases.values())
    st.info(f"知识库数量: {len(st.session_state.knowledge_bases)}")
    st.info(f"总文件数: {total_files}")
    # 创建新知识库
    with st.expander("➕ 创建新知识库", expanded=False):
        new_kb_name = st.text_input("知识库名称", key="new_kb_name")
        if st.button("创建知识库", use_container_width=True):
            if new_kb_name and new_kb_name not in st.session_state.knowledge_bases:
                st.session_state.knowledge_bases[new_kb_name] = []
                st.success(f"知识库 '{new_kb_name}' 创建成功！")
            elif new_kb_name in st.session_state.knowledge_bases:
                st.error("知识库名称已存在！")
            else:
                st.error("请输入知识库名称！")

    if st.session_state.pending_delete:
        info = st.session_state.pending_delete
        if info['type'] == 'file':
            st.warning(f"删除文件 {info['file_name']}？")
        else:
            st.error(f"删除知识库 {info['kb_name']}？")
        co1, co2 = st.columns(2)
        with co1:
            st.button("✅ 确认", on_click=confirm_delete)
        with co2:
            st.button("❌ 取消", on_click=cancel_delete)

    st.markdown("---")

    # 显示所有知识库及其文件
    st.markdown("### 知识库列表")
    # 为每个知识库创建可展开的区域
    for kb_name, files in st.session_state.knowledge_bases.items():
        with st.expander(f"📚 {kb_name} ({len(files)}个文件)", expanded=False):
            # 显示该知识库下的文件列表
            if files:
                for i, file_name in enumerate(files):
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        st.text(f"📄 {file_name}")
                    with col2:
                        # 删除单个文件的按钮
                        if st.button("🗑️", key=f"del_file_{kb_name}_{i}",
                                     help=f"删除{file_name}",
                                     on_click=request_delete,
                                     args=('file', kb_name, file_name)):
                            pass
            else:
                st.info("该知识库暂无文件")

            # 向该知识库上传文件
            st.markdown("---")
            uploaded_file = st.file_uploader(
                f"上传文件到 {kb_name}",
                type=['txt', 'pdf', 'docx', 'md', 'csv'],
                key=f"uploader_{kb_name}",
                label_visibility="collapsed"
            )
            # 文件上传说明
            st.info("📌 支持的文档类型：TXT, PDF, DOCX, Markdown, CSV (单个文件不超过200MB)")
            if uploaded_file is not None:
                if uploaded_file.name not in st.session_state.knowledge_bases[kb_name]:
                    st.session_state.knowledge_bases[kb_name].append(uploaded_file.name)
                    st.success(f"文件 {uploaded_file.name} 上传到 {kb_name} 成功！")
                else:
                    st.warning("文件已存在于该知识库中！")

            # 如果不是默认知识库，显示删除知识库按钮
            if kb_name != "默认知识库":
                if st.button(f"🗑️ 删除整个知识库", key=f"del_kb_{kb_name}",
                             use_container_width=True,
                             on_click=request_delete,
                             args=('kb', kb_name)):
                    pass

    st.markdown("---")

    st.markdown("### 🕒 历史对话")
    if 'conversation_summaries' not in st.session_state:
        st.session_state.conversation_summaries = []


    def summary():
        if st.session_state.chat_history:
            s = st.session_state.chat_history[0]['content'][:20]
            return f"{datetime.now().strftime('%H:%M')} | {s}"
        return "空"


    if st.session_state.conversation_summaries:
        st.selectbox("记录", ["当前：" + summary()] + st.session_state.conversation_summaries)
    else:
        st.selectbox("记录", ["无"], disabled=True)
