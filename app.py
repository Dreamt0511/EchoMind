import streamlit as st
import pandas as pd
from datetime import datetime
import os

# 页面配置
st.set_page_config(
    page_title="个性化问答助手",
    layout="wide"
)

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
    st.session_state.pending_delete = None  # 存储待删除的项目信息

st.markdown("<h1 style='text-align: center;'>个性化问答助手-EchoMind</h1>", unsafe_allow_html=True)


# 定义发送消息的函数
def send_message():
    """处理发送消息的逻辑"""
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
    """请求删除，显示确认界面"""
    st.session_state.pending_delete = {
        'type': delete_type,
        'kb_name': kb_name,
        'file_name': file_name
    }


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
                    st.success(f"文件 '{file_name}' 已删除！")

        elif delete_info['type'] == 'kb':
            kb_name = delete_info['kb_name']
            if kb_name in st.session_state.knowledge_bases and kb_name != "默认知识库":
                del st.session_state.knowledge_bases[kb_name]
                if st.session_state.selected_kb == kb_name:
                    st.session_state.selected_kb = "默认知识库"
                st.success(f"知识库 '{kb_name}' 已删除！")

        # 清除待删除状态
        st.session_state.pending_delete = None
        st.rerun()


def cancel_delete():
    """取消删除"""
    st.session_state.pending_delete = None


# 右侧面板 - 对话区域
with st.container():
    chat_container = st.container()
    with chat_container:
        if st.session_state.chat_history:
            for chat in st.session_state.chat_history:
                with st.chat_message(chat["role"]):
                    st.write(chat["content"])
                    if "timestamp" in chat:
                        st.caption(f"发送时间: {chat['timestamp']}")

    # 对话输入区域
    st.markdown("---")
    st.markdown('<div style="margin-top: 25rem;"></div>', unsafe_allow_html=True)

    input_col, button_col = st.columns([5, 1])

    with input_col:
        user_input = st.text_input(
            "输入您的问题...",
            key="user_question",
            placeholder="请输入您的问题... (按Enter键发送)",
            label_visibility="collapsed",
            on_change=send_message
        )

    with button_col:
        send_button = st.button("📤 发送", use_container_width=True, on_click=send_message)

    st.markdown('</div>', unsafe_allow_html=True)

    # 知识库选择下拉框
    kb_list = list(st.session_state.knowledge_bases.keys())
    if kb_list:
        selected_kb = st.selectbox(
            "💬选择知识库进行提问",
            options=kb_list,
            index=kb_list.index(st.session_state.selected_kb) if st.session_state.selected_kb in kb_list else 0,
            key="kb_selector"
        )
        if selected_kb != st.session_state.selected_kb:
            st.session_state.selected_kb = selected_kb

    # 创建两列，比例相等
    col1, col2 = st.columns(2)

    # 清空对话历史按钮
    with col1:
        if st.button("🗑️ 清空当前对话历史", use_container_width=True):
            st.session_state.chat_history = []
            st.success("当前对话历史已清空！")
            st.rerun()

    # 对话历史导出按钮
    with col2:
        if st.button("📥 导出对话历史", use_container_width=True):
            if st.session_state.chat_history:
                df = pd.DataFrame(st.session_state.chat_history)
                csv = df.to_csv(index=False)
                st.download_button(
                    label="点击下载CSV文件",
                    data=csv,
                    file_name=f"chat_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
            else:
                st.warning("没有对话历史可以导出")

# 底部信息
st.markdown("---")
# GitHub仓库链接
col1, col2, col3,col4 = st.columns([3, 1, 3,3])
with col2:
    st.markdown(
        "[![GitHub](https://img.shields.io/badge/GitHub-View_on_GitHub-blue?logo=GitHub)](https://github.com/Dreamt0511/EchoMind)")
    #st.markdown("跳转到本项目GitHub仓库")
with col3:
    st.markdown(
        """
        <div style='text-align: center; color: #666; padding: 1rem;'>
            <p>个性化问答助手-EchoMind | 作者 Dreamt</p>
        </div>
        """,
        unsafe_allow_html=True
    )

# 在侧边栏添加知识库管理功能
with st.sidebar:
    st.markdown("### 📚 知识库管理")
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
                st.rerun()
            elif new_kb_name in st.session_state.knowledge_bases:
                st.error("知识库名称已存在！")
            else:
                st.error("请输入知识库名称！")

    st.markdown("---")

    # 如果有待删除的项目，显示确认区域
    if st.session_state.pending_delete:
        with st.container():
            delete_info = st.session_state.pending_delete

            if delete_info['type'] == 'file':
                st.warning(f"确定要删除文件 '{delete_info['file_name']}' 吗？")
                st.caption("此操作不可恢复！")
            else:
                kb_name = delete_info['kb_name']
                file_count = len(st.session_state.knowledge_bases.get(kb_name, []))
                st.error(f"确定要删除知识库 '{kb_name}' 吗？")
                st.warning(f"该知识库包含 {file_count} 个文件，删除后将全部丢失！")
                st.caption("此操作不可恢复！")

            col1, col2 = st.columns(2)
            with col1:
                if st.button("✅ 确认删除", use_container_width=True, key="confirm_btn"):
                    confirm_delete()
            with col2:
                if st.button("❌ 取消", use_container_width=True, key="cancel_btn"):
                    cancel_delete()
            st.markdown('</div>', unsafe_allow_html=True)
        st.markdown("---")

    # 显示所有知识库及其文件
    st.markdown("### 📁 知识库列表")

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
                    st.rerun()
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

    #对话历史
    st.markdown("### 🕒 对话历史")

    # 初始化
    if 'conversation_summaries' not in st.session_state:
        st.session_state.conversation_summaries = []


    # 生成当前对话摘要
    def get_current_summary():
        if st.session_state.chat_history:
            first_msg = st.session_state.chat_history[0]["content"][:20] + "..." if len(
                st.session_state.chat_history[0]["content"]) > 20 else st.session_state.chat_history[0]["content"]
            return f"{datetime.now().strftime('%H:%M')} | {first_msg}"
        return "空对话"


    # 创建下拉列表
    if st.session_state.conversation_summaries:
        # 合并当前和历史记录
        all_conversations = ["📝 当前: " + get_current_summary()] + st.session_state.conversation_summaries

        selected = st.selectbox(
            "历史对话记录",
            options=all_conversations,
            key="history_selector"
        )

        # 当选择历史记录时
        if selected and not selected.startswith("📝 当前"):
            st.info(f"查看: {selected}")
    else:
        st.selectbox(
            "历史对话记录",
            options=["暂无历史记录"],
            disabled=True
        )

    # 保存按钮
    col1, col2 = st.columns(2)
    with col1:
        if st.button("💾 保存当前", use_container_width=True):
            if st.session_state.chat_history:
                summary = get_current_summary()
                st.session_state.conversation_summaries.append(summary)
                st.success("已保存!")
                st.rerun()

    with col2:
        if st.button("🗑️ 清空历史", use_container_width=True):
            st.session_state.conversation_summaries = []
            st.rerun()

# 运行说明
if __name__ == "__main__":
    st.markdown("""
    <!-- 
    运行方法：
    1. 安装依赖：pip install streamlit pandas
    2. 运行：streamlit run app.py
    -->
    """, unsafe_allow_html=True)