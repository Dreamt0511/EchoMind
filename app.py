import streamlit as st
import pandas as pd
from datetime import datetime

# 页面配置
st.set_page_config(
    page_title="个人知识库助手",
    page_icon="📚",
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
</style>
""", unsafe_allow_html=True)

# 初始化session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'knowledge_bases' not in st.session_state:
    st.session_state.knowledge_bases = ['默认知识库']
if 'user_info' not in st.session_state:
    st.session_state.user_info = {
        'name': '访客',
        'email': '',
        'preferences': {}
    }
if 'selected_kb' not in st.session_state:
    st.session_state.selected_kb = '默认知识库'


# 定义发送消息的函数
def send_message():
    """处理发送消息的逻辑"""
    user_input = st.session_state.user_question

    if user_input and user_input.strip():
        # 获取当前选中的知识库
        selected_kb = st.session_state.selected_kb

        # 添加用户消息到历史记录
        st.session_state.chat_history.append({
            "role": "user",
            "content": user_input,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })

        # 模拟AI响应（实际应用中这里应该调用知识库查询）
        with st.spinner("正在思考中..."):
            # 这里可以添加实际的知识库查询逻辑
            ai_response = f"这是关于 '{user_input}' 在知识库 '{selected_kb}' 中的回答。"

            st.session_state.chat_history.append({
                "role": "assistant",
                "content": ai_response,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            })

        # 清空输入框
        st.session_state.user_question = ""

# 右侧面板 - 对话区域
with st.container():
    # 对话历史容器
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
    # 添加一个空的div来向下移动标题
    st.markdown('<div style="margin-top: 25rem;"></div>', unsafe_allow_html=True)
    # 使用两列布局放置输入框和发送按钮
    input_col, button_col = st.columns([5, 1])

    with input_col:
        # 添加Enter键发送功能：通过on_change回调
        user_input = st.text_input(
            "输入您的问题...",
            key="user_question",
            placeholder="请输入您的问题... (按Enter键发送)",
            label_visibility="collapsed",
            on_change=send_message  # 当按下Enter键时触发
        )

    with button_col:
        send_button = st.button("📤 发送", use_container_width=True, on_click=send_message)

    st.markdown('</div>', unsafe_allow_html=True)
    # 知识库选择下拉框
    selected_kb = st.selectbox(
        "💬选择知识库进行提问",
        options=st.session_state.knowledge_bases,
        index=st.session_state.knowledge_bases.index(
            st.session_state.selected_kb) if st.session_state.selected_kb in st.session_state.knowledge_bases else 0,
        key="kb_selector"
    )
    # 更新选中的知识库
    if selected_kb != st.session_state.selected_kb:
        st.session_state.selected_kb = selected_kb

    # 创建两列，比例相等
    col1, col2 = st.columns(2)

    # 清空对话历史按钮 - 放在第一列
    with col1:
        if st.button("🗑️ 清空对话历史", use_container_width=True):
            st.session_state.chat_history = []
            st.success("对话历史已清空！")
            st.rerun()

    # 对话历史导出按钮 - 放在第二列
    with col2:
        if st.button("📥 导出对话历史", use_container_width=True):
            if st.session_state.chat_history:
                # 创建DataFrame
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
st.markdown(
    """
    <div style='text-align: center; color: #666; padding: 1rem;'>
        <p>个性化问答助手-EchoMind | 作者 Dreamt</p>
    </div>
    """,
    unsafe_allow_html=True
)

# 在侧边栏添加一些额外的功能
with st.sidebar:
    # 主标题
    st.markdown('<h1 class="main-header">📚 个性化问答助手-EchoMind 📚</h1>', unsafe_allow_html=True)

    # 已上传的知识库列表 - 放在最前面
    st.markdown("### 📁 已上传的知识库")
    for kb in st.session_state.knowledge_bases:
        col1, col2 = st.columns([3, 1])
        with col1:
            st.write(f"• {kb}")
        with col2:
            if st.button("选择", key=f"select_{kb}"):
                st.session_state.selected_kb = kb
                st.success(f"已选择: {kb}")

    st.markdown("---")

    # 上传新知识库
    st.markdown("### 📤 上传知识库")
    uploaded_file = st.file_uploader(
        "拖拽或点击上传文件",
        type=['txt', 'pdf', 'docx', 'md', 'csv'],
        help="支持的文档类型：TXT, PDF, DOCX, Markdown, CSV",
        key="sidebar_file_uploader"
    )
    if uploaded_file is not None:
        st.success(f"文件 {uploaded_file.name} 上传成功！")
        if uploaded_file.name not in st.session_state.knowledge_bases:
            st.session_state.knowledge_bases.append(uploaded_file.name)
            st.rerun()

    # 支持的文档类型提示
    st.info("📌 支持的文档类型：TXT, PDF, DOCX, Markdown, CSV (单个文件不超过200MB)")
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("---")

    # 系统状态
    st.markdown("### 📊 系统状态")
    st.info(f"知识库数量: {len(st.session_state.knowledge_bases)}")
    st.info(f"对话条数: {len(st.session_state.chat_history)}")

    # GitHub仓库链接
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown(
            "[![GitHub](https://img.shields.io/badge/GitHub-View_on_GitHub-blue?logo=GitHub)](https://github.com/Dreamt0511/EchoMind)")
        st.markdown("跳转到本项目GitHub仓库")

    st.markdown('</div>', unsafe_allow_html=True)

# 运行说明
if __name__ == "__main__":
    # 这个注释可以帮助用户理解如何运行应用
    st.markdown("""
    <!-- 
    运行方法：
    1. 安装依赖：pip install streamlit pandas
    2. 运行：streamlit run app.py
    -->
    """, unsafe_allow_html=True)