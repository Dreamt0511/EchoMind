import streamlit as st
from konwlege_base import KnowledgeBaseService
from rag import RagService
import config_data as config
import os

# 页面标题
st.title("知识库助手")
st.subheader("开发者：Dreamt")

filesName = []
try:
    #读取已存储的知识库名，显示在左侧
    with open(config.filesName_path,'r',encoding='utf-8') as f:
        filesName = [fileName.strip() for fileName in f]
except FileNotFoundError:
    open(config.filesName_path, 'w', encoding='utf-8').close()

# 初始化session_state

if "service" not in st.session_state:
    st.session_state["service"] = KnowledgeBaseService()

if "rag" not in st.session_state:
    st.session_state["rag"] = RagService()

if "message" not in st.session_state:
    st.session_state["message"] = []

if "file_names" not in st.session_state:
    st.session_state["file_names"] = filesName


# 创建选项卡
tab1, tab2 = st.tabs(["📤 文件上传", "💬 对话助手"])

# 文件上传选项卡
with tab1:
    # 获取上传的文件对象
    uploader_file = st.file_uploader(
        "请上传.txt文件",
        type=["txt"],
        accept_multiple_files=False,  # 一次只允许上传一个文件
        key="file_uploader"
    )

    if uploader_file is not None:
        # 提取文件信息
        file_name = uploader_file.name
        file_size = uploader_file.size / 1024
        file_type = uploader_file.type
        # 获得文件内容
        text = uploader_file.getvalue().decode("utf-8")

        # 显示上传的文件信息
        st.subheader(f"文件名：{file_name}")
        st.write(f"文件格式：{file_type} | 大小：{file_size:.2f} kb")

        # 使用expander折叠面板，点击展开显示内容
        with st.expander("查看文件内容"):
            st.write(text)

        # 添加上传按钮，避免自动上传
        if st.button("上传到知识库", type="primary"):
            # 使用转圈动画，然后显示加载结果
            with st.spinner("载入数据库中..."):
                result = st.session_state["service"].upload_to_chroma_by_str(text, file_name)
                st.success(result)
                if result == "[成功]内容已存入向量数据库":# 添加庆祝效果
                    st.balloons()
                    st.session_state["file_names"].append(file_name)

# 对话助手选项卡
with tab2:
    # 显示历史消息
    for message in st.session_state["message"]:
        st.chat_message(message["role"]).write(message["content"])

    # 读取提问
    query = st.chat_input("请输入您的问题...")

    if query:
        # 显示用户提问
        st.chat_message("user").write(query)
        # 记录用户提问
        st.session_state["message"].append({"role": "user", "content": query})

        with st.spinner("正在检索知识库内容..."):
            ai_res_list = []  # 这个列表会作为缓存容器，收集所有流式返回的文本块。
            stream_res = st.session_state["rag"].chain.stream({"input": query}, config.session_config)


            # 调用后返回一个生成器对象
            def capture():
                for chunk in stream_res:
                    ai_res_list.append(chunk)
                    yield chunk


            # 显示ai回复,write_stream 期望传入的是一个生成器（或可迭代对象）
            st.chat_message("ai").write_stream(capture())
            # 记录回复
            st.session_state["message"].append({"role": "ai", "content": "".join(ai_res_list)})

# 侧边栏显示当前知识库状态
with st.sidebar:
    st.header("知识库状态")

    # 这里可以添加显示知识库统计信息的代码
    # 例如：文档数量、片段数量等
    st.info(f"知识库文档数量：{len(st.session_state["file_names"])}")
    # 使用expander折叠面板，点击展开显示上传的文件名
    with st.expander("已上传的知识库："):
        for file_name in st.session_state["file_names"]:
            st.write(file_name)
            







