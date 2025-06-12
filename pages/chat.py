import streamlit as st
from collections import defaultdict

from gemini_chatbot import answer_query as gemini_answer_query

st.set_page_config(
    page_title="HUST Assistant",
    initial_sidebar_state="expanded",
)

with st.sidebar:
    models = ["Gemini_RAG", "VinaLlama-7b"]
    selected_model = st.selectbox("Chọn mô hình", models, index=0)
    st.session_state["model"] = selected_model
    
st.markdown(
    """
    <style>
        .fixed-title {
            position: fixed;
            width: calc(100% - 32px);
            max-width: 730px;
            left: 58%;
            transform: translateX(-50%);
            background-color: #D32F2F;
            color: #FFFFFF;
            padding: 12px;
            text-align: center;
            font-size: 30px;
            font-weight: bold;
            z-index: 1000;
            border-radius: 8px;
            border-bottom: 3px solid #B71C1C;
        }
        .main-container {
            padding-top: 70px;
        }
    </style>
    <div class="fixed-title">HUST Assitant</div>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="main-container">', unsafe_allow_html=True)

if "messages" not in st.session_state:
    st.session_state.messages = []
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "search_results" not in st.session_state:
    st.session_state.search_results = []
if "see_search_results" not in st.session_state:
    st.session_state.see_search_results = False
if "crawl" not in st.session_state:
    st.session_state.crawl = False

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


if input_text := st.chat_input():
    with st.chat_message("user"):
        st.markdown(input_text)
        st.session_state.messages.append({"role": "user", "content": input_text})
        st.session_state.chat_history.append({"role": "user", "content": input_text})

    with st.chat_message("bot"):
        if st.session_state["model"] == "Gemini_RAG":         
#             response_text = "response"
#             search_results = defaultdict(str, {
#     "https://hust.edu.vn/vi/co-cau-to-chuc-bai-viet/giam-doc-hieu-truong-dhbk-ha-noi-qua-cac-thoi-ky.html":
#         "# Giám đốc/ Hiệu trưởng ĐHBK Hà Nội qua các thời kỳ...",
    
#     "https://nld.com.vn/giao-duc-khoa-hoc/truong-dh-bach-khoa-ha-noi-thanh-dh-bach-khoa-ha-noi-20230317134449418.htm":
#         "# Trường ĐH Bách khoa Hà Nội trở thành ĐH Bách khoa Hà Nội...",
    
#     "https://hust.edu.vn/vi/co-cau-to-chuc-bai-viet/ban-giam-doc-dai-hoc.html":
#         "# Ban Giám đốc đại học\nĐại học Bách khoa Hà Nội\n2025-05-07T15:07:15",
    
#     "https://hust.edu.vn/vi/news/hoat-dong-chung/giam-doc-dai-hoc-bach-khoa-ha-noi-truyen-cam-hung-cho-nhung-chu-ca-bach-khoa-vuon-ra-bien-lon-65519.html":
#         "# Giám đốc truyền cảm hứng cho sinh viên Bách khoa vươn ra biển lớn..."
# })

            response_text, search_results = gemini_answer_query(input_text, crawl = st.session_state.crawl)
            st.markdown(response_text)
            st.session_state.search_results = search_results

        else:
            response_text = "Model not supported."
            st.markdown(response_text)

        st.session_state.messages.append({"role": "bot", "content": response_text})

if st.session_state.search_results:
    if st.button("See search results"):  
        st.session_state.see_search_results = True

if st.session_state.see_search_results:
    st.session_state.see_search_results = False
    st.switch_page(r"pages\search_results.py")