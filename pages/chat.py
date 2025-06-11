import streamlit as st

def gemini_response(input_text, num_result=10):
    response = "Đây là câu trả lời từ HUST Assistant cho câu hỏi của bạn."
    top_result = [
        {"url": "https://hust.edu.vn", "data": "Trang chủ của Đại học Bách Khoa Hà Nội."},
        {"url": "https://ctt-daotao.hust.edu.vn", "data": "Cổng thông tin đào tạo của sinh viên."}
    ]
    return response, top_result

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
            padding-top: 60px;
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
if "top_results" not in st.session_state:
    st.session_state.top_results = []
if "see_top_results" not in st.session_state:
    st.session_state.see_top_results = False




    
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
            response_text, top_results = gemini_response(input_text)
            st.markdown(response_text)
            st.session_state.top_results = top_results

        else:
            response_text = "Model not supported."
            st.markdown(response_text)

        st.session_state.messages.append({"role": "bot", "content": response_text})

if st.session_state.top_results:
    if st.button("See top results"):  
        st.session_state.see_top_results = True

if st.session_state.see_top_results:
    st.session_state.see_top_results = False
    st.switch_page(r"pages\top_results.py")