import streamlit as st

# query_params = st.query_params
# show_results = query_params.get("page", "chat") == "top_results"

st.set_page_config(
    page_title="HUST Assistant",
    initial_sidebar_state="expanded",
)

st.title("Welcome to HUST Assitant")
st.write("Trợ lý ảo giúp bạn tra cứu nhanh chóng thông tin chính thống của Đại học Bách Khoa Hà Nội.")
st.markdown("---")

if "crawl" not in st.session_state:
    st.session_state.crawl = False

st.session_state.crawl = st.toggle("Deeper Search Mode", value=st.session_state.get("crawl", False))

if st.button("Bắt đầu trò chuyện"):
    st.switch_page("pages/chat.py")
