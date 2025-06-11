import streamlit as st

# query_params = st.query_params
# show_results = query_params.get("page", "chat") == "top_results"
import streamlit as st

st.set_page_config(
    page_title="HUST Assistant",
    initial_sidebar_state="expanded",
)

st.title("Welcome to HUST Assitant")

if st.button("Start!"):
    st.switch_page("pages/chat.py")
