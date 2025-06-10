import asyncio
import os
import random
import time
import torch 

import streamlit as st
import torch
from google import genai
from transformers import AutoTokenizer, BitsAndBytesConfig

# from gemini_chatbot import answer_query as gemini_answer_query
# from vinallama_qa import answer_query as vinallama_answer_query

print("Libraries loaded successfully")

def gemini_response(input_text, num_result=10):
    response = "Đây là câu trả lời từ HUST Assistant cho câu hỏi của bạn."
    top_result = [
        {"url": "https://hust.edu.vn", "data": "Trang chủ của Đại học Bách Khoa Hà Nội."},
        {"url": "https://ctt-daotao.hust.edu.vn", "data": "Cổng thông tin đào tạo của sinh viên."}
    ]
    return response, top_result




query_params = st.experimental_get_query_params()
show_results = query_params.get("page", ["chat"])[0] == "top_results"

# -----------------------------
# Nếu đang ở trang top_results thì render riêng
if show_results:
    st.set_page_config(
        page_title="Top Results - HUST Assistant",
        initial_sidebar_state="collapsed",
    )
    st.title("🔍 Top Results từ HUST Assistant")
    st.markdown("Các nguồn thông tin có thể hữu ích cho câu hỏi của bạn:")

    top_results = st.session_state.get("top_results", [])

    if top_results:
        for idx, result in enumerate(top_results, 1):
            st.markdown(f"**Nguồn {idx}:**")
            st.markdown(f"- 🔗 [Link]({result['url']})")
            st.markdown(f"- 📄 Nội dung: {result['data']}")
            st.markdown("---")
    else:
        st.warning("❌ Không có kết quả nào để hiển thị.")

    st.markdown("[⬅️ Quay lại trang chat](?page=chat)")
    st.stop()











st.set_page_config(
    page_title="HUST Assistant",
    initial_sidebar_state="collapsed",
)

gemini_api_key = os.getenv("GEMINI_API_KEY") or "your-default-api-key"


st.markdown(
    """
    <style>
        .fixed-title {
            position: fixed;
            width: calc(100% - 32px);
            max-width: 730px;
            left: 50%;
            transform: translateX(-50%);
            background-color: #D32F2F;
            color: #FFFFFF;
            padding: 12px;
            text-align: center;
            font-size: 20px;
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


models = ["Gemini_RAG", "VinaLlama-7b"]
st.session_state["model"] = st.sidebar.selectbox("Select model", models, index=0)

st.markdown('<div class="main-container">', unsafe_allow_html=True)

if "messages" not in st.session_state:
    st.session_state.messages = []
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

                
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

            if top_results:
                # Link điều hướng tới trang top_results
                st.markdown("[➡️ Xem chi tiết nguồn](?page=top_results)")
        # elif st.session_state["model"] == "vinallama-7b":
        #     response_text = vinallama_answer_query(input_text, model=llama2_model, tokenizer=tokenizer)    
        else:
            response_text = "❌ Model not supported."
            st.markdown(response_text)
        
        st.session_state.messages.append({"role": "bot", "content": response_text})


        
if st.session_state.chat_history:
    if st.button("Show Full Query History"):
        with st.expander("Full Query History", expanded=False):
            for idx, msg in enumerate(st.session_state.chat_history, 1):
                role = "User" if msg["role"] == "user" else "Bot"
                st.markdown(f"**{role} {idx}:** {msg['content']}")
