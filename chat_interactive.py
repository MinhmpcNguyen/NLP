import asyncio
import os
import random
import time

import streamlit as st
import torch
from google import genai

from gemini_chatbot import answer_query

print("Libraries loaded successfully")
st.set_page_config(
    page_title="HUST Assistant",
    initial_sidebar_state="collapsed",
)

gemini_api_key = os.getenv("GEMINI_API_KEY") or "your-default-api-key"


# def chat(input, max_new_tokens=100):
#     """Generate response"""
#     input_text = format_input(input)

#     response = "Bot mimics your message", input_text
#     return response, "10s"

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

models = ["vinllama-7B", "vietnamese Llama2-7B", "Gemini_RAG"]
st.session_state["model"] = st.sidebar.selectbox("Select model", models, index=0)

st.markdown('<div class="main-container">', unsafe_allow_html=True)


# if "model" not in st.session_state:
#     model_path = "results_final1\llama3-mental-health-lora"
#     tokenizer = AutoTokenizer.from_pretrained(model_path)
#     st.session_state.tokenizer = tokenizer = AutoTokenizer.from_pretrained(model_path)
#     st.session_state.model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "user_input" not in st.session_state:
    st.session_state.user_input = ""

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if input_text := st.chat_input():
    with st.chat_message("user"):
        st.markdown(input_text)
        st.session_state.messages.append({"role": "user", "content": input_text})
        st.session_state.chat_history.append({"role": "user", "content": input_text})

    with st.chat_message("bot"):
        response_text = asyncio.run(answer_query(input_text))
        st.markdown(response_text)
        st.session_state.messages.append({"role": "bot", "content": response_text})
