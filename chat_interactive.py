import asyncio
import os
import random
import time
import torch 

import streamlit as st
import torch
from google import genai
from transformers import AutoTokenizer, BitsAndBytesConfig
from peft import AutoPeftModelForCausalLM

from gemini_chatbot import answer_query as gemini_answer_query
from vinallama_qa import answer_query as vinallama_answer_query

print("Libraries loaded successfully")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vinallama_model = "model/vinallama-7b-finetuned-25-percent"
    
bnb_config = BitsAndBytesConfig(
    load_in_4bit = True,
    bnb_4bit_use_double_quant = False,
    bnb_4bit_quant_type = "nf4",
    bnb_4bit_compute_dtype = torch.float16
)

tokenizer = AutoTokenizer.from_pretrained(vinallama_model)

llama2_model = AutoPeftModelForCausalLM.from_pretrained(
    vinallama_model,
    quantization_config = bnb_config,
    low_cpu_mem_usage = True,
    return_dict = True,
    torch_dtype=torch.float16,
    device_map=device
)

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


# if "model" not in st.session_state:
#     model_path = "results_final1\llama3-mental-health-lora"
#     tokenizer = AutoTokenizer.from_pretrained(model_path)
#     st.session_state.tokenizer = tokenizer = AutoTokenizer.from_pretrained(model_path)
#     st.session_state.model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto")

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
            response_text = asyncio.run(gemini_answer_query(input_text))
        elif st.session_state["model"] == "vinallama-7b":
            response_text = vinallama_answer_query(input_text, model=llama2_model, tokenizer=tokenizer)    
        else:
            response_text = "❌ Model not supported."

        st.markdown(response_text)
        st.session_state.messages.append({"role": "bot", "content": response_text})
