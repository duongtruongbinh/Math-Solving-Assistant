
import os
import streamlit as st
from dotenv import load_dotenv
import google.generativeai as gen_ai
from streamlit_option_menu import option_menu
import torch
import transformers

# Additional imports from app.py
import pandas as pd
import numpy as np
import bitsandbytes as bnb
from googletrans import Translator
from datasets import load_dataset
from huggingface_hub import notebook_login
from peft import (LoraConfig, PeftConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training)
from transformers import (AutoConfig, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig)

# Environment setup from app.py
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Load model paths and configs from app.py
model_path = "model/models--vilm--vinallama-2.7b-chat/snapshots/b31d5f1306494b2bf10ecb0c6031077af3f5b39a"
peft_model_path = 'model\models--duongtruongbinh--vinallama-peft-2.7b-chat\snapshots\e90135fd01b4e813a99397be1fa1564af3b55714'
PEFT_MODEL = "duongtruongbinh/vinallama-peft-2.7b-chat"
device = 'cuda' if torch.cuda.is_available() else 'cpu'
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    load_in_8bit_fp32_cpu_offload=True
)

@st.cache_resource
def load_model(model_path):
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        trust_remote_code=True,
        quantization_config=bnb_config
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token
    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)
    config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=[
            "q_proj",
            "up_proj",
            "o_proj",
            "k_proj",
            "down_proj",
            "gate_proj",
            "v_proj"
        ],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )

    model = get_peft_model(model, config)
    return model, tokenizer

# Load environment variables
load_dotenv()

# Configure Streamlit page settings
st.set_page_config(
    page_title="Chat with AI Models!",
    page_icon=":brain:",
    layout="centered",
)

with st.sidebar:
    selected = option_menu(
        menu_title="Select Model",
        options=["Gemini", "Vinallama"],
        orientation="horizontal"
    )
def translate_role_for_streamlit(user_role):
    if user_role == "model":
        return "assistant"
    else:
        return user_role
def handle_chat_history(user_input, response, role='user'):
    if "chat_session" not in st.session_state:
        st.session_state.chat_session = {'history': []}
    # Append user message and model response to history
    st.session_state.chat_session['history'].append({'role': role, 'text': user_input})
    st.session_state.chat_session['history'].append({'role': 'assistant', 'text': response})

# Define the function to display chat
def display_chat():
    for message in st.session_state.chat_session['history']:
        with st.chat_message(message['role']):
            st.markdown(message['text'])

if selected == "Gemini":
    # Google Gemini-Pro AI model setup and chat logic
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    gen_ai.configure(api_key=GOOGLE_API_KEY)
    model = gen_ai.GenerativeModel('gemini-pro')
    if "chat_session" not in st.session_state:
        st.session_state.chat_session = model.start_chat(history=[])
    st.title("🤖 Gemini Pro - ChatBot")
    for message in st.session_state.chat_session.history:
        with st.chat_message(translate_role_for_streamlit(message.role)):
            st.markdown(message.parts[0].text)
    user_prompt = st.chat_input("Ask Gemini-Pro...")
    if user_prompt:
        # Add user's message to chat and display it
        st.chat_message("user").markdown(user_prompt)

        # Send user's message to Gemini-Pro and get the response
        gemini_response = st.session_state.chat_session.send_message(user_prompt)

        # Display Gemini-Pro's response
        with st.chat_message("assistant"):
            st.markdown(gemini_response.text)

elif selected == "Vinallama":
    # Vinallama model loading and chat logic
    vinallama_model, vinallama_tokenizer = load_model(model_path)
    st.title("Vinallama ChatBot")
    user_input = st.text_input("Type your message:", key="vinallama_input")
    if user_input:
        input_ids = vinallama_tokenizer.encode(user_input, return_tensors="pt").to(device)
        outputs = vinallama_model.generate(input_ids, max_length=50, num_return_sequences=1)
        response = vinallama_tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Handle chat history and display chat
        
        handle_chat_history(user_input, response, role='user')  # Store user input
        handle_chat_history("Vinallama: " + response, response, role='assistant')  # Store model response
        display_chat()  # Display the chat history
