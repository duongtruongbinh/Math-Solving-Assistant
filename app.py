import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json
import os
import bitsandbytes as bnb
import torch
import torch.nn as nn
import transformers

from googletrans import Translator
from pprint import pprint
from datasets import load_dataset
from huggingface_hub import notebook_login
from peft import (
    LoraConfig,
    PeftConfig,
    PeftModel,
    get_peft_model,
    prepare_model_for_kbit_training
)
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig
)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
model_path = "model/models--vilm--vinallama-2.7b-chat/snapshots/b31d5f1306494b2bf10ecb0c6031077af3f5b39a"
device = 'cuda' if torch.cuda.is_available() else 'cpu'


@st.cache_resource
def load_model(model_path):
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        load_in_8bit_fp32_cpu_offload=True
    )
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


def generate_response(prompt, model, tokenizer, device):
    # Tokenize input prompt
    encoding = tokenizer(prompt, return_tensors="pt").to(device)
    print(device)
    # Generate response
    generation_config = model.generation_config
    generation_config.max_new_tokens = 200
    # generation_config.temperature = 0.7
    generation_config.top_p = 0.7
    generation_config.num_return_sequences = 1
    generation_config.pad_token_id = tokenizer.eos_token_id
    generation_config.eos_token_id = tokenizer.eos_token_id
    with torch.inference_mode():
        outputs = model.generate(
            input_ids=encoding.input_ids,
            attention_mask=encoding.attention_mask,
            generation_config=generation_config
        )

    # Decode and return response
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


def main():
    st.title("Vinallama Chatbot")

    # Load model
    model, tokenizer = load_model(model_path)

    user_input = st.text_input("Hãy nhập gì đó để chat với tôi")
    prompt = """
    <|im_start|>system
    Bạn là một trợ lí AI hữu ích. Hãy trả lời người dùng một cách chính xác.
    <|im_end|>
    <|im_start|>user
    {user_input}
    <|im_end|>
    <|im_start|>assistant
    """.strip()

    if st.button("Gửi"):
        response = generate_response(prompt.format(
            user_input=user_input), model, tokenizer, device)
        st.write("Vinallama:", response)


if __name__ == "__main__":
    main()
