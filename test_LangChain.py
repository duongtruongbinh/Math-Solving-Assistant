import streamlit as st
import pandas as pd
import numpy as np
import os
import bitsandbytes as bnb
import torch
import torch.nn as nn
import transformers

from langchain_community.llms import HuggingFacePipeline
from langchain.chains import LLMChain
from langchain_core.prompts import PromptTemplate

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
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path)
    return model, tokenizer


# def generate_response(prompt, model, tokenizer, device):
#     # Tokenize input prompt
#     encoding = tokenizer(prompt, return_tensors="pt").to(device)
#     print(device)
#     # Generate response
#     generation_config = model.generation_config
#     generation_config.max_new_tokens = 200
#     # generation_config.temperature = 0.7
#     generation_config.top_p = 0.7
#     generation_config.num_return_sequences = 1
#     generation_config.pad_token_id = tokenizer.eos_token_id
#     generation_config.eos_token_id = tokenizer.eos_token_id
#     with torch.inference_mode():
#         outputs = model.generate(
#             input_ids=encoding.input_ids,
#             attention_mask=encoding.attention_mask,
#             generation_config=generation_config
#         )

#     # Decode and return response
#     return tokenizer.decode(outputs[0], skip_special_tokens=True)


def main():
    st.title("Vinallama Chatbot")

    # Load model
    model, tokenizer = load_model(model_path)

    text_generation_pipeline = transformers.pipeline(
        model=model,
        tokenizer=tokenizer,
        task="text-generation",
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
        repetition_penalty=1.1,
        return_full_text=True,
        max_new_tokens=200,
    )
    my_pipeline = HuggingFacePipeline(pipeline=text_generation_pipeline)
    user_input = st.text_input("Hãy nhập gì đó để chat với tôi")

    # Định nghĩa template cho prompt
    template = prompt = """<|im_start|>system
    Bạn là một trợ lí AI hữu ích. Hãy trả lời người dùng một cách chính xác.
    <|im_end|>
    <|im_start|>user
    {user_input}<|im_end|>
    <|im_start|>assistant"""

    prompt = PromptTemplate(template=template, input_variables=["user_input"])

    llm_chain = LLMChain(prompt=prompt,
                         llm=my_pipeline
                         )
    # Tạo đối tượng LLMChain
    # prompt = PromptTemplate(template=template, input_variables=["user_input"])

    if st.button("Gửi"):
        response = llm_chain.invoke({"user_input": user_input})
        # print response with beautiful formatting
        st.write(response)


if __name__ == "__main__":
    main()
