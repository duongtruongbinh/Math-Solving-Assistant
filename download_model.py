import streamlit as st
import pandas as pd
import numpy as np
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

base_model_path = "model/models--vilm--vinallama-2.7b-chat/snapshots/b31d5f1306494b2bf10ecb0c6031077af3f5b39a"
# download base model
model_name = 'vilm/vinallama-2.7b-chat'
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    trust_remote_code=True,
    quantization_config=bnb_config,
    cache_dir="./model"
)
tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir="./model")
tokenizer.pad_token = tokenizer.eos_token

# download fine-tuned model
# PEFT_MODEL = "duongtruongbinh/vinallama-peft-2.7b-chat"
# config = PeftConfig.from_pretrained(PEFT_MODEL, cache_dir="./model")
# model = AutoModelForCausalLM.from_pretrained(
#     config.base_model_name_or_path,
#     return_dict=True,
#     quantization_config=bnb_config,
#     device_map="auto",
#     trust_remote_code=True,
#     cache_dir="./model"
# )

# tokenizer = AutoTokenizer.from_pretrained(
#     config.base_model_name_or_path, cache_dir="./model")
# tokenizer.pad_token = tokenizer.eos_token

# model = PeftModel.from_pretrained(model, PEFT_MODEL, cache_dir="./model")


def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0

    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainables%: {100 * trainable_params / all_param}"
    )


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
print_trainable_parameters(model)
