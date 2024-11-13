MODEL_NAME = "CohereForAI/aya-23-8b"

# you may want to change the following parameters depending on your GPU configuration

# free T4 instance
# QUANTIZE_4BIT = True
# USE_GRAD_CHECKPOINTING = True
# TRAIN_BATCH_SIZE = 2
# TRAIN_MAX_SEQ_LENGTH = 512
# USE_FLASH_ATTENTION = False
# GRAD_ACC_STEPS = 16

# equivalent A100 setting
QUANTIZE_4BIT = True
USE_GRAD_CHECKPOINTING = True
TRAIN_BATCH_SIZE = 16
TRAIN_MAX_SEQ_LENGTH = 512
USE_FLASH_ATTENTION = True
GRAD_ACC_STEPS = 2
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,HfArgumentParser,TrainingArguments,pipeline, logging
from transformers.generation import GenerationConfig
from peft import LoraConfig, PeftModel, prepare_model_for_kbit_training, get_peft_model
import os,torch
import bitsandbytes as bnb
from datasets import load_dataset
from trl import SFTTrainer
from datasets import Dataset
import pyarrow as pa
import pyarrow.dataset as ds
import pandas as pd
import re
import wandb

# Load Model
quantization_config = None
if QUANTIZE_4BIT:
  quantization_config = BitsAndBytesConfig(
      load_in_4bit=True,
      bnb_4bit_quant_type="nf4",
      bnb_4bit_use_double_quant=True,
      bnb_4bit_compute_dtype=torch.bfloat16,
  )

attn_implementation = None
if USE_FLASH_ATTENTION:
  attn_implementation="flash_attention_2"
device = torch.device("cuda")

model = AutoModelForCausalLM.from_pretrained(
          MODEL_NAME,
          quantization_config=quantization_config,
          attn_implementation=attn_implementation,
          torch_dtype=torch.bfloat16,
          device_map="auto",
        )

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
def get_message_format(prompts):
  messages = []

  for p in prompts:
    messages.append(
        [{"role": "user", "content": p}]
      )

  return messages

def generate_aya_23(
      prompts,
      model,
      temperature=0.3,
      top_p=0.75,
      top_k=0,
      max_new_tokens=1024
    ):
  messages = get_message_format(prompts)
  input_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        padding=True,
        return_tensors="pt",
      )
  input_ids = input_ids.to(model.device)
  print(model.device)
  prompt_padded_len = len(input_ids[0])

  gen_tokens = model.generate(
        input_ids,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        max_new_tokens=max_new_tokens,
        do_sample=True,
      )

  # get only generated tokens
  gen_tokens = [
      gt[prompt_padded_len:] for gt in gen_tokens
    ]

  gen_text = tokenizer.batch_decode(gen_tokens, skip_special_tokens=True)
  return gen_text

  # Test generations on langauges in Aya 23 set
prompts = [
    "Write a list of three fruits and tell me about each of them", # English
]

generations = generate_aya_23(prompts, model)

for p, g in zip(prompts, generations):
  print(
      "PROMPT", p ,"RESPONSE", g, "\n", sep="\n"
    )
