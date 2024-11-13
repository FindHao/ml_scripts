# pip install transformers==4.41.1
import os

import torch
from huggingface_hub import login
from transformers import AutoModelForCausalLM, AutoTokenizer

login(token=os.getenv("HUGGINGFACE_TOKEN"))
device = torch.device("cuda")

model_id = "CohereForAI/aya-23-8B"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
)
device = torch.device("cuda")
# Format message with the command-r-plus chat template
messages = [
    {"role": "user", "content": "Anneme onu ne kadar sevdiğimi anlatan bir mektup yaz"}
]
input_ids = tokenizer.apply_chat_template(
    messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
)
input_ids = input_ids.to(model.device)
## <BOS_TOKEN><|START_OF_TURN_TOKEN|><|USER_TOKEN|>Anneme onu ne kadar sevdiğimi anlatan bir mektup yaz<|END_OF_TURN_TOKEN|><|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>

gen_tokens = model.generate(
    input_ids,
    max_new_tokens=100,
    do_sample=True,
    temperature=0.3,
)

gen_text = tokenizer.decode(gen_tokens[0])
print(gen_text)
