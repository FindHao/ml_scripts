# pip install transformers==4.41.1
import os

import torch
from huggingface_hub import login
from torch.export.dynamic_shapes import Dim
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

# gen_fun = torch.compile(model.generate)
gen_fun = model.generate
gen_tokens = gen_fun(
    input_ids,
    max_new_tokens=100,
    do_sample=True,
    temperature=0.3,
)
gen_text = tokenizer.decode(gen_tokens[0])
print("gen_text", gen_text)


# export part
print("======export part=======")

print("======wrap model=======")


class CustomModelWrapper(torch.nn.Module):
    def __init__(self, model):
        super(CustomModelWrapper, self).__init__()
        self.model = model

    def forward(self, input_ids, max_new_tokens=100, do_sample=True, temperature=0.3):
        return self.model.generate(
            input_ids,
            # max_new_tokens=max_new_tokens,
            # do_sample=do_sample,
            # temperature=temperature,
        )


# Create an instance of the custom model wrapper
custom_model = CustomModelWrapper(model)

# Use the custom model's forward method to generate tokens
gen_tokens = custom_model(input_ids)

gen_text = tokenizer.decode(gen_tokens[0])
print("gen_text", gen_text)

# export part
print("======export part=======")
ep = torch.export.export(
    custom_model,
    (input_ids,),
    dynamic_shapes=({0: Dim.AUTO, 1: Dim.AUTO},),
    strict=False,
)
print(ep)
output_path = torch._inductor.aoti_compile_and_package(
    ep,
    input_ids,
    package_path=os.path.join(os.getcwd(), "model.pt2"),
)
