from transformers import pipeline
import tritonparse.structured_logging
tritonparse.structured_logging.init("./logs/", enable_trace_launch=True)

generator = pipeline(
    "text-generation",
    model="openai/gpt-oss-20b",
    torch_dtype="auto",
    device_map="auto",  # Automatically place on available GPUs
)

messages = [
    {"role": "user", "content": "Explain what MXFP4 quantization is."},
]

result = generator(
    messages,
    max_new_tokens=200,
    temperature=1.0,
)

print(result[0]["generated_text"])

import tritonparse.utils
tritonparse.utils.unified_parse("./logs/")