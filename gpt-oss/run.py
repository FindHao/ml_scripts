from transformers import pipeline
import tritonparse.structured_logging
from datetime import datetime

# Generate a unique log directory name with a timestamp
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_dir = f"./logs_{timestamp}/"
output_dir = f"./parsed_output_{timestamp}/"

tritonparse.structured_logging.init(log_dir, enable_trace_launch=True)

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
tritonparse.utils.unified_parse(log_dir, out=output_dir, overwrite=True)