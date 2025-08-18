from transformers import pipeline
from datetime import datetime
import os

# Check environment variable to enable/disable tritonparse-related functionality
TRITONPARSE_ENABLE = os.environ.get("TRITONPARSE_ENABLE") == "1"

if TRITONPARSE_ENABLE:
    # Generate a unique log directory name with a timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Use a single environment variable as the common parent directory
    parent_dir = os.environ.get("TRITONPARSE_LOG_DIR", ".")
    log_dir = os.path.join(parent_dir, f"logs_{timestamp}") + "/"
    print(f"===>debug: Log directory: {log_dir}")
    output_dir = os.path.join(parent_dir, f"parsed_output_{timestamp}") + "/"
    import tritonparse.structured_logging
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

if TRITONPARSE_ENABLE:
    import tritonparse.utils
    tritonparse.utils.unified_parse(log_dir, out=output_dir, overwrite=True)
