from transformers import pipeline
from datetime import datetime
import os
import torch
import torch.profiler
import contextlib

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

ENABLE_PROFILER = os.environ.get("ENABLE_PROFILER") == "1"
profiler_context = contextlib.nullcontext()
if ENABLE_PROFILER:
    print("PyTorch profiler is enabled.")
    profiler_context = torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        on_trace_ready=torch.profiler.tensorboard_trace_handler('./pytorch_profiler/', use_gzip=True),
        record_shapes=False,
        profile_memory=False,
        with_stack=True,
    )

generator = pipeline(
    "text-generation",
    model="openai/gpt-oss-20b",
    torch_dtype="auto",
    device_map="auto",  # Automatically place on available GPUs
)

messages = [
    {"role": "user", "content": "Explain what MXFP4 quantization is."},
]

# max_tokens = 10 if ENABLE_PROFILER else 200
max_tokens = 200

with profiler_context as prof:
    result = generator(
        messages,
        max_new_tokens=max_tokens,
        temperature=1.0,
    )

print(result[0]["generated_text"])

if TRITONPARSE_ENABLE:
    import tritonparse.utils
    tritonparse.utils.unified_parse(log_dir, out=output_dir, overwrite=True)
