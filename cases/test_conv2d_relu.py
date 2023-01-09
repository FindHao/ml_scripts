
import time
from torch import profiler
import torch
import argparse

out_channels = 64
input_shape = [1, 3, 224, 224]
weight_shape = [64, 3, 7, 7]
stride = (2, 2)
padding = (3, 3)
dilation = (1, 1)
groups = 1

input = torch.ones(input_shape, dtype=torch.float32, device='cpu')
# use torch.quantize_per_tensor to quantize the input
input = torch.quantize_per_tensor(input, scale=1.0, zero_point=0, dtype=torch.quint8)
scale=1.0
zero_point=0
w = torch.ones(weight_shape, dtype=torch.float32, device='cpu')
w = torch.quantize_per_tensor(w, scale=1.0, zero_point=0, dtype=torch.qint8)
b = torch.ones(out_channels, dtype=torch.float32, device='cpu')
packed_params = torch.ops.quantized.conv2d_prepack(w, b, stride, padding, dilation, groups)
def run():
    for i in range(10):
        torch.ops.quantized.conv2d_relu(input, packed_params, scale, zero_point)
    # measure execution time
    t0 = time.time_ns()
    for i in range(100):
        torch.ops.quantized.conv2d_relu(
            input, packed_params, scale, zero_point)
    t1 = time.time_ns()
    print("time (ms):", (t1 - t0) / 1000000)

def profile_run():
    # warmup
    for i in range(10):
        torch.ops.quantized.conv2d_relu(input, packed_params, scale, zero_point)
    nwarmup = 4
    with profiler.profile(
        schedule=profiler.schedule(wait=0, warmup=nwarmup, active=1),
        activities=[profiler.ProfilerActivity.CPU],
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
        with_flops=False,
        on_trace_ready=profiler.tensorboard_trace_handler("./logs")
    ) as prof:
        for i in range(nwarmup + 1):
            torch.ops.quantized.conv2d_relu(
                input, packed_params, scale, zero_point)
            prof.step()

if __name__ == "__main__":
    # add a parser to parse the arguments, the argument is --profile to enable profiling
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()
    if args.profile:
        profile_run()
    else:
        run()
        
        

