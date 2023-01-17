"""Test different kernel selections for cudnn 8.5.0 and cudnn 8.3.2"""

from torch import profiler
import torch
import torch.nn.functional as F
import argparse

# out_channels = 1024
# input_shape = [1, 1024, 50, 75]
# stride = (1, 1)
# padding = (1, 1)
# dilation = (1, 1)
# groups = 1
# weight_shape = (out_channels, input_shape[1] // groups, 3, 3)
# weight = torch.ones(weight_shape, dtype=torch.float32, device='cuda')
# x = torch.ones(input_shape, dtype=torch.float32, device='cuda')
# bias_shape = (out_channels,)
# bias = torch.ones(bias_shape, dtype=torch.float32, device='cuda')

input_shape = [1, 1024, 50, 75]
stride = (1, 1)
padding = (1, 1)
dilation = (1, 1)
groups = 1
weight_shape = (1024, 1024, 3, 3)
weight = torch.ones(weight_shape, dtype=torch.float32, device='cuda')
x = torch.ones(input_shape, dtype=torch.float32, device='cuda')
bias_shape = (1024,)
bias = torch.ones((1024, ), dtype=torch.float32, device='cuda')

def run2():
    
    for i in range(100):
        F.conv2d(x, weight, bias, (1, 1), (1, 1), (1, 1), 1)

def run():
    
    for i in range(100):
        F.conv2d(x, weight, bias, stride, padding, dilation, groups)

def profile_run():
    # warmup
    for i in range(10):
        F.conv2d(x, weight, bias, stride, padding, dilation, groups)
    nwarmup = 4
    with profiler.profile(
        schedule=profiler.schedule(wait=0, warmup=nwarmup, active=1),
        activities=[profiler.ProfilerActivity.CUDA],
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
        with_flops=True,
        on_trace_ready=profiler.tensorboard_trace_handler("./logs")
    ) as prof:
        for i in range(nwarmup + 1):
            F.conv2d(x, weight, bias, stride, padding, dilation, groups)
    print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=10))

if __name__ == "__main__":
    # add a parser to parse the arguments, the argument is --profile to enable profiling
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()
    if args.profile:
        profile_run()
    else:
        run2()
        
        
    