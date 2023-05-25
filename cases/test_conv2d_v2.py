from torch import profiler
import torch
import torch.nn.functional as F
import argparse

import time

def timer(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"{func.__name__} executed in {end - start} seconds")
        return result
    return wrapper

def gpu_timer(func):
    def wrapper(*args, **kwargs):
        torch.cuda.synchronize()
        start = time.time_ns()
        result = func(*args, **kwargs)
        torch.cuda.synchronize()
        end = time.time_ns()
        duration_ms = (end - start) / 1e6
        if duration_ms > 1000:
            print(f"{func.__name__} executed in {duration_ms / 1000:.2f} seconds")
        else:
            print(f"{func.__name__} executed in {duration_ms:.2f} milliseconds")
        return result
    return wrapper

@gpu_timer
def run_conv2d(input, weight, other_args, profile_folder):
    bias = other_args[0]
    stride = other_args[1]
    padding = other_args[2]
    dilation = other_args[3]
    groups = other_args[4]
    # warmup
    for i in range(11):
        x = F.conv2d(input, weight, bias, stride, padding, dilation, groups)
    for i in range(1000):
        x = F.conv2d(input, weight, bias, stride, padding, dilation, groups)
    return x


def profile(input_shape, weight_shape, other_args, profile_folder):
    activity_groups = []
    activity_groups.append(profiler.ProfilerActivity.CUDA)
    activity_groups.append(profiler.ProfilerActivity.CPU)
    profile_detailed = True

    input = torch.ones(input_shape, dtype=torch.float32, device='cuda')
    weight = torch.ones(weight_shape, dtype=torch.float32, device='cuda')
    bias = other_args[0]
    stride = other_args[1]
    padding = other_args[2]
    dilation = other_args[3]
    groups = other_args[4]
    
    with profiler.profile(
        schedule=profiler.schedule(wait=0, warmup=0, active=1),
        activities=activity_groups,
        record_shapes=profile_detailed,
        profile_memory=profile_detailed,
        with_stack=profile_detailed,
        with_flops=profile_detailed,
        on_trace_ready=profiler.tensorboard_trace_handler(profile_folder)
    ) as prof:
      x = F.conv2d(input, weight, bias, stride, padding, dilation, groups)
    return x


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    SUPPORT_BATCHSIZE_LIST = ['32', '64']
    parser.add_argument("--bs", choices=SUPPORT_BATCHSIZE_LIST, required=False,default='64',
                        help="Specify batch size to the test.")
    parser.add_argument("--profile-folder", default="./logs", help="Save profiling model traces to this directory.")
    args, extra_args = parser.parse_known_args()
    torch.backends.cudnn.benchmark = True
    if args.bs == '64':
        input_shape = (64, 224, 112, 112)
        other_args = [None, (2, 2), (1, 1), (1, 1), 2]
    else:
        input_shape = (32, 224, 56, 56)
        other_args = [None, (1, 1), (1, 1), (1, 1), 2]
    weight_shape = (224, 112, 3, 3)
    input = torch.ones(input_shape, dtype=torch.float32, device='cuda')
    weight = torch.ones(weight_shape, dtype=torch.float32, device='cuda')
    # profile(input_shape, weight_shape, other_args, args.profile_folder)
    run_conv2d(input, weight, other_args, args.profile_folder)
