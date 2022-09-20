
import torch
import torch.nn.functional as F
from torch import profiler


def run_conv2d(input_shape, weight_shape, other_args, profile_folder):
    input = torch.ones(input_shape, dtype=torch.float32, device='cuda')
    weight = torch.ones(weight_shape, dtype=torch.float32, device='cuda')
    bias = other_args[0]
    stride = other_args[1]
    padding = other_args[2]
    dilation = other_args[3]
    groups = other_args[4]
    # warmup
    torch.cuda.nvtx.range_push("warmup")
    for i in range(11):
        x = F.conv2d(input, weight, bias, stride, padding, dilation, groups)
    torch.cuda.nvtx.range_pop()
    torch.cuda.nvtx.range_push("run")
    for i in range(1000):
        x = F.conv2d(input, weight, bias, stride, padding, dilation, groups)
    return x


def work():

    input_shape = (64, 224, 112, 112)
    weight_shape = (224, 112, 3, 3)
    other_args = [None, (2, 2), (1, 1), (1, 1), 2]
    activity_groups = []
    activity_groups.append(profiler.ProfilerActivity.CUDA)
    activity_groups.append(profiler.ProfilerActivity.CPU)
    profile_detailed = True
    profile_folder = "/tmp/logs-9.19"

    with profiler.profile(
        schedule=profiler.schedule(wait=0, warmup=0, active=1),
        activities=activity_groups,
        record_shapes=profile_detailed,
        profile_memory=profile_detailed,
        with_stack=profile_detailed,
        with_flops=profile_detailed,
        on_trace_ready=profiler.tensorboard_trace_handler(profile_folder)
    ) as prof:
        run_conv2d(input_shape, weight_shape, other_args, profile_folder)





work()