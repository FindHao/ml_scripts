import torch
from torch import profiler
import numpy as np



def profile():
    activity_groups = []
    activity_groups.append(profiler.ProfilerActivity.CUDA)
    activity_groups.append(profiler.ProfilerActivity.CPU)
    profile_detailed = True

    input_shape = [8, 12, 64, 64, 64]
    input = torch.ones(input_shape, dtype=torch.float32, device='cuda')
    
    with profiler.profile(
        schedule=profiler.schedule(wait=0, warmup=0, active=1),
        activities=activity_groups,
        record_shapes=profile_detailed,
        profile_memory=profile_detailed,
        with_stack=profile_detailed,
        with_flops=profile_detailed,
        on_trace_ready=profiler.tensorboard_trace_handler('/tmp/logs/')
    ) as prof:
       x = input / 20.0
       x2 = input * 4
       x3 = input / np.sqrt(16)

    
# profile()

def work():
    input_shape = [8, 12, 64, 64, 64]
    input = torch.ones(input_shape, dtype=torch.float32, device='cuda')
    x = input / 20.0

work()