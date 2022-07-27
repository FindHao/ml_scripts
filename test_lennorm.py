import torch
from torch import profiler
import numpy as np


def _len_and_dim_norm(vectors):
    """
    length and attention head size dim normalization
    """
    vectors = vectors * torch.rsqrt(
        torch.tensor(64, device=vectors.device, dtype=vectors.dtype)
    )
    return vectors

def _len_and_dim_norm2(vectors):
    """
    length and attention head size dim normalization
    """
    # tmp = torch.tensor(64, device=vectors.device, dtype=vectors.dtype)
    vectors = vectors / np.sqrt(64)
    return vectors



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
       x  = _len_and_dim_norm(input)
       y = _len_and_dim_norm2(input)
       return y
    
profile()