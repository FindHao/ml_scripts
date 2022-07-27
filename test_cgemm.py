import torch
from torch import Tensor
import torch.nn.functional as F



import pickle


def work():
    # input = pickle.load(open('/tmp/findhao2.input', 'rb'))
    # weight = pickle.load(open('/tmp/findhao2.weight', 'rb'))
    # tmp = [bias, self.stride, self.padding, self.dilation, self.groups]
    # tmp = pickle.load(open('/tmp/findhao2.tmp', 'rb'))


    input_shape = (64, 224, 112, 112)
    tmp = [None, (2, 2), (1, 1), (1, 1), 2]


    input_shape = (32, 224, 112, 112)


    # input_shape = (32, 224, 56, 56)
    tmp = [None, (1, 1), (1, 1), (1, 1), 2]

    
    w_shape = (224, 112, 3, 3)
    input = torch.ones(input_shape, dtype=torch.float32, device='cuda')
    weight = torch.ones(w_shape, dtype=torch.float32, device='cuda')
    
    bias = tmp[0]
    stride = tmp[1]
    padding = tmp[2]
    dilation = tmp[3]
    groups = tmp[4]
    # torch._foreach_zero_([weight, input])
    x =  F.conv2d(input, weight, bias, stride, padding, dilation, groups)
    return x
    
# work()

from torch import profiler
activity_groups = []
activity_groups.append(profiler.ProfilerActivity.CUDA)
activity_groups.append(profiler.ProfilerActivity.CPU)
profile_detailed=True
with profiler.profile(
        schedule=profiler.schedule(wait=0, warmup=0, active=1),
        activities=activity_groups,
        record_shapes=profile_detailed,
        profile_memory=profile_detailed,
        with_stack=profile_detailed,
        with_flops=profile_detailed,
        on_trace_ready=profiler.tensorboard_trace_handler("/tmp/logs2/")
    ) as prof:
  # add your code here.
    work()


