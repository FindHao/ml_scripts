from torch import profiler
import torch
import torch.nn.functional as F
import argparse
import time

MAX_ITER = 1

def run_conv2d_time_measure(input_shape, weight_shape, other_args):
    input = torch.ones(input_shape, dtype=torch.float32, device='cuda')
    weight = torch.ones(weight_shape, dtype=torch.float32, device='cuda')
    bias = other_args[0]
    stride = other_args[1]
    padding = other_args[2]
    dilation = other_args[3]
    groups = other_args[4]
    torch.cuda.synchronize()
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    t0 = time.time_ns()
    start_event.record()
    for _i in range(MAX_ITER):  
      x = F.conv2d(input, weight, bias, stride, padding, dilation, groups)
    end_event.record()
    torch.cuda.synchronize()
    t1 = time.time_ns()
    gpu_time = start_event.elapsed_time(end_event)
    print('{:<20} {:>20}'.format("GPU Time per batch:", "%.3f milliseconds" % (gpu_time), sep=''))
    print('{:<20} {:>20}'.format("Total Wall Time:", "%.3f milliseconds" % ((t1 - t0) / 1_000_000)), sep='')
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

    parser.add_argument("--bs", choices=SUPPORT_BATCHSIZE_LIST, required=True,
                        help="Specify batch size to the test.")
    parser.add_argument("-p", "--profile", action="store_true", default=False,
                        help="Use pytorch profiler to profile the function.")
    parser.add_argument("--profile-folder", default="./logs", help="Save profiling model traces to this directory.")
    args, extra_args = parser.parse_known_args()
    if args.bs == '64':
        input_shape = (64, 224, 112, 112)
        other_args = [None, (2, 2), (1, 1), (1, 1), 2]
    else:
        input_shape = (32, 224, 56, 56)
        other_args = [None, (1, 1), (1, 1), (1, 1), 2]
    weight_shape = (224, 112, 3, 3)
    if args.profile == True:
        profile(input_shape, weight_shape, other_args, args.profile_folder)
    else:
        run_conv2d_time_measure(input_shape, weight_shape, other_args)
