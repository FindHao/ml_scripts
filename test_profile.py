from torch import profiler
import torch
import torch.nn.functional as F
import argparse
from typing import Dict, List
from torch._C._autograd import _ProfilerEvent, DeviceType

class event_light:
    def __init__(self, event):
        self.duration_us = event.duration_us()
        self.start_us = event.start_us()
        self.end_us = event.duration_us() + event.start_us()
        self.include_events = [event]


def run_conv2d(input_shape, weight_shape, other_args, profile_folder):
    input = torch.ones(input_shape, dtype=torch.float32, device='cuda')
    weight = torch.ones(weight_shape, dtype=torch.float32, device='cuda')
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


def merge_interval(intervals):
    if not intervals:
        return []
    intervals.sort(key=lambda x: x.start_us)
    res = []
    res.append(intervals[0])
    for i in range(1, len(intervals)):
        last = res[-1]
        if intervals[i].start_us <= last.end_us:
            last.end_us = max(last.end_us, intervals[i].end_us)
            last.include_events.extend(intervals[i].include_events)
        else:
            res.append(intervals[i])
    return res
        
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
    # event_tree = prof.profiler.kineto_results.experimental_event_tree()
    events_raw = prof.profiler.kineto_results.events()
    events = []
    for event in events_raw:
        # tid_root.setdefault(event.start_tid, []).append(event)
        if event.device_type() == DeviceType.CUDA:
            events.append(event_light(event))
            print(event.name())
    intervals = merge_interval(events)
    print("=====after merge====")
    sum_gpu_active = 0
    for interval in intervals:
        print(interval.start_us, interval.end_us)
        sum_gpu_active += interval.end_us - interval.start_us
        for event in interval.include_events:
            print(event.name())
    print("GPU active time:", sum_gpu_active/1e3, "ms")
    return x


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    SUPPORT_BATCHSIZE_LIST = ['32', '64']
    parser.add_argument("--bs", choices=SUPPORT_BATCHSIZE_LIST, required=True,
                        help="Specify batch size to the test.")
    parser.add_argument("--profile-folder", default="./logs",
                        help="Save profiling model traces to this directory.")
    args, extra_args = parser.parse_known_args()
    torch.backends.cudnn.benchmark = False
    if args.bs == '64':
        input_shape = (64, 224, 112, 112)
        other_args = [None, (2, 2), (1, 1), (1, 1), 2]
    else:
        input_shape = (32, 224, 56, 56)
        other_args = [None, (1, 1), (1, 1), (1, 1), 2]
    weight_shape = (224, 112, 3, 3)
    profile(input_shape, weight_shape, other_args, args.profile_folder)
    # run_conv2d(input_shape, weight_shape, other_args, args.profile_folder)
