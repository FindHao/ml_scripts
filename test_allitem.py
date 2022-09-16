import torch


def fun1():
    a = torch.rand(1, 2, device="cuda").bool()
    assert torch.all(a).item(), "Verify that `shifted_input_ids` has only positive values"

def fun2():
    a = torch.rand(1, 2,device="cuda").bool()
    if torch.any(torch.all(a)):
        v = False
    else:
        v = True
    assert v, "Verify that `shifted_input_ids` has only positive values"


if __name__ == "__main__":
    from torch import profiler
    activity_groups = []
    activity_groups.append(profiler.ProfilerActivity.CUDA)
    activity_groups.append(profiler.ProfilerActivity.CPU)
    profile_detailed=True
    with profiler.profile(
            schedule=profiler.schedule(wait=0, warmup=3, active=1),
            activities=activity_groups,
            record_shapes=profile_detailed,
            profile_memory=profile_detailed,
            with_stack=profile_detailed,
            with_flops=profile_detailed,
            on_trace_ready=profiler.tensorboard_trace_handler("/tmp/logs2/")
        ) as prof:
        for i in range(4):
            fun1()
            fun2()
            prof.step()
