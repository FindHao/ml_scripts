import torch

@torch.compile()
def foo(x):
    return x.view(torch.float64).view([10, 10])
    # return x.view([10, 10]).view(torch.int32)

foo(torch.rand([100, 1], device="cuda", dtype=torch.cfloat))
