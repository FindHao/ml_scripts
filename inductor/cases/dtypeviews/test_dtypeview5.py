import torch

@torch.compile()
def foo(x):
    return x.view([10, 10]).view(torch.int32)

foo(torch.rand([100, 1]))
