import torch

# @torch.compile()
# def f(x, y):
#     return x + y.sum()

@torch.compile()
def f(x, y):
    return x + y.sum()

f(torch.randn(3, device='cuda'), torch.randn(3))
