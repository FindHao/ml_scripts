import torch
torch._dynamo.config.capture_scalar_outputs = True

@torch.compile()
def foo(x, y):
    a = x.sum().item()
    return y+a
foo(torch.rand([20]), torch.rand([4]))
