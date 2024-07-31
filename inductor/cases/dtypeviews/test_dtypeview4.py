import torch
from torch._dynamo.testing import rand_strided


# def fn(x, y, x_dtype):
#     x = x.view(x_dtype)
#     y = y.view(x_dtype) + 1
#     return x@y

# x = rand_strided((2, 2), (2, 1), device='cuda:0', dtype=torch.float16)
# y = rand_strided((2, 2), (2, 1), device='cuda:0', dtype=torch.int16)
# x1 = x.clone()
# y1 = y.clone()

# z = fn(x, y, torch.float16)
# print(type(z))
# print(z)

# compiled_fn = torch.compile(fn)
# z1 = compiled_fn(x1, y1, torch.float16)
# print(type(z1))
# print(z1)

# def fn(x, x_dtype):
#     return x.view(x_dtype)

# x = rand_strided((2, 2), (2, 1), device='cuda:0', dtype=torch.float16)
# compiled_fn = torch.compile(fn)
# z1 = compiled_fn(x, torch.float32)
# print(z1.dtype)

device='cuda'

@torch.compile
def fn(x, y):
    x = x.view(torch.float16)
    y = y.view(torch.float16) + 1
    return x @ y

x = torch.randn((2, 2), device=device, dtype=torch.bfloat16)
y = torch.randn((2, 2), device=device, dtype=torch.bfloat16)
fn(x, y)
