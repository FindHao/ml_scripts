import torch

device = torch.device("cuda")
full = torch.randn((16, 16)).to(device)
view = torch.as_strided(full, (16, 8), full.stride())

def foo(x):
    result = x + x
    result_strided = torch.empty_strided(x.size(), x.stride(), device=device)
    result_strided[:] = result
    return result_strided

eager_result = foo(view)

compiled_foo = torch.compile(foo, backend="inductor")
compiled_result = compiled_foo(view)

assert compiled_result.stride() == eager_result.stride()
