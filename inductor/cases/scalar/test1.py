import torch

# @torch.compile()
# def f(x, y):
#     return x + y.sum()


# def f(x, y):
#     # return x + y.sum()
#     return y.sum().item()

# input_gpu = torch.randn(3, device='cuda')
# input_cpu = torch.randn(3)
# input_gpu2 = input_gpu.clone()
# input_cpu2 = input_cpu.clone()

# reference_out = f(input_gpu, input_cpu)
# compiled_f = torch.compile(f)
# out = compiled_f(input_gpu2, input_cpu2)
# print("actual out: ", out)
# print("type of out: ", type(out))
# print("reference out: ", reference_out)
# print("type of reference out: ", type(reference_out))

# torch._dynamo.config.capture_scalar_outputs = True

# @torch.compile()
# def foo(x, y):
#     a = x.sum().item() + 1
#     return y+a
# foo(torch.rand([20]), torch.rand([4]))

def fn(x):
    return x.sum().item()

input = torch.rand([20], dtype=torch.float64)
# input = torch.randint(0, 100, (5, 5), dtype=torch.int32)
# random_array_int64 = torch.randint(0, 2**32, (5, 5), dtype=torch.int64)

# 转换为 uint32 数组
# input = random_array_int64.to(torch.uint32)

input2 = input.clone()
reference_out = fn(input)
compiled_f = torch.compile(fn)
out = compiled_f(input2)
print("actual out: ", out)
print("type of out: ", type(out))
print("reference out: ", reference_out)
print("type of reference out: ", type(reference_out))


import math
# assert(math.isclose(out, reference_out, abs_tol=1e-10, rel_tol=1e-10))
