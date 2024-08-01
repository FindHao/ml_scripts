import torch


torch._dynamo.config.capture_scalar_outputs = True


def f(x, y):
    return x + y.sum()


data_type = torch.float32
test_dtypes = [
    torch.float32,
    torch.float64,
    torch.float16,
    torch.bfloat16,
    torch.int64,
]

for cpu_dtype in test_dtypes:
    print("cpu dtype:", cpu_dtype)
    x = torch.randn(3, device="cuda")
    y = torch.randn(3, device="cpu", dtype=cpu_dtype)
    x1 = x.clone()
    y1 = y.clone()
    reference_out = f(x, y)
    print("reference out: ", reference_out)
    # compiled_f = torch.compile(f)
    # out = compiled_f(x1, y1)
    # print("actual out: ", out)
    print("="*10)
