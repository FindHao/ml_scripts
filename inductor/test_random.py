import torch

torch.manual_seed(12345)

# print(torch.randint())

def fn():
    random_tensor1 = torch.randint(10, [1024], device="cuda")
    random_tensor2 = torch.randint(11, [1024], device="cuda")
    # random_tensor3 = torch.randint(10, [1024], device="cuda")
    # tensor4 = random_tensor1 + random_tensor2 +random_tensor3
    # return tensor4
    return random_tensor1, random_tensor2, random_tensor3

fn()
fn_opt = torch.compile(fn)
fn_opt()


# def fn2(tensor1, x):
#     tensor2 = torch.cos(tensor1) + x
#     return tensor1, tensor2

# t1 = torch.randn(1024, device="cuda")
# fn2_opt = torch.compile(fn2)
# x=2
# fn2_opt(t1, x)
# fn2_opt(t1, 3)

# def fn():
#     random_tensor1 = torch.randn(1024, device="cuda")
#     return random_tensor1.cos()

# fn_opt = torch.compile(fn)
# fn_opt()