import torch

# Setting a seed for reproducibility
torch.manual_seed(0)

def fn():
    random_tensor1 = torch.randint(10, [1024], device="cuda"),
    random_tensor2 = torch.randint(11, [1024], device="cuda"),
    return random_tensor1, random_tensor2

t1, t2 = fn()
fn_opt = torch.compile(fn)
fn_opt()
