import torch

print(torch.randint(low=5, high=6, size=(1,), device='cuda', dtype=torch.uint8))

print("testing case 1")
def fn():
    # this bug only occurs on constant tensors
    x = torch.tensor(5, device='cuda', dtype=torch.uint8)
    # x = torch.randint(low=5, high=6, size=(1,), device='cuda', dtype=torch.uint8)
    y = torch.neg(x)
    return x < y
# supposed to be False
print(fn())
fn_compiled = torch.compile(fn)
print(fn_compiled())


# print("\ntesting case 2")
# def fn2():
#     x = torch.tensor(0, device='cuda', dtype=torch.uint8)
#     y = x - 1
#     return x < y
# # supposed to be False
# print(fn2())
# fn2_compiled = torch.compile(fn2)
# print(fn2_compiled())

# print("\ntesting case 3")
# def fn3():
#     x = torch.tensor(255, device='cuda', dtype=torch.uint8)
#     y = x + 1
#     return x < y

# # supposed to be True
# print(fn3())
# fn3_compiled = torch.compile(fn3)
# print(fn3_compiled())


