
import torch

# Step 1: Generate 2 random tensors
tensor1 = torch.rand((3, 3), device='cuda')
tensor2 = torch.rand((3, 3), device='cuda')

aflag = True

def fn(x, y):
    # Step 2: Compute the cosine of the tensors
    cos_tensor1 = torch.cos(tensor1)
    cos_tensor2 = torch.cos(tensor2)
    if aflag:
        cos_tensor1 = torch.cos(cos_tensor1)
    else:
        cos_tensor2 = torch.cos(cos_tensor2)
    return cos_tensor1, cos_tensor2
# print("\nCosine of Tensor 1:")
# print(cos_tensor1)
# print("\nCosine of Tensor 2:")
# print(cos_tensor2)

print(torch.compile(fn)(tensor1, tensor2))
aflag = False
print(torch.compile(fn)(tensor1, tensor2))
