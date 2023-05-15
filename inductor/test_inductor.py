
import torch

# Step 1: Generate 2 random tensors
tensor1 = torch.rand((3, 3))
tensor2 = torch.rand((3, 3))

print("Tensor 1:")
print(tensor1)
print("\nTensor 2:")
print(tensor2)

def fn(x, y):
    # Step 2: Compute the cosine of the tensors
    cos_tensor1 = torch.cos(tensor1)
    cos_tensor2 = torch.cos(tensor2)
    return cos_tensor1, cos_tensor2
# print("\nCosine of Tensor 1:")
# print(cos_tensor1)
# print("\nCosine of Tensor 2:")
# print(cos_tensor2)

fn(tensor1, tensor2)