import torch
import torch._dynamo as dynamo

# @torch.compile
# def change_view(tensor):
#     reshaped_tensor = tensor.view(2, 4)
#     return reshaped_tensor

# @torch.compile
# def compute_on_view(tensor):
#     reshaped_tensor = change_view(tensor)
#     sum_of_elements = torch.sum(reshaped_tensor)
#     return sum_of_elements


# tensor = torch.arange(8, device='cuda')
# tensor2 = tensor.clone()

# print(change_view(tensor))
# print(compute_on_view(tensor2))

@torch.compile
def change_view(tensor):
    return tensor.view(2, 4)

tensor = torch.arange(8, device='cuda')
tensor2 = tensor.clone()
change_view(tensor)
