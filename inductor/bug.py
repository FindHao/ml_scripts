import torch

def fn(v3):
    # v3: ()
    v7 = torch.neg(v3) # v7: () 250
    v4 = torch.lt(v7, torch.tensor(4, dtype=torch.uint8)) # v4: () False
    return v7, v4


v3 = torch.tensor(6, dtype=torch.uint8)
print(fn(v3)) # tensor(False)

compiled = torch.compile(fn)
print(compiled(v3))



def forward():
    x = torch.tensor(5, device='cuda', dtype=torch.uint8)
    y = torch.neg(x)
    return x < y
print(forward())
fn_compiled = torch.compile(forward)
print(fn_compiled())