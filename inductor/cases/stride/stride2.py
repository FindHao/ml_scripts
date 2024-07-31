import torch
import triton
import triton.language as tl

@triton.jit
def add_kernel(
    in_ptr0,
    in_ptr1,
    out_ptr,
    n_elements,
    BLOCK_SIZE: "tl.constexpr",
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(in_ptr0 + offsets, mask=mask)
    y = tl.load(in_ptr1 + offsets, mask=mask)
    output = x + y
    tl.store(out_ptr + offsets, output, mask=mask)

x = torch.randn(2, 2, device="cuda")
other = torch.randn(2, 2, device="cuda")

def f(x, other):
    y = x.t().contiguous().t()
    z = y.sin().t()
    grid = (z.numel(),)
    out = torch.empty_like(other)
    add_kernel[grid](z, other, out, z.numel(), BLOCK_SIZE=16)
    return out

f_compile = torch.compile(f)

out = f(x, other)
out_compile = f_compile(x, other)
print(out)
print(out_compile)
assert torch.allclose(out_compile, out)
