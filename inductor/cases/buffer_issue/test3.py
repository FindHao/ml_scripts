import torch
import torch.utils._pytree as pytree
from torch.export._tree_utils import reorder_kwargs
import torch.profiler as profiler
import copy
from torch._dynamo.testing import rand_strided, same

# get current file path
import os
import sys
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
py_impl_path = os.path.join(currentdir, "py_impl.py")
aoti_impl_path = os.path.join(currentdir, "aoti_impl.so")


device="cuda"
s0 = 727828
s1 = 512
example_inputs = (
    torch.rand(2, 4, device=device),
    torch.rand(2, device=device),
    torch.rand(s0, s1, device=device),
)

ref_inputs = copy.deepcopy(example_inputs)
ref_inputs2 = copy.deepcopy(example_inputs)
# ref_inputs3 = copy.deepcopy(example_inputs)
# ref_inputs4 = copy.deepcopy(example_inputs)
# model = Model()
# ref_model=copy.deepcopy(model)
# expeceted = ref_model(*ref_inputs)
import time

def forward(primals_1, primals_2, primals_5):
    view = torch.ops.aten.reshape.default(primals_5, [-1, 4, 128])
    primals_5 = None
    permute = torch.ops.aten.permute.default(view, [0, 2, 1])
    clone = torch.ops.aten.clone.default(
        permute, memory_format=torch.contiguous_format
    )
    return clone
    # permute = None

    # view_1 = torch.ops.aten.reshape.default(clone, [-1, 4])
    # clone = None
    # permute_1 = torch.ops.aten.permute.default(primals_1, [1, 0])
    # primals_1 = None
    # addmm = torch.ops.aten.addmm.default(primals_2, view_1, permute_1)
    # primals_2 = None
    # return addmm


compiled_1 = torch.compile(forward)

print(compiled_1(*ref_inputs))

print("====")
print(forward(*ref_inputs2))
