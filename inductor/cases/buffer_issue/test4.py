import torch
import torch.utils._pytree as pytree
from torch.export._tree_utils import reorder_kwargs
import torch.profiler as profiler
import copy
from torch._dynamo.testing import rand_strided, same
from torch._dynamo.testing import rand_strided, same

# get current file path
import os
import sys
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))


device="cuda"
# s0 = 727828
# y_grid=65536
s0=524288
# y_grid=65535
# s0=524280
s1 = 512

s0 = 727828

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

print("====torchinductor=====")
compiled_1 = torch.compile(forward)
actual = compiled_1(*ref_inputs)
print(actual)

print("====eager=====")
expeceted = forward(*ref_inputs2)
print(expeceted)
print("===same or not?===")
print(same(actual, expeceted))
