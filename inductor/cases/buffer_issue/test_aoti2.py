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
    permute = None

    view_1 = torch.ops.aten.reshape.default(clone, [-1, 4])
    clone = None
    permute_1 = torch.ops.aten.permute.default(primals_1, [1, 0])
    primals_1 = None
    addmm = torch.ops.aten.addmm.default(primals_2, view_1, permute_1)
    primals_2 = None
    return addmm


original_output=forward(*example_inputs)
print('======eager mode======')
print(original_output)



# torchinductor part
import importlib.util
import sys




def dynamic_import(module_path, function_name):
    module_name = module_path.split("/")[-1].split(".")[0]
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return getattr(module, function_name)


function_name = "call"
call = dynamic_import(py_impl_path, function_name)

ref_inputs = list(ref_inputs)
# ref_inputs4 = copy.deepcopy(inputs)
full_py_output = call(ref_inputs)
py_output = full_py_output[0]
print("py output id  ",hex(id(py_output)))

# print(same(py_output, expeceted))
# profile_model(call, ref_inputs4, py=True, worker_name="torchinductor")


# aoti part
runner = torch._C._aoti.AOTIModelContainerRunnerCuda(aoti_impl_path, 1, device)  # type: ignore[assignment, call-arg]
import time
time.sleep(10)
def optimized(*args, **kwargs):
    call_spec = runner.get_call_spec()  # type: ignore[attr-defined]
    in_spec = pytree.treespec_loads(call_spec[0])
    out_spec = pytree.treespec_loads(call_spec[1])
    flat_inputs = pytree.tree_flatten((args, reorder_kwargs(kwargs, in_spec)))[0]
    flat_outputs = runner.run(flat_inputs)  # type: ignore[attr-defined]
    print("aoti output id", hex(id(flat_outputs[0])))
    return pytree.tree_unflatten(flat_outputs, out_spec)
aoti_output = optimized(*ref_inputs2)


# reproduce issue

# py_output = run_py_impl(ref_inputs)
# aoti_output = run_aoti(ref_inputs2)

# print output and compare
print("====aoti===")
print(aoti_output)
print("====py_output")
print(py_output)
print("====same or not")
print(same(aoti_output, py_output))

# print("=====compare input ====")
# print("refinputs")
# print(ref_inputs)
# print("refinputs2")
# print(ref_inputs2)
# print(same(ref_inputs, ref_inputs2))


# print("===== a workaround for the buffer issue=====")
# py_output = run_py_impl(ref_inputs3)
# # padding to fix the buffer issue
# empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda
# buf10 = empty_strided_cuda((727828, 128, 4), (512, 4, 1), torch.float32)
# aoti_output = run_aoti(ref_inputs4)
# # print output and compare
# print("====aoti===")
# print(aoti_output)
# print("====py_output")
# print(py_output)
# print("====same or not")
# print(same(aoti_output, py_output))








# profile_model(optimized, ref_inputs3, py=False, worker_name="aoti")
