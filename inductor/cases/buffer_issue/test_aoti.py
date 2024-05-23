import torch
import torch.utils._pytree as pytree
from torch.export._tree_utils import reorder_kwargs
import torch.profiler as profiler
import copy
from torch._dynamo.testing import rand_strided, same

py_impl_path="/home/yhao/p9/ml_scripts/inductor/cases/buffer_issue/py_impl.py"
aoti_impl_path="/home/yhao/p9/ml_scripts/inductor/cases/buffer_issue/aoti_impl.so"


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
ref_inputs3 = copy.deepcopy(example_inputs)
# model = Model()
# ref_model=copy.deepcopy(model)
# expeceted = ref_model(*ref_inputs)




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

ref_inputs2 = list(ref_inputs2)
ref_inputs4 = copy.deepcopy(ref_inputs2)
full_py_output = call(ref_inputs2)
py_output = full_py_output[0]

# print(same(py_output, expeceted))
# profile_model(call, ref_inputs4, py=True, worker_name="torchinductor")

empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda

buf10 = empty_strided_cuda((727828, 128, 4), (512, 4, 1), torch.float32)


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
    return pytree.tree_unflatten(flat_outputs, out_spec)
aoti_output = optimized(*ref_inputs3)










# print output and compare
print("====aoti===")
print(aoti_output)
print("====py_output")
print(py_output)
print(same(aoti_output, py_output))

# profile_model(optimized, ref_inputs3, py=False, worker_name="aoti")
