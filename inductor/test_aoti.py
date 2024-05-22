import torch
import torch.utils._pytree as pytree
from torch.export._tree_utils import reorder_kwargs
import torch.profiler as profiler
import copy
from torch._dynamo.testing import rand_strided, same

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, primals_1, primals_2, primals_5):
        view = torch.ops.aten.reshape.default(primals_5, [-1, 4, 128])
        primals_5 = None
        permute = torch.ops.aten.permute.default(view, [0, 2, 1])
        clone = torch.ops.aten.clone.default(
            permute, memory_format=torch.contiguous_format
        )
        permute = None
        view_1 = torch.ops.aten.reshape.default(clone, [-1, 4])
        clone = None
        permute_1 = torch.ops.aten.permute.default(primals_1, [1, 0])
        primals_1 = None
        addmm = torch.ops.aten.addmm.default(primals_2, view_1, permute_1)
        primals_2 = None
        return addmm

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
model = Model()
ref_model=copy.deepcopy(model)
expeceted = ref_model(*ref_inputs)

# # profile
# activity_groups = []
# result_summary = []
# device_to_activity = {
#     "cuda": profiler.ProfilerActivity.CUDA,
#     "cpu": profiler.ProfilerActivity.CPU,
# }
# nwarmup=1
# with profiler.profile(
#     schedule=profiler.schedule(wait=0, warmup=nwarmup, active=1, repeat=1),
#     activities=activity_groups,
#     record_shapes=True,
#     # profile_memory=False,
#     # with_stack=True,
#     # with_flops=False,
#     on_trace_ready=(profiler.tensorboard_trace_handler("/tmp/yhao/profile/")),
# ) as prof:
#     for i in range(nwarmup + 1):
#         ref_model(*ref_inputs)
#         prof.step()

# torchinductor
import importlib.util
import sys

def dynamic_import(module_path, function_name):
    module_name = module_path.split("/")[-1].split(".")[0]
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return getattr(module, function_name)

py_path = "/tmp/torchinductor_yhao/ry/cry6wn35qurxhwrzmfhnkmi3onn6ek6wh2x7l6z3oyldbvqcgjlf.py"
function_name = "call"
call = dynamic_import(py_path, function_name)

py_output = call(list(ref_inputs2))[0]
print(same(py_output, expeceted))


# aoti part
so_path="/tmp/torchinductor_yhao/cfiwgqmhnw54zqatavj57riugdirk75nrpqhdgyyaa4dlx62na34/cxurg4jagc5ladi6scephaotyisxw7dooonxt62ckzujcwjhxxhu.so"
runner = torch._C._aoti.AOTIModelContainerRunnerCuda(so_path, 1, device)  # type: ignore[assignment, call-arg]


def optimized(*args, **kwargs):
    call_spec = runner.get_call_spec()  # type: ignore[attr-defined]
    in_spec = pytree.treespec_loads(call_spec[0])
    out_spec = pytree.treespec_loads(call_spec[1])
    flat_inputs = pytree.tree_flatten((args, reorder_kwargs(kwargs, in_spec)))[0]
    flat_outputs = runner.run(flat_inputs)  # type: ignore[attr-defined]
    return pytree.tree_unflatten(flat_outputs, out_spec)

aoti_output = optimized(*example_inputs)
print(same(aoti_output, expeceted))
