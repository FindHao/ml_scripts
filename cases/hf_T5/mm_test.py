from ctypes import c_void_p, c_long
import torch
import math
import random
import os
import tempfile
from math import inf, nan
from torch._inductor.hooks import run_intermediate_hooks
from torch._inductor.utils import maybe_profile

from torch import empty_strided, as_strided, device
from torch._inductor.codecache import AsyncCompile
from torch._inductor.select_algorithm import extern_kernels

aten = torch.ops.aten
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
async_compile = AsyncCompile()



import triton
import triton.language as tl
from torch._inductor.triton_heuristics import grid, start_graph, end_graph
from torch._C import _cuda_getCurrentRawStream as get_cuda_stream

async_compile.wait(globals())
del async_compile
stream1_raw = torch.cuda.Stream()
stream1 = stream1_raw.cuda_stream
stream0_raw = torch.cuda.default_stream()
stream0 = get_cuda_stream(0)

def call(args):
    with torch.cuda._DeviceGuard(0):
        arg33_1, arg34_1, arg75_1, buf17, buf20 = args
        buf18 = empty_strided((8192, 512), (512, 1), device='cuda', dtype=torch.bfloat16)
        extern_kernels.mm(as_strided(buf17, (8192, 512), (512, 1)), as_strided(arg75_1, (512, 512), (1, 512)), out=buf18)
        buf21 = empty_strided((8192, 512), (512, 1), device='cuda', dtype=torch.bfloat16)
        extern_kernels.mm(as_strided(buf20, (8192, 512), (512, 1)), as_strided(arg33_1, (512, 512), (1, 512)), out=buf21)
        del arg33_1
        buf22 = empty_strided((8192, 512), (512, 1), device='cuda', dtype=torch.bfloat16)
        extern_kernels.mm(as_strided(buf20, (8192, 512), (512, 1)), as_strided(arg34_1, (512, 512), (1, 512)), out=buf22)
        return (buf18, buf21, buf22)

def call_opt(args):
    with torch.cuda._DeviceGuard(0):
        arg33_1, arg34_1, arg75_1, buf17, buf20 = args
        torch.cuda.set_stream(stream1_raw)
        buf18 = empty_strided((8192, 512), (512, 1), device='cuda', dtype=torch.bfloat16)
        extern_kernels.mm(as_strided(buf17, (8192, 512), (512, 1)), as_strided(arg75_1, (512, 512), (1, 512)), out=buf18)
        torch.cuda.set_stream(stream0_raw)
        buf21 = empty_strided((8192, 512), (512, 1), device='cuda', dtype=torch.bfloat16)
        extern_kernels.mm(as_strided(buf20, (8192, 512), (512, 1)), as_strided(arg33_1, (512, 512), (1, 512)), out=buf21)
        del arg33_1
        buf22 = empty_strided((8192, 512), (512, 1), device='cuda', dtype=torch.bfloat16)
        extern_kernels.mm(as_strided(buf20, (8192, 512), (512, 1)), as_strided(arg34_1, (512, 512), (1, 512)), out=buf22)
        return (buf18, buf21, buf22)

if __name__ == '__main__':
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg33_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.bfloat16)
    arg34_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.bfloat16)
    arg75_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.bfloat16)
    arg33_1_clone = arg33_1.clone()
    arg34_1_clone = arg34_1.clone()
    arg75_1_clone = arg75_1.clone()
    buf17 = rand_strided((4, 2048, 512), (1048576, 512, 1), device='cuda:0', dtype=torch.bfloat16)
    buf20 = rand_strided((4, 2048, 512), (1048576, 512, 1), device='cuda:0', dtype=torch.bfloat16)
    buf17_clone = buf17.clone()
    buf20_clone = buf20.clone()
    results_origin = call((arg33_1, arg34_1, arg75_1, buf17, buf20))
    results = call_opt((arg33_1_clone, arg34_1_clone, arg75_1_clone, buf17_clone, buf20_clone))
    all_equal = True
    print("====compare results ====")
    args = "buf18, buf21, buf22".split(",")
    def nan_allclose(tensor1, tensor2):
        tensor1_nan_mask = torch.isnan(tensor1)
        tensor2_nan_mask = torch.isnan(tensor2)
        nan_mask_equal = torch.all(tensor1_nan_mask.eq(tensor2_nan_mask))
        tensor1_non_nan = torch.where(tensor1_nan_mask, torch.zeros_like(tensor1), tensor1)
        tensor2_non_nan = torch.where(tensor2_nan_mask, torch.zeros_like(tensor2), tensor2)
        non_nan_values_allclose = torch.allclose(tensor1_non_nan, tensor2_non_nan)
        return nan_mask_equal and non_nan_values_allclose

    for i in range(len(results)):
        if not nan_allclose(results[i], results_origin[i]):
            print("Error in output ", args[i])
            print("mutli_stream output:")
            print(results[i])
            print("origin output:")
            print(results_origin[i])
            print("max diff:")
            print(torch.max(torch.abs(results[i] - results_origin[i])))
            all_equal = False

    if all_equal:
        print("All outputs are equal")
