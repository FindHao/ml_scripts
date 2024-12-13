# AOT ID: ['0_forward']
"""
This file is a optimized inductor code for embedding.
"""
from ctypes import c_void_p, c_long, c_int
import torch
import math
import random
import os
import tempfile
from math import inf, nan
from torch._inductor.hooks import run_intermediate_hooks
from torch._inductor.utils import maybe_profile
from torch._inductor.codegen.memory_planning import _align as align
from torch import device, empty_strided
from torch._inductor.async_compile import AsyncCompile
from torch._inductor.select_algorithm import extern_kernels
from torch._inductor.codegen.multi_kernel import MultiKernelCall
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import (
    grid,
    split_scan_grid,
    grid_combo_kernels,
    start_graph,
    end_graph,
    cooperative_reduction_grid,
)
from torch._C import _cuda_getCurrentRawStream as get_raw_stream

aten = torch.ops.aten
inductor_ops = torch.ops.inductor
_quantized = torch.ops._quantized
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cpu = torch._C._dynamo.guards._empty_strided_cpu
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda
empty_strided_xpu = torch._C._dynamo.guards._empty_strided_xpu
reinterpret_tensor = torch._C._dynamo.guards._reinterpret_tensor
alloc_from_pool = torch.ops.inductor._alloc_from_pool

empty_strided_p2p = torch._C._distributed_c10d._SymmetricMemory.empty_strided_p2p


# kernel path: /tmp/torchinductor_yhao24/27/c27is6lglieveyt4lhwy7wdswiotq3jl2x4ovyonypjc5ysukvsw.py
# Topologically Sorted Source Nodes: [embedding], Original ATen: [aten.embedding]
# Source node to ATen node mapping:
#   embedding => embedding
# Graph fragment:
#   %embedding : [num_users=1] = call_function[target=torch.ops.aten.embedding.default](args = (%primals_1, %primals_2), kwargs = {})
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import (
    AutotuneHint,
    ReductionHint,
    TileHint,
    DeviceProperties,
)

triton_helpers.set_driver_to_gpu()


@triton.jit
def triton_poi_fused_embedding_0(
    in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK: tl.constexpr
):
    xnumel = 67108864
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x1_scalar = tl.program_id(0) * XBLOCK // 4096
    x0 = xindex % 4096
    x2 = xindex
    tmp0_scalar = tl.load(in_ptr0 + (x1_scalar), None, eviction_policy="evict_last")
    tmp0 = tmp0_scalar
    tmp1 = tl.full([XBLOCK], 8192, tl.int32)
    tmp2 = tmp0 + tmp1
    tmp3 = tmp0 < 0
    tmp4 = tl.where(tmp3, tmp2, tmp0)
    tl.device_assert(
        (0 <= tmp4) & (tmp4 < 8192), "index out of bounds: 0 <= tmp4 < 8192"
    )
    tmp6 = tl.load(in_ptr1 + (x0 + 4096 * tmp4), None)
    tl.store(out_ptr0 + (x2), tmp6, None)


def call(args):
    primals_1, primals_2 = args
    args.clear()
    assert_size_stride(primals_1, (8192, 4096), (4096, 1))
    assert_size_stride(primals_2, (8, 2048), (2048, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((8, 2048, 4096), (8388608, 4096, 1), torch.float32)
        # Topologically Sorted Source Nodes: [embedding], Original ATen: [aten.embedding]
        stream0 = get_raw_stream(0)
        triton_poi_fused_embedding_0[grid(67108864)](
            primals_2, primals_1, buf0, 67108864, XBLOCK=1024, num_warps=4, num_stages=1
        )
        del primals_1
    return (
        buf0,
        primals_2,
    )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance

    primals_1 = rand_strided(
        (8192, 4096), (4096, 1), device="cuda:0", dtype=torch.float32
    )
    # primals_2 = rand_strided((8, 2048), (2048, 1), device='cuda:0', dtype=torch.int64)
    primals_2 = torch.randint(8192, (8, 2048), device="cuda:0", dtype=torch.int64)
    fn = lambda: call([primals_1, primals_2])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main

    compiled_module_main("None", benchmark_compiled_module)
