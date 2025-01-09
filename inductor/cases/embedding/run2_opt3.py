# tried to change loop order without XYBLOCK
# AOT ID: ['0_forward']
"""
This file is a original inductor code for embedding.
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
    x1 = xindex // 16384
    x0 = xindex % 16384
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None, eviction_policy="evict_last")
    tmp1 = tl.full([XBLOCK], 8192, tl.int32)
    tmp2 = tmp0 + tmp1
    tmp3 = tmp0 < 0
    tmp4 = tl.where(tmp3, tmp2, tmp0)
    tl.device_assert(
        (0 <= tmp4) & (tmp4 < 8192), "index out of bounds: 0 <= tmp4 < 8192"
    )
    tmp6 = tl.load(in_ptr1 + (x1 + 16384 * tmp4), None)
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


import torch
import triton
import triton.language as tl


@triton.jit
def embedding_forward_kernel(
    embeddings_ptr,
    indices_ptr,
    output_ptr,
    n_elements,
    embedding_dim: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    start_m = pid_m * BLOCK_SIZE_M
    start_n = pid_n * BLOCK_SIZE_N
    offsets_m = start_m + tl.arange(0, BLOCK_SIZE_M)
    mask_m = offsets_m < n_elements
    indices = tl.load(indices_ptr + offsets_m, mask=mask_m, other=0)
    offsets_n = start_n + tl.arange(0, BLOCK_SIZE_N)
    mask_n = offsets_n < embedding_dim

    embedding_offsets = indices[:, None] * embedding_dim + offsets_n[None, :]
    embeddings = tl.load(
        embeddings_ptr + embedding_offsets,
        mask=mask_m[:, None] & mask_n[None, :],
        other=0.0,
    )

    output_offsets = offsets_m[:, None] * embedding_dim + offsets_n[None, :]
    tl.store(
        output_ptr + output_offsets, embeddings, mask=mask_m[:, None] & mask_n[None, :]
    )


class LigerEmbeddingFunction(torch.autograd.Function):
    @staticmethod
    def forward(embeddings: torch.Tensor, indices: torch.Tensor):
        ori_shape = indices.shape
        indices = indices.view(-1)
        output = torch.empty(
            indices.shape[0],
            embeddings.shape[1],
            device=indices.device,
            dtype=embeddings.dtype,
        )

        n_elements = indices.numel()
        embedding_dim = embeddings.shape[1]

        BLOCK_SIZE_M = triton.next_power_of_2(min(128, embedding_dim))
        BLOCK_SIZE_N = triton.next_power_of_2(min(128, embedding_dim))
        grid = (
            triton.cdiv(n_elements, BLOCK_SIZE_M),
            triton.cdiv(embedding_dim, BLOCK_SIZE_N),
        )

        embedding_forward_kernel[grid](
            embeddings,
            indices,
            output,
            n_elements,
            embedding_dim=embedding_dim,
            BLOCK_SIZE_M=BLOCK_SIZE_M,
            BLOCK_SIZE_N=BLOCK_SIZE_N,
        )

        return output.view(*ori_shape, -1)

# def benchmark_compiled_module(times=10, repeat=10):
#     from torch._dynamo.testing import rand_strided
#     from torch._inductor.utils import print_performance

#     primals_1 = rand_strided(
#         (8192, 4096), (4096, 1), device="cuda:0", dtype=torch.float32
#     )
#     # primals_2 = rand_strided((8, 2048), (2048, 1), device='cuda:0', dtype=torch.int64)
#     primals_2 = torch.randint(8192, (8, 2048), device="cuda:0", dtype=torch.int64)
#     fn = lambda: call([primals_1, primals_2])
#     return print_performance(fn, times=times, repeat=repeat)


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance

    primals_1 = rand_strided(
        (8192, 4096), (4096, 1), device="cuda:0", dtype=torch.float32
    )
    primals_2 = torch.randint(8192, (8, 2048), device="cuda:0", dtype=torch.int64)

    XBLOCK = 128
    YBLOCK = 128
    nwarps = 4

    fn = lambda: call(
        [primals_1, primals_2],
        # XBLOCK=XBLOCK,
        # YBLOCK=YBLOCK,
        # num_warps=nwarps,
    )
    ares = print_performance(fn, times=times, repeat=repeat)
    print(f"=====XBLOCK={XBLOCK}, YBLOCK={YBLOCK}, nwarps={nwarps}, ares={ares}")

    fn = lambda: LigerEmbeddingFunction.forward(primals_1, primals_2)
    ref_result = print_performance(fn, times=times, repeat=repeat)
    print(f"=====ref: XBLOCK=128, YBLOCK=128, nwarps=4, ares={ref_result}")
    # Check correctness
    with torch.no_grad():
        out1 = call([primals_1, primals_2])[0]
        out2 = LigerEmbeddingFunction.forward(primals_1, primals_2)
        
        # Compare outputs
        max_diff = torch.max(torch.abs(out1 - out2))
        print(f"\nMax absolute difference: {max_diff}")
        
        # Check if outputs are close enough
        rtol = 1e-5
        atol = 1e-5
        is_close = torch.allclose(out1, out2, rtol=rtol, atol=atol)
        print(f"Outputs match within tolerance (rtol={rtol}, atol={atol}): {is_close}")
        
        if not is_close:
            # Print more detailed error statistics if outputs don't match
            rel_diff = torch.abs(out1 - out2) / (torch.abs(out2) + atol)
            print(f"Max relative difference: {torch.max(rel_diff)}")
            print(f"Mean absolute difference: {torch.mean(torch.abs(out1 - out2))}")
            print(f"Number of differing elements: {torch.sum(~torch.isclose(out1, out2, rtol=rtol, atol=atol))}")
    return ref_result
    return ref_result


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main

    compiled_module_main("None", benchmark_compiled_module)
