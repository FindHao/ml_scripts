# AOT ID: ['0_forward']
import math
import os
import random
import tempfile
from ctypes import c_int, c_long, c_void_p
from math import inf, nan

import torch
import triton
import triton.language as tl
from torch import device, empty_strided
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
from torch._inductor.async_compile import AsyncCompile
from torch._inductor.codegen.memory_planning import _align as align
from torch._inductor.codegen.multi_kernel import MultiKernelCall
from torch._inductor.hooks import run_intermediate_hooks
from torch._inductor.runtime.triton_heuristics import (
    cooperative_reduction_grid,
    end_graph,
    grid,
    grid_combo_kernels,
    split_scan_grid,
    start_graph,
)
from torch._inductor.select_algorithm import extern_kernels
from torch._inductor.utils import maybe_profile

aten = torch.ops.aten
inductor_ops = torch.ops.inductor
_quantized = torch.ops._quantized
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cpu = torch._C._dynamo.guards._empty_strided_cpu
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda
empty_strided_xpu = torch._C._dynamo.guards._empty_strided_xpu
reinterpret_tensor = torch._C._dynamo.guards._reinterpret_tensor
alloc_from_pool = torch.ops.inductor._alloc_from_pool
async_compile = AsyncCompile()
empty_strided_p2p = torch._C._distributed_c10d._SymmetricMemory.empty_strided_p2p


# kernel path: /tmp/torchinductor_yhao/tmplr3_cwfy/dj/cdj4ucqslufamcxongvllsoyurgyzwb7j4s55jx7kyltmzs4dvlx.py
# Topologically Sorted Source Nodes: [pow_1, variance, add, rsqrt, hidden_states_1, mul_1], Original ATen: [aten.pow, aten.mean, aten.add, aten.rsqrt, aten.mul]
# Source node to ATen node mapping:
#   add => add
#   hidden_states_1 => mul
#   mul_1 => mul_1
#   pow_1 => pow_1
#   rsqrt => rsqrt
#   variance => mean
# Graph fragment:
#   %pow_1 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%primals_1, 2), kwargs = {})
#   %mean : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_1, [-1], True), kwargs = {})
#   %add : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mean, 1e-06), kwargs = {})
#   %rsqrt : [num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add,), kwargs = {})
#   %mul : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%primals_1, %rsqrt), kwargs = {})
#   %mul_1 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%primals_2, %mul), kwargs = {})
triton_red_fused_add_mean_mul_pow_rsqrt_0 = async_compile.triton(
    "triton_red_fused_add_mean_mul_pow_rsqrt_0",
    """
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints=[2048, 32768],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_mean_mul_pow_rsqrt_0', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 1, 'backend_hash': '1D985D9C0CD26A9BB5BAD8A051C971598DEF7D09C84A1EB5695128BF97C731E4', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False}
)
@triton.jit
def triton_red_fused_add_mean_mul_pow_rsqrt_0(in_out_ptr0, in_ptr0, in_ptr1, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 2048
    rnumel = 32768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp3 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_ptr0 + (r1 + (32768*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tmp0 * tmp0
        tmp2 = tl.broadcast_to(tmp1, [XBLOCK, RBLOCK])
        tmp4 = _tmp3 + tmp2
        _tmp3 = tl.where(rmask & xmask, tmp4, _tmp3)
    tmp3 = tl.sum(_tmp3, 1)[:, None]
    tmp5 = 32768.0
    tmp6 = tmp3 / tmp5
    tmp7 = 1e-06
    tmp8 = tmp6 + tmp7
    tmp9 = libdevice.rsqrt(tmp8)
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp9, xmask)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp10 = tl.load(in_ptr1 + (r1), rmask, eviction_policy='evict_last', other=0.0)
        tmp11 = tl.load(in_ptr0 + (r1 + (32768*x0)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp12 = tmp11 * tmp9
        tmp13 = tmp10 * tmp12
        tl.store(out_ptr0 + (r1 + (32768*x0)), tmp13, rmask & xmask)
""",
    device_str="cuda",
)


async_compile.wait(globals())
del async_compile


def call(args):
    primals_1, primals_2 = args
    args.clear()
    assert_size_stride(primals_1, (2048, 32768), (32768, 1))
    assert_size_stride(primals_2, (32768,), (1,))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((2048, 1), (1, 2048), torch.float32)
        buf1 = reinterpret_tensor(buf0, (2048, 1), (1, 1), 0)
        del buf0  # reuse
        buf2 = empty_strided_cuda((2048, 32768), (32768, 1), torch.float32)
        # Topologically Sorted Source Nodes: [pow_1, variance, add, rsqrt, hidden_states_1, mul_1], Original ATen: [aten.pow, aten.mean, aten.add, aten.rsqrt, aten.mul]
        stream0 = get_raw_stream(0)
        triton_red_fused_add_mean_mul_pow_rsqrt_0.run(
            buf1,
            primals_1,
            primals_2,
            buf2,
            2048,
            32768,
            grid=grid(2048),
            stream=stream0,
        )
        del primals_2
    return (
        buf2,
        primals_1,
        buf1,
    )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance

    primals_1 = rand_strided(
        (2048, 32768), (32768, 1), device="cuda:0", dtype=torch.float32
    )
    primals_2 = rand_strided((32768,), (1,), device="cuda:0", dtype=torch.float32)
    fn = lambda: call([primals_1, primals_2])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main

    compiled_module_main("None", benchmark_compiled_module)
