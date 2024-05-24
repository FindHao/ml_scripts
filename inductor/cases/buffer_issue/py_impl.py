

from ctypes import c_void_p, c_long
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
from torch._inductor.codecache import AsyncCompile
from torch._inductor.select_algorithm import extern_kernels
from torch._inductor.codegen.multi_kernel import MultiKernelCall

aten = torch.ops.aten
inductor_ops = torch.ops.inductor
_quantized = torch.ops._quantized
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cpu = torch._C._dynamo.guards._empty_strided_cpu
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda
alloc_from_pool = torch.ops.inductor._alloc_from_pool
reinterpret_tensor = torch.ops.inductor._reinterpret_tensor
async_compile = AsyncCompile()


# kernel path: /tmp/torchinductor_yhao/xz/cxzwl5p3b23mlry6mtiw6zdv3wuitfq74tzd3oa2ezqnddyumzya.py
# Source Nodes: [clone], Original ATen: [aten.clone]
# clone => clone
triton_poi_fused_clone_0 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[134217728, 4], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_0', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'ed7dfdc092a430929eaf844af6fc1617b738d894bfa950e27f89799327fe9f10', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': False, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': True},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 93161984
    xnumel = 4
    yoffset = tl.program_id(1) * (tl.program_id(2) + 1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 128
    y1 = (yindex // 128)
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + (128*x2) + (512*y1)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x2 + (4*y3)), tmp0, xmask & ymask)
''', device_str='cuda')

import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import grid, split_scan_grid, start_graph, end_graph
from torch._C import _cuda_getCurrentRawStream as get_raw_stream


async_compile.wait(globals())
del async_compile

import time
def call(args):
    arg0_1, arg1_1, arg2_1 = args
    # args.clear()
    assert_size_stride(arg0_1, (2, 4), (4, 1))
    assert_size_stride(arg1_1, (2, ), (1, ))
    assert_size_stride(arg2_1, (727828, 512), (512, 1))

    for kernel in globals().values():
        if isinstance(kernel, torch._inductor.runtime.triton_heuristics.CachingAutotuner):
            kernel.cuda_kernel_saved = False
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((727828, 128, 4), (512, 4, 1), torch.float32)
        # Source Nodes: [clone], Original ATen: [aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_0.run(arg2_1, buf0, 93161984, 4, grid=grid(93161984, 4), stream=stream0)
        # torch.cuda.synchronize()

        # time.sleep(5)


    # #     @Yueming: the belowing is generated from the unit test, but we don't need it for our testing.
    #     del arg2_1
    #     buf1 = empty_strided_cuda((93161984, 2), (2, 1), torch.float32)
    #     # Source Nodes: [addmm], Original ATen: [aten.addmm]
    #     extern_kernels.addmm(arg1_1, reinterpret_tensor(buf0, (93161984, 4), (4, 1), 0), reinterpret_tensor(arg0_1, (4, 2), (1, 4), 0), alpha=1, beta=1, out=buf1)
    #     del arg0_1
    #     del arg1_1
    #     # del buf0
    # for kernel in globals().values():
    #     if isinstance(kernel, torch._inductor.runtime.triton_heuristics.CachingAutotuner):
    #         if not kernel.cuda_kernel_saved:
    #             if len(kernel.launchers) == 0:
    #                 kernel.precompile()
    #             kernel.save_cuda_kernel(
    #                 grid=(0, 0, 0),   # use dummy grid
    #                 stream="stream",  # use dummy stream
    #                 launcher=kernel.launchers[0],
    #             )
    # return (buf1, )


    return (buf0, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((2, 4), (4, 1), device='cuda:0', dtype=torch.float32)
    arg1_1 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg2_1 = rand_strided((727828, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([arg0_1, arg1_1, arg2_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
