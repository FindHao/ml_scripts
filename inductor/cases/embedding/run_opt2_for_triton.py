import torch
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import (
    grid,
)

aten = torch.ops.aten
inductor_ops = torch.ops.inductor
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_embedding_0(
    in_ptr0,
    in_ptr1,
    out_ptr0,
    ynumel,
    xnumel,
    YBLOCK: tl.constexpr,
    XBLOCK: tl.constexpr,
):
    ynumel = 16384
    xnumel = 4096
    yoffset = tl.program_id(0) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = tl.full([XBLOCK, YBLOCK], True, tl.int1)
    xoffset = tl.program_id(1) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, YBLOCK], True, tl.int1)
    y0 = yindex
    x1 = xindex
    tmp0 = tl.load(in_ptr0 + (y0), None, eviction_policy="evict_last")
    tmp1 = tl.full([XBLOCK, YBLOCK], 8192, tl.int32)
    tmp2 = tmp0 + tmp1
    tmp3 = tmp0 < 0
    tmp4 = tl.where(tmp3, tmp2, tmp0)
    tl.device_assert(
        (0 <= tmp4) & (tmp4 < 8192), "index out of bounds: 0 <= tmp4 < 8192"
    )
    tmp6 = tl.load(in_ptr1 + (x1 + 4096 * tmp4), None)
    tl.store(out_ptr0 + (x1 + 4096 * y0), tmp6, None)


def call(
    args,
    XBLOCK: int = 128,
    YBLOCK: int = 128,
    num_warps: int = 8,
):
    primals_1, primals_2 = args
    args.clear()
    assert_size_stride(primals_1, (8192, 4096), (4096, 1))
    assert_size_stride(primals_2, (8, 2048), (2048, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((8, 2048, 4096), (8388608, 4096, 1), torch.float32)
        # Topologically Sorted Source Nodes: [embedding], Original ATen: [aten.embedding]
        triton_poi_fused_embedding_0[grid(4096, 16384)](
            primals_2,
            primals_1,
            buf0,
            4096,
            16384,
            XBLOCK=XBLOCK,
            YBLOCK=YBLOCK,
            num_warps=num_warps,
            num_stages=1,
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
    primals_2 = torch.randint(8192, (8, 2048), device="cuda:0", dtype=torch.int64)

    XBLOCK = 128
    YBLOCK = 128
    nwarps = 4

    fn = lambda: call(
        [primals_1, primals_2],
        XBLOCK=XBLOCK,
        YBLOCK=YBLOCK,
        num_warps=nwarps,
    )
    ares = print_performance(fn, times=times, repeat=repeat)
    print(f"ares={ares}")
    return ares


if __name__ == "__main__":
    # --ncu
    # from torch._inductor.wrapper_benchmark import compiled_module_main
    # compiled_module_main("None", benchmark_compiled_module)
    # pure run
    benchmark_compiled_module()
