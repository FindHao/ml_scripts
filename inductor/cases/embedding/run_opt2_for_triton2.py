import torch
import triton
import triton.language as tl

aten = torch.ops.aten

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
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = torch.empty_strided((8, 2048, 4096), (8388608, 4096, 1), dtype=torch.float32, device="cuda")

        # Topologically Sorted Source Nodes: [embedding], Original ATen: [aten.embedding]
        triton_poi_fused_embedding_0[(128, 32, 1)](
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

def benchmark_compiled_module(times=10, repeat=10, XBLOCK=128, YBLOCK=128, num_warps=8):
    size = (8192, 4096)
    stride = (4096, 1)
    needed_size = (
        sum((shape - 1) * stride for shape, stride in zip(size, stride))
        + 1
    )
    primals_1 = torch.as_strided(torch.randn(needed_size, device="cuda:0", dtype=torch.float32
    ), size, stride)
    primals_2 = torch.randint(8192, (8, 2048), device="cuda:0", dtype=torch.int64)

    print(call([primals_1, primals_2], XBLOCK=XBLOCK, YBLOCK=YBLOCK))


if __name__ == "__main__":
    # --ncu
    # from torch._inductor.wrapper_benchmark import compiled_module_main
    # compiled_module_main("None", benchmark_compiled_module)
    # pure run
    import argparse

    parser = argparse.ArgumentParser(description="Benchmark Compiled Module")
    parser.add_argument("--times", type=int, default=10, help="Number of times to run the benchmark")
    parser.add_argument("--repeat", type=int, default=10, help="Number of repeats for the benchmark")
    parser.add_argument("--XBLOCK", type=int, default=128, help="XBLOCK size")
    parser.add_argument("--YBLOCK", type=int, default=128, help="YBLOCK size")
    parser.add_argument("--num_warps", type=int, default=8, help="Number of warps")

    args = parser.parse_args()

    benchmark_compiled_module(
        times=args.times,
        repeat=args.repeat,
        XBLOCK=args.XBLOCK,
        YBLOCK=args.YBLOCK,
    )
