
import triton
import triton.language as tl
# from triton.compiler.compiler import AttrsDescriptor

# from torch._inductor.runtime import triton_helpers, triton_heuristics
# from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
# from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

# @triton_heuristics.pointwise(
#     size_hints=[67108864, 4], tile_hint=TileHint.SQUARE,
#     filename=__file__,
#     triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]},
#     inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_0', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'ed7dfdc092a430929eaf844af6fc1617b738d894bfa950e27f89799327fe9f10', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': False, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
#     min_elem_per_thread=0
# )
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 67108864
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
    tmp0 = tl.load(in_ptr0 + (y0 + (128*x2) + (512*y1)), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x2 + (4*y3)), tmp0, xmask)


def run_triton(x, y):
    x = torch.randn((67108864, 4), device="cuda")
    y = torch.empty((32768, 4), dtype=torch.float32, device='cuda')
    triton_(x, y)
    return y
    arg0_1 = ones_tensor.as_strided((524288, 512), (512, 1))
    buf0 = empty_strided_cuda((524288, 128, 4), (512, 4, 1), torch.float32)
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        # Source Nodes: [clone], Original ATen: [aten.clone]
        stream0 = get_raw_stream(0)
        # triton_poi_fused_clone_0.run(arg0_1, buf0, 67108864, 4, grid=grid(67108864, 4), stream=stream0)
        triton_(arg0_1, buf0, 67108864, 4, YBLOCK=67108864, XBLOCK=4)
        del arg0_1
    return (buf0, )
