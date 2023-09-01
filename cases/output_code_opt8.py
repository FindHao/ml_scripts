# remove all other buffers
# based on opt6
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

# from torch._dynamo.testing import one_strided

aten = torch.ops.aten
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
async_compile = AsyncCompile()


# kernel path: /tmp/torchinductor_yhao/xe/cxejnwdak7bi5k2erwzd3jmm4ehloi7go3yrc4tkwjfx2kfqv4ex.py
# Original ATen: aten._to_copy, aten.add, aten.embedding, aten.mean, aten.mul, aten.pow, aten.rsqrt

# aten._to_copy => convert_element_type_45, convert_element_type_46
# aten.add => add_35
# aten.embedding => embedding_2
# aten.mean => mean_13
# aten.mul => mul_32, mul_33
# aten.pow => pow_14
# aten.rsqrt => rsqrt_13
triton_red_fused__to_copy_add_embedding_mean_mul_pow_rsqrt_0 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[8192, 512],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    meta={'signature': {0: '*i64', 1: '*bf16', 2: '*bf16', 3: '*bf16', 4: 'i32', 5: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 8192
    rnumel = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    _tmp5 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp1 = triton_helpers.promote_to_tensor(tmp0)
        tl.device_assert((0 <= tmp1) & (tmp1 < 32128), "index out of bounds: 0 <= tmp1 < 32128")
        tmp2 = tl.load(in_ptr1 + (r1 + (512*tmp0)), rmask, eviction_policy='evict_last', other=0).to(tl.float32)
        tmp3 = tmp2.to(tl.float32)
        tmp4 = tmp3 * tmp3
        tmp6 = _tmp5 + tmp4
        _tmp5 = tl.where(rmask, tmp6, _tmp5)
    tmp5 = tl.sum(_tmp5, 1)[:, None]
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp7 = tl.load(in_ptr2 + (r1), rmask, eviction_policy='evict_last', other=0).to(tl.float32)
        tmp8 = triton_helpers.promote_to_tensor(tmp0)
        tl.device_assert((0 <= tmp8) & (tmp8 < 32128), "index out of bounds: 0 <= tmp8 < 32128")
        tmp9 = tl.load(in_ptr1 + (r1 + (512*tmp0)), rmask, other=0).to(tl.float32)
        tmp10 = tmp9.to(tl.float32)
        tmp11 = 512.0
        tmp12 = tmp5 / tmp11
        tmp13 = 1e-06
        tmp14 = tmp12 + tmp13
        tmp15 = tl.math.rsqrt(tmp14)
        tmp16 = tmp10 * tmp15
        tmp17 = tmp16.to(tl.float32)
        tmp18 = tmp7 * tmp17
        tl.store(out_ptr1 + (r1 + (512*x0)), tmp18, rmask)
''')


# kernel path: /tmp/torchinductor_yhao/v2/cv2deel4qswmf45ofu4tkutq5qv6pk3vcwyrwoi2htdtz5mumt3q.py
# Original ATen: aten.clone

# aten.clone => clone_24
triton_poi_fused_clone_1 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(size_hints=[4194304], filename=__file__, meta={'signature': {0: '*bf16', 1: '*bf16', 2: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]})
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4194304
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 64
    x1 = (xindex // 64) % 2048
    x2 = (xindex // 131072) % 8
    x3 = (xindex // 1048576)
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (64*x2) + (512*x1) + (1048576*x3)), None).to(tl.float32)
    tl.store(out_ptr0 + (x4), tmp0, None)
''')


# kernel path: /tmp/torchinductor_yhao/cl/ccl6p6rbhtcoofthtukjedz6obcn5qbijdprggx5gyk4oik3x4rq.py
# Original ATen: aten.clone

# aten.clone => clone_25
triton_poi_fused_clone_2 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(size_hints=[2048, 2048], tile_hint=TileHint.SQUARE,filename=__file__, meta={'signature': {0: '*bf16', 1: '*bf16', 2: 'i32', 3: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]})
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, ynumel, XBLOCK : tl.constexpr, YBLOCK : tl.constexpr):
    xnumel = 2048
    ynumel = 2048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    x0 = xindex % 512
    x1 = (xindex // 512)
    y2 = yindex
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (512*y2) + (1048576*x1)), None).to(tl.float32)
    tl.store(out_ptr0 + (y2 + (2048*x3)), tmp0, None)
''')


# kernel path: /tmp/torchinductor_yhao/7q/c7qjojmegtxqf4u5czqab4hmgcxfxwghsqvq4ahqnnj4xmg4nr6k.py
# Original ATen: aten._softmax, aten._to_copy, aten.add, aten.mul, aten.rsub

# aten._softmax => amax_6, div_10, exp_6, sub_11, sum_7
# aten._to_copy => convert_element_type_43, convert_element_type_49, convert_element_type_50
# aten.add => add_38, add_39
# aten.mul => mul_29, mul_30
# aten.rsub => sub_8
triton_red_fused__softmax__to_copy_add_mul_rsub_3 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[65536, 2048],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    meta={'signature': {0: '*bf16', 1: '*bf16', 2: '*bf16', 3: 'i32', 4: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=())]}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 65536
    rnumel = 2048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x4 = xindex
    x0 = xindex % 2048
    x1 = (xindex // 2048) % 8
    _tmp35 = tl.full([XBLOCK, RBLOCK], float("-inf"), tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r3 = rindex
        tmp0 = tl.load(in_out_ptr0 + (r3 + (2048*x4)), rmask, other=0).to(tl.float32)
        tmp1 = r3 + ((-1)*x0)
        tmp2 = 0
        tmp3 = triton_helpers.minimum(tmp1, tmp2)
        tmp4 = -tmp3
        tmp5 = 16
        tmp6 = tmp4 < tmp5
        tmp7 = tmp4.to(tl.float32)
        tmp8 = 16.0
        tmp9 = tmp7 / tmp8
        tmp10 = tl.log(tmp9)
        tmp11 = 2.0794415416798357
        tmp12 = tmp10 / tmp11
        tmp13 = tmp12 * tmp8
        tmp14 = tmp13.to(tl.int64)
        tmp15 = tmp14 + tmp5
        tmp16 = 31
        tmp17 = triton_helpers.minimum(tmp15, tmp16)
        tmp18 = tl.where(tmp6, tmp4, tmp17)
        tmp19 = tmp18 + tmp2
        tmp20 = triton_helpers.promote_to_tensor(tmp19)
        tl.device_assert((0 <= tmp20) & (tmp20 < 32), "index out of bounds: 0 <= tmp20 < 32")
        tmp21 = tl.load(in_ptr0 + (x1 + (8*tmp19)), None).to(tl.float32)
        tmp22 = r3
        tmp23 = x0
        tmp24 = tmp22 <= tmp23
        tmp25 = tmp24.to(tl.float32)
        tmp26 = 1.0
        tmp27 = tmp25 * tmp26
        tmp28 = tmp27.to(tl.float32)
        tmp29 = tmp26 - tmp28
        tmp30 = -3.3895313892515355e+38
        tmp31 = tmp29 * tmp30
        tmp32 = tmp21 + tmp31
        tmp33 = tmp0 + tmp32
        tmp34 = tmp33.to(tl.float32)
        tmp36 = triton_helpers.maximum(_tmp35, tmp34)
        _tmp35 = tl.where(rmask, tmp36, _tmp35)
        tl.store(in_out_ptr0 + (r3 + (2048*x4)), tmp33, rmask)
    tmp35 = triton_helpers.max2(_tmp35, 1)[:, None]
    _tmp41 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r3 = rindex
        tmp37 = tl.load(in_out_ptr0 + (r3 + (2048*x4)), rmask, eviction_policy='evict_last', other=0).to(tl.float32)
        tmp38 = tmp37.to(tl.float32)
        tmp39 = tmp38 - tmp35
        tmp40 = tl.exp(tmp39)
        tmp42 = _tmp41 + tmp40
        _tmp41 = tl.where(rmask, tmp42, _tmp41)
    tmp41 = tl.sum(_tmp41, 1)[:, None]
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r3 = rindex
        tmp43 = tl.load(in_out_ptr0 + (r3 + (2048*x4)), rmask, other=0).to(tl.float32)
        tmp44 = tmp43.to(tl.float32)
        tmp45 = tmp44 - tmp35
        tmp46 = tl.exp(tmp45)
        tmp47 = tmp46 / tmp41
        tmp48 = tmp47.to(tl.float32)
        tl.store(out_ptr2 + (r3 + (2048*x4)), tmp48, rmask)
''')


# kernel path: /tmp/torchinductor_yhao/n6/cn6qdbyzz6y67nai4uvafcsh2wkgocv3ubddm7zclaeypyk2zc2m.py
# Original ATen: aten.clone

# aten.clone => clone_27
triton_poi_fused_clone_4 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(size_hints=[4194304], filename=__file__, meta={'signature': {0: '*bf16', 1: '*bf16', 2: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]})
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4194304
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 64
    x1 = (xindex // 64) % 8
    x2 = (xindex // 512) % 2048
    x3 = (xindex // 1048576)
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (64*x2) + (131072*x1) + (1048576*x3)), None).to(tl.float32)
    tl.store(out_ptr0 + (x4), tmp0, None)
''')


# kernel path: /tmp/torchinductor_yhao/43/c43y7l5m7lofr2kkzhnnnk6ll3eq67h2idbi5sw4mqn5dlieq635.py
# Original ATen: aten._to_copy, aten.add, aten.embedding, aten.mean, aten.mul, aten.pow, aten.rsqrt

# aten._to_copy => convert_element_type_1, convert_element_type_2, convert_element_type_51, convert_element_type_52
# aten.add => add, add_40, add_41
# aten.embedding => embedding, embedding_2
# aten.mean => mean, mean_14
# aten.mul => mul_1, mul_2, mul_35, mul_36
# aten.pow => pow_1, pow_15
# aten.rsqrt => rsqrt, rsqrt_14
triton_red_fused__to_copy_add_embedding_mean_mul_pow_rsqrt_5 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[8192, 512],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    meta={'signature': {0: '*i64', 1: '*bf16', 2: '*bf16', 3: '*i64', 4: '*bf16', 5: '*bf16', 6: '*bf16', 7: '*bf16', 8: 'i32', 9: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=())]}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr2, out_ptr3, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 8192
    rnumel = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    _tmp7 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    tmp9 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    _tmp14 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp3 = tl.load(in_ptr2 + (r1 + (512*x0)), rmask, eviction_policy='evict_last', other=0).to(tl.float32)
        tmp1 = triton_helpers.promote_to_tensor(tmp0)
        tl.device_assert((0 <= tmp1) & (tmp1 < 32128), "index out of bounds: 0 <= tmp1 < 32128")
        tmp2 = tl.load(in_ptr1 + (r1 + (512*tmp0)), rmask, eviction_policy='evict_last', other=0).to(tl.float32)
        tmp4 = tmp2 + tmp3
        tmp5 = tmp4.to(tl.float32)
        tmp6 = tmp5 * tmp5
        tmp8 = _tmp7 + tmp6
        _tmp7 = tl.where(rmask, tmp8, _tmp7)
        tmp10 = triton_helpers.promote_to_tensor(tmp9)
        tl.device_assert((0 <= tmp10) & (tmp10 < 32128), "index out of bounds: 0 <= tmp10 < 32128")
        tmp11 = tl.load(in_ptr1 + (r1 + (512*tmp9)), rmask, eviction_policy='evict_last', other=0).to(tl.float32)
        tmp12 = tmp11.to(tl.float32)
        tmp13 = tmp12 * tmp12
        tmp15 = _tmp14 + tmp13
        _tmp14 = tl.where(rmask, tmp15, _tmp14)
    tmp7 = tl.sum(_tmp7, 1)[:, None]
    tmp14 = tl.sum(_tmp14, 1)[:, None]
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp16 = tl.load(in_ptr4 + (r1), rmask, eviction_policy='evict_last', other=0).to(tl.float32)
        tmp19 = tl.load(in_ptr2 + (r1 + (512*x0)), rmask, other=0).to(tl.float32)
        tmp30 = tl.load(in_ptr5 + (r1), rmask, eviction_policy='evict_last', other=0).to(tl.float32)
        tmp17 = triton_helpers.promote_to_tensor(tmp0)
        tl.device_assert((0 <= tmp17) & (tmp17 < 32128), "index out of bounds: 0 <= tmp17 < 32128")
        tmp18 = tl.load(in_ptr1 + (r1 + (512*tmp0)), rmask, other=0).to(tl.float32)
        tmp20 = tmp18 + tmp19
        tmp21 = tmp20.to(tl.float32)
        tmp22 = 512.0
        tmp23 = tmp7 / tmp22
        tmp24 = 1e-06
        tmp25 = tmp23 + tmp24
        tmp26 = tl.math.rsqrt(tmp25)
        tmp27 = tmp21 * tmp26
        tmp28 = tmp27.to(tl.float32)
        tmp29 = tmp16 * tmp28
        tmp31 = triton_helpers.promote_to_tensor(tmp9)
        tl.device_assert((0 <= tmp31) & (tmp31 < 32128), "index out of bounds: 0 <= tmp31 < 32128")
        tmp32 = tl.load(in_ptr1 + (r1 + (512*tmp9)), rmask, other=0).to(tl.float32)
        tmp33 = tmp32.to(tl.float32)
        tmp34 = tmp14 / tmp22
        tmp35 = tmp34 + tmp24
        tmp36 = tl.math.rsqrt(tmp35)
        tmp37 = tmp33 * tmp36
        tmp38 = tmp37.to(tl.float32)
        tmp39 = tmp30 * tmp38
        tl.store(out_ptr2 + (r1 + (512*x0)), tmp29, rmask)
        tl.store(out_ptr3 + (r1 + (512*x0)), tmp39, rmask)
''')



import triton
import triton.language as tl
from torch._inductor.triton_heuristics import grid, start_graph, end_graph
from torch._C import _cuda_getCurrentRawStream as get_cuda_stream

stream1_raw = torch.cuda.Stream()
stream1 = stream1_raw.cuda_stream
stream2_raw = torch.cuda.Stream()
stream2 = stream2_raw.cuda_stream
stream3_raw = torch.cuda.Stream()
stream3 = stream3_raw.cuda_stream
stream4_raw = torch.cuda.Stream()
stream4 = stream4_raw.cuda_stream
stream5_raw = torch.cuda.Stream()
stream5 = stream5_raw.cuda_stream
stream0_raw = torch.cuda.default_stream()
stream0 = get_cuda_stream(0)


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, arg13_1, arg14_1, arg32_1, arg33_1, arg34_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg132_1, arg133_1 = args
    args.clear()
    assert_size_stride(arg0_1, (512, ), (1, ))
    assert_size_stride(arg13_1, (512, ), (1, ))
    assert_size_stride(arg14_1, (512, ), (1, ))
    assert_size_stride(arg32_1, (32128, 512), (512, 1))
    assert_size_stride(arg33_1, (512, 512), (512, 1))
    assert_size_stride(arg34_1, (512, 512), (512, 1))
    assert_size_stride(arg70_1, (512, 512), (512, 1))
    assert_size_stride(arg71_1, (512, 512), (512, 1))
    assert_size_stride(arg72_1, (512, 512), (512, 1))
    assert_size_stride(arg73_1, (32, 8), (8, 1))
    assert_size_stride(arg74_1, (512, 512), (512, 1))
    assert_size_stride(arg75_1, (512, 512), (512, 1))
    assert_size_stride(arg132_1, (4, 2048), (2048, 1))
    assert_size_stride(arg133_1, (4, 2048), (2048, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0) # no-op to ensure context
        buf1 = empty_strided((4, 2048, 512), (1048576, 512, 1), device='cuda', dtype=torch.bfloat16)
        triton_red_fused__to_copy_add_embedding_mean_mul_pow_rsqrt_0.run(arg133_1, arg32_1, arg13_1, buf1, 8192, 512, grid=grid(8192), stream=stream0)
        del arg13_1
        buf2 = empty_strided((8192, 512), (512, 1), device='cuda', dtype=torch.bfloat16)
        extern_kernels.mm(as_strided(buf1, (8192, 512), (512, 1)), as_strided(arg70_1, (512, 512), (1, 512)), out=buf2)
        del arg70_1
        buf3 = empty_strided((8192, 512), (512, 1), device='cuda', dtype=torch.bfloat16)
        extern_kernels.mm(as_strided(buf1, (8192, 512), (512, 1)), as_strided(arg71_1, (512, 512), (1, 512)), out=buf3)
        del arg71_1
        buf4 = empty_strided((4, 8, 2048, 64), (1048576, 131072, 64, 1), device='cuda', dtype=torch.bfloat16)
        triton_poi_fused_clone_1.run(buf2, buf4, 4194304, grid=grid(4194304), stream=stream0)
        del buf2
        buf5 = empty_strided((4, 8, 64, 2048), (1048576, 131072, 2048, 1), device='cuda', dtype=torch.bfloat16)
        triton_poi_fused_clone_2.run(buf3, buf5, 2048, 2048, grid=grid(2048, 2048), stream=stream0)
        buf6 = empty_strided((32, 2048, 2048), (4194304, 2048, 1), device='cuda', dtype=torch.bfloat16)
        extern_kernels.bmm(as_strided(buf4, (32, 2048, 64), (131072, 64, 1)), as_strided(buf5, (32, 64, 2048), (131072, 2048, 1)), out=buf6)
        del buf4
        del buf5
        buf7 = as_strided(buf6, (4, 8, 2048, 2048), (33554432, 4194304, 2048, 1)); del buf6  # reuse
        buf11 = empty_strided((4, 8, 2048, 2048), (33554432, 4194304, 2048, 1), device='cuda', dtype=torch.bfloat16)
        triton_red_fused__softmax__to_copy_add_mul_rsub_3.run(buf7, arg73_1, buf11, 65536, 2048, grid=grid(65536), stream=stream0)
        del buf7
        buf10 = empty_strided((8192, 512), (512, 1), device='cuda', dtype=torch.bfloat16)
        extern_kernels.mm(as_strided(buf1, (8192, 512), (512, 1)), as_strided(arg72_1, (512, 512), (1, 512)), out=buf10)
        del arg72_1
        del buf1
        buf12 = empty_strided((4, 8, 2048, 64), (1048576, 131072, 64, 1), device='cuda', dtype=torch.bfloat16)
        triton_poi_fused_clone_1.run(buf10, buf12, 4194304, grid=grid(4194304), stream=stream0)
        buf13 = empty_strided((32, 2048, 64), (131072, 64, 1), device='cuda', dtype=torch.bfloat16)
        extern_kernels.bmm(as_strided(buf11, (32, 2048, 2048), (4194304, 2048, 1)), as_strided(buf12, (32, 2048, 64), (131072, 64, 1)), out=buf13)
        del buf11
        del buf12
        buf14 = empty_strided((4, 2048, 8, 64), (1048576, 512, 64, 1), device='cuda', dtype=torch.bfloat16)
        triton_poi_fused_clone_4.run(buf13, buf14, 4194304, grid=grid(4194304), stream=stream0)
        del buf13
        buf15 = empty_strided((8192, 512), (512, 1), device='cuda', dtype=torch.bfloat16)
        extern_kernels.mm(as_strided(buf14, (8192, 512), (512, 1)), as_strided(arg74_1, (512, 512), (1, 512)), out=buf15)
        del arg74_1
        del buf14
        buf17 = empty_strided((4, 2048, 512), (1048576, 512, 1), device='cuda', dtype=torch.bfloat16)
        buf20 = empty_strided((4, 2048, 512), (1048576, 512, 1), device='cuda', dtype=torch.bfloat16)
        event_buf16_buf19_buf17_buf20 = torch.cuda.Event()
        triton_red_fused__to_copy_add_embedding_mean_mul_pow_rsqrt_5.run(arg133_1, arg32_1, buf15, arg132_1, arg14_1, arg0_1, buf17, buf20, 8192, 512, grid=grid(8192), stream=stream0)
        event_buf16_buf19_buf17_buf20.record(stream0_raw)
        del arg0_1
        del arg14_1
        # =====put buf18 to stream5 =======
        torch.cuda.set_stream(stream5_raw)
        event_buf18 = torch.cuda.Event()
        buf18 = empty_strided((8192, 512), (512, 1), device='cuda', dtype=torch.bfloat16)
        torch.cuda.set_stream(stream0_raw)
        stream5_raw.wait_event(event_buf16_buf19_buf17_buf20)
        torch.cuda.set_stream(stream5_raw)
        extern_kernels.mm(as_strided(buf17, (8192, 512), (512, 1)), as_strided(arg75_1, (512, 512), (1, 512)), out=buf18)
        event_buf18.record(stream5_raw)
        torch.cuda.set_stream(stream0_raw)
        # ======== end ======
        del arg75_1
        del buf17
        buf21 = empty_strided((8192, 512), (512, 1), device='cuda', dtype=torch.bfloat16)
        extern_kernels.mm(as_strided(buf20, (8192, 512), (512, 1)), as_strided(arg33_1, (512, 512), (1, 512)), out=buf21)
        del arg33_1
        buf22 = empty_strided((8192, 512), (512, 1), device='cuda', dtype=torch.bfloat16)
        extern_kernels.mm(as_strided(buf20, (8192, 512), (512, 1)), as_strided(arg34_1, (512, 512), (1, 512)), out=buf22)
        del arg34_1
        return buf18, buf21, buf22

def benchmark_compiled_module_origin(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg13_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg14_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg32_1 = rand_strided((32128, 512), (512, 1), device='cuda:0', dtype=torch.bfloat16)
    arg33_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.bfloat16)
    arg34_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.bfloat16)
    arg70_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.bfloat16)
    arg71_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.bfloat16)
    arg72_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.bfloat16)
    arg73_1 = rand_strided((32, 8), (8, 1), device='cuda:0', dtype=torch.bfloat16)
    arg74_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.bfloat16)
    arg75_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.bfloat16)
    arg132_1 = rand_strided((4, 2048), (2048, 1), device='cuda:0', dtype=torch.int64)
    arg133_1 = rand_strided((4, 2048), (2048, 1), device='cuda:0', dtype=torch.int64)
    return print_performance(lambda: call([arg0_1, arg13_1, arg14_1, arg32_1, arg33_1, arg34_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg132_1, arg133_1]), times=times, repeat=repeat)

def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg13_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg14_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg32_1 = rand_strided((32128, 512), (512, 1), device='cuda:0', dtype=torch.bfloat16)
    arg33_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.bfloat16)
    arg34_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.bfloat16)
    arg70_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.bfloat16)
    arg71_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.bfloat16)
    arg72_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.bfloat16)
    arg73_1 = rand_strided((32, 8), (8, 1), device='cuda:0', dtype=torch.bfloat16)
    arg74_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.bfloat16)
    arg75_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.bfloat16)
    arg132_1 = rand_strided((4, 2048), (2048, 1), device='cuda:0', dtype=torch.int64)
    arg133_1 = rand_strided((4, 2048), (2048, 1), device='cuda:0', dtype=torch.int64)
    arg0_1_clone = arg0_1.clone()
    arg13_1_clone = arg13_1.clone()
    arg14_1_clone = arg14_1.clone()
    arg32_1_clone = arg32_1.clone()
    arg33_1_clone = arg33_1.clone()
    arg34_1_clone = arg34_1.clone()
    arg70_1_clone = arg70_1.clone()
    arg71_1_clone = arg71_1.clone()
    arg72_1_clone = arg72_1.clone()
    arg73_1_clone = arg73_1.clone()
    arg74_1_clone = arg74_1.clone()
    arg75_1_clone = arg75_1.clone()
    arg132_1_clone = arg132_1.clone()
    arg133_1_clone = arg133_1.clone()
    from output_code_buf18 import call as call_origin
    results_origin = call_origin(    [arg0_1_clone, arg13_1_clone, arg14_1_clone, arg32_1_clone, arg33_1_clone, arg34_1_clone, arg70_1_clone, arg71_1_clone, arg72_1_clone, arg73_1_clone, arg74_1_clone, arg75_1_clone, arg132_1_clone, arg133_1_clone])
    torch.cuda.synchronize()
    print("====with streams ====")
    results = call([arg0_1, arg13_1, arg14_1, arg32_1, arg33_1, arg34_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg132_1, arg133_1])
    torch.cuda.synchronize()



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

if __name__ == "__main__":
    # from torch._inductor.utils import compiled_module_main
    # compiled_module_main('hf_T5', benchmark_compiled_module_origin)

    benchmark_compiled_module()
