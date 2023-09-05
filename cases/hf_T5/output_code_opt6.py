# move buf151 to stream0
# based on opt5
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


# kernel path: /tmp/torchinductor_yhao/y4/cy45fv5fbub65lzctajyo2kofsgyuzvuzhpao37p7nnhz7432wlp.py
# Original ATen: aten._softmax, aten._to_copy

# aten._softmax => amax, div_2, exp, sub_2, sum_1
# aten._to_copy => convert_element_type_6, convert_element_type_7
triton_red_fused__softmax__to_copy_6 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[65536, 2048],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    meta={'signature': {0: '*bf16', 1: '*bf16', 2: '*bf16', 3: 'i32', 4: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=())]}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr3, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 65536
    rnumel = 2048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x4 = xindex
    x0 = xindex % 2048
    x1 = (xindex // 2048) % 8
    _tmp30 = tl.full([XBLOCK, RBLOCK], float("-inf"), tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r3 = rindex
        tmp0 = tl.load(in_ptr0 + (r3 + (2048*x4)), rmask, eviction_policy='evict_last', other=0).to(tl.float32)
        tmp1 = r3 + ((-1)*x0)
        tmp2 = 0
        tmp3 = tmp1 > tmp2
        tmp4 = tmp3.to(tl.int64)
        tmp5 = 16
        tmp6 = tmp4 * tmp5
        tmp7 = tmp6 + tmp2
        tmp8 = tl.abs(tmp1)
        tmp9 = 8
        tmp10 = tmp8 < tmp9
        tmp11 = tmp8.to(tl.float32)
        tmp12 = 8.0
        tmp13 = tmp11 / tmp12
        tmp14 = tl.log(tmp13)
        tmp15 = 2.772588722239781
        tmp16 = tmp14 / tmp15
        tmp17 = tmp16 * tmp12
        tmp18 = tmp17.to(tl.int64)
        tmp19 = tmp18 + tmp9
        tmp20 = 15
        tmp21 = triton_helpers.minimum(tmp19, tmp20)
        tmp22 = tl.where(tmp10, tmp8, tmp21)
        tmp23 = tmp7 + tmp22
        tmp24 = triton_helpers.promote_to_tensor(tmp23)
        tl.device_assert(((0 <= tmp24) & (tmp24 < 32)) | ~rmask, "index out of bounds: 0 <= tmp24 < 32")
        tmp25 = tl.load(in_ptr1 + (x1 + (8*tmp23)), rmask, eviction_policy='evict_last', other=0).to(tl.float32)
        tmp26 = 0.0
        tmp27 = tmp25 + tmp26
        tmp28 = tmp0 + tmp27
        tmp29 = tmp28.to(tl.float32)
        tmp31 = triton_helpers.maximum(_tmp30, tmp29)
        _tmp30 = tl.where(rmask, tmp31, _tmp30)
    tmp30 = triton_helpers.max2(_tmp30, 1)[:, None]
    _tmp64 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r3 = rindex
        tmp32 = tl.load(in_ptr0 + (r3 + (2048*x4)), rmask, eviction_policy='evict_last', other=0).to(tl.float32)
        tmp33 = r3 + ((-1)*x0)
        tmp34 = 0
        tmp35 = tmp33 > tmp34
        tmp36 = tmp35.to(tl.int64)
        tmp37 = 16
        tmp38 = tmp36 * tmp37
        tmp39 = tmp38 + tmp34
        tmp40 = tl.abs(tmp33)
        tmp41 = 8
        tmp42 = tmp40 < tmp41
        tmp43 = tmp40.to(tl.float32)
        tmp44 = 8.0
        tmp45 = tmp43 / tmp44
        tmp46 = tl.log(tmp45)
        tmp47 = 2.772588722239781
        tmp48 = tmp46 / tmp47
        tmp49 = tmp48 * tmp44
        tmp50 = tmp49.to(tl.int64)
        tmp51 = tmp50 + tmp41
        tmp52 = 15
        tmp53 = triton_helpers.minimum(tmp51, tmp52)
        tmp54 = tl.where(tmp42, tmp40, tmp53)
        tmp55 = tmp39 + tmp54
        tmp56 = triton_helpers.promote_to_tensor(tmp55)
        tl.device_assert(((0 <= tmp56) & (tmp56 < 32)) | ~rmask, "index out of bounds: 0 <= tmp56 < 32")
        tmp57 = tl.load(in_ptr1 + (x1 + (8*tmp55)), rmask, eviction_policy='evict_last', other=0).to(tl.float32)
        tmp58 = 0.0
        tmp59 = tmp57 + tmp58
        tmp60 = tmp32 + tmp59
        tmp61 = tmp60.to(tl.float32)
        tmp62 = tmp61 - tmp30
        tmp63 = tl.exp(tmp62)
        tmp65 = _tmp64 + tmp63
        _tmp64 = tl.where(rmask, tmp65, _tmp64)
    tmp64 = tl.sum(_tmp64, 1)[:, None]
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r3 = rindex
        tmp66 = tl.load(in_ptr0 + (r3 + (2048*x4)), rmask, other=0).to(tl.float32)
        tmp67 = r3 + ((-1)*x0)
        tmp68 = 0
        tmp69 = tmp67 > tmp68
        tmp70 = tmp69.to(tl.int64)
        tmp71 = 16
        tmp72 = tmp70 * tmp71
        tmp73 = tmp72 + tmp68
        tmp74 = tl.abs(tmp67)
        tmp75 = 8
        tmp76 = tmp74 < tmp75
        tmp77 = tmp74.to(tl.float32)
        tmp78 = 8.0
        tmp79 = tmp77 / tmp78
        tmp80 = tl.log(tmp79)
        tmp81 = 2.772588722239781
        tmp82 = tmp80 / tmp81
        tmp83 = tmp82 * tmp78
        tmp84 = tmp83.to(tl.int64)
        tmp85 = tmp84 + tmp75
        tmp86 = 15
        tmp87 = triton_helpers.minimum(tmp85, tmp86)
        tmp88 = tl.where(tmp76, tmp74, tmp87)
        tmp89 = tmp73 + tmp88
        tmp90 = triton_helpers.promote_to_tensor(tmp89)
        tl.device_assert(((0 <= tmp90) & (tmp90 < 32)) | ~rmask, "index out of bounds: 0 <= tmp90 < 32")
        tmp91 = tl.load(in_ptr1 + (x1 + (8*tmp89)), rmask, other=0).to(tl.float32)
        tmp92 = 0.0
        tmp93 = tmp91 + tmp92
        tmp94 = tmp66 + tmp93
        tmp95 = tmp94.to(tl.float32)
        tmp96 = tmp95 - tmp30
        tmp97 = tl.exp(tmp96)
        tmp98 = tmp97 / tmp64
        tmp99 = tmp98.to(tl.float32)
        tl.store(out_ptr3 + (r3 + (2048*x4)), tmp99, rmask)
''')


# kernel path: /tmp/torchinductor_yhao/lo/clom2cwbwefckp4kdi42ejyfbtpngthibsekrwxocqdetaixe7dm.py
# Original ATen: aten._to_copy, aten.add, aten.embedding, aten.mean, aten.mul, aten.pow, aten.rsqrt

# aten._to_copy => convert_element_type_8, convert_element_type_9
# aten.add => add_6, add_7
# aten.embedding => embedding
# aten.mean => mean_1
# aten.mul => mul_5, mul_6
# aten.pow => pow_2
# aten.rsqrt => rsqrt_1
triton_red_fused__to_copy_add_embedding_mean_mul_pow_rsqrt_7 = async_compile.triton('triton_', '''
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
    meta={'signature': {0: '*i64', 1: '*bf16', 2: '*bf16', 3: '*bf16', 4: '*bf16', 5: 'i32', 6: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 8192
    rnumel = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    _tmp7 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
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
    tmp7 = tl.sum(_tmp7, 1)[:, None]
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp9 = tl.load(in_ptr3 + (r1), rmask, eviction_policy='evict_last', other=0).to(tl.float32)
        tmp12 = tl.load(in_ptr2 + (r1 + (512*x0)), rmask, other=0).to(tl.float32)
        tmp10 = triton_helpers.promote_to_tensor(tmp0)
        tl.device_assert((0 <= tmp10) & (tmp10 < 32128), "index out of bounds: 0 <= tmp10 < 32128")
        tmp11 = tl.load(in_ptr1 + (r1 + (512*tmp0)), rmask, other=0).to(tl.float32)
        tmp13 = tmp11 + tmp12
        tmp14 = tmp13.to(tl.float32)
        tmp15 = 512.0
        tmp16 = tmp7 / tmp15
        tmp17 = 1e-06
        tmp18 = tmp16 + tmp17
        tmp19 = tl.math.rsqrt(tmp18)
        tmp20 = tmp14 * tmp19
        tmp21 = tmp20.to(tl.float32)
        tmp22 = tmp9 * tmp21
        tl.store(out_ptr1 + (r1 + (512*x0)), tmp22, rmask)
''')


# kernel path: /tmp/torchinductor_yhao/5j/c5jt72tpen4lq3u2m2c2dytpqzsgxgor3bskb3yu5ekyezdmntry.py
# Original ATen: aten.relu

# aten.relu => relu
triton_poi_fused_relu_8 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(size_hints=[16777216], filename=__file__, meta={'signature': {0: '*bf16', 1: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'configs': [instance_descriptor(divisible_by_16=(0, 1), equal_to_1=())]})
@triton.jit
def triton_(in_out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16777216
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (x0), None).to(tl.float32)
    tmp1 = triton_helpers.maximum(0, tmp0)
    tl.store(in_out_ptr0 + (x0), tmp1, None)
''')


# kernel path: /tmp/torchinductor_yhao/bq/cbqpe5hjrgdd63fxitv24ml5hpz7vlxo6nxam7w24h4vg5godf6w.py
# Original ATen: aten._to_copy, aten.add, aten.embedding, aten.mean, aten.mul, aten.pow, aten.rsqrt

# aten._to_copy => convert_element_type_10, convert_element_type_11
# aten.add => add_6, add_8, add_9
# aten.embedding => embedding
# aten.mean => mean_2
# aten.mul => mul_7, mul_8
# aten.pow => pow_3
# aten.rsqrt => rsqrt_2
triton_red_fused__to_copy_add_embedding_mean_mul_pow_rsqrt_9 = async_compile.triton('triton_', '''
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
    meta={'signature': {0: '*i64', 1: '*bf16', 2: '*bf16', 3: '*bf16', 4: '*bf16', 5: '*bf16', 6: 'i32', 7: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=())]}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 8192
    rnumel = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    _tmp9 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp3 = tl.load(in_ptr2 + (r1 + (512*x0)), rmask, eviction_policy='evict_last', other=0).to(tl.float32)
        tmp5 = tl.load(in_ptr3 + (r1 + (512*x0)), rmask, eviction_policy='evict_last', other=0).to(tl.float32)
        tmp1 = triton_helpers.promote_to_tensor(tmp0)
        tl.device_assert((0 <= tmp1) & (tmp1 < 32128), "index out of bounds: 0 <= tmp1 < 32128")
        tmp2 = tl.load(in_ptr1 + (r1 + (512*tmp0)), rmask, eviction_policy='evict_last', other=0).to(tl.float32)
        tmp4 = tmp2 + tmp3
        tmp6 = tmp4 + tmp5
        tmp7 = tmp6.to(tl.float32)
        tmp8 = tmp7 * tmp7
        tmp10 = _tmp9 + tmp8
        _tmp9 = tl.where(rmask, tmp10, _tmp9)
    tmp9 = tl.sum(_tmp9, 1)[:, None]
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp11 = tl.load(in_ptr4 + (r1), rmask, eviction_policy='evict_last', other=0).to(tl.float32)
        tmp14 = tl.load(in_ptr2 + (r1 + (512*x0)), rmask, other=0).to(tl.float32)
        tmp16 = tl.load(in_ptr3 + (r1 + (512*x0)), rmask, other=0).to(tl.float32)
        tmp12 = triton_helpers.promote_to_tensor(tmp0)
        tl.device_assert((0 <= tmp12) & (tmp12 < 32128), "index out of bounds: 0 <= tmp12 < 32128")
        tmp13 = tl.load(in_ptr1 + (r1 + (512*tmp0)), rmask, other=0).to(tl.float32)
        tmp15 = tmp13 + tmp14
        tmp17 = tmp15 + tmp16
        tmp18 = tmp17.to(tl.float32)
        tmp19 = 512.0
        tmp20 = tmp9 / tmp19
        tmp21 = 1e-06
        tmp22 = tmp20 + tmp21
        tmp23 = tl.math.rsqrt(tmp22)
        tmp24 = tmp18 * tmp23
        tmp25 = tmp24.to(tl.float32)
        tmp26 = tmp11 * tmp25
        tl.store(out_ptr1 + (r1 + (512*x0)), tmp26, rmask)
''')


# kernel path: /tmp/torchinductor_yhao/jx/cjx4dk4spqjj75xdoro6li5wpkr37b7no7kq5cw3hl5lmu6lypsk.py
# Original ATen: aten._to_copy, aten.add, aten.embedding, aten.mean, aten.mul, aten.pow, aten.rsqrt

# aten._to_copy => convert_element_type_14, convert_element_type_15
# aten.add => add_11, add_12, add_6, add_8
# aten.embedding => embedding
# aten.mean => mean_3
# aten.mul => mul_10, mul_9
# aten.pow => pow_4
# aten.rsqrt => rsqrt_3
triton_per_fused__to_copy_add_embedding_mean_mul_pow_rsqrt_10 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[8192, 512],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    meta={'signature': {0: '*bf16', 1: '*i64', 2: '*bf16', 3: '*bf16', 4: '*bf16', 5: '*bf16', 6: '*bf16', 7: 'i32', 8: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=())]}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 8192
    rnumel = 512
    RBLOCK: tl.constexpr = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    x0 = xindex
    r1 = rindex
    tmp0 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_out_ptr0 + (r1 + (512*x0)), rmask, other=0).to(tl.float32)
    tmp5 = tl.load(in_ptr2 + (r1 + (512*x0)), rmask, other=0).to(tl.float32)
    tmp7 = tl.load(in_ptr3 + (r1 + (512*x0)), rmask, other=0).to(tl.float32)
    tmp14 = tl.load(in_ptr4 + (r1), rmask, eviction_policy='evict_last', other=0).to(tl.float32)
    tmp1 = triton_helpers.promote_to_tensor(tmp0)
    tl.device_assert((0 <= tmp1) & (tmp1 < 32128), "index out of bounds: 0 <= tmp1 < 32128")
    tmp2 = tl.load(in_ptr1 + (r1 + (512*tmp0)), rmask, other=0).to(tl.float32)
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 + tmp5
    tmp8 = tmp6 + tmp7
    tmp9 = tmp8.to(tl.float32)
    tmp10 = tmp9 * tmp9
    tmp12 = tl.where(rmask, tmp10, 0)
    tmp13 = triton_helpers.promote_to_tensor(tl.sum(tmp12, 0))
    tmp15 = 512.0
    tmp16 = tmp13 / tmp15
    tmp17 = 1e-06
    tmp18 = tmp16 + tmp17
    tmp19 = tl.math.rsqrt(tmp18)
    tmp20 = tmp9 * tmp19
    tmp21 = tmp20.to(tl.float32)
    tmp22 = tmp14 * tmp21
    tl.store(in_out_ptr0 + (r1 + (512*x0)), tmp8, rmask)
    tl.store(out_ptr1 + (r1 + (512*x0)), tmp22, rmask)
''')


# kernel path: /tmp/torchinductor_yhao/kg/ckgscrub6wwwi4wdftznbh37d3ov32dbnswzezhik63bmadeipn4.py
# Original ATen: aten._to_copy, aten.add, aten.mean, aten.mul, aten.pow, aten.rsqrt

# aten._to_copy => convert_element_type_16, convert_element_type_17
# aten.add => add_13, add_14
# aten.mean => mean_4
# aten.mul => mul_11, mul_12
# aten.pow => pow_5
# aten.rsqrt => rsqrt_4
triton_per_fused__to_copy_add_mean_mul_pow_rsqrt_11 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[8192, 512],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    meta={'signature': {0: '*bf16', 1: '*bf16', 2: '*bf16', 3: '*bf16', 4: 'i32', 5: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 8192
    rnumel = 512
    RBLOCK: tl.constexpr = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (512*x0)), rmask, other=0).to(tl.float32)
    tmp1 = tl.load(in_ptr1 + (r1 + (512*x0)), rmask, other=0).to(tl.float32)
    tmp8 = tl.load(in_ptr2 + (r1), rmask, eviction_policy='evict_last', other=0).to(tl.float32)
    tmp2 = tmp0 + tmp1
    tmp3 = tmp2.to(tl.float32)
    tmp4 = tmp3 * tmp3
    tmp6 = tl.where(rmask, tmp4, 0)
    tmp7 = triton_helpers.promote_to_tensor(tl.sum(tmp6, 0))
    tmp9 = 512.0
    tmp10 = tmp7 / tmp9
    tmp11 = 1e-06
    tmp12 = tmp10 + tmp11
    tmp13 = tl.math.rsqrt(tmp12)
    tmp14 = tmp3 * tmp13
    tmp15 = tmp14.to(tl.float32)
    tmp16 = tmp8 * tmp15
    tl.store(out_ptr1 + (r1 + (512*x0)), tmp16, rmask)
''')


# kernel path: /tmp/torchinductor_yhao/k7/ck7tz2moipiqjy4m4qqizjuk5efuwjgy4ogb5rjufeb72ahhhlbz.py
# Original ATen: aten._to_copy, aten.add, aten.mean, aten.mul, aten.pow, aten.rsqrt

# aten._to_copy => convert_element_type_20, convert_element_type_21
# aten.add => add_13, add_16, add_17
# aten.mean => mean_5
# aten.mul => mul_13, mul_14
# aten.pow => pow_6
# aten.rsqrt => rsqrt_5
triton_per_fused__to_copy_add_mean_mul_pow_rsqrt_12 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[8192, 512],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    meta={'signature': {0: '*bf16', 1: '*bf16', 2: '*bf16', 3: '*bf16', 4: '*bf16', 5: 'i32', 6: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 8192
    rnumel = 512
    RBLOCK: tl.constexpr = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (512*x0)), rmask, other=0).to(tl.float32)
    tmp1 = tl.load(in_ptr1 + (r1 + (512*x0)), rmask, other=0).to(tl.float32)
    tmp3 = tl.load(in_ptr2 + (r1 + (512*x0)), rmask, other=0).to(tl.float32)
    tmp10 = tl.load(in_ptr3 + (r1), rmask, eviction_policy='evict_last', other=0).to(tl.float32)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp5 = tmp4.to(tl.float32)
    tmp6 = tmp5 * tmp5
    tmp8 = tl.where(rmask, tmp6, 0)
    tmp9 = triton_helpers.promote_to_tensor(tl.sum(tmp8, 0))
    tmp11 = 512.0
    tmp12 = tmp9 / tmp11
    tmp13 = 1e-06
    tmp14 = tmp12 + tmp13
    tmp15 = tl.math.rsqrt(tmp14)
    tmp16 = tmp5 * tmp15
    tmp17 = tmp16.to(tl.float32)
    tmp18 = tmp10 * tmp17
    tl.store(out_ptr1 + (r1 + (512*x0)), tmp18, rmask)
''')


# kernel path: /tmp/torchinductor_yhao/i6/ci6364x34ichu7zhsu4gngfoxep7kxk4k334mzbq7g6zakbzbrzj.py
# Original ATen: aten._to_copy, aten.add, aten.mean, aten.mul, aten.pow, aten.rsqrt

# aten._to_copy => convert_element_type_22, convert_element_type_23
# aten.add => add_13, add_16, add_18, add_19
# aten.mean => mean_6
# aten.mul => mul_15, mul_16
# aten.pow => pow_7
# aten.rsqrt => rsqrt_6
triton_per_fused__to_copy_add_mean_mul_pow_rsqrt_13 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[8192, 512],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    meta={'signature': {0: '*bf16', 1: '*bf16', 2: '*bf16', 3: '*bf16', 4: '*bf16', 5: '*bf16', 6: 'i32', 7: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=())]}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 8192
    rnumel = 512
    RBLOCK: tl.constexpr = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (512*x0)), rmask, other=0).to(tl.float32)
    tmp1 = tl.load(in_ptr1 + (r1 + (512*x0)), rmask, other=0).to(tl.float32)
    tmp3 = tl.load(in_ptr2 + (r1 + (512*x0)), rmask, other=0).to(tl.float32)
    tmp5 = tl.load(in_ptr3 + (r1 + (512*x0)), rmask, other=0).to(tl.float32)
    tmp12 = tl.load(in_ptr4 + (r1), rmask, eviction_policy='evict_last', other=0).to(tl.float32)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 + tmp5
    tmp7 = tmp6.to(tl.float32)
    tmp8 = tmp7 * tmp7
    tmp10 = tl.where(rmask, tmp8, 0)
    tmp11 = triton_helpers.promote_to_tensor(tl.sum(tmp10, 0))
    tmp13 = 512.0
    tmp14 = tmp11 / tmp13
    tmp15 = 1e-06
    tmp16 = tmp14 + tmp15
    tmp17 = tl.math.rsqrt(tmp16)
    tmp18 = tmp7 * tmp17
    tmp19 = tmp18.to(tl.float32)
    tmp20 = tmp12 * tmp19
    tl.store(out_ptr1 + (r1 + (512*x0)), tmp20, rmask)
''')


# kernel path: /tmp/torchinductor_yhao/hu/chufsxnyhlubmlrz4ou6ho33umym5ihpwqeblggl35siqbo4ynob.py
# Original ATen: aten._to_copy, aten.add, aten.mean, aten.mul, aten.pow, aten.rsqrt

# aten._to_copy => convert_element_type_26, convert_element_type_27
# aten.add => add_13, add_16, add_18, add_21, add_22
# aten.mean => mean_7
# aten.mul => mul_17, mul_18
# aten.pow => pow_8
# aten.rsqrt => rsqrt_7
triton_per_fused__to_copy_add_mean_mul_pow_rsqrt_14 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[8192, 512],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    meta={'signature': {0: '*bf16', 1: '*bf16', 2: '*bf16', 3: '*bf16', 4: '*bf16', 5: '*bf16', 6: '*bf16', 7: 'i32', 8: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=())]}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 8192
    rnumel = 512
    RBLOCK: tl.constexpr = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (r1 + (512*x0)), rmask, other=0).to(tl.float32)
    tmp1 = tl.load(in_ptr0 + (r1 + (512*x0)), rmask, other=0).to(tl.float32)
    tmp3 = tl.load(in_ptr1 + (r1 + (512*x0)), rmask, other=0).to(tl.float32)
    tmp5 = tl.load(in_ptr2 + (r1 + (512*x0)), rmask, other=0).to(tl.float32)
    tmp7 = tl.load(in_ptr3 + (r1 + (512*x0)), rmask, other=0).to(tl.float32)
    tmp14 = tl.load(in_ptr4 + (r1), rmask, eviction_policy='evict_last', other=0).to(tl.float32)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 + tmp5
    tmp8 = tmp6 + tmp7
    tmp9 = tmp8.to(tl.float32)
    tmp10 = tmp9 * tmp9
    tmp12 = tl.where(rmask, tmp10, 0)
    tmp13 = triton_helpers.promote_to_tensor(tl.sum(tmp12, 0))
    tmp15 = 512.0
    tmp16 = tmp13 / tmp15
    tmp17 = 1e-06
    tmp18 = tmp16 + tmp17
    tmp19 = tl.math.rsqrt(tmp18)
    tmp20 = tmp9 * tmp19
    tmp21 = tmp20.to(tl.float32)
    tmp22 = tmp14 * tmp21
    tl.store(in_out_ptr0 + (r1 + (512*x0)), tmp8, rmask)
    tl.store(out_ptr1 + (r1 + (512*x0)), tmp22, rmask)
''')


# kernel path: /tmp/torchinductor_yhao/mz/cmz575gkwdopplde3rokxbkre4k4an6js6jqfb5ettz27aju2jx6.py
# Original ATen: aten._to_copy, aten.add, aten.mean, aten.mul, aten.pow, aten.rsqrt

# aten._to_copy => convert_element_type_38, convert_element_type_39
# aten.add => add_23, add_26, add_28, add_31, add_32
# aten.mean => mean_11
# aten.mul => mul_25, mul_26
# aten.pow => pow_12
# aten.rsqrt => rsqrt_11
triton_per_fused__to_copy_add_mean_mul_pow_rsqrt_15 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[8192, 512],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    meta={'signature': {0: '*bf16', 1: '*bf16', 2: '*bf16', 3: '*bf16', 4: '*bf16', 5: '*bf16', 6: '*bf16', 7: 'i32', 8: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=())]}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 8192
    rnumel = 512
    RBLOCK: tl.constexpr = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (512*x0)), rmask, other=0).to(tl.float32)
    tmp1 = tl.load(in_out_ptr0 + (r1 + (512*x0)), rmask, other=0).to(tl.float32)
    tmp3 = tl.load(in_ptr1 + (r1 + (512*x0)), rmask, other=0).to(tl.float32)
    tmp5 = tl.load(in_ptr2 + (r1 + (512*x0)), rmask, other=0).to(tl.float32)
    tmp7 = tl.load(in_ptr3 + (r1 + (512*x0)), rmask, other=0).to(tl.float32)
    tmp14 = tl.load(in_ptr4 + (r1), rmask, eviction_policy='evict_last', other=0).to(tl.float32)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 + tmp5
    tmp8 = tmp6 + tmp7
    tmp9 = tmp8.to(tl.float32)
    tmp10 = tmp9 * tmp9
    tmp12 = tl.where(rmask, tmp10, 0)
    tmp13 = triton_helpers.promote_to_tensor(tl.sum(tmp12, 0))
    tmp15 = 512.0
    tmp16 = tmp13 / tmp15
    tmp17 = 1e-06
    tmp18 = tmp16 + tmp17
    tmp19 = tl.math.rsqrt(tmp18)
    tmp20 = tmp9 * tmp19
    tmp21 = tmp20.to(tl.float32)
    tmp22 = tmp14 * tmp21
    tl.store(in_out_ptr0 + (r1 + (512*x0)), tmp8, rmask)
    tl.store(out_ptr1 + (r1 + (512*x0)), tmp22, rmask)
''')


# kernel path: /tmp/torchinductor_yhao/qu/cquh6opgbxe5kmjquzqk555ib7g2fint2ejkrxy2zti55t4bmv27.py
# Original ATen: aten._softmax, aten._to_copy

# aten._softmax => amax_7, div_11, exp_7, sub_12, sum_8
# aten._to_copy => convert_element_type_53, convert_element_type_54
triton_red_fused__softmax__to_copy_16 = async_compile.triton('triton_', '''
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
    meta={'signature': {0: '*bf16', 1: '*bf16', 2: 'i32', 3: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]}
)
@triton.jit
def triton_(in_ptr0, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 65536
    rnumel = 2048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp2 = tl.full([XBLOCK, RBLOCK], float("-inf"), tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_ptr0 + (r1 + (2048*x0)), rmask, eviction_policy='evict_last', other=0).to(tl.float32)
        tmp1 = tmp0.to(tl.float32)
        tmp3 = triton_helpers.maximum(_tmp2, tmp1)
        _tmp2 = tl.where(rmask, tmp3, _tmp2)
    tmp2 = triton_helpers.max2(_tmp2, 1)[:, None]
    _tmp8 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp4 = tl.load(in_ptr0 + (r1 + (2048*x0)), rmask, eviction_policy='evict_last', other=0).to(tl.float32)
        tmp5 = tmp4.to(tl.float32)
        tmp6 = tmp5 - tmp2
        tmp7 = tl.exp(tmp6)
        tmp9 = _tmp8 + tmp7
        _tmp8 = tl.where(rmask, tmp9, _tmp8)
    tmp8 = tl.sum(_tmp8, 1)[:, None]
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp10 = tl.load(in_ptr0 + (r1 + (2048*x0)), rmask, other=0).to(tl.float32)
        tmp11 = tmp10.to(tl.float32)
        tmp12 = tmp11 - tmp2
        tmp13 = tl.exp(tmp12)
        tmp14 = tmp13 / tmp8
        tmp15 = tmp14.to(tl.float32)
        tl.store(out_ptr2 + (r1 + (2048*x0)), tmp15, rmask)
''')


# kernel path: /tmp/torchinductor_yhao/oe/coeq2hzxyamsfjswquo6wup7rhkr6v542m7l6cdjqoy4alm7pd7x.py
# Original ATen: aten._to_copy, aten.add, aten.mean, aten.mul, aten.pow, aten.rsqrt

# aten._to_copy => convert_element_type_107, convert_element_type_108
# aten.add => add_81, add_84, add_86, add_87
# aten.mean => mean_31
# aten.mul => mul_69, mul_70, mul_71
# aten.pow => pow_32
# aten.rsqrt => rsqrt_31
triton_per_fused__to_copy_add_mean_mul_pow_rsqrt_17 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[8192, 512],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    meta={'signature': {0: '*bf16', 1: '*bf16', 2: '*bf16', 3: '*bf16', 4: '*bf16', 5: '*bf16', 6: 'i32', 7: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=())]}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 8192
    rnumel = 512
    RBLOCK: tl.constexpr = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (512*x0)), rmask, other=0).to(tl.float32)
    tmp1 = tl.load(in_ptr1 + (r1 + (512*x0)), rmask, other=0).to(tl.float32)
    tmp3 = tl.load(in_ptr2 + (r1 + (512*x0)), rmask, other=0).to(tl.float32)
    tmp5 = tl.load(in_ptr3 + (r1 + (512*x0)), rmask, other=0).to(tl.float32)
    tmp12 = tl.load(in_ptr4 + (r1), rmask, eviction_policy='evict_last', other=0).to(tl.float32)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 + tmp5
    tmp7 = tmp6.to(tl.float32)
    tmp8 = tmp7 * tmp7
    tmp10 = tl.where(rmask, tmp8, 0)
    tmp11 = triton_helpers.promote_to_tensor(tl.sum(tmp10, 0))
    tmp13 = 512.0
    tmp14 = tmp11 / tmp13
    tmp15 = 1e-06
    tmp16 = tmp14 + tmp15
    tmp17 = tl.math.rsqrt(tmp16)
    tmp18 = tmp7 * tmp17
    tmp19 = tmp18.to(tl.float32)
    tmp20 = tmp12 * tmp19
    tmp21 = 0.04419417382415922
    tmp22 = tmp20 * tmp21
    tl.store(out_ptr1 + (r1 + (512*x0)), tmp22, rmask)
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
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1 = args
    args.clear()
    assert_size_stride(arg0_1, (512, ), (1, ))
    assert_size_stride(arg1_1, (512, ), (1, ))
    assert_size_stride(arg2_1, (512, ), (1, ))
    assert_size_stride(arg3_1, (512, ), (1, ))
    assert_size_stride(arg4_1, (512, ), (1, ))
    assert_size_stride(arg5_1, (512, ), (1, ))
    assert_size_stride(arg6_1, (512, ), (1, ))
    assert_size_stride(arg7_1, (512, ), (1, ))
    assert_size_stride(arg8_1, (512, ), (1, ))
    assert_size_stride(arg9_1, (512, ), (1, ))
    assert_size_stride(arg10_1, (512, ), (1, ))
    assert_size_stride(arg11_1, (512, ), (1, ))
    assert_size_stride(arg12_1, (512, ), (1, ))
    assert_size_stride(arg13_1, (512, ), (1, ))
    assert_size_stride(arg14_1, (512, ), (1, ))
    assert_size_stride(arg15_1, (512, ), (1, ))
    assert_size_stride(arg16_1, (512, ), (1, ))
    assert_size_stride(arg17_1, (512, ), (1, ))
    assert_size_stride(arg18_1, (512, ), (1, ))
    assert_size_stride(arg19_1, (512, ), (1, ))
    assert_size_stride(arg20_1, (512, ), (1, ))
    assert_size_stride(arg21_1, (512, ), (1, ))
    assert_size_stride(arg22_1, (512, ), (1, ))
    assert_size_stride(arg23_1, (512, ), (1, ))
    assert_size_stride(arg24_1, (512, ), (1, ))
    assert_size_stride(arg25_1, (512, ), (1, ))
    assert_size_stride(arg26_1, (512, ), (1, ))
    assert_size_stride(arg27_1, (512, ), (1, ))
    assert_size_stride(arg28_1, (512, ), (1, ))
    assert_size_stride(arg29_1, (512, ), (1, ))
    assert_size_stride(arg30_1, (512, ), (1, ))
    assert_size_stride(arg31_1, (512, ), (1, ))
    assert_size_stride(arg32_1, (32128, 512), (512, 1))
    assert_size_stride(arg33_1, (512, 512), (512, 1))
    assert_size_stride(arg34_1, (512, 512), (512, 1))
    assert_size_stride(arg35_1, (512, 512), (512, 1))
    assert_size_stride(arg36_1, (32, 8), (8, 1))
    assert_size_stride(arg37_1, (512, 512), (512, 1))
    assert_size_stride(arg38_1, (2048, 512), (512, 1))
    assert_size_stride(arg39_1, (512, 2048), (2048, 1))
    assert_size_stride(arg40_1, (512, 512), (512, 1))
    assert_size_stride(arg41_1, (512, 512), (512, 1))
    assert_size_stride(arg42_1, (512, 512), (512, 1))
    assert_size_stride(arg43_1, (512, 512), (512, 1))
    assert_size_stride(arg44_1, (2048, 512), (512, 1))
    assert_size_stride(arg45_1, (512, 2048), (2048, 1))
    assert_size_stride(arg46_1, (512, 512), (512, 1))
    assert_size_stride(arg47_1, (512, 512), (512, 1))
    assert_size_stride(arg48_1, (512, 512), (512, 1))
    assert_size_stride(arg49_1, (512, 512), (512, 1))
    assert_size_stride(arg50_1, (2048, 512), (512, 1))
    assert_size_stride(arg51_1, (512, 2048), (2048, 1))
    assert_size_stride(arg52_1, (512, 512), (512, 1))
    assert_size_stride(arg53_1, (512, 512), (512, 1))
    assert_size_stride(arg54_1, (512, 512), (512, 1))
    assert_size_stride(arg55_1, (512, 512), (512, 1))
    assert_size_stride(arg56_1, (2048, 512), (512, 1))
    assert_size_stride(arg57_1, (512, 2048), (2048, 1))
    assert_size_stride(arg58_1, (512, 512), (512, 1))
    assert_size_stride(arg59_1, (512, 512), (512, 1))
    assert_size_stride(arg60_1, (512, 512), (512, 1))
    assert_size_stride(arg61_1, (512, 512), (512, 1))
    assert_size_stride(arg62_1, (2048, 512), (512, 1))
    assert_size_stride(arg63_1, (512, 2048), (2048, 1))
    assert_size_stride(arg64_1, (512, 512), (512, 1))
    assert_size_stride(arg65_1, (512, 512), (512, 1))
    assert_size_stride(arg66_1, (512, 512), (512, 1))
    assert_size_stride(arg67_1, (512, 512), (512, 1))
    assert_size_stride(arg68_1, (2048, 512), (512, 1))
    assert_size_stride(arg69_1, (512, 2048), (2048, 1))
    assert_size_stride(arg70_1, (512, 512), (512, 1))
    assert_size_stride(arg71_1, (512, 512), (512, 1))
    assert_size_stride(arg72_1, (512, 512), (512, 1))
    assert_size_stride(arg73_1, (32, 8), (8, 1))
    assert_size_stride(arg74_1, (512, 512), (512, 1))
    assert_size_stride(arg75_1, (512, 512), (512, 1))
    assert_size_stride(arg76_1, (512, 512), (512, 1))
    assert_size_stride(arg77_1, (512, 512), (512, 1))
    assert_size_stride(arg78_1, (512, 512), (512, 1))
    assert_size_stride(arg79_1, (2048, 512), (512, 1))
    assert_size_stride(arg80_1, (512, 2048), (2048, 1))
    assert_size_stride(arg81_1, (512, 512), (512, 1))
    assert_size_stride(arg82_1, (512, 512), (512, 1))
    assert_size_stride(arg83_1, (512, 512), (512, 1))
    assert_size_stride(arg84_1, (512, 512), (512, 1))
    assert_size_stride(arg85_1, (512, 512), (512, 1))
    assert_size_stride(arg86_1, (512, 512), (512, 1))
    assert_size_stride(arg87_1, (512, 512), (512, 1))
    assert_size_stride(arg88_1, (512, 512), (512, 1))
    assert_size_stride(arg89_1, (2048, 512), (512, 1))
    assert_size_stride(arg90_1, (512, 2048), (2048, 1))
    assert_size_stride(arg91_1, (512, 512), (512, 1))
    assert_size_stride(arg92_1, (512, 512), (512, 1))
    assert_size_stride(arg93_1, (512, 512), (512, 1))
    assert_size_stride(arg94_1, (512, 512), (512, 1))
    assert_size_stride(arg95_1, (512, 512), (512, 1))
    assert_size_stride(arg96_1, (512, 512), (512, 1))
    assert_size_stride(arg97_1, (512, 512), (512, 1))
    assert_size_stride(arg98_1, (512, 512), (512, 1))
    assert_size_stride(arg99_1, (2048, 512), (512, 1))
    assert_size_stride(arg100_1, (512, 2048), (2048, 1))
    assert_size_stride(arg101_1, (512, 512), (512, 1))
    assert_size_stride(arg102_1, (512, 512), (512, 1))
    assert_size_stride(arg103_1, (512, 512), (512, 1))
    assert_size_stride(arg104_1, (512, 512), (512, 1))
    assert_size_stride(arg105_1, (512, 512), (512, 1))
    assert_size_stride(arg106_1, (512, 512), (512, 1))
    assert_size_stride(arg107_1, (512, 512), (512, 1))
    assert_size_stride(arg108_1, (512, 512), (512, 1))
    assert_size_stride(arg109_1, (2048, 512), (512, 1))
    assert_size_stride(arg110_1, (512, 2048), (2048, 1))
    assert_size_stride(arg111_1, (512, 512), (512, 1))
    assert_size_stride(arg112_1, (512, 512), (512, 1))
    assert_size_stride(arg113_1, (512, 512), (512, 1))
    assert_size_stride(arg114_1, (512, 512), (512, 1))
    assert_size_stride(arg115_1, (512, 512), (512, 1))
    assert_size_stride(arg116_1, (512, 512), (512, 1))
    assert_size_stride(arg117_1, (512, 512), (512, 1))
    assert_size_stride(arg118_1, (512, 512), (512, 1))
    assert_size_stride(arg119_1, (2048, 512), (512, 1))
    assert_size_stride(arg120_1, (512, 2048), (2048, 1))
    assert_size_stride(arg121_1, (512, 512), (512, 1))
    assert_size_stride(arg122_1, (512, 512), (512, 1))
    assert_size_stride(arg123_1, (512, 512), (512, 1))
    assert_size_stride(arg124_1, (512, 512), (512, 1))
    assert_size_stride(arg125_1, (512, 512), (512, 1))
    assert_size_stride(arg126_1, (512, 512), (512, 1))
    assert_size_stride(arg127_1, (512, 512), (512, 1))
    assert_size_stride(arg128_1, (512, 512), (512, 1))
    assert_size_stride(arg129_1, (2048, 512), (512, 1))
    assert_size_stride(arg130_1, (512, 2048), (2048, 1))
    assert_size_stride(arg131_1, (32128, 512), (512, 1))
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
        torch.cuda.set_stream(stream5_raw)
        event_buf18 = torch.cuda.Event()
        buf18 = empty_strided((8192, 512), (512, 1), device='cuda', dtype=torch.bfloat16)
        torch.cuda.set_stream(stream0_raw)
        stream5_raw.wait_event(event_buf16_buf19_buf17_buf20)
        torch.cuda.set_stream(stream5_raw)
        extern_kernels.mm(as_strided(buf17, (8192, 512), (512, 1)), as_strided(arg75_1, (512, 512), (1, 512)), out=buf18)
        event_buf18.record(stream5_raw)
        torch.cuda.set_stream(stream0_raw)
        del arg75_1
        del buf17
        buf21 = empty_strided((8192, 512), (512, 1), device='cuda', dtype=torch.bfloat16)
        extern_kernels.mm(as_strided(buf20, (8192, 512), (512, 1)), as_strided(arg33_1, (512, 512), (1, 512)), out=buf21)
        del arg33_1
        buf22 = empty_strided((8192, 512), (512, 1), device='cuda', dtype=torch.bfloat16)
        extern_kernels.mm(as_strided(buf20, (8192, 512), (512, 1)), as_strided(arg34_1, (512, 512), (1, 512)), out=buf22)
        del arg34_1
        buf23 = empty_strided((4, 8, 2048, 64), (1048576, 131072, 64, 1), device='cuda', dtype=torch.bfloat16)
        triton_poi_fused_clone_1.run(buf21, buf23, 4194304, grid=grid(4194304), stream=stream0)
        del buf21
        buf24 = empty_strided((4, 8, 64, 2048), (1048576, 131072, 2048, 1), device='cuda', dtype=torch.bfloat16)
        triton_poi_fused_clone_2.run(buf22, buf24, 2048, 2048, grid=grid(2048, 2048), stream=stream0)
        del buf22
        buf25 = empty_strided((32, 2048, 2048), (4194304, 2048, 1), device='cuda', dtype=torch.bfloat16)
        extern_kernels.bmm(as_strided(buf23, (32, 2048, 64), (131072, 64, 1)), as_strided(buf24, (32, 64, 2048), (131072, 2048, 1)), out=buf25)
        del buf23
        del buf24
        buf30 = empty_strided((4, 8, 2048, 2048), (33554432, 4194304, 2048, 1), device='cuda', dtype=torch.bfloat16)
        triton_red_fused__softmax__to_copy_6.run(buf25, arg36_1, buf30, 65536, 2048, grid=grid(65536), stream=stream0)
        del buf25
        buf29 = empty_strided((8192, 512), (512, 1), device='cuda', dtype=torch.bfloat16)
        extern_kernels.mm(as_strided(buf20, (8192, 512), (512, 1)), as_strided(arg35_1, (512, 512), (1, 512)), out=buf29)
        del arg35_1
        del buf20
        buf31 = empty_strided((4, 8, 2048, 64), (1048576, 131072, 64, 1), device='cuda', dtype=torch.bfloat16)
        triton_poi_fused_clone_1.run(buf29, buf31, 4194304, grid=grid(4194304), stream=stream0)
        del buf29
        buf32 = empty_strided((32, 2048, 64), (131072, 64, 1), device='cuda', dtype=torch.bfloat16)
        extern_kernels.bmm(as_strided(buf30, (32, 2048, 2048), (4194304, 2048, 1)), as_strided(buf31, (32, 2048, 64), (131072, 64, 1)), out=buf32)
        del buf30
        del buf31
        buf33 = empty_strided((4, 2048, 8, 64), (1048576, 512, 64, 1), device='cuda', dtype=torch.bfloat16)
        triton_poi_fused_clone_4.run(buf32, buf33, 4194304, grid=grid(4194304), stream=stream0)
        del buf32
        buf34 = empty_strided((8192, 512), (512, 1), device='cuda', dtype=torch.bfloat16)
        extern_kernels.mm(as_strided(buf33, (8192, 512), (512, 1)), as_strided(arg37_1, (512, 512), (1, 512)), out=buf34)
        del arg37_1
        del buf33
        buf36 = empty_strided((4, 2048, 512), (1048576, 512, 1), device='cuda', dtype=torch.bfloat16)
        triton_red_fused__to_copy_add_embedding_mean_mul_pow_rsqrt_7.run(arg132_1, arg32_1, buf34, arg1_1, buf36, 8192, 512, grid=grid(8192), stream=stream0)
        del arg1_1
        buf37 = empty_strided((8192, 2048), (2048, 1), device='cuda', dtype=torch.bfloat16)
        extern_kernels.mm(as_strided(buf36, (8192, 512), (512, 1)), as_strided(arg38_1, (512, 2048), (1, 512)), out=buf37)
        del arg38_1
        del buf36
        buf38 = as_strided(buf37, (4, 2048, 2048), (4194304, 2048, 1)); del buf37  # reuse
        triton_poi_fused_relu_8.run(buf38, 16777216, grid=grid(16777216), stream=stream0)
        buf39 = empty_strided((8192, 512), (512, 1), device='cuda', dtype=torch.bfloat16)
        extern_kernels.mm(as_strided(buf38, (8192, 2048), (2048, 1)), as_strided(arg39_1, (2048, 512), (1, 2048)), out=buf39)
        del arg39_1
        del buf38
        buf41 = empty_strided((4, 2048, 512), (1048576, 512, 1), device='cuda', dtype=torch.bfloat16)
        triton_red_fused__to_copy_add_embedding_mean_mul_pow_rsqrt_9.run(arg132_1, arg32_1, buf34, buf39, arg2_1, buf41, 8192, 512, grid=grid(8192), stream=stream0)
        del arg2_1
        buf42 = empty_strided((8192, 512), (512, 1), device='cuda', dtype=torch.bfloat16)
        extern_kernels.mm(as_strided(buf41, (8192, 512), (512, 1)), as_strided(arg40_1, (512, 512), (1, 512)), out=buf42)
        del arg40_1
        buf43 = empty_strided((8192, 512), (512, 1), device='cuda', dtype=torch.bfloat16)
        extern_kernels.mm(as_strided(buf41, (8192, 512), (512, 1)), as_strided(arg41_1, (512, 512), (1, 512)), out=buf43)
        del arg41_1
        buf44 = empty_strided((4, 8, 2048, 64), (1048576, 131072, 64, 1), device='cuda', dtype=torch.bfloat16)
        triton_poi_fused_clone_1.run(buf42, buf44, 4194304, grid=grid(4194304), stream=stream0)
        del buf42
        buf45 = empty_strided((4, 8, 64, 2048), (1048576, 131072, 2048, 1), device='cuda', dtype=torch.bfloat16)
        triton_poi_fused_clone_2.run(buf43, buf45, 2048, 2048, grid=grid(2048, 2048), stream=stream0)
        del buf43
        buf46 = empty_strided((32, 2048, 2048), (4194304, 2048, 1), device='cuda', dtype=torch.bfloat16)
        extern_kernels.bmm(as_strided(buf44, (32, 2048, 64), (131072, 64, 1)), as_strided(buf45, (32, 64, 2048), (131072, 2048, 1)), out=buf46)
        del buf44
        del buf45
        buf51 = empty_strided((4, 8, 2048, 2048), (33554432, 4194304, 2048, 1), device='cuda', dtype=torch.bfloat16)
        triton_red_fused__softmax__to_copy_6.run(buf46, arg36_1, buf51, 65536, 2048, grid=grid(65536), stream=stream0)
        del buf46
        buf50 = empty_strided((8192, 512), (512, 1), device='cuda', dtype=torch.bfloat16)
        extern_kernels.mm(as_strided(buf41, (8192, 512), (512, 1)), as_strided(arg42_1, (512, 512), (1, 512)), out=buf50)
        del arg42_1
        del buf41
        buf52 = empty_strided((4, 8, 2048, 64), (1048576, 131072, 64, 1), device='cuda', dtype=torch.bfloat16)
        triton_poi_fused_clone_1.run(buf50, buf52, 4194304, grid=grid(4194304), stream=stream0)
        del buf50
        buf53 = empty_strided((32, 2048, 64), (131072, 64, 1), device='cuda', dtype=torch.bfloat16)
        extern_kernels.bmm(as_strided(buf51, (32, 2048, 2048), (4194304, 2048, 1)), as_strided(buf52, (32, 2048, 64), (131072, 64, 1)), out=buf53)
        del buf51
        del buf52
        buf54 = empty_strided((4, 2048, 8, 64), (1048576, 512, 64, 1), device='cuda', dtype=torch.bfloat16)
        triton_poi_fused_clone_4.run(buf53, buf54, 4194304, grid=grid(4194304), stream=stream0)
        del buf53
        buf55 = empty_strided((8192, 512), (512, 1), device='cuda', dtype=torch.bfloat16)
        extern_kernels.mm(as_strided(buf54, (8192, 512), (512, 1)), as_strided(arg43_1, (512, 512), (1, 512)), out=buf55)
        del arg43_1
        del buf54
        buf56 = as_strided(buf34, (4, 2048, 512), (1048576, 512, 1)); del buf34  # reuse
        buf58 = empty_strided((4, 2048, 512), (1048576, 512, 1), device='cuda', dtype=torch.bfloat16)
        triton_per_fused__to_copy_add_embedding_mean_mul_pow_rsqrt_10.run(buf56, arg132_1, arg32_1, buf39, buf55, arg3_1, buf58, 8192, 512, grid=grid(8192), stream=stream0)
        del arg132_1
        del arg3_1
        del buf39
        del buf55
        buf59 = empty_strided((8192, 2048), (2048, 1), device='cuda', dtype=torch.bfloat16)
        extern_kernels.mm(as_strided(buf58, (8192, 512), (512, 1)), as_strided(arg44_1, (512, 2048), (1, 512)), out=buf59)
        del arg44_1
        del buf58
        buf60 = as_strided(buf59, (4, 2048, 2048), (4194304, 2048, 1)); del buf59  # reuse
        triton_poi_fused_relu_8.run(buf60, 16777216, grid=grid(16777216), stream=stream0)
        buf61 = empty_strided((8192, 512), (512, 1), device='cuda', dtype=torch.bfloat16)
        extern_kernels.mm(as_strided(buf60, (8192, 2048), (2048, 1)), as_strided(arg45_1, (2048, 512), (1, 2048)), out=buf61)
        del arg45_1
        del buf60
        buf63 = empty_strided((4, 2048, 512), (1048576, 512, 1), device='cuda', dtype=torch.bfloat16)
        triton_per_fused__to_copy_add_mean_mul_pow_rsqrt_11.run(buf56, buf61, arg4_1, buf63, 8192, 512, grid=grid(8192), stream=stream0)
        del arg4_1
        buf64 = empty_strided((8192, 512), (512, 1), device='cuda', dtype=torch.bfloat16)
        extern_kernels.mm(as_strided(buf63, (8192, 512), (512, 1)), as_strided(arg46_1, (512, 512), (1, 512)), out=buf64)
        del arg46_1
        buf65 = empty_strided((8192, 512), (512, 1), device='cuda', dtype=torch.bfloat16)
        extern_kernels.mm(as_strided(buf63, (8192, 512), (512, 1)), as_strided(arg47_1, (512, 512), (1, 512)), out=buf65)
        del arg47_1
        buf66 = empty_strided((4, 8, 2048, 64), (1048576, 131072, 64, 1), device='cuda', dtype=torch.bfloat16)
        triton_poi_fused_clone_1.run(buf64, buf66, 4194304, grid=grid(4194304), stream=stream0)
        del buf64
        buf67 = empty_strided((4, 8, 64, 2048), (1048576, 131072, 2048, 1), device='cuda', dtype=torch.bfloat16)
        triton_poi_fused_clone_2.run(buf65, buf67, 2048, 2048, grid=grid(2048, 2048), stream=stream0)
        del buf65
        buf68 = empty_strided((32, 2048, 2048), (4194304, 2048, 1), device='cuda', dtype=torch.bfloat16)
        extern_kernels.bmm(as_strided(buf66, (32, 2048, 64), (131072, 64, 1)), as_strided(buf67, (32, 64, 2048), (131072, 2048, 1)), out=buf68)
        del buf66
        del buf67
        buf73 = empty_strided((4, 8, 2048, 2048), (33554432, 4194304, 2048, 1), device='cuda', dtype=torch.bfloat16)
        triton_red_fused__softmax__to_copy_6.run(buf68, arg36_1, buf73, 65536, 2048, grid=grid(65536), stream=stream0)
        del buf68
        buf72 = empty_strided((8192, 512), (512, 1), device='cuda', dtype=torch.bfloat16)
        extern_kernels.mm(as_strided(buf63, (8192, 512), (512, 1)), as_strided(arg48_1, (512, 512), (1, 512)), out=buf72)
        del arg48_1
        del buf63
        buf74 = empty_strided((4, 8, 2048, 64), (1048576, 131072, 64, 1), device='cuda', dtype=torch.bfloat16)
        triton_poi_fused_clone_1.run(buf72, buf74, 4194304, grid=grid(4194304), stream=stream0)
        del buf72
        buf75 = empty_strided((32, 2048, 64), (131072, 64, 1), device='cuda', dtype=torch.bfloat16)
        extern_kernels.bmm(as_strided(buf73, (32, 2048, 2048), (4194304, 2048, 1)), as_strided(buf74, (32, 2048, 64), (131072, 64, 1)), out=buf75)
        del buf73
        del buf74
        buf76 = empty_strided((4, 2048, 8, 64), (1048576, 512, 64, 1), device='cuda', dtype=torch.bfloat16)
        triton_poi_fused_clone_4.run(buf75, buf76, 4194304, grid=grid(4194304), stream=stream0)
        del buf75
        buf77 = empty_strided((8192, 512), (512, 1), device='cuda', dtype=torch.bfloat16)
        extern_kernels.mm(as_strided(buf76, (8192, 512), (512, 1)), as_strided(arg49_1, (512, 512), (1, 512)), out=buf77)
        del arg49_1
        del buf76
        buf79 = empty_strided((4, 2048, 512), (1048576, 512, 1), device='cuda', dtype=torch.bfloat16)
        triton_per_fused__to_copy_add_mean_mul_pow_rsqrt_12.run(buf56, buf61, buf77, arg5_1, buf79, 8192, 512, grid=grid(8192), stream=stream0)
        del arg5_1
        buf80 = empty_strided((8192, 2048), (2048, 1), device='cuda', dtype=torch.bfloat16)
        extern_kernels.mm(as_strided(buf79, (8192, 512), (512, 1)), as_strided(arg50_1, (512, 2048), (1, 512)), out=buf80)
        del arg50_1
        del buf79
        buf81 = as_strided(buf80, (4, 2048, 2048), (4194304, 2048, 1)); del buf80  # reuse
        triton_poi_fused_relu_8.run(buf81, 16777216, grid=grid(16777216), stream=stream0)
        buf82 = empty_strided((8192, 512), (512, 1), device='cuda', dtype=torch.bfloat16)
        extern_kernels.mm(as_strided(buf81, (8192, 2048), (2048, 1)), as_strided(arg51_1, (2048, 512), (1, 2048)), out=buf82)
        del arg51_1
        del buf81
        buf84 = empty_strided((4, 2048, 512), (1048576, 512, 1), device='cuda', dtype=torch.bfloat16)
        triton_per_fused__to_copy_add_mean_mul_pow_rsqrt_13.run(buf56, buf61, buf77, buf82, arg6_1, buf84, 8192, 512, grid=grid(8192), stream=stream0)
        del arg6_1
        buf85 = empty_strided((8192, 512), (512, 1), device='cuda', dtype=torch.bfloat16)
        extern_kernels.mm(as_strided(buf84, (8192, 512), (512, 1)), as_strided(arg52_1, (512, 512), (1, 512)), out=buf85)
        del arg52_1
        buf86 = empty_strided((8192, 512), (512, 1), device='cuda', dtype=torch.bfloat16)
        extern_kernels.mm(as_strided(buf84, (8192, 512), (512, 1)), as_strided(arg53_1, (512, 512), (1, 512)), out=buf86)
        del arg53_1
        buf87 = empty_strided((4, 8, 2048, 64), (1048576, 131072, 64, 1), device='cuda', dtype=torch.bfloat16)
        triton_poi_fused_clone_1.run(buf85, buf87, 4194304, grid=grid(4194304), stream=stream0)
        del buf85
        buf88 = empty_strided((4, 8, 64, 2048), (1048576, 131072, 2048, 1), device='cuda', dtype=torch.bfloat16)
        triton_poi_fused_clone_2.run(buf86, buf88, 2048, 2048, grid=grid(2048, 2048), stream=stream0)
        del buf86
        buf89 = empty_strided((32, 2048, 2048), (4194304, 2048, 1), device='cuda', dtype=torch.bfloat16)
        extern_kernels.bmm(as_strided(buf87, (32, 2048, 64), (131072, 64, 1)), as_strided(buf88, (32, 64, 2048), (131072, 2048, 1)), out=buf89)
        del buf87
        del buf88
        buf94 = empty_strided((4, 8, 2048, 2048), (33554432, 4194304, 2048, 1), device='cuda', dtype=torch.bfloat16)
        triton_red_fused__softmax__to_copy_6.run(buf89, arg36_1, buf94, 65536, 2048, grid=grid(65536), stream=stream0)
        del buf89
        buf93 = empty_strided((8192, 512), (512, 1), device='cuda', dtype=torch.bfloat16)
        extern_kernels.mm(as_strided(buf84, (8192, 512), (512, 1)), as_strided(arg54_1, (512, 512), (1, 512)), out=buf93)
        del arg54_1
        del buf84
        buf95 = empty_strided((4, 8, 2048, 64), (1048576, 131072, 64, 1), device='cuda', dtype=torch.bfloat16)
        triton_poi_fused_clone_1.run(buf93, buf95, 4194304, grid=grid(4194304), stream=stream0)
        del buf93
        buf96 = empty_strided((32, 2048, 64), (131072, 64, 1), device='cuda', dtype=torch.bfloat16)
        extern_kernels.bmm(as_strided(buf94, (32, 2048, 2048), (4194304, 2048, 1)), as_strided(buf95, (32, 2048, 64), (131072, 64, 1)), out=buf96)
        del buf94
        del buf95
        buf97 = empty_strided((4, 2048, 8, 64), (1048576, 512, 64, 1), device='cuda', dtype=torch.bfloat16)
        triton_poi_fused_clone_4.run(buf96, buf97, 4194304, grid=grid(4194304), stream=stream0)
        del buf96
        buf98 = empty_strided((8192, 512), (512, 1), device='cuda', dtype=torch.bfloat16)
        extern_kernels.mm(as_strided(buf97, (8192, 512), (512, 1)), as_strided(arg55_1, (512, 512), (1, 512)), out=buf98)
        del arg55_1
        del buf97
        buf99 = buf56; del buf56  # reuse
        buf101 = empty_strided((4, 2048, 512), (1048576, 512, 1), device='cuda', dtype=torch.bfloat16)
        triton_per_fused__to_copy_add_mean_mul_pow_rsqrt_14.run(buf99, buf61, buf77, buf82, buf98, arg7_1, buf101, 8192, 512, grid=grid(8192), stream=stream0)
        del arg7_1
        del buf61
        del buf77
        del buf82
        del buf98
        buf102 = empty_strided((8192, 2048), (2048, 1), device='cuda', dtype=torch.bfloat16)
        extern_kernels.mm(as_strided(buf101, (8192, 512), (512, 1)), as_strided(arg56_1, (512, 2048), (1, 512)), out=buf102)
        del arg56_1
        del buf101
        buf103 = as_strided(buf102, (4, 2048, 2048), (4194304, 2048, 1)); del buf102  # reuse
        triton_poi_fused_relu_8.run(buf103, 16777216, grid=grid(16777216), stream=stream0)
        buf104 = empty_strided((8192, 512), (512, 1), device='cuda', dtype=torch.bfloat16)
        extern_kernels.mm(as_strided(buf103, (8192, 2048), (2048, 1)), as_strided(arg57_1, (2048, 512), (1, 2048)), out=buf104)
        del arg57_1
        del buf103
        buf106 = empty_strided((4, 2048, 512), (1048576, 512, 1), device='cuda', dtype=torch.bfloat16)
        triton_per_fused__to_copy_add_mean_mul_pow_rsqrt_11.run(buf99, buf104, arg8_1, buf106, 8192, 512, grid=grid(8192), stream=stream0)
        del arg8_1
        buf107 = empty_strided((8192, 512), (512, 1), device='cuda', dtype=torch.bfloat16)
        extern_kernels.mm(as_strided(buf106, (8192, 512), (512, 1)), as_strided(arg58_1, (512, 512), (1, 512)), out=buf107)
        del arg58_1
        buf108 = empty_strided((8192, 512), (512, 1), device='cuda', dtype=torch.bfloat16)
        extern_kernels.mm(as_strided(buf106, (8192, 512), (512, 1)), as_strided(arg59_1, (512, 512), (1, 512)), out=buf108)
        del arg59_1
        buf109 = empty_strided((4, 8, 2048, 64), (1048576, 131072, 64, 1), device='cuda', dtype=torch.bfloat16)
        triton_poi_fused_clone_1.run(buf107, buf109, 4194304, grid=grid(4194304), stream=stream0)
        del buf107
        buf110 = empty_strided((4, 8, 64, 2048), (1048576, 131072, 2048, 1), device='cuda', dtype=torch.bfloat16)
        triton_poi_fused_clone_2.run(buf108, buf110, 2048, 2048, grid=grid(2048, 2048), stream=stream0)
        del buf108
        buf111 = empty_strided((32, 2048, 2048), (4194304, 2048, 1), device='cuda', dtype=torch.bfloat16)
        extern_kernels.bmm(as_strided(buf109, (32, 2048, 64), (131072, 64, 1)), as_strided(buf110, (32, 64, 2048), (131072, 2048, 1)), out=buf111)
        del buf109
        del buf110
        buf116 = empty_strided((4, 8, 2048, 2048), (33554432, 4194304, 2048, 1), device='cuda', dtype=torch.bfloat16)
        triton_red_fused__softmax__to_copy_6.run(buf111, arg36_1, buf116, 65536, 2048, grid=grid(65536), stream=stream0)
        del buf111
        buf115 = empty_strided((8192, 512), (512, 1), device='cuda', dtype=torch.bfloat16)
        extern_kernels.mm(as_strided(buf106, (8192, 512), (512, 1)), as_strided(arg60_1, (512, 512), (1, 512)), out=buf115)
        del arg60_1
        del buf106
        buf117 = empty_strided((4, 8, 2048, 64), (1048576, 131072, 64, 1), device='cuda', dtype=torch.bfloat16)
        triton_poi_fused_clone_1.run(buf115, buf117, 4194304, grid=grid(4194304), stream=stream0)
        del buf115
        buf118 = empty_strided((32, 2048, 64), (131072, 64, 1), device='cuda', dtype=torch.bfloat16)
        extern_kernels.bmm(as_strided(buf116, (32, 2048, 2048), (4194304, 2048, 1)), as_strided(buf117, (32, 2048, 64), (131072, 64, 1)), out=buf118)
        del buf116
        del buf117
        buf119 = empty_strided((4, 2048, 8, 64), (1048576, 512, 64, 1), device='cuda', dtype=torch.bfloat16)
        triton_poi_fused_clone_4.run(buf118, buf119, 4194304, grid=grid(4194304), stream=stream0)
        del buf118
        buf120 = empty_strided((8192, 512), (512, 1), device='cuda', dtype=torch.bfloat16)
        extern_kernels.mm(as_strided(buf119, (8192, 512), (512, 1)), as_strided(arg61_1, (512, 512), (1, 512)), out=buf120)
        del arg61_1
        del buf119
        buf122 = empty_strided((4, 2048, 512), (1048576, 512, 1), device='cuda', dtype=torch.bfloat16)
        triton_per_fused__to_copy_add_mean_mul_pow_rsqrt_12.run(buf99, buf104, buf120, arg9_1, buf122, 8192, 512, grid=grid(8192), stream=stream0)
        del arg9_1
        buf123 = empty_strided((8192, 2048), (2048, 1), device='cuda', dtype=torch.bfloat16)
        extern_kernels.mm(as_strided(buf122, (8192, 512), (512, 1)), as_strided(arg62_1, (512, 2048), (1, 512)), out=buf123)
        del arg62_1
        del buf122
        buf124 = as_strided(buf123, (4, 2048, 2048), (4194304, 2048, 1)); del buf123  # reuse
        triton_poi_fused_relu_8.run(buf124, 16777216, grid=grid(16777216), stream=stream0)
        buf125 = empty_strided((8192, 512), (512, 1), device='cuda', dtype=torch.bfloat16)
        extern_kernels.mm(as_strided(buf124, (8192, 2048), (2048, 1)), as_strided(arg63_1, (2048, 512), (1, 2048)), out=buf125)
        del arg63_1
        del buf124
        buf127 = empty_strided((4, 2048, 512), (1048576, 512, 1), device='cuda', dtype=torch.bfloat16)
        triton_per_fused__to_copy_add_mean_mul_pow_rsqrt_13.run(buf99, buf104, buf120, buf125, arg10_1, buf127, 8192, 512, grid=grid(8192), stream=stream0)
        del arg10_1
        buf128 = empty_strided((8192, 512), (512, 1), device='cuda', dtype=torch.bfloat16)
        extern_kernels.mm(as_strided(buf127, (8192, 512), (512, 1)), as_strided(arg64_1, (512, 512), (1, 512)), out=buf128)
        del arg64_1
        buf129 = empty_strided((8192, 512), (512, 1), device='cuda', dtype=torch.bfloat16)
        extern_kernels.mm(as_strided(buf127, (8192, 512), (512, 1)), as_strided(arg65_1, (512, 512), (1, 512)), out=buf129)
        del arg65_1
        buf130 = empty_strided((4, 8, 2048, 64), (1048576, 131072, 64, 1), device='cuda', dtype=torch.bfloat16)
        triton_poi_fused_clone_1.run(buf128, buf130, 4194304, grid=grid(4194304), stream=stream0)
        del buf128
        buf131 = empty_strided((4, 8, 64, 2048), (1048576, 131072, 2048, 1), device='cuda', dtype=torch.bfloat16)
        triton_poi_fused_clone_2.run(buf129, buf131, 2048, 2048, grid=grid(2048, 2048), stream=stream0)
        del buf129
        buf132 = empty_strided((32, 2048, 2048), (4194304, 2048, 1), device='cuda', dtype=torch.bfloat16)
        extern_kernels.bmm(as_strided(buf130, (32, 2048, 64), (131072, 64, 1)), as_strided(buf131, (32, 64, 2048), (131072, 2048, 1)), out=buf132)
        del buf130
        del buf131
        buf137 = empty_strided((4, 8, 2048, 2048), (33554432, 4194304, 2048, 1), device='cuda', dtype=torch.bfloat16)
        triton_red_fused__softmax__to_copy_6.run(buf132, arg36_1, buf137, 65536, 2048, grid=grid(65536), stream=stream0)
        del arg36_1
        del buf132
        buf136 = empty_strided((8192, 512), (512, 1), device='cuda', dtype=torch.bfloat16)
        extern_kernels.mm(as_strided(buf127, (8192, 512), (512, 1)), as_strided(arg66_1, (512, 512), (1, 512)), out=buf136)
        del arg66_1
        del buf127
        buf138 = empty_strided((4, 8, 2048, 64), (1048576, 131072, 64, 1), device='cuda', dtype=torch.bfloat16)
        triton_poi_fused_clone_1.run(buf136, buf138, 4194304, grid=grid(4194304), stream=stream0)
        del buf136
        buf139 = empty_strided((32, 2048, 64), (131072, 64, 1), device='cuda', dtype=torch.bfloat16)
        extern_kernels.bmm(as_strided(buf137, (32, 2048, 2048), (4194304, 2048, 1)), as_strided(buf138, (32, 2048, 64), (131072, 64, 1)), out=buf139)
        del buf137
        del buf138
        buf140 = empty_strided((4, 2048, 8, 64), (1048576, 512, 64, 1), device='cuda', dtype=torch.bfloat16)
        triton_poi_fused_clone_4.run(buf139, buf140, 4194304, grid=grid(4194304), stream=stream0)
        del buf139
        buf141 = empty_strided((8192, 512), (512, 1), device='cuda', dtype=torch.bfloat16)
        extern_kernels.mm(as_strided(buf140, (8192, 512), (512, 1)), as_strided(arg67_1, (512, 512), (1, 512)), out=buf141)
        del arg67_1
        del buf140
        buf142 = as_strided(buf104, (4, 2048, 512), (1048576, 512, 1)); del buf104  # reuse
        buf144 = empty_strided((4, 2048, 512), (1048576, 512, 1), device='cuda', dtype=torch.bfloat16)
        triton_per_fused__to_copy_add_mean_mul_pow_rsqrt_15.run(buf142, buf99, buf120, buf125, buf141, arg11_1, buf144, 8192, 512, grid=grid(8192), stream=stream0)
        del arg11_1
        del buf120
        del buf125
        del buf141
        del buf99
        buf145 = empty_strided((8192, 2048), (2048, 1), device='cuda', dtype=torch.bfloat16)
        extern_kernels.mm(as_strided(buf144, (8192, 512), (512, 1)), as_strided(arg68_1, (512, 2048), (1, 512)), out=buf145)
        del arg68_1
        del buf144
        buf146 = as_strided(buf145, (4, 2048, 2048), (4194304, 2048, 1)); del buf145  # reuse
        triton_poi_fused_relu_8.run(buf146, 16777216, grid=grid(16777216), stream=stream0)
        buf147 = empty_strided((8192, 512), (512, 1), device='cuda', dtype=torch.bfloat16)
        extern_kernels.mm(as_strided(buf146, (8192, 2048), (2048, 1)), as_strided(arg69_1, (2048, 512), (1, 2048)), out=buf147)
        del arg69_1
        del buf146
        buf149 = empty_strided((4, 2048, 512), (1048576, 512, 1), device='cuda', dtype=torch.bfloat16)
        triton_per_fused__to_copy_add_mean_mul_pow_rsqrt_11.run(buf142, buf147, arg12_1, buf149, 8192, 512, grid=grid(8192), stream=stream0)
        del arg12_1
        del buf142
        del buf147
        buf150 = empty_strided((8192, 512), (512, 1), device='cuda', dtype=torch.bfloat16)
        extern_kernels.mm(as_strided(buf149, (8192, 512), (512, 1)), as_strided(arg76_1, (512, 512), (1, 512)), out=buf150)
        del arg76_1

        stream0_raw.wait_event(event_buf18)
        # torch.cuda.set_stream(stream5_raw)
        buf151 = empty_strided((4, 8, 2048, 64), (1048576, 131072, 64, 1), device='cuda', dtype=torch.bfloat16)
        # torch.cuda.set_stream(stream0_raw)
        # event_buf151 = torch.cuda.Event()
        # torch.cuda.set_stream(stream5_raw)
        triton_poi_fused_clone_1.run(buf18, buf151, 4194304, grid=grid(4194304), stream=stream0)
        # torch.cuda.set_stream(stream0_raw)
        # event_buf151.record(stream0_raw)
        del buf18
        buf152 = empty_strided((4, 8, 64, 2048), (1048576, 131072, 2048, 1), device='cuda', dtype=torch.bfloat16)
        triton_poi_fused_clone_2.run(buf150, buf152, 2048, 2048, grid=grid(2048, 2048), stream=stream0)
        buf153 = empty_strided((32, 2048, 2048), (4194304, 2048, 1), device='cuda', dtype=torch.bfloat16)
        # stream0_raw.wait_event(event_buf151)
        extern_kernels.bmm(as_strided(buf151, (32, 2048, 64), (131072, 64, 1)), as_strided(buf152, (32, 64, 2048), (131072, 2048, 1)), out=buf153)
        del buf151
        del buf152
        buf157 = empty_strided((4, 8, 2048, 2048), (33554432, 4194304, 2048, 1), device='cuda', dtype=torch.bfloat16)
        triton_red_fused__softmax__to_copy_16.run(buf153, buf157, 65536, 2048, grid=grid(65536), stream=stream0)
        del buf153
        buf156 = empty_strided((8192, 512), (512, 1), device='cuda', dtype=torch.bfloat16)
        extern_kernels.mm(as_strided(buf149, (8192, 512), (512, 1)), as_strided(arg77_1, (512, 512), (1, 512)), out=buf156)
        del arg77_1
        buf158 = empty_strided((4, 8, 2048, 64), (1048576, 131072, 64, 1), device='cuda', dtype=torch.bfloat16)
        triton_poi_fused_clone_1.run(buf156, buf158, 4194304, grid=grid(4194304), stream=stream0)
        buf159 = empty_strided((32, 2048, 64), (131072, 64, 1), device='cuda', dtype=torch.bfloat16)
        extern_kernels.bmm(as_strided(buf157, (32, 2048, 2048), (4194304, 2048, 1)), as_strided(buf158, (32, 2048, 64), (131072, 64, 1)), out=buf159)
        del buf157
        del buf158
        buf160 = empty_strided((4, 2048, 8, 64), (1048576, 512, 64, 1), device='cuda', dtype=torch.bfloat16)
        triton_poi_fused_clone_4.run(buf159, buf160, 4194304, grid=grid(4194304), stream=stream0)
        del buf159
        buf161 = empty_strided((8192, 512), (512, 1), device='cuda', dtype=torch.bfloat16)
        extern_kernels.mm(as_strided(buf160, (8192, 512), (512, 1)), as_strided(arg78_1, (512, 512), (1, 512)), out=buf161)
        del arg78_1
        del buf160
        buf163 = empty_strided((4, 2048, 512), (1048576, 512, 1), device='cuda', dtype=torch.bfloat16)
        triton_red_fused__to_copy_add_embedding_mean_mul_pow_rsqrt_9.run(arg133_1, arg32_1, buf15, buf161, arg15_1, buf163, 8192, 512, grid=grid(8192), stream=stream0)
        del arg15_1
        buf164 = empty_strided((8192, 2048), (2048, 1), device='cuda', dtype=torch.bfloat16)
        extern_kernels.mm(as_strided(buf163, (8192, 512), (512, 1)), as_strided(arg79_1, (512, 2048), (1, 512)), out=buf164)
        del arg79_1
        del buf163
        buf165 = as_strided(buf164, (4, 2048, 2048), (4194304, 2048, 1)); del buf164  # reuse
        triton_poi_fused_relu_8.run(buf165, 16777216, grid=grid(16777216), stream=stream0)
        buf166 = empty_strided((8192, 512), (512, 1), device='cuda', dtype=torch.bfloat16)
        extern_kernels.mm(as_strided(buf165, (8192, 2048), (2048, 1)), as_strided(arg80_1, (2048, 512), (1, 2048)), out=buf166)
        del arg80_1
        del buf165
        buf167 = as_strided(buf15, (4, 2048, 512), (1048576, 512, 1)); del buf15  # reuse
        buf169 = empty_strided((4, 2048, 512), (1048576, 512, 1), device='cuda', dtype=torch.bfloat16)
        triton_per_fused__to_copy_add_embedding_mean_mul_pow_rsqrt_10.run(buf167, arg133_1, arg32_1, buf161, buf166, arg16_1, buf169, 8192, 512, grid=grid(8192), stream=stream0)
        del arg133_1
        del arg16_1
        del arg32_1
        del buf161
        del buf166
        buf170 = empty_strided((8192, 512), (512, 1), device='cuda', dtype=torch.bfloat16)
        extern_kernels.mm(as_strided(buf169, (8192, 512), (512, 1)), as_strided(arg81_1, (512, 512), (1, 512)), out=buf170)
        del arg81_1
        buf171 = empty_strided((8192, 512), (512, 1), device='cuda', dtype=torch.bfloat16)
        extern_kernels.mm(as_strided(buf169, (8192, 512), (512, 1)), as_strided(arg82_1, (512, 512), (1, 512)), out=buf171)
        del arg82_1
        buf172 = empty_strided((4, 8, 2048, 64), (1048576, 131072, 64, 1), device='cuda', dtype=torch.bfloat16)
        triton_poi_fused_clone_1.run(buf170, buf172, 4194304, grid=grid(4194304), stream=stream0)
        del buf170
        buf173 = empty_strided((4, 8, 64, 2048), (1048576, 131072, 2048, 1), device='cuda', dtype=torch.bfloat16)
        triton_poi_fused_clone_2.run(buf171, buf173, 2048, 2048, grid=grid(2048, 2048), stream=stream0)
        buf174 = empty_strided((32, 2048, 2048), (4194304, 2048, 1), device='cuda', dtype=torch.bfloat16)
        extern_kernels.bmm(as_strided(buf172, (32, 2048, 64), (131072, 64, 1)), as_strided(buf173, (32, 64, 2048), (131072, 2048, 1)), out=buf174)
        del buf172
        del buf173
        buf175 = as_strided(buf174, (4, 8, 2048, 2048), (33554432, 4194304, 2048, 1)); del buf174  # reuse
        buf179 = empty_strided((4, 8, 2048, 2048), (33554432, 4194304, 2048, 1), device='cuda', dtype=torch.bfloat16)
        triton_red_fused__softmax__to_copy_add_mul_rsub_3.run(buf175, arg73_1, buf179, 65536, 2048, grid=grid(65536), stream=stream0)
        del buf175
        buf178 = empty_strided((8192, 512), (512, 1), device='cuda', dtype=torch.bfloat16)
        extern_kernels.mm(as_strided(buf169, (8192, 512), (512, 1)), as_strided(arg83_1, (512, 512), (1, 512)), out=buf178)
        del arg83_1
        del buf169
        buf180 = empty_strided((4, 8, 2048, 64), (1048576, 131072, 64, 1), device='cuda', dtype=torch.bfloat16)
        triton_poi_fused_clone_1.run(buf178, buf180, 4194304, grid=grid(4194304), stream=stream0)
        buf181 = empty_strided((32, 2048, 64), (131072, 64, 1), device='cuda', dtype=torch.bfloat16)
        extern_kernels.bmm(as_strided(buf179, (32, 2048, 2048), (4194304, 2048, 1)), as_strided(buf180, (32, 2048, 64), (131072, 64, 1)), out=buf181)
        del buf179
        del buf180
        buf182 = empty_strided((4, 2048, 8, 64), (1048576, 512, 64, 1), device='cuda', dtype=torch.bfloat16)
        triton_poi_fused_clone_4.run(buf181, buf182, 4194304, grid=grid(4194304), stream=stream0)
        del buf181
        buf183 = empty_strided((8192, 512), (512, 1), device='cuda', dtype=torch.bfloat16)
        extern_kernels.mm(as_strided(buf182, (8192, 512), (512, 1)), as_strided(arg84_1, (512, 512), (1, 512)), out=buf183)
        del arg84_1
        del buf182
        buf185 = empty_strided((4, 2048, 512), (1048576, 512, 1), device='cuda', dtype=torch.bfloat16)
        triton_per_fused__to_copy_add_mean_mul_pow_rsqrt_11.run(buf167, buf183, arg17_1, buf185, 8192, 512, grid=grid(8192), stream=stream0)
        del arg17_1
        buf186 = empty_strided((8192, 512), (512, 1), device='cuda', dtype=torch.bfloat16)
        extern_kernels.mm(as_strided(buf185, (8192, 512), (512, 1)), as_strided(arg85_1, (512, 512), (1, 512)), out=buf186)
        del arg85_1
        del buf185
        buf187 = empty_strided((8192, 512), (512, 1), device='cuda', dtype=torch.bfloat16)
        extern_kernels.mm(as_strided(buf149, (8192, 512), (512, 1)), as_strided(arg86_1, (512, 512), (1, 512)), out=buf187)
        del arg86_1
        buf188 = empty_strided((4, 8, 2048, 64), (1048576, 131072, 64, 1), device='cuda', dtype=torch.bfloat16)
        triton_poi_fused_clone_1.run(buf186, buf188, 4194304, grid=grid(4194304), stream=stream0)
        del buf186
        buf189 = empty_strided((4, 8, 64, 2048), (1048576, 131072, 2048, 1), device='cuda', dtype=torch.bfloat16)
        triton_poi_fused_clone_2.run(buf187, buf189, 2048, 2048, grid=grid(2048, 2048), stream=stream0)
        buf190 = empty_strided((32, 2048, 2048), (4194304, 2048, 1), device='cuda', dtype=torch.bfloat16)
        extern_kernels.bmm(as_strided(buf188, (32, 2048, 64), (131072, 64, 1)), as_strided(buf189, (32, 64, 2048), (131072, 2048, 1)), out=buf190)
        del buf188
        del buf189
        buf194 = empty_strided((4, 8, 2048, 2048), (33554432, 4194304, 2048, 1), device='cuda', dtype=torch.bfloat16)
        triton_red_fused__softmax__to_copy_16.run(buf190, buf194, 65536, 2048, grid=grid(65536), stream=stream0)
        del buf190
        buf193 = empty_strided((8192, 512), (512, 1), device='cuda', dtype=torch.bfloat16)
        extern_kernels.mm(as_strided(buf149, (8192, 512), (512, 1)), as_strided(arg87_1, (512, 512), (1, 512)), out=buf193)
        del arg87_1
        buf195 = empty_strided((4, 8, 2048, 64), (1048576, 131072, 64, 1), device='cuda', dtype=torch.bfloat16)
        triton_poi_fused_clone_1.run(buf193, buf195, 4194304, grid=grid(4194304), stream=stream0)
        buf196 = empty_strided((32, 2048, 64), (131072, 64, 1), device='cuda', dtype=torch.bfloat16)
        extern_kernels.bmm(as_strided(buf194, (32, 2048, 2048), (4194304, 2048, 1)), as_strided(buf195, (32, 2048, 64), (131072, 64, 1)), out=buf196)
        del buf194
        del buf195
        buf197 = empty_strided((4, 2048, 8, 64), (1048576, 512, 64, 1), device='cuda', dtype=torch.bfloat16)
        triton_poi_fused_clone_4.run(buf196, buf197, 4194304, grid=grid(4194304), stream=stream0)
        del buf196
        buf198 = empty_strided((8192, 512), (512, 1), device='cuda', dtype=torch.bfloat16)
        extern_kernels.mm(as_strided(buf197, (8192, 512), (512, 1)), as_strided(arg88_1, (512, 512), (1, 512)), out=buf198)
        del arg88_1
        del buf197
        buf200 = empty_strided((4, 2048, 512), (1048576, 512, 1), device='cuda', dtype=torch.bfloat16)
        triton_per_fused__to_copy_add_mean_mul_pow_rsqrt_12.run(buf167, buf183, buf198, arg18_1, buf200, 8192, 512, grid=grid(8192), stream=stream0)
        del arg18_1
        buf201 = empty_strided((8192, 2048), (2048, 1), device='cuda', dtype=torch.bfloat16)
        extern_kernels.mm(as_strided(buf200, (8192, 512), (512, 1)), as_strided(arg89_1, (512, 2048), (1, 512)), out=buf201)
        del arg89_1
        del buf200
        buf202 = as_strided(buf201, (4, 2048, 2048), (4194304, 2048, 1)); del buf201  # reuse
        triton_poi_fused_relu_8.run(buf202, 16777216, grid=grid(16777216), stream=stream0)
        buf203 = empty_strided((8192, 512), (512, 1), device='cuda', dtype=torch.bfloat16)
        extern_kernels.mm(as_strided(buf202, (8192, 2048), (2048, 1)), as_strided(arg90_1, (2048, 512), (1, 2048)), out=buf203)
        del arg90_1
        del buf202
        buf205 = empty_strided((4, 2048, 512), (1048576, 512, 1), device='cuda', dtype=torch.bfloat16)
        triton_per_fused__to_copy_add_mean_mul_pow_rsqrt_13.run(buf167, buf183, buf198, buf203, arg19_1, buf205, 8192, 512, grid=grid(8192), stream=stream0)
        del arg19_1
        buf206 = empty_strided((8192, 512), (512, 1), device='cuda', dtype=torch.bfloat16)
        extern_kernels.mm(as_strided(buf205, (8192, 512), (512, 1)), as_strided(arg91_1, (512, 512), (1, 512)), out=buf206)
        del arg91_1
        buf207 = empty_strided((8192, 512), (512, 1), device='cuda', dtype=torch.bfloat16)
        extern_kernels.mm(as_strided(buf205, (8192, 512), (512, 1)), as_strided(arg92_1, (512, 512), (1, 512)), out=buf207)
        del arg92_1
        buf208 = empty_strided((4, 8, 2048, 64), (1048576, 131072, 64, 1), device='cuda', dtype=torch.bfloat16)
        triton_poi_fused_clone_1.run(buf206, buf208, 4194304, grid=grid(4194304), stream=stream0)
        del buf206
        buf209 = empty_strided((4, 8, 64, 2048), (1048576, 131072, 2048, 1), device='cuda', dtype=torch.bfloat16)
        triton_poi_fused_clone_2.run(buf207, buf209, 2048, 2048, grid=grid(2048, 2048), stream=stream0)
        buf210 = empty_strided((32, 2048, 2048), (4194304, 2048, 1), device='cuda', dtype=torch.bfloat16)
        extern_kernels.bmm(as_strided(buf208, (32, 2048, 64), (131072, 64, 1)), as_strided(buf209, (32, 64, 2048), (131072, 2048, 1)), out=buf210)
        del buf208
        del buf209
        buf211 = as_strided(buf210, (4, 8, 2048, 2048), (33554432, 4194304, 2048, 1)); del buf210  # reuse
        buf215 = empty_strided((4, 8, 2048, 2048), (33554432, 4194304, 2048, 1), device='cuda', dtype=torch.bfloat16)
        triton_red_fused__softmax__to_copy_add_mul_rsub_3.run(buf211, arg73_1, buf215, 65536, 2048, grid=grid(65536), stream=stream0)
        del buf211
        buf214 = empty_strided((8192, 512), (512, 1), device='cuda', dtype=torch.bfloat16)
        extern_kernels.mm(as_strided(buf205, (8192, 512), (512, 1)), as_strided(arg93_1, (512, 512), (1, 512)), out=buf214)
        del arg93_1
        del buf205
        buf216 = empty_strided((4, 8, 2048, 64), (1048576, 131072, 64, 1), device='cuda', dtype=torch.bfloat16)
        triton_poi_fused_clone_1.run(buf214, buf216, 4194304, grid=grid(4194304), stream=stream0)
        buf217 = empty_strided((32, 2048, 64), (131072, 64, 1), device='cuda', dtype=torch.bfloat16)
        extern_kernels.bmm(as_strided(buf215, (32, 2048, 2048), (4194304, 2048, 1)), as_strided(buf216, (32, 2048, 64), (131072, 64, 1)), out=buf217)
        del buf215
        del buf216
        buf218 = empty_strided((4, 2048, 8, 64), (1048576, 512, 64, 1), device='cuda', dtype=torch.bfloat16)
        triton_poi_fused_clone_4.run(buf217, buf218, 4194304, grid=grid(4194304), stream=stream0)
        del buf217
        buf219 = empty_strided((8192, 512), (512, 1), device='cuda', dtype=torch.bfloat16)
        extern_kernels.mm(as_strided(buf218, (8192, 512), (512, 1)), as_strided(arg94_1, (512, 512), (1, 512)), out=buf219)
        del arg94_1
        del buf218
        buf220 = buf167; del buf167  # reuse
        buf222 = empty_strided((4, 2048, 512), (1048576, 512, 1), device='cuda', dtype=torch.bfloat16)
        triton_per_fused__to_copy_add_mean_mul_pow_rsqrt_14.run(buf220, buf183, buf198, buf203, buf219, arg20_1, buf222, 8192, 512, grid=grid(8192), stream=stream0)
        del arg20_1
        del buf183
        del buf198
        del buf203
        del buf219
        buf223 = empty_strided((8192, 512), (512, 1), device='cuda', dtype=torch.bfloat16)
        extern_kernels.mm(as_strided(buf222, (8192, 512), (512, 1)), as_strided(arg95_1, (512, 512), (1, 512)), out=buf223)
        del arg95_1
        del buf222
        buf224 = empty_strided((8192, 512), (512, 1), device='cuda', dtype=torch.bfloat16)
        extern_kernels.mm(as_strided(buf149, (8192, 512), (512, 1)), as_strided(arg96_1, (512, 512), (1, 512)), out=buf224)
        del arg96_1
        buf225 = empty_strided((4, 8, 2048, 64), (1048576, 131072, 64, 1), device='cuda', dtype=torch.bfloat16)
        triton_poi_fused_clone_1.run(buf223, buf225, 4194304, grid=grid(4194304), stream=stream0)
        del buf223
        buf226 = empty_strided((4, 8, 64, 2048), (1048576, 131072, 2048, 1), device='cuda', dtype=torch.bfloat16)
        triton_poi_fused_clone_2.run(buf224, buf226, 2048, 2048, grid=grid(2048, 2048), stream=stream0)
        buf227 = empty_strided((32, 2048, 2048), (4194304, 2048, 1), device='cuda', dtype=torch.bfloat16)
        extern_kernels.bmm(as_strided(buf225, (32, 2048, 64), (131072, 64, 1)), as_strided(buf226, (32, 64, 2048), (131072, 2048, 1)), out=buf227)
        del buf225
        del buf226
        buf231 = empty_strided((4, 8, 2048, 2048), (33554432, 4194304, 2048, 1), device='cuda', dtype=torch.bfloat16)
        triton_red_fused__softmax__to_copy_16.run(buf227, buf231, 65536, 2048, grid=grid(65536), stream=stream0)
        del buf227
        buf230 = empty_strided((8192, 512), (512, 1), device='cuda', dtype=torch.bfloat16)
        extern_kernels.mm(as_strided(buf149, (8192, 512), (512, 1)), as_strided(arg97_1, (512, 512), (1, 512)), out=buf230)
        del arg97_1
        buf232 = empty_strided((4, 8, 2048, 64), (1048576, 131072, 64, 1), device='cuda', dtype=torch.bfloat16)
        triton_poi_fused_clone_1.run(buf230, buf232, 4194304, grid=grid(4194304), stream=stream0)
        buf233 = empty_strided((32, 2048, 64), (131072, 64, 1), device='cuda', dtype=torch.bfloat16)
        extern_kernels.bmm(as_strided(buf231, (32, 2048, 2048), (4194304, 2048, 1)), as_strided(buf232, (32, 2048, 64), (131072, 64, 1)), out=buf233)
        del buf231
        del buf232
        buf234 = empty_strided((4, 2048, 8, 64), (1048576, 512, 64, 1), device='cuda', dtype=torch.bfloat16)
        triton_poi_fused_clone_4.run(buf233, buf234, 4194304, grid=grid(4194304), stream=stream0)
        del buf233
        buf235 = empty_strided((8192, 512), (512, 1), device='cuda', dtype=torch.bfloat16)
        extern_kernels.mm(as_strided(buf234, (8192, 512), (512, 1)), as_strided(arg98_1, (512, 512), (1, 512)), out=buf235)
        del arg98_1
        del buf234
        buf237 = empty_strided((4, 2048, 512), (1048576, 512, 1), device='cuda', dtype=torch.bfloat16)
        triton_per_fused__to_copy_add_mean_mul_pow_rsqrt_11.run(buf220, buf235, arg21_1, buf237, 8192, 512, grid=grid(8192), stream=stream0)
        del arg21_1
        buf238 = empty_strided((8192, 2048), (2048, 1), device='cuda', dtype=torch.bfloat16)
        extern_kernels.mm(as_strided(buf237, (8192, 512), (512, 1)), as_strided(arg99_1, (512, 2048), (1, 512)), out=buf238)
        del arg99_1
        del buf237
        buf239 = as_strided(buf238, (4, 2048, 2048), (4194304, 2048, 1)); del buf238  # reuse
        triton_poi_fused_relu_8.run(buf239, 16777216, grid=grid(16777216), stream=stream0)
        buf240 = empty_strided((8192, 512), (512, 1), device='cuda', dtype=torch.bfloat16)
        extern_kernels.mm(as_strided(buf239, (8192, 2048), (2048, 1)), as_strided(arg100_1, (2048, 512), (1, 2048)), out=buf240)
        del arg100_1
        del buf239
        buf242 = empty_strided((4, 2048, 512), (1048576, 512, 1), device='cuda', dtype=torch.bfloat16)
        triton_per_fused__to_copy_add_mean_mul_pow_rsqrt_12.run(buf220, buf235, buf240, arg22_1, buf242, 8192, 512, grid=grid(8192), stream=stream0)
        del arg22_1
        buf243 = empty_strided((8192, 512), (512, 1), device='cuda', dtype=torch.bfloat16)
        extern_kernels.mm(as_strided(buf242, (8192, 512), (512, 1)), as_strided(arg101_1, (512, 512), (1, 512)), out=buf243)
        del arg101_1
        buf244 = empty_strided((8192, 512), (512, 1), device='cuda', dtype=torch.bfloat16)
        extern_kernels.mm(as_strided(buf242, (8192, 512), (512, 1)), as_strided(arg102_1, (512, 512), (1, 512)), out=buf244)
        del arg102_1
        buf245 = empty_strided((4, 8, 2048, 64), (1048576, 131072, 64, 1), device='cuda', dtype=torch.bfloat16)
        triton_poi_fused_clone_1.run(buf243, buf245, 4194304, grid=grid(4194304), stream=stream0)
        del buf243
        buf246 = empty_strided((4, 8, 64, 2048), (1048576, 131072, 2048, 1), device='cuda', dtype=torch.bfloat16)
        triton_poi_fused_clone_2.run(buf244, buf246, 2048, 2048, grid=grid(2048, 2048), stream=stream0)
        buf247 = empty_strided((32, 2048, 2048), (4194304, 2048, 1), device='cuda', dtype=torch.bfloat16)
        extern_kernels.bmm(as_strided(buf245, (32, 2048, 64), (131072, 64, 1)), as_strided(buf246, (32, 64, 2048), (131072, 2048, 1)), out=buf247)
        del buf245
        del buf246
        buf248 = as_strided(buf247, (4, 8, 2048, 2048), (33554432, 4194304, 2048, 1)); del buf247  # reuse
        buf252 = empty_strided((4, 8, 2048, 2048), (33554432, 4194304, 2048, 1), device='cuda', dtype=torch.bfloat16)
        triton_red_fused__softmax__to_copy_add_mul_rsub_3.run(buf248, arg73_1, buf252, 65536, 2048, grid=grid(65536), stream=stream0)
        del buf248
        buf251 = empty_strided((8192, 512), (512, 1), device='cuda', dtype=torch.bfloat16)
        extern_kernels.mm(as_strided(buf242, (8192, 512), (512, 1)), as_strided(arg103_1, (512, 512), (1, 512)), out=buf251)
        del arg103_1
        del buf242
        buf253 = empty_strided((4, 8, 2048, 64), (1048576, 131072, 64, 1), device='cuda', dtype=torch.bfloat16)
        triton_poi_fused_clone_1.run(buf251, buf253, 4194304, grid=grid(4194304), stream=stream0)
        buf254 = empty_strided((32, 2048, 64), (131072, 64, 1), device='cuda', dtype=torch.bfloat16)
        extern_kernels.bmm(as_strided(buf252, (32, 2048, 2048), (4194304, 2048, 1)), as_strided(buf253, (32, 2048, 64), (131072, 64, 1)), out=buf254)
        del buf252
        del buf253
        buf255 = empty_strided((4, 2048, 8, 64), (1048576, 512, 64, 1), device='cuda', dtype=torch.bfloat16)
        triton_poi_fused_clone_4.run(buf254, buf255, 4194304, grid=grid(4194304), stream=stream0)
        del buf254
        buf256 = empty_strided((8192, 512), (512, 1), device='cuda', dtype=torch.bfloat16)
        extern_kernels.mm(as_strided(buf255, (8192, 512), (512, 1)), as_strided(arg104_1, (512, 512), (1, 512)), out=buf256)
        del arg104_1
        del buf255
        buf258 = empty_strided((4, 2048, 512), (1048576, 512, 1), device='cuda', dtype=torch.bfloat16)
        triton_per_fused__to_copy_add_mean_mul_pow_rsqrt_13.run(buf220, buf235, buf240, buf256, arg23_1, buf258, 8192, 512, grid=grid(8192), stream=stream0)
        del arg23_1
        buf259 = empty_strided((8192, 512), (512, 1), device='cuda', dtype=torch.bfloat16)
        extern_kernels.mm(as_strided(buf258, (8192, 512), (512, 1)), as_strided(arg105_1, (512, 512), (1, 512)), out=buf259)
        del arg105_1
        del buf258
        buf260 = empty_strided((8192, 512), (512, 1), device='cuda', dtype=torch.bfloat16)
        extern_kernels.mm(as_strided(buf149, (8192, 512), (512, 1)), as_strided(arg106_1, (512, 512), (1, 512)), out=buf260)
        del arg106_1
        buf261 = empty_strided((4, 8, 2048, 64), (1048576, 131072, 64, 1), device='cuda', dtype=torch.bfloat16)
        triton_poi_fused_clone_1.run(buf259, buf261, 4194304, grid=grid(4194304), stream=stream0)
        del buf259
        buf262 = empty_strided((4, 8, 64, 2048), (1048576, 131072, 2048, 1), device='cuda', dtype=torch.bfloat16)
        triton_poi_fused_clone_2.run(buf260, buf262, 2048, 2048, grid=grid(2048, 2048), stream=stream0)
        buf263 = empty_strided((32, 2048, 2048), (4194304, 2048, 1), device='cuda', dtype=torch.bfloat16)
        extern_kernels.bmm(as_strided(buf261, (32, 2048, 64), (131072, 64, 1)), as_strided(buf262, (32, 64, 2048), (131072, 2048, 1)), out=buf263)
        del buf261
        del buf262
        buf267 = empty_strided((4, 8, 2048, 2048), (33554432, 4194304, 2048, 1), device='cuda', dtype=torch.bfloat16)
        triton_red_fused__softmax__to_copy_16.run(buf263, buf267, 65536, 2048, grid=grid(65536), stream=stream0)
        del buf263
        buf266 = empty_strided((8192, 512), (512, 1), device='cuda', dtype=torch.bfloat16)
        extern_kernels.mm(as_strided(buf149, (8192, 512), (512, 1)), as_strided(arg107_1, (512, 512), (1, 512)), out=buf266)
        del arg107_1
        buf268 = empty_strided((4, 8, 2048, 64), (1048576, 131072, 64, 1), device='cuda', dtype=torch.bfloat16)
        triton_poi_fused_clone_1.run(buf266, buf268, 4194304, grid=grid(4194304), stream=stream0)
        buf269 = empty_strided((32, 2048, 64), (131072, 64, 1), device='cuda', dtype=torch.bfloat16)
        extern_kernels.bmm(as_strided(buf267, (32, 2048, 2048), (4194304, 2048, 1)), as_strided(buf268, (32, 2048, 64), (131072, 64, 1)), out=buf269)
        del buf267
        del buf268
        buf270 = empty_strided((4, 2048, 8, 64), (1048576, 512, 64, 1), device='cuda', dtype=torch.bfloat16)
        triton_poi_fused_clone_4.run(buf269, buf270, 4194304, grid=grid(4194304), stream=stream0)
        del buf269
        buf271 = empty_strided((8192, 512), (512, 1), device='cuda', dtype=torch.bfloat16)
        extern_kernels.mm(as_strided(buf270, (8192, 512), (512, 1)), as_strided(arg108_1, (512, 512), (1, 512)), out=buf271)
        del arg108_1
        del buf270
        buf272 = buf220; del buf220  # reuse
        buf274 = empty_strided((4, 2048, 512), (1048576, 512, 1), device='cuda', dtype=torch.bfloat16)
        triton_per_fused__to_copy_add_mean_mul_pow_rsqrt_14.run(buf272, buf235, buf240, buf256, buf271, arg24_1, buf274, 8192, 512, grid=grid(8192), stream=stream0)
        del arg24_1
        del buf235
        del buf240
        del buf256
        del buf271
        buf275 = empty_strided((8192, 2048), (2048, 1), device='cuda', dtype=torch.bfloat16)
        extern_kernels.mm(as_strided(buf274, (8192, 512), (512, 1)), as_strided(arg109_1, (512, 2048), (1, 512)), out=buf275)
        del arg109_1
        del buf274
        buf276 = as_strided(buf275, (4, 2048, 2048), (4194304, 2048, 1)); del buf275  # reuse
        triton_poi_fused_relu_8.run(buf276, 16777216, grid=grid(16777216), stream=stream0)
        buf277 = empty_strided((8192, 512), (512, 1), device='cuda', dtype=torch.bfloat16)
        extern_kernels.mm(as_strided(buf276, (8192, 2048), (2048, 1)), as_strided(arg110_1, (2048, 512), (1, 2048)), out=buf277)
        del arg110_1
        del buf276
        buf279 = empty_strided((4, 2048, 512), (1048576, 512, 1), device='cuda', dtype=torch.bfloat16)
        triton_per_fused__to_copy_add_mean_mul_pow_rsqrt_11.run(buf272, buf277, arg25_1, buf279, 8192, 512, grid=grid(8192), stream=stream0)
        del arg25_1
        buf280 = empty_strided((8192, 512), (512, 1), device='cuda', dtype=torch.bfloat16)
        extern_kernels.mm(as_strided(buf279, (8192, 512), (512, 1)), as_strided(arg111_1, (512, 512), (1, 512)), out=buf280)
        del arg111_1
        buf281 = empty_strided((8192, 512), (512, 1), device='cuda', dtype=torch.bfloat16)
        extern_kernels.mm(as_strided(buf279, (8192, 512), (512, 1)), as_strided(arg112_1, (512, 512), (1, 512)), out=buf281)
        del arg112_1
        buf282 = empty_strided((4, 8, 2048, 64), (1048576, 131072, 64, 1), device='cuda', dtype=torch.bfloat16)
        triton_poi_fused_clone_1.run(buf280, buf282, 4194304, grid=grid(4194304), stream=stream0)
        del buf280
        buf283 = empty_strided((4, 8, 64, 2048), (1048576, 131072, 2048, 1), device='cuda', dtype=torch.bfloat16)
        triton_poi_fused_clone_2.run(buf281, buf283, 2048, 2048, grid=grid(2048, 2048), stream=stream0)
        buf284 = empty_strided((32, 2048, 2048), (4194304, 2048, 1), device='cuda', dtype=torch.bfloat16)
        extern_kernels.bmm(as_strided(buf282, (32, 2048, 64), (131072, 64, 1)), as_strided(buf283, (32, 64, 2048), (131072, 2048, 1)), out=buf284)
        del buf282
        del buf283
        buf285 = as_strided(buf284, (4, 8, 2048, 2048), (33554432, 4194304, 2048, 1)); del buf284  # reuse
        buf289 = empty_strided((4, 8, 2048, 2048), (33554432, 4194304, 2048, 1), device='cuda', dtype=torch.bfloat16)
        triton_red_fused__softmax__to_copy_add_mul_rsub_3.run(buf285, arg73_1, buf289, 65536, 2048, grid=grid(65536), stream=stream0)
        del buf285
        buf288 = empty_strided((8192, 512), (512, 1), device='cuda', dtype=torch.bfloat16)
        extern_kernels.mm(as_strided(buf279, (8192, 512), (512, 1)), as_strided(arg113_1, (512, 512), (1, 512)), out=buf288)
        del arg113_1
        del buf279
        buf290 = empty_strided((4, 8, 2048, 64), (1048576, 131072, 64, 1), device='cuda', dtype=torch.bfloat16)
        triton_poi_fused_clone_1.run(buf288, buf290, 4194304, grid=grid(4194304), stream=stream0)
        buf291 = empty_strided((32, 2048, 64), (131072, 64, 1), device='cuda', dtype=torch.bfloat16)
        extern_kernels.bmm(as_strided(buf289, (32, 2048, 2048), (4194304, 2048, 1)), as_strided(buf290, (32, 2048, 64), (131072, 64, 1)), out=buf291)
        del buf289
        del buf290
        buf292 = empty_strided((4, 2048, 8, 64), (1048576, 512, 64, 1), device='cuda', dtype=torch.bfloat16)
        triton_poi_fused_clone_4.run(buf291, buf292, 4194304, grid=grid(4194304), stream=stream0)
        del buf291
        buf293 = empty_strided((8192, 512), (512, 1), device='cuda', dtype=torch.bfloat16)
        extern_kernels.mm(as_strided(buf292, (8192, 512), (512, 1)), as_strided(arg114_1, (512, 512), (1, 512)), out=buf293)
        del arg114_1
        del buf292
        buf295 = empty_strided((4, 2048, 512), (1048576, 512, 1), device='cuda', dtype=torch.bfloat16)
        triton_per_fused__to_copy_add_mean_mul_pow_rsqrt_12.run(buf272, buf277, buf293, arg26_1, buf295, 8192, 512, grid=grid(8192), stream=stream0)
        del arg26_1
        buf296 = empty_strided((8192, 512), (512, 1), device='cuda', dtype=torch.bfloat16)
        extern_kernels.mm(as_strided(buf295, (8192, 512), (512, 1)), as_strided(arg115_1, (512, 512), (1, 512)), out=buf296)
        del arg115_1
        del buf295
        buf297 = empty_strided((8192, 512), (512, 1), device='cuda', dtype=torch.bfloat16)
        extern_kernels.mm(as_strided(buf149, (8192, 512), (512, 1)), as_strided(arg116_1, (512, 512), (1, 512)), out=buf297)
        del arg116_1
        buf298 = empty_strided((4, 8, 2048, 64), (1048576, 131072, 64, 1), device='cuda', dtype=torch.bfloat16)
        triton_poi_fused_clone_1.run(buf296, buf298, 4194304, grid=grid(4194304), stream=stream0)
        del buf296
        buf299 = empty_strided((4, 8, 64, 2048), (1048576, 131072, 2048, 1), device='cuda', dtype=torch.bfloat16)
        triton_poi_fused_clone_2.run(buf297, buf299, 2048, 2048, grid=grid(2048, 2048), stream=stream0)
        buf300 = empty_strided((32, 2048, 2048), (4194304, 2048, 1), device='cuda', dtype=torch.bfloat16)
        extern_kernels.bmm(as_strided(buf298, (32, 2048, 64), (131072, 64, 1)), as_strided(buf299, (32, 64, 2048), (131072, 2048, 1)), out=buf300)
        del buf298
        del buf299
        buf304 = empty_strided((4, 8, 2048, 2048), (33554432, 4194304, 2048, 1), device='cuda', dtype=torch.bfloat16)
        triton_red_fused__softmax__to_copy_16.run(buf300, buf304, 65536, 2048, grid=grid(65536), stream=stream0)
        del buf300
        buf303 = empty_strided((8192, 512), (512, 1), device='cuda', dtype=torch.bfloat16)
        extern_kernels.mm(as_strided(buf149, (8192, 512), (512, 1)), as_strided(arg117_1, (512, 512), (1, 512)), out=buf303)
        del arg117_1
        buf305 = empty_strided((4, 8, 2048, 64), (1048576, 131072, 64, 1), device='cuda', dtype=torch.bfloat16)
        triton_poi_fused_clone_1.run(buf303, buf305, 4194304, grid=grid(4194304), stream=stream0)
        buf306 = empty_strided((32, 2048, 64), (131072, 64, 1), device='cuda', dtype=torch.bfloat16)
        extern_kernels.bmm(as_strided(buf304, (32, 2048, 2048), (4194304, 2048, 1)), as_strided(buf305, (32, 2048, 64), (131072, 64, 1)), out=buf306)
        del buf304
        del buf305
        buf307 = empty_strided((4, 2048, 8, 64), (1048576, 512, 64, 1), device='cuda', dtype=torch.bfloat16)
        triton_poi_fused_clone_4.run(buf306, buf307, 4194304, grid=grid(4194304), stream=stream0)
        del buf306
        buf308 = empty_strided((8192, 512), (512, 1), device='cuda', dtype=torch.bfloat16)
        extern_kernels.mm(as_strided(buf307, (8192, 512), (512, 1)), as_strided(arg118_1, (512, 512), (1, 512)), out=buf308)
        del arg118_1
        del buf307
        buf310 = empty_strided((4, 2048, 512), (1048576, 512, 1), device='cuda', dtype=torch.bfloat16)
        triton_per_fused__to_copy_add_mean_mul_pow_rsqrt_13.run(buf272, buf277, buf293, buf308, arg27_1, buf310, 8192, 512, grid=grid(8192), stream=stream0)
        del arg27_1
        buf311 = empty_strided((8192, 2048), (2048, 1), device='cuda', dtype=torch.bfloat16)
        extern_kernels.mm(as_strided(buf310, (8192, 512), (512, 1)), as_strided(arg119_1, (512, 2048), (1, 512)), out=buf311)
        del arg119_1
        del buf310
        buf312 = as_strided(buf311, (4, 2048, 2048), (4194304, 2048, 1)); del buf311  # reuse
        triton_poi_fused_relu_8.run(buf312, 16777216, grid=grid(16777216), stream=stream0)
        buf313 = empty_strided((8192, 512), (512, 1), device='cuda', dtype=torch.bfloat16)
        extern_kernels.mm(as_strided(buf312, (8192, 2048), (2048, 1)), as_strided(arg120_1, (2048, 512), (1, 2048)), out=buf313)
        del arg120_1
        del buf312
        buf314 = buf272; del buf272  # reuse
        buf316 = empty_strided((4, 2048, 512), (1048576, 512, 1), device='cuda', dtype=torch.bfloat16)
        triton_per_fused__to_copy_add_mean_mul_pow_rsqrt_14.run(buf314, buf277, buf293, buf308, buf313, arg28_1, buf316, 8192, 512, grid=grid(8192), stream=stream0)
        del arg28_1
        del buf277
        del buf293
        del buf308
        del buf313
        buf317 = empty_strided((8192, 512), (512, 1), device='cuda', dtype=torch.bfloat16)
        extern_kernels.mm(as_strided(buf316, (8192, 512), (512, 1)), as_strided(arg121_1, (512, 512), (1, 512)), out=buf317)
        del arg121_1
        buf318 = empty_strided((8192, 512), (512, 1), device='cuda', dtype=torch.bfloat16)
        extern_kernels.mm(as_strided(buf316, (8192, 512), (512, 1)), as_strided(arg122_1, (512, 512), (1, 512)), out=buf318)
        del arg122_1
        buf319 = empty_strided((4, 8, 2048, 64), (1048576, 131072, 64, 1), device='cuda', dtype=torch.bfloat16)
        triton_poi_fused_clone_1.run(buf317, buf319, 4194304, grid=grid(4194304), stream=stream0)
        del buf317
        buf320 = empty_strided((4, 8, 64, 2048), (1048576, 131072, 2048, 1), device='cuda', dtype=torch.bfloat16)
        triton_poi_fused_clone_2.run(buf318, buf320, 2048, 2048, grid=grid(2048, 2048), stream=stream0)
        buf321 = empty_strided((32, 2048, 2048), (4194304, 2048, 1), device='cuda', dtype=torch.bfloat16)
        extern_kernels.bmm(as_strided(buf319, (32, 2048, 64), (131072, 64, 1)), as_strided(buf320, (32, 64, 2048), (131072, 2048, 1)), out=buf321)
        del buf319
        del buf320
        buf322 = as_strided(buf321, (4, 8, 2048, 2048), (33554432, 4194304, 2048, 1)); del buf321  # reuse
        buf326 = empty_strided((4, 8, 2048, 2048), (33554432, 4194304, 2048, 1), device='cuda', dtype=torch.bfloat16)
        triton_red_fused__softmax__to_copy_add_mul_rsub_3.run(buf322, arg73_1, buf326, 65536, 2048, grid=grid(65536), stream=stream0)
        del arg73_1
        del buf322
        buf325 = empty_strided((8192, 512), (512, 1), device='cuda', dtype=torch.bfloat16)
        extern_kernels.mm(as_strided(buf316, (8192, 512), (512, 1)), as_strided(arg123_1, (512, 512), (1, 512)), out=buf325)
        del arg123_1
        del buf316
        buf327 = empty_strided((4, 8, 2048, 64), (1048576, 131072, 64, 1), device='cuda', dtype=torch.bfloat16)
        triton_poi_fused_clone_1.run(buf325, buf327, 4194304, grid=grid(4194304), stream=stream0)
        buf328 = empty_strided((32, 2048, 64), (131072, 64, 1), device='cuda', dtype=torch.bfloat16)
        extern_kernels.bmm(as_strided(buf326, (32, 2048, 2048), (4194304, 2048, 1)), as_strided(buf327, (32, 2048, 64), (131072, 64, 1)), out=buf328)
        del buf326
        del buf327
        buf329 = empty_strided((4, 2048, 8, 64), (1048576, 512, 64, 1), device='cuda', dtype=torch.bfloat16)
        triton_poi_fused_clone_4.run(buf328, buf329, 4194304, grid=grid(4194304), stream=stream0)
        del buf328
        buf330 = empty_strided((8192, 512), (512, 1), device='cuda', dtype=torch.bfloat16)
        extern_kernels.mm(as_strided(buf329, (8192, 512), (512, 1)), as_strided(arg124_1, (512, 512), (1, 512)), out=buf330)
        del arg124_1
        del buf329
        buf332 = empty_strided((4, 2048, 512), (1048576, 512, 1), device='cuda', dtype=torch.bfloat16)
        triton_per_fused__to_copy_add_mean_mul_pow_rsqrt_11.run(buf314, buf330, arg29_1, buf332, 8192, 512, grid=grid(8192), stream=stream0)
        del arg29_1
        buf333 = empty_strided((8192, 512), (512, 1), device='cuda', dtype=torch.bfloat16)
        extern_kernels.mm(as_strided(buf332, (8192, 512), (512, 1)), as_strided(arg125_1, (512, 512), (1, 512)), out=buf333)
        del arg125_1
        del buf332
        buf334 = empty_strided((8192, 512), (512, 1), device='cuda', dtype=torch.bfloat16)
        extern_kernels.mm(as_strided(buf149, (8192, 512), (512, 1)), as_strided(arg126_1, (512, 512), (1, 512)), out=buf334)
        del arg126_1
        buf335 = empty_strided((4, 8, 2048, 64), (1048576, 131072, 64, 1), device='cuda', dtype=torch.bfloat16)
        triton_poi_fused_clone_1.run(buf333, buf335, 4194304, grid=grid(4194304), stream=stream0)
        del buf333
        buf336 = empty_strided((4, 8, 64, 2048), (1048576, 131072, 2048, 1), device='cuda', dtype=torch.bfloat16)
        triton_poi_fused_clone_2.run(buf334, buf336, 2048, 2048, grid=grid(2048, 2048), stream=stream0)
        buf337 = empty_strided((32, 2048, 2048), (4194304, 2048, 1), device='cuda', dtype=torch.bfloat16)
        extern_kernels.bmm(as_strided(buf335, (32, 2048, 64), (131072, 64, 1)), as_strided(buf336, (32, 64, 2048), (131072, 2048, 1)), out=buf337)
        del buf335
        del buf336
        buf341 = empty_strided((4, 8, 2048, 2048), (33554432, 4194304, 2048, 1), device='cuda', dtype=torch.bfloat16)
        triton_red_fused__softmax__to_copy_16.run(buf337, buf341, 65536, 2048, grid=grid(65536), stream=stream0)
        del buf337
        buf340 = empty_strided((8192, 512), (512, 1), device='cuda', dtype=torch.bfloat16)
        extern_kernels.mm(as_strided(buf149, (8192, 512), (512, 1)), as_strided(arg127_1, (512, 512), (1, 512)), out=buf340)
        del arg127_1
        buf342 = empty_strided((4, 8, 2048, 64), (1048576, 131072, 64, 1), device='cuda', dtype=torch.bfloat16)
        triton_poi_fused_clone_1.run(buf340, buf342, 4194304, grid=grid(4194304), stream=stream0)
        buf343 = empty_strided((32, 2048, 64), (131072, 64, 1), device='cuda', dtype=torch.bfloat16)
        extern_kernels.bmm(as_strided(buf341, (32, 2048, 2048), (4194304, 2048, 1)), as_strided(buf342, (32, 2048, 64), (131072, 64, 1)), out=buf343)
        del buf341
        del buf342
        buf344 = empty_strided((4, 2048, 8, 64), (1048576, 512, 64, 1), device='cuda', dtype=torch.bfloat16)
        triton_poi_fused_clone_4.run(buf343, buf344, 4194304, grid=grid(4194304), stream=stream0)
        del buf343
        buf345 = empty_strided((8192, 512), (512, 1), device='cuda', dtype=torch.bfloat16)
        extern_kernels.mm(as_strided(buf344, (8192, 512), (512, 1)), as_strided(arg128_1, (512, 512), (1, 512)), out=buf345)
        del arg128_1
        del buf344
        buf347 = empty_strided((4, 2048, 512), (1048576, 512, 1), device='cuda', dtype=torch.bfloat16)
        triton_per_fused__to_copy_add_mean_mul_pow_rsqrt_12.run(buf314, buf330, buf345, arg30_1, buf347, 8192, 512, grid=grid(8192), stream=stream0)
        del arg30_1
        buf348 = empty_strided((8192, 2048), (2048, 1), device='cuda', dtype=torch.bfloat16)
        extern_kernels.mm(as_strided(buf347, (8192, 512), (512, 1)), as_strided(arg129_1, (512, 2048), (1, 512)), out=buf348)
        del arg129_1
        del buf347
        buf349 = as_strided(buf348, (4, 2048, 2048), (4194304, 2048, 1)); del buf348  # reuse
        triton_poi_fused_relu_8.run(buf349, 16777216, grid=grid(16777216), stream=stream0)
        buf350 = empty_strided((8192, 512), (512, 1), device='cuda', dtype=torch.bfloat16)
        extern_kernels.mm(as_strided(buf349, (8192, 2048), (2048, 1)), as_strided(arg130_1, (2048, 512), (1, 2048)), out=buf350)
        del arg130_1
        del buf349
        buf352 = empty_strided((4, 2048, 512), (1048576, 512, 1), device='cuda', dtype=torch.bfloat16)
        triton_per_fused__to_copy_add_mean_mul_pow_rsqrt_17.run(buf314, buf330, buf345, buf350, arg31_1, buf352, 8192, 512, grid=grid(8192), stream=stream0)
        del arg31_1
        del buf314
        del buf330
        del buf345
        del buf350
        buf353 = empty_strided((8192, 32128), (32128, 1), device='cuda', dtype=torch.bfloat16)
        extern_kernels.mm(as_strided(buf352, (8192, 512), (512, 1)), as_strided(arg131_1, (512, 32128), (1, 512)), out=buf353)
        del arg131_1
        return (as_strided(buf353, (4, 2048, 32128), (65798144, 32128, 1)), as_strided(buf3, (4, 8, 2048, 64), (1048576, 64, 512, 1)), as_strided(buf10, (4, 8, 2048, 64), (1048576, 64, 512, 1)), as_strided(buf150, (4, 8, 2048, 64), (1048576, 64, 512, 1)), as_strided(buf156, (4, 8, 2048, 64), (1048576, 64, 512, 1)), as_strided(buf171, (4, 8, 2048, 64), (1048576, 64, 512, 1)), as_strided(buf178, (4, 8, 2048, 64), (1048576, 64, 512, 1)), as_strided(buf187, (4, 8, 2048, 64), (1048576, 64, 512, 1)), as_strided(buf193, (4, 8, 2048, 64), (1048576, 64, 512, 1)), as_strided(buf207, (4, 8, 2048, 64), (1048576, 64, 512, 1)), as_strided(buf214, (4, 8, 2048, 64), (1048576, 64, 512, 1)), as_strided(buf224, (4, 8, 2048, 64), (1048576, 64, 512, 1)), as_strided(buf230, (4, 8, 2048, 64), (1048576, 64, 512, 1)), as_strided(buf244, (4, 8, 2048, 64), (1048576, 64, 512, 1)), as_strided(buf251, (4, 8, 2048, 64), (1048576, 64, 512, 1)), as_strided(buf260, (4, 8, 2048, 64), (1048576, 64, 512, 1)), as_strided(buf266, (4, 8, 2048, 64), (1048576, 64, 512, 1)), as_strided(buf281, (4, 8, 2048, 64), (1048576, 64, 512, 1)), as_strided(buf288, (4, 8, 2048, 64), (1048576, 64, 512, 1)), as_strided(buf297, (4, 8, 2048, 64), (1048576, 64, 512, 1)), as_strided(buf303, (4, 8, 2048, 64), (1048576, 64, 512, 1)), as_strided(buf318, (4, 8, 2048, 64), (1048576, 64, 512, 1)), as_strided(buf325, (4, 8, 2048, 64), (1048576, 64, 512, 1)), as_strided(buf334, (4, 8, 2048, 64), (1048576, 64, 512, 1)), as_strided(buf340, (4, 8, 2048, 64), (1048576, 64, 512, 1)), buf149, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg1_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg2_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg3_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg4_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg5_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg6_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg7_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg8_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg9_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg10_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg11_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg12_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg13_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg14_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg15_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg16_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg17_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg18_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg19_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg20_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg21_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg22_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg23_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg24_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg25_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg26_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg27_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg28_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg29_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg30_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg31_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg32_1 = rand_strided((32128, 512), (512, 1), device='cuda:0', dtype=torch.bfloat16)
    arg33_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.bfloat16)
    arg34_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.bfloat16)
    arg35_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.bfloat16)
    arg36_1 = rand_strided((32, 8), (8, 1), device='cuda:0', dtype=torch.bfloat16)
    arg37_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.bfloat16)
    arg38_1 = rand_strided((2048, 512), (512, 1), device='cuda:0', dtype=torch.bfloat16)
    arg39_1 = rand_strided((512, 2048), (2048, 1), device='cuda:0', dtype=torch.bfloat16)
    arg40_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.bfloat16)
    arg41_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.bfloat16)
    arg42_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.bfloat16)
    arg43_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.bfloat16)
    arg44_1 = rand_strided((2048, 512), (512, 1), device='cuda:0', dtype=torch.bfloat16)
    arg45_1 = rand_strided((512, 2048), (2048, 1), device='cuda:0', dtype=torch.bfloat16)
    arg46_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.bfloat16)
    arg47_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.bfloat16)
    arg48_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.bfloat16)
    arg49_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.bfloat16)
    arg50_1 = rand_strided((2048, 512), (512, 1), device='cuda:0', dtype=torch.bfloat16)
    arg51_1 = rand_strided((512, 2048), (2048, 1), device='cuda:0', dtype=torch.bfloat16)
    arg52_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.bfloat16)
    arg53_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.bfloat16)
    arg54_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.bfloat16)
    arg55_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.bfloat16)
    arg56_1 = rand_strided((2048, 512), (512, 1), device='cuda:0', dtype=torch.bfloat16)
    arg57_1 = rand_strided((512, 2048), (2048, 1), device='cuda:0', dtype=torch.bfloat16)
    arg58_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.bfloat16)
    arg59_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.bfloat16)
    arg60_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.bfloat16)
    arg61_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.bfloat16)
    arg62_1 = rand_strided((2048, 512), (512, 1), device='cuda:0', dtype=torch.bfloat16)
    arg63_1 = rand_strided((512, 2048), (2048, 1), device='cuda:0', dtype=torch.bfloat16)
    arg64_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.bfloat16)
    arg65_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.bfloat16)
    arg66_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.bfloat16)
    arg67_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.bfloat16)
    arg68_1 = rand_strided((2048, 512), (512, 1), device='cuda:0', dtype=torch.bfloat16)
    arg69_1 = rand_strided((512, 2048), (2048, 1), device='cuda:0', dtype=torch.bfloat16)
    arg70_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.bfloat16)
    arg71_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.bfloat16)
    arg72_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.bfloat16)
    arg73_1 = rand_strided((32, 8), (8, 1), device='cuda:0', dtype=torch.bfloat16)
    arg74_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.bfloat16)
    arg75_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.bfloat16)
    arg76_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.bfloat16)
    arg77_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.bfloat16)
    arg78_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.bfloat16)
    arg79_1 = rand_strided((2048, 512), (512, 1), device='cuda:0', dtype=torch.bfloat16)
    arg80_1 = rand_strided((512, 2048), (2048, 1), device='cuda:0', dtype=torch.bfloat16)
    arg81_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.bfloat16)
    arg82_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.bfloat16)
    arg83_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.bfloat16)
    arg84_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.bfloat16)
    arg85_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.bfloat16)
    arg86_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.bfloat16)
    arg87_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.bfloat16)
    arg88_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.bfloat16)
    arg89_1 = rand_strided((2048, 512), (512, 1), device='cuda:0', dtype=torch.bfloat16)
    arg90_1 = rand_strided((512, 2048), (2048, 1), device='cuda:0', dtype=torch.bfloat16)
    arg91_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.bfloat16)
    arg92_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.bfloat16)
    arg93_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.bfloat16)
    arg94_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.bfloat16)
    arg95_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.bfloat16)
    arg96_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.bfloat16)
    arg97_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.bfloat16)
    arg98_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.bfloat16)
    arg99_1 = rand_strided((2048, 512), (512, 1), device='cuda:0', dtype=torch.bfloat16)
    arg100_1 = rand_strided((512, 2048), (2048, 1), device='cuda:0', dtype=torch.bfloat16)
    arg101_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.bfloat16)
    arg102_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.bfloat16)
    arg103_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.bfloat16)
    arg104_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.bfloat16)
    arg105_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.bfloat16)
    arg106_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.bfloat16)
    arg107_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.bfloat16)
    arg108_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.bfloat16)
    arg109_1 = rand_strided((2048, 512), (512, 1), device='cuda:0', dtype=torch.bfloat16)
    arg110_1 = rand_strided((512, 2048), (2048, 1), device='cuda:0', dtype=torch.bfloat16)
    arg111_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.bfloat16)
    arg112_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.bfloat16)
    arg113_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.bfloat16)
    arg114_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.bfloat16)
    arg115_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.bfloat16)
    arg116_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.bfloat16)
    arg117_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.bfloat16)
    arg118_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.bfloat16)
    arg119_1 = rand_strided((2048, 512), (512, 1), device='cuda:0', dtype=torch.bfloat16)
    arg120_1 = rand_strided((512, 2048), (2048, 1), device='cuda:0', dtype=torch.bfloat16)
    arg121_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.bfloat16)
    arg122_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.bfloat16)
    arg123_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.bfloat16)
    arg124_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.bfloat16)
    arg125_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.bfloat16)
    arg126_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.bfloat16)
    arg127_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.bfloat16)
    arg128_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.bfloat16)
    arg129_1 = rand_strided((2048, 512), (512, 1), device='cuda:0', dtype=torch.bfloat16)
    arg130_1 = rand_strided((512, 2048), (2048, 1), device='cuda:0', dtype=torch.bfloat16)
    arg131_1 = rand_strided((32128, 512), (512, 1), device='cuda:0', dtype=torch.bfloat16)
    arg132_1 = rand_strided((4, 2048), (2048, 1), device='cuda:0', dtype=torch.int64)
    arg133_1 = rand_strided((4, 2048), (2048, 1), device='cuda:0', dtype=torch.int64)
    arg0_1_clone = arg0_1.clone()
    arg1_1_clone = arg1_1.clone()
    arg2_1_clone = arg2_1.clone()
    arg3_1_clone = arg3_1.clone()
    arg4_1_clone = arg4_1.clone()
    arg5_1_clone = arg5_1.clone()
    arg6_1_clone = arg6_1.clone()
    arg7_1_clone = arg7_1.clone()
    arg8_1_clone = arg8_1.clone()
    arg9_1_clone = arg9_1.clone()
    arg10_1_clone = arg10_1.clone()
    arg11_1_clone = arg11_1.clone()
    arg12_1_clone = arg12_1.clone()
    arg13_1_clone = arg13_1.clone()
    arg14_1_clone = arg14_1.clone()
    arg15_1_clone = arg15_1.clone()
    arg16_1_clone = arg16_1.clone()
    arg17_1_clone = arg17_1.clone()
    arg18_1_clone = arg18_1.clone()
    arg19_1_clone = arg19_1.clone()
    arg20_1_clone = arg20_1.clone()
    arg21_1_clone = arg21_1.clone()
    arg22_1_clone = arg22_1.clone()
    arg23_1_clone = arg23_1.clone()
    arg24_1_clone = arg24_1.clone()
    arg25_1_clone = arg25_1.clone()
    arg26_1_clone = arg26_1.clone()
    arg27_1_clone = arg27_1.clone()
    arg28_1_clone = arg28_1.clone()
    arg29_1_clone = arg29_1.clone()
    arg30_1_clone = arg30_1.clone()
    arg31_1_clone = arg31_1.clone()
    arg32_1_clone = arg32_1.clone()
    arg33_1_clone = arg33_1.clone()
    arg34_1_clone = arg34_1.clone()
    arg35_1_clone = arg35_1.clone()
    arg36_1_clone = arg36_1.clone()
    arg37_1_clone = arg37_1.clone()
    arg38_1_clone = arg38_1.clone()
    arg39_1_clone = arg39_1.clone()
    arg40_1_clone = arg40_1.clone()
    arg41_1_clone = arg41_1.clone()
    arg42_1_clone = arg42_1.clone()
    arg43_1_clone = arg43_1.clone()
    arg44_1_clone = arg44_1.clone()
    arg45_1_clone = arg45_1.clone()
    arg46_1_clone = arg46_1.clone()
    arg47_1_clone = arg47_1.clone()
    arg48_1_clone = arg48_1.clone()
    arg49_1_clone = arg49_1.clone()
    arg50_1_clone = arg50_1.clone()
    arg51_1_clone = arg51_1.clone()
    arg52_1_clone = arg52_1.clone()
    arg53_1_clone = arg53_1.clone()
    arg54_1_clone = arg54_1.clone()
    arg55_1_clone = arg55_1.clone()
    arg56_1_clone = arg56_1.clone()
    arg57_1_clone = arg57_1.clone()
    arg58_1_clone = arg58_1.clone()
    arg59_1_clone = arg59_1.clone()
    arg60_1_clone = arg60_1.clone()
    arg61_1_clone = arg61_1.clone()
    arg62_1_clone = arg62_1.clone()
    arg63_1_clone = arg63_1.clone()
    arg64_1_clone = arg64_1.clone()
    arg65_1_clone = arg65_1.clone()
    arg66_1_clone = arg66_1.clone()
    arg67_1_clone = arg67_1.clone()
    arg68_1_clone = arg68_1.clone()
    arg69_1_clone = arg69_1.clone()
    arg70_1_clone = arg70_1.clone()
    arg71_1_clone = arg71_1.clone()
    arg72_1_clone = arg72_1.clone()
    arg73_1_clone = arg73_1.clone()
    arg74_1_clone = arg74_1.clone()
    arg75_1_clone = arg75_1.clone()
    arg76_1_clone = arg76_1.clone()
    arg77_1_clone = arg77_1.clone()
    arg78_1_clone = arg78_1.clone()
    arg79_1_clone = arg79_1.clone()
    arg80_1_clone = arg80_1.clone()
    arg81_1_clone = arg81_1.clone()
    arg82_1_clone = arg82_1.clone()
    arg83_1_clone = arg83_1.clone()
    arg84_1_clone = arg84_1.clone()
    arg85_1_clone = arg85_1.clone()
    arg86_1_clone = arg86_1.clone()
    arg87_1_clone = arg87_1.clone()
    arg88_1_clone = arg88_1.clone()
    arg89_1_clone = arg89_1.clone()
    arg90_1_clone = arg90_1.clone()
    arg91_1_clone = arg91_1.clone()
    arg92_1_clone = arg92_1.clone()
    arg93_1_clone = arg93_1.clone()
    arg94_1_clone = arg94_1.clone()
    arg95_1_clone = arg95_1.clone()
    arg96_1_clone = arg96_1.clone()
    arg97_1_clone = arg97_1.clone()
    arg98_1_clone = arg98_1.clone()
    arg99_1_clone = arg99_1.clone()
    arg100_1_clone = arg100_1.clone()
    arg101_1_clone = arg101_1.clone()
    arg102_1_clone = arg102_1.clone()
    arg103_1_clone = arg103_1.clone()
    arg104_1_clone = arg104_1.clone()
    arg105_1_clone = arg105_1.clone()
    arg106_1_clone = arg106_1.clone()
    arg107_1_clone = arg107_1.clone()
    arg108_1_clone = arg108_1.clone()
    arg109_1_clone = arg109_1.clone()
    arg110_1_clone = arg110_1.clone()
    arg111_1_clone = arg111_1.clone()
    arg112_1_clone = arg112_1.clone()
    arg113_1_clone = arg113_1.clone()
    arg114_1_clone = arg114_1.clone()
    arg115_1_clone = arg115_1.clone()
    arg116_1_clone = arg116_1.clone()
    arg117_1_clone = arg117_1.clone()
    arg118_1_clone = arg118_1.clone()
    arg119_1_clone = arg119_1.clone()
    arg120_1_clone = arg120_1.clone()
    arg121_1_clone = arg121_1.clone()
    arg122_1_clone = arg122_1.clone()
    arg123_1_clone = arg123_1.clone()
    arg124_1_clone = arg124_1.clone()
    arg125_1_clone = arg125_1.clone()
    arg126_1_clone = arg126_1.clone()
    arg127_1_clone = arg127_1.clone()
    arg128_1_clone = arg128_1.clone()
    arg129_1_clone = arg129_1.clone()
    arg130_1_clone = arg130_1.clone()
    arg131_1_clone = arg131_1.clone()
    arg132_1_clone = arg132_1.clone()
    arg133_1_clone = arg133_1.clone()
    from output_code import call as call_origin
    results_origin = call_origin(    [arg0_1_clone, arg1_1_clone, arg2_1_clone, arg3_1_clone, arg4_1_clone, arg5_1_clone, arg6_1_clone, arg7_1_clone, arg8_1_clone, arg9_1_clone, arg10_1_clone, arg11_1_clone, arg12_1_clone, arg13_1_clone, arg14_1_clone, arg15_1_clone, arg16_1_clone, arg17_1_clone, arg18_1_clone, arg19_1_clone, arg20_1_clone, arg21_1_clone, arg22_1_clone, arg23_1_clone, arg24_1_clone, arg25_1_clone, arg26_1_clone, arg27_1_clone, arg28_1_clone, arg29_1_clone, arg30_1_clone, arg31_1_clone, arg32_1_clone, arg33_1_clone, arg34_1_clone, arg35_1_clone, arg36_1_clone, arg37_1_clone, arg38_1_clone, arg39_1_clone, arg40_1_clone, arg41_1_clone, arg42_1_clone, arg43_1_clone, arg44_1_clone, arg45_1_clone, arg46_1_clone, arg47_1_clone, arg48_1_clone, arg49_1_clone, arg50_1_clone, arg51_1_clone, arg52_1_clone, arg53_1_clone, arg54_1_clone, arg55_1_clone, arg56_1_clone, arg57_1_clone, arg58_1_clone, arg59_1_clone, arg60_1_clone, arg61_1_clone, arg62_1_clone, arg63_1_clone, arg64_1_clone, arg65_1_clone, arg66_1_clone, arg67_1_clone, arg68_1_clone, arg69_1_clone, arg70_1_clone, arg71_1_clone, arg72_1_clone, arg73_1_clone, arg74_1_clone, arg75_1_clone, arg76_1_clone, arg77_1_clone, arg78_1_clone, arg79_1_clone, arg80_1_clone, arg81_1_clone, arg82_1_clone, arg83_1_clone, arg84_1_clone, arg85_1_clone, arg86_1_clone, arg87_1_clone, arg88_1_clone, arg89_1_clone, arg90_1_clone, arg91_1_clone, arg92_1_clone, arg93_1_clone, arg94_1_clone, arg95_1_clone, arg96_1_clone, arg97_1_clone, arg98_1_clone, arg99_1_clone, arg100_1_clone, arg101_1_clone, arg102_1_clone, arg103_1_clone, arg104_1_clone, arg105_1_clone, arg106_1_clone, arg107_1_clone, arg108_1_clone, arg109_1_clone, arg110_1_clone, arg111_1_clone, arg112_1_clone, arg113_1_clone, arg114_1_clone, arg115_1_clone, arg116_1_clone, arg117_1_clone, arg118_1_clone, arg119_1_clone, arg120_1_clone, arg121_1_clone, arg122_1_clone, arg123_1_clone, arg124_1_clone, arg125_1_clone, arg126_1_clone, arg127_1_clone, arg128_1_clone, arg129_1_clone, arg130_1_clone, arg131_1_clone, arg132_1_clone, arg133_1_clone])
    torch.cuda.synchronize()
    print("====with streams ====")
    results = call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1])
    torch.cuda.synchronize()



    all_equal = True
    print("====compare results ====")
    args = "buf353, buf3, buf10, buf150, buf156, buf171, buf178, buf187, buf193, buf207, buf214, buf224, buf230, buf244, buf251, buf260, buf266, buf281, buf288, buf297, buf303, buf318, buf325, buf334, buf340, buf149".split()
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

    # return print_performance(lambda: call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1]), times=times, repeat=repeat)
if __name__ == "__main__":
    # from torch._inductor.utils import compiled_module_main
    # compiled_module_main('hf_T5', benchmark_compiled_module)
    benchmark_compiled_module()
