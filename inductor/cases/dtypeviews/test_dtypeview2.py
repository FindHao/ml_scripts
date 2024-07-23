import torch
import struct
device = torch.device("cuda")
device = torch.device("cpu")
from torch._dynamo.testing import rand_strided

# # Function to convert a float to its hex representation
# def float_to_hex(f):
#     return hex(struct.unpack('<I', struct.pack('<f', f))[0])
#     # return ''.join('{:0>8b}'.format(c) for c in struct.pack('!f', f))

# # Convert each tensor value to its hex representation
# def api(tensor):
#     return [[float_to_hex(x.item()) for x in row] for row in tensor]

# # def fn(x, y, x_dtype):
# #     x = x.view(x_dtype)
# #     y = y.view(x_dtype) + 1
# #     return x @ y
# def fn(x, x_dtype):
#     return x.view(x_dtype) + 1

# x = torch.randn((2, 2), device='cuda', dtype=torch.float16)
# # x = torch.tensor([[-0.3499,  0.8379],
# #         [-0.2217, -1.1514]], device='cuda', dtype=torch.float16)


# # y = torch.randn((2, 2), device='cuda', dtype=torch.float16)
# x1 = x.clone()
# x2 = x.clone()
# # y1 = y.clone()

# ref_results = fn(x, torch.bfloat16)
# # print(fn(x, torch.bfloat16))
# # print(ref_results)

# # def fn2(x):
# #     x_bfloat16 = x.to(dtype=torch.bfloat16, bitcast=True)
# #     x_float32 = x_bfloat16.to(dtype=torch.float32)
# #     x_float32 = x_float32 + 1
# #     x_bfloat16_final = x_float32.to(dtype=torch.bfloat16)
# #     return x_bfloat16_final

# # fn2_result = fn2(x1)
# # print(fn2_result)
# # print(torch.allclose(ref_results, fn2_result))
# compiled_fn = torch.compile(fn)
# compiled_results = compiled_fn(x1, torch.bfloat16)
# # print in hex format

# print('input,',api(x2))
# print('ref,',api(ref_results))
# print("compiled,", api(compiled_results))

# print("ref, ", ref_results)
# print("compiled,", compiled_results)
# print(torch.allclose(ref_results, compiled_results))




# # ref_results = fn(x, y, torch.bfloat16)
# # compiled_fn = torch.compile(fn)
# # compiled_results = compiled_fn(x1, y1, torch.bfloat16)
# # print(ref_results)
# # print(compiled_results)
# # print(torch.allclose(ref_results, compiled_results))

# # ref_results = fn(x, torch.bfloat16)
# # compiled_fn = torch.compile(fn)
# # compiled_results = compiled_fn(x1, torch.bfloat16)
# # print(ref_results)
# # print(compiled_results)
# # print(torch.allclose(ref_results, compiled_results))


def fn(x, y, x_dtype, x2):
    x = x.view(x_dtype)
    y = y.view(x_dtype) + 1
    x2 = x2.view(x_dtype) + 1
    return x @ y, x2 @ x

def _get_primitive_bitwidth(dtype):
    if dtype.is_floating_point:
        return torch.finfo(dtype).bits
    else:
        return torch.iinfo(dtype).bits

test_dtypes = [
    torch.float32,
    torch.float64,
    torch.float16,
    torch.bfloat16,
    torch.uint8,
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
]
xx = set()
yy = set()
zz = set()
# for test_dtype_x in test_dtypes:
#     for test_dtype_y in test_dtypes:
#         # @ operation needs arguments to be the same dtype
#         for view_dtype in test_dtypes:
#             if (
#                 _get_primitive_bitwidth(test_dtype_x)
#                 != _get_primitive_bitwidth(test_dtype_y)
#                 or _get_primitive_bitwidth(test_dtype_x)
#                 != _get_primitive_bitwidth(view_dtype)
#                 or view_dtype
#                 in [
#                     torch.int32,
#                     torch.int64,
#                     torch.int16,
#                     torch.uint8,
#                     torch.int8,
#                 ]
#             ):
#                 continue

for test_dtype_x, test_dtype_y, view_dtype in [
(torch.float32, torch.float32, torch.int32),
#     (torch.float32, torch.float32, torch.float32),
# (torch.float32, torch.int32, torch.float32),
# (torch.float64, torch.float64, torch.float64),
# (torch.float64, torch.int64, torch.float64),
# (torch.float16, torch.float16, torch.float16),
# (torch.float16, torch.bfloat16, torch.float16),
# (torch.float16, torch.bfloat16, torch.bfloat16),
# (torch.float16, torch.int16, torch.float16),
# (torch.float16, torch.int16, torch.bfloat16),
# (torch.bfloat16, torch.float16, torch.float16),
# (torch.bfloat16, torch.float16, torch.bfloat16),
# (torch.bfloat16, torch.bfloat16, torch.float16),
# (torch.bfloat16, torch.bfloat16, torch.bfloat16),
# (torch.bfloat16, torch.int16, torch.float16),
# (torch.bfloat16, torch.int16, torch.bfloat16),
# (torch.int16, torch.float16, torch.float16),
# (torch.int16, torch.float16, torch.bfloat16),
# (torch.int16, torch.bfloat16, torch.float16),
# (torch.int16, torch.bfloat16, torch.bfloat16),
# (torch.int16, torch.int16, torch.float16),
# (torch.int16, torch.int16, torch.bfloat16),
# (torch.int32, torch.float32, torch.float32),
# (torch.int32, torch.int32, torch.float32),
# (torch.int64, torch.float64, torch.float64),
# (torch.int64, torch.int64, torch.float64)
]:
    print(f"({test_dtype_x}, {test_dtype_y}, {view_dtype})")
    xx.add(test_dtype_x)
    yy.add(test_dtype_y)
    zz.add(view_dtype)
    # x = rand_strided(
    #     (2, 2), (2, 1), device=device, dtype=test_dtype_x
    # )
    # z = rand_strided(
    #     (2, 2), (2, 1), device=device, dtype=test_dtype_x
    # )
    # y = rand_strided(
    #     (2, 2), (2, 1), device=device, dtype=test_dtype_y
    # )
    x = torch.tensor([[1066185791, 1073983379], [1066185791, 1073983379]], device=device, dtype=test_dtype_x)

    y = torch.tensor([[1066185791, 1073983379], [1066185791, 1073983379]], device=device, dtype=test_dtype_y)
    z = torch.tensor([[1066185791, 1073983379], [1066185791, 1073983379]], device=device, dtype=test_dtype_x)
    x1 = x.clone()
    y1 = y.clone()
    z1 = z.clone()
    try:
        ref_results = fn(x, y, view_dtype, z)
    except Exception as e:
        continue
    compiled_fn = torch.compile(fn)
    compiled_results = compiled_fn(x1, y1,  view_dtype, z1)
    print(ref_results)
    print(compiled_results)
    print(f"(>>findhao===={test_dtype_x}, {test_dtype_y}, {view_dtype})", torch.allclose(ref_results[0], compiled_results[0]))
