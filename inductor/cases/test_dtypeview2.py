import torch
import struct
device = torch.device("cuda")

# Function to convert a float to its hex representation
def float_to_hex(f):
    return hex(struct.unpack('<I', struct.pack('<f', f))[0])
    # return ''.join('{:0>8b}'.format(c) for c in struct.pack('!f', f))

# Convert each tensor value to its hex representation
def api(tensor):
    return [[float_to_hex(x.item()) for x in row] for row in tensor]

# def fn(x, y, x_dtype):
#     x = x.view(x_dtype)
#     y = y.view(x_dtype) + 1
#     return x @ y
def fn(x, x_dtype):
    return x.view(x_dtype) + 1

x = torch.randn((2, 2), device='cuda', dtype=torch.float16)
# x = torch.tensor([[-0.3499,  0.8379],
#         [-0.2217, -1.1514]], device='cuda', dtype=torch.float16)


# y = torch.randn((2, 2), device='cuda', dtype=torch.float16)
x1 = x.clone()
x2 = x.clone()
# y1 = y.clone()

ref_results = fn(x, torch.bfloat16)
# print(fn(x, torch.bfloat16))
# print(ref_results)

# def fn2(x):
#     x_bfloat16 = x.to(dtype=torch.bfloat16, bitcast=True)
#     x_float32 = x_bfloat16.to(dtype=torch.float32)
#     x_float32 = x_float32 + 1
#     x_bfloat16_final = x_float32.to(dtype=torch.bfloat16)
#     return x_bfloat16_final

# fn2_result = fn2(x1)
# print(fn2_result)
# print(torch.allclose(ref_results, fn2_result))
compiled_fn = torch.compile(fn)
compiled_results = compiled_fn(x1, torch.bfloat16)
# print in hex format

print('input,',api(x2))
print('ref,',api(ref_results))
print("compiled,", api(compiled_results))

print("ref, ", ref_results)
print("compiled,", compiled_results)
print(torch.allclose(ref_results, compiled_results))




# ref_results = fn(x, y, torch.bfloat16)
# compiled_fn = torch.compile(fn)
# compiled_results = compiled_fn(x1, y1, torch.bfloat16)
# print(ref_results)
# print(compiled_results)
# print(torch.allclose(ref_results, compiled_results))

# ref_results = fn(x, torch.bfloat16)
# compiled_fn = torch.compile(fn)
# compiled_results = compiled_fn(x1, torch.bfloat16)
# print(ref_results)
# print(compiled_results)
# print(torch.allclose(ref_results, compiled_results))
