import torch
import struct
# Function to convert a float to its hex representation
def float_to_hex(f):
    return hex(struct.unpack('<I', struct.pack('<f', f))[0])
    # return ''.join('{:0>8b}'.format(c) for c in struct.pack('!f', f))

# Convert each tensor value to its hex representation
def api(tensor):
    return [[float_to_hex(x.item()) for x in row] for row in tensor]

# 输入数据
data = torch.tensor([[-0.3499, 0.8379], [-0.2217, -1.1514]], dtype=torch.float16)

# 打印初始的 float16 数据
print("Initial float16 data:")
print(data)

print(api(data))

# 将 float16 转换为 bfloat16 (通过 int16 的 bitcast)
data_bfloat16_int = data.view(torch.int16)
data_bfloat16 = data_bfloat16_int.view(torch.bfloat16)

# 打印转换后的 bfloat16 数据
print("Data bitcast to bfloat16:")
print(data_bfloat16)
print(api(data_bfloat16))

# 将 bfloat16 转换为 float32
data_float32 = data_bfloat16.to(torch.float32)

# 打印转换后的 float32 数据
print("Data upcast to float32:")
print(data_float32)
print(api(data_float32))

# 执行 +1 操作
data_float32 += 1

# 打印加法运算后的 float32 数据
print("Data after +1 in float32:")
print(data_float32)
print(api(data_float32))

# 将结果 downcast 为 bfloat16
result_bfloat16 = data_float32.to(torch.bfloat16)

# 打印最终的 bfloat16 数据
print("Final result downcast to bfloat16:")
print(result_bfloat16)
print(api(result_bfloat16))
