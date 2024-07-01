import torch

device = torch.device("cuda")

# # @torch.compile
# # def fn3(x, y):
# #     x = x.view(torch.float16)
# #     y = y.view(torch.float16) + 1
# #     return x @ y
# # @torch.compile
# # def fn4(x, y):
# #     # x1 = x.view(torch.int16)
# #     # y1 = y.view(torch.int16)
# #     x1 = x.view(torch.bfloat16)
# #     y1 = y.view(torch.bfloat16)
# #     return torch.mm(x1, y1)

# @torch.compile
# def fn3(x,y):

#     x = x + 1
#     x = torch.ops.aten.view.dtype(x, torch.int16)
#     x = x * 2
#     return x


# # x = torch.randn((10,10), device=device, dtype=torch.int16)
# # y = torch.randn((10,10), device=device, dtype=torch.int16)
# # x = torch.randint(low=-32768, high=32767, size=(10,10), device='cuda', dtype=torch.int16)
# # y = torch.randint(low=-32768, high=32767, size=(10,10), device='cuda', dtype=torch.int16)
# # x = torch.randn((2,2), device=device, dtype=torch.bfloat16)
# y = torch.randn((2,2), device=device, dtype=torch.bfloat16)
# # x_clone = x.clone()
# # y_clone = y.clone()
# x = torch.randn([1024], dtype=torch.float16, device=device)

# inputs = [x, y]
# out1 = fn3(*inputs)
# # out2 = fn3(x_clone, y_clone)
# # are_close = torch.allclose(out1, out2, rtol=1e-05, atol=1e-08)
# # print(are_close)
# # fn4(x, y)
# # fn4(x)

# # x = torch.randn(100, device=device, dtype=torch.bfloat16)
# # y = torch.randn(100, device=device, dtype=torch.bfloat16)

# # @torch.compile
# # def fn5(x, y):
# #     x1 = x.view(torch.int16)
# #     x2 = x.view(torch.int16)
# #     return x1@y, x2

# # fn5(x, y)

# import torch
# @torch.compile
# def fn6(x):
#     x = x + 1
#     x = torch.ops.aten.view.dtype(x, torch.uint16)
#     x = x * 2
#     return x

# x = torch.randn([1024], dtype=torch.float16, device='cuda')
# fn6(x)

@torch.compile
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
dtype_ranges = {
    torch.uint8: (0, 256),
    torch.int8: (-128, 128),
    torch.int16: (-32768, 32767),
    torch.int32: (-2147483648, 2147483647),
    torch.int64: (-9223372036854775808, 9223372036854775807)
}
test_dtypes = [torch.float, torch.float64, torch.float16, torch.bfloat16, torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64]
for test_dtype_x in test_dtypes:
    for test_dtype_y in test_dtypes:
        # @ operation needs arguments to be the same dtype
        for view_dtype in test_dtypes:
            print(f"({test_dtype_x}, {test_dtype_y}, {view_dtype})")
            if _get_primitive_bitwidth(test_dtype_x) != _get_primitive_bitwidth(test_dtype_y) or _get_primitive_bitwidth(test_dtype_x) != _get_primitive_bitwidth(view_dtype) or view_dtype in [torch.int32, torch.int64, torch.int16, torch.uint8, torch.int8]:
                print("skip")
                continue
            if test_dtype_x.is_floating_point:
                x = torch.randn((2, 2), device='cuda', dtype=test_dtype_x)
            else:
                # Use the range from dtype_ranges if available
                if test_dtype_x in dtype_ranges:
                    low, high = dtype_ranges[test_dtype_x]
                    x = torch.randint(low=low, high=high, size=(2, 2), device='cuda', dtype=test_dtype_x)
                else:
                    raise ValueError(f"Unsupported dtype: {test_dtype_x}")
            if test_dtype_y.is_floating_point:
                y = torch.randn((2, 2), device='cuda', dtype=test_dtype_y)
            else:
                # Use the range from dtype_ranges if available
                if test_dtype_y in dtype_ranges:
                    low, high = dtype_ranges[test_dtype_y]
                    y = torch.randint(low=low, high=high, size=(2, 2), device='cuda', dtype=test_dtype_y)
                else:
                    raise ValueError(f"Unsupported dtype: {test_dtype_y}")
            x2 = x.clone()
            fn(x, y, view_dtype, x2)
            # # print the final valued combination to /tmp/yhao.log
            # with open("/tmp/yhao.log", "a") as f:
            #     f.write(f"({test_dtype_x}, {test_dtype_y}, {view_dtype})\n")
