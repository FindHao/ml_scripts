import torch
import time


def gpu_timer(func):
    def wrapper(*args, **kwargs):
        torch.cuda.synchronize()
        start = time.time_ns()
        result = func(*args, **kwargs)
        torch.cuda.synchronize()
        end = time.time_ns()
        duration_ms = (end - start) / 1e6
        if duration_ms > 1000:
            print(f"{func.__name__} executed in {duration_ms / 1000:.2f} seconds")
        else:
            print(f"{func.__name__} executed in {duration_ms:.2f} milliseconds")
        return result
    return wrapper