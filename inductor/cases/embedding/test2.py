import torch
import torch.nn.functional as F


def position_bias_softmax(scores, weight=None, pw_bias=False):
    scores = scores.to(torch.float32)
    context_position = torch.arange(2048, dtype=torch.long, device="cuda")[
        :, None
    ].repeat(1, 2048)
    pos = context_position.clamp(0, 31)
    if weight is not None:
        scores = scores + F.embedding(pos, weight).squeeze(-1)
    elif pw_bias:
        scores = scores + pos * (1.0 / 32)
    return F.softmax(scores, dim=-1).to(torch.float16)


def test_position_bias_softmax():
    # Create test inputs
    batch_size = 4
    seq_len = 2048
    scores = torch.randn(
        batch_size, seq_len, seq_len, device="cuda", dtype=torch.float16
    )

    # Warm up
    for _ in range(10):
        _ = position_bias_softmax(scores)

    # Synchronize CUDA for accurate timing
    torch.cuda.synchronize()
    start_time = torch.cuda.Event(enable_timing=True)
    end_time = torch.cuda.Event(enable_timing=True)

    start_time.record()
    result_basic = position_bias_softmax(scores)
    end_time.record()

    torch.cuda.synchronize()
    elapsed_time = start_time.elapsed_time(end_time)

    print(f"Basic result shape: {result_basic.shape}, dtype: {result_basic.dtype}")
    print(f"Execution time: {elapsed_time:.4f} ms")

    # Test with torch.compile
    compiled_position_bias_softmax = torch.compile(position_bias_softmax)

    # Warm up
    for _ in range(10):
        _ = compiled_position_bias_softmax(scores)

    # Synchronize CUDA for accurate timing
    torch.cuda.synchronize()
    start_time = torch.cuda.Event(enable_timing=True)
    end_time = torch.cuda.Event(enable_timing=True)

    start_time.record()
    result_compiled = compiled_position_bias_softmax(scores)
    end_time.record()

    torch.cuda.synchronize()
    elapsed_time_compiled = start_time.elapsed_time(end_time)

    print(
        f"Compiled result shape: {result_compiled.shape}, dtype: {result_compiled.dtype}"
    )
    print(f"Compiled execution time: {elapsed_time_compiled:.4f} ms")
    print(f"Speedup: {elapsed_time / elapsed_time_compiled:.2f}x")

    return result_basic


if __name__ == "__main__":
    import torch.nn.functional as F

    test_position_bias_softmax()
