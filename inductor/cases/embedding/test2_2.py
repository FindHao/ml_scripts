import torch
import torch.nn.functional as F
from triton.testing import do_bench


def position_bias_softmax(scores, weight=None, pw_bias=False):
    scores = scores.to(torch.float32)
    context_position = (
        torch.arange(2048, dtype=torch.long, device="cuda")[:, None].repeat(1, 2048).T
    )
    pos = context_position.clamp(0, 31)
    if weight is not None:
        scores = scores + F.embedding(pos, weight).squeeze(-1)
    elif pw_bias:
        scores = scores + pos * (1.0 / 32)
    return F.softmax(scores, dim=-1).to(torch.float16)


scores = torch.randn(8, 2048, 2048, device="cuda", dtype=torch.float16)
weight = torch.randn(32, 1, device="cuda")
position_bias_softmax(scores, weight)
compiled = torch.compile(position_bias_softmax)
# compiled(scores)
# compiled(scores, pw_bias=True)
compiled(scores, weight=weight)
gb = 2 * scores.element_size() * scores.numel() / 1e9
# sec = do_bench(lambda: compiled(scores)) / 1e3
# print(f"no bias gb/s: {gb/sec}")
# sec = do_bench(lambda: compiled(scores, pw_bias=True)) / 1e3
# print(f"pointwise bias gb/s: {gb/sec}")
sec = do_bench(lambda: compiled(scores, weight=weight)) / 1e3
print(f"weighted bias gb/s: {gb/sec}")
