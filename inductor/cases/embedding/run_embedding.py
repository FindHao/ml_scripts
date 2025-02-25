import torch
from torch.nn import Embedding
from liger_kernel.transformers.experimental.embedding import LigerEmbedding

# (8, 2048, 4096, 2048)
B, T, D, V = 8, 2048, 4096, 2048

device = "cuda"
dtype = torch.float32
_input = torch.randint(0, V, (B, T), device=device)
tmp_embed = Embedding(V, D).to(device).to(dtype)
shared_weight = tmp_embed.weight.data


torch_embedding = Embedding(V, D).to(device).to(dtype)
torch_embedding.weight.data.copy_(shared_weight)
liger_embedding = LigerEmbedding(V, D).to(device).to(dtype)
liger_embedding.weight.data.copy_(shared_weight)

tmp_embedding = Embedding(V, D).to(device).to(dtype)
tmp_embedding.weight.data.copy_(shared_weight)
inductor_embedding = torch.compile(tmp_embedding)

from triton.testing import do_bench


def get_func(embedding_fn):
    return lambda: embedding_fn(_input)


torch_exec_time = do_bench(get_func(torch_embedding), return_mode="mean")
liger_exec_time = do_bench(get_func(liger_embedding), return_mode="mean")
inductor_exec_time = do_bench(get_func(inductor_embedding), return_mode="mean")

print(f"Torch embedding average time: {torch_exec_time:.2f} ms")
print(
    f"Liger embedding average time: {liger_exec_time:.2f} ms, speedup: {torch_exec_time/liger_exec_time:.2f}x"
)
print(
    f"Inductor embedding average time: {inductor_exec_time:.2f} ms, speedup: {torch_exec_time/inductor_exec_time:.2f}x"
)
