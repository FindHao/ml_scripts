from typing import Optional, Tuple

import torch
import triton
import triton.language as tl
import os
import sys

# Add generative-recommenders directory to the Python path
generative_recommenders_path = os.path.join("/home/yhao/ml_scripts/triton", "generative-recommenders")
sys.path.append(generative_recommenders_path)


# Import necessary functions from generative_recommenders
from generative_recommenders.common import (
    apply_sampling,
    generate_sparse_seq_len,
    set_use_runtime_max_seq_len,
)

# Import necessary functions from triton_hstu_attention
from generative_recommenders.ops.triton.triton_hstu_attention import (
    autotune_max_seq_len,
    prev_power_of_2,
    switch_to_contiguous_if_needed,
    triton_hstu_mha,
)

# Always autotune based on the actual max_seq_len
set_use_runtime_max_seq_len(True)


def hstu(
    max_seq_len: int,
    alpha: float,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    seq_offsets: torch.Tensor,
    num_targets: Optional[torch.Tensor] = None,
    max_attn_len: int = 0,
    contextual_seq_len: int = 0,
    sort_by_length: bool = True,
):
    """
    Extracted HSTU (Hierarchical Sparse Transformer Unit) function from the ragged_attention operator.

    Args:
        max_seq_len: Maximum sequence length
        alpha: Scaling factor for attention scores (typically 1.0 / attn_dim)
        q: Query tensor of shape [L, H, D_attn]
        k: Key tensor of shape [L, H, D_attn]
        v: Value tensor of shape [L, H, D_hidden]
        seq_offsets: Tensor of shape [batch_size + 1] containing offsets for each sequence
        num_targets: Optional tensor of shape [batch_size] containing number of targets for each sequence
        max_attn_len: Maximum attention length (0 means no limit)
        contextual_seq_len: Contextual sequence length (0 means no contextual sequence)
        sort_by_length: Whether to sort sequences by length for better performance

    Returns:
        Output tensor of shape [L, H, D_hidden]
    """
    return triton_hstu_mha(
        max_seq_len,
        alpha=alpha,
        q=q,
        k=k,
        v=v,
        seq_offsets=seq_offsets,
        num_targets=num_targets,
        max_attn_len=max_attn_len,
        contextual_seq_len=contextual_seq_len,
        sort_by_length=sort_by_length,
    )


def get_test_inputs(
    batch_size,
    heads,
    seq_len,
    attn_dim,
    hidden_dim,
    seq_sparsity,
    has_delta_q,
    delta_size,
    target_size,
    max_attn_len,
    dtype,
    requires_grad,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
    lengths = generate_sparse_seq_len(
        size=batch_size,
        max_seq_len=seq_len,
        sparsity=seq_sparsity,
        device=torch.device("cuda"),
    )
    lengths = apply_sampling(lengths, alpha=2.0, max_seq_len=seq_len)
    if has_delta_q:
        lengths = lengths + delta_size
        num_targets = torch.ones_like(lengths) * delta_size
        seq_len = seq_len + delta_size
    else:
        delta_size = 0
        num_targets = None
        if target_size != 0:
            num_targets = torch.randint(
                target_size,
                target_size + 1,
                (batch_size,),
                device=lengths.device,
                dtype=lengths.dtype,
            )
            num_targets = torch.where(num_targets > lengths, lengths, num_targets).to(
                torch.int32
            )
    max_attn_len = max_attn_len if max_attn_len < seq_len else seq_len
    seq_offsets = torch.zeros(
        (batch_size + 1,), dtype=torch.int32, device=torch.device("cuda")
    )
    seq_offsets[1:] = torch.cumsum(lengths, dim=0)
    L = int(seq_offsets[-1].item())
    x = torch.empty(
        (L, heads, attn_dim * 2 + hidden_dim),
        dtype=dtype,
        device=torch.device("cuda"),
    ).uniform_(-0.01, 0.01)
    q, k, v = torch.split(x, [attn_dim, attn_dim, hidden_dim], dim=-1)
    # Only create delta_q and delta_x_offsets if has_delta_q is True
    if has_delta_q:
        delta_q = torch.empty(
            (batch_size * delta_size, heads, attn_dim),
            dtype=dtype,
            device=torch.device("cuda"),
        ).uniform_(-0.1, 0.1)
        delta_x_offsets = torch.arange(0, delta_size, device=torch.device("cuda"))
        delta_x_offsets = (seq_offsets[1:] - delta_size).view(
            batch_size, 1
        ) + delta_x_offsets.view(1, delta_size)
        delta_x_offsets = delta_x_offsets.view(-1)
    q = q.requires_grad_(requires_grad)
    k = k.requires_grad_(requires_grad)
    v = v.requires_grad_(requires_grad)
    return q, k, v, seq_offsets, num_targets, seq_len


def test():
    """
    Test the HSTU function with some sample inputs.
    """
    # Parameters
    batch_size = 128
    num_heads = 4
    seq_len = 128
    attn_dim = 128
    hidden_dim = 128
    sparsity = 0.8  # Variable length sequences
    target_size = 20
    max_attn_len = 64

    # Generate test inputs
    q, k, v, seq_offsets, num_targets, max_seq_len = get_test_inputs(
        batch_size=batch_size,
        heads=num_heads,
        seq_len=seq_len,
        attn_dim=attn_dim,
        hidden_dim=hidden_dim,
        seq_sparsity=sparsity,
        has_delta_q=False,
        delta_size=0,  # Set to 0 since has_delta_q is False
        target_size=target_size,
        max_attn_len=max_attn_len,
        dtype=torch.float16,
        requires_grad=True,
    )

    # Set alpha (scaling factor)
    alpha = 1.0 / attn_dim

    # Run the HSTU function
    output = hstu(
        max_seq_len=max_seq_len,
        alpha=alpha,
        q=q,
        k=k,
        v=v,
        seq_offsets=seq_offsets,
        num_targets=num_targets,
        max_attn_len=max_attn_len,
        contextual_seq_len=0,
        sort_by_length=True,
    )

    # print(f"Input shapes: q={q.shape}, k={k.shape}, v={v.shape}")
    # print(f"Output shape: {output.shape}")
    # print(f"Sequence offsets: {seq_offsets}")
    # if num_targets is not None:
    #     print(f"Number of targets: {num_targets}")

    # # Test backward pass
    # if q.requires_grad:
    #     loss = output.sum()
    #     loss.backward()
    #     print(
    #         f"Gradients: q.grad={q.grad.shape}, k.grad={k.grad.shape}, v.grad={v.grad.shape}"
    #     )
    do = torch.rand_like(output)
    output.backward(do, retain_graph=True)
    print(f"output: {output.shape}")
    return output


if __name__ == "__main__":
    test()
