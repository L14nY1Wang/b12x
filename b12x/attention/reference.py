"""Reference attention helpers for b12x attention correctness checks."""

from __future__ import annotations

import torch


def attention_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    softmax_scale: float | None = None,
    causal: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute exact self-attention for contiguous rank-3 or rank-4 tensors.

    Supported layouts:
    - `q`: `[seqlen_q, q_heads, head_dim]` or `[batch, seqlen_q, q_heads, head_dim]`
    - `k`, `v`: same rank, with `kv_heads` in place of `q_heads`

    Returns:
    - `out` with the same shape/dtype as `q`
    - `lse` with shape `[q_heads, seqlen_q]` or `[batch, q_heads, seqlen_q]`
    """
    if q.ndim not in (3, 4):
        raise ValueError(f"expected rank-3 or rank-4 q tensor, got rank {q.ndim}")
    if q.ndim != k.ndim or q.ndim != v.ndim:
        raise ValueError("q, k, and v must have the same rank")

    squeeze_batch = q.ndim == 3
    if squeeze_batch:
        q = q.unsqueeze(0)
        k = k.unsqueeze(0)
        v = v.unsqueeze(0)

    batch, seqlen_q, q_heads, head_dim = q.shape
    _, seqlen_k, kv_heads, head_dim_k = k.shape
    _, seqlen_v, kv_heads_v, head_dim_v = v.shape
    if head_dim != head_dim_k or head_dim != head_dim_v:
        raise ValueError("reference path currently requires matching Q/K/V head dims")
    if seqlen_k != seqlen_v or kv_heads != kv_heads_v:
        raise ValueError("k and v must have the same sequence length and head count")
    if q_heads % kv_heads != 0:
        raise ValueError(f"q_heads={q_heads} must be divisible by kv_heads={kv_heads}")

    if softmax_scale is None:
        softmax_scale = head_dim ** -0.5

    q_per_kv = q_heads // kv_heads
    if q_per_kv != 1:
        k = k.repeat_interleave(q_per_kv, dim=2)
        v = v.repeat_interleave(q_per_kv, dim=2)

    q_f = q.permute(0, 2, 1, 3).to(torch.float32)
    k_f = k.permute(0, 2, 1, 3).to(torch.float32)
    v_f = v.permute(0, 2, 1, 3).to(torch.float32)

    scores = torch.matmul(q_f, k_f.transpose(-1, -2)) * float(softmax_scale)
    if causal:
        causal_mask = torch.triu(
            torch.ones(seqlen_q, seqlen_k, device=scores.device, dtype=torch.bool),
            diagonal=1,
        )
        scores = scores.masked_fill(causal_mask.view(1, 1, seqlen_q, seqlen_k), float("-inf"))
    probs = torch.softmax(scores, dim=-1)
    out = torch.matmul(probs, v_f).permute(0, 2, 1, 3).to(q.dtype)
    lse = torch.logsumexp(scores, dim=-1).to(torch.float32)

    if squeeze_batch:
        out = out.squeeze(0)
        lse = lse.squeeze(0)
    return out, lse
