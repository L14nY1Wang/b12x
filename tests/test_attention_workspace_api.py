from __future__ import annotations

import pytest
import torch

from b12x.attention.reference import attention_reference
from b12x.integration.attention import (
    allocate_attention_workspace,
    allocate_attention_workspace_pool,
    b12x_attention_forward,
    clear_attention_caches,
)

from .helpers import require_sm120


def _make_gqa_inputs(
    shape: tuple[int, int, int, int],
    *,
    kv_heads: int,
    dtype: torch.dtype = torch.bfloat16,
    seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    torch.manual_seed(seed)
    batch, seqlen, q_heads, head_dim = shape
    q = torch.randn(batch, seqlen, q_heads, head_dim, device="cuda", dtype=dtype) / 4
    k = torch.randn(batch, seqlen, kv_heads, head_dim, device="cuda", dtype=dtype) / 4
    v = torch.randn(batch, seqlen, kv_heads, head_dim, device="cuda", dtype=dtype) / 4
    return q, k, v


def _cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> float:
    a_f = a.to(torch.float32).reshape(-1)
    b_f = b.to(torch.float32).reshape(-1)
    return torch.nn.functional.cosine_similarity(a_f, b_f, dim=0).item()


def test_exact_workspace_matches_reference_for_qwen_like_shape() -> None:
    require_sm120()
    clear_attention_caches()

    q, k, v = _make_gqa_inputs((1, 48, 8, 256), kv_heads=1, seed=7)
    workspace = allocate_attention_workspace(q, k, v, causal=True)
    out, lse = b12x_attention_forward(q, k, v, workspace=workspace)
    ref_out, ref_lse = attention_reference(q, k, v, causal=True)
    torch.cuda.synchronize()

    max_abs = (out - ref_out).abs().max().item()
    max_lse_abs = (lse - ref_lse).abs().max().item()
    assert max_abs <= 0.02
    assert max_lse_abs <= 0.03
    assert _cosine_similarity(out, ref_out) >= 0.99999


def test_workspace_pool_reuses_exact_shape_buffers() -> None:
    require_sm120()
    clear_attention_caches()

    q, k, v = _make_gqa_inputs((1, 32, 8, 256), kv_heads=1, seed=11)
    pool = allocate_attention_workspace_pool()

    out0, lse0 = b12x_attention_forward(q, k, v, workspace=pool)
    ref_out, ref_lse = attention_reference(q, k, v, causal=True)
    out1, lse1 = b12x_attention_forward(q, k, v, workspace=pool)
    torch.cuda.synchronize()

    assert len(pool.workspaces) == 1
    assert out0.data_ptr() == out1.data_ptr()
    assert lse0.data_ptr() == lse1.data_ptr()
    assert (out1 - ref_out).abs().max().item() <= 0.02
    assert (lse1 - ref_lse).abs().max().item() <= 0.03


def test_exact_workspace_rejects_shape_mismatch() -> None:
    require_sm120()
    clear_attention_caches()

    q0, k0, v0 = _make_gqa_inputs((1, 16, 8, 256), kv_heads=1, seed=13)
    q1, k1, v1 = _make_gqa_inputs((1, 32, 8, 256), kv_heads=1, seed=17)
    workspace = allocate_attention_workspace(q0, k0, v0, causal=True)

    with pytest.raises(ValueError, match="workspace shape mismatch"):
        b12x_attention_forward(q1, k1, v1, workspace=workspace)
