from __future__ import annotations

import pytest
import torch

from b12x.attention.reference import attention_reference
from b12x.integration.attention import (
    allocate_attention_workspace_pool,
    allocate_attention_workspace_for_plan,
    b12x_attention_forward,
    clear_attention_caches,
    create_attention_plan,
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
    plan = create_attention_plan(q, k, v, causal=True)
    workspace = allocate_attention_workspace_for_plan(plan)
    out, lse = b12x_attention_forward(q, k, v, workspace=workspace, plan=plan)
    ref_out, ref_lse = attention_reference(q, k, v, causal=True)
    torch.cuda.synchronize()

    max_abs = (out - ref_out).abs().max().item()
    max_lse_abs = (lse - ref_lse).abs().max().item()
    assert max_abs <= 0.02
    assert max_lse_abs <= 0.03
    assert _cosine_similarity(out, ref_out) >= 0.99999


def test_contiguous_plan_exposes_logical_gqa_dimensions() -> None:
    require_sm120()
    clear_attention_caches()

    q, k, v = _make_gqa_inputs((2, 48, 8, 256), kv_heads=1, seed=5)
    plan = create_attention_plan(q, k, v, causal=True)

    assert plan.num_batch == 2
    assert plan.num_q_heads == 8
    assert plan.num_kv_heads == 1
    assert plan.qhead_per_kvhead == 8
    assert plan.seqlen_q_static == 48
    assert plan.seqlen_k_static == 48
    assert plan.logical_q_rows_static == 48 * 8
    assert plan.logical_total_q_rows == 2 * 48 * 8


def test_contiguous_workspace_pool_reuses_plan_exact_shape() -> None:
    require_sm120()
    clear_attention_caches()

    q, k, v = _make_gqa_inputs((1, 48, 8, 256), kv_heads=1, seed=31)
    plan = create_attention_plan(q, k, v, causal=True)
    pool = allocate_attention_workspace_pool()

    out0, lse0 = b12x_attention_forward(q, k, v, workspace=pool, plan=plan)
    out1, lse1 = b12x_attention_forward(q, k, v, workspace=pool, plan=plan)

    assert out0.data_ptr() == out1.data_ptr()
    assert lse0.data_ptr() == lse1.data_ptr()
    assert len(pool.workspaces) == 1


def test_contiguous_workspace_pool_requires_explicit_plan() -> None:
    require_sm120()
    clear_attention_caches()

    q, k, v = _make_gqa_inputs((1, 48, 8, 256), kv_heads=1, seed=37)
    pool = allocate_attention_workspace_pool()

    with pytest.raises(TypeError, match="require an explicit AttentionPlan"):
        b12x_attention_forward(q, k, v, workspace=pool)


def test_right_aligned_causal_multi_tile_shape_matches_reference() -> None:
    require_sm120()
    clear_attention_caches()

    torch.manual_seed(19)
    q = torch.randn(1, 6, 8, 256, device="cuda", dtype=torch.bfloat16) / 4
    k = torch.randn(1, 33, 1, 256, device="cuda", dtype=torch.bfloat16) / 4
    v = torch.randn(1, 33, 1, 256, device="cuda", dtype=torch.bfloat16) / 4
    plan = create_attention_plan(q, k, v, causal=True)
    workspace = allocate_attention_workspace_for_plan(plan)

    out, lse = b12x_attention_forward(q, k, v, workspace=workspace, plan=plan)
    ref_out, ref_lse = attention_reference(q, k, v, causal=True)
    torch.cuda.synchronize()

    assert not torch.isnan(out).any()
    assert not torch.isnan(lse).any()
    assert (out - ref_out).abs().max().item() <= 0.02
    assert (lse - ref_lse).abs().max().item() <= 0.03
    assert _cosine_similarity(out, ref_out) >= 0.99999


def test_single_token_single_key_corner_is_rejected() -> None:
    require_sm120()
    clear_attention_caches()

    torch.manual_seed(29)
    q = torch.randn(1, 1, 8, 256, device="cuda", dtype=torch.bfloat16) / 4
    k = torch.randn(1, 1, 1, 256, device="cuda", dtype=torch.bfloat16) / 4
    v = torch.randn(1, 1, 1, 256, device="cuda", dtype=torch.bfloat16) / 4
    plan = create_attention_plan(q, k, v, causal=True)
    workspace = allocate_attention_workspace_for_plan(plan)

    with pytest.raises(ValueError, match="single-token single-key corner"):
        b12x_attention_forward(q, k, v, workspace=workspace, plan=plan)


def test_exact_workspace_rejects_shape_mismatch() -> None:
    require_sm120()
    clear_attention_caches()

    q0, k0, v0 = _make_gqa_inputs((1, 16, 8, 256), kv_heads=1, seed=13)
    q1, k1, v1 = _make_gqa_inputs((1, 32, 8, 256), kv_heads=1, seed=17)
    plan0 = create_attention_plan(q0, k0, v0, causal=True)
    workspace = allocate_attention_workspace_for_plan(plan0)

    with pytest.raises(ValueError, match="attention plan mismatch"):
        b12x_attention_forward(q1, k1, v1, workspace=workspace, plan=plan0)
