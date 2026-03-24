from __future__ import annotations

import math

import torch

from b12x.attention.paged.api import paged_attention_forward
from b12x.attention.paged.planner import create_paged_plan
from b12x.attention.paged.workspace import allocate_paged_workspace_for_plan
from b12x.attention.reference import paged_attention_reference

from .helpers import require_sm120
from .test_attention_paged_planner import _make_inputs
from .test_paged_attention_workspace_api import _quantize_paged_kv_cache_e4m3


def _cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> float:
    a_f = a.to(torch.float32).reshape(-1)
    b_f = b.to(torch.float32).reshape(-1)
    return torch.nn.functional.cosine_similarity(a_f, b_f, dim=0).item()


@torch.inference_mode()
def test_paged_forward_matches_reference_without_split() -> None:
    require_sm120()
    q, k_cache, v_cache, page_table, cache_seqlens, cu_seqlens_q = _make_inputs(
        q_seqlens=[1, 1, 1],
        cache_seqlens=[64, 128, 192],
        dtype=torch.bfloat16,
        kv_dtype=torch.bfloat16,
    )
    plan = create_paged_plan(
        q,
        k_cache,
        v_cache,
        page_table,
        cache_seqlens,
        cu_seqlens_q,
        disable_split_kv=True,
    )
    workspace = allocate_paged_workspace_for_plan(plan)

    output, lse_base2 = paged_attention_forward(
        q,
        k_cache,
        v_cache,
        page_table,
        cache_seqlens,
        cu_seqlens_q,
        workspace=workspace,
        plan=plan,
    )
    torch.cuda.synchronize()

    ref_out, ref_lse = paged_attention_reference(
        q,
        k_cache,
        v_cache,
        page_table,
        cache_seqlens,
        cu_seqlens_q,
        causal=True,
    )
    lse_natural = lse_base2.transpose(0, 1) * math.log(2.0)
    assert (output - ref_out).abs().max().item() <= 0.03
    assert (lse_natural - ref_lse).abs().max().item() <= 0.05
    assert _cosine_similarity(output, ref_out) >= 0.99999


@torch.inference_mode()
def test_paged_forward_matches_reference_without_split_fp8_decode_batch8() -> None:
    require_sm120()
    q, k_cache, v_cache, page_table, cache_seqlens, cu_seqlens_q = _make_inputs(
        q_seqlens=[1, 1, 1, 1, 1, 1, 1, 1],
        cache_seqlens=[64, 64, 64, 64, 64, 64, 64, 64],
        dtype=torch.bfloat16,
        kv_dtype=torch.bfloat16,
    )
    k_fp8, v_fp8, k_descale, v_descale = _quantize_paged_kv_cache_e4m3(
        k_cache,
        v_cache,
        page_table,
        cache_seqlens,
    )
    plan = create_paged_plan(
        q,
        k_fp8,
        v_fp8,
        page_table,
        cache_seqlens,
        cu_seqlens_q,
        disable_split_kv=True,
    )
    workspace = allocate_paged_workspace_for_plan(plan)

    output, lse_base2 = paged_attention_forward(
        q,
        k_fp8,
        v_fp8,
        page_table,
        cache_seqlens,
        cu_seqlens_q,
        workspace=workspace,
        plan=plan,
        k_descale=k_descale,
        v_descale=v_descale,
    )
    torch.cuda.synchronize()

    ref_out, ref_lse = paged_attention_reference(
        q,
        k_fp8,
        v_fp8,
        page_table,
        cache_seqlens,
        cu_seqlens_q,
        k_descale=k_descale,
        v_descale=v_descale,
        causal=True,
    )
    lse_natural = lse_base2.transpose(0, 1) * math.log(2.0)
    assert (output - ref_out).abs().max().item() <= 0.05
    assert (lse_natural - ref_lse).abs().max().item() <= 0.08
    assert _cosine_similarity(output, ref_out) >= 0.999


@torch.inference_mode()
def test_paged_forward_matches_reference_without_split_bf16_extend() -> None:
    require_sm120()
    q, k_cache, v_cache, page_table, cache_seqlens, cu_seqlens_q = _make_inputs(
        q_seqlens=[6, 5],
        cache_seqlens=[64, 64],
        dtype=torch.bfloat16,
        kv_dtype=torch.bfloat16,
    )
    plan = create_paged_plan(
        q,
        k_cache,
        v_cache,
        page_table,
        cache_seqlens,
        cu_seqlens_q,
        disable_split_kv=True,
    )
    workspace = allocate_paged_workspace_for_plan(plan)

    output, lse_base2 = paged_attention_forward(
        q,
        k_cache,
        v_cache,
        page_table,
        cache_seqlens,
        cu_seqlens_q,
        workspace=workspace,
        plan=plan,
    )
    torch.cuda.synchronize()

    ref_out, ref_lse = paged_attention_reference(
        q,
        k_cache,
        v_cache,
        page_table,
        cache_seqlens,
        cu_seqlens_q,
        causal=True,
    )
    lse_natural = lse_base2.transpose(0, 1) * math.log(2.0)
    assert (output - ref_out).abs().max().item() <= 0.03
    assert (lse_natural - ref_lse).abs().max().item() <= 0.05
    assert _cosine_similarity(output, ref_out) >= 0.99999


@torch.inference_mode()
def test_paged_forward_matches_reference_with_split_fp8_kv() -> None:
    require_sm120()
    q, k_cache, v_cache, page_table, cache_seqlens, cu_seqlens_q = _make_inputs(
        q_seqlens=[6, 5],
        cache_seqlens=[2048, 4096],
        dtype=torch.bfloat16,
        kv_dtype=torch.bfloat16,
    )
    k_fp8, v_fp8, k_descale, v_descale = _quantize_paged_kv_cache_e4m3(
        k_cache,
        v_cache,
        page_table,
        cache_seqlens,
    )
    plan = create_paged_plan(
        q,
        k_fp8,
        v_fp8,
        page_table,
        cache_seqlens,
        cu_seqlens_q,
        fixed_split_size=8,
    )
    workspace = allocate_paged_workspace_for_plan(plan)

    output, lse_base2 = paged_attention_forward(
        q,
        k_fp8,
        v_fp8,
        page_table,
        cache_seqlens,
        cu_seqlens_q,
        workspace=workspace,
        plan=plan,
        k_descale=k_descale,
        v_descale=v_descale,
    )
    torch.cuda.synchronize()

    ref_out, ref_lse = paged_attention_reference(
        q,
        k_fp8,
        v_fp8,
        page_table,
        cache_seqlens,
        cu_seqlens_q,
        k_descale=k_descale,
        v_descale=v_descale,
        causal=True,
    )
    lse_natural = lse_base2.transpose(0, 1) * math.log(2.0)
    assert (output - ref_out).abs().max().item() <= 0.05
    assert (lse_natural - ref_lse).abs().max().item() <= 0.08
    assert _cosine_similarity(output, ref_out) >= 0.999


@torch.inference_mode()
def test_paged_forward_matches_reference_with_split_bf16_kv() -> None:
    require_sm120()
    q, k_cache, v_cache, page_table, cache_seqlens, cu_seqlens_q = _make_inputs(
        q_seqlens=[6, 5],
        cache_seqlens=[2048, 4096],
        dtype=torch.bfloat16,
        kv_dtype=torch.bfloat16,
    )
    plan = create_paged_plan(
        q,
        k_cache,
        v_cache,
        page_table,
        cache_seqlens,
        cu_seqlens_q,
        fixed_split_size=8,
    )
    workspace = allocate_paged_workspace_for_plan(plan)

    output, lse_base2 = paged_attention_forward(
        q,
        k_cache,
        v_cache,
        page_table,
        cache_seqlens,
        cu_seqlens_q,
        workspace=workspace,
        plan=plan,
    )
    torch.cuda.synchronize()

    ref_out, ref_lse = paged_attention_reference(
        q,
        k_cache,
        v_cache,
        page_table,
        cache_seqlens,
        cu_seqlens_q,
        causal=True,
    )
    lse_natural = lse_base2.transpose(0, 1) * math.log(2.0)
    assert (output - ref_out).abs().max().item() <= 0.03
    assert (lse_natural - ref_lse).abs().max().item() <= 0.05
    assert _cosine_similarity(output, ref_out) >= 0.99999
