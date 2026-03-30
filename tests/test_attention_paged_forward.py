from __future__ import annotations

import math

import torch

from benchmarks.benchmark_paged_attention import (
    _capture_backend_graph,
    _capture_flashinfer_fa2_graph,
    _make_uniform_paged_inputs,
    _quantize_paged_kv_cache_global_e4m3,
)
from b12x.attention.reference import paged_attention_reference
from b12x.integration.attention import PagedAttentionWorkspace

from .helpers import require_sm120
from .test_attention_paged_planner import _make_inputs
from .test_paged_attention_workspace_api import _quantize_paged_kv_cache_e4m3


def _cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> float:
    a_f = a.to(torch.float32).reshape(-1)
    b_f = b.to(torch.float32).reshape(-1)
    return torch.nn.functional.cosine_similarity(a_f, b_f, dim=0).item()


def _make_workspace(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    *,
    mode: str,
    attn_mode: str | None = None,
) -> PagedAttentionWorkspace:
    return PagedAttentionWorkspace.for_tensors(
        mode=mode,
        q=q,
        k_cache=k_cache,
        v_cache=v_cache,
        attn_mode=attn_mode,
    )


def _run_decode_graph_check(
    *,
    cache_seqlen: int,
    b12x_attn_mode: str = "default",
) -> tuple[torch.Tensor, torch.Tensor, str]:
    q, k_cache, v_cache, page_table, cache_seqlens, cu_seqlens_q = _make_uniform_paged_inputs(
        batch=8,
        q_seqlen=1,
        cache_seqlen=cache_seqlen,
        page_size=64,
        q_heads=8,
        kv_heads=1,
        head_dim=256,
        dtype=torch.bfloat16,
        seed=1,
    )
    k_fp8, v_fp8, k_descale, v_descale, k_scale, v_scale = _quantize_paged_kv_cache_global_e4m3(
        k_cache,
        v_cache,
        batch=8,
        kv_heads=1,
    )
    backend = _capture_backend_graph(
        q=q,
        k_cache=k_fp8,
        v_cache=v_fp8,
        page_table=page_table,
        cache_seqlens=cache_seqlens,
        cu_seqlens_q=cu_seqlens_q,
        fixed_split_pages=None,
        k_descale=k_descale,
        v_descale=v_descale,
        warmup=1,
        b12x_attn_mode=b12x_attn_mode,
    )
    _fa2_graph, fa2_out = _capture_flashinfer_fa2_graph(
        q=q,
        k_cache=k_fp8,
        v_cache=v_fp8,
        page_table=page_table,
        cache_seqlens=cache_seqlens,
        q_seqlen=1,
        page_size=64,
        q_heads=8,
        kv_heads=1,
        head_dim=256,
        q_dtype=torch.bfloat16,
        kv_dtype=torch.float8_e4m3fn,
        k_scale=k_scale,
        v_scale=v_scale,
        workspace_bytes=512 * 1024 * 1024,
        warmup=1,
    )
    return backend.output, fa2_out, backend.plan_desc


@torch.inference_mode()
def test_paged_forward_matches_reference_decode_short_context() -> None:
    require_sm120()
    q, k_cache, v_cache, page_table, cache_seqlens, cu_seqlens_q = _make_inputs(
        q_seqlens=[1, 1, 1],
        cache_seqlens=[64, 128, 192],
        dtype=torch.bfloat16,
        kv_dtype=torch.bfloat16,
    )
    workspace = _make_workspace(q, k_cache, v_cache, mode="decode")
    workspace.prepare(page_table, cache_seqlens, cu_seqlens_q)
    output, lse_base2 = workspace.run(
        q,
        k_cache,
        v_cache,
        output=torch.empty_like(q),
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
    lse_natural = lse_base2 * math.log(2.0)
    assert (output - ref_out).abs().max().item() <= 0.03
    assert (lse_natural - ref_lse).abs().max().item() <= 0.05
    assert _cosine_similarity(output, ref_out) >= 0.99999


@torch.inference_mode()
def test_paged_forward_matches_reference_fp8_decode_short_context_batch8() -> None:
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
    workspace = _make_workspace(q, k_fp8, v_fp8, mode="decode")
    workspace.prepare(page_table, cache_seqlens, cu_seqlens_q)
    output, lse_base2 = workspace.run(
        q,
        k_fp8,
        v_fp8,
        output=torch.empty_like(q),
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
    lse_natural = lse_base2 * math.log(2.0)
    assert (output - ref_out).abs().max().item() <= 0.05
    assert (lse_natural - ref_lse).abs().max().item() <= 0.08
    assert _cosine_similarity(output, ref_out) >= 0.999


@torch.inference_mode()
def test_paged_forward_turbo_matches_reference_fp8_decode_short_context_batch8() -> None:
    require_sm120()
    output, fa2_out, plan_desc = _run_decode_graph_check(cache_seqlen=64, b12x_attn_mode="turbo")
    assert plan_desc == "chunk=128,split"
    assert (output - fa2_out).abs().max().item() <= 0.02
    assert _cosine_similarity(output, fa2_out) >= 0.998


@torch.inference_mode()
def test_paged_forward_matches_reference_without_split_bf16_extend() -> None:
    require_sm120()
    q, k_cache, v_cache, page_table, cache_seqlens, cu_seqlens_q = _make_inputs(
        q_seqlens=[6, 5],
        cache_seqlens=[64, 64],
        dtype=torch.bfloat16,
        kv_dtype=torch.bfloat16,
    )
    workspace = _make_workspace(q, k_cache, v_cache, mode="extend")
    workspace.prepare(
        page_table,
        cache_seqlens,
        cu_seqlens_q,
        disable_split_kv=True,
    )
    output, lse_base2 = workspace.run(q, k_cache, v_cache, output=torch.empty_like(q))
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
    lse_natural = lse_base2 * math.log(2.0)
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
    workspace = _make_workspace(q, k_fp8, v_fp8, mode="extend")
    workspace.prepare(
        page_table,
        cache_seqlens,
        cu_seqlens_q,
        fixed_split_size=8,
    )
    output, lse_base2 = workspace.run(
        q,
        k_fp8,
        v_fp8,
        output=torch.empty_like(q),
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
    lse_natural = lse_base2 * math.log(2.0)
    assert (output - ref_out).abs().max().item() <= 0.05
    assert (lse_natural - ref_lse).abs().max().item() <= 0.08
    assert _cosine_similarity(output, ref_out) >= 0.999


@torch.inference_mode()
def test_paged_forward_matches_reference_with_split_fp8_decode() -> None:
    require_sm120()
    output, fa2_out, plan_desc = _run_decode_graph_check(cache_seqlen=512)
    assert plan_desc.endswith(",split")
    assert (output - fa2_out).abs().max().item() <= 0.01
    assert _cosine_similarity(output, fa2_out) >= 0.999


@torch.inference_mode()
def test_paged_forward_turbo_matches_reference_with_split_fp8_decode() -> None:
    require_sm120()
    output, fa2_out, plan_desc = _run_decode_graph_check(cache_seqlen=8192, b12x_attn_mode="turbo")
    assert plan_desc == "chunk=384,split"
    assert (output - fa2_out).abs().max().item() <= 0.01
    assert _cosine_similarity(output, fa2_out) >= 0.998


@torch.inference_mode()
def test_paged_forward_matches_reference_with_split_bf16_kv() -> None:
    require_sm120()
    q, k_cache, v_cache, page_table, cache_seqlens, cu_seqlens_q = _make_inputs(
        q_seqlens=[6, 5],
        cache_seqlens=[2048, 4096],
        dtype=torch.bfloat16,
        kv_dtype=torch.bfloat16,
    )
    workspace = _make_workspace(q, k_cache, v_cache, mode="extend")
    workspace.prepare(
        page_table,
        cache_seqlens,
        cu_seqlens_q,
        fixed_split_size=8,
    )
    output, lse_base2 = workspace.run(q, k_cache, v_cache, output=torch.empty_like(q))
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
    lse_natural = lse_base2 * math.log(2.0)
    assert (output - ref_out).abs().max().item() <= 0.03
    assert (lse_natural - ref_lse).abs().max().item() <= 0.05
    assert _cosine_similarity(output, ref_out) >= 0.99999
