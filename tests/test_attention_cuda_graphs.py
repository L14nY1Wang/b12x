from __future__ import annotations

import math

import pytest
import torch

from b12x.attention.reference import paged_attention_reference
from b12x.integration.attention import (
    allocate_paged_attention_workspace_for_plan,
    b12x_paged_attention_forward,
    clear_attention_caches,
    create_paged_attention_plan,
)

from .helpers import require_sm120
from .test_paged_attention_workspace_api import (
    _make_paged_inputs,
    _quantize_paged_kv_cache_e4m3,
)


def _cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> float:
    a_f = a.to(torch.float32).reshape(-1)
    b_f = b.to(torch.float32).reshape(-1)
    return torch.nn.functional.cosine_similarity(a_f, b_f, dim=0).item()


def _lse_base2_to_natural(lse: torch.Tensor) -> torch.Tensor:
    return lse * math.log(2.0)


@torch.inference_mode()
@pytest.mark.parametrize("fixed_split_size", [None, 4])
def test_paged_attention_replays_under_cuda_graph_with_fixed_metadata(
    fixed_split_size: int | None,
) -> None:
    require_sm120()
    clear_attention_caches()

    q, k_cache, v_cache, page_table, cache_seqlens, cu_seqlens_q = _make_paged_inputs(
        q_seqlens=[6, 5, 7, 4],
        cache_seqlens=[97, 81, 113, 68],
        page_size=64,
        seed=73,
    )
    plan = create_paged_attention_plan(
        q,
        k_cache,
        v_cache,
        page_table,
        cache_seqlens,
        cu_seqlens_q,
        causal=True,
        fixed_split_size=fixed_split_size,
    )
    workspace = allocate_paged_attention_workspace_for_plan(plan, total_q=q.shape[0])

    b12x_paged_attention_forward(
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

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        b12x_paged_attention_forward(
            q,
            k_cache,
            v_cache,
            page_table,
            cache_seqlens,
            cu_seqlens_q,
            workspace=workspace,
            plan=plan,
            output=workspace.output,
        )

    ref_out_1, ref_lse_1 = paged_attention_reference(
        q,
        k_cache,
        v_cache,
        page_table,
        cache_seqlens,
        cu_seqlens_q,
        causal=True,
    )
    graph.replay()
    torch.cuda.synchronize()
    assert (workspace.output - ref_out_1).abs().max().item() <= 0.02
    assert (_lse_base2_to_natural(workspace.lse.transpose(0, 1)) - ref_lse_1).abs().max().item() <= 0.03
    assert _cosine_similarity(workspace.output, ref_out_1) >= 0.99999

    torch.manual_seed(79)
    q.copy_(torch.randn_like(q) / 4)
    k_cache.copy_(torch.randn_like(k_cache) / 4)
    v_cache.copy_(torch.randn_like(v_cache) / 4)
    ref_out_2, ref_lse_2 = paged_attention_reference(
        q,
        k_cache,
        v_cache,
        page_table,
        cache_seqlens,
        cu_seqlens_q,
        causal=True,
    )
    graph.replay()
    torch.cuda.synchronize()
    assert (workspace.output - ref_out_2).abs().max().item() <= 0.02
    assert (_lse_base2_to_natural(workspace.lse.transpose(0, 1)) - ref_lse_2).abs().max().item() <= 0.03
    assert _cosine_similarity(workspace.output, ref_out_2) >= 0.99999


@torch.inference_mode()
def test_paged_attention_fp8_kv_replays_under_cuda_graph() -> None:
    require_sm120()
    clear_attention_caches()

    q, k_cache, v_cache, page_table, cache_seqlens, cu_seqlens_q = _make_paged_inputs(
        q_seqlens=[6, 5, 7, 4],
        cache_seqlens=[97, 81, 113, 68],
        page_size=64,
        seed=83,
    )
    k_fp8, v_fp8, k_descale, v_descale = _quantize_paged_kv_cache_e4m3(
        k_cache,
        v_cache,
        page_table,
        cache_seqlens,
    )
    plan = create_paged_attention_plan(
        q,
        k_fp8,
        v_fp8,
        page_table,
        cache_seqlens,
        cu_seqlens_q,
        causal=True,
    )
    workspace = allocate_paged_attention_workspace_for_plan(plan, total_q=q.shape[0])

    b12x_paged_attention_forward(
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

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        b12x_paged_attention_forward(
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
            output=workspace.output,
        )

    ref_out_1, ref_lse_1 = paged_attention_reference(
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
    graph.replay()
    torch.cuda.synchronize()
    assert (workspace.output - ref_out_1).abs().max().item() <= 0.05
    assert (_lse_base2_to_natural(workspace.lse.transpose(0, 1)) - ref_lse_1).abs().max().item() <= 0.05
    assert _cosine_similarity(workspace.output, ref_out_1) >= 0.9999

    q_2, k_cache_2, v_cache_2, _, _, _ = _make_paged_inputs(
        q_seqlens=[6, 5, 7, 4],
        cache_seqlens=[97, 81, 113, 68],
        page_size=64,
        seed=89,
        page_table_width=page_table.shape[1],
        num_pages=k_cache.shape[0],
    )
    k_fp8_2, v_fp8_2, k_descale_2, v_descale_2 = _quantize_paged_kv_cache_e4m3(
        k_cache_2,
        v_cache_2,
        page_table,
        cache_seqlens,
    )
    q.copy_(q_2)
    k_fp8.copy_(k_fp8_2)
    v_fp8.copy_(v_fp8_2)
    k_descale.copy_(k_descale_2)
    v_descale.copy_(v_descale_2)

    ref_out_2, ref_lse_2 = paged_attention_reference(
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
    graph.replay()
    torch.cuda.synchronize()
    assert (workspace.output - ref_out_2).abs().max().item() <= 0.05
    assert (_lse_base2_to_natural(workspace.lse.transpose(0, 1)) - ref_lse_2).abs().max().item() <= 0.05
    assert _cosine_similarity(workspace.output, ref_out_2) >= 0.9999
