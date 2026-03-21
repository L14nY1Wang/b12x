from __future__ import annotations

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


@torch.inference_mode()
@pytest.mark.parametrize("num_splits", [1, 4])
def test_paged_attention_replays_under_cuda_graph_with_dynamic_metadata(num_splits: int) -> None:
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
        num_splits=num_splits,
    )
    workspace = allocate_paged_attention_workspace_for_plan(plan)

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
    assert (workspace.lse.transpose(0, 1) - ref_lse_1).abs().max().item() <= 0.03
    assert _cosine_similarity(workspace.output, ref_out_1) >= 0.99999

    torch.manual_seed(79)
    q.copy_(torch.randn_like(q) / 4)
    k_cache.copy_(torch.randn_like(k_cache) / 4)
    v_cache.copy_(torch.randn_like(v_cache) / 4)
    q_seqlens_2 = [4, 8, 5, 5]
    cache_seqlens_2 = [64, 96, 128, 70]
    cu_seqlens_q.copy_(
        torch.tensor([0, 4, 12, 17, 22], dtype=torch.int32, device=q.device)
    )
    cache_seqlens.copy_(torch.tensor(cache_seqlens_2, dtype=torch.int32, device=q.device))
    assert sum(q_seqlens_2) == q.shape[0]
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
    assert (workspace.lse.transpose(0, 1) - ref_lse_2).abs().max().item() <= 0.03
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
        num_splits=1,
    )
    workspace = allocate_paged_attention_workspace_for_plan(plan)

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
    assert (workspace.lse.transpose(0, 1) - ref_lse_1).abs().max().item() <= 0.05
    assert _cosine_similarity(workspace.output, ref_out_1) >= 0.9999

    q_2, k_cache_2, v_cache_2, _, cache_seqlens_2, cu_seqlens_q_2 = _make_paged_inputs(
        q_seqlens=[4, 8, 5, 5],
        cache_seqlens=[64, 96, 128, 70],
        page_size=64,
        seed=89,
        page_table_width=page_table.shape[1],
        num_pages=k_cache.shape[0],
    )
    k_fp8_2, v_fp8_2, k_descale_2, v_descale_2 = _quantize_paged_kv_cache_e4m3(
        k_cache_2,
        v_cache_2,
        page_table,
        cache_seqlens_2,
    )
    q.copy_(q_2)
    k_fp8.copy_(k_fp8_2)
    v_fp8.copy_(v_fp8_2)
    cache_seqlens.copy_(cache_seqlens_2)
    cu_seqlens_q.copy_(cu_seqlens_q_2)
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
    assert (workspace.lse.transpose(0, 1) - ref_lse_2).abs().max().item() <= 0.05
    assert _cosine_similarity(workspace.output, ref_out_2) >= 0.9999
