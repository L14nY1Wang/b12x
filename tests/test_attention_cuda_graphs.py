from __future__ import annotations

import torch

from b12x.attention.reference import attention_reference, paged_attention_reference
from b12x.integration.attention import (
    allocate_attention_workspace,
    allocate_paged_attention_workspace,
    b12x_attention_forward,
    b12x_paged_attention_forward,
    clear_attention_caches,
)

from .helpers import require_sm120
from .test_paged_attention_workspace_api import _make_paged_inputs


def _cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> float:
    a_f = a.to(torch.float32).reshape(-1)
    b_f = b.to(torch.float32).reshape(-1)
    return torch.nn.functional.cosine_similarity(a_f, b_f, dim=0).item()


def test_contiguous_attention_replays_under_cuda_graph() -> None:
    require_sm120()
    clear_attention_caches()

    torch.manual_seed(71)
    q = torch.randn(1, 48, 8, 256, device="cuda", dtype=torch.bfloat16) / 4
    k = torch.randn(1, 48, 1, 256, device="cuda", dtype=torch.bfloat16) / 4
    v = torch.randn(1, 48, 1, 256, device="cuda", dtype=torch.bfloat16) / 4
    workspace = allocate_attention_workspace(q, k, v, causal=True)

    b12x_attention_forward(q, k, v, workspace=workspace)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        b12x_attention_forward(q, k, v, workspace=workspace)

    graph.replay()
    torch.cuda.synchronize()

    ref_out, ref_lse = attention_reference(q, k, v, causal=True)
    assert (workspace.output - ref_out).abs().max().item() <= 0.02
    assert (workspace.lse - ref_lse).abs().max().item() <= 0.03
    assert _cosine_similarity(workspace.output, ref_out) >= 0.99999


def test_paged_attention_replays_under_cuda_graph_with_dynamic_metadata() -> None:
    require_sm120()
    clear_attention_caches()

    q, k_cache, v_cache, page_table, cache_seqlens, cu_seqlens_q = _make_paged_inputs(
        q_seqlens=[6, 5, 7, 4],
        cache_seqlens=[97, 81, 113, 68],
        page_size=64,
        seed=73,
    )
    workspace = allocate_paged_attention_workspace(
        q,
        k_cache,
        v_cache,
        page_table,
        cache_seqlens,
        cu_seqlens_q,
        causal=True,
    )

    b12x_paged_attention_forward(
        q,
        k_cache,
        v_cache,
        page_table,
        cache_seqlens,
        cu_seqlens_q,
        workspace=workspace,
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
