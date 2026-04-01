from __future__ import annotations

import math

import cutlass.base_dsl.dsl as cutlass_dsl
import pytest
import torch

from b12x.attention.reference import paged_attention_reference
from b12x.integration.attention import (
    PagedAttentionWorkspace,
    clear_attention_caches,
    infer_paged_attention_mode,
)

from .helpers import require_sm120


def _cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> float:
    a_f = a.to(torch.float32).reshape(-1)
    b_f = b.to(torch.float32).reshape(-1)
    return torch.nn.functional.cosine_similarity(a_f, b_f, dim=0).item()


def _lse_base2_to_natural(lse: torch.Tensor) -> torch.Tensor:
    return lse * math.log(2.0)


def _make_paged_inputs(
    *,
    q_seqlens: list[int],
    cache_seqlens: list[int],
    page_size: int,
    q_heads: int = 8,
    kv_heads: int = 1,
    head_dim: int = 256,
    dtype: torch.dtype = torch.bfloat16,
    seed: int = 0,
    page_table_width: int | None = None,
    num_pages: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if len(q_seqlens) != len(cache_seqlens):
        raise ValueError("q_seqlens and cache_seqlens must have the same length")
    torch.manual_seed(seed)
    device = "cuda"
    batch = len(q_seqlens)
    total_q = sum(q_seqlens)
    q = torch.randn(total_q, q_heads, head_dim, device=device, dtype=dtype) / 4

    pages_per_request = [(cache_len + page_size - 1) // page_size for cache_len in cache_seqlens]
    max_pages = max(pages_per_request, default=0)
    if page_table_width is not None:
        if page_table_width < max_pages:
            raise ValueError(
                f"page_table_width={page_table_width} is smaller than the required max_pages={max_pages}"
            )
        max_pages = page_table_width
    total_pages_needed = sum(pages_per_request)
    if num_pages is None:
        num_pages = max(1, total_pages_needed * 2)
    if num_pages < total_pages_needed:
        raise ValueError(f"num_pages={num_pages} is smaller than the required total {total_pages_needed}")

    k_cache = torch.randn(num_pages, page_size, kv_heads, head_dim, device=device, dtype=dtype) / 4
    v_cache = torch.randn(num_pages, page_size, kv_heads, head_dim, device=device, dtype=dtype) / 4
    page_table = torch.zeros(batch, max_pages, dtype=torch.int32, device=device)
    page_order = torch.randperm(num_pages, device=device)
    cursor = 0
    for request_idx, num_req_pages in enumerate(pages_per_request):
        if num_req_pages == 0:
            continue
        page_ids = page_order[cursor : cursor + num_req_pages].to(torch.int32)
        cursor += num_req_pages
        page_table[request_idx, :num_req_pages] = page_ids
        page_table[request_idx, num_req_pages:] = page_ids[-1]

    cache_seqlens_t = torch.tensor(cache_seqlens, dtype=torch.int32, device=device)
    q_offsets = [0]
    for q_len in q_seqlens:
        q_offsets.append(q_offsets[-1] + q_len)
    cu_seqlens_q = torch.tensor(q_offsets, dtype=torch.int32, device=device)
    return q, k_cache, v_cache, page_table, cache_seqlens_t, cu_seqlens_q


def _quantize_paged_kv_cache_e4m3(
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    page_table: torch.Tensor,
    cache_seqlens: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    batch, _max_pages = page_table.shape
    _, page_size, kv_heads, _head_dim = k_cache.shape
    finfo = torch.finfo(torch.float8_e4m3fn)
    k_quant = torch.empty_like(k_cache, dtype=torch.float8_e4m3fn)
    v_quant = torch.empty_like(v_cache, dtype=torch.float8_e4m3fn)
    k_descale = torch.ones((batch, kv_heads), dtype=torch.float32, device=k_cache.device)
    v_descale = torch.ones((batch, kv_heads), dtype=torch.float32, device=v_cache.device)
    for request_idx in range(batch):
        cache_len = int(cache_seqlens[request_idx].item())
        num_pages = (cache_len + page_size - 1) // page_size
        if num_pages == 0:
            continue
        page_ids = page_table[request_idx, :num_pages].to(torch.long)
        k_pages = k_cache.index_select(0, page_ids).to(torch.float32)
        v_pages = v_cache.index_select(0, page_ids).to(torch.float32)
        k_scale = k_pages.abs().amax(dim=(0, 1, 3)) / finfo.max
        v_scale = v_pages.abs().amax(dim=(0, 1, 3)) / finfo.max
        k_scale = torch.where(k_scale > 0, k_scale, torch.ones_like(k_scale))
        v_scale = torch.where(v_scale > 0, v_scale, torch.ones_like(v_scale))
        k_descale[request_idx] = k_scale
        v_descale[request_idx] = v_scale
        k_quant[page_ids] = (k_pages / k_scale.view(1, 1, kv_heads, 1)).clamp(
            min=finfo.min,
            max=finfo.max,
        ).to(torch.float8_e4m3fn)
        v_quant[page_ids] = (v_pages / v_scale.view(1, 1, kv_heads, 1)).clamp(
            min=finfo.min,
            max=finfo.max,
        ).to(torch.float8_e4m3fn)
    return k_quant.contiguous(), v_quant.contiguous(), k_descale.contiguous(), v_descale.contiguous()


def _make_workspace(
    *,
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    use_cuda_graph: bool = False,
    attn_mode: str | None = None,
) -> PagedAttentionWorkspace:
    return PagedAttentionWorkspace.for_tensors(
        mode=infer_paged_attention_mode(cu_seqlens_q),
        q=q,
        k_cache=k_cache,
        v_cache=v_cache,
        use_cuda_graph=use_cuda_graph,
        attn_mode=attn_mode,
    )


@pytest.mark.parametrize("fixed_split_size", [None, 4])
def test_paged_workspace_matches_reference_for_qwen_like_extend_shape(
    fixed_split_size: int | None,
) -> None:
    require_sm120()
    clear_attention_caches()

    q, k_cache, v_cache, page_table, cache_seqlens, cu_seqlens_q = _make_paged_inputs(
        q_seqlens=[6, 5, 7, 4],
        cache_seqlens=[97, 81, 113, 68],
        page_size=64,
        seed=23,
    )
    workspace = _make_workspace(
        q=q,
        k_cache=k_cache,
        v_cache=v_cache,
        cu_seqlens_q=cu_seqlens_q,
    )
    workspace.prepare(
        page_table,
        cache_seqlens,
        cu_seqlens_q,
        fixed_split_size=fixed_split_size,
    )
    output = torch.empty_like(q)
    out, lse = workspace.run(q, k_cache, v_cache, output=output)
    ref_out, ref_lse = paged_attention_reference(
        q,
        k_cache,
        v_cache,
        page_table,
        cache_seqlens,
        cu_seqlens_q,
        causal=True,
    )
    torch.cuda.synchronize()

    assert (out - ref_out).abs().max().item() <= 0.02
    assert (_lse_base2_to_natural(lse) - ref_lse).abs().max().item() <= 0.03
    assert _cosine_similarity(out, ref_out) >= 0.99999


def test_paged_workspace_exposes_primary_backend_metadata() -> None:
    require_sm120()
    clear_attention_caches()

    q, k_cache, v_cache, page_table, cache_seqlens_t, cu_seqlens_q = _make_paged_inputs(
        q_seqlens=[6, 5, 7, 4],
        cache_seqlens=[97, 81, 113, 68],
        page_size=64,
        seed=29,
    )
    workspace = _make_workspace(
        q=q,
        k_cache=k_cache,
        v_cache=v_cache,
        cu_seqlens_q=cu_seqlens_q,
    )
    workspace.prepare(page_table, cache_seqlens_t, cu_seqlens_q)
    plan = workspace.plan

    assert plan.num_q_heads == 8
    assert plan.num_kv_heads == 1
    assert plan.gqa_group_size == 8
    assert plan.head_dim_qk == 256
    assert plan.head_dim_vo == 256
    assert plan.mode == "extend"
    assert plan.cta_tile_q == 64
    assert plan.kv_chunk_size == 64
    assert plan.split_kv is True
    assert plan.total_q == q.shape[0]
    assert plan.page_table_shape == tuple(page_table.shape)


def test_paged_workspace_preserves_opt_in_attention_mode() -> None:
    require_sm120()
    clear_attention_caches()

    q, k_cache, v_cache, _page_table, _cache_seqlens_t, cu_seqlens_q = _make_paged_inputs(
        q_seqlens=[1, 1, 1, 1],
        cache_seqlens=[64, 64, 64, 64],
        page_size=64,
        seed=31,
    )
    workspace = _make_workspace(
        q=q,
        k_cache=k_cache,
        v_cache=v_cache,
        cu_seqlens_q=cu_seqlens_q,
        use_cuda_graph=True,
        attn_mode="turbo",
    )

    assert workspace.attn_mode == "turbo"
    assert workspace.use_cuda_graph is True


def test_paged_workspace_matches_reference_for_fp8_kv_cache() -> None:
    require_sm120()
    clear_attention_caches()

    q, k_cache, v_cache, page_table, cache_seqlens, cu_seqlens_q = _make_paged_inputs(
        q_seqlens=[6, 5, 7, 4],
        cache_seqlens=[97, 81, 113, 68],
        page_size=64,
        seed=123,
    )
    k_fp8, v_fp8, k_descale, v_descale = _quantize_paged_kv_cache_e4m3(
        k_cache,
        v_cache,
        page_table,
        cache_seqlens,
    )
    workspace = _make_workspace(
        q=q,
        k_cache=k_fp8,
        v_cache=v_fp8,
        cu_seqlens_q=cu_seqlens_q,
    )
    workspace.prepare(page_table, cache_seqlens, cu_seqlens_q)
    output = torch.empty_like(q)
    out, lse = workspace.run(
        q,
        k_fp8,
        v_fp8,
        output=output,
        k_descale=k_descale,
        v_descale=v_descale,
    )
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
    torch.cuda.synchronize()

    assert workspace.plan.kv_dtype == torch.float8_e4m3fn
    assert (out - ref_out).abs().max().item() <= 0.05
    assert (_lse_base2_to_natural(lse) - ref_lse).abs().max().item() <= 0.05
    assert _cosine_similarity(out, ref_out) >= 0.9999


def test_paged_workspace_matches_reference_for_bf16_splitkv_extend_shape() -> None:
    require_sm120()
    clear_attention_caches()

    q, k_cache, v_cache, page_table, cache_seqlens, cu_seqlens_q = _make_paged_inputs(
        q_seqlens=[128],
        cache_seqlens=[128],
        page_size=64,
        q_heads=8,
        kv_heads=1,
        head_dim=256,
        seed=29,
    )
    workspace = _make_workspace(
        q=q,
        k_cache=k_cache,
        v_cache=v_cache,
        cu_seqlens_q=cu_seqlens_q,
    )
    workspace.prepare(page_table, cache_seqlens, cu_seqlens_q)

    assert workspace.plan.mode == "extend"
    assert workspace.plan.split_kv is True
    assert workspace.plan.kv_chunk_size == 64

    output = torch.empty_like(q)
    out, lse = workspace.run(q, k_cache, v_cache, output=output)
    ref_out, ref_lse = paged_attention_reference(
        q,
        k_cache,
        v_cache,
        page_table,
        cache_seqlens,
        cu_seqlens_q,
        causal=True,
    )
    torch.cuda.synchronize()

    assert (out - ref_out).abs().max().item() <= 0.02
    assert (_lse_base2_to_natural(lse) - ref_lse).abs().max().item() <= 0.02
    assert _cosine_similarity(out, ref_out) >= 0.9999


def test_paged_mode_inference_distinguishes_decode_from_extend() -> None:
    require_sm120()
    clear_attention_caches()

    _, _, _, _, _, cu_seqlens_decode = _make_paged_inputs(
        q_seqlens=[1, 1, 1, 1],
        cache_seqlens=[64, 64, 64, 64],
        page_size=64,
        seed=33,
    )
    _, _, _, _, _, cu_seqlens_extend = _make_paged_inputs(
        q_seqlens=[6, 5, 7, 4],
        cache_seqlens=[97, 81, 113, 68],
        page_size=64,
        seed=35,
    )

    assert infer_paged_attention_mode(cu_seqlens_decode) == "decode"
    assert infer_paged_attention_mode(cu_seqlens_extend) == "extend"


def test_decode_prepare_uses_small_q_tile() -> None:
    require_sm120()
    clear_attention_caches()

    q, k_cache, v_cache, page_table, cache_seqlens, cu_seqlens_q = _make_paged_inputs(
        q_seqlens=[1, 1, 1, 1],
        cache_seqlens=[64, 96, 128, 70],
        page_size=64,
        seed=39,
    )
    workspace = _make_workspace(
        q=q,
        k_cache=k_cache,
        v_cache=v_cache,
        cu_seqlens_q=cu_seqlens_q,
    )
    workspace.prepare(page_table, cache_seqlens, cu_seqlens_q)

    assert workspace.plan.mode == "decode"
    assert workspace.plan.cta_tile_q == 16
    assert workspace.plan.kv_chunk_size == 2 * 64


def test_workspace_eager_path_grows_with_larger_shape() -> None:
    require_sm120()
    clear_attention_caches()

    small = _make_paged_inputs(
        q_seqlens=[1, 1, 1, 1],
        cache_seqlens=[64, 64, 64, 64],
        page_size=64,
        seed=41,
    )
    large = _make_paged_inputs(
        q_seqlens=[1, 1, 1, 1],
        cache_seqlens=[2048, 2048, 4096, 4096],
        page_size=64,
        seed=43,
        page_table_width=64,
        num_pages=512,
    )
    q_s, k_s, v_s, pt_s, cs_s, cu_s = small
    q_l, k_l, v_l, pt_l, cs_l, cu_l = large
    workspace = _make_workspace(
        q=q_s,
        k_cache=k_s,
        v_cache=v_s,
        cu_seqlens_q=cu_s,
    )
    workspace.prepare(pt_s, cs_s, cu_s)
    request_items_small = int(workspace.request_indices.shape[0])
    merge_small = int(workspace.merge_indptr.shape[0])

    workspace.prepare(pt_l, cs_l, cu_l)
    assert int(workspace.request_indices.shape[0]) >= request_items_small
    assert int(workspace.merge_indptr.shape[0]) >= merge_small


def test_workspace_mode_validation_rejects_mismatched_prepare() -> None:
    require_sm120()
    clear_attention_caches()

    q, k_cache, v_cache, page_table, cache_seqlens, cu_seqlens_q = _make_paged_inputs(
        q_seqlens=[6, 5, 7, 4],
        cache_seqlens=[97, 81, 113, 68],
        page_size=64,
        seed=47,
    )
    workspace = PagedAttentionWorkspace.for_tensors(
        mode="decode",
        q=q[:4],
        k_cache=k_cache,
        v_cache=v_cache,
    )
    with pytest.raises(ValueError, match="workspace mode decode does not match prepared mode extend"):
        workspace.prepare(page_table, cache_seqlens, cu_seqlens_q)


def test_workspace_fixed_split_size_pins_chunk_pages() -> None:
    require_sm120()
    clear_attention_caches()

    q, k_cache, v_cache, page_table, cache_seqlens, cu_seqlens_q = _make_paged_inputs(
        q_seqlens=[1, 1],
        cache_seqlens=[2048, 4096],
        page_size=64,
        seed=51,
    )
    workspace = _make_workspace(
        q=q,
        k_cache=k_cache,
        v_cache=v_cache,
        cu_seqlens_q=cu_seqlens_q,
    )
    workspace.prepare(
        page_table,
        cache_seqlens,
        cu_seqlens_q,
        fixed_split_size=8,
    )

    assert workspace.plan.fixed_split_size == 8
    assert workspace.plan.kv_chunk_size == 8 * 64


def test_graph_workspace_reuses_stable_metadata_buffers_for_smaller_replay() -> None:
    require_sm120()
    clear_attention_caches()

    capture_inputs = _make_paged_inputs(
        q_seqlens=[6, 5, 7, 4],
        cache_seqlens=[2048, 2048, 4096, 4096],
        page_size=64,
        seed=59,
        page_table_width=64,
        num_pages=512,
    )
    replay_inputs = _make_paged_inputs(
        q_seqlens=[4, 4, 4, 4],
        cache_seqlens=[1024, 1536, 2048, 2048],
        page_size=64,
        seed=61,
        page_table_width=64,
        num_pages=512,
    )
    q0, k0, v0, pt0, cs0, cu0 = capture_inputs
    q1, k1, v1, pt1, cs1, cu1 = replay_inputs

    workspace = _make_workspace(
        q=q0,
        k_cache=k0,
        v_cache=v0,
        cu_seqlens_q=cu0,
        use_cuda_graph=True,
    )
    workspace.prepare(pt0, cs0, cu0)
    ptrs = (
        workspace.request_indices.data_ptr(),
        workspace.page_table.data_ptr(),
        workspace.cache_seqlens.data_ptr(),
        workspace.cu_seqlens_q.data_ptr(),
    )
    workspace.prepare(pt1, cs1, cu1)

    assert ptrs == (
        workspace.request_indices.data_ptr(),
        workspace.page_table.data_ptr(),
        workspace.cache_seqlens.data_ptr(),
        workspace.cu_seqlens_q.data_ptr(),
    )
    assert workspace.active_total_q == q1.shape[0]


def test_graph_workspace_uses_narrower_fp8_extend_chunks_at_8192() -> None:
    require_sm120()
    clear_attention_caches()

    q, k_cache, v_cache, page_table, cache_seqlens, cu_seqlens_q = _make_paged_inputs(
        q_seqlens=[6] * 8,
        cache_seqlens=[8192] * 8,
        page_size=64,
        seed=73,
        page_table_width=128,
        num_pages=2048,
    )
    k_fp8, v_fp8, _, _ = _quantize_paged_kv_cache_e4m3(
        k_cache,
        v_cache,
        page_table,
        cache_seqlens,
    )
    workspace = _make_workspace(
        q=q,
        k_cache=k_fp8,
        v_cache=v_fp8,
        cu_seqlens_q=cu_seqlens_q,
        use_cuda_graph=True,
    )
    workspace.prepare(page_table, cache_seqlens, cu_seqlens_q)

    assert workspace.plan.mode == "extend"
    assert workspace.plan.kv_dtype == torch.float8_e4m3fn
    assert workspace.plan.kv_chunk_size == 3 * 64
    assert workspace.plan.split_kv is True


def test_graph_workspace_rejects_capacity_growth() -> None:
    require_sm120()
    clear_attention_caches()

    small = _make_paged_inputs(
        q_seqlens=[1, 1],
        cache_seqlens=[64, 64],
        page_size=64,
        seed=67,
    )
    large = _make_paged_inputs(
        q_seqlens=[1, 1, 1, 1],
        cache_seqlens=[8192, 8192, 8192, 8192],
        page_size=64,
        seed=71,
        page_table_width=128,
        num_pages=1024,
    )
    q_s, k_s, v_s, pt_s, cs_s, cu_s = small
    q_l, k_l, v_l, pt_l, cs_l, cu_l = large
    workspace = _make_workspace(
        q=q_s,
        k_cache=k_s,
        v_cache=v_s,
        cu_seqlens_q=cu_s,
        use_cuda_graph=True,
    )
    workspace.prepare(pt_s, cs_s, cu_s)

    with pytest.raises(ValueError, match="graph-mode paged workspace capacity exceeded"):
        workspace.prepare(pt_l, cs_l, cu_l)


def _count_generate_original_ir_calls(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[list[int], object]:
    calls = [0]
    original_generate_original_ir = cutlass_dsl.BaseDSL.generate_original_ir

    def counted_generate_original_ir(self, *args, **kwargs):
        calls[0] += 1
        return original_generate_original_ir(self, *args, **kwargs)

    monkeypatch.setattr(cutlass_dsl.BaseDSL, "generate_original_ir", counted_generate_original_ir)
    return calls, original_generate_original_ir


def test_eager_workspace_reuses_compiled_host_launcher_for_identical_nosplit_shape(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    require_sm120()
    clear_attention_caches()

    q, k_cache, v_cache, page_table, cache_seqlens, cu_seqlens_q = _make_paged_inputs(
        q_seqlens=[4],
        cache_seqlens=[4],
        page_size=64,
        q_heads=48,
        kv_heads=8,
        head_dim=128,
        seed=79,
    )
    workspace = _make_workspace(
        q=q,
        k_cache=k_cache,
        v_cache=v_cache,
        cu_seqlens_q=cu_seqlens_q,
    )
    calls, _ = _count_generate_original_ir_calls(monkeypatch)

    workspace.prepare(page_table, cache_seqlens, cu_seqlens_q)
    workspace.run(q, k_cache, v_cache, output=torch.empty_like(q))
    first_run_calls = calls[0]

    workspace.prepare(page_table, cache_seqlens, cu_seqlens_q)
    workspace.run(q, k_cache, v_cache, output=torch.empty_like(q))

    assert first_run_calls > 0
    assert calls[0] == first_run_calls


def test_eager_workspace_reuses_compiled_host_launcher_for_identical_splitkv_shape(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    require_sm120()
    clear_attention_caches()

    q, k_cache, v_cache, page_table, cache_seqlens, cu_seqlens_q = _make_paged_inputs(
        q_seqlens=[6, 5, 7, 4],
        cache_seqlens=[97, 81, 113, 68],
        page_size=64,
        seed=83,
    )
    workspace = _make_workspace(
        q=q,
        k_cache=k_cache,
        v_cache=v_cache,
        cu_seqlens_q=cu_seqlens_q,
    )
    calls, _ = _count_generate_original_ir_calls(monkeypatch)

    workspace.prepare(page_table, cache_seqlens, cu_seqlens_q)
    assert workspace.plan.split_kv is True
    workspace.run(q, k_cache, v_cache, output=torch.empty_like(q))
    first_run_calls = calls[0]

    workspace.prepare(page_table, cache_seqlens, cu_seqlens_q)
    workspace.run(q, k_cache, v_cache, output=torch.empty_like(q))

    assert first_run_calls > 0
    assert calls[0] == first_run_calls
