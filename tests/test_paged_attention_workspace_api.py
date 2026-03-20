from __future__ import annotations

import pytest
import torch

from b12x.attention.reference import paged_attention_reference
from b12x.integration.attention import (
    allocate_paged_attention_workspace_pool,
    allocate_paged_attention_workspace_for_plan,
    b12x_paged_decode,
    b12x_paged_extend,
    b12x_paged_attention_forward,
    choose_paged_attention_num_splits,
    clear_attention_caches,
    create_paged_attention_plan,
    infer_paged_attention_mode,
)

from .helpers import require_sm120


def _cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> float:
    a_f = a.to(torch.float32).reshape(-1)
    b_f = b.to(torch.float32).reshape(-1)
    return torch.nn.functional.cosine_similarity(a_f, b_f, dim=0).item()


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
    batch, max_pages = page_table.shape
    _, _page_size, kv_heads, _head_dim = k_cache.shape
    finfo = torch.finfo(torch.float8_e4m3fn)
    k_quant = torch.empty_like(k_cache, dtype=torch.float8_e4m3fn)
    v_quant = torch.empty_like(v_cache, dtype=torch.float8_e4m3fn)
    k_descale = torch.ones((batch, kv_heads), dtype=torch.float32, device=k_cache.device)
    v_descale = torch.ones((batch, kv_heads), dtype=torch.float32, device=v_cache.device)
    for request_idx in range(batch):
        cache_len = int(cache_seqlens[request_idx].item())
        num_pages = (cache_len + k_cache.shape[1] - 1) // k_cache.shape[1]
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


@pytest.mark.parametrize("num_splits", [1, 4])
def test_paged_workspace_matches_reference_for_qwen_like_extend_shape(num_splits: int) -> None:
    require_sm120()
    clear_attention_caches()

    q, k_cache, v_cache, page_table, cache_seqlens, cu_seqlens_q = _make_paged_inputs(
        q_seqlens=[6, 5, 7, 4],
        cache_seqlens=[97, 81, 113, 68],
        page_size=64,
        seed=23,
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
    out, lse = b12x_paged_attention_forward(
        q,
        k_cache,
        v_cache,
        page_table,
        cache_seqlens,
        cu_seqlens_q,
        workspace=workspace,
        plan=plan,
    )
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
    assert (lse - ref_lse).abs().max().item() <= 0.03
    assert _cosine_similarity(out, ref_out) >= 0.99999


def test_paged_plan_exposes_logical_gqa_dimensions() -> None:
    require_sm120()
    clear_attention_caches()

    q_seqlens = [6, 5, 7, 4]
    cache_seqlens = [97, 81, 113, 68]
    page_size = 64
    q, k_cache, v_cache, page_table, cache_seqlens_t, cu_seqlens_q = _make_paged_inputs(
        q_seqlens=q_seqlens,
        cache_seqlens=cache_seqlens,
        page_size=page_size,
        seed=29,
    )
    plan = create_paged_attention_plan(
        q,
        k_cache,
        v_cache,
        page_table,
        cache_seqlens_t,
        cu_seqlens_q,
        causal=True,
    )

    assert plan.num_batch == len(q_seqlens)
    assert plan.num_q_heads == 8
    assert plan.num_kv_heads == 1
    assert plan.qhead_per_kvhead == 8
    assert plan.seqlen_q_static == sum(q_seqlens)
    assert plan.seqlen_k_static == page_size * page_table.shape[1]
    assert plan.logical_q_rows_static == sum(q_seqlens) * 8
    assert plan.logical_total_q_rows == sum(q_seqlens) * 8
    assert plan.mode == "extend"
    assert plan.kernel_family == "main"
    assert plan.tile_m == 32
    assert plan.num_compute_warps == 2
    assert plan.num_stages == 1
    assert plan.q_in_regs is False
    assert plan.kv_dtype == torch.bfloat16


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
    out, lse = b12x_paged_attention_forward(
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

    assert plan.kv_dtype == torch.float8_e4m3fn
    assert (out - ref_out).abs().max().item() <= 0.05
    assert (lse - ref_lse).abs().max().item() <= 0.05
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


def test_decode_plan_selects_decode_micro_kernel_family() -> None:
    require_sm120()
    clear_attention_caches()

    q, k_cache, v_cache, page_table, cache_seqlens, cu_seqlens_q = _make_paged_inputs(
        q_seqlens=[1, 1, 1, 1],
        cache_seqlens=[64, 96, 128, 70],
        page_size=64,
        seed=39,
    )
    plan = create_paged_attention_plan(
        q,
        k_cache,
        v_cache,
        page_table,
        cache_seqlens,
        cu_seqlens_q,
        causal=True,
    )

    assert plan.mode == "decode"
    assert plan.kernel_family == "decode_micro"
    assert plan.tile_m == 16
    assert plan.tile_n == 64
    assert plan.num_compute_warps == 1
    assert plan.q_in_regs is True


def test_long_decode_plan_falls_back_to_main_kernel_family() -> None:
    require_sm120()
    clear_attention_caches()

    q, k_cache, v_cache, page_table, cache_seqlens, cu_seqlens_q = _make_paged_inputs(
        q_seqlens=[1, 1, 1, 1],
        cache_seqlens=[512, 768, 640, 1024],
        page_size=64,
        seed=45,
    )
    plan = create_paged_attention_plan(
        q,
        k_cache,
        v_cache,
        page_table,
        cache_seqlens,
        cu_seqlens_q,
        causal=True,
    )

    assert plan.mode == "decode"
    assert plan.kernel_family == "main"
    assert plan.tile_m == 32
    assert plan.num_compute_warps == 2
    assert plan.q_in_regs is False


def test_paged_workspace_pool_reuses_plan_exact_shape() -> None:
    require_sm120()
    clear_attention_caches()

    q, k_cache, v_cache, page_table, cache_seqlens, cu_seqlens_q = _make_paged_inputs(
        q_seqlens=[6, 5, 7, 4],
        cache_seqlens=[97, 81, 113, 68],
        page_size=64,
        seed=41,
    )
    plan = create_paged_attention_plan(
        q,
        k_cache,
        v_cache,
        page_table,
        cache_seqlens,
        cu_seqlens_q,
        causal=True,
    )
    pool = allocate_paged_attention_workspace_pool()

    out0, lse0 = b12x_paged_attention_forward(
        q,
        k_cache,
        v_cache,
        page_table,
        cache_seqlens,
        cu_seqlens_q,
        workspace=pool,
        plan=plan,
    )
    out1, lse1 = b12x_paged_attention_forward(
        q,
        k_cache,
        v_cache,
        page_table,
        cache_seqlens,
        cu_seqlens_q,
        workspace=pool,
        plan=plan,
    )

    assert out0.data_ptr() == out1.data_ptr()
    assert lse0.data_ptr() == lse1.data_ptr()
    assert len(pool.workspaces) == 1


def test_paged_workspace_pool_requires_explicit_plan() -> None:
    require_sm120()
    clear_attention_caches()

    q, k_cache, v_cache, page_table, cache_seqlens, cu_seqlens_q = _make_paged_inputs(
        q_seqlens=[6, 5, 7, 4],
        cache_seqlens=[97, 81, 113, 68],
        page_size=64,
        seed=43,
    )
    pool = allocate_paged_attention_workspace_pool()

    with pytest.raises(TypeError, match="require an explicit PagedAttentionPlan"):
        b12x_paged_attention_forward(
            q,
            k_cache,
            v_cache,
            page_table,
            cache_seqlens,
            cu_seqlens_q,
            workspace=pool,
        )


def test_paged_decode_and_extend_surfaces_validate_mode() -> None:
    require_sm120()
    clear_attention_caches()

    decode_inputs = _make_paged_inputs(
        q_seqlens=[1, 1, 1, 1],
        cache_seqlens=[64, 96, 128, 70],
        page_size=64,
        seed=47,
    )
    extend_inputs = _make_paged_inputs(
        q_seqlens=[6, 5, 7, 4],
        cache_seqlens=[97, 81, 113, 68],
        page_size=64,
        seed=49,
    )
    q_d, k_d, v_d, pt_d, cs_d, cu_d = decode_inputs
    q_e, k_e, v_e, pt_e, cs_e, cu_e = extend_inputs
    decode_plan = create_paged_attention_plan(q_d, k_d, v_d, pt_d, cs_d, cu_d, causal=True)
    extend_plan = create_paged_attention_plan(q_e, k_e, v_e, pt_e, cs_e, cu_e, causal=True)
    decode_workspace = allocate_paged_attention_workspace_for_plan(decode_plan)
    extend_workspace = allocate_paged_attention_workspace_for_plan(extend_plan)

    b12x_paged_decode(
        q_d,
        k_d,
        v_d,
        pt_d,
        cs_d,
        cu_d,
        workspace=decode_workspace,
        plan=decode_plan,
    )
    b12x_paged_extend(
        q_e,
        k_e,
        v_e,
        pt_e,
        cs_e,
        cu_e,
        workspace=extend_workspace,
        plan=extend_plan,
    )

    with pytest.raises(ValueError, match="expected a decode plan"):
        b12x_paged_decode(
            q_e,
            k_e,
            v_e,
            pt_e,
            cs_e,
            cu_e,
            workspace=extend_workspace,
            plan=extend_plan,
        )
    with pytest.raises(ValueError, match="expected an extend plan"):
        b12x_paged_extend(
            q_d,
            k_d,
            v_d,
            pt_d,
            cs_d,
            cu_d,
            workspace=decode_workspace,
            plan=decode_plan,
        )


def test_exact_paged_workspace_rejects_shape_mismatch() -> None:
    require_sm120()
    clear_attention_caches()

    q0, k0, v0, page_table0, cache_seqlens0, cu_seqlens_q0 = _make_paged_inputs(
        q_seqlens=[2, 2],
        cache_seqlens=[65, 65],
        page_size=64,
        seed=37,
        page_table_width=3,
        num_pages=10,
    )
    q1, k1, v1, page_table1, cache_seqlens1, cu_seqlens_q1 = _make_paged_inputs(
        q_seqlens=[2, 3],
        cache_seqlens=[65, 129],
        page_size=64,
        seed=41,
        page_table_width=3,
        num_pages=10,
    )
    plan0 = create_paged_attention_plan(
        q0,
        k0,
        v0,
        page_table0,
        cache_seqlens0,
        cu_seqlens_q0,
        causal=True,
    )
    workspace = allocate_paged_attention_workspace_for_plan(plan0)

    with pytest.raises(ValueError, match="paged attention plan mismatch"):
        b12x_paged_attention_forward(
            q1,
            k1,
            v1,
            page_table1,
            cache_seqlens1,
            cu_seqlens_q1,
            workspace=workspace,
            plan=plan0,
        )


def test_single_token_single_key_paged_corner_is_rejected() -> None:
    require_sm120()
    clear_attention_caches()

    q, k_cache, v_cache, page_table, cache_seqlens, cu_seqlens_q = _make_paged_inputs(
        q_seqlens=[1],
        cache_seqlens=[1],
        page_size=64,
        seed=53,
    )

    with pytest.raises(ValueError, match="single-token single-key corner"):
        create_paged_attention_plan(
            q,
            k_cache,
            v_cache,
            page_table,
            cache_seqlens,
            cu_seqlens_q,
            causal=True,
        )


def test_invalid_num_splits_is_rejected() -> None:
    require_sm120()
    clear_attention_caches()

    q, k_cache, v_cache, page_table, cache_seqlens, cu_seqlens_q = _make_paged_inputs(
        q_seqlens=[4, 4],
        cache_seqlens=[64, 64],
        page_size=64,
        seed=71,
    )

    with pytest.raises(ValueError, match="num_splits must be one of"):
        create_paged_attention_plan(
            q,
            k_cache,
            v_cache,
            page_table,
            cache_seqlens,
            cu_seqlens_q,
            causal=True,
            num_splits=3,
        )


def test_auto_split_bucket_selection_is_deterministic() -> None:
    require_sm120()
    clear_attention_caches()

    q0, k0, v0, page_table0, cache_seqlens0, cu_seqlens_q0 = _make_paged_inputs(
        q_seqlens=[1] * 8,
        cache_seqlens=[64] * 8,
        page_size=64,
        seed=83,
    )
    q1, k1, v1, page_table1, cache_seqlens1, cu_seqlens_q1 = _make_paged_inputs(
        q_seqlens=[6] * 8,
        cache_seqlens=[2048] * 8,
        page_size=64,
        seed=89,
    )

    assert choose_paged_attention_num_splits(cache_seqlens0, page_size=64, mode="decode") == 1
    assert choose_paged_attention_num_splits(cache_seqlens1, page_size=64, mode="extend") == 8

    plan0 = create_paged_attention_plan(
        q0,
        k0,
        v0,
        page_table0,
        cache_seqlens0,
        cu_seqlens_q0,
        causal=True,
        num_splits=0,
    )
    plan1 = create_paged_attention_plan(
        q1,
        k1,
        v1,
        page_table1,
        cache_seqlens1,
        cu_seqlens_q1,
        causal=True,
        num_splits=0,
    )

    assert plan0.num_splits == 1
    assert plan1.num_splits == 8


def test_mode_aware_split_bucket_selection_matches_planner_policy() -> None:
    require_sm120()
    clear_attention_caches()

    short_decode = torch.tensor([64] * 8, dtype=torch.int32, device="cuda")
    mid_decode = torch.tensor([128] * 8, dtype=torch.int32, device="cuda")
    long_decode = torch.tensor([2048] * 8, dtype=torch.int32, device="cuda")
    ultra_long_decode = torch.tensor([32768] * 2, dtype=torch.int32, device="cuda")
    short_extend = torch.tensor([128] * 8, dtype=torch.int32, device="cuda")
    mid_extend = torch.tensor([256] * 8, dtype=torch.int32, device="cuda")
    long_extend = torch.tensor([2048] * 8, dtype=torch.int32, device="cuda")
    ultra_long_extend = torch.tensor([32768] * 2, dtype=torch.int32, device="cuda")

    assert choose_paged_attention_num_splits(short_decode, page_size=64, mode="decode") == 1
    assert choose_paged_attention_num_splits(mid_decode, page_size=64, mode="decode") == 2
    assert choose_paged_attention_num_splits(long_decode, page_size=64, mode="decode") == 8
    assert (
        choose_paged_attention_num_splits(
            ultra_long_decode,
            page_size=64,
            mode="decode",
            kv_dtype=torch.float8_e4m3fn,
        )
        == 24
    )
    assert choose_paged_attention_num_splits(short_extend, page_size=64, mode="extend") == 1
    assert choose_paged_attention_num_splits(mid_extend, page_size=64, mode="extend") == 4
    assert choose_paged_attention_num_splits(long_extend, page_size=64, mode="extend") == 8
    assert (
        choose_paged_attention_num_splits(
            ultra_long_extend,
            page_size=64,
            mode="extend",
            kv_dtype=torch.float8_e4m3fn,
        )
        == 24
    )


def test_page_size_other_than_64_is_rejected() -> None:
    require_sm120()
    clear_attention_caches()

    q, k_cache, v_cache, page_table, cache_seqlens, cu_seqlens_q = _make_paged_inputs(
        q_seqlens=[4, 4],
        cache_seqlens=[32, 64],
        page_size=32,
        seed=61,
    )

    with pytest.raises(ValueError, match="page_size=64"):
        create_paged_attention_plan(
            q,
            k_cache,
            v_cache,
            page_table,
            cache_seqlens,
            cu_seqlens_q,
            causal=True,
        )
