from __future__ import annotations

import torch

from b12x.attention.paged.planner import create_paged_plan, infer_paged_mode
from b12x.integration.attention import PagedAttentionWorkspace


_EXPLICIT_TOKEN_LENGTHS = [
    1,
    2,
    4,
    8,
    16,
    32,
    64,
    128,
    256,
    512,
    1024,
    2048,
    4096,
    8192,
    16384,
    32768,
    65536,
    131072,
]


def _assert_chunk_table(
    *,
    q_seqlens: list[int],
    kv_dtype: torch.dtype,
    expected_chunk_pages_by_cache_len: dict[int, int],
) -> None:
    for cache_len in _EXPLICIT_TOKEN_LENGTHS:
        q, k_cache, v_cache, page_table, cache_seqlens, cu_seqlens_q = _make_inputs(
            q_seqlens=q_seqlens,
            cache_seqlens=[cache_len] * 8,
            kv_dtype=kv_dtype,
        )
        plan = create_paged_plan(
            q,
            k_cache,
            v_cache,
            page_table,
            cache_seqlens,
            cu_seqlens_q,
        )
        assert plan.kv_chunk_size == expected_chunk_pages_by_cache_len[cache_len] * 64


def _make_inputs(
    *,
    q_seqlens: list[int],
    cache_seqlens: list[int],
    page_size: int = 64,
    q_heads: int = 8,
    kv_heads: int = 1,
    head_dim_qk: int = 256,
    head_dim_vo: int = 256,
    dtype: torch.dtype = torch.bfloat16,
    kv_dtype: torch.dtype = torch.float8_e4m3fn,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    device = "cuda"
    batch = len(q_seqlens)
    total_q = sum(q_seqlens)
    q = torch.randn(total_q, q_heads, head_dim_qk, dtype=dtype, device=device)
    max_pages = max((cache_len + page_size - 1) // page_size for cache_len in cache_seqlens)
    num_pages = sum((cache_len + page_size - 1) // page_size for cache_len in cache_seqlens) + 8
    k_cache = torch.randn(
        num_pages, page_size, kv_heads, head_dim_qk, dtype=torch.float32, device=device
    ).to(kv_dtype)
    v_cache = torch.randn(
        num_pages, page_size, kv_heads, head_dim_vo, dtype=torch.float32, device=device
    ).to(kv_dtype)
    page_table = torch.zeros(batch, max_pages, dtype=torch.int32, device=device)
    cursor = 0
    for request_idx, cache_len in enumerate(cache_seqlens):
        req_pages = (cache_len + page_size - 1) // page_size
        page_ids = torch.arange(cursor, cursor + req_pages, dtype=torch.int32, device=device)
        cursor += req_pages
        page_table[request_idx, :req_pages] = page_ids
        page_table[request_idx, req_pages:] = page_ids[-1]
    cache_seqlens_t = torch.tensor(cache_seqlens, dtype=torch.int32, device=device)
    offsets = [0]
    for q_len in q_seqlens:
        offsets.append(offsets[-1] + q_len)
    cu_seqlens_q = torch.tensor(offsets, dtype=torch.int32, device=device)
    return q, k_cache, v_cache, page_table, cache_seqlens_t, cu_seqlens_q


def test_paged_infers_decode_mode() -> None:
    _, _, _, _, _, cu_seqlens_q = _make_inputs(q_seqlens=[1, 1, 1], cache_seqlens=[64, 128, 192])
    assert infer_paged_mode(cu_seqlens_q) == "decode"


def test_paged_infers_extend_mode() -> None:
    _, _, _, _, _, cu_seqlens_q = _make_inputs(q_seqlens=[1, 6], cache_seqlens=[64, 128])
    assert infer_paged_mode(cu_seqlens_q) == "extend"


def test_paged_decode_plan_emits_exact_split_metadata() -> None:
    q, k_cache, v_cache, page_table, cache_seqlens, cu_seqlens_q = _make_inputs(
        q_seqlens=[1, 1],
        cache_seqlens=[2048, 4096],
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

    assert plan.mode == "decode"
    assert plan.cta_tile_q == 16
    assert plan.split_kv is True
    assert plan.kv_chunk_size == 8 * 64
    assert plan.request_indices == (0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1)
    assert plan.qo_tile_indices == (0,) * 12
    assert plan.kv_tile_indices == (0, 1, 2, 3, 0, 1, 2, 3, 4, 5, 6, 7)
    assert plan.merge_indptr == (0, 4, 12)
    assert plan.o_indptr == (0, 4, 12)
    assert plan.total_num_partial_rows == 12


def test_paged_workspace_shapes_follow_plan_metadata() -> None:
    q, k_cache, v_cache, page_table, cache_seqlens, cu_seqlens_q = _make_inputs(
        q_seqlens=[1, 6],
        cache_seqlens=[2048, 8192],
    )
    plan = create_paged_plan(
        q,
        k_cache,
        v_cache,
        page_table,
        cache_seqlens,
        cu_seqlens_q,
        fixed_split_size=16,
    )
    workspace = PagedAttentionWorkspace.for_tensors(
        mode="extend",
        q=q,
        k_cache=k_cache,
        v_cache=v_cache,
    )
    workspace.prepare(
        page_table,
        cache_seqlens,
        cu_seqlens_q,
        fixed_split_size=16,
    )

    assert workspace.lse.shape == (8, 7)
    assert workspace.kv_chunk_size_ptr.item() == 16 * 64
    assert workspace.total_num_rows_ptr.item() == 7
    assert workspace.request_indices.shape[0] == plan.new_batch_size
    assert workspace.merge_indptr.shape[0] == plan.total_q + 1
    assert workspace.o_indptr.shape[0] == page_table.shape[0] + 1
    assert workspace.tmp_output is not None
    assert workspace.tmp_output.shape[0] == plan.total_num_partial_rows
    assert workspace.tmp_lse is not None
    assert workspace.tmp_lse.shape == (plan.total_num_partial_rows, plan.num_q_heads)


def test_paged_fp8_auto_chunk_heuristic_uses_larger_decode_chunks() -> None:
    q, k_cache, v_cache, page_table, cache_seqlens, cu_seqlens_q = _make_inputs(
        q_seqlens=[1] * 8,
        cache_seqlens=[8192] * 8,
    )
    plan = create_paged_plan(
        q,
        k_cache,
        v_cache,
        page_table,
        cache_seqlens,
        cu_seqlens_q,
    )

    assert plan.mode == "decode"
    assert plan.kv_chunk_size == 6 * 64
    assert plan.split_kv is True


def test_paged_fp8_auto_chunk_heuristic_uses_coarser_extend_chunks_at_very_long_context() -> None:
    q, k_cache, v_cache, page_table, cache_seqlens, cu_seqlens_q = _make_inputs(
        q_seqlens=[6] * 8,
        cache_seqlens=[32768] * 8,
    )
    plan = create_paged_plan(
        q,
        k_cache,
        v_cache,
        page_table,
        cache_seqlens,
        cu_seqlens_q,
    )

    assert plan.mode == "extend"
    assert plan.kv_chunk_size == 24 * 64
    assert plan.split_kv is True


def test_paged_fp8_auto_chunk_heuristic_keeps_mid_long_extend_chunks_stable() -> None:
    q, k_cache, v_cache, page_table, cache_seqlens, cu_seqlens_q = _make_inputs(
        q_seqlens=[6] * 8,
        cache_seqlens=[8192] * 8,
    )
    plan = create_paged_plan(
        q,
        k_cache,
        v_cache,
        page_table,
        cache_seqlens,
        cu_seqlens_q,
    )

    assert plan.mode == "extend"
    assert plan.kv_chunk_size == 6 * 64
    assert plan.split_kv is True


def test_paged_fp8_auto_chunk_heuristic_uses_single_page_chunks_for_small_mid_extend() -> None:
    q, k_cache, v_cache, page_table, cache_seqlens, cu_seqlens_q = _make_inputs(
        q_seqlens=[6] * 8,
        cache_seqlens=[512] * 8,
    )
    plan = create_paged_plan(
        q,
        k_cache,
        v_cache,
        page_table,
        cache_seqlens,
        cu_seqlens_q,
    )

    assert plan.mode == "extend"
    assert plan.kv_chunk_size == 1 * 64
    assert plan.split_kv is True


def test_paged_fp8_auto_chunk_heuristic_uses_two_page_chunks_for_mid_extend() -> None:
    q, k_cache, v_cache, page_table, cache_seqlens, cu_seqlens_q = _make_inputs(
        q_seqlens=[6] * 8,
        cache_seqlens=[2048] * 8,
    )
    plan = create_paged_plan(
        q,
        k_cache,
        v_cache,
        page_table,
        cache_seqlens,
        cu_seqlens_q,
    )

    assert plan.mode == "extend"
    assert plan.kv_chunk_size == 2 * 64
    assert plan.split_kv is True


def test_paged_fp8_auto_chunk_heuristic_uses_three_page_chunks_for_extend_4096() -> None:
    q, k_cache, v_cache, page_table, cache_seqlens, cu_seqlens_q = _make_inputs(
        q_seqlens=[6] * 8,
        cache_seqlens=[4096] * 8,
    )
    plan = create_paged_plan(
        q,
        k_cache,
        v_cache,
        page_table,
        cache_seqlens,
        cu_seqlens_q,
    )

    assert plan.mode == "extend"
    assert plan.kv_chunk_size == 3 * 64
    assert plan.split_kv is True


def test_paged_fp8_auto_chunk_heuristic_uses_coarser_decode_chunks_at_very_long_context() -> None:
    q, k_cache, v_cache, page_table, cache_seqlens, cu_seqlens_q = _make_inputs(
        q_seqlens=[1] * 8,
        cache_seqlens=[32768] * 8,
    )
    plan = create_paged_plan(
        q,
        k_cache,
        v_cache,
        page_table,
        cache_seqlens,
        cu_seqlens_q,
    )

    assert plan.mode == "decode"
    assert plan.kv_chunk_size == 24 * 64
    assert plan.split_kv is True


def test_paged_bf16_auto_chunk_heuristic_uses_single_page_chunks_for_small_extend() -> None:
    for cache_len in (512,):
        q, k_cache, v_cache, page_table, cache_seqlens, cu_seqlens_q = _make_inputs(
            q_seqlens=[6] * 8,
            cache_seqlens=[cache_len] * 8,
            kv_dtype=torch.bfloat16,
        )
        plan = create_paged_plan(
            q,
            k_cache,
            v_cache,
            page_table,
            cache_seqlens,
            cu_seqlens_q,
        )

        assert plan.mode == "extend"
        assert plan.kv_chunk_size == 1 * 64
        assert plan.split_kv is True


def test_paged_bf16_auto_chunk_heuristic_uses_two_pages_for_extend_2048() -> None:
    q, k_cache, v_cache, page_table, cache_seqlens, cu_seqlens_q = _make_inputs(
        q_seqlens=[6] * 8,
        cache_seqlens=[2048] * 8,
        kv_dtype=torch.bfloat16,
    )
    plan = create_paged_plan(
        q,
        k_cache,
        v_cache,
        page_table,
        cache_seqlens,
        cu_seqlens_q,
    )

    assert plan.mode == "extend"
    assert plan.kv_chunk_size == 2 * 64
    assert plan.split_kv is True


def test_paged_bf16_auto_chunk_heuristic_uses_three_pages_for_extend_4096() -> None:
    q, k_cache, v_cache, page_table, cache_seqlens, cu_seqlens_q = _make_inputs(
        q_seqlens=[6] * 8,
        cache_seqlens=[4096] * 8,
        kv_dtype=torch.bfloat16,
    )
    plan = create_paged_plan(
        q,
        k_cache,
        v_cache,
        page_table,
        cache_seqlens,
        cu_seqlens_q,
    )

    assert plan.mode == "extend"
    assert plan.kv_chunk_size == 3 * 64
    assert plan.split_kv is True


def test_paged_bf16_auto_chunk_heuristic_uses_three_pages_for_extend_8192() -> None:
    q, k_cache, v_cache, page_table, cache_seqlens, cu_seqlens_q = _make_inputs(
        q_seqlens=[6] * 8,
        cache_seqlens=[8192] * 8,
        kv_dtype=torch.bfloat16,
    )
    plan = create_paged_plan(
        q,
        k_cache,
        v_cache,
        page_table,
        cache_seqlens,
        cu_seqlens_q,
    )

    assert plan.mode == "extend"
    assert plan.kv_chunk_size == 6 * 64
    assert plan.split_kv is True


def test_paged_bf16_auto_chunk_heuristic_uses_six_pages_for_extend_16384() -> None:
    q, k_cache, v_cache, page_table, cache_seqlens, cu_seqlens_q = _make_inputs(
        q_seqlens=[6] * 8,
        cache_seqlens=[16384] * 8,
        kv_dtype=torch.bfloat16,
    )
    plan = create_paged_plan(
        q,
        k_cache,
        v_cache,
        page_table,
        cache_seqlens,
        cu_seqlens_q,
    )

    assert plan.mode == "extend"
    assert plan.kv_chunk_size == 6 * 64
    assert plan.split_kv is True


def test_paged_bf16_auto_chunk_heuristic_uses_thirty_two_pages_for_extend_32768() -> None:
    q, k_cache, v_cache, page_table, cache_seqlens, cu_seqlens_q = _make_inputs(
        q_seqlens=[6] * 8,
        cache_seqlens=[32768] * 8,
        kv_dtype=torch.bfloat16,
    )
    plan = create_paged_plan(
        q,
        k_cache,
        v_cache,
        page_table,
        cache_seqlens,
        cu_seqlens_q,
    )

    assert plan.mode == "extend"
    assert plan.kv_chunk_size == 32 * 64
    assert plan.split_kv is True


def test_paged_bf16_auto_chunk_heuristic_uses_two_pages_for_decode_2048() -> None:
    q, k_cache, v_cache, page_table, cache_seqlens, cu_seqlens_q = _make_inputs(
        q_seqlens=[1] * 8,
        cache_seqlens=[2048] * 8,
        kv_dtype=torch.bfloat16,
    )
    plan = create_paged_plan(
        q,
        k_cache,
        v_cache,
        page_table,
        cache_seqlens,
        cu_seqlens_q,
    )

    assert plan.mode == "decode"
    assert plan.kv_chunk_size == 2 * 64
    assert plan.split_kv is True


def test_paged_bf16_auto_chunk_heuristic_uses_six_pages_for_decode_8192() -> None:
    q, k_cache, v_cache, page_table, cache_seqlens, cu_seqlens_q = _make_inputs(
        q_seqlens=[1] * 8,
        cache_seqlens=[8192] * 8,
        kv_dtype=torch.bfloat16,
    )
    plan = create_paged_plan(
        q,
        k_cache,
        v_cache,
        page_table,
        cache_seqlens,
        cu_seqlens_q,
    )

    assert plan.mode == "decode"
    assert plan.kv_chunk_size == 6 * 64
    assert plan.split_kv is True


def test_paged_bf16_auto_chunk_heuristic_uses_forty_eight_pages_for_decode_32768() -> None:
    q, k_cache, v_cache, page_table, cache_seqlens, cu_seqlens_q = _make_inputs(
        q_seqlens=[1] * 8,
        cache_seqlens=[32768] * 8,
        kv_dtype=torch.bfloat16,
    )
    plan = create_paged_plan(
        q,
        k_cache,
        v_cache,
        page_table,
        cache_seqlens,
        cu_seqlens_q,
    )

    assert plan.mode == "decode"
    assert plan.kv_chunk_size == 48 * 64
    assert plan.split_kv is True


def test_paged_decode_fp8_chunk_policy_is_explicit_out_to_128k() -> None:
    _assert_chunk_table(
        q_seqlens=[1] * 8,
        kv_dtype=torch.float8_e4m3fn,
        expected_chunk_pages_by_cache_len={
            1: 2,
            2: 2,
            4: 2,
            8: 2,
            16: 2,
            32: 2,
            64: 2,
            128: 2,
            256: 2,
            512: 1,
            1024: 1,
            2048: 2,
            4096: 3,
            8192: 6,
            16384: 12,
            32768: 24,
            65536: 64,
            131072: 128,
        },
    )


def test_paged_decode_bf16_chunk_policy_is_explicit_out_to_128k() -> None:
    _assert_chunk_table(
        q_seqlens=[1] * 8,
        kv_dtype=torch.bfloat16,
        expected_chunk_pages_by_cache_len={
            1: 1,
            2: 1,
            4: 1,
            8: 1,
            16: 1,
            32: 1,
            64: 1,
            128: 2,
            256: 1,
            512: 1,
            1024: 1,
            2048: 2,
            4096: 3,
            8192: 6,
            16384: 12,
            32768: 48,
            65536: 128,
            131072: 128,
        },
    )


def test_paged_extend_fp8_chunk_policy_is_explicit_out_to_128k() -> None:
    _assert_chunk_table(
        q_seqlens=[6] * 8,
        kv_dtype=torch.float8_e4m3fn,
        expected_chunk_pages_by_cache_len={
            1: 1,
            2: 1,
            4: 1,
            8: 1,
            16: 1,
            32: 1,
            64: 1,
            128: 1,
            256: 1,
            512: 1,
            1024: 1,
            2048: 2,
            4096: 3,
            8192: 6,
            16384: 6,
            32768: 24,
            65536: 24,
            131072: 24,
        },
    )


def test_paged_extend_bf16_chunk_policy_is_explicit_out_to_128k() -> None:
    _assert_chunk_table(
        q_seqlens=[6] * 8,
        kv_dtype=torch.bfloat16,
        expected_chunk_pages_by_cache_len={
            1: 1,
            2: 1,
            4: 1,
            8: 1,
            16: 1,
            32: 1,
            64: 1,
            128: 1,
            256: 1,
            512: 1,
            1024: 1,
            2048: 2,
            4096: 3,
            8192: 6,
            16384: 6,
            32768: 32,
            65536: 32,
            131072: 32,
        },
    )
