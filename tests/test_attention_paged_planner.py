from __future__ import annotations

import pytest
import torch

from b12x.attention.paged.planner import (
    build_decode_chunk_pages_lut,
    create_paged_plan,
    decode_chunk_pages_for_graph,
    infer_paged_mode,
)
from b12x.attention.paged.tuning.registry import (
    DECODE_GRAPH_POLICY,
    register_decode_graph_policy,
)
from b12x.integration.attention import PagedAttentionWorkspace


@pytest.fixture(autouse=True)
def _isolate_decode_graph_policy_registry():
    snapshot = {key: value.copy() for key, value in DECODE_GRAPH_POLICY.items()}
    DECODE_GRAPH_POLICY.clear()
    try:
        yield
    finally:
        DECODE_GRAPH_POLICY.clear()
        DECODE_GRAPH_POLICY.update({key: value.copy() for key, value in snapshot.items()})


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
    assert plan.kv_chunk_size > 0
    assert plan.split_kv is True


def test_paged_plan_disables_split_kv_when_merge_backend_is_unsupported() -> None:
    q, k_cache, v_cache, page_table, cache_seqlens, cu_seqlens_q = _make_inputs(
        q_seqlens=[6] * 8,
        cache_seqlens=[8192] * 8,
        q_heads=48,
        kv_heads=8,
        head_dim_qk=128,
        head_dim_vo=128,
        kv_dtype=torch.bfloat16,
    )
    plan = create_paged_plan(
        q,
        k_cache,
        v_cache,
        page_table,
        cache_seqlens,
        cu_seqlens_q,
        enable_cuda_graph=True,
        graph_chunk_policy=True,
    )

    assert plan.split_kv is False


def test_paged_graph_budget_is_independent_of_cache_length() -> None:
    short_inputs = _make_inputs(
        q_seqlens=[1] * 8,
        cache_seqlens=[2048] * 8,
        kv_dtype=torch.bfloat16,
    )
    long_inputs = _make_inputs(
        q_seqlens=[1] * 8,
        cache_seqlens=[32768] * 8,
        kv_dtype=torch.bfloat16,
    )

    short_plan = create_paged_plan(
        *short_inputs,
        enable_cuda_graph=True,
        graph_chunk_policy=True,
        graph_ctas_per_sm=2,
    )
    long_plan = create_paged_plan(
        *long_inputs,
        enable_cuda_graph=True,
        graph_chunk_policy=True,
        graph_ctas_per_sm=2,
    )

    expected_budget = int(torch.cuda.get_device_properties("cuda").multi_processor_count) * 2
    assert short_plan.graph_ctas_per_sm == 2
    assert long_plan.graph_ctas_per_sm == 2
    assert short_plan.max_batch_size_if_split == expected_budget
    assert long_plan.max_batch_size_if_split == expected_budget
    assert short_plan.padded_batch_size == long_plan.padded_batch_size == expected_budget


def test_paged_graph_mode_falls_back_when_heuristic_overflows_budget() -> None:
    q, k_cache, v_cache, page_table, cache_seqlens, cu_seqlens_q = _make_inputs(
        q_seqlens=[6] * 8,
        cache_seqlens=[128000] * 8,
        kv_dtype=torch.bfloat16,
    )
    plan = create_paged_plan(
        q,
        k_cache,
        v_cache,
        page_table,
        cache_seqlens,
        cu_seqlens_q,
        enable_cuda_graph=True,
        graph_chunk_policy=True,
        graph_ctas_per_sm=2,
    )

    assert plan.new_batch_size <= plan.padded_batch_size
    assert plan.split_kv is True


def test_paged_graph_mode_uses_registered_decode_graph_policy() -> None:
    batch = 7
    register_decode_graph_policy(
        kv_dtype="bf16",
        regime="decode",
        batch=batch,
        graph_ctas_per_sm=6,
        capture_fixed_split_pages=4,
        capture_page_count=4096,
        page_size=64,
        chunk_ladder=((127, 1), (4096, 9)),
    )
    q, k_cache, v_cache, page_table, cache_seqlens, cu_seqlens_q = _make_inputs(
        q_seqlens=[1] * batch,
        cache_seqlens=[8192] * batch,
        kv_dtype=torch.bfloat16,
    )
    plan = create_paged_plan(
        q,
        k_cache,
        v_cache,
        page_table,
        cache_seqlens,
        cu_seqlens_q,
        enable_cuda_graph=True,
        graph_chunk_policy=True,
    )
    expected_budget = int(torch.cuda.get_device_properties("cuda").multi_processor_count) * 6
    assert plan.graph_ctas_per_sm == 6
    assert plan.max_batch_size_if_split == expected_budget
    assert plan.kv_chunk_size == 9 * 64


def test_decode_graph_chunk_pages_for_graph_uses_registered_policy() -> None:
    register_decode_graph_policy(
        kv_dtype="bf16",
        regime="decode",
        batch=4,
        graph_ctas_per_sm=6,
        capture_fixed_split_pages=4,
        capture_page_count=4096,
        page_size=64,
        chunk_ladder=((127, 1), (1024, 7), (4096, 9)),
    )

    assert (
        decode_chunk_pages_for_graph(
            q_dtype=torch.bfloat16,
            kv_dtype=torch.bfloat16,
            batch=4,
            page_size=64,
            head_dim_qk=256,
            head_dim_vo=256,
            gqa_group_size=8,
            max_effective_kv_pages=32,
        )
        == 1
    )
    assert (
        decode_chunk_pages_for_graph(
            q_dtype=torch.bfloat16,
            kv_dtype=torch.bfloat16,
            batch=4,
            page_size=64,
            head_dim_qk=256,
            head_dim_vo=256,
            gqa_group_size=8,
            max_effective_kv_pages=256,
        )
        == 7
    )


def test_build_decode_chunk_pages_lut_uses_registered_policy() -> None:
    register_decode_graph_policy(
        kv_dtype="bf16",
        regime="decode",
        batch=3,
        graph_ctas_per_sm=5,
        capture_fixed_split_pages=4,
        capture_page_count=4096,
        page_size=64,
        chunk_ladder=((4, 1), (8, 2), (16, 3)),
    )

    lut = build_decode_chunk_pages_lut(
        q_dtype=torch.bfloat16,
        kv_dtype=torch.bfloat16,
        batch=3,
        page_size=64,
        head_dim_qk=256,
        head_dim_vo=256,
        gqa_group_size=8,
        max_effective_kv_pages=16,
    )

    assert lut[:4] == (1, 1, 1, 1)
    assert lut[4:8] == (2, 2, 2, 2)
    assert lut[8:] == (3, 3, 3, 3, 3, 3, 3, 3)


@pytest.mark.parametrize(
    ("q_seqlens", "cache_seqlens", "kv_dtype"),
    [
        ([1] * 8, [8192] * 8, torch.float8_e4m3fn),
        ([1] * 8, [32768] * 8, torch.bfloat16),
        ([6] * 8, [8192] * 8, torch.float8_e4m3fn),
        ([6] * 8, [32768] * 8, torch.bfloat16),
    ],
)
def test_paged_non_policy_chunk_selection_still_produces_valid_split_kv_plans(
    q_seqlens: list[int],
    cache_seqlens: list[int],
    kv_dtype: torch.dtype,
) -> None:
    q, k_cache, v_cache, page_table, cache_seqlens_t, cu_seqlens_q = _make_inputs(
        q_seqlens=q_seqlens,
        cache_seqlens=cache_seqlens,
        kv_dtype=kv_dtype,
    )
    plan = create_paged_plan(
        q,
        k_cache,
        v_cache,
        page_table,
        cache_seqlens_t,
        cu_seqlens_q,
    )

    assert plan.kv_chunk_size > 0
    assert plan.total_num_partial_rows >= plan.total_q
    assert plan.new_batch_size >= page_table.shape[0]
    assert plan.split_kv is True
