from __future__ import annotations

import pytest
import torch

from b12x.attention.nsa_indexer.kernel import _split_index_k_cache_runtime_views
from b12x.integration.nsa_indexer import (
    NSAIndexerDecodeMetadata,
    NSAIndexerExtendMetadata,
    clear_nsa_indexer_caches,
    pack_nsa_index_k_cache_reference,
    sparse_nsa_index_decode_topk,
    sparse_nsa_index_extend_topk,
)
from b12x.attention.nsa_indexer.triton_topk import (
    run_sparse_nsa_dynamic_topk_kernel,
    run_sparse_nsa_topk_kernel,
    supports_sparse_nsa_dynamic_topk_kernel,
    supports_sparse_nsa_topk_kernel,
)
from b12x.attention.nsa_indexer.reference import sparse_nsa_index_reference


def _row_token_set(row: torch.Tensor) -> set[int]:
    return {int(token) for token in row.tolist() if int(token) >= 0}


def _assert_decode_matches_cuda_contract(
    *,
    actual: torch.Tensor,
    expected: torch.Tensor,
    page_table_1: torch.Tensor,
    query_row_to_batch: torch.Tensor,
    seqlens: torch.Tensor,
    topk: int,
) -> None:
    if actual.shape != expected.shape:
        raise AssertionError(f"shape mismatch: {tuple(actual.shape)} vs {tuple(expected.shape)}")
    for row_idx in range(actual.shape[0]):
        seq_len = int(seqlens[row_idx].item())
        batch_row = int(query_row_to_batch[row_idx].item())
        if seq_len <= topk:
            prefix = page_table_1[batch_row, :seq_len]
            assert torch.equal(actual[row_idx, :seq_len], prefix)
            assert torch.equal(
                actual[row_idx, seq_len:],
                torch.full((actual.shape[1] - seq_len,), -1, dtype=torch.int32, device=actual.device),
            )
            assert _row_token_set(actual[row_idx]) == _row_token_set(expected[row_idx])
        else:
            assert torch.equal(actual[row_idx], expected[row_idx])


def test_sparse_nsa_index_runtime_views_preserve_page_stride() -> None:
    device = torch.device("cpu")
    page_count = 3
    page_bytes = 64 * (128 + 4)
    index_k_cache = torch.arange(page_count * page_bytes, dtype=torch.uint8, device=device).view(
        page_count, page_bytes
    )

    quant, scales = _split_index_k_cache_runtime_views(index_k_cache)

    assert quant.shape == (page_count, 64, 128)
    assert quant.stride() == (page_bytes, 128, 1)
    assert quant.untyped_storage().data_ptr() == index_k_cache.untyped_storage().data_ptr()
    assert quant[1, 2, 127].item() == index_k_cache[1, 2 * 128 + 127].item()

    data_bytes = 64 * 128
    assert scales.shape == (page_count, 64)
    assert scales.stride() == (page_bytes // 4, 1)
    assert scales.untyped_storage().data_ptr() == index_k_cache.untyped_storage().data_ptr()
    scale_bytes = scales.view(torch.uint8).view(page_count, 64, 4)
    assert scale_bytes[1, 2, 0].item() == index_k_cache[1, data_bytes + 2 * 4].item()


def test_sparse_nsa_index_decode_topk_uses_decode_metadata() -> None:
    device = torch.device("cpu")
    gen = torch.Generator(device="cpu")
    gen.manual_seed(72_001)

    num_tokens = 80
    num_heads = 5
    q_rows = 3
    topk = 6

    index_k_cache = pack_nsa_index_k_cache_reference(
        torch.randn((num_tokens, 128), generator=gen, dtype=torch.float32, device=device) / 3
    )
    q_fp8 = (
        torch.randn((q_rows + 1, num_heads, 128), generator=gen, dtype=torch.float32, device=device) / 2
    ).to(torch.float8_e4m3fn)
    weights = torch.randn((q_rows + 1, num_heads), generator=gen, dtype=torch.float32, device=device)
    page_table_1 = torch.full((q_rows, 12), -1, dtype=torch.int32, device=device)
    page_table_1[0, :6] = torch.tensor([0, 1, 2, 3, 4, 5], dtype=torch.int32)
    page_table_1[1, :7] = torch.tensor([8, 9, 10, 11, 12, 13, 14], dtype=torch.int32)
    page_table_1[2, :8] = torch.tensor([16, 17, 18, 19, 20, 21, 22, 23], dtype=torch.int32)
    metadata = NSAIndexerDecodeMetadata(
        page_table_1=page_table_1,
        cache_seqlens_int32=torch.tensor([6, 7, 8], dtype=torch.int32, device=device),
    )

    output = sparse_nsa_index_decode_topk(
        q_fp8=q_fp8,
        weights=weights,
        index_k_cache=index_k_cache,
        metadata=metadata,
        topk=topk,
    )

    assert output.shape == (q_rows + 1, topk)
    assert torch.equal(output[-1], torch.full((topk,), -1, dtype=torch.int32))


def test_sparse_nsa_index_extend_topk_expands_batch_rows() -> None:
    device = torch.device("cpu")
    gen = torch.Generator(device="cpu")
    gen.manual_seed(72_002)

    num_tokens = 96
    num_heads = 2
    topk = 4

    index_k_cache = pack_nsa_index_k_cache_reference(
        torch.randn((num_tokens, 128), generator=gen, dtype=torch.float32, device=device) / 3
    )
    q_fp8 = (
        torch.randn((6, num_heads, 128), generator=gen, dtype=torch.float32, device=device) / 2
    ).to(torch.float8_e4m3fn)
    weights = torch.randn((6, num_heads), generator=gen, dtype=torch.float32, device=device)
    page_table_1 = torch.full((2, 12), -1, dtype=torch.int32, device=device)
    page_table_1[0, :7] = torch.tensor([3, 5, 7, 9, 11, 13, 15], dtype=torch.int32)
    page_table_1[1, :9] = torch.tensor([32, 34, 36, 38, 40, 42, 44, 46, 48], dtype=torch.int32)
    metadata = NSAIndexerExtendMetadata(
        page_table_1=page_table_1,
        nsa_seqlens_expanded=torch.tensor([6, 7, 7, 8, 9], dtype=torch.int32, device=device),
        nsa_extend_seq_lens_list=[2, 3],
    )

    output = sparse_nsa_index_extend_topk(
        q_fp8=q_fp8,
        weights=weights,
        index_k_cache=index_k_cache,
        metadata=metadata,
        topk=topk,
    )

    assert output.shape == (6, topk)
    assert torch.equal(output[5], torch.full((topk,), -1, dtype=torch.int32))


def test_sparse_nsa_index_extend_validates_expanded_lengths() -> None:
    device = torch.device("cpu")
    q_fp8 = torch.zeros((4, 1, 128), dtype=torch.float8_e4m3fn, device=device)
    weights = torch.zeros((4, 1), dtype=torch.float32, device=device)
    index_k_cache = torch.zeros((1, 64 * (128 + 4)), dtype=torch.uint8, device=device)
    metadata = NSAIndexerExtendMetadata(
        page_table_1=torch.zeros((2, 8), dtype=torch.int32, device=device),
        nsa_seqlens_expanded=torch.tensor([3, 4], dtype=torch.int32, device=device),
        nsa_extend_seq_lens_list=[2, 2],
    )

    try:
        sparse_nsa_index_extend_topk(
            q_fp8=q_fp8,
            weights=weights,
            index_k_cache=index_k_cache,
            metadata=metadata,
            topk=2,
        )
    except ValueError as exc:
        assert "fewer than the expanded query rows" in str(exc)
    else:
        raise AssertionError("expected expanded-length validation to fail")


def test_sparse_nsa_index_decode_topk_matches_reference_with_zero_valid_rows() -> None:
    device = torch.device("cpu")
    gen = torch.Generator(device="cpu")
    gen.manual_seed(72_003)

    num_tokens = 72
    q_rows = 3
    num_heads = 4
    topk = 6

    index_k_cache = pack_nsa_index_k_cache_reference(
        torch.randn((num_tokens, 128), generator=gen, dtype=torch.float32, device=device) / 3
    )
    q_fp8 = (
        torch.randn((q_rows, num_heads, 128), generator=gen, dtype=torch.float32, device=device) / 2
    ).to(torch.float8_e4m3fn)
    weights = torch.randn((q_rows, num_heads), generator=gen, dtype=torch.float32, device=device)
    page_table_1 = torch.full((q_rows, 8), -1, dtype=torch.int32, device=device)
    page_table_1[0, :4] = torch.tensor([4, 17, -1, 29], dtype=torch.int32)
    page_table_1[2, :5] = torch.tensor([33, -1, 47, 51, -1], dtype=torch.int32)
    seqlens = torch.tensor([4, 0, 5], dtype=torch.int32, device=device)

    actual = sparse_nsa_index_decode_topk(
        q_fp8=q_fp8,
        weights=weights,
        index_k_cache=index_k_cache,
        metadata=NSAIndexerDecodeMetadata(
            page_table_1=page_table_1,
            cache_seqlens_int32=seqlens,
        ),
        topk=topk,
    )
    expected = sparse_nsa_index_reference(
        q_fp8=q_fp8,
        weights=weights,
        index_k_cache=index_k_cache,
        page_table_1=page_table_1,
        query_row_to_batch=torch.arange(q_rows, dtype=torch.int32, device=device),
        seqlens_per_query=seqlens,
        topk=topk,
    )

    assert torch.equal(actual, expected)
    assert torch.equal(actual[1], torch.full((topk,), -1, dtype=torch.int32))


def test_sparse_nsa_index_extend_topk_matches_reference_on_repeated_dense_prefix_rows() -> None:
    device = torch.device("cpu")
    gen = torch.Generator(device="cpu")
    gen.manual_seed(72_004)

    num_tokens = 160
    num_heads = 3
    topk = 5

    index_k_cache = pack_nsa_index_k_cache_reference(
        torch.randn((num_tokens, 128), generator=gen, dtype=torch.float32, device=device) / 3
    )
    q_fp8 = (
        torch.randn((6, num_heads, 128), generator=gen, dtype=torch.float32, device=device) / 2
    ).to(torch.float8_e4m3fn)
    weights = torch.randn((6, num_heads, 1), generator=gen, dtype=torch.float32, device=device)
    page_table_1 = torch.stack(
        [
            torch.tensor([0, 1, 2, 3, 4, 5, 6, 7], dtype=torch.int32),
            torch.tensor([64, 66, 68, 70, 72, 74, 76, 78], dtype=torch.int32),
        ],
        dim=0,
    ).to(device=device)
    metadata = NSAIndexerExtendMetadata(
        page_table_1=page_table_1,
        nsa_seqlens_expanded=torch.tensor([4, 5, 6, 6, 7], dtype=torch.int32, device=device),
        nsa_extend_seq_lens_list=[2, 3],
    )

    actual = sparse_nsa_index_extend_topk(
        q_fp8=q_fp8,
        weights=weights,
        index_k_cache=index_k_cache,
        metadata=metadata,
        topk=topk,
    )
    query_row_to_batch = torch.tensor([0, 0, 1, 1, 1], dtype=torch.int32, device=device)
    expected = sparse_nsa_index_reference(
        q_fp8=q_fp8,
        weights=weights,
        index_k_cache=index_k_cache,
        page_table_1=page_table_1,
        query_row_to_batch=query_row_to_batch,
        seqlens_per_query=metadata.nsa_seqlens_expanded,
        topk=topk,
    )

    assert torch.equal(actual[:5], expected[:5])
    assert torch.equal(actual[5], torch.full((topk,), -1, dtype=torch.int32))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for kernel coverage")
def test_sparse_nsa_index_decode_topk_cuda_kernel_matches_reference() -> None:
    device = torch.device("cuda")
    gen = torch.Generator(device="cpu")
    gen.manual_seed(72_005)

    num_tokens = 192
    q_rows = 4
    num_heads = 8
    topk = 6

    index_k_cache = pack_nsa_index_k_cache_reference(
        torch.randn((num_tokens, 128), generator=gen, dtype=torch.float32).to(device=device) / 3
    )
    q_fp8 = (
        torch.randn((q_rows, num_heads, 128), generator=gen, dtype=torch.float32).to(device=device) / 2
    ).to(torch.float8_e4m3fn)
    weights = torch.randn((q_rows, num_heads, 1), generator=gen, dtype=torch.float32).to(device=device)
    page_table_1 = torch.full((q_rows, 10), -1, dtype=torch.int32, device=device)
    page_table_1[0, :6] = torch.tensor([4, 9, 11, 18, 33, 40], dtype=torch.int32, device=device)
    page_table_1[1, :7] = torch.tensor([2, 8, 15, 21, 45, 64, 65], dtype=torch.int32, device=device)
    page_table_1[2, :8] = torch.tensor([79, 81, 96, 97, 111, 127, 128, 129], dtype=torch.int32, device=device)
    page_table_1[3, :5] = torch.tensor([140, 141, 143, 151, 159], dtype=torch.int32, device=device)
    seqlens = torch.tensor([6, 7, 8, 5], dtype=torch.int32, device=device)

    actual = sparse_nsa_index_decode_topk(
        q_fp8=q_fp8,
        weights=weights,
        index_k_cache=index_k_cache,
        metadata=NSAIndexerDecodeMetadata(
            page_table_1=page_table_1,
            cache_seqlens_int32=seqlens,
        ),
        topk=topk,
    )
    expected = sparse_nsa_index_reference(
        q_fp8=q_fp8,
        weights=weights,
        index_k_cache=index_k_cache,
        page_table_1=page_table_1,
        query_row_to_batch=torch.arange(q_rows, dtype=torch.int32, device=device),
        seqlens_per_query=seqlens,
        topk=topk,
    )

    _assert_decode_matches_cuda_contract(
        actual=actual,
        expected=expected,
        page_table_1=page_table_1,
        query_row_to_batch=torch.arange(q_rows, dtype=torch.int32, device=device),
        seqlens=seqlens,
        topk=topk,
    )



@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for kernel coverage")
def test_sparse_nsa_index_decode_topk_cuda_kernel_returns_live_prefix_for_trivial_rows() -> None:
    device = torch.device("cuda")
    num_tokens = 160
    q_rows = 2
    num_heads = 6
    topk = 8

    index_k_cache = pack_nsa_index_k_cache_reference(
        torch.randn((num_tokens, 128), dtype=torch.float32, device=device) / 3
    )
    q_fp8 = (
        torch.randn((q_rows, num_heads, 128), dtype=torch.float32, device=device) / 2
    ).to(torch.float8_e4m3fn)
    weights = torch.randn((q_rows, num_heads, 1), dtype=torch.float32, device=device)
    page_table_1 = torch.full((q_rows, 10), -1, dtype=torch.int32, device=device)
    page_table_1[0, :6] = torch.tensor([11, 7, 29, 5, 43, 3], dtype=torch.int32, device=device)
    page_table_1[1, :8] = torch.tensor([80, 78, 76, 74, 72, 70, 68, 66], dtype=torch.int32, device=device)
    seqlens = torch.tensor([6, 8], dtype=torch.int32, device=device)

    actual = sparse_nsa_index_decode_topk(
        q_fp8=q_fp8,
        weights=weights,
        index_k_cache=index_k_cache,
        metadata=NSAIndexerDecodeMetadata(
            page_table_1=page_table_1,
            cache_seqlens_int32=seqlens,
        ),
        topk=topk,
    )
    expected = sparse_nsa_index_reference(
        q_fp8=q_fp8,
        weights=weights,
        index_k_cache=index_k_cache,
        page_table_1=page_table_1,
        query_row_to_batch=torch.arange(q_rows, dtype=torch.int32, device=device),
        seqlens_per_query=seqlens,
        topk=topk,
    )

    _assert_decode_matches_cuda_contract(
        actual=actual,
        expected=expected,
        page_table_1=page_table_1,
        query_row_to_batch=torch.arange(q_rows, dtype=torch.int32, device=device),
        seqlens=seqlens,
        topk=topk,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for kernel coverage")
def test_sparse_nsa_index_decode_topk_cuda_kernel_prefers_earlier_positions_on_ties() -> None:
    device = torch.device("cuda")
    num_tokens = 128
    q_rows = 2
    num_heads = 6
    topk = 5

    index_k_cache = pack_nsa_index_k_cache_reference(
        torch.randn((num_tokens, 128), dtype=torch.float32, device=device) / 3
    )
    q_fp8 = torch.zeros((q_rows, num_heads, 128), dtype=torch.float32, device=device).to(torch.float8_e4m3fn)
    weights = torch.randn((q_rows, num_heads), dtype=torch.float32, device=device)
    page_table_1 = torch.full((q_rows, 8), -1, dtype=torch.int32, device=device)
    page_table_1[0, :7] = torch.tensor([20, 18, 16, 14, 12, 10, 8], dtype=torch.int32, device=device)
    page_table_1[1, :6] = torch.tensor([41, 39, 37, 35, 33, 31], dtype=torch.int32, device=device)
    seqlens = torch.tensor([7, 6], dtype=torch.int32, device=device)

    actual = sparse_nsa_index_decode_topk(
        q_fp8=q_fp8,
        weights=weights,
        index_k_cache=index_k_cache,
        metadata=NSAIndexerDecodeMetadata(
            page_table_1=page_table_1,
            cache_seqlens_int32=seqlens,
        ),
        topk=topk,
    )

    assert torch.equal(actual[0], torch.tensor([20, 18, 16, 14, 12], dtype=torch.int32, device=device))
    assert torch.equal(actual[1], torch.tensor([41, 39, 37, 35, 33], dtype=torch.int32, device=device))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for graph capture coverage")
def test_sparse_nsa_index_decode_topk_cuda_graph_replay_tracks_live_width_without_stale_output() -> None:
    device = torch.device("cuda")
    num_tokens = 4096
    q_rows = 2
    num_heads = 8
    topk = 1024
    width_capacity = 2048

    index_k_cache = pack_nsa_index_k_cache_reference(
        torch.randn((num_tokens, 128), dtype=torch.float32, device=device) / 3
    )
    q_fp8 = (
        torch.randn((q_rows, num_heads, 128), dtype=torch.float32, device=device) / 2
    ).to(torch.float8_e4m3fn)
    weights = torch.randn((q_rows, num_heads, 1), dtype=torch.float32, device=device)
    live_page_table_1 = torch.full((q_rows, width_capacity), -1, dtype=torch.int32, device=device)
    live_seqlens = torch.empty((q_rows,), dtype=torch.int32, device=device)
    graph_page_table_1 = torch.full((q_rows, width_capacity), -1, dtype=torch.int32, device=device)
    graph_seqlens = torch.empty_like(live_seqlens)

    def prepare(case: int) -> None:
        if case == 0:
            live_page_table_1[0, :1536] = torch.arange(2000, 3536, dtype=torch.int32, device=device)
            live_page_table_1[1, :768] = torch.arange(900, 1668, dtype=torch.int32, device=device)
            live_page_table_1[0, 1536:] = -1
            live_page_table_1[1, 768:] = -1
            live_seqlens.copy_(torch.tensor([1536, 768], dtype=torch.int32, device=device))
        else:
            live_page_table_1[0, :1280] = torch.arange(2600, 3880, dtype=torch.int32, device=device)
            live_page_table_1[1, :512] = torch.arange(1200, 1712, dtype=torch.int32, device=device)
            live_page_table_1[0, 1280:] = -1
            live_page_table_1[1, 512:] = -1
            live_seqlens.copy_(torch.tensor([1280, 512], dtype=torch.int32, device=device))
        graph_page_table_1.copy_(live_page_table_1)
        graph_seqlens.copy_(live_seqlens)

    captured_out = None

    def run() -> None:
        nonlocal captured_out
        captured_out = sparse_nsa_index_decode_topk(
            q_fp8=q_fp8,
            weights=weights,
            index_k_cache=index_k_cache,
            metadata=NSAIndexerDecodeMetadata(
                page_table_1=graph_page_table_1,
                cache_seqlens_int32=graph_seqlens,
            ),
            topk=topk,
        )

    clear_nsa_indexer_caches()
    prepare(0)
    run()
    torch.cuda.synchronize(device)
    run()
    torch.cuda.synchronize(device)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()

    graph.replay()
    torch.cuda.synchronize(device)
    assert captured_out is not None
    expected0 = sparse_nsa_index_reference(
        q_fp8=q_fp8,
        weights=weights,
        index_k_cache=index_k_cache,
        page_table_1=graph_page_table_1,
        query_row_to_batch=torch.arange(q_rows, dtype=torch.int32, device=device),
        seqlens_per_query=graph_seqlens,
        topk=topk,
    )
    _assert_decode_matches_cuda_contract(
        actual=captured_out,
        expected=expected0,
        page_table_1=graph_page_table_1,
        query_row_to_batch=torch.arange(q_rows, dtype=torch.int32, device=device),
        seqlens=graph_seqlens,
        topk=topk,
    )

    prepare(1)
    graph.replay()
    torch.cuda.synchronize(device)
    expected1 = sparse_nsa_index_reference(
        q_fp8=q_fp8,
        weights=weights,
        index_k_cache=index_k_cache,
        page_table_1=graph_page_table_1,
        query_row_to_batch=torch.arange(q_rows, dtype=torch.int32, device=device),
        seqlens_per_query=graph_seqlens,
        topk=topk,
    )
    _assert_decode_matches_cuda_contract(
        actual=captured_out,
        expected=expected1,
        page_table_1=graph_page_table_1,
        query_row_to_batch=torch.arange(q_rows, dtype=torch.int32, device=device),
        seqlens=graph_seqlens,
        topk=topk,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for kernel coverage")
def test_sparse_nsa_index_extend_topk_cuda_kernel_matches_reference() -> None:
    device = torch.device("cuda")
    gen = torch.Generator(device="cpu")
    gen.manual_seed(72_006)

    num_tokens = 224
    num_heads = 6
    topk = 4

    index_k_cache = pack_nsa_index_k_cache_reference(
        torch.randn((num_tokens, 128), generator=gen, dtype=torch.float32).to(device=device) / 3
    )
    q_fp8 = (
        torch.randn((7, num_heads, 128), generator=gen, dtype=torch.float32).to(device=device) / 2
    ).to(torch.float8_e4m3fn)
    weights = torch.randn((7, num_heads), generator=gen, dtype=torch.float32).to(device=device)
    page_table_1 = torch.full((3, 12), -1, dtype=torch.int32, device=device)
    page_table_1[0, :7] = torch.tensor([1, 3, 5, 7, 9, 11, 13], dtype=torch.int32, device=device)
    page_table_1[1, :9] = torch.tensor([64, 66, 68, 70, 72, 74, 76, 78, 80], dtype=torch.int32, device=device)
    page_table_1[2, :10] = torch.tensor([128, 131, 135, 139, 143, 147, 151, 155, 159, 163], dtype=torch.int32, device=device)
    metadata = NSAIndexerExtendMetadata(
        page_table_1=page_table_1,
        nsa_seqlens_expanded=torch.tensor([6, 7, 8, 9, 9, 10], dtype=torch.int32, device=device),
        nsa_extend_seq_lens_list=[2, 1, 3],
    )

    actual = sparse_nsa_index_extend_topk(
        q_fp8=q_fp8,
        weights=weights,
        index_k_cache=index_k_cache,
        metadata=metadata,
        topk=topk,
    )
    query_row_to_batch = torch.tensor([0, 0, 1, 2, 2, 2], dtype=torch.int32, device=device)
    expected = sparse_nsa_index_reference(
        q_fp8=q_fp8,
        weights=weights,
        index_k_cache=index_k_cache,
        page_table_1=page_table_1,
        query_row_to_batch=query_row_to_batch,
        seqlens_per_query=metadata.nsa_seqlens_expanded,
        topk=topk,
    )

    assert torch.equal(actual[:6], expected[:6])
    assert torch.equal(actual[6], torch.full((topk,), -1, dtype=torch.int32, device=device))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for kernel coverage")
def test_sparse_nsa_topk_kernel_matches_stable_sort_at_full_width() -> None:
    device = torch.device("cuda")
    rows = 4
    width = 2048
    topk = 2048

    logits = torch.randn((rows, width), dtype=torch.float32, device=device)
    page_table_1 = torch.randint(0, 8192, (2, width), dtype=torch.int32, device=device)
    query_row_to_batch = torch.tensor([0, 0, 1, 1], dtype=torch.int32, device=device)
    seqlens = torch.full((rows,), width, dtype=torch.int32, device=device)
    output = torch.full((rows, topk), -1, dtype=torch.int32, device=device)

    assert supports_sparse_nsa_topk_kernel(
        logits=logits,
        page_table_1=page_table_1,
        query_row_to_batch=query_row_to_batch,
        seqlens_per_query=seqlens,
        gather_k=topk,
    )
    run_sparse_nsa_topk_kernel(
        logits=logits,
        page_table_1=page_table_1,
        query_row_to_batch=query_row_to_batch,
        seqlens_per_query=seqlens,
        output=output,
        gather_k=topk,
    )

    order = torch.argsort(logits, dim=1, descending=True, stable=True)
    expected = page_table_1[
        query_row_to_batch.to(torch.long).unsqueeze(1).expand(-1, topk),
        order.to(torch.long),
    ]
    assert torch.equal(output, expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for kernel coverage")
def test_sparse_nsa_dynamic_topk_kernel_uses_runtime_active_width() -> None:
    device = torch.device("cuda")
    rows = 3
    width = 8192
    active_width = 1024
    topk = 2048

    logits = torch.randn((rows, width), dtype=torch.float32, device=device)
    page_table_1 = torch.randint(0, 65536, (2, width), dtype=torch.int32, device=device)
    query_row_to_batch = torch.tensor([0, 0, 1], dtype=torch.int32, device=device)
    seqlens = torch.full((rows,), active_width, dtype=torch.int32, device=device)
    active_width_tensor = torch.tensor([active_width], dtype=torch.int32, device=device)
    output = torch.full((rows, topk), -1, dtype=torch.int32, device=device)

    assert supports_sparse_nsa_dynamic_topk_kernel(
        logits=logits,
        page_table_1=page_table_1,
        query_row_to_batch=query_row_to_batch,
        seqlens_per_query=seqlens,
        active_width=active_width_tensor,
        gather_k=topk,
    )
    run_sparse_nsa_dynamic_topk_kernel(
        logits=logits,
        page_table_1=page_table_1,
        query_row_to_batch=query_row_to_batch,
        seqlens_per_query=seqlens,
        active_width=active_width_tensor,
        output=output,
        gather_k=topk,
    )

    expected = torch.full((rows, topk), -1, dtype=torch.int32, device=device)
    expected[:, :active_width] = page_table_1[
        query_row_to_batch.to(torch.long).unsqueeze(1).expand(-1, active_width),
        torch.arange(active_width, device=device, dtype=torch.long).unsqueeze(0).expand(rows, -1),
    ]
    assert torch.equal(output, expected)
