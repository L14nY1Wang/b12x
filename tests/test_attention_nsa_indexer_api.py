from __future__ import annotations

import pytest
import torch

from b12x.attention.nsa_indexer.kernel import _split_index_k_cache_runtime_views
from b12x.attention.nsa_indexer.reference import (
    sparse_nsa_extend_logits_reference,
    sparse_nsa_paged_logits_reference,
)
from b12x.integration.nsa_indexer import (
    NSAIndexerExtendLogitsMetadata,
    NSAIndexerPagedDecodeMetadata,
    clear_nsa_indexer_caches,
    get_paged_mqa_logits_metadata,
    pack_nsa_index_k_cache_reference,
    sparse_nsa_index_decode_logits_paged,
    sparse_nsa_index_extend_logits,
)


_FP8_E4M3_MAX = float(torch.finfo(torch.float8_e4m3fn).max)


def _make_real_page_table(
    *,
    page_starts: list[int],
    seqlens: list[int],
    width_blocks: int,
    device: torch.device,
) -> torch.Tensor:
    real_page_table = torch.full(
        (len(seqlens), width_blocks),
        -1,
        dtype=torch.int32,
        device=device,
    )
    for row_idx, (page_start, seq_len) in enumerate(zip(page_starts, seqlens, strict=True)):
        block_count = (int(seq_len) + 63) // 64
        if block_count:
            real_page_table[row_idx, :block_count] = torch.arange(
                page_start,
                page_start + block_count,
                dtype=torch.int32,
                device=device,
            )
    return real_page_table


def _quantize_rows_to_kv_fp8(k: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    scale = k.abs().amax(dim=1) / _FP8_E4M3_MAX
    scale = torch.where(scale > 0, scale, torch.ones_like(scale))
    quant = (k / scale.unsqueeze(1)).clamp(-_FP8_E4M3_MAX, _FP8_E4M3_MAX).to(torch.float8_e4m3fn)
    return quant, scale.to(torch.float32)


def _assert_logits_close(actual: torch.Tensor, expected: torch.Tensor) -> None:
    torch.testing.assert_close(actual, expected, atol=1e-4, rtol=1e-4)


def test_sparse_nsa_index_runtime_views_preserve_page_stride() -> None:
    device = torch.device("cpu")
    page_count = 3
    page_bytes = 64 * (128 + 4)
    index_k_cache = torch.arange(page_count * page_bytes, dtype=torch.uint8, device=device).view(
        page_count,
        page_bytes,
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


def test_get_paged_mqa_logits_metadata_matches_deepgemm_shape() -> None:
    context_lens_1d = torch.tensor([64, 128, 192], dtype=torch.int32)
    schedule_1d = get_paged_mqa_logits_metadata(context_lens_1d, 64, 5)
    assert schedule_1d.shape == (6, 2)
    assert schedule_1d.dtype == torch.int32
    assert schedule_1d.is_contiguous()
    assert torch.count_nonzero(schedule_1d).item() == 0

    context_lens_2d = torch.tensor([[64, 65], [128, 129]], dtype=torch.int32)
    schedule_2d = get_paged_mqa_logits_metadata(context_lens_2d, 64, 7)
    assert schedule_2d.shape == (8, 2)
    assert schedule_2d.dtype == torch.int32
    assert schedule_2d.is_contiguous()
    assert torch.count_nonzero(schedule_2d).item() == 0


def test_sparse_nsa_index_decode_logits_paged_matches_reference_cpu() -> None:
    device = torch.device("cpu")
    gen = torch.Generator(device="cpu")
    gen.manual_seed(72_100)

    q_rows = 3
    num_heads = 4
    page_starts = [1, 3, 5]
    width_blocks = 3
    num_tokens = (max(page_starts) + width_blocks) * 64
    seqlens = torch.tensor([65, 128, 150], dtype=torch.int32, device=device)
    real_page_table = _make_real_page_table(
        page_starts=page_starts,
        seqlens=seqlens.tolist(),
        width_blocks=width_blocks,
        device=device,
    )
    q_fp8 = (
        torch.randn((q_rows + 1, num_heads, 128), generator=gen, dtype=torch.float32, device=device) / 2
    ).to(torch.float8_e4m3fn)
    weights = torch.randn((q_rows + 1, num_heads), generator=gen, dtype=torch.float32, device=device)
    index_k_cache = pack_nsa_index_k_cache_reference(
        torch.randn((num_tokens, 128), generator=gen, dtype=torch.float32, device=device) / 3
    )

    actual = sparse_nsa_index_decode_logits_paged(
        q_fp8=q_fp8,
        weights=weights,
        index_k_cache=index_k_cache,
        metadata=NSAIndexerPagedDecodeMetadata(
            real_page_table=real_page_table,
            cache_seqlens_int32=seqlens,
            paged_mqa_schedule_metadata=get_paged_mqa_logits_metadata(seqlens, 64, 8),
        ),
    )
    expected = sparse_nsa_paged_logits_reference(
        q_fp8=q_fp8,
        weights=weights,
        index_k_cache=index_k_cache,
        real_page_table=real_page_table,
        query_row_to_batch=torch.arange(q_rows, dtype=torch.int32, device=device),
        seqlens_per_query=seqlens,
    )

    _assert_logits_close(actual, expected)
    assert torch.isneginf(actual[-1]).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for paged kernel coverage")
def test_sparse_nsa_index_decode_logits_paged_cuda_kernel_matches_reference() -> None:
    device = torch.device("cuda")
    gen = torch.Generator(device="cpu")
    gen.manual_seed(72_101)

    q_rows = 4
    num_heads = 8
    page_starts = [2, 8, 12, 16]
    num_tokens = (max(page_starts) + 3) * 64
    seqlens = torch.tensor([65, 96, 128, 191], dtype=torch.int32, device=device)
    real_page_table = _make_real_page_table(
        page_starts=page_starts,
        seqlens=seqlens.tolist(),
        width_blocks=3,
        device=device,
    )
    q_fp8 = (
        torch.randn((q_rows, num_heads, 128), generator=gen, dtype=torch.float32).to(device=device) / 2
    ).to(torch.float8_e4m3fn)
    weights = torch.randn((q_rows, num_heads), generator=gen, dtype=torch.float32).to(device=device)
    index_k_cache = pack_nsa_index_k_cache_reference(
        torch.randn((num_tokens, 128), generator=gen, dtype=torch.float32).to(device=device) / 3
    )

    actual = sparse_nsa_index_decode_logits_paged(
        q_fp8=q_fp8,
        weights=weights,
        index_k_cache=index_k_cache,
        metadata=NSAIndexerPagedDecodeMetadata(
            real_page_table=real_page_table,
            cache_seqlens_int32=seqlens,
            paged_mqa_schedule_metadata=get_paged_mqa_logits_metadata(seqlens, 64, 8),
        ),
    )
    expected = sparse_nsa_paged_logits_reference(
        q_fp8=q_fp8,
        weights=weights,
        index_k_cache=index_k_cache,
        real_page_table=real_page_table,
        query_row_to_batch=torch.arange(q_rows, dtype=torch.int32, device=device),
        seqlens_per_query=seqlens,
    )

    torch.cuda.synchronize(device)
    _assert_logits_close(actual, expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for graph capture coverage")
def test_sparse_nsa_index_decode_logits_paged_cuda_graph_replay_tracks_live_width_without_stale_output() -> None:
    device = torch.device("cuda")
    gen = torch.Generator(device="cpu")
    gen.manual_seed(72_102)

    rows = 2
    num_heads = 8
    num_tokens = 1024
    graph_width_blocks = 4
    live_width_blocks = 3

    q_fp8 = (
        torch.randn((rows, num_heads, 128), generator=gen, dtype=torch.float32).to(device=device) / 2
    ).to(torch.float8_e4m3fn)
    weights = torch.randn((rows, num_heads), generator=gen, dtype=torch.float32).to(device=device)
    index_k_cache = pack_nsa_index_k_cache_reference(
        torch.randn((num_tokens, 128), generator=gen, dtype=torch.float32).to(device=device) / 3
    )
    live_real_page_table0 = _make_real_page_table(
        page_starts=[2, 8],
        seqlens=[150, 129],
        width_blocks=live_width_blocks,
        device=device,
    )
    live_real_page_table1 = _make_real_page_table(
        page_starts=[4, 9],
        seqlens=[65, 64],
        width_blocks=live_width_blocks,
        device=device,
    )
    graph_real_page_table = torch.full(
        (rows, graph_width_blocks),
        -1,
        dtype=torch.int32,
        device=device,
    )
    graph_seqlens = torch.empty((rows,), dtype=torch.int32, device=device)

    def prepare(page_table: torch.Tensor, seqlens: torch.Tensor) -> None:
        graph_real_page_table[:, :live_width_blocks].copy_(page_table)
        graph_seqlens.copy_(seqlens)

    metadata = NSAIndexerPagedDecodeMetadata(
        real_page_table=graph_real_page_table,
        cache_seqlens_int32=graph_seqlens,
        paged_mqa_schedule_metadata=get_paged_mqa_logits_metadata(graph_seqlens, 64, 8),
    )

    clear_nsa_indexer_caches()
    prepare(live_real_page_table0, torch.tensor([150, 129], dtype=torch.int32, device=device))
    sparse_nsa_index_decode_logits_paged(
        q_fp8=q_fp8,
        weights=weights,
        index_k_cache=index_k_cache,
        metadata=metadata,
    )
    torch.cuda.synchronize(device)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured_out = sparse_nsa_index_decode_logits_paged(
            q_fp8=q_fp8,
            weights=weights,
            index_k_cache=index_k_cache,
            metadata=metadata,
        )
    graph.replay()
    torch.cuda.synchronize(device)
    actual0 = captured_out.clone()
    expected0 = sparse_nsa_paged_logits_reference(
        q_fp8=q_fp8,
        weights=weights,
        index_k_cache=index_k_cache,
        real_page_table=graph_real_page_table,
        query_row_to_batch=torch.arange(rows, dtype=torch.int32, device=device),
        seqlens_per_query=graph_seqlens,
    )
    _assert_logits_close(actual0, expected0)

    prepare(live_real_page_table1, torch.tensor([65, 64], dtype=torch.int32, device=device))
    graph.replay()
    torch.cuda.synchronize(device)
    actual1 = captured_out.clone()
    expected1 = sparse_nsa_paged_logits_reference(
        q_fp8=q_fp8,
        weights=weights,
        index_k_cache=index_k_cache,
        real_page_table=graph_real_page_table,
        query_row_to_batch=torch.arange(rows, dtype=torch.int32, device=device),
        seqlens_per_query=graph_seqlens,
    )
    _assert_logits_close(actual1, expected1)
    assert torch.isneginf(actual1[:, 65:]).all()


@pytest.mark.parametrize(
    "device",
    [torch.device("cpu")] + ([torch.device("cuda")] if torch.cuda.is_available() else []),
)
def test_sparse_nsa_index_extend_logits_matches_reference(device: torch.device) -> None:
    gen = torch.Generator(device="cpu")
    gen.manual_seed(72_103)

    q_rows = 5
    num_heads = 3
    k_rows = 64
    q_fp8 = (
        torch.randn((q_rows + 1, num_heads, 128), generator=gen, dtype=torch.float32).to(device=device)
        / 2
    ).to(torch.float8_e4m3fn)
    weights = torch.randn((q_rows + 1, num_heads), generator=gen, dtype=torch.float32).to(device=device)
    k = torch.randn((k_rows, 128), generator=gen, dtype=torch.float32).to(device=device) / 3
    kv_fp8 = _quantize_rows_to_kv_fp8(k)
    k_start = torch.tensor([0, 5, 12, 12, 40], dtype=torch.int32, device=device)
    k_end = torch.tensor([8, 16, 20, 12, 55], dtype=torch.int32, device=device)

    actual = sparse_nsa_index_extend_logits(
        q_fp8=q_fp8,
        weights=weights,
        kv_fp8=kv_fp8,
        metadata=NSAIndexerExtendLogitsMetadata(
            k_start=k_start,
            k_end=k_end,
        ),
    )
    expected = sparse_nsa_extend_logits_reference(
        q_fp8=q_fp8,
        weights=weights,
        kv_fp8=kv_fp8,
        k_start=k_start,
        k_end=k_end,
    )

    _assert_logits_close(actual, expected)
    assert torch.isneginf(actual[-1]).all()


@pytest.mark.parametrize(
    "device",
    [torch.device("cpu")] + ([torch.device("cuda")] if torch.cuda.is_available() else []),
)
def test_sparse_nsa_index_extend_logits_matches_reference_for_sparse_tile_ranges(
    device: torch.device,
) -> None:
    gen = torch.Generator(device="cpu")
    gen.manual_seed(72_104)

    q_rows = 40
    num_heads = 4
    k_rows = 130
    q_fp8 = (
        torch.randn((q_rows, num_heads, 128), generator=gen, dtype=torch.float32).to(device=device) / 2
    ).to(torch.float8_e4m3fn)
    weights = torch.randn((q_rows, num_heads), generator=gen, dtype=torch.float32).to(device=device)
    k = torch.randn((k_rows, 128), generator=gen, dtype=torch.float32).to(device=device) / 3
    kv_fp8 = _quantize_rows_to_kv_fp8(k)
    k_start = torch.tensor(([0] * 32) + ([128] * 8), dtype=torch.int32, device=device)
    k_end = torch.tensor(([32] * 32) + ([130] * 8), dtype=torch.int32, device=device)

    actual = sparse_nsa_index_extend_logits(
        q_fp8=q_fp8,
        weights=weights,
        kv_fp8=kv_fp8,
        metadata=NSAIndexerExtendLogitsMetadata(
            k_start=k_start,
            k_end=k_end,
        ),
    )
    expected = sparse_nsa_extend_logits_reference(
        q_fp8=q_fp8,
        weights=weights,
        kv_fp8=kv_fp8,
        k_start=k_start,
        k_end=k_end,
    )

    _assert_logits_close(actual, expected)
    assert torch.isneginf(actual[:32, 32:]).all()
    assert torch.isneginf(actual[32:, :128]).all()
