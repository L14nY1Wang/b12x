"""NSA indexer API oriented around the current sglang boundary."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from functools import lru_cache

import torch

from .fused_decode import (
    clear_sparse_nsa_fused_decode_kernel_cache,
    run_sparse_nsa_fused_decode_kernel,
    supports_sparse_nsa_fused_decode_kernel,
)
from .kernel import (
    clear_sparse_nsa_indexer_kernel_cache,
    run_sparse_nsa_index_logits_kernel,
    supports_sparse_nsa_indexer_kernel,
)
from .reference import sparse_nsa_index_reference
from .triton_topk import (
    clear_sparse_nsa_topk_kernel_cache,
    run_sparse_nsa_dynamic_topk_kernel,
    run_sparse_nsa_topk_kernel,
    supports_sparse_nsa_dynamic_topk_kernel,
    supports_sparse_nsa_topk_kernel,
)


_INDEX_HEAD_DIM = 128


def _is_cuda_graph_capture_active(device: torch.device) -> bool:
    return device.type == "cuda" and torch.cuda.is_current_stream_capturing()


@dataclass(frozen=True)
class NSAIndexerDecodeMetadata:
    page_table_1: torch.Tensor
    cache_seqlens_int32: torch.Tensor


@dataclass(frozen=True)
class NSAIndexerExtendMetadata:
    page_table_1: torch.Tensor
    nsa_seqlens_expanded: torch.Tensor
    nsa_extend_seq_lens_list: Sequence[int]


def clear_nsa_indexer_caches() -> None:
    """Clear any cached NSA indexer runtime state."""
    clear_sparse_nsa_fused_decode_kernel_cache()
    clear_sparse_nsa_indexer_kernel_cache()
    clear_sparse_nsa_topk_kernel_cache()
    _cached_extend_lengths_tensor.cache_clear()
    _cached_query_row_to_batch.cache_clear()
    _cached_width_cap_tensor.cache_clear()


def _normalize_weights(weights: torch.Tensor, *, q_rows: int, num_heads: int) -> torch.Tensor:
    if weights.ndim == 3:
        if weights.shape[2] != 1:
            raise ValueError(
                f"weights rank-3 input must have trailing dimension 1, got {tuple(weights.shape)}"
            )
        weights = weights.squeeze(2)
    if weights.ndim != 2:
        raise ValueError(f"weights must be rank-2 or rank-3, got {tuple(weights.shape)}")
    if weights.shape != (q_rows, num_heads):
        raise ValueError(f"weights shape must be {(q_rows, num_heads)}, got {tuple(weights.shape)}")
    return weights.to(torch.float32)


@lru_cache(maxsize=64)
def _cached_extend_lengths_tensor(
    lengths: tuple[int, ...],
    device_type: str,
    device_index: int | None,
) -> torch.Tensor:
    return torch.tensor(lengths, dtype=torch.int32, device=torch.device(device_type, device_index))


@lru_cache(maxsize=64)
def _cached_query_row_to_batch(
    lengths: tuple[int, ...],
    device_type: str,
    device_index: int | None,
) -> torch.Tensor:
    rows = []
    for batch_row, length in enumerate(lengths):
        rows.extend([batch_row] * int(length))
    return torch.tensor(rows, dtype=torch.int32, device=torch.device(device_type, device_index))


@lru_cache(maxsize=64)
def _cached_width_cap_tensor(
    width: int,
    device_type: str,
    device_index: int | None,
) -> torch.Tensor:
    return torch.tensor([width], dtype=torch.int32, device=torch.device(device_type, device_index))


def _validate_sparse_index_inputs(
    *,
    q_fp8: torch.Tensor,
    weights: torch.Tensor,
    page_table_1: torch.Tensor,
    query_row_to_batch: torch.Tensor,
    seqlens_per_query: torch.Tensor,
    topk: int,
) -> torch.Tensor:
    if q_fp8.ndim != 3:
        raise ValueError(f"q_fp8 must be rank-3, got {tuple(q_fp8.shape)}")
    if q_fp8.shape[2] != _INDEX_HEAD_DIM:
        raise ValueError(f"q_fp8 head_dim must be {_INDEX_HEAD_DIM}, got {q_fp8.shape[2]}")
    if page_table_1.ndim != 2:
        raise ValueError(f"page_table_1 must be rank-2, got {tuple(page_table_1.shape)}")
    if page_table_1.dtype != torch.int32:
        raise ValueError(f"page_table_1 must have dtype torch.int32, got {page_table_1.dtype}")
    if page_table_1.device != q_fp8.device:
        raise ValueError(
            f"page_table_1 device {page_table_1.device} does not match q_fp8 device {q_fp8.device}"
        )
    if query_row_to_batch.ndim != 1:
        raise ValueError(
            f"query_row_to_batch must be rank-1, got {tuple(query_row_to_batch.shape)}"
        )
    if seqlens_per_query.ndim != 1:
        raise ValueError(
            f"seqlens_per_query must be rank-1, got {tuple(seqlens_per_query.shape)}"
        )
    if query_row_to_batch.shape != seqlens_per_query.shape:
        raise ValueError(
            "query_row_to_batch and seqlens_per_query must have the same shape, got "
            f"{tuple(query_row_to_batch.shape)} vs {tuple(seqlens_per_query.shape)}"
        )
    if query_row_to_batch.device != q_fp8.device:
        raise ValueError(
            f"query_row_to_batch device {query_row_to_batch.device} does not match "
            f"q_fp8 device {q_fp8.device}"
        )
    if seqlens_per_query.device != q_fp8.device:
        raise ValueError(
            f"seqlens_per_query device {seqlens_per_query.device} does not match "
            f"q_fp8 device {q_fp8.device}"
        )
    if topk <= 0:
        raise ValueError(f"topk must be positive, got {topk}")

    num_queries, num_heads, _ = q_fp8.shape
    weights_f = _normalize_weights(weights, q_rows=num_queries, num_heads=num_heads)
    valid_q_rows = int(query_row_to_batch.numel())
    if valid_q_rows > num_queries:
        raise ValueError(f"metadata describes {valid_q_rows} query rows, but q_fp8 has {num_queries}")
    if valid_q_rows and not _is_cuda_graph_capture_active(q_fp8.device):
        invalid_rows = torch.nonzero(
            (query_row_to_batch < 0) | (query_row_to_batch >= page_table_1.shape[0]),
            as_tuple=False,
        )
        if invalid_rows.numel():
            query_row = int(invalid_rows[0, 0].item())
            batch_row = int(query_row_to_batch[query_row].item())
            raise ValueError(
                f"query_row_to_batch[{query_row}]={batch_row} is out of bounds for "
                f"{page_table_1.shape[0]} batch rows"
            )
    return weights_f


def _stable_topk_ids_from_logits(
    *,
    logits: torch.Tensor,
    page_table_1: torch.Tensor,
    query_row_to_batch: torch.Tensor,
    output: torch.Tensor,
    gather_k: int,
) -> None:
    topk_pos = torch.argsort(logits, dim=1, descending=True, stable=True)[:, :gather_k]
    topk_values = torch.gather(logits, 1, topk_pos)
    batch_rows = query_row_to_batch.to(torch.long).unsqueeze(1).expand(-1, gather_k)
    gathered = page_table_1[batch_rows, topk_pos.to(torch.long)]
    output[:, :gather_k] = torch.where(
        torch.isfinite(topk_values),
        gathered,
        torch.full_like(gathered, -1),
    )


def _make_active_width_tensor(
    *,
    seqlens_per_query: torch.Tensor,
    width: int,
) -> torch.Tensor:
    if seqlens_per_query.ndim != 1:
        raise ValueError(
            "seqlens_per_query must be rank-1 when computing active width, got "
            f"{tuple(seqlens_per_query.shape)}"
        )
    width_cap = _cached_width_cap_tensor(
        int(width),
        seqlens_per_query.device.type,
        seqlens_per_query.device.index,
    )
    return torch.minimum(seqlens_per_query.amax().reshape(1), width_cap)


def _sparse_nsa_topk_impl(
    *,
    q_fp8: torch.Tensor,
    weights: torch.Tensor,
    index_k_cache: torch.Tensor,
    page_table_1: torch.Tensor,
    query_row_to_batch: torch.Tensor,
    seqlens_per_query: torch.Tensor,
    topk: int,
    page_size: int,
    allow_fused_decode: bool,
) -> torch.Tensor:
    weights_f = _validate_sparse_index_inputs(
        q_fp8=q_fp8,
        weights=weights,
        page_table_1=page_table_1,
        query_row_to_batch=query_row_to_batch,
        seqlens_per_query=seqlens_per_query,
        topk=topk,
    )
    num_queries = q_fp8.shape[0]
    valid_q_rows = int(query_row_to_batch.numel())
    output = torch.full((num_queries, topk), -1, dtype=torch.int32, device=q_fp8.device)
    if valid_q_rows == 0:
        return output

    if not supports_sparse_nsa_indexer_kernel(
        q_fp8=q_fp8[:valid_q_rows],
        weights=weights_f[:valid_q_rows],
        index_k_cache=index_k_cache,
        page_table_1=page_table_1,
        query_row_to_batch=query_row_to_batch[:valid_q_rows],
        seqlens_per_query=seqlens_per_query[:valid_q_rows],
        page_size=page_size,
    ):
        return sparse_nsa_index_reference(
            q_fp8=q_fp8,
            weights=weights_f,
            index_k_cache=index_k_cache,
            page_table_1=page_table_1,
            query_row_to_batch=query_row_to_batch,
            seqlens_per_query=seqlens_per_query,
            topk=topk,
            page_size=page_size,
        )

    width = page_table_1.shape[1]
    if width == 0:
        return output

    seqlens_valid = seqlens_per_query[:valid_q_rows].contiguous()
    active_width = _make_active_width_tensor(seqlens_per_query=seqlens_valid, width=width)
    max_token_capacity = index_k_cache.shape[0] * page_size
    if not _is_cuda_graph_capture_active(q_fp8.device):
        active_width_host = min(width, int(active_width.item()))
        candidate_tokens = page_table_1.index_select(
            0, query_row_to_batch[:valid_q_rows].to(torch.long)
        )[:, :active_width_host]
        positions = torch.arange(active_width_host, dtype=torch.int32, device=q_fp8.device).unsqueeze(0)
        candidate_valid_mask = (positions < seqlens_valid.unsqueeze(1)) & (candidate_tokens >= 0)
        overflow_mask = candidate_valid_mask & (candidate_tokens >= max_token_capacity)
        if torch.any(overflow_mask):
            bad = int(candidate_tokens[overflow_mask].max().item())
            raise ValueError(
                f"page_table_1 token id {bad} exceeds index_k_cache capacity {max_token_capacity}"
            )

    gather_k = min(topk, width)
    if gather_k == 0:
        return output

    if allow_fused_decode and supports_sparse_nsa_fused_decode_kernel(
        q_fp8=q_fp8[:valid_q_rows],
        weights=weights_f[:valid_q_rows],
        index_k_cache=index_k_cache,
        page_table_1=page_table_1,
        query_row_to_batch=query_row_to_batch[:valid_q_rows],
        seqlens_per_query=seqlens_valid,
        active_width=active_width,
        gather_k=gather_k,
        page_size=page_size,
    ):
        run_sparse_nsa_fused_decode_kernel(
            q_fp8=q_fp8[:valid_q_rows],
            weights=weights_f[:valid_q_rows],
            index_k_cache=index_k_cache,
            page_table_1=page_table_1,
            query_row_to_batch=query_row_to_batch[:valid_q_rows],
            seqlens_per_query=seqlens_valid,
            active_width=active_width,
            output=output[:valid_q_rows],
            page_size=page_size,
        )
        return output

    logits = run_sparse_nsa_index_logits_kernel(
        q_fp8=q_fp8[:valid_q_rows],
        weights=weights_f[:valid_q_rows],
        index_k_cache=index_k_cache,
        page_table_1=page_table_1,
        query_row_to_batch=query_row_to_batch[:valid_q_rows],
        seqlens_per_query=seqlens_valid,
        active_width=active_width,
        trivial_topk=gather_k if allow_fused_decode else 0,
        page_size=page_size,
    )

    if supports_sparse_nsa_dynamic_topk_kernel(
        logits=logits,
        page_table_1=page_table_1,
        query_row_to_batch=query_row_to_batch[:valid_q_rows],
        seqlens_per_query=seqlens_valid,
        active_width=active_width,
        gather_k=gather_k,
    ):
        run_sparse_nsa_dynamic_topk_kernel(
            logits=logits,
            page_table_1=page_table_1,
            query_row_to_batch=query_row_to_batch[:valid_q_rows],
            seqlens_per_query=seqlens_valid,
            active_width=active_width,
            output=output[:valid_q_rows],
            gather_k=gather_k,
        )
    elif supports_sparse_nsa_topk_kernel(
        logits=logits,
        page_table_1=page_table_1,
        query_row_to_batch=query_row_to_batch[:valid_q_rows],
        seqlens_per_query=seqlens_valid,
        gather_k=gather_k,
    ):
        run_sparse_nsa_topk_kernel(
            logits=logits,
            page_table_1=page_table_1,
            query_row_to_batch=query_row_to_batch[:valid_q_rows],
            seqlens_per_query=seqlens_valid,
            output=output[:valid_q_rows],
            gather_k=gather_k,
        )
    else:
        _stable_topk_ids_from_logits(
            logits=logits,
            page_table_1=page_table_1,
            query_row_to_batch=query_row_to_batch[:valid_q_rows],
            output=output[:valid_q_rows],
            gather_k=gather_k,
        )
    return output


def sparse_nsa_index_decode_topk(
    *,
    q_fp8: torch.Tensor,
    weights: torch.Tensor,
    index_k_cache: torch.Tensor,
    metadata: NSAIndexerDecodeMetadata,
    topk: int,
    page_size: int = 64,
) -> torch.Tensor:
    # Decode uses a graph-safe trivial-row fast path: when a live row already
    # fits under `topk`, the CUDA path returns the live page-table prefix
    # directly instead of score-sorting an identical token set.
    if metadata.cache_seqlens_int32.ndim != 1:
        raise ValueError(
            "cache_seqlens_int32 must be rank-1, got "
            f"{tuple(metadata.cache_seqlens_int32.shape)}"
        )
    if metadata.page_table_1.shape[0] != metadata.cache_seqlens_int32.shape[0]:
        raise ValueError(
            f"page_table_1 rows {metadata.page_table_1.shape[0]} do not match "
            f"cache_seqlens rows {metadata.cache_seqlens_int32.shape[0]}"
        )
    if metadata.cache_seqlens_int32.device != q_fp8.device:
        raise ValueError(
            f"cache_seqlens_int32 device {metadata.cache_seqlens_int32.device} does not match "
            f"q_fp8 device {q_fp8.device}"
        )

    query_row_to_batch = torch.arange(metadata.page_table_1.shape[0], dtype=torch.int32, device=q_fp8.device)
    return _sparse_nsa_topk_impl(
        q_fp8=q_fp8,
        weights=weights,
        index_k_cache=index_k_cache,
        page_table_1=metadata.page_table_1,
        query_row_to_batch=query_row_to_batch,
        seqlens_per_query=metadata.cache_seqlens_int32,
        topk=topk,
        page_size=page_size,
        allow_fused_decode=True,
    )


def sparse_nsa_index_extend_topk(
    *,
    q_fp8: torch.Tensor,
    weights: torch.Tensor,
    index_k_cache: torch.Tensor,
    metadata: NSAIndexerExtendMetadata,
    topk: int,
    page_size: int = 64,
) -> torch.Tensor:
    if metadata.nsa_seqlens_expanded.ndim != 1:
        raise ValueError(
            "nsa_seqlens_expanded must be rank-1, got "
            f"{tuple(metadata.nsa_seqlens_expanded.shape)}"
        )
    if metadata.nsa_seqlens_expanded.device != q_fp8.device:
        raise ValueError(
            f"nsa_seqlens_expanded device {metadata.nsa_seqlens_expanded.device} does not match "
            f"q_fp8 device {q_fp8.device}"
        )

    extend_lengths = _cached_extend_lengths_tensor(
        tuple(metadata.nsa_extend_seq_lens_list),
        q_fp8.device.type,
        q_fp8.device.index,
    )
    if extend_lengths.ndim != 1:
        raise ValueError(f"extend lengths must be rank-1, got {tuple(extend_lengths.shape)}")
    if metadata.page_table_1.shape[0] != extend_lengths.shape[0]:
        raise ValueError(
            f"page_table_1 rows {metadata.page_table_1.shape[0]} do not match "
            f"extend length rows {extend_lengths.shape[0]}"
        )

    query_row_to_batch = _cached_query_row_to_batch(
        tuple(metadata.nsa_extend_seq_lens_list),
        q_fp8.device.type,
        q_fp8.device.index,
    )
    if metadata.nsa_seqlens_expanded.shape[0] < query_row_to_batch.shape[0]:
        raise ValueError(
            f"nsa_seqlens_expanded rows {metadata.nsa_seqlens_expanded.shape[0]} are fewer than "
            f"the expanded query rows {query_row_to_batch.shape[0]}"
        )
    seqlens_per_query = metadata.nsa_seqlens_expanded[: query_row_to_batch.shape[0]]
    return _sparse_nsa_topk_impl(
        q_fp8=q_fp8,
        weights=weights,
        index_k_cache=index_k_cache,
        page_table_1=metadata.page_table_1,
        query_row_to_batch=query_row_to_batch,
        seqlens_per_query=seqlens_per_query,
        topk=topk,
        page_size=page_size,
        allow_fused_decode=False,
    )
