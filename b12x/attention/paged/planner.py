"""Host planner for the primary paged-attention backend.

This module models the host-side work decomposition used by FlashInfer's paged
attention kernels:

- choose `CTA_TILE_Q` from packed Q rows,
- choose `kv_chunk_size` on the host,
- emit exact `(request_idx, qo_tile_idx, kv_tile_idx)` worklists,
- emit `merge_indptr` / `o_indptr` for split reduction.

No kernel-side split LUT or legacy scheduler assumptions live here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch

_FP8_KV_DTYPE = torch.float8_e4m3fn
_PAGED_EXTEND_FP8_CHUNK_TABLE_PAGES = (
    (1, 1),
    (2, 1),
    (4, 1),
    (8, 1),
    (16, 1),
    (32, 2),
    (64, 3),
    (128, 6),
    (256, 6),
    (512, 24),
    (1024, 24),
    (2048, 24),
)
_PAGED_EXTEND_FP8_GRAPH_CHUNK_TABLE_PAGES = (
    (1, 1),
    (2, 1),
    (4, 1),
    (8, 1),
    (16, 1),
    (32, 2),
    (64, 3),
    (128, 6),
    (256, 6),
    (512, 24),
    (1024, 24),
    (2048, 24),
)
_PAGED_EXTEND_BF16_CHUNK_TABLE_PAGES = (
    (1, 1),
    (2, 1),
    (4, 1),
    (8, 1),
    (16, 1),
    (32, 2),
    (64, 3),
    (128, 6),
    (256, 6),
    (512, 32),
    (1024, 32),
    (2048, 32),
)
_PAGED_DECODE_BF16_CHUNK_TABLE_PAGES = (
    (1, 1),
    (2, 2),
    (16, 1),
    (32, 2),
    (64, 3),
    (128, 6),
    (256, 12),
    (320, 16),
    (512, 48),
    (640, 64),
    (960, 96),
    (2048, 128),
)
_PAGED_DECODE_FP8_CHUNK_TABLE_PAGES = (
    (1, 2),
    (2, 2),
    (4, 2),
    (16, 1),
    (32, 2),
    (64, 3),
    (128, 6),
    (256, 12),
    (320, 16),
    (512, 24),
    (640, 32),
    (768, 48),
    (1024, 64),
    (1536, 96),
    (2048, 128),
)


def _merge_backend_supports_split_kv(
    *,
    output_dtype: torch.dtype,
    head_dim_vo: int,
) -> bool:
    return output_dtype in (torch.float16, torch.bfloat16, torch.float32) and head_dim_vo == 256


def _ceil_div(x: int, y: int) -> int:
    return (x + y - 1) // y


def _metadata_to_cpu_int_list(t: torch.Tensor, *, name: str) -> list[int]:
    if t.dtype not in (torch.int32, torch.int64):
        raise TypeError(f"{name} must be torch.int32 or torch.int64")
    if not t.is_contiguous():
        raise ValueError(f"{name} must be contiguous")
    return [int(v) for v in t.detach().cpu().tolist()]


def _q_lengths_from_cu_seqlens(cu_seqlens_q: torch.Tensor) -> list[int]:
    cu_seqlens_q_list = _metadata_to_cpu_int_list(cu_seqlens_q, name="cu_seqlens_q")
    q_lengths: list[int] = []
    for start, end in zip(cu_seqlens_q_list[:-1], cu_seqlens_q_list[1:]):
        if end < start:
            raise ValueError("cu_seqlens_q must be non-decreasing")
        q_lengths.append(end - start)
    return q_lengths


def infer_paged_mode(cu_seqlens_q: torch.Tensor) -> Literal["decode", "extend"]:
    q_lengths = _q_lengths_from_cu_seqlens(cu_seqlens_q)
    return "decode" if q_lengths and all(q_len == 1 for q_len in q_lengths) else "extend"


def _fa2_determine_cta_tile_q(avg_packed_qo_len: int, head_dim: int) -> int:
    # Faithful to FlashInfer's FA2DetermineCtaTileQ.
    if avg_packed_qo_len > 64 and head_dim < 256:
        return 128
    if avg_packed_qo_len > 16:
        return 64
    return 16


def _paged_determine_cta_tile_q(
    *,
    mode: Literal["decode", "extend"],
    kv_dtype: torch.dtype,
    packed_qo_len: int,
    head_dim: int,
    max_effective_kv_pages: int,
) -> int:
    cta_tile_q = _fa2_determine_cta_tile_q(packed_qo_len, head_dim)
    if mode == "extend" and kv_dtype == _FP8_KV_DTYPE and cta_tile_q == 64:
        if max_effective_kv_pages <= 8:
            return 32
        return 48
    return cta_tile_q


def _prefill_binary_search_kv_chunk_size(
    *,
    enable_cuda_graph: bool,
    max_batch_size_if_split: int,
    packed_qo_len_arr: list[int],
    kv_len_arr: list[int],
    qo_chunk_size: int,
    min_kv_chunk_size: int = 1,
) -> tuple[bool, int]:
    batch_size = len(packed_qo_len_arr)
    max_kv_len = max(max(kv_len_arr, default=1), 1)
    low = min_kv_chunk_size
    high = max_kv_len
    while low < high:
        mid = (low + high) // 2
        new_batch_size = 0
        for i in range(batch_size):
            new_batch_size += _ceil_div(packed_qo_len_arr[i], qo_chunk_size) * _ceil_div(
                max(kv_len_arr[i], 1), mid
            )
        if new_batch_size > max_batch_size_if_split:
            low = mid + 1
        else:
            high = mid
    return (enable_cuda_graph or low < max_kv_len, low)


def _lookup_chunk_pages_from_table(
    max_effective_kv_pages: int,
    table: tuple[tuple[int, int | None], ...],
) -> int | None:
    for max_pages, chunk_pages in table:
        if max_effective_kv_pages <= max_pages:
            return chunk_pages
    return None


def _paged_chunk_table_pages(
    *,
    mode: Literal["decode", "extend"],
    q_dtype: torch.dtype,
    kv_dtype: torch.dtype,
    page_size: int,
    head_dim_qk: int,
    head_dim_vo: int,
    gqa_group_size: int,
    max_effective_kv_pages: int,
    graph_chunk_policy: bool,
) -> int | None:
    """Explicit table-driven chunk override for the tuned serving matrix.

    FlashInfer's planner is tuned around its own kernel/merge cost model. Our
    current paged backend currently has materially higher per-chunk overhead on
    the shipped serving matrix, so the same binary-search objective ends up
    over-partitioning requests in some regimes. Keep these overrides narrow
    and table-driven.
    """
    if page_size != 64 or head_dim_qk != 256 or head_dim_vo != 256 or gqa_group_size != 8:
        return None
    if q_dtype != torch.bfloat16:
        return None
    if mode == "extend" and kv_dtype == _FP8_KV_DTYPE:
        table = (
            _PAGED_EXTEND_FP8_GRAPH_CHUNK_TABLE_PAGES
            if graph_chunk_policy
            else _PAGED_EXTEND_FP8_CHUNK_TABLE_PAGES
        )
        return _lookup_chunk_pages_from_table(
            max_effective_kv_pages,
            table,
        )
    if mode == "decode" and kv_dtype == _FP8_KV_DTYPE:
        return _lookup_chunk_pages_from_table(
            max_effective_kv_pages,
            _PAGED_DECODE_FP8_CHUNK_TABLE_PAGES,
        )
    if mode == "extend" and kv_dtype == torch.bfloat16:
        return _lookup_chunk_pages_from_table(
            max_effective_kv_pages,
            _PAGED_EXTEND_BF16_CHUNK_TABLE_PAGES,
        )
    if mode == "decode" and kv_dtype == torch.bfloat16:
        return _lookup_chunk_pages_from_table(
            max_effective_kv_pages,
            _PAGED_DECODE_BF16_CHUNK_TABLE_PAGES,
        )
    return None


def decode_chunk_pages_for_graph(
    *,
    q_dtype: torch.dtype,
    kv_dtype: torch.dtype,
    page_size: int,
    head_dim_qk: int,
    head_dim_vo: int,
    gqa_group_size: int,
    max_effective_kv_pages: int,
) -> int:
    """Return the decode split chunk size in pages for graph replay.

    Default decode now stays on the split-capable generic family, so replay only
    needs the runtime chunk size, not a full host work decomposition. For the
    tuned decode contracts we reuse the explicit table overrides. Outside the
    tuned table range, `create_paged_plan()` falls back to the binary-search
    path, which in graph mode immediately selects the minimum chunk size because
    the decode graph budget is intentionally overprovisioned. Mirror that here
    without rebuilding the full host plan.
    """

    chunk_pages = _paged_chunk_table_pages(
        mode="decode",
        q_dtype=q_dtype,
        kv_dtype=kv_dtype,
        page_size=page_size,
        head_dim_qk=head_dim_qk,
        head_dim_vo=head_dim_vo,
        gqa_group_size=gqa_group_size,
        max_effective_kv_pages=max(max_effective_kv_pages, 1),
        graph_chunk_policy=True,
    )
    if chunk_pages is not None:
        return int(chunk_pages)
    return max(128 // page_size, 1)


@dataclass(frozen=True)
class PagedPlanKey:
    total_q: int
    num_q_heads: int
    head_dim_qk: int
    head_dim_vo: int
    k_cache_shape: tuple[int, ...]
    v_cache_shape: tuple[int, ...]
    page_table_shape: tuple[int, ...]
    dtype: torch.dtype
    kv_dtype: torch.dtype
    mode: Literal["decode", "extend"]
    cta_tile_q: int
    kv_chunk_size: int
    split_kv: bool
    fixed_split_size: int
    disable_split_kv: bool
    enable_cuda_graph: bool
    graph_chunk_policy: bool
    max_batch_size_if_split: int
    padded_batch_size: int
    new_batch_size: int
    num_qo_tiles: int
    total_num_partial_rows: int
    page_size: int
    num_kv_heads: int
    gqa_group_size: int
    device_index: int


@dataclass(frozen=True, kw_only=True)
class PagedPlan:
    key: PagedPlanKey
    request_indices: tuple[int, ...]
    qo_tile_indices: tuple[int, ...]
    kv_tile_indices: tuple[int, ...]
    merge_indptr: tuple[int, ...]
    o_indptr: tuple[int, ...]
    block_valid_mask: tuple[bool, ...]

    def __getattr__(self, name: str):
        return getattr(self.key, name)

    @property
    def device(self) -> torch.device:
        return torch.device("cuda", self.device_index)


def create_paged_plan(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    page_table: torch.Tensor,
    cache_seqlens: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    *,
    mode: Literal["decode", "extend"] | None = None,
    fixed_split_size: int = -1,
    disable_split_kv: bool = False,
    force_split_kv: bool | None = None,
    enable_cuda_graph: bool = False,
    graph_chunk_policy: bool = False,
    max_batch_size_if_split: int | None = None,
    window_left: int = -1,
) -> PagedPlan:
    if q.ndim != 3:
        raise ValueError(f"q must be rank-3 [total_q, q_heads, head_dim], got {tuple(q.shape)}")
    if k_cache.ndim != 4:
        raise ValueError(
            f"k_cache must be rank-4 [num_pages, page_size, kv_heads, head_dim], got {tuple(k_cache.shape)}"
        )
    if v_cache.ndim != 4:
        raise ValueError(
            f"v_cache must be rank-4 [num_pages, page_size, kv_heads, head_dim_v], got {tuple(v_cache.shape)}"
        )
    if page_table.ndim != 2:
        raise ValueError(f"page_table must be rank-2 [batch, max_pages], got {tuple(page_table.shape)}")
    if cache_seqlens.ndim != 1:
        raise ValueError(f"cache_seqlens must be rank-1 [batch], got {tuple(cache_seqlens.shape)}")
    if cu_seqlens_q.ndim != 1:
        raise ValueError(f"cu_seqlens_q must be rank-1 [batch+1], got {tuple(cu_seqlens_q.shape)}")
    if q.device.type != "cuda":
        raise ValueError("q must be on CUDA")
    if not (k_cache.device == v_cache.device == page_table.device == cache_seqlens.device == cu_seqlens_q.device == q.device):
        raise ValueError("all inputs must be on the same CUDA device")
    if q.dtype not in (torch.float16, torch.bfloat16):
        raise TypeError(f"unsupported q dtype {q.dtype}")
    if k_cache.dtype != v_cache.dtype:
        raise TypeError("k_cache and v_cache must have matching dtypes")
    if k_cache.dtype not in (torch.float16, torch.bfloat16, _FP8_KV_DTYPE):
        raise TypeError(f"unsupported kv dtype {k_cache.dtype}")

    total_q, num_q_heads, head_dim_qk = [int(dim) for dim in q.shape]
    num_pages, page_size, num_kv_heads, head_dim_k = [int(dim) for dim in k_cache.shape]
    v_num_pages, v_page_size, v_num_kv_heads, head_dim_vo = [int(dim) for dim in v_cache.shape]
    batch, max_pages_per_request = [int(dim) for dim in page_table.shape]

    if num_pages != v_num_pages or page_size != v_page_size or num_kv_heads != v_num_kv_heads:
        raise ValueError("k_cache and v_cache structural shapes must match except head_dim")
    if head_dim_k != head_dim_qk:
        raise ValueError("primary paged backend expects head_dim_qk to match k_cache head_dim")
    if page_size != 64:
        raise ValueError(f"primary paged backend expects page_size=64, got {page_size}")
    if num_q_heads % num_kv_heads != 0:
        raise ValueError("num_q_heads must be divisible by num_kv_heads")
    if tuple(cache_seqlens.shape) != (batch,):
        raise ValueError("cache_seqlens shape must match page_table batch")
    if tuple(cu_seqlens_q.shape) != (batch + 1,):
        raise ValueError("cu_seqlens_q shape must be [batch + 1]")

    q_lengths = _q_lengths_from_cu_seqlens(cu_seqlens_q)
    cache_lengths = _metadata_to_cpu_int_list(cache_seqlens, name="cache_seqlens")
    if any(cache_len <= 0 for cache_len in cache_lengths):
        raise ValueError("primary paged backend requires cache_seqlens > 0")
    cache_pages_arr = [_ceil_div(cache_len, page_size) for cache_len in cache_lengths]
    if any(cache_pages > max_pages_per_request for cache_pages in cache_pages_arr):
        raise ValueError("page_table width is smaller than required by cache_seqlens")

    inferred_mode = infer_paged_mode(cu_seqlens_q)
    mode = inferred_mode if mode is None else mode
    if force_split_kv is None:
        force_split_kv = mode == "decode"

    gqa_group_size = num_q_heads // num_kv_heads
    packed_qo_len_arr = [q_len * gqa_group_size for q_len in q_lengths]
    kv_len_arr = list(cache_pages_arr)

    if enable_cuda_graph:
        total_num_rows = total_q
        max_seq_len = total_num_rows - batch + 1
        max_qo_len = max_seq_len * gqa_group_size
        max_effective_kv_pages = max(kv_len_arr) if window_left < 0 else min(
            _ceil_div(window_left + _fa2_determine_cta_tile_q(max_qo_len, head_dim_qk), page_size),
            max(kv_len_arr),
        )
        cta_tile_q = _paged_determine_cta_tile_q(
            mode=mode,
            kv_dtype=k_cache.dtype,
            packed_qo_len=max_qo_len,
            head_dim=head_dim_qk,
            max_effective_kv_pages=max(max_effective_kv_pages, 1),
        )
        total_num_qo_tiles = _ceil_div(total_num_rows * gqa_group_size, cta_tile_q) + batch - 1
    else:
        avg_packed_qo_len = sum(packed_qo_len_arr) // max(batch, 1)
        max_effective_kv_pages = max(kv_len_arr) if window_left < 0 else min(
            _ceil_div(window_left + _fa2_determine_cta_tile_q(avg_packed_qo_len, head_dim_qk), page_size),
            max(kv_len_arr),
        )
        cta_tile_q = _paged_determine_cta_tile_q(
            mode=mode,
            kv_dtype=k_cache.dtype,
            packed_qo_len=avg_packed_qo_len,
            head_dim=head_dim_qk,
            max_effective_kv_pages=max(max_effective_kv_pages, 1),
        )
        total_num_qo_tiles = sum(_ceil_div(packed_qo_len, cta_tile_q) for packed_qo_len in packed_qo_len_arr)

    effective_kv_len_arr = [
        min(_ceil_div(window_left + cta_tile_q, page_size), kv_len) if window_left >= 0 else kv_len
        for kv_len in kv_len_arr
    ]
    min_kv_chunk_size = max(128 // page_size, 1)
    if max_batch_size_if_split is None:
        max_batch_size_if_split = max(total_num_qo_tiles, 1) * max(max(effective_kv_len_arr), 1)

    if not _merge_backend_supports_split_kv(output_dtype=q.dtype, head_dim_vo=head_dim_vo):
        disable_split_kv = True

    if disable_split_kv and not force_split_kv:
        split_kv = False
        kv_chunk_size_pages = 1 << 30
    elif fixed_split_size > 0:
        split_kv = False
        kv_chunk_size_pages = fixed_split_size
    else:
        heuristic_kv_chunk_size_pages = _paged_chunk_table_pages(
            mode=mode,
            q_dtype=q.dtype,
            kv_dtype=k_cache.dtype,
            page_size=page_size,
            head_dim_qk=head_dim_qk,
            head_dim_vo=head_dim_vo,
            gqa_group_size=gqa_group_size,
            max_effective_kv_pages=max(max(effective_kv_len_arr), 1),
            graph_chunk_policy=graph_chunk_policy,
        )
        if heuristic_kv_chunk_size_pages is not None:
            split_kv = False
            kv_chunk_size_pages = heuristic_kv_chunk_size_pages
        else:
            split_kv, kv_chunk_size_pages = _prefill_binary_search_kv_chunk_size(
                enable_cuda_graph=enable_cuda_graph,
                max_batch_size_if_split=max_batch_size_if_split,
                packed_qo_len_arr=packed_qo_len_arr,
                kv_len_arr=effective_kv_len_arr,
                qo_chunk_size=cta_tile_q,
                min_kv_chunk_size=min_kv_chunk_size,
            )

    request_indices: list[int] = []
    qo_tile_indices: list[int] = []
    kv_tile_indices: list[int] = []
    merge_indptr: list[int] = [0]
    o_indptr: list[int] = [0]
    new_batch_size = 0

    for request_idx, (packed_qo_len, qo_len, kv_len) in enumerate(
        zip(packed_qo_len_arr, q_lengths, effective_kv_len_arr)
    ):
        num_tiles_q = _ceil_div(packed_qo_len, cta_tile_q)
        num_chunks_kv = 1 if disable_split_kv and not force_split_kv else _ceil_div(max(kv_len, 1), kv_chunk_size_pages)
        if not disable_split_kv or force_split_kv:
            split_kv = split_kv or num_chunks_kv > 1
        for q_tile_idx in range(num_tiles_q):
            for kv_tile_idx in range(num_chunks_kv):
                new_batch_size += 1
                request_indices.append(request_idx)
                qo_tile_indices.append(q_tile_idx)
                kv_tile_indices.append(kv_tile_idx)
        for _ in range(qo_len):
            merge_indptr.append(merge_indptr[-1] + num_chunks_kv)
        o_indptr.append(o_indptr[-1] + qo_len * num_chunks_kv)

    padded_batch_size = (
        max(max_batch_size_if_split, total_num_qo_tiles) if enable_cuda_graph else new_batch_size
    )
    if new_batch_size > padded_batch_size:
        raise ValueError(
            "new_batch_size exceeds padded_batch_size; fixed_split_size is incompatible with the chosen graph budget"
        )
    if force_split_kv:
        split_kv = True

    block_valid_mask = [idx < new_batch_size for idx in range(padded_batch_size)]
    kv_chunk_size = kv_chunk_size_pages * page_size

    key = PagedPlanKey(
        total_q=total_q,
        num_q_heads=num_q_heads,
        head_dim_qk=head_dim_qk,
        head_dim_vo=head_dim_vo,
        k_cache_shape=tuple(int(dim) for dim in k_cache.shape),
        v_cache_shape=tuple(int(dim) for dim in v_cache.shape),
        page_table_shape=tuple(int(dim) for dim in page_table.shape),
        dtype=q.dtype,
        kv_dtype=k_cache.dtype,
        mode=mode,
        cta_tile_q=cta_tile_q,
        kv_chunk_size=kv_chunk_size,
        split_kv=split_kv,
        fixed_split_size=fixed_split_size,
        disable_split_kv=disable_split_kv,
        enable_cuda_graph=enable_cuda_graph,
        graph_chunk_policy=graph_chunk_policy,
        max_batch_size_if_split=max_batch_size_if_split,
        padded_batch_size=padded_batch_size,
        new_batch_size=new_batch_size,
        num_qo_tiles=total_num_qo_tiles,
        total_num_partial_rows=o_indptr[-1],
        page_size=page_size,
        num_kv_heads=num_kv_heads,
        gqa_group_size=gqa_group_size,
        device_index=q.device.index if q.device.index is not None else torch.cuda.current_device(),
    )
    return PagedPlan(
        key=key,
        request_indices=tuple(request_indices),
        qo_tile_indices=tuple(qo_tile_indices),
        kv_tile_indices=tuple(kv_tile_indices),
        merge_indptr=tuple(merge_indptr),
        o_indptr=tuple(o_indptr),
        block_valid_mask=tuple(block_valid_mask),
    )
