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

from bisect import bisect_left
from dataclasses import dataclass
from typing import Literal

import torch

_FP8_KV_DTYPE = torch.float8_e4m3fn
_BF16_DECODE_UPPER_BOUNDS = (
    4, 6, 7, 8, 31, 104, 120, 160, 200, 223, 238, 260, 265, 280, 286,
    297, 315, 351, 372, 373, 382, 383, 384, 441, 442, 497, 498, 560,
    582, 583, 584, 587, 700, 701, 702, 719, 746, 747, 748, 789, 856,
    857, 859, 864, 865, 866, 1007, 1008, 1009, 1010, 1011, 1013, 1014,
    1015, 1024,
)
_BF16_DECODE_WINNERS = (
    1, 3, 5, 8, 9, 10, 13, 14, 17, 18, 20, 22, 24, 26, 28,
    29, 32, 35, 37, 40, 42, 43, 47, 48, 51, 56, 58, 60,
    62, 64, 66, 69, 73, 75, 78, 81, 82, 84, 89, 94, 95,
    97, 99, 101, 103, 106, 109, 111, 113, 115, 118, 120, 122,
    125, 127,
)
_FP8_DECODE_UPPER_BOUNDS = (
    1, 2, 3, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 38, 55, 69, 96,
    129, 163, 205, 249, 298, 303, 325, 342, 367, 393, 420, 430, 453,
    463, 466, 474, 503, 531, 547, 558, 578, 580, 612, 628, 647, 665,
    705, 719, 780, 786, 793, 820, 832, 854, 878, 887, 951, 1370, 1406,
    1465, 1507, 1545, 1609, 1631, 1648, 1649, 1762, 1801, 1802, 1827,
    1855, 1960, 1961, 2048,
)
_FP8_DECODE_WINNERS = (
    1, 2, 3, 4, 6, 8, 10, 13, 14, 15, 16, 17, 18, 19, 22, 23, 24, 25,
    26, 28, 30, 32, 34, 36, 38, 40, 41, 42, 43, 44, 47, 48, 50, 51, 52,
    53, 54, 56, 57, 58, 59, 60, 61, 62, 63, 65, 66, 67, 68, 69, 70, 71,
    78, 85, 93, 95, 97, 99, 101, 102, 103, 105, 111, 118, 121, 127, 134,
    136, 144, 145, 155,
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


def _stub_chunk_pages(page_count: int) -> int:
    return max(1, int(round(max(1, int(page_count)) * 0.2)))


def bf16_decode_chunk_pages(page_count: int) -> int:
    page_count = max(1, int(page_count))
    if page_count <= _BF16_DECODE_UPPER_BOUNDS[-1]:
        return int(_BF16_DECODE_WINNERS[bisect_left(_BF16_DECODE_UPPER_BOUNDS, page_count)])
    return max(127, int(round(127 + 0.11732302295918368 * (page_count - 1024))))


def fp8_decode_chunk_pages(page_count: int) -> int:
    page_count = max(1, int(page_count))
    if page_count <= _FP8_DECODE_UPPER_BOUNDS[-1]:
        return int(_FP8_DECODE_WINNERS[bisect_left(_FP8_DECODE_UPPER_BOUNDS, page_count)])
    return max(227, int(round(227 + 0.047918528318 * (page_count - 2048))))


def bf16_extend_chunk_pages(page_count: int) -> int:
    return max(1, int(page_count))


def fp8_extend_chunk_pages(page_count: int) -> int:
    return max(1, int(page_count))


def chunk_pages_for_family(
    *,
    mode: Literal["decode", "extend"],
    q_dtype: torch.dtype,
    kv_dtype: torch.dtype,
    page_size: int,
    head_dim_qk: int,
    head_dim_vo: int,
    gqa_group_size: int,
    max_effective_kv_pages: int,
) -> int:
    del q_dtype, page_size, head_dim_qk, head_dim_vo, gqa_group_size
    if mode == "decode" and kv_dtype == torch.bfloat16:
        return bf16_decode_chunk_pages(max_effective_kv_pages)
    if mode == "decode" and kv_dtype == _FP8_KV_DTYPE:
        return fp8_decode_chunk_pages(max_effective_kv_pages)
    if mode == "extend" and kv_dtype == torch.bfloat16:
        return bf16_extend_chunk_pages(max_effective_kv_pages)
    if mode == "extend" and kv_dtype == _FP8_KV_DTYPE:
        return fp8_extend_chunk_pages(max_effective_kv_pages)
    raise TypeError(f"unsupported chunk-policy family: mode={mode} kv_dtype={kv_dtype}")


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
    needs the runtime chunk size, not a full host work decomposition.
    """
    return chunk_pages_for_family(
        mode="decode",
        q_dtype=q_dtype,
        kv_dtype=kv_dtype,
        page_size=page_size,
        head_dim_qk=head_dim_qk,
        head_dim_vo=head_dim_vo,
        gqa_group_size=gqa_group_size,
        max_effective_kv_pages=max(max_effective_kv_pages, 1),
    )


def build_decode_chunk_pages_lut(
    *,
    q_dtype: torch.dtype,
    kv_dtype: torch.dtype,
    page_size: int,
    head_dim_qk: int,
    head_dim_vo: int,
    gqa_group_size: int,
    max_effective_kv_pages: int,
) -> tuple[int, ...]:
    max_effective_kv_pages = max(int(max_effective_kv_pages), 1)
    return tuple(
        decode_chunk_pages_for_graph(
            q_dtype=q_dtype,
            kv_dtype=kv_dtype,
            page_size=page_size,
            head_dim_qk=head_dim_qk,
            head_dim_vo=head_dim_vo,
            gqa_group_size=gqa_group_size,
            max_effective_kv_pages=page_count,
        )
        for page_count in range(1, max_effective_kv_pages + 1)
    )


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
        heuristic_kv_chunk_size_pages = chunk_pages_for_family(
            mode=mode,
            q_dtype=q.dtype,
            kv_dtype=k_cache.dtype,
            page_size=page_size,
            head_dim_qk=head_dim_qk,
            head_dim_vo=head_dim_vo,
            gqa_group_size=gqa_group_size,
            max_effective_kv_pages=max(max(effective_kv_len_arr), 1),
        )
        split_kv = False
        kv_chunk_size_pages = heuristic_kv_chunk_size_pages

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
