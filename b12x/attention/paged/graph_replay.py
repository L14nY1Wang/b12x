"""Device-side decode graph replay helpers for the paged attention backend."""

from __future__ import annotations

from collections.abc import Sequence

import torch
import triton
import triton.language as tl

_DECODE_BLOCK_CHUNKS = 128
_DECODE_BLOCK_PAGES = 128


@triton.jit
def build_decode_graph_page_table_triton(
    req_to_token_ptr,
    req_pool_indices_ptr,
    page_table_ptr,
    req_to_token_row_stride,
    page_table_row_stride,
    max_pages_per_req,
    PAGE_SIZE: tl.constexpr,
    BLOCK_PAGES: tl.constexpr,
):
    req_idx = tl.program_id(axis=0)
    page_block_idx = tl.program_id(axis=1)

    req_pool_idx = tl.load(req_pool_indices_ptr + req_idx).to(tl.int64)
    page_offsets = page_block_idx * BLOCK_PAGES + tl.arange(0, BLOCK_PAGES)
    page_mask = page_offsets < max_pages_per_req
    flat_token_offsets = req_pool_idx * req_to_token_row_stride + page_offsets.to(tl.int64) * PAGE_SIZE
    token_indices = tl.load(req_to_token_ptr + flat_token_offsets, mask=page_mask, other=0)
    tl.store(
        page_table_ptr + req_idx * page_table_row_stride + page_offsets,
        (token_indices // PAGE_SIZE).to(tl.int32),
        mask=page_mask,
    )


@triton.jit
def update_decode_graph_metadata_triton(
    cache_seqlens_ptr,
    merge_indptr_ptr,
    block_valid_mask_ptr,
    chunk_pages_ptr,
    max_chunks_per_req,
    PAGE_SIZE: tl.constexpr,
    BLOCK_CHUNKS: tl.constexpr,
):
    req_idx = tl.program_id(axis=0)
    chunk_block_idx = tl.program_id(axis=1)

    cache_len = tl.load(cache_seqlens_ptr + req_idx).to(tl.int32)
    chunk_pages = tl.load(chunk_pages_ptr).to(tl.int32)
    num_pages = tl.maximum((cache_len + (PAGE_SIZE - 1)) // PAGE_SIZE, 1)
    num_chunks = (num_pages + chunk_pages - 1) // chunk_pages

    tl.store(merge_indptr_ptr + req_idx + 1, num_chunks)

    chunk_offsets = chunk_block_idx * BLOCK_CHUNKS + tl.arange(0, BLOCK_CHUNKS)
    chunk_mask = chunk_offsets < max_chunks_per_req
    is_active = chunk_offsets < num_chunks
    tl.store(
        block_valid_mask_ptr + req_idx * max_chunks_per_req + chunk_offsets,
        is_active.to(tl.int32),
        mask=chunk_mask,
    )


@triton.jit
def update_regular_decode_graph_metadata_triton(
    cache_seqlens_ptr,
    merge_indptr_ptr,
    chunk_pages_ptr,
    PAGE_SIZE: tl.constexpr,
):
    req_idx = tl.program_id(axis=0)

    cache_len = tl.load(cache_seqlens_ptr + req_idx).to(tl.int32)
    chunk_pages = tl.load(chunk_pages_ptr).to(tl.int32)
    kv_chunk_size = chunk_pages * PAGE_SIZE
    num_chunks = tl.maximum((cache_len + kv_chunk_size - 1) // kv_chunk_size, 1)

    tl.store(merge_indptr_ptr + req_idx + 1, num_chunks)


def make_decode_chunk_pages_lut_tensor(
    decode_chunk_pages_lut: Sequence[int],
    *,
    device: torch.device,
) -> torch.Tensor:
    if not decode_chunk_pages_lut:
        raise ValueError("decode chunk-pages LUT must be non-empty")
    if any(int(chunk_pages) <= 0 for chunk_pages in decode_chunk_pages_lut):
        raise ValueError("decode chunk-pages LUT must contain only positive values")
    return torch.tensor(
        (int(decode_chunk_pages_lut[0]), *(int(chunk_pages) for chunk_pages in decode_chunk_pages_lut)),
        dtype=torch.int32,
        device=device,
    )


def summarize_decode_chunk_pages_lut(
    decode_chunk_pages_lut: Sequence[int],
) -> tuple[int, int]:
    if not decode_chunk_pages_lut:
        raise ValueError("decode chunk-pages LUT must be non-empty")
    worst_page_count = 1
    max_chunks_per_req = 1
    for page_count, chunk_pages in enumerate(decode_chunk_pages_lut, start=1):
        num_chunks = (page_count + int(chunk_pages) - 1) // int(chunk_pages)
        if num_chunks > max_chunks_per_req:
            max_chunks_per_req = num_chunks
            worst_page_count = page_count
    return int(worst_page_count), int(max_chunks_per_req)


def update_decode_graph_replay_metadata(
    *,
    req_to_token: torch.Tensor,
    req_pool_indices: torch.Tensor,
    page_table: torch.Tensor,
    cache_seqlens: torch.Tensor,
    request_indices: torch.Tensor,
    merge_indptr: torch.Tensor,
    o_indptr: torch.Tensor,
    block_valid_mask: torch.Tensor,
    kv_chunk_size_ptr: torch.Tensor,
    decode_chunk_pages_lut: torch.Tensor,
    page_size: int,
) -> None:
    if req_to_token.device != page_table.device:
        raise ValueError("req_to_token and page_table must be on the same device")
    if req_pool_indices.device != page_table.device:
        raise ValueError("req_pool_indices and page_table must be on the same device")
    if cache_seqlens.device != page_table.device:
        raise ValueError("cache_seqlens and page_table must be on the same device")
    if decode_chunk_pages_lut.device != page_table.device:
        raise ValueError("decode_chunk_pages_lut and page_table must be on the same device")
    if page_size <= 0:
        raise ValueError("page_size must be positive")

    bs = int(cache_seqlens.shape[0])
    if bs <= 0:
        raise ValueError("decode graph replay requires bs > 0")
    if int(req_pool_indices.shape[0]) != bs:
        raise ValueError("req_pool_indices shape must match cache_seqlens batch")
    work_items_capacity = int(request_indices.shape[0])
    if work_items_capacity % bs != 0:
        raise RuntimeError("decode graph workspace request_indices shape is incompatible with the batch bucket")
    max_chunks_per_req = work_items_capacity // bs
    if max_chunks_per_req <= 0:
        raise RuntimeError("decode graph workspace must allocate at least one chunk per request")

    max_cache_pages = torch.div(
        cache_seqlens[:bs].amax() + (page_size - 1),
        page_size,
        rounding_mode="floor",
    ).clamp_(min=1, max=page_table.shape[1]).to(torch.int64)
    decode_chunk_pages = torch.index_select(decode_chunk_pages_lut, 0, max_cache_pages.view(1))

    page_blocks = triton.cdiv(int(page_table.shape[1]), _DECODE_BLOCK_PAGES)
    build_decode_graph_page_table_triton[(bs, page_blocks)](
        req_to_token,
        req_pool_indices,
        page_table,
        req_to_token.stride(0),
        page_table.stride(0),
        page_table.shape[1],
        PAGE_SIZE=page_size,
        BLOCK_PAGES=_DECODE_BLOCK_PAGES,
    )

    block_valid_mask.zero_()
    merge_indptr.zero_()
    chunk_blocks = triton.cdiv(max_chunks_per_req, _DECODE_BLOCK_CHUNKS)
    update_decode_graph_metadata_triton[(bs, chunk_blocks)](
        cache_seqlens,
        merge_indptr,
        block_valid_mask,
        decode_chunk_pages,
        max_chunks_per_req,
        PAGE_SIZE=page_size,
        BLOCK_CHUNKS=_DECODE_BLOCK_CHUNKS,
    )
    torch.cumsum(
        merge_indptr[1 : bs + 1],
        dim=0,
        out=merge_indptr[1 : bs + 1],
    )
    o_indptr[: bs + 1].copy_(merge_indptr[: bs + 1])
    kv_chunk_size_ptr.copy_(decode_chunk_pages * page_size)


def update_regular_decode_graph_replay_metadata(
    *,
    req_to_token: torch.Tensor,
    req_pool_indices: torch.Tensor,
    page_table: torch.Tensor,
    cache_seqlens: torch.Tensor,
    merge_indptr: torch.Tensor,
    o_indptr: torch.Tensor,
    kv_chunk_size_ptr: torch.Tensor,
    decode_chunk_pages_lut: torch.Tensor,
    page_size: int,
) -> None:
    if req_to_token.device != page_table.device:
        raise ValueError("req_to_token and page_table must be on the same device")
    if req_pool_indices.device != page_table.device:
        raise ValueError("req_pool_indices and page_table must be on the same device")
    if cache_seqlens.device != page_table.device:
        raise ValueError("cache_seqlens and page_table must be on the same device")
    if decode_chunk_pages_lut.device != page_table.device:
        raise ValueError("decode_chunk_pages_lut and page_table must be on the same device")
    if page_size <= 0:
        raise ValueError("page_size must be positive")

    bs = int(cache_seqlens.shape[0])
    if bs <= 0:
        raise ValueError("decode graph replay requires bs > 0")
    if int(req_pool_indices.shape[0]) != bs:
        raise ValueError("req_pool_indices shape must match cache_seqlens batch")

    max_cache_pages = torch.div(
        cache_seqlens[:bs].amax() + (page_size - 1),
        page_size,
        rounding_mode="floor",
    ).clamp_(min=1, max=page_table.shape[1]).to(torch.int64)
    decode_chunk_pages = torch.index_select(decode_chunk_pages_lut, 0, max_cache_pages.view(1))

    page_blocks = triton.cdiv(int(page_table.shape[1]), _DECODE_BLOCK_PAGES)
    build_decode_graph_page_table_triton[(bs, page_blocks)](
        req_to_token,
        req_pool_indices,
        page_table,
        req_to_token.stride(0),
        page_table.stride(0),
        page_table.shape[1],
        PAGE_SIZE=page_size,
        BLOCK_PAGES=_DECODE_BLOCK_PAGES,
    )

    merge_indptr.zero_()
    update_regular_decode_graph_metadata_triton[(bs,)](
        cache_seqlens,
        merge_indptr,
        decode_chunk_pages,
        PAGE_SIZE=page_size,
    )
    torch.cumsum(
        merge_indptr[1 : bs + 1],
        dim=0,
        out=merge_indptr[1 : bs + 1],
    )
    o_indptr[: bs + 1].copy_(merge_indptr[: bs + 1])
    kv_chunk_size_ptr.copy_(decode_chunk_pages * page_size)


def update_regular_decode_graph_chunk_metadata(
    *,
    cache_seqlens: torch.Tensor,
    merge_indptr: torch.Tensor,
    o_indptr: torch.Tensor,
    kv_chunk_size_ptr: torch.Tensor,
    kv_chunk_size: int,
    max_chunks_per_req: int,
) -> None:
    device = cache_seqlens.device
    if merge_indptr.device != device or o_indptr.device != device:
        raise ValueError("indptr buffers and cache_seqlens must be on the same device")
    if kv_chunk_size_ptr.device != device:
        raise ValueError("decode graph buffers and cache_seqlens must be on the same device")
    if kv_chunk_size <= 0:
        raise ValueError("kv_chunk_size must be positive")
    if max_chunks_per_req <= 0:
        raise ValueError("max_chunks_per_req must be positive")

    bs = int(cache_seqlens.shape[0])
    if bs <= 0:
        raise ValueError("decode graph replay requires bs > 0")

    merge_indptr.zero_()
    num_chunks = torch.maximum(
        torch.div(
            cache_seqlens[:bs] + (int(kv_chunk_size) - 1),
            int(kv_chunk_size),
            rounding_mode="floor",
        ),
        torch.ones(bs, dtype=torch.int32, device=device),
    )
    merge_indptr[1 : bs + 1].copy_(num_chunks)
    torch.cumsum(
        merge_indptr[1 : bs + 1],
        dim=0,
        out=merge_indptr[1 : bs + 1],
    )
    o_indptr[: bs + 1].copy_(merge_indptr[: bs + 1])
    kv_chunk_size_ptr[0] = int(kv_chunk_size)
