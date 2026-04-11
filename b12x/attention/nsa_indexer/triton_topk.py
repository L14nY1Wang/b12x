from __future__ import annotations

import os
from functools import lru_cache

import torch
import triton
import triton.language as tl


_DEFAULT_MAX_WIDTH_BLOCK = 2048
_LARGE_MAX_WIDTH_BLOCK = 4096
_MAX_TOPK_BLOCK = 2048
_MAX_MERGE_WIDTH_BLOCK = 4096


def _dynamic_width_block(gather_k: int) -> int:
    override = os.environ.get("B12X_NSA_INDEXER_DYNAMIC_TOPK_BLOCK")
    if override is not None:
        block = int(override)
        if block not in (_DEFAULT_MAX_WIDTH_BLOCK, _LARGE_MAX_WIDTH_BLOCK):
            raise ValueError(
                "B12X_NSA_INDEXER_DYNAMIC_TOPK_BLOCK must be 2048 or 4096, "
                f"got {override}"
            )
        return block
    if gather_k >= 2048:
        return _LARGE_MAX_WIDTH_BLOCK
    return _DEFAULT_MAX_WIDTH_BLOCK


@triton.jit
def _float32_to_ordered_uint32(x):
    bits = x.to(tl.uint32, bitcast=True)
    sign = bits >> 31
    mask = tl.where(sign != 0, 0xFFFFFFFF, 0x80000000).to(tl.uint32)
    return bits ^ mask


@triton.jit
def _sparse_nsa_topk_ids_kernel(
    logits_ptr,
    page_table_1_ptr,
    query_row_to_batch_ptr,
    seqlens_per_query_ptr,
    output_ptr,
    logits_row_stride,
    page_table_row_stride,
    output_row_stride,
    width,
    gather_k,
    WIDTH_BLOCK: tl.constexpr,
    TOPK_BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = tl.arange(0, WIDTH_BLOCK)
    width_mask = offsets < width

    batch_row = tl.load(query_row_to_batch_ptr + pid).to(tl.int32)
    seq_len = tl.load(seqlens_per_query_ptr + pid).to(tl.int32)

    logits = tl.load(
        logits_ptr + pid * logits_row_stride + offsets,
        mask=width_mask,
        other=float("-inf"),
    ).to(tl.float32)
    token_ids = tl.load(
        page_table_1_ptr + batch_row * page_table_row_stride + offsets,
        mask=width_mask,
        other=-1,
    ).to(tl.int32)

    valid = width_mask & (offsets < seq_len) & (token_ids >= 0)
    ordered = _float32_to_ordered_uint32(logits)
    pos_rank = 0xFFFFFFFF - offsets.to(tl.uint32)
    keys = (ordered.to(tl.uint64) << 32) | pos_rank.to(tl.uint64)
    keys = tl.where(valid, keys, 0)

    top_keys = tl.topk(keys, k=TOPK_BLOCK)
    out_offsets = tl.arange(0, TOPK_BLOCK)
    store_mask = out_offsets < gather_k

    selected_pos = (0xFFFFFFFF - (top_keys & 0xFFFFFFFF).to(tl.uint32)).to(tl.int32)
    selected_valid = store_mask & (top_keys != 0) & (selected_pos < width)
    selected_ids = tl.load(
        page_table_1_ptr + batch_row * page_table_row_stride + selected_pos,
        mask=selected_valid,
        other=-1,
    ).to(tl.int32)
    tl.store(
        output_ptr + pid * output_row_stride + out_offsets,
        tl.where(selected_valid, selected_ids, -1),
        mask=store_mask,
    )


@triton.jit
def _sparse_nsa_block_topk_keys_kernel(
    logits_ptr,
    page_table_1_ptr,
    query_row_to_batch_ptr,
    seqlens_per_query_ptr,
    active_width_ptr,
    partial_keys_ptr,
    logits_row_stride,
    page_table_row_stride,
    partial_row_stride,
    partial_block_stride,
    width_capacity,
    gather_k,
    BLOCK_WIDTH: tl.constexpr,
    TOPK_BLOCK: tl.constexpr,
):
    row_idx = tl.program_id(0)
    block_idx = tl.program_id(1)
    offsets = tl.arange(0, BLOCK_WIDTH)
    block_base = block_idx * BLOCK_WIDTH
    token_pos = block_base + offsets

    active_width = tl.load(active_width_ptr).to(tl.int32)
    batch_row = tl.load(query_row_to_batch_ptr + row_idx).to(tl.int32)
    seq_len = tl.load(seqlens_per_query_ptr + row_idx).to(tl.int32)

    active_width = tl.minimum(active_width, width_capacity)
    out_offsets = tl.arange(0, TOPK_BLOCK)
    store_mask = out_offsets < gather_k
    if seq_len <= gather_k:
        tl.store(
            partial_keys_ptr
            + row_idx * partial_row_stride
            + block_idx * partial_block_stride
            + out_offsets,
            0,
            mask=store_mask,
        )
        return
    if block_base >= active_width:
        tl.store(
            partial_keys_ptr
            + row_idx * partial_row_stride
            + block_idx * partial_block_stride
            + out_offsets,
            0,
            mask=store_mask,
        )
        return
    width_mask = token_pos < width_capacity
    logits = tl.load(
        logits_ptr + row_idx * logits_row_stride + token_pos,
        mask=width_mask,
        other=float("-inf"),
    ).to(tl.float32)
    token_ids = tl.load(
        page_table_1_ptr + batch_row * page_table_row_stride + token_pos,
        mask=width_mask,
        other=-1,
    ).to(tl.int32)

    valid = width_mask & (token_pos < active_width) & (token_pos < seq_len) & (token_ids >= 0)
    ordered = _float32_to_ordered_uint32(logits)
    pos_rank = 0xFFFFFFFF - token_pos.to(tl.uint32)
    keys = (ordered.to(tl.uint64) << 32) | pos_rank.to(tl.uint64)
    keys = tl.where(valid, keys, 0)

    top_keys = tl.topk(keys, k=TOPK_BLOCK)
    tl.store(
        partial_keys_ptr
        + row_idx * partial_row_stride
        + block_idx * partial_block_stride
        + out_offsets,
        tl.where(store_mask, top_keys, 0),
        mask=store_mask,
    )


@triton.jit
def _sparse_nsa_merge_topk_keys_kernel(
    src_keys_ptr,
    dst_keys_ptr,
    seqlens_per_query_ptr,
    active_width_ptr,
    src_row_stride,
    src_block_stride,
    dst_row_stride,
    dst_block_stride,
    width_capacity,
    gather_k,
    src_blocks,
    block_span,
    BLOCK_WIDTH: tl.constexpr,
    TOPK_BLOCK: tl.constexpr,
    MERGE_WIDTH_BLOCK: tl.constexpr,
):
    row_idx = tl.program_id(0)
    pair_idx = tl.program_id(1)

    active_width = tl.load(active_width_ptr).to(tl.int32)
    active_width = tl.minimum(active_width, width_capacity)
    seq_len = tl.load(seqlens_per_query_ptr + row_idx).to(tl.int32)
    if seq_len <= gather_k:
        return
    active_blocks = (active_width + BLOCK_WIDTH - 1) // BLOCK_WIDTH
    active_src_blocks = (active_blocks + block_span - 1) // block_span
    pair_base = pair_idx * 2
    out_offsets = tl.arange(0, TOPK_BLOCK)
    store_mask = out_offsets < gather_k
    if pair_base >= active_src_blocks:
        tl.store(
            dst_keys_ptr + row_idx * dst_row_stride + pair_idx * dst_block_stride + out_offsets,
            0,
            mask=store_mask,
        )
        return
    if pair_base + 1 >= active_src_blocks:
        copied_keys = tl.load(
            src_keys_ptr + row_idx * src_row_stride + pair_base * src_block_stride + out_offsets,
            mask=store_mask,
            other=0,
        ).to(tl.uint64)
        tl.store(
            dst_keys_ptr + row_idx * dst_row_stride + pair_idx * dst_block_stride + out_offsets,
            copied_keys,
            mask=store_mask,
        )
        return

    merge_offsets = tl.arange(0, MERGE_WIDTH_BLOCK)
    from_left = merge_offsets < TOPK_BLOCK
    src_block = tl.where(from_left, pair_base, pair_base + 1)
    src_offset = tl.where(from_left, merge_offsets, merge_offsets - TOPK_BLOCK)
    src_active = (src_block < src_blocks) & (src_block < active_blocks) & (src_offset < gather_k)

    merged_keys = tl.load(
        src_keys_ptr + row_idx * src_row_stride + src_block * src_block_stride + src_offset,
        mask=src_active,
        other=0,
    ).to(tl.uint64)
    top_keys = tl.topk(merged_keys, k=TOPK_BLOCK)

    tl.store(
        dst_keys_ptr
        + row_idx * dst_row_stride
        + pair_idx * dst_block_stride
        + out_offsets,
        tl.where(store_mask, top_keys, 0),
        mask=store_mask,
    )


@triton.jit
def _sparse_nsa_ids_from_keys_kernel(
    keys_ptr,
    page_table_1_ptr,
    query_row_to_batch_ptr,
    seqlens_per_query_ptr,
    output_ptr,
    keys_row_stride,
    page_table_row_stride,
    output_row_stride,
    width_capacity,
    gather_k,
    TOPK_BLOCK: tl.constexpr,
):
    row_idx = tl.program_id(0)
    offsets = tl.arange(0, TOPK_BLOCK)
    store_mask = offsets < gather_k

    keys = tl.load(
        keys_ptr + row_idx * keys_row_stride + offsets,
        mask=store_mask,
        other=0,
    ).to(tl.uint64)
    batch_row = tl.load(query_row_to_batch_ptr + row_idx).to(tl.int32)
    seq_len = tl.load(seqlens_per_query_ptr + row_idx).to(tl.int32)
    if seq_len <= gather_k:
        prefix_valid = store_mask & (offsets < seq_len)
        prefix_ids = tl.load(
            page_table_1_ptr + batch_row * page_table_row_stride + offsets,
            mask=prefix_valid,
            other=-1,
        ).to(tl.int32)
        tl.store(
            output_ptr + row_idx * output_row_stride + offsets,
            tl.where(prefix_valid, prefix_ids, -1),
            mask=store_mask,
        )
        return
    selected_pos = (0xFFFFFFFF - (keys & 0xFFFFFFFF).to(tl.uint32)).to(tl.int32)
    valid = store_mask & (keys != 0) & (selected_pos < width_capacity)
    selected_ids = tl.load(
        page_table_1_ptr + batch_row * page_table_row_stride + selected_pos,
        mask=valid,
        other=-1,
    ).to(tl.int32)
    tl.store(
        output_ptr + row_idx * output_row_stride + offsets,
        tl.where(valid, selected_ids, -1),
        mask=store_mask,
    )


@lru_cache(maxsize=32)
def _cached_partial_key_workspace(
    rows: int,
    num_blocks: int,
    topk_block: int,
    device_type: str,
    device_index: int | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    device = torch.device(device_type, device_index)
    shape = (rows, num_blocks, topk_block)
    return (
        torch.empty(shape, dtype=torch.uint64, device=device),
        torch.empty(shape, dtype=torch.uint64, device=device),
    )


def clear_sparse_nsa_topk_kernel_cache() -> None:
    _cached_partial_key_workspace.cache_clear()


def supports_sparse_nsa_topk_kernel(
    *,
    logits: torch.Tensor,
    page_table_1: torch.Tensor,
    query_row_to_batch: torch.Tensor,
    seqlens_per_query: torch.Tensor,
    gather_k: int,
) -> bool:
    if os.environ.get("B12X_NSA_INDEXER_FORCE_TORCH_TOPK", "0") == "1":
        return False
    if logits.device.type != "cuda":
        return False
    if not (
        logits.device == page_table_1.device == query_row_to_batch.device == seqlens_per_query.device
    ):
        return False
    if logits.ndim != 2 or page_table_1.ndim != 2:
        return False
    if query_row_to_batch.ndim != 1 or seqlens_per_query.ndim != 1:
        return False
    if logits.shape[0] != query_row_to_batch.shape[0] or logits.shape[0] != seqlens_per_query.shape[0]:
        return False
    if logits.shape[1] != page_table_1.shape[1]:
        return False
    if logits.dtype != torch.float32:
        return False
    if page_table_1.dtype != torch.int32:
        return False
    if query_row_to_batch.dtype != torch.int32 or seqlens_per_query.dtype != torch.int32:
        return False
    if gather_k <= 0:
        return False
    width_block = triton.next_power_of_2(int(logits.shape[1]))
    topk_block = triton.next_power_of_2(int(gather_k))
    if width_block > _DEFAULT_MAX_WIDTH_BLOCK or topk_block > _MAX_TOPK_BLOCK:
        return False
    if topk_block > width_block:
        return False
    return True


def run_sparse_nsa_topk_kernel(
    *,
    logits: torch.Tensor,
    page_table_1: torch.Tensor,
    query_row_to_batch: torch.Tensor,
    seqlens_per_query: torch.Tensor,
    output: torch.Tensor,
    gather_k: int,
) -> None:
    if not supports_sparse_nsa_topk_kernel(
        logits=logits,
        page_table_1=page_table_1,
        query_row_to_batch=query_row_to_batch,
        seqlens_per_query=seqlens_per_query,
        gather_k=gather_k,
    ):
        raise ValueError("sparse NSA Triton top-k kernel does not support this configuration")
    if output.ndim != 2 or output.shape[0] != logits.shape[0] or output.shape[1] < gather_k:
        raise ValueError(
            "output must have shape [rows, >= gather_k], got "
            f"{tuple(output.shape)} for rows={logits.shape[0]} gather_k={gather_k}"
        )
    if output.dtype != torch.int32 or output.device != logits.device:
        raise ValueError("output must be an int32 CUDA tensor on the same device as logits")
    if output.stride(-1) != 1:
        raise ValueError("output must be contiguous in the top-k dimension")

    rows, width = logits.shape
    if rows == 0 or width == 0 or gather_k == 0:
        return

    width_block = triton.next_power_of_2(int(width))
    topk_block = triton.next_power_of_2(int(gather_k))
    num_warps = 4 if width_block <= 512 else 8

    _sparse_nsa_topk_ids_kernel[(rows,)](
        logits,
        page_table_1,
        query_row_to_batch,
        seqlens_per_query,
        output,
        logits.stride(0),
        page_table_1.stride(0),
        output.stride(0),
        width,
        gather_k,
        WIDTH_BLOCK=width_block,
        TOPK_BLOCK=topk_block,
        num_warps=num_warps,
    )


def supports_sparse_nsa_dynamic_topk_kernel(
    *,
    logits: torch.Tensor,
    page_table_1: torch.Tensor,
    query_row_to_batch: torch.Tensor,
    seqlens_per_query: torch.Tensor,
    active_width: torch.Tensor,
    gather_k: int,
) -> bool:
    if os.environ.get("B12X_NSA_INDEXER_FORCE_TORCH_TOPK", "0") == "1":
        return False
    if logits.device.type != "cuda":
        return False
    if not (
        logits.device == page_table_1.device == query_row_to_batch.device == seqlens_per_query.device
    ):
        return False
    if logits.ndim != 2 or page_table_1.ndim != 2:
        return False
    if query_row_to_batch.ndim != 1 or seqlens_per_query.ndim != 1:
        return False
    if logits.shape[0] != query_row_to_batch.shape[0] or logits.shape[0] != seqlens_per_query.shape[0]:
        return False
    if logits.shape[1] != page_table_1.shape[1]:
        return False
    if logits.dtype != torch.float32 or page_table_1.dtype != torch.int32:
        return False
    if query_row_to_batch.dtype != torch.int32 or seqlens_per_query.dtype != torch.int32:
        return False
    if gather_k <= 0:
        return False
    if active_width.shape != (1,):
        return False
    if active_width.dtype != torch.int32 or active_width.device != logits.device:
        return False
    topk_block = triton.next_power_of_2(int(gather_k))
    width_block = _dynamic_width_block(int(gather_k))
    return (
        topk_block <= _MAX_TOPK_BLOCK
        and width_block <= _MAX_MERGE_WIDTH_BLOCK
        and (2 * topk_block) <= _MAX_MERGE_WIDTH_BLOCK
    )

def run_sparse_nsa_dynamic_topk_kernel(
    *,
    logits: torch.Tensor,
    page_table_1: torch.Tensor,
    query_row_to_batch: torch.Tensor,
    seqlens_per_query: torch.Tensor,
    active_width: torch.Tensor,
    output: torch.Tensor,
    gather_k: int,
) -> None:
    if not supports_sparse_nsa_dynamic_topk_kernel(
        logits=logits,
        page_table_1=page_table_1,
        query_row_to_batch=query_row_to_batch,
        seqlens_per_query=seqlens_per_query,
        active_width=active_width,
        gather_k=gather_k,
    ):
        raise ValueError("sparse NSA dynamic Triton top-k kernel does not support this configuration")
    if output.ndim != 2 or output.shape[0] != logits.shape[0] or output.shape[1] < gather_k:
        raise ValueError(
            "output must have shape [rows, >= gather_k], got "
            f"{tuple(output.shape)} for rows={logits.shape[0]} gather_k={gather_k}"
        )
    if output.dtype != torch.int32 or output.device != logits.device:
        raise ValueError("output must be an int32 CUDA tensor on the same device as logits")
    if output.stride(-1) != 1:
        raise ValueError("output must be contiguous in the top-k dimension")

    rows, width = logits.shape
    if rows == 0 or width == 0 or gather_k == 0:
        return

    topk_block = triton.next_power_of_2(int(gather_k))
    width_block = _dynamic_width_block(int(gather_k))
    num_blocks = triton.cdiv(int(width), width_block)
    partial_keys_a, partial_keys_b = _cached_partial_key_workspace(
        rows,
        num_blocks,
        topk_block,
        logits.device.type,
        logits.device.index,
    )

    _sparse_nsa_block_topk_keys_kernel[(rows, num_blocks)](
        logits,
        page_table_1,
        query_row_to_batch,
        seqlens_per_query,
        active_width,
        partial_keys_a,
        logits.stride(0),
        page_table_1.stride(0),
        partial_keys_a.stride(0),
        partial_keys_a.stride(1),
        width,
        gather_k,
        BLOCK_WIDTH=width_block,
        TOPK_BLOCK=topk_block,
        num_warps=8,
    )

    src = partial_keys_a
    dst = partial_keys_b
    src_blocks = num_blocks
    block_span = 1
    while src_blocks > 1:
        blocks_per_round = (src_blocks + 1) // 2
        _sparse_nsa_merge_topk_keys_kernel[(rows, blocks_per_round)](
            src,
            dst,
            seqlens_per_query,
            active_width,
            src.stride(0),
            src.stride(1),
            dst.stride(0),
            dst.stride(1),
            width,
            gather_k,
            src_blocks,
            block_span,
            BLOCK_WIDTH=width_block,
            TOPK_BLOCK=topk_block,
            MERGE_WIDTH_BLOCK=2 * topk_block,
            num_warps=8,
        )
        src, dst = dst, src
        src_blocks = blocks_per_round
        block_span *= 2

    _sparse_nsa_ids_from_keys_kernel[(rows,)](
        src,
        page_table_1,
        query_row_to_batch,
        seqlens_per_query,
        output,
        src.stride(0),
        page_table_1.stride(0),
        output.stride(0),
        width,
        gather_k,
        TOPK_BLOCK=topk_block,
        num_warps=8,
    )
