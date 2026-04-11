"""Fused decode-only NSA score+select path.

This path keeps the graph-safe trivial-row prefix fast path and fuses the
non-trivial decode path into:

1. CuTeDSL score generation directly into compact per-CTA packed-key leaves.
2. Triton pairwise merge of those leaves into a single top-k list.
3. Triton materialization of final token ids from packed positions.

Unlike the unfused path, this never materializes a full `[rows, width]` logits
buffer or runs a separate block-topk pass over it.
"""

from __future__ import annotations

from functools import lru_cache
import os

import cutlass
import cutlass.cute as cute
import torch
import triton
import triton.language as tl
from cutlass import Float32, Int32, Uint32, Uint64
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op

from b12x.attention import utils as attention_utils
from b12x.cute.utils import current_cuda_stream

from .kernel import (
    _INDEX_HEAD_DIM,
    _MAX_Q_HEADS,
    _PAGE_SIZE,
    _SCORE_CTAS_PER_ROW,
    _THREADS_PER_CTA,
    _TOKENS_PER_CTA,
    _WARP_THREADS,
    _load_fp8x4,
    _load_fp8x4_2d,
    _run_cached_host_launcher,
    _split_index_k_cache_runtime_views,
    _tensor_meta_key,
    _to_kernel_tensor,
    _warp_allreduce_sum,
    supports_sparse_nsa_indexer_kernel,
)


_MAX_TOPK_BLOCK = 2048
_MIN_SCORE_CTAS = 4


def _next_power_of_two(value: int) -> int:
    if value <= 0:
        raise ValueError(f"value must be positive, got {value}")
    return 1 << (value - 1).bit_length()


def _ceil_div(x: int, y: int) -> int:
    return (x + y - 1) // y


def _score_leaf_width(width: int, score_ctas: int) -> int:
    steps = _ceil_div(int(width), int(score_ctas) * _TOKENS_PER_CTA)
    return _next_power_of_two(max(steps * _TOKENS_PER_CTA, _TOKENS_PER_CTA))


def _select_score_ctas(width: int, topk_block: int) -> int:
    override = os.environ.get("B12X_NSA_INDEXER_FUSED_SCORE_CTAS")
    if override is not None:
        score_ctas = int(override)
        if (
            score_ctas < _MIN_SCORE_CTAS
            or score_ctas > _SCORE_CTAS_PER_ROW
            or (score_ctas & (score_ctas - 1)) != 0
        ):
            raise ValueError(
                "B12X_NSA_INDEXER_FUSED_SCORE_CTAS must be a positive power of two "
                f"between {_MIN_SCORE_CTAS} and {_SCORE_CTAS_PER_ROW}, got {override}"
            )
        return score_ctas
    for score_ctas in (32, 16, 8, 4):
        if score_ctas < _MIN_SCORE_CTAS:
            continue
        if _score_leaf_width(width, score_ctas) <= topk_block:
            return score_ctas
    return _SCORE_CTAS_PER_ROW


@dsl_user_op
def _bitcast_f32_to_u32(value: Float32, *, loc=None, ip=None) -> Uint32:
    return Uint32(
        llvm.inline_asm(
            T.i32(),
            [Float32(value).ir_value(loc=loc, ip=ip)],
            "mov.b32 $0, $1;",
            "=r,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@cute.jit
def _float32_to_ordered_uint32(value: Float32) -> Uint32:
    bits = _bitcast_f32_to_u32(value)
    sign = bits >> Uint32(31)
    mask = cutlass.select_(sign != Uint32(0), Uint32(0xFFFFFFFF), Uint32(0x80000000))
    return bits ^ mask


@cute.jit
def _pack_score_key(score: Float32, token_pos: Int32) -> Uint64:
    ordered = _float32_to_ordered_uint32(score)
    pos_rank = Uint32(0xFFFFFFFF) - Uint32(token_pos)
    return (Uint64(ordered) << Uint64(32)) | Uint64(pos_rank)


class SparseNSAFusedLeafKeysKernel:
    """Reuse the strong score launch geometry and emit compact per-CTA key leaves."""

    def __init__(self, *, leaf_width: int, score_ctas: int):
        self.leaf_width = int(leaf_width)
        self.score_ctas = int(score_ctas)

    @cute.jit
    def __call__(
        self,
        q_bytes: cute.Tensor,
        weights: cute.Tensor,
        k_quant_bytes: cute.Tensor,
        k_scales: cute.Tensor,
        page_table_1: cute.Tensor,
        query_row_to_batch: cute.Tensor,
        seqlens_per_query: cute.Tensor,
        active_width: cute.Tensor,
        gather_k: cute.Tensor,
        partial_keys: cute.Tensor,
        stream,
    ):
        self.kernel(
            q_bytes,
            weights,
            k_quant_bytes,
            k_scales,
            page_table_1,
            query_row_to_batch,
            seqlens_per_query,
            active_width,
            gather_k,
            partial_keys,
        ).launch(
            grid=(q_bytes.shape[0], self.score_ctas, 1),
            block=[_THREADS_PER_CTA, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        q_bytes: cute.Tensor,
        weights: cute.Tensor,
        k_quant_bytes: cute.Tensor,
        k_scales: cute.Tensor,
        page_table_1: cute.Tensor,
        query_row_to_batch: cute.Tensor,
        seqlens_per_query: cute.Tensor,
        active_width: cute.Tensor,
        gather_k: cute.Tensor,
        partial_keys: cute.Tensor,
    ):
        tx, _, _ = cute.arch.thread_idx()
        q_idx, cta_idx, _ = cute.arch.block_idx()
        q_idx = Int32(q_idx)
        cta_idx = Int32(cta_idx)
        lane = tx % Int32(_WARP_THREADS)
        warp_idx = tx // Int32(_WARP_THREADS)

        smem = cutlass.utils.SmemAllocator()

        @cute.struct
        class SharedStorage:
            qBytes: cute.struct.Align[
                cute.struct.MemRange[cutlass.Uint8, _MAX_Q_HEADS * _INDEX_HEAD_DIM],
                16,
            ]
            weights: cute.struct.Align[
                cute.struct.MemRange[cutlass.Float32, _MAX_Q_HEADS],
                16,
            ]

        storage = smem.allocate(SharedStorage)
        s_q = storage.qBytes.get_tensor(
            cute.make_layout((_MAX_Q_HEADS, _INDEX_HEAD_DIM), stride=(_INDEX_HEAD_DIM, 1))
        )
        s_w = storage.weights.get_tensor(cute.make_layout((_MAX_Q_HEADS,), stride=(1,)))

        num_heads = Int32(q_bytes.shape[1])
        q_linear = tx
        total_q_bytes = num_heads * Int32(_INDEX_HEAD_DIM)
        while q_linear < total_q_bytes:
            head_idx = q_linear // Int32(_INDEX_HEAD_DIM)
            col_idx = q_linear - head_idx * Int32(_INDEX_HEAD_DIM)
            s_q[head_idx, col_idx] = q_bytes[q_idx, head_idx, col_idx]
            q_linear += Int32(_THREADS_PER_CTA)

        w_linear = tx
        while w_linear < num_heads:
            s_w[w_linear] = Float32(weights[q_idx, w_linear])
            w_linear += Int32(_THREADS_PER_CTA)
        cute.arch.sync_threads()

        width_capacity = Int32(page_table_1.shape[1])
        live_width = Int32(active_width[Int32(0)])
        if live_width > width_capacity:
            live_width = width_capacity
        seq_len = Int32(seqlens_per_query[q_idx])
        if seq_len < Int32(0):
            seq_len = Int32(0)
        if seq_len > live_width:
            seq_len = live_width
        gather_k_i32 = Int32(gather_k[Int32(0)])
        active_leaf_blocks = (seq_len + Int32(_TOKENS_PER_CTA) - Int32(1)) // Int32(_TOKENS_PER_CTA)
        if active_leaf_blocks > Int32(self.score_ctas):
            active_leaf_blocks = Int32(self.score_ctas)
        if seq_len > gather_k_i32 and cta_idx < active_leaf_blocks:
            init_linear = tx
            while init_linear < Int32(self.leaf_width):
                partial_keys[q_idx, cta_idx, init_linear] = Uint64(0)
                init_linear += Int32(_THREADS_PER_CTA)
            cute.arch.sync_threads()

            batch_row = Int32(query_row_to_batch[q_idx])
            token_pos = cta_idx * Int32(_TOKENS_PER_CTA) + warp_idx
            token_stride = Int32(self.score_ctas * _TOKENS_PER_CTA)
            local_slot = warp_idx
            leaf_width_i32 = Int32(self.leaf_width)

            while local_slot < leaf_width_i32:
                if token_pos < seq_len:
                    token_id = Int32(page_table_1[batch_row, token_pos])
                    if token_id >= Int32(0):
                        page_idx = token_id // Int32(_PAGE_SIZE)
                        slot_idx = token_id - page_idx * Int32(_PAGE_SIZE)
                        base = lane * Int32(4)

                        logit = Float32(0.0)
                        head_idx = Int32(0)
                        while head_idx < num_heads:
                            q0, q1, q2, q3 = _load_fp8x4_2d(s_q, head_idx, base)
                            k0, k1, k2, k3 = _load_fp8x4(k_quant_bytes, page_idx, slot_idx, base)
                            dot = Float32(q0 * k0 + q1 * k1 + q2 * k2 + q3 * k3)
                            dot = _warp_allreduce_sum(dot)
                            if lane == Int32(0):
                                logit = Float32(
                                    logit + attention_utils.fmax(dot, Float32(0.0)) * s_w[head_idx]
                                )
                            head_idx += Int32(1)

                        if lane == Int32(0):
                            score = Float32(logit * Float32(k_scales[page_idx, slot_idx]))
                            partial_keys[q_idx, cta_idx, local_slot] = _pack_score_key(
                                score, token_pos
                            )
                token_pos += token_stride
                local_slot += Int32(_TOKENS_PER_CTA)


@triton.jit
def _merge_uint64_topk_keys_kernel(
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
    src_blocks_capacity,
    block_span,
    SRC_WIDTH_BLOCK: tl.constexpr,
    DST_WIDTH_BLOCK: tl.constexpr,
    MERGE_WIDTH_BLOCK: tl.constexpr,
):
    row_idx = tl.program_id(0)
    pair_idx = tl.program_id(1)

    active_width = tl.load(active_width_ptr).to(tl.int32)
    active_width = tl.minimum(active_width, width_capacity)
    seq_len = tl.load(seqlens_per_query_ptr + row_idx).to(tl.int32)
    seq_len = tl.minimum(seq_len, active_width)
    out_offsets = tl.arange(0, DST_WIDTH_BLOCK)
    store_mask = out_offsets < gather_k
    if seq_len <= gather_k:
        return

    active_leaf_blocks = (seq_len + 4 - 1) // 4
    active_leaf_blocks = tl.minimum(active_leaf_blocks, src_blocks_capacity * block_span)
    active_src_blocks = (active_leaf_blocks + block_span - 1) // block_span
    pair_base = pair_idx * 2
    if pair_base >= active_src_blocks:
        tl.store(
            dst_keys_ptr + row_idx * dst_row_stride + pair_idx * dst_block_stride + out_offsets,
            0,
            mask=store_mask,
        )
        return

    if pair_base + 1 >= active_src_blocks:
        copy_valid = out_offsets < SRC_WIDTH_BLOCK
        copied_keys = tl.load(
            src_keys_ptr + row_idx * src_row_stride + pair_base * src_block_stride + out_offsets,
            mask=copy_valid,
            other=0,
        ).to(tl.uint64)
        tl.store(
            dst_keys_ptr + row_idx * dst_row_stride + pair_idx * dst_block_stride + out_offsets,
            tl.where(copy_valid, copied_keys, 0),
            mask=store_mask,
        )
        return

    merge_offsets = tl.arange(0, MERGE_WIDTH_BLOCK)
    from_left = merge_offsets < SRC_WIDTH_BLOCK
    src_block = tl.where(from_left, pair_base, pair_base + 1)
    src_offset = tl.where(from_left, merge_offsets, merge_offsets - SRC_WIDTH_BLOCK)
    merged_keys = tl.load(
        src_keys_ptr + row_idx * src_row_stride + src_block * src_block_stride + src_offset,
        mask=src_offset < SRC_WIDTH_BLOCK,
        other=0,
    ).to(tl.uint64)
    top_keys = tl.topk(merged_keys, k=DST_WIDTH_BLOCK)
    tl.store(
        dst_keys_ptr + row_idx * dst_row_stride + pair_idx * dst_block_stride + out_offsets,
        tl.where(store_mask, top_keys, 0),
        mask=store_mask,
    )


@triton.jit
def _ids_from_uint64_keys_kernel(
    keys_ptr,
    page_table_1_ptr,
    query_row_to_batch_ptr,
    seqlens_per_query_ptr,
    active_width_ptr,
    output_ptr,
    keys_row_stride,
    keys_block_stride,
    page_table_row_stride,
    output_row_stride,
    width_capacity,
    gather_k,
    TOPK_BLOCK: tl.constexpr,
):
    row_idx = tl.program_id(0)
    offsets = tl.arange(0, TOPK_BLOCK)
    store_mask = offsets < gather_k

    active_width = tl.load(active_width_ptr).to(tl.int32)
    active_width = tl.minimum(active_width, width_capacity)
    seq_len = tl.load(seqlens_per_query_ptr + row_idx).to(tl.int32)
    seq_len = tl.minimum(seq_len, active_width)
    if seq_len <= gather_k:
        return

    keys = tl.load(
        keys_ptr + row_idx * keys_row_stride + offsets,
        mask=store_mask,
        other=0,
    ).to(tl.uint64)
    batch_row = tl.load(query_row_to_batch_ptr + row_idx).to(tl.int32)
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


@triton.jit
def _fill_trivial_prefix_kernel(
    page_table_1_ptr,
    query_row_to_batch_ptr,
    seqlens_per_query_ptr,
    active_width_ptr,
    output_ptr,
    page_table_row_stride,
    output_row_stride,
    width_capacity,
    gather_k,
    TOPK_BLOCK: tl.constexpr,
):
    row_idx = tl.program_id(0)
    offsets = tl.arange(0, TOPK_BLOCK)
    store_mask = offsets < gather_k

    active_width = tl.load(active_width_ptr).to(tl.int32)
    active_width = tl.minimum(active_width, width_capacity)
    seq_len = tl.load(seqlens_per_query_ptr + row_idx).to(tl.int32)
    seq_len = tl.minimum(seq_len, active_width)
    if seq_len > gather_k:
        return

    batch_row = tl.load(query_row_to_batch_ptr + row_idx).to(tl.int32)
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


@lru_cache(maxsize=32)
def _cached_int32_scalar(
    value: int,
    device_type: str,
    device_index: int | None,
) -> torch.Tensor:
    return torch.tensor([value], dtype=torch.int32, device=torch.device(device_type, device_index))


@lru_cache(maxsize=16)
def _build_sparse_nsa_fused_leaf_kernel(
    *,
    leaf_width: int,
    score_ctas: int,
) -> SparseNSAFusedLeafKeysKernel:
    return SparseNSAFusedLeafKeysKernel(leaf_width=leaf_width, score_ctas=score_ctas)


def clear_sparse_nsa_fused_decode_kernel_cache() -> None:
    _cached_partial_key_workspace.cache_clear()
    _cached_int32_scalar.cache_clear()
    _build_sparse_nsa_fused_leaf_kernel.cache_clear()


def supports_sparse_nsa_fused_decode_kernel(
    *,
    q_fp8: torch.Tensor,
    weights: torch.Tensor,
    index_k_cache: torch.Tensor,
    page_table_1: torch.Tensor,
    query_row_to_batch: torch.Tensor,
    seqlens_per_query: torch.Tensor,
    active_width: torch.Tensor,
    gather_k: int,
    page_size: int,
) -> bool:
    if os.environ.get("B12X_NSA_INDEXER_ENABLE_FUSED_DECODE", "0") != "1":
        return False
    if os.environ.get("B12X_NSA_INDEXER_FORCE_REFERENCE", "0") == "1":
        return False
    if not supports_sparse_nsa_indexer_kernel(
        q_fp8=q_fp8,
        weights=weights,
        index_k_cache=index_k_cache,
        page_table_1=page_table_1,
        query_row_to_batch=query_row_to_batch,
        seqlens_per_query=seqlens_per_query,
        page_size=page_size,
    ):
        return False
    if gather_k <= 0:
        return False
    if active_width.shape != (1,):
        return False
    if active_width.dtype != torch.int32 or active_width.device != q_fp8.device:
        return False
    topk_block = _next_power_of_two(gather_k)
    if topk_block > _MAX_TOPK_BLOCK:
        return False
    score_ctas = _select_score_ctas(page_table_1.shape[1], topk_block)
    return _score_leaf_width(page_table_1.shape[1], score_ctas) <= topk_block


def run_sparse_nsa_fused_decode_kernel(
    *,
    q_fp8: torch.Tensor,
    weights: torch.Tensor,
    index_k_cache: torch.Tensor,
    page_table_1: torch.Tensor,
    query_row_to_batch: torch.Tensor,
    seqlens_per_query: torch.Tensor,
    active_width: torch.Tensor,
    output: torch.Tensor,
    page_size: int = _PAGE_SIZE,
) -> None:
    gather_k = int(output.shape[1])
    if not supports_sparse_nsa_fused_decode_kernel(
        q_fp8=q_fp8,
        weights=weights,
        index_k_cache=index_k_cache,
        page_table_1=page_table_1,
        query_row_to_batch=query_row_to_batch,
        seqlens_per_query=seqlens_per_query,
        active_width=active_width,
        gather_k=gather_k,
        page_size=page_size,
    ):
        raise ValueError("fused sparse NSA decode kernel does not support this configuration")
    if output.ndim != 2 or output.shape[0] != q_fp8.shape[0]:
        raise ValueError(
            f"output must have shape ({q_fp8.shape[0]}, {gather_k}), got {tuple(output.shape)}"
        )
    if output.dtype != torch.int32 or output.device != q_fp8.device:
        raise ValueError("output must be int32 on the same device as q_fp8")
    if output.stride(-1) != 1:
        raise ValueError("output must be contiguous in the top-k dimension")

    rows = q_fp8.shape[0]
    width = page_table_1.shape[1]
    if rows == 0 or width == 0 or gather_k == 0:
        return

    topk_block = _next_power_of_two(gather_k)
    score_ctas = _select_score_ctas(width, topk_block)
    leaf_width = _score_leaf_width(width, score_ctas)
    q_bytes = q_fp8.contiguous().view(torch.uint8)
    weights_t = weights.contiguous()
    page_table_t = page_table_1.contiguous()
    query_rows_t = query_row_to_batch.contiguous()
    seqlens_t = seqlens_per_query.contiguous()
    active_width_t = active_width.contiguous()
    gather_k_t = _cached_int32_scalar(gather_k, q_fp8.device.type, q_fp8.device.index)
    k_quant_bytes, k_scales = _split_index_k_cache_runtime_views(index_k_cache)
    partial_a, partial_b = _cached_partial_key_workspace(
        rows,
        score_ctas,
        topk_block,
        q_fp8.device.type,
        q_fp8.device.index,
    )

    leaf_kernel = _build_sparse_nsa_fused_leaf_kernel(
        leaf_width=leaf_width,
        score_ctas=score_ctas,
    )
    leaf_args = (
        _to_kernel_tensor(q_bytes, cutlass.Uint8),
        _to_kernel_tensor(weights_t, cutlass.Float32, assumed_align=4),
        _to_kernel_tensor(k_quant_bytes, cutlass.Uint8),
        _to_kernel_tensor(k_scales, cutlass.Float32, assumed_align=4),
        _to_kernel_tensor(page_table_t, cutlass.Int32, assumed_align=4),
        _to_kernel_tensor(query_rows_t, cutlass.Int32, assumed_align=4),
        _to_kernel_tensor(seqlens_t, cutlass.Int32, assumed_align=4),
        _to_kernel_tensor(active_width_t, cutlass.Int32, assumed_align=4),
        _to_kernel_tensor(gather_k_t, cutlass.Int32, assumed_align=4),
        _to_kernel_tensor(partial_a, cutlass.Uint64, assumed_align=8),
        current_cuda_stream(),
    )
    leaf_cache_key = (
        leaf_width,
        score_ctas,
        _tensor_meta_key(q_bytes),
        _tensor_meta_key(weights_t),
        _tensor_meta_key(k_quant_bytes),
        _tensor_meta_key(k_scales),
        _tensor_meta_key(page_table_t),
        _tensor_meta_key(query_rows_t),
        _tensor_meta_key(seqlens_t),
        _tensor_meta_key(active_width_t),
        _tensor_meta_key(gather_k_t),
        _tensor_meta_key(partial_a),
    )
    _run_cached_host_launcher(leaf_kernel, leaf_cache_key, leaf_args)

    src = partial_a
    dst = partial_b
    src_width = leaf_width
    src_blocks = score_ctas
    block_span = 1
    while src_blocks > 1 or src_width < topk_block:
        blocks_per_round = (src_blocks + 1) // 2
        dst_width = min(topk_block, src_width * 2)
        _merge_uint64_topk_keys_kernel[(rows, blocks_per_round)](
            src,
            dst,
            seqlens_t,
            active_width_t,
            src.stride(0),
            src.stride(1),
            dst.stride(0),
            dst.stride(1),
            width,
            gather_k,
            score_ctas,
            block_span,
            SRC_WIDTH_BLOCK=src_width,
            DST_WIDTH_BLOCK=dst_width,
            MERGE_WIDTH_BLOCK=2 * src_width,
            num_warps=8,
        )
        src, dst = dst, src
        src_blocks = blocks_per_round
        src_width = dst_width
        block_span *= 2

    _ids_from_uint64_keys_kernel[(rows,)](
        src,
        page_table_t,
        query_rows_t,
        seqlens_t,
        active_width_t,
        output,
        src.stride(0),
        src.stride(1),
        page_table_t.stride(0),
        output.stride(0),
        width,
        gather_k,
        TOPK_BLOCK=topk_block,
        num_warps=8,
    )
    _fill_trivial_prefix_kernel[(rows,)](
        page_table_t,
        query_rows_t,
        seqlens_t,
        active_width_t,
        output,
        page_table_t.stride(0),
        output.stride(0),
        width,
        gather_k,
        TOPK_BLOCK=topk_block,
        num_warps=4,
    )
