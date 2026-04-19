"""CuTeDSL extend logits kernel for the DeepGEMM-style non-paged NSA contract."""

from __future__ import annotations

from collections import OrderedDict
from functools import lru_cache
import os
import warnings

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass import Float32, Int32, Uint32
from cutlass._mlir.dialects import llvm
from cutlass.cute.runtime import from_dlpack
from cutlass.cutlass_dsl import Int64, dsl_user_op

from b12x.attention import pipeline
from b12x.attention import utils as attention_utils
from b12x.cute.fp4 import (
    frag_layout_swizzle_16b_to_8b,
    ld_shared_v4_u32,
    ldmatrix_m8n8x4_b16,
    ldmatrix_m8n8x4_left_half_b16,
    ldmatrix_m8n8x4_right_half_b16,
    mxfp8_mma_m16n8k32_f32_e4m3,
    shared_ptr_to_u32,
    st_shared_v4_u32,
)
from b12x.runtime_control import raise_if_kernel_resolution_frozen
from b12x.cute.utils import current_cuda_stream


_INDEX_HEAD_DIM = 128
_FP8_ROW_U32 = _INDEX_HEAD_DIM // 4
_FP8_ROW_VECS = _INDEX_HEAD_DIM // 16
_BLOCK_Q = 32
_BLOCK_K = 64
_WARP_THREADS = 32
_WARPS_Q = _BLOCK_Q // 16
_WARPS_K = _BLOCK_K // 16
_WARPS_PER_CTA = _WARPS_Q * _WARPS_K
_THREADS_PER_CTA = _WARPS_PER_CTA * _WARP_THREADS
_MAX_Q_HEADS = 64
_EAGER_HOST_LAUNCHER_CACHE_SIZE = 32
_EXTEND_TMA_DESC_CACHE_SIZE = 32


def _to_kernel_tensor(
    tensor: torch.Tensor,
    dtype: type[cutlass.Numeric],
    *,
    assumed_align: int = 16,
) -> cutlass.cute.Tensor:
    cute_tensor = from_dlpack(tensor, assumed_align=assumed_align)
    cute_tensor.element_type = dtype
    leading_dim = next((idx for idx, stride in enumerate(tensor.stride()) if stride == 1), None)
    if leading_dim is not None and tensor.ndim >= 2:
        cute_tensor = cute_tensor.mark_layout_dynamic(leading_dim=leading_dim)
    return cute_tensor


def _tensor_meta_key(
    tensor: torch.Tensor,
) -> tuple[tuple[int, ...], tuple[int, ...], str, tuple[str, int | None]]:
    return (
        tuple(tensor.shape),
        tuple(tensor.stride()),
        str(tensor.dtype),
        (tensor.device.type, tensor.device.index),
    )


def _launcher_cache_lookup(
    kernel: object,
    cache_key: tuple[object, ...],
):
    cache = getattr(kernel, "_eager_host_launchers", None)
    if cache is None:
        cache = OrderedDict()
        setattr(kernel, "_eager_host_launchers", cache)
        return cache, None
    compiled = cache.get(cache_key)
    if compiled is not None:
        cache.move_to_end(cache_key)
    return cache, compiled


def _run_cached_host_launcher(
    kernel: object,
    cache_key: tuple[object, ...],
    args: tuple[object, ...],
) -> None:
    cache, compiled = _launcher_cache_lookup(kernel, cache_key)
    if compiled is None:
        raise_if_kernel_resolution_frozen(
            "eager host launcher compile",
            target=kernel,
            cache_key=cache_key,
        )
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="Cache is disabled as user wants to compile only.",
                category=UserWarning,
            )
            compiled = kernel(*args, compile_only=True)
        cache[cache_key] = compiled
        if len(cache) > _EAGER_HOST_LAUNCHER_CACHE_SIZE:
            cache.popitem(last=False)
    exe_args, _ = compiled.generate_execution_args(*args)
    compiled.run_compiled_program(exe_args)


def _pad_kv_rows(
    *,
    k_quant: torch.Tensor,
    k_scale: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    padded_rows = ((k_quant.shape[0] + _BLOCK_K - 1) // _BLOCK_K) * _BLOCK_K
    k_quant = k_quant.contiguous()
    k_scale = k_scale.contiguous()
    if padded_rows == k_quant.shape[0]:
        return k_quant, k_scale

    k_quant_padded = torch.empty(
        (padded_rows, _INDEX_HEAD_DIM),
        dtype=k_quant.dtype,
        device=k_quant.device,
    )
    k_scale_padded = torch.empty(
        (padded_rows,),
        dtype=k_scale.dtype,
        device=k_scale.device,
    )
    k_quant_padded[: k_quant.shape[0]].copy_(k_quant)
    k_quant_padded[k_quant.shape[0] :].zero_()
    k_scale_padded[: k_scale.shape[0]].copy_(k_scale)
    k_scale_padded[k_scale.shape[0] :].zero_()
    return k_quant_padded, k_scale_padded


def _view_last_dim_as_u32(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.dtype != torch.uint8:
        raise ValueError(f"expected uint8 tensor, got {tensor.dtype}")
    if tensor.stride(-1) != 1:
        raise ValueError(f"expected contiguous last dim, got stride={tensor.stride()}")
    if tensor.shape[-1] % 4 != 0:
        raise ValueError(f"last dim must be divisible by 4, got {tensor.shape[-1]}")
    out_shape = (*tensor.shape[:-1], tensor.shape[-1] // 4)
    return tensor.view(torch.uint32).view(out_shape)


def _encode_extend_k_tma_descriptor(
    k_quant_bytes: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if k_quant_bytes.ndim != 2 or k_quant_bytes.shape[1] != _INDEX_HEAD_DIM:
        raise ValueError(
            f"k_quant_bytes must have shape (rows, {_INDEX_HEAD_DIM}), got {tuple(k_quant_bytes.shape)}"
        )
    if k_quant_bytes.dtype != torch.uint8:
        raise TypeError(f"k_quant_bytes must have dtype torch.uint8, got {k_quant_bytes.dtype}")

    swizzle_name = os.environ.get("B12X_NSA_EXTEND_TMA_SWIZZLE", "")
    swizzle = (
        cuda.CUtensorMapSwizzle.CU_TENSOR_MAP_SWIZZLE_128B
        if swizzle_name == "128B"
        else cuda.CUtensorMapSwizzle.CU_TENSOR_MAP_SWIZZLE_NONE
    )
    U64 = cuda.cuuint64_t
    U32 = cuda.cuuint32_t
    row_bytes = int(k_quant_bytes.stride(0)) * k_quant_bytes.element_size()
    base_ptr = int(k_quant_bytes.data_ptr())
    total_rows = int(k_quant_bytes.shape[0])

    result, tensor_map = cuda.cuTensorMapEncodeTiled(
        cuda.CUtensorMapDataType.CU_TENSOR_MAP_DATA_TYPE_UINT8,
        2,
        base_ptr,
        [U64(_INDEX_HEAD_DIM), U64(total_rows)],
        [U64(row_bytes)],
        [U32(_INDEX_HEAD_DIM), U32(_BLOCK_K)],
        [U32(1), U32(1)],
        cuda.CUtensorMapInterleave.CU_TENSOR_MAP_INTERLEAVE_NONE,
        swizzle,
        cuda.CUtensorMapL2promotion.CU_TENSOR_MAP_L2_PROMOTION_NONE,
        cuda.CUtensorMapFloatOOBfill.CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE,
    )
    if result != cuda.CUresult.CUDA_SUCCESS:
        raise RuntimeError(f"cuTensorMapEncodeTiled failed: {result}")

    desc = torch.tensor(
        [int(word) for word in tensor_map.opaque],
        dtype=torch.uint64,
        device=k_quant_bytes.device,
    )
    desc_ptrs = torch.tensor([int(desc.data_ptr())], dtype=torch.int64, device=k_quant_bytes.device)
    return desc, desc_ptrs


def _get_cached_extend_k_tma_descriptor(
    k_quant_bytes: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    key = (
        int(k_quant_bytes.data_ptr()),
        tuple(k_quant_bytes.shape),
        tuple(k_quant_bytes.stride()),
        str(k_quant_bytes.dtype),
        (k_quant_bytes.device.type, k_quant_bytes.device.index),
    )
    cache = getattr(_get_cached_extend_k_tma_descriptor, "_cache", None)
    if cache is None:
        cache = OrderedDict()
        setattr(_get_cached_extend_k_tma_descriptor, "_cache", cache)
    cached = cache.get(key)
    if cached is not None:
        cache.move_to_end(key)
        return cached
    desc = _encode_extend_k_tma_descriptor(k_quant_bytes)
    cache[key] = desc
    if len(cache) > _EXTEND_TMA_DESC_CACHE_SIZE:
        cache.popitem(last=False)
    return desc


@dsl_user_op
def _cp_async_bulk_tensor_2d(
    dst_smem_addr: Int32,
    tensor_map_ptr: Int64,
    coord0: Int32,
    coord1: Int32,
    mbar_smem_addr: Int32,
    *,
    loc=None,
    ip=None,
):
    llvm.inline_asm(
        None,
        [
            Int32(dst_smem_addr).ir_value(loc=loc, ip=ip),
            Int64(tensor_map_ptr).ir_value(loc=loc, ip=ip),
            Int32(coord0).ir_value(loc=loc, ip=ip),
            Int32(coord1).ir_value(loc=loc, ip=ip),
            Int32(mbar_smem_addr).ir_value(loc=loc, ip=ip),
        ],
        "cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes "
        "[$0], [$1, {$2, $3}], [$4];",
        "r,l,r,r,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


@cute.jit
def _permuted_offset_128b(row_idx, vec_idx, row_stride_128b):
    return row_idx * row_stride_128b + (vec_idx ^ (row_idx % 8))


@cute.jit
def _smem_addr_from_b128_offset(base_addr: Int32, offset_128b):
    return base_addr + Int32(offset_128b * 16)


@cute.jit
def _advance_offset_by_row_128b(offset_128b, step_size, row_stride_128b):
    return offset_128b + step_size * row_stride_128b


@cute.jit
def _advance_offset_by_column_128b_2(offset_128b, step_idx):
    xor_term = Int32(0x2) + (Int32(0x4) if step_idx % 2 == 1 else Int32(0))
    extra = Int32(8) if step_idx % 4 == 3 else Int32(0)
    return (offset_128b ^ xor_term) + extra


@cute.jit
def _zero_score_frag(score_frag: cute.Tensor) -> None:
    for reg_id in cutlass.range_constexpr(8):
        score_frag[0, 0, reg_id] = Float32(0.0)


@cute.jit
def _pack_q_mxfp8_reg(
    s_q_bytes: cute.Tensor,
    row: Int32,
    col_pair_base: Int32,
) -> Uint32:
    b0 = Uint32(s_q_bytes[row, col_pair_base + Int32(0)])
    b1 = Uint32(s_q_bytes[row, col_pair_base + Int32(1)])
    b2 = Uint32(s_q_bytes[row, col_pair_base + Int32(8)])
    b3 = Uint32(s_q_bytes[row, col_pair_base + Int32(9)])
    return b0 | (b1 << Int32(8)) | (b2 << Int32(16)) | (b3 << Int32(24))


@cute.jit
def _literal_qk_mma_into_sfrag_mxfp8_raw(
    s_frag: cute.Tensor,
    s_q_bytes: cute.Tensor,
    k_base_addr: Int32,
    lane,
    warp_q_idx,
    warp_kv_idx,
    row_base,
    num_mma_q,
    num_mma_kv,
    num_mma_d_qk,
    upcast_stride_q,
    upcast_stride_k,
):
    unit_scale = Uint32(0x7F7F7F7F)
    k_offset = _permuted_offset_128b(
        row_base + warp_kv_idx * num_mma_kv * Int32(16) + Int32(8) * (lane // Int32(16)) + lane % Int32(8),
        (lane % Int32(16)) // Int32(8),
        upcast_stride_k,
    )
    for mma_pair in cutlass.range_constexpr(num_mma_d_qk // 2):
        q_regs = cute.make_rmem_tensor(
            cute.make_layout((num_mma_q, 4), stride=(4, 1)),
            Uint32,
        )
        group_id = lane // Int32(4)
        thread_id_in_group = lane % Int32(4)
        col_base = Int32(mma_pair * 32) + thread_id_in_group * Int32(2)
        for mma_q in cutlass.range_constexpr(num_mma_q):
            row_base_q = warp_q_idx * Int32(16) + mma_q * Int32(16)
            q_regs[mma_q, 0] = _pack_q_mxfp8_reg(s_q_bytes, row_base_q + group_id, col_base)
            q_regs[mma_q, 1] = _pack_q_mxfp8_reg(
                s_q_bytes, row_base_q + group_id + Int32(8), col_base
            )
            q_regs[mma_q, 2] = _pack_q_mxfp8_reg(
                s_q_bytes, row_base_q + group_id, col_base + Int32(16)
            )
            q_regs[mma_q, 3] = _pack_q_mxfp8_reg(
                s_q_bytes, row_base_q + group_id + Int32(8), col_base + Int32(16)
            )

        k_offset_cur = k_offset
        for mma_kv in cutlass.range_constexpr(num_mma_kv):
            b0_k0, b1_k0 = ldmatrix_m8n8x4_left_half_b16(
                _smem_addr_from_b128_offset(k_base_addr, k_offset_cur)
            )
            b0_k1, b1_k1 = ldmatrix_m8n8x4_right_half_b16(
                _smem_addr_from_b128_offset(k_base_addr, k_offset_cur)
            )
            b0_k0 = frag_layout_swizzle_16b_to_8b(b0_k0)
            b1_k0 = frag_layout_swizzle_16b_to_8b(b1_k0)
            b0_k1 = frag_layout_swizzle_16b_to_8b(b0_k1)
            b1_k1 = frag_layout_swizzle_16b_to_8b(b1_k1)
            k_offset_cur = _advance_offset_by_row_128b(k_offset_cur, Int32(16), upcast_stride_k)

            for mma_q in cutlass.range_constexpr(num_mma_q):
                d0, d1, d2, d3 = mxfp8_mma_m16n8k32_f32_e4m3(
                    s_frag[mma_q, mma_kv, 0],
                    s_frag[mma_q, mma_kv, 1],
                    s_frag[mma_q, mma_kv, 2],
                    s_frag[mma_q, mma_kv, 3],
                    q_regs[mma_q, 0],
                    q_regs[mma_q, 1],
                    q_regs[mma_q, 2],
                    q_regs[mma_q, 3],
                    b0_k0,
                    b0_k1,
                    unit_scale,
                    unit_scale,
                )
                d4, d5, d6, d7 = mxfp8_mma_m16n8k32_f32_e4m3(
                    s_frag[mma_q, mma_kv, 4],
                    s_frag[mma_q, mma_kv, 5],
                    s_frag[mma_q, mma_kv, 6],
                    s_frag[mma_q, mma_kv, 7],
                    q_regs[mma_q, 0],
                    q_regs[mma_q, 1],
                    q_regs[mma_q, 2],
                    q_regs[mma_q, 3],
                    b1_k0,
                    b1_k1,
                    unit_scale,
                    unit_scale,
                )
                s_frag[mma_q, mma_kv, 0] = d0
                s_frag[mma_q, mma_kv, 1] = d1
                s_frag[mma_q, mma_kv, 2] = d2
                s_frag[mma_q, mma_kv, 3] = d3
                s_frag[mma_q, mma_kv, 4] = d4
                s_frag[mma_q, mma_kv, 5] = d5
                s_frag[mma_q, mma_kv, 6] = d6
                s_frag[mma_q, mma_kv, 7] = d7

        k_offset = _advance_offset_by_column_128b_2(k_offset_cur, mma_pair) - Int32(
            num_mma_kv * Int32(16) * upcast_stride_k
        )


@cute.jit
def _stage_q_tile_raw(
    q_u32: cute.Tensor,
    head_idx: Int32,
    q_tile_base: Int32,
    q_base_addr: Int32,
    valid_q_rows: Int32,
    lane: Int32,
):
    linear = lane
    total = Int32(_BLOCK_Q * _FP8_ROW_VECS)
    while linear < total:
        row = linear // Int32(_FP8_ROW_VECS)
        vec_idx = linear - row * Int32(_FP8_ROW_VECS)
        q_row = q_tile_base + row
        dst_addr = q_base_addr + row * Int32(_INDEX_HEAD_DIM) + vec_idx * Int32(16)
        v0 = Uint32(0)
        v1 = Uint32(0)
        v2 = Uint32(0)
        v3 = Uint32(0)
        if q_row < valid_q_rows:
            src_u32 = vec_idx * Int32(4)
            v0 = Uint32(q_u32[q_row, head_idx, src_u32 + Int32(0)])
            v1 = Uint32(q_u32[q_row, head_idx, src_u32 + Int32(1)])
            v2 = Uint32(q_u32[q_row, head_idx, src_u32 + Int32(2)])
            v3 = Uint32(q_u32[q_row, head_idx, src_u32 + Int32(3)])
        st_shared_v4_u32(dst_addr, v0, v1, v2, v3)
        linear += Int32(_THREADS_PER_CTA)


@cute.jit
def _repack_k_tile_to_permuted(
    k_linear_base_addr: Int32,
    k_perm_base_addr: Int32,
    lane: Int32,
):
    linear = lane
    total = Int32(_BLOCK_K * _FP8_ROW_VECS)
    while linear < total:
        row = linear // Int32(_FP8_ROW_VECS)
        vec_idx = linear - row * Int32(_FP8_ROW_VECS)
        src_addr = k_linear_base_addr + Int32((row * Int32(_INDEX_HEAD_DIM) + vec_idx * Int32(16)))
        dst_addr = _smem_addr_from_b128_offset(
            k_perm_base_addr,
            _permuted_offset_128b(row, vec_idx, Int32(_FP8_ROW_VECS)),
        )
        v0, v1, v2, v3 = ld_shared_v4_u32(src_addr)
        st_shared_v4_u32(dst_addr, v0, v1, v2, v3)
        linear += Int32(_THREADS_PER_CTA)


class SparseNSAExtendLogitsKernel:
    """Tile-wise ragged logits kernel with TMA K staging and tensor-core inner loops."""

    @cute.jit
    def __call__(
        self,
        q_u32: cute.Tensor,
        weights: cute.Tensor,
        k_quant_bytes: cute.Tensor,
        k_tma_desc_ptrs: cute.Tensor,
        k_scales: cute.Tensor,
        k_start: cute.Tensor,
        k_end: cute.Tensor,
        logits_out: cute.Tensor,
        valid_q_rows: Int32,
        valid_k_rows: Int32,
        stream: cuda.CUstream,
    ):
        self.kernel(
            q_u32,
            weights,
            k_quant_bytes,
            k_tma_desc_ptrs,
            k_scales,
            k_start,
            k_end,
            logits_out,
            valid_q_rows,
            valid_k_rows,
        ).launch(
            grid=(
                (valid_q_rows + _BLOCK_Q - 1) // _BLOCK_Q,
                (valid_k_rows + _BLOCK_K - 1) // _BLOCK_K,
                1,
            ),
            block=[_THREADS_PER_CTA, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        q_u32: cute.Tensor,
        weights: cute.Tensor,
        k_quant_bytes: cute.Tensor,
        k_tma_desc_ptrs: cute.Tensor,
        k_scales: cute.Tensor,
        k_start: cute.Tensor,
        k_end: cute.Tensor,
        logits_out: cute.Tensor,
        valid_q_rows: Int32,
        valid_k_rows: Int32,
    ):
        tx, _, _ = cute.arch.thread_idx()
        q_tile_idx, k_tile_idx, _ = cute.arch.block_idx()
        lane = tx % Int32(_WARP_THREADS)
        warp_idx = tx // Int32(_WARP_THREADS)
        warp_q_idx = warp_idx // Int32(_WARPS_K)
        warp_k_idx = warp_idx - warp_q_idx * Int32(_WARPS_K)

        q_tile_base = q_tile_idx * Int32(_BLOCK_Q)
        k_tile_base = k_tile_idx * Int32(_BLOCK_K)
        valid_q_rows = Int32(valid_q_rows)
        k_total_rows = Int32(valid_k_rows)
        num_heads = Int32(q_u32.shape[1])

        smem = cutlass.utils.SmemAllocator()

        @cute.struct
        class SharedStorage:
            mbar_ptr_k: cute.struct.MemRange[cutlass.Int64, 1]
            q_tile: cute.struct.Align[
                cute.struct.MemRange[cutlass.Uint8, _BLOCK_Q * _INDEX_HEAD_DIM],
                16,
            ]
            weights: cute.struct.Align[
                cute.struct.MemRange[cutlass.Float32, _BLOCK_Q],
                16,
            ]
            k_linear: cute.struct.Align[
                cute.struct.MemRange[cutlass.Uint8, _BLOCK_K * _INDEX_HEAD_DIM],
                1024,
            ]
            k_perm: cute.struct.Align[
                cute.struct.MemRange[cutlass.Uint8, _BLOCK_K * _INDEX_HEAD_DIM],
                1024,
            ]
            scales: cute.struct.Align[
                cute.struct.MemRange[cutlass.Float32, _BLOCK_K],
                16,
            ]
            k_start: cute.struct.Align[
                cute.struct.MemRange[cutlass.Int32, _BLOCK_Q],
                16,
            ]
            k_end: cute.struct.Align[
                cute.struct.MemRange[cutlass.Int32, _BLOCK_Q],
                16,
            ]

        storage = smem.allocate(SharedStorage)
        mbar_ptr_k = storage.mbar_ptr_k.data_ptr()
        q_base_addr = shared_ptr_to_u32(storage.q_tile.data_ptr())
        k_linear_base_addr = shared_ptr_to_u32(storage.k_linear.data_ptr())
        k_perm_base_addr = shared_ptr_to_u32(storage.k_perm.data_ptr())
        s_weights = storage.weights.get_tensor(cute.make_layout((_BLOCK_Q,), stride=(1,)))
        s_scales = storage.scales.get_tensor(cute.make_layout((_BLOCK_K,), stride=(1,)))
        s_k_start = storage.k_start.get_tensor(cute.make_layout((_BLOCK_Q,), stride=(1,)))
        s_k_end = storage.k_end.get_tensor(cute.make_layout((_BLOCK_Q,), stride=(1,)))
        s_q_bytes = storage.q_tile.get_tensor(
            cute.make_layout((_BLOCK_Q, _INDEX_HEAD_DIM), stride=(_INDEX_HEAD_DIM, 1))
        )
        s_k_linear_bytes = storage.k_linear.get_tensor(
            cute.make_layout((_BLOCK_K * _INDEX_HEAD_DIM,), stride=(1,))
        )

        if tx == 0:
            cute.arch.mbarrier_init(mbar_ptr_k, Int32(1))
        row_linear = tx
        while row_linear < Int32(_BLOCK_Q):
            q_row = q_tile_base + row_linear
            s_k_start[row_linear] = Int32(k_start[q_row]) if q_row < valid_q_rows else Int32(0)
            s_k_end[row_linear] = Int32(k_end[q_row]) if q_row < valid_q_rows else Int32(0)
            row_linear += Int32(_THREADS_PER_CTA)
        cute.arch.sync_threads()

        if tx == 0:
            tile_k_end = k_tile_base + Int32(_BLOCK_K)
            if tile_k_end > k_total_rows:
                tile_k_end = k_total_rows
            tile_live = Int32(0)
            if k_tile_base < tile_k_end:
                for row_idx in cutlass.range_constexpr(_BLOCK_Q):
                    row_start = Int32(s_k_start[row_idx])
                    row_end = Int32(s_k_end[row_idx])
                    row_live = cutlass.select_(row_end > k_tile_base, Int32(1), Int32(0))
                    row_live = cutlass.select_(row_start < tile_k_end, row_live, Int32(0))
                    tile_live = cutlass.select_(row_live != Int32(0), Int32(1), tile_live)
            s_q_bytes[Int32(0), Int32(0)] = cutlass.Int8(tile_live)
        cute.arch.sync_threads()
        if Int32(s_q_bytes[Int32(0), Int32(0)]) != Int32(0):
            producer_state = pipeline.PipelineStateSimple(1, Int32(0))
            consumer_state = pipeline.PipelineStateSimple(1, Int32(0))
            if warp_idx == Int32(0):
                full_mbar_ptr = mbar_ptr_k + producer_state.index
                with cute.arch.elect_one():
                    cute.arch.mbarrier_arrive_and_expect_tx(
                        full_mbar_ptr,
                        Int32(_BLOCK_K * _INDEX_HEAD_DIM),
                    )
                    _cp_async_bulk_tensor_2d(
                        shared_ptr_to_u32(s_k_linear_bytes.iterator),
                        Int64(k_tma_desc_ptrs[Int32(0)]),
                        Int32(0),
                        k_tile_base,
                        shared_ptr_to_u32(full_mbar_ptr),
                    )
            cute.arch.mbarrier_wait(
                mbar_ptr_k + consumer_state.index,
                phase=consumer_state.phase,
            )
            cute.arch.sync_threads()

            _repack_k_tile_to_permuted(k_linear_base_addr, k_perm_base_addr, tx)
            scale_linear = tx
            while scale_linear < Int32(_BLOCK_K):
                s_scales[scale_linear] = Float32(k_scales[k_tile_base + scale_linear])
                scale_linear += Int32(_THREADS_PER_CTA)
            cute.arch.sync_threads()

            frag_layout = cute.make_layout((1, 1, 8), stride=(8, 8, 1))
            acc_frag = cute.make_rmem_tensor(frag_layout, Float32)
            _zero_score_frag(acc_frag)

            head_idx = Int32(0)
            while head_idx < num_heads:
                _stage_q_tile_raw(q_u32, head_idx, q_tile_base, q_base_addr, valid_q_rows, tx)

                weight_linear = tx
                while weight_linear < Int32(_BLOCK_Q):
                    q_row = q_tile_base + weight_linear
                    s_weights[weight_linear] = (
                        Float32(weights[q_row, head_idx]) if q_row < valid_q_rows else Float32(0.0)
                    )
                    weight_linear += Int32(_THREADS_PER_CTA)
                cute.arch.sync_threads()

                score_frag = cute.make_rmem_tensor(frag_layout, Float32)
                _zero_score_frag(score_frag)
                _literal_qk_mma_into_sfrag_mxfp8_raw(
                    score_frag,
                    s_q_bytes,
                    k_perm_base_addr,
                    lane,
                    warp_q_idx,
                    warp_k_idx,
                    Int32(0),
                    Int32(1),
                    Int32(1),
                    Int32(_INDEX_HEAD_DIM // 16),
                    Int32(0),
                    Int32(_FP8_ROW_VECS),
                )
                lane_group = lane // Int32(4)
                for reg_id in cutlass.range_constexpr(8):
                    row_slot = (reg_id % 4) // 2
                    q_local = warp_q_idx * Int32(16) + lane_group + Int32(8 * row_slot)
                    if q_local < Int32(_BLOCK_Q):
                        acc_frag[0, 0, reg_id] = Float32(
                            acc_frag[0, 0, reg_id]
                            + attention_utils.fmax(score_frag[0, 0, reg_id], Float32(0.0))
                            * s_weights[q_local]
                        )
                cute.arch.sync_threads()
                head_idx += Int32(1)

            lane_group = lane // Int32(4)
            lane_pair_base = Int32(2) * (lane % Int32(4))
            for reg_id in cutlass.range_constexpr(8):
                row_slot = (reg_id % 4) // 2
                q_local = warp_q_idx * Int32(16) + lane_group + Int32(8 * row_slot)
                k_local = (
                    warp_k_idx * Int32(16)
                    + lane_pair_base
                    + Int32(8 * (reg_id // 4))
                    + Int32(reg_id % 2)
                )
                q_row = q_tile_base + q_local
                k_row = k_tile_base + k_local
                if (
                    q_local < Int32(_BLOCK_Q)
                    and k_local < Int32(_BLOCK_K)
                    and q_row < valid_q_rows
                    and k_row < k_total_rows
                ):
                    row_start = Int32(s_k_start[q_local])
                    row_end = Int32(s_k_end[q_local])
                    if k_row >= row_start and k_row < row_end:
                        logits_out[q_row, k_row] = Float32(acc_frag[0, 0, reg_id] * s_scales[k_local])


@lru_cache(maxsize=16)
def _build_sparse_nsa_extend_kernel() -> SparseNSAExtendLogitsKernel:
    return SparseNSAExtendLogitsKernel()


def supports_sparse_nsa_extend_logits_kernel(
    *,
    q_fp8: torch.Tensor,
    weights: torch.Tensor,
    k_quant: torch.Tensor,
    k_scale: torch.Tensor,
    k_start: torch.Tensor,
    k_end: torch.Tensor,
) -> bool:
    if os.environ.get("B12X_NSA_INDEXER_FORCE_REFERENCE", "0") == "1":
        return False
    if q_fp8.device.type != "cuda":
        return False
    if q_fp8.ndim != 3 or q_fp8.shape[2] != _INDEX_HEAD_DIM:
        return False
    if q_fp8.shape[1] > _MAX_Q_HEADS:
        return False
    if weights.ndim != 2 or weights.shape != q_fp8.shape[:2]:
        return False
    if k_quant.ndim != 2 or k_quant.shape[1] != _INDEX_HEAD_DIM:
        return False
    if k_scale.ndim != 1 or k_scale.shape[0] != k_quant.shape[0]:
        return False
    if k_start.ndim != 1 or k_end.ndim != 1 or k_start.shape != k_end.shape:
        return False
    if k_start.shape[0] > q_fp8.shape[0]:
        return False
    if q_fp8.dtype != torch.float8_e4m3fn:
        return False
    if weights.dtype != torch.float32:
        return False
    if k_quant.dtype != torch.float8_e4m3fn:
        return False
    if k_scale.dtype != torch.float32:
        return False
    if k_start.dtype != torch.int32 or k_end.dtype != torch.int32:
        return False
    if not (
        q_fp8.device
        == weights.device
        == k_quant.device
        == k_scale.device
        == k_start.device
        == k_end.device
    ):
        return False
    return True


def run_sparse_nsa_extend_logits_kernel(
    *,
    q_fp8: torch.Tensor,
    weights: torch.Tensor,
    k_quant: torch.Tensor,
    k_scale: torch.Tensor,
    k_start: torch.Tensor,
    k_end: torch.Tensor,
    contract_phantoms: dict[str, torch.Tensor] | None = None,
    workspace=None,
) -> torch.Tensor:
    if not supports_sparse_nsa_extend_logits_kernel(
        q_fp8=q_fp8,
        weights=weights,
        k_quant=k_quant,
        k_scale=k_scale,
        k_start=k_start,
        k_end=k_end,
    ):
        raise ValueError("sparse NSA extend logits kernel only supports the exact CUDA FP8 contract")

    q_rows_total = int(q_fp8.shape[0])
    valid_q_rows = int(k_start.shape[0])
    k_rows = int(k_quant.shape[0])

    if workspace is not None:
        staged = workspace.stage_nsa_indexer_extend(
            q_fp8=q_fp8,
            weights=weights,
            k_quant=k_quant,
            k_scale=k_scale,
            k_start=k_start,
            k_end=k_end,
        )
        q_u32 = staged["q_u32"]
        weights_kernel = staged["weights"]
        k_quant_bytes = staged["k_quant_bytes"]
        k_scale_kernel = staged["k_scales"]
        k_start_kernel = staged["k_start"]
        k_end_kernel = staged["k_end"]
        out_kernel = staged["logits"]
        out_view = staged["logits_view"]
        if contract_phantoms is None:
            contract_phantoms = workspace.get_indexer_contract_phantoms()
    else:
        out_kernel = torch.full(
            (q_rows_total, k_rows),
            float("-inf"),
            dtype=torch.float32,
            device=q_fp8.device,
        )
        out_view = out_kernel
        if valid_q_rows == 0 or k_rows == 0:
            return out_view

        q_bytes = q_fp8.contiguous().view(torch.uint8)
        k_quant_padded, k_scale_padded = _pad_kv_rows(k_quant=k_quant, k_scale=k_scale)
        k_quant_bytes = k_quant_padded.contiguous().view(torch.uint8)
        q_u32 = _view_last_dim_as_u32(q_bytes)
        weights_kernel = weights.contiguous()
        k_scale_kernel = k_scale_padded.contiguous()
        k_start_kernel = k_start.contiguous()
        k_end_kernel = k_end.contiguous()

    if valid_q_rows == 0 or k_rows == 0:
        return out_view

    _, k_tma_desc_ptrs = _get_cached_extend_k_tma_descriptor(k_quant_bytes)
    kernel = _build_sparse_nsa_extend_kernel()
    args = (
        _to_kernel_tensor(q_u32, cutlass.Uint32),
        _to_kernel_tensor(weights_kernel, cutlass.Float32, assumed_align=4),
        _to_kernel_tensor(k_quant_bytes, cutlass.Uint8),
        _to_kernel_tensor(k_tma_desc_ptrs, cutlass.Int64, assumed_align=8),
        _to_kernel_tensor(k_scale_kernel, cutlass.Float32, assumed_align=4),
        _to_kernel_tensor(k_start_kernel, cutlass.Int32, assumed_align=4),
        _to_kernel_tensor(k_end_kernel, cutlass.Int32, assumed_align=4),
        _to_kernel_tensor(out_kernel, cutlass.Float32, assumed_align=4),
        valid_q_rows,
        k_rows,
        current_cuda_stream(),
    )
    _cp = contract_phantoms or {}
    cache_key = (
        _tensor_meta_key(_cp.get("extend_q_u32", q_u32)),
        _tensor_meta_key(_cp.get("extend_weights", weights_kernel)),
        _tensor_meta_key(_cp.get("extend_k_quant", k_quant_bytes)),
        _tensor_meta_key(k_tma_desc_ptrs),
        _tensor_meta_key(_cp.get("extend_k_scale", k_scale_kernel)),
        _tensor_meta_key(_cp.get("extend_k_start", k_start_kernel)),
        _tensor_meta_key(_cp.get("extend_k_end", k_end_kernel)),
        _tensor_meta_key(_cp.get("extend_logits", out_kernel)),
    )
    _run_cached_host_launcher(kernel, cache_key, args)
    return out_view
