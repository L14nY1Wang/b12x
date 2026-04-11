"""CuTeDSL kernel path for NSA top-k logits under the current GLM-5.1 contract."""

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
from cutlass.cute.runtime import from_dlpack

from b12x.attention import utils as attention_utils
from b12x.cute.fp4 import bfloat2_to_float2_scaled, fp8x4_e4m3_to_bfloat2x2
from b12x.cute.utils import current_cuda_stream


_INDEX_HEAD_DIM = 128
_PAGE_SIZE = 64
_SCALE_BYTES = 4
_WARP_THREADS = 32
_TOKENS_PER_CTA = 4
_THREADS_PER_CTA = _WARP_THREADS * _TOKENS_PER_CTA
_SCORE_CTAS_PER_ROW = 32
_MAX_Q_HEADS = 64
_EAGER_HOST_LAUNCHER_CACHE_SIZE = 32


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


@cute.jit
def _warp_allreduce_sum(value: Float32) -> Float32:
    for shift in cutlass.range_constexpr(5):
        value = Float32(value + cute.arch.shuffle_sync_bfly(value, offset=1 << shift))
    return value


@cute.jit
def _load_fp8x4(mBytes: cute.Tensor, row0: Int32, row1: Int32, base: Int32):
    packed = (
        Uint32(mBytes[row0, row1, base + Int32(0)])
        | (Uint32(mBytes[row0, row1, base + Int32(1)]) << Uint32(8))
        | (Uint32(mBytes[row0, row1, base + Int32(2)]) << Uint32(16))
        | (Uint32(mBytes[row0, row1, base + Int32(3)]) << Uint32(24))
    )
    bf2_01, bf2_23 = fp8x4_e4m3_to_bfloat2x2(packed)
    v0, v1 = bfloat2_to_float2_scaled(bf2_01, Float32(1.0))
    v2, v3 = bfloat2_to_float2_scaled(bf2_23, Float32(1.0))
    return v0, v1, v2, v3


@cute.jit
def _load_fp8x4_2d(mBytes: cute.Tensor, row0: Int32, base: Int32):
    packed = (
        Uint32(mBytes[row0, base + Int32(0)])
        | (Uint32(mBytes[row0, base + Int32(1)]) << Uint32(8))
        | (Uint32(mBytes[row0, base + Int32(2)]) << Uint32(16))
        | (Uint32(mBytes[row0, base + Int32(3)]) << Uint32(24))
    )
    bf2_01, bf2_23 = fp8x4_e4m3_to_bfloat2x2(packed)
    v0, v1 = bfloat2_to_float2_scaled(bf2_01, Float32(1.0))
    v2, v3 = bfloat2_to_float2_scaled(bf2_23, Float32(1.0))
    return v0, v1, v2, v3


class SparseNSAIndexerLogitsKernel:
    """One CTA reuses a query row across a small tile of candidate positions."""

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
        trivial_topk: cute.Tensor,
        logits_out: cute.Tensor,
        stream: cuda.CUstream,
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
            trivial_topk,
            logits_out,
        ).launch(
            grid=(
                q_bytes.shape[0],
                _SCORE_CTAS_PER_ROW,
                1,
            ),
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
        trivial_topk: cute.Tensor,
        logits_out: cute.Tensor,
    ):
        tx, _, _ = cute.arch.thread_idx()
        q_idx, cta_idx, _ = cute.arch.block_idx()
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
        sQ = storage.qBytes.get_tensor(
            cute.make_layout((_MAX_Q_HEADS, _INDEX_HEAD_DIM), stride=(_INDEX_HEAD_DIM, 1))
        )
        sW = storage.weights.get_tensor(cute.make_layout((_MAX_Q_HEADS,), stride=(1,)))

        num_heads = Int32(q_bytes.shape[1])
        q_linear = tx
        total_q_bytes = num_heads * Int32(_INDEX_HEAD_DIM)
        while q_linear < total_q_bytes:
            head_idx = q_linear // Int32(_INDEX_HEAD_DIM)
            col_idx = q_linear - head_idx * Int32(_INDEX_HEAD_DIM)
            sQ[head_idx, col_idx] = q_bytes[q_idx, head_idx, col_idx]
            q_linear += Int32(_THREADS_PER_CTA)

        w_linear = tx
        while w_linear < num_heads:
            sW[w_linear] = Float32(weights[q_idx, w_linear])
            w_linear += Int32(_THREADS_PER_CTA)
        cute.arch.sync_threads()

        width = Int32(page_table_1.shape[1])
        seq_len = Int32(seqlens_per_query[q_idx])
        live_width = Int32(active_width[Int32(0)])
        skip_topk = Int32(trivial_topk[Int32(0)])
        if live_width > width:
            live_width = width
        batch_row = Int32(query_row_to_batch[q_idx])
        token_pos = cta_idx * Int32(_TOKENS_PER_CTA) + warp_idx
        token_stride = Int32(_SCORE_CTAS_PER_ROW * _TOKENS_PER_CTA)
        if seq_len > skip_topk:
            while token_pos < live_width:
                if lane == 0:
                    logits_out[q_idx, token_pos] = Float32(-Float32.inf)

                token_id = Int32(-1)
                if token_pos < seq_len:
                    token_id = Int32(page_table_1[batch_row, token_pos])

                if token_pos < seq_len and token_id >= Int32(0):
                    page_idx = token_id // Int32(_PAGE_SIZE)
                    slot_idx = token_id - page_idx * Int32(_PAGE_SIZE)
                    base = lane * Int32(4)

                    logit = Float32(0.0)
                    head_idx = Int32(0)
                    while head_idx < num_heads:
                        q0, q1, q2, q3 = _load_fp8x4_2d(sQ, head_idx, base)
                        k0, k1, k2, k3 = _load_fp8x4(k_quant_bytes, page_idx, slot_idx, base)
                        dot = Float32(q0 * k0 + q1 * k1 + q2 * k2 + q3 * k3)
                        dot = _warp_allreduce_sum(dot)
                        if lane == 0:
                            logit = Float32(
                                logit + attention_utils.fmax(dot, Float32(0.0)) * sW[head_idx]
                            )
                        head_idx += Int32(1)

                    if lane == 0:
                        logits_out[q_idx, token_pos] = Float32(
                            logit * Float32(k_scales[page_idx, slot_idx])
                        )

                token_pos += token_stride


@lru_cache(maxsize=8)
def _build_sparse_nsa_indexer_kernel() -> SparseNSAIndexerLogitsKernel:
    return SparseNSAIndexerLogitsKernel()


@lru_cache(maxsize=64)
def _cached_trivial_topk_tensor(
    topk: int,
    device_type: str,
    device_index: int | None,
) -> torch.Tensor:
    return torch.tensor([topk], dtype=torch.int32, device=torch.device(device_type, device_index))


def clear_sparse_nsa_indexer_kernel_cache() -> None:
    _build_sparse_nsa_indexer_kernel.cache_clear()
    _cached_trivial_topk_tensor.cache_clear()


def _split_index_k_cache_runtime_views(index_k_cache: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if index_k_cache.stride(-1) != 1:
        raise ValueError(
            f"index_k_cache must have a contiguous last dimension, got stride={index_k_cache.stride()}"
        )
    num_pages = index_k_cache.shape[0]
    data_bytes = _PAGE_SIZE * _INDEX_HEAD_DIM
    k_quant_bytes = index_k_cache[:, :data_bytes].view(num_pages, _PAGE_SIZE, _INDEX_HEAD_DIM)
    k_scales = (
        index_k_cache[:, data_bytes : data_bytes + _PAGE_SIZE * _SCALE_BYTES]
        .view(num_pages, _PAGE_SIZE, _SCALE_BYTES)
        .view(torch.float32)
        .squeeze(-1)
    )
    return k_quant_bytes, k_scales


def supports_sparse_nsa_indexer_kernel(
    *,
    q_fp8: torch.Tensor,
    weights: torch.Tensor,
    index_k_cache: torch.Tensor,
    page_table_1: torch.Tensor,
    query_row_to_batch: torch.Tensor,
    seqlens_per_query: torch.Tensor,
    page_size: int,
) -> bool:
    if os.environ.get("B12X_NSA_INDEXER_FORCE_REFERENCE", "0") == "1":
        return False
    if page_size != _PAGE_SIZE:
        return False
    if q_fp8.device.type != "cuda":
        return False
    if not (
        weights.device
        == index_k_cache.device
        == page_table_1.device
        == query_row_to_batch.device
        == seqlens_per_query.device
        == q_fp8.device
    ):
        return False
    if q_fp8.ndim != 3 or q_fp8.shape[2] != _INDEX_HEAD_DIM:
        return False
    if q_fp8.shape[1] > _MAX_Q_HEADS:
        return False
    if weights.ndim != 2 or weights.shape != q_fp8.shape[:2]:
        return False
    if page_table_1.ndim != 2:
        return False
    if query_row_to_batch.ndim != 1 or query_row_to_batch.shape[0] != q_fp8.shape[0]:
        return False
    if seqlens_per_query.ndim != 1 or seqlens_per_query.shape[0] != q_fp8.shape[0]:
        return False
    if index_k_cache.ndim != 2 or index_k_cache.shape[1] != _PAGE_SIZE * (_INDEX_HEAD_DIM + _SCALE_BYTES):
        return False
    if q_fp8.dtype != torch.float8_e4m3fn:
        return False
    if weights.dtype != torch.float32:
        return False
    if index_k_cache.dtype != torch.uint8:
        return False
    if index_k_cache.stride(-1) != 1:
        return False
    if (
        page_table_1.dtype != torch.int32
        or query_row_to_batch.dtype != torch.int32
        or seqlens_per_query.dtype != torch.int32
    ):
        return False
    return True


def run_sparse_nsa_index_logits_kernel(
    *,
    q_fp8: torch.Tensor,
    weights: torch.Tensor,
    index_k_cache: torch.Tensor,
    page_table_1: torch.Tensor,
    query_row_to_batch: torch.Tensor,
    seqlens_per_query: torch.Tensor,
    active_width: torch.Tensor | None = None,
    trivial_topk: int | None = None,
    page_size: int = _PAGE_SIZE,
) -> torch.Tensor:
    if not supports_sparse_nsa_indexer_kernel(
        q_fp8=q_fp8,
        weights=weights,
        index_k_cache=index_k_cache,
        page_table_1=page_table_1,
        query_row_to_batch=query_row_to_batch,
        seqlens_per_query=seqlens_per_query,
        page_size=page_size,
    ):
        raise ValueError("sparse NSA indexer kernel only supports the exact CUDA page_size=64 FP8 contract")

    rows = q_fp8.shape[0]
    width = page_table_1.shape[1]
    if rows == 0 or width == 0:
        return torch.empty((rows, width), dtype=torch.float32, device=q_fp8.device)
    if active_width is None:
        active_width = torch.tensor([width], dtype=torch.int32, device=q_fp8.device)
    if active_width.shape != (1,):
        raise ValueError(f"active_width must have shape (1,), got {tuple(active_width.shape)}")
    if active_width.dtype != torch.int32:
        raise ValueError(f"active_width must have dtype torch.int32, got {active_width.dtype}")
    if active_width.device != q_fp8.device:
        raise ValueError(
            f"active_width device {active_width.device} does not match q_fp8 device {q_fp8.device}"
        )
    if trivial_topk is None:
        trivial_topk = 0
    trivial_topk_tensor = _cached_trivial_topk_tensor(
        int(trivial_topk),
        q_fp8.device.type,
        q_fp8.device.index,
    )

    k_quant_bytes, k_scales = _split_index_k_cache_runtime_views(index_k_cache)
    q_bytes = q_fp8.contiguous().view(torch.uint8)
    logits = torch.empty((rows, width), dtype=torch.float32, device=q_fp8.device)

    kernel = _build_sparse_nsa_indexer_kernel()
    args = (
        _to_kernel_tensor(q_bytes, cutlass.Uint8),
        _to_kernel_tensor(weights.contiguous(), cutlass.Float32, assumed_align=4),
        _to_kernel_tensor(k_quant_bytes, cutlass.Uint8),
        _to_kernel_tensor(k_scales, cutlass.Float32, assumed_align=4),
        _to_kernel_tensor(page_table_1.contiguous(), cutlass.Int32, assumed_align=4),
        _to_kernel_tensor(query_row_to_batch.contiguous(), cutlass.Int32, assumed_align=4),
        _to_kernel_tensor(seqlens_per_query.contiguous(), cutlass.Int32, assumed_align=4),
        _to_kernel_tensor(active_width.contiguous(), cutlass.Int32, assumed_align=4),
        _to_kernel_tensor(trivial_topk_tensor, cutlass.Int32, assumed_align=4),
        _to_kernel_tensor(logits, cutlass.Float32, assumed_align=4),
        current_cuda_stream(),
    )
    cache_key = (
        _tensor_meta_key(q_bytes),
        _tensor_meta_key(weights),
        _tensor_meta_key(k_quant_bytes),
        _tensor_meta_key(k_scales),
        _tensor_meta_key(page_table_1),
        _tensor_meta_key(query_row_to_batch),
        _tensor_meta_key(seqlens_per_query),
        _tensor_meta_key(active_width),
        _tensor_meta_key(trivial_topk_tensor),
        _tensor_meta_key(logits),
    )
    _run_cached_host_launcher(kernel, cache_key, args)
    return logits
