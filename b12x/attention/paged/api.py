"""Public isolated API for the primary paged-attention backend."""

from __future__ import annotations

from functools import lru_cache
import os
from typing import Literal

import cuda.bindings.driver as cuda
import cutlass
import torch
from cutlass.cute.runtime import from_dlpack

from b12x.cute.utils import current_cuda_stream

from .forward_paged import (
    PagedForwardKernel,
    PagedFp8DecodeRawForwardKernel,
    PagedFp8ExtendRawForwardKernel,
    PagedFp8RawPlaneDumpKernel,
)
from .merge import PagedPersistentMergeKernel, default_paged_persistent_ctas
from .planner import PagedPlan
from .traits import PagedForwardTraits, select_paged_forward_traits_from_plan
from .workspace import PagedAttentionWorkspace


def _torch_to_cutlass_dtype(dtype: torch.dtype) -> type[cutlass.Numeric]:
    if dtype == torch.bfloat16:
        return cutlass.BFloat16
    if dtype == torch.float16:
        return cutlass.Float16
    if dtype == torch.float8_e4m3fn:
        return cutlass.Float8E4M3FN
    if dtype == torch.float32:
        return cutlass.Float32
    raise TypeError(f"unsupported dtype {dtype}")


def _torch_to_cutlass_storage_dtype(dtype: torch.dtype) -> type[cutlass.Numeric]:
    if dtype == torch.float8_e4m3fn:
        return cutlass.Uint8
    return _torch_to_cutlass_dtype(dtype)


def _to_kernel_tensor(
    tensor: torch.Tensor | None,
    dtype: type[cutlass.Numeric],
    *,
    assumed_align: int = 16,
) -> torch.Tensor | cutlass.cute.Tensor | None:
    if tensor is None:
        return None
    cute_tensor = from_dlpack(tensor, assumed_align=assumed_align)
    cute_tensor.element_type = dtype
    leading_dim = next((idx for idx, stride in enumerate(tensor.stride()) if stride == 1), None)
    if leading_dim is not None and tensor.ndim >= 2:
        cute_tensor = cute_tensor.mark_layout_dynamic(leading_dim=leading_dim)
    return cute_tensor


def _as_int32_tensor(tensor: torch.Tensor) -> torch.Tensor:
    return tensor if tensor.dtype == torch.int32 else tensor.to(torch.int32)


def _attn_turbo_enabled(attn_mode: Literal["default", "turbo"] | None) -> bool:
    if attn_mode == "turbo":
        return True
    if attn_mode == "default":
        return False
    return os.environ.get("B12X_ATTN", "").upper() == "TURBO"


@lru_cache(maxsize=4)
def _dummy_fp8_tma_desc(device_index: int) -> torch.Tensor:
    return torch.zeros((1, 16), dtype=torch.uint64, device=torch.device("cuda", device_index))


@lru_cache(maxsize=4)
def _dummy_fp8_tma_desc_ptrs(device_index: int) -> torch.Tensor:
    return torch.zeros((1,), dtype=torch.int64, device=torch.device("cuda", device_index))


def _encode_fp8_plane_tma_descriptors(
    cache: torch.Tensor,
    *,
    plane_cols: int,
    tile_rows: int | None = None,
) -> torch.Tensor:
    if cache.dtype != torch.float8_e4m3fn:
        raise TypeError("fp8 plane TMA descriptors require float8_e4m3fn cache tensors")
    if cache.ndim != 4:
        raise ValueError("cache must have shape [num_pages, page_size, kv_heads, head_dim]")
    num_pages, page_size, kv_heads, head_dim = [int(dim) for dim in cache.shape]
    if plane_cols <= 0 or head_dim % plane_cols != 0:
        raise ValueError(f"plane_cols={plane_cols} must divide head_dim={head_dim}")
    if tile_rows is None:
        tile_rows = page_size
    if tile_rows <= 0 or page_size % tile_rows != 0:
        raise ValueError(f"tile_rows={tile_rows} must be positive and divide page_size={page_size}")

    swizzle_name = os.environ.get("B12X_PAGED_KV_TMA_PLANE_SWIZZLE", "")
    swizzle = (
        cuda.CUtensorMapSwizzle.CU_TENSOR_MAP_SWIZZLE_NONE
        if swizzle_name == "none"
        else cuda.CUtensorMapSwizzle.CU_TENSOR_MAP_SWIZZLE_128B
    )
    U64 = cuda.cuuint64_t
    U32 = cuda.cuuint32_t
    row_bytes = kv_heads * head_dim
    total_rows = num_pages * page_size
    base_ptr = int(cache.view(torch.uint8).data_ptr())
    head_stride_bytes = head_dim

    host_desc = torch.empty((kv_heads, 16), dtype=torch.uint64)
    for kv_head_idx in range(kv_heads):
        result, tensor_map = cuda.cuTensorMapEncodeTiled(
            cuda.CUtensorMapDataType.CU_TENSOR_MAP_DATA_TYPE_UINT8,
            2,
            base_ptr + kv_head_idx * head_stride_bytes,
            [U64(head_dim), U64(total_rows)],
            [U64(row_bytes)],
            [U32(plane_cols), U32(tile_rows)],
            [U32(1), U32(1)],
            cuda.CUtensorMapInterleave.CU_TENSOR_MAP_INTERLEAVE_NONE,
            swizzle,
            cuda.CUtensorMapL2promotion.CU_TENSOR_MAP_L2_PROMOTION_NONE,
            cuda.CUtensorMapFloatOOBfill.CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE,
        )
        if result != cuda.CUresult.CUDA_SUCCESS:
            raise RuntimeError(f"cuTensorMapEncodeTiled failed: {result}")
        host_desc[kv_head_idx] = torch.tensor(
            [int(word) for word in tensor_map.opaque],
            dtype=torch.uint64,
        )
    return host_desc.to(device=cache.device, non_blocking=False)


def _descriptor_row_ptrs(desc: torch.Tensor) -> torch.Tensor:
    row_bytes = int(desc.stride(0)) * desc.element_size()
    base_ptr = int(desc.data_ptr())
    ptrs = [base_ptr + idx * row_bytes for idx in range(int(desc.shape[0]))]
    return torch.tensor(ptrs, dtype=torch.int64, device=desc.device)


def _use_fp8_decode_raw_specialization(
    traits: PagedForwardTraits,
    *,
    split_kv: bool,
    enable_paged_kv_tma: bool,
) -> bool:
    return (
        enable_paged_kv_tma
        and os.environ.get("B12X_PAGED_KV_TMA", "1") != "0"
        and not split_kv
        and traits.kv_dtype == torch.float8_e4m3fn
        and traits.q_dtype == torch.bfloat16
        and traits.o_dtype == torch.bfloat16
        and traits.num_warps_q == 1
        and traits.num_warps_kv == 4
        and traits.num_threads == 128
        and traits.cta_tile_q == 16
        and traits.num_mma_q == 1
        and traits.num_mma_kv == 1
        and traits.head_dim_qk == 256
        and traits.head_dim_vo == 256
    )


def _use_fp8_extend_raw_specialization(
    traits: PagedForwardTraits,
    *,
    split_kv: bool,
    enable_paged_kv_tma: bool,
) -> bool:
    del split_kv
    return (
        enable_paged_kv_tma
        and os.environ.get("B12X_PAGED_KV_TMA", "1") != "0"
        and traits.kv_dtype == torch.float8_e4m3fn
        and traits.q_dtype == torch.bfloat16
        and traits.o_dtype == torch.bfloat16
        and traits.num_warps_q == 4
        and traits.num_warps_kv == 1
        and traits.num_threads == 128
        and traits.cta_tile_q == 64
        and traits.num_mma_q == 1
        and traits.num_mma_kv == 2
        and traits.head_dim_qk == 256
        and traits.head_dim_vo == 256
    )


@lru_cache(maxsize=64)
def _build_forward_kernel(
    traits: PagedForwardTraits,
    split_kv: bool,
    mxfp8_turbo: bool,
    enable_mxfp8_pv: bool,
    enable_paged_kv_tma: bool,
) -> PagedForwardKernel:
    return PagedForwardKernel(
        _torch_to_cutlass_dtype(traits.q_dtype),
        _torch_to_cutlass_dtype(traits.kv_dtype),
        _torch_to_cutlass_storage_dtype(traits.kv_dtype),
        _torch_to_cutlass_dtype(traits.o_dtype),
        traits=traits,
        split_kv=split_kv,
        mxfp8_turbo=mxfp8_turbo,
        enable_mxfp8_pv=enable_mxfp8_pv,
        enable_paged_kv_tma=enable_paged_kv_tma,
    )


@lru_cache(maxsize=16)
def _build_fp8_decode_raw_forward_kernel(
    traits: PagedForwardTraits,
    split_kv: bool,
    mxfp8_turbo: bool,
    enable_mxfp8_pv: bool,
    enable_paged_kv_tma: bool,
) -> PagedFp8DecodeRawForwardKernel:
    del traits, split_kv, mxfp8_turbo, enable_mxfp8_pv, enable_paged_kv_tma
    return PagedFp8DecodeRawForwardKernel()


@lru_cache(maxsize=16)
def _build_fp8_extend_raw_forward_kernel(
    traits: PagedForwardTraits,
    split_kv: bool,
    mxfp8_turbo: bool,
    enable_mxfp8_pv: bool,
    enable_paged_kv_tma: bool,
) -> PagedFp8ExtendRawForwardKernel:
    del traits, mxfp8_turbo, enable_mxfp8_pv, enable_paged_kv_tma
    return PagedFp8ExtendRawForwardKernel(split_kv=split_kv)


@lru_cache(maxsize=16)
def _build_merge_kernel(
    dtype: torch.dtype,
    head_dim: int,
    persistent_ctas: int,
) -> PagedPersistentMergeKernel:
    cutlass_dtype = _torch_to_cutlass_dtype(dtype)
    return PagedPersistentMergeKernel(
        cutlass_dtype,
        cutlass_dtype,
        head_dim=head_dim,
        persistent_ctas=persistent_ctas,
    )


@lru_cache(maxsize=4)
def _build_fp8_planewords_dump_kernel() -> PagedFp8RawPlaneDumpKernel:
    return PagedFp8RawPlaneDumpKernel()


def _get_cached_fp8_tma_descs(
    workspace: PagedAttentionWorkspace,
    *,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    plane_cols: int,
    tile_rows: int,
) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor, torch.Tensor] | None:
    cached = getattr(workspace, "_live_fp8_tma_desc_cache", None)
    if cached is None:
        return None
    key = (
        int(k_cache.data_ptr()),
        int(v_cache.data_ptr()),
        tuple(k_cache.shape),
        tuple(v_cache.shape),
        plane_cols,
        tile_rows,
    )
    if cached.get("key") != key:
        return None
    return (
        cached.get("k_desc"),
        cached.get("v_desc"),
        cached["k_ptrs"],
        cached["v_ptrs"],
    )


def paged_attention_forward(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    *,
    workspace: PagedAttentionWorkspace,
    output: torch.Tensor,
    k_descale: torch.Tensor | None = None,
    v_descale: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    plan = workspace.plan
    page_table = workspace.page_table
    cache_seqlens = workspace.cache_seqlens
    cu_seqlens_q = workspace.cu_seqlens_q
    if page_table is None or cache_seqlens is None or cu_seqlens_q is None:
        raise RuntimeError("paged workspace metadata has not been prepared")
    if plan.split_kv and (workspace.tmp_output is None or workspace.tmp_lse is None):
        raise ValueError("split-kv plan requires tmp_output and tmp_lse in the workspace")
    if k_descale is not None and k_descale.ndim == 2 and int(k_descale.shape[1]) == 1:
        k_descale = k_descale[:, 0].contiguous()
    if v_descale is not None and v_descale.ndim == 2 and int(v_descale.shape[1]) == 1:
        v_descale = v_descale[:, 0].contiguous()
    if output.ndim != 3:
        raise ValueError(f"output must be rank-3 [total_q, heads, head_dim], got {tuple(output.shape)}")
    if int(output.shape[0]) < int(plan.total_q):
        raise ValueError(
            f"output first dimension must be at least total_q={plan.total_q}, got {int(output.shape[0])}"
        )
    if tuple(output.shape[1:]) != (plan.num_q_heads, plan.head_dim_vo):
        raise ValueError(
            "output shape must match the prepared workspace contract: "
            f"expected (*, {plan.num_q_heads}, {plan.head_dim_vo}), got {tuple(output.shape)}"
        )

    if (k_cache.dtype == torch.float8_e4m3fn or v_cache.dtype == torch.float8_e4m3fn) and (
        k_descale is None or v_descale is None
    ):
        raise ValueError("fp8 paged caches require k_descale and v_descale")

    traits = select_paged_forward_traits_from_plan(plan)
    mxfp8_turbo = _attn_turbo_enabled(workspace.attn_mode) and plan.kv_dtype == torch.float8_e4m3fn
    enable_mxfp8_pv = mxfp8_turbo and plan.mode == "decode" and plan.kv_chunk_size <= 384
    enable_paged_kv_tma = os.environ.get("B12X_PAGED_KV_TMA", "1") != "0"
    if _use_fp8_decode_raw_specialization(
        traits,
        split_kv=plan.split_kv,
        enable_paged_kv_tma=enable_paged_kv_tma,
    ):
        forward_kernel = _build_fp8_decode_raw_forward_kernel(
            traits,
            plan.split_kv,
            mxfp8_turbo,
            enable_mxfp8_pv,
            enable_paged_kv_tma,
        )
    elif _use_fp8_extend_raw_specialization(
        traits,
        split_kv=plan.split_kv,
        enable_paged_kv_tma=enable_paged_kv_tma,
    ):
        forward_kernel = _build_fp8_extend_raw_forward_kernel(
            traits,
            plan.split_kv,
            mxfp8_turbo,
            enable_mxfp8_pv,
            enable_paged_kv_tma,
        )
    else:
        forward_kernel = _build_forward_kernel(
            traits,
            plan.split_kv,
            mxfp8_turbo,
            enable_mxfp8_pv,
            enable_paged_kv_tma,
        )
    forward_output = workspace.tmp_output if plan.split_kv else output
    forward_lse = workspace.tmp_lse if plan.split_kv else workspace.lse
    assert forward_output is not None
    assert forward_lse is not None

    q_arg = _to_kernel_tensor(q, _torch_to_cutlass_dtype(q.dtype))
    k_cache_arg = (
        _to_kernel_tensor(k_cache.view(torch.uint8), cutlass.Uint8)
        if k_cache.dtype == torch.float8_e4m3fn
        else _to_kernel_tensor(k_cache, _torch_to_cutlass_dtype(k_cache.dtype))
    )
    v_cache_arg = (
        _to_kernel_tensor(v_cache.view(torch.uint8), cutlass.Uint8)
        if v_cache.dtype == torch.float8_e4m3fn
        else _to_kernel_tensor(v_cache, _torch_to_cutlass_dtype(v_cache.dtype))
    )
    forward_output_arg = _to_kernel_tensor(forward_output, _torch_to_cutlass_dtype(forward_output.dtype))
    forward_lse_arg = _to_kernel_tensor(forward_lse, cutlass.Float32)
    page_table_arg = _to_kernel_tensor(_as_int32_tensor(page_table), cutlass.Int32, assumed_align=4)
    cache_seqlens_arg = _to_kernel_tensor(
        _as_int32_tensor(cache_seqlens), cutlass.Int32, assumed_align=4
    )
    cu_seqlens_q_arg = _to_kernel_tensor(
        _as_int32_tensor(cu_seqlens_q), cutlass.Int32, assumed_align=4
    )
    request_indices_arg = _to_kernel_tensor(workspace.request_indices, cutlass.Int32, assumed_align=4)
    qo_tile_indices_arg = _to_kernel_tensor(workspace.qo_tile_indices, cutlass.Int32, assumed_align=4)
    kv_tile_indices_arg = _to_kernel_tensor(workspace.kv_tile_indices, cutlass.Int32, assumed_align=4)
    o_indptr_arg = _to_kernel_tensor(workspace.o_indptr, cutlass.Int32, assumed_align=4)
    kv_chunk_size_arg = _to_kernel_tensor(workspace.kv_chunk_size_ptr, cutlass.Int32, assumed_align=4)
    block_valid_mask_arg = _to_kernel_tensor(workspace.block_valid_mask, cutlass.Int32, assumed_align=4)
    k_descale_arg = _to_kernel_tensor(k_descale, cutlass.Float32)
    v_descale_arg = _to_kernel_tensor(v_descale, cutlass.Float32)
    dummy_desc_ptrs = _dummy_fp8_tma_desc_ptrs(torch.cuda.current_device())
    k_tma_desc_ptrs = dummy_desc_ptrs
    v_tma_desc_ptrs = dummy_desc_ptrs
    k_tma_desc = None
    v_tma_desc = None
    if getattr(forward_kernel, "use_paged_kv_tma_fp8_raw_issue", False):
        cached_descs = _get_cached_fp8_tma_descs(
            workspace,
            k_cache=k_cache,
            v_cache=v_cache,
            plane_cols=forward_kernel.kv_tma_plane_head_dim,
            tile_rows=forward_kernel.stage_tile_rows,
        )
        if cached_descs is not None:
            k_tma_desc, v_tma_desc, k_tma_desc_ptrs, v_tma_desc_ptrs = cached_descs
        else:
            k_tma_desc = _encode_fp8_plane_tma_descriptors(
                k_cache,
                plane_cols=forward_kernel.kv_tma_plane_head_dim,
                tile_rows=forward_kernel.stage_tile_rows,
            )
            k_tma_desc_ptrs = _descriptor_row_ptrs(k_tma_desc)
            v_tma_desc = _encode_fp8_plane_tma_descriptors(
                v_cache,
                plane_cols=forward_kernel.kv_tma_plane_head_dim,
                tile_rows=forward_kernel.stage_tile_rows,
            )
            v_tma_desc_ptrs = _descriptor_row_ptrs(v_tma_desc)
            workspace._live_fp8_tma_desc_cache = {
                "key": (
                    int(k_cache.data_ptr()),
                    int(v_cache.data_ptr()),
                    tuple(k_cache.shape),
                    tuple(v_cache.shape),
                    forward_kernel.kv_tma_plane_head_dim,
                    forward_kernel.stage_tile_rows,
                ),
                "k_desc": k_tma_desc,
                "v_desc": v_tma_desc,
                "k_ptrs": k_tma_desc_ptrs,
                "v_ptrs": v_tma_desc_ptrs,
            }
    workspace._live_fp8_tma_descs = (k_tma_desc, v_tma_desc, k_tma_desc_ptrs, v_tma_desc_ptrs)
    k_tma_desc_arg = _to_kernel_tensor(k_tma_desc_ptrs, cutlass.Int64, assumed_align=8)
    v_tma_desc_arg = _to_kernel_tensor(v_tma_desc_ptrs, cutlass.Int64, assumed_align=8)

    stream = current_cuda_stream()
    if getattr(forward_kernel, "use_paged_kv_tma_fp8_raw_issue", False) and os.environ.get(
        "B12X_PAGED_KV_DEBUG_DUMP", ""
    ) == "PLANEWORDS":
        dump_kernel = _build_fp8_planewords_dump_kernel()
        debug_words = torch.zeros_like(forward_output.view(torch.int32))
        dump_kernel(
            page_table_arg,
            _to_kernel_tensor(debug_words, cutlass.Int32, assumed_align=4),
            v_tma_desc_arg,
            stream,
        )
        forward_output.view(torch.int32).copy_(debug_words)
        return forward_output, forward_lse
    forward_kernel(
        q_arg,
        k_cache_arg,
        v_cache_arg,
        page_table_arg,
        cache_seqlens_arg,
        cu_seqlens_q_arg,
        request_indices_arg,
        qo_tile_indices_arg,
        kv_tile_indices_arg,
        o_indptr_arg,
        kv_chunk_size_arg,
        block_valid_mask_arg,
        forward_output_arg,
        forward_lse_arg,
        k_descale_arg,
        v_descale_arg,
        k_tma_desc_arg,
        v_tma_desc_arg,
        stream,
    )

    if plan.split_kv:
        persistent_ctas = default_paged_persistent_ctas(
            total_rows=plan.total_q,
            num_heads=plan.num_q_heads,
            device=output.device,
        )
        merge_kernel = _build_merge_kernel(output.dtype, plan.head_dim_vo, persistent_ctas)
        tmp_output_arg = _to_kernel_tensor(workspace.tmp_output, _torch_to_cutlass_dtype(workspace.tmp_output.dtype))
        tmp_lse_arg = _to_kernel_tensor(workspace.tmp_lse, cutlass.Float32)
        merge_indptr_arg = _to_kernel_tensor(workspace.merge_indptr, cutlass.Int32, assumed_align=4)
        output_arg = _to_kernel_tensor(output, _torch_to_cutlass_dtype(output.dtype))
        lse_arg = _to_kernel_tensor(workspace.lse, cutlass.Float32)
        total_num_rows_arg = _to_kernel_tensor(workspace.total_num_rows_ptr, cutlass.Int32, assumed_align=4)
        merge_kernel(
            tmp_output_arg,
            tmp_lse_arg,
            merge_indptr_arg,
            output_arg,
            lse_arg,
            total_num_rows_arg,
            stream=stream,
        )

    return output[: plan.total_q], workspace.current_lse_view()


def clear_paged_caches() -> None:
    """Clear compiled-kernel caches for the primary paged backend."""
    _build_forward_kernel.cache_clear()
    _build_merge_kernel.cache_clear()
