"""Public isolated API for the primary paged-attention backend."""

from __future__ import annotations

from functools import lru_cache
import os
from typing import Literal

import cutlass
import torch
from cutlass.cute.runtime import from_dlpack

from b12x.cute.utils import current_cuda_stream

from .forward_paged import PagedForwardKernel
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


@lru_cache(maxsize=64)
def _build_forward_kernel(
    traits: PagedForwardTraits,
    split_kv: bool,
    mxfp8_turbo: bool,
    enable_mxfp8_pv: bool,
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
    )


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
    forward_kernel = _build_forward_kernel(
        traits,
        plan.split_kv,
        mxfp8_turbo,
        enable_mxfp8_pv,
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

    stream = current_cuda_stream()
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
