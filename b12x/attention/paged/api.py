"""Public isolated API for the primary paged-attention backend."""

from __future__ import annotations

from functools import lru_cache

import cutlass
import torch
from cutlass.cute.runtime import from_dlpack

from b12x.cute.utils import current_cuda_stream

from .forward_paged import PagedForwardKernel
from .merge import PagedPersistentMergeKernel, default_paged_persistent_ctas
from .planner import PagedPlan
from .traits import PagedForwardTraits, select_paged_forward_traits_from_plan
from .workspace import PagedWorkspace


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


@lru_cache(maxsize=32)
def _build_forward_kernel(
    traits: PagedForwardTraits,
    split_kv: bool,
) -> PagedForwardKernel:
    return PagedForwardKernel(
        _torch_to_cutlass_dtype(traits.q_dtype),
        _torch_to_cutlass_dtype(traits.kv_dtype),
        _torch_to_cutlass_storage_dtype(traits.kv_dtype),
        _torch_to_cutlass_dtype(traits.o_dtype),
        traits=traits,
        split_kv=split_kv,
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
    page_table: torch.Tensor,
    cache_seqlens: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    *,
    workspace: PagedWorkspace,
    plan: PagedPlan,
    k_descale: torch.Tensor | None = None,
    v_descale: torch.Tensor | None = None,
    output: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if workspace.plan_key is not None and workspace.plan_key != plan.key:
        raise ValueError("workspace.plan_key does not match the provided plan")
    if plan.split_kv and (workspace.tmp_output is None or workspace.tmp_lse is None):
        raise ValueError("split-kv plan requires tmp_output and tmp_lse in the workspace")
    if output is None:
        output = workspace.output
    if k_descale is not None and k_descale.ndim == 2 and int(k_descale.shape[1]) == 1:
        k_descale = k_descale[:, 0].contiguous()
    if v_descale is not None and v_descale.ndim == 2 and int(v_descale.shape[1]) == 1:
        v_descale = v_descale[:, 0].contiguous()

    if (k_cache.dtype == torch.float8_e4m3fn or v_cache.dtype == torch.float8_e4m3fn) and (
        k_descale is None or v_descale is None
    ):
        raise ValueError("fp8 paged caches require k_descale and v_descale")

    traits = select_paged_forward_traits_from_plan(plan)
    forward_kernel = _build_forward_kernel(
        traits,
        plan.split_kv,
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

    return output, workspace.lse


def clear_paged_caches() -> None:
    """Clear compiled-kernel caches for the primary paged backend."""
    _build_forward_kernel.cache_clear()
    _build_merge_kernel.cache_clear()
