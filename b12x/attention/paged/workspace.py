"""Workspace allocation for the primary paged-attention backend."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .planner import PagedPlan


def _paged_lse_storage_shape(total_q: int, num_q_heads: int) -> tuple[int, int]:
    return (num_q_heads, total_q)


@dataclass(kw_only=True)
class PagedWorkspace:
    device: torch.device
    dtype: torch.dtype
    kv_dtype: torch.dtype
    output: torch.Tensor
    lse: torch.Tensor
    request_indices: torch.Tensor
    qo_tile_indices: torch.Tensor
    kv_tile_indices: torch.Tensor
    merge_indptr: torch.Tensor
    o_indptr: torch.Tensor
    kv_chunk_size_ptr: torch.Tensor
    total_num_rows_ptr: torch.Tensor
    block_valid_mask: torch.Tensor | None = None
    tmp_output: torch.Tensor | None = None
    tmp_lse: torch.Tensor | None = None
    plan_key: object | None = None


def _copy_int_metadata(values: tuple[int, ...], *, device: torch.device) -> torch.Tensor:
    return torch.tensor(values, dtype=torch.int32, device=device)


def allocate_paged_workspace_for_plan(plan: PagedPlan) -> PagedWorkspace:
    device = plan.device
    kv_chunk_size_runtime = min(int(plan.kv_chunk_size), 2**31 - 1)
    output = torch.empty(
        (plan.total_q, plan.num_q_heads, plan.head_dim_vo), dtype=plan.dtype, device=device
    )
    lse = torch.empty(
        _paged_lse_storage_shape(plan.total_q, plan.num_q_heads), dtype=torch.float32, device=device
    )
    request_indices = _copy_int_metadata(plan.request_indices, device=device)
    qo_tile_indices = _copy_int_metadata(plan.qo_tile_indices, device=device)
    kv_tile_indices = _copy_int_metadata(plan.kv_tile_indices, device=device)
    merge_indptr = _copy_int_metadata(plan.merge_indptr, device=device)
    o_indptr = _copy_int_metadata(plan.o_indptr, device=device)
    kv_chunk_size_ptr = torch.tensor([kv_chunk_size_runtime], dtype=torch.int32, device=device)
    total_num_rows_ptr = torch.tensor([plan.total_q], dtype=torch.int32, device=device)
    block_valid_mask = None
    tmp_output = None
    tmp_lse = None
    if plan.split_kv:
        tmp_output = torch.empty(
            (plan.total_num_partial_rows, plan.num_q_heads, plan.head_dim_vo),
            dtype=plan.dtype,
            device=device,
        )
        tmp_lse = torch.empty(
            (plan.total_num_partial_rows, plan.num_q_heads),
            dtype=torch.float32,
            device=device,
        )
        block_valid_mask = torch.tensor(plan.block_valid_mask, dtype=torch.bool, device=device)
    return PagedWorkspace(
        device=device,
        dtype=plan.dtype,
        kv_dtype=plan.kv_dtype,
        output=output,
        lse=lse,
        request_indices=request_indices,
        qo_tile_indices=qo_tile_indices,
        kv_tile_indices=kv_tile_indices,
        merge_indptr=merge_indptr,
        o_indptr=o_indptr,
        kv_chunk_size_ptr=kv_chunk_size_ptr,
        total_num_rows_ptr=total_num_rows_ptr,
        block_valid_mask=block_valid_mask,
        tmp_output=tmp_output,
        tmp_lse=tmp_lse,
        plan_key=plan.key,
    )
