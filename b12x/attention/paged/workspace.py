"""Dynamic workspace state for the primary paged-attention backend."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch

from .planner import PagedPlan, create_paged_plan, infer_paged_mode


def _paged_lse_storage_shape(total_q: int, num_q_heads: int) -> tuple[int, int]:
    return (num_q_heads, total_q)


def _copy_int_metadata(values: tuple[int, ...], *, device: torch.device) -> torch.Tensor:
    return torch.tensor(values, dtype=torch.int32, device=device)


def _canonical_device(device: torch.device | str) -> torch.device:
    device = torch.device(device)
    if device.type == "cuda" and device.index is None:
        return torch.device("cuda", torch.cuda.current_device())
    return device


def _shape_only_cuda_tensor(
    shape: tuple[int, ...],
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """Return a tiny CUDA tensor view with the requested logical shape.

    The paged planner only inspects shape, dtype, and device for the
    contract-side KV tensors; it never reads their data. Represent the
    contract with a single-element zero-stride view so graph workspaces do
    not allocate full shadow copies of the KV cache.
    """

    if any(dim <= 0 for dim in shape):
        raise ValueError(f"shape must be positive in every dimension, got {shape}")
    base = torch.empty(1, dtype=dtype, device=device)
    return base.as_strided(shape, (0,) * len(shape))


@dataclass(kw_only=True)
class PagedAttentionWorkspace:
    mode: Literal["decode", "extend"]
    device: torch.device
    dtype: torch.dtype
    kv_dtype: torch.dtype
    num_q_heads: int
    num_kv_heads: int
    head_dim_qk: int
    head_dim_vo: int
    attn_mode: Literal["default", "turbo"] | None = None
    page_size: int = 64
    use_cuda_graph: bool = False
    request_indices: torch.Tensor | None = None
    qo_tile_indices: torch.Tensor | None = None
    kv_tile_indices: torch.Tensor | None = None
    merge_indptr: torch.Tensor | None = None
    o_indptr: torch.Tensor | None = None
    kv_chunk_size_ptr: torch.Tensor | None = None
    total_num_rows_ptr: torch.Tensor | None = None
    block_valid_mask: torch.Tensor | None = None
    page_table: torch.Tensor | None = None
    cache_seqlens: torch.Tensor | None = None
    cu_seqlens_q: torch.Tensor | None = None
    lse: torch.Tensor | None = None
    tmp_output: torch.Tensor | None = None
    tmp_lse: torch.Tensor | None = None
    _plan_q: torch.Tensor | None = None
    _plan_k_cache: torch.Tensor | None = None
    _plan_v_cache: torch.Tensor | None = None
    _plan: PagedPlan | None = None

    @classmethod
    def for_contract(
        cls,
        *,
        mode: Literal["decode", "extend"],
        device: torch.device | str,
        dtype: torch.dtype,
        kv_dtype: torch.dtype,
        num_q_heads: int,
        num_kv_heads: int,
        head_dim_qk: int,
        head_dim_vo: int,
        page_size: int,
        max_total_q: int,
        num_cache_pages: int,
        use_cuda_graph: bool = False,
        attn_mode: Literal["default", "turbo"] | None = None,
    ) -> PagedAttentionWorkspace:
        device = _canonical_device(device)
        if max_total_q <= 0:
            raise ValueError("max_total_q must be positive")
        if num_cache_pages <= 0:
            raise ValueError("num_cache_pages must be positive")
        plan_q = torch.empty((max_total_q, num_q_heads, head_dim_qk), dtype=dtype, device=device)
        plan_k_cache = _shape_only_cuda_tensor(
            (num_cache_pages, page_size, num_kv_heads, head_dim_qk),
            dtype=kv_dtype,
            device=device,
        )
        plan_v_cache = _shape_only_cuda_tensor(
            (num_cache_pages, page_size, num_kv_heads, head_dim_vo),
            dtype=kv_dtype,
            device=device,
        )
        return cls(
            mode=mode,
            device=device,
            dtype=dtype,
            kv_dtype=kv_dtype,
            num_q_heads=num_q_heads,
            num_kv_heads=num_kv_heads,
            head_dim_qk=head_dim_qk,
            head_dim_vo=head_dim_vo,
            attn_mode=attn_mode,
            page_size=page_size,
            use_cuda_graph=use_cuda_graph,
            _plan_q=plan_q,
            _plan_k_cache=plan_k_cache,
            _plan_v_cache=plan_v_cache,
        )

    @classmethod
    def for_tensors(
        cls,
        *,
        mode: Literal["decode", "extend"],
        q: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        use_cuda_graph: bool = False,
        attn_mode: Literal["default", "turbo"] | None = None,
    ) -> PagedAttentionWorkspace:
        if q.ndim != 3:
            raise ValueError(f"q must have shape [total_q, q_heads, head_dim], got {tuple(q.shape)}")
        if k_cache.ndim != 4 or v_cache.ndim != 4:
            raise ValueError("k_cache and v_cache must be rank-4 paged tensors")
        return cls.for_contract(
            mode=mode,
            device=q.device,
            dtype=q.dtype,
            kv_dtype=k_cache.dtype,
            num_q_heads=int(q.shape[1]),
            num_kv_heads=int(k_cache.shape[2]),
            head_dim_qk=int(q.shape[2]),
            head_dim_vo=int(v_cache.shape[3]),
            page_size=int(k_cache.shape[1]),
            max_total_q=int(q.shape[0]),
            num_cache_pages=int(k_cache.shape[0]),
            use_cuda_graph=use_cuda_graph,
            attn_mode=attn_mode,
        )

    @property
    def prepared(self) -> bool:
        return self._plan is not None

    @property
    def plan(self) -> PagedPlan:
        if self._plan is None:
            raise RuntimeError("paged workspace has not been prepared")
        return self._plan

    @property
    def active_total_q(self) -> int:
        return self.plan.total_q

    @property
    def active_batch(self) -> int:
        return self.plan.page_table_shape[0]

    @property
    def total_q_capacity(self) -> int:
        if self._plan_q is None:
            raise RuntimeError("paged workspace planning contract is not initialized")
        return int(self._plan_q.shape[0])

    def prepare(
        self,
        page_table: torch.Tensor,
        cache_seqlens: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        *,
        fixed_split_size: int | None = None,
        disable_split_kv: bool = False,
    ) -> PagedAttentionWorkspace:
        inferred_mode = infer_paged_mode(cu_seqlens_q)
        if inferred_mode != self.mode:
            raise ValueError(f"workspace mode {self.mode} does not match prepared mode {inferred_mode}")
        active_total_q = int(cu_seqlens_q[-1].item())
        self._ensure_plan_contract(active_total_q)
        assert self._plan_q is not None
        assert self._plan_k_cache is not None
        assert self._plan_v_cache is not None
        if active_total_q <= 0 or active_total_q > int(self._plan_q.shape[0]):
            raise ValueError(
                f"cu_seqlens_q implies total_q={active_total_q}, but workspace capacity is {int(self._plan_q.shape[0])}"
            )

        plan = create_paged_plan(
            self._plan_q[:active_total_q],
            self._plan_k_cache,
            self._plan_v_cache,
            page_table,
            cache_seqlens,
            cu_seqlens_q,
            mode=self.mode,
            fixed_split_size=-1 if fixed_split_size is None else int(fixed_split_size),
            disable_split_kv=disable_split_kv,
            enable_cuda_graph=False,
            graph_chunk_policy=self.use_cuda_graph,
        )
        self._ensure_capacity(plan)
        self._copy_runtime_metadata(page_table, cache_seqlens, cu_seqlens_q)
        self._copy_plan_metadata(plan)
        self._plan = plan
        return self

    def _ensure_plan_contract(self, active_total_q: int) -> None:
        if self._plan_q is None or self._plan_k_cache is None or self._plan_v_cache is None:
            raise RuntimeError("paged workspace planning contract is not initialized")
        if active_total_q <= int(self._plan_q.shape[0]):
            return
        if self.use_cuda_graph:
            raise ValueError(
                "graph-mode paged workspace capacity exceeded; construct a larger workspace or capture a larger graph bucket"
            )
        self._plan_q = torch.empty(
            (active_total_q, self.num_q_heads, self.head_dim_qk),
            dtype=self.dtype,
            device=self.device,
        )

    @torch._dynamo.disable
    def run(
        self,
        q: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        *,
        output: torch.Tensor,
        k_descale: torch.Tensor | None = None,
        v_descale: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        from .api import paged_attention_forward

        out, lse = paged_attention_forward(
            q,
            k_cache,
            v_cache,
            workspace=self,
            output=output,
            k_descale=k_descale,
            v_descale=v_descale,
        )
        return out, lse

    def current_lse_view(self) -> torch.Tensor:
        if self.lse is None:
            raise RuntimeError("workspace has not been prepared")
        return self.lse[:, : self.active_total_q].transpose(0, 1)

    def _validate_static_shapes(
        self,
        q: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
    ) -> None:
        if q.device != self.device or k_cache.device != self.device or v_cache.device != self.device:
            raise ValueError("workspace inputs must stay on the workspace device")
        if q.dtype != self.dtype:
            raise TypeError(f"workspace expects q dtype {self.dtype}, got {q.dtype}")
        if k_cache.dtype != self.kv_dtype or v_cache.dtype != self.kv_dtype:
            raise TypeError(f"workspace expects kv dtype {self.kv_dtype}, got {k_cache.dtype}/{v_cache.dtype}")
        if tuple(q.shape[1:]) != (self.num_q_heads, self.head_dim_qk):
            raise ValueError(
                "q shape does not match the workspace contract: "
                f"expected (*, {self.num_q_heads}, {self.head_dim_qk}), got {tuple(q.shape)}"
            )
        if int(k_cache.shape[1]) != self.page_size or int(v_cache.shape[1]) != self.page_size:
            raise ValueError(f"workspace expects page_size={self.page_size}")
        if int(k_cache.shape[2]) != self.num_kv_heads or int(v_cache.shape[2]) != self.num_kv_heads:
            raise ValueError("kv head count does not match the workspace contract")
        if int(k_cache.shape[3]) != self.head_dim_qk:
            raise ValueError("k_cache head_dim does not match the workspace contract")
        if int(v_cache.shape[3]) != self.head_dim_vo:
            raise ValueError("v_cache head_dim does not match the workspace contract")

    def _ensure_capacity(self, plan: PagedPlan) -> None:
        work_items_needed = int(plan.new_batch_size)
        total_q_needed = int(plan.total_q)
        batch_needed = int(plan.page_table_shape[0])
        page_table_width_needed = int(plan.page_table_shape[1])
        partial_rows_needed = int(plan.total_num_partial_rows) if plan.split_kv else 0

        work_items_capacity = 0 if self.request_indices is None else int(self.request_indices.shape[0])
        total_q_capacity = 0 if self.lse is None else int(self.lse.shape[1])
        batch_capacity = 0 if self.o_indptr is None else int(self.o_indptr.shape[0] - 1)
        page_table_width_capacity = 0 if self.page_table is None else int(self.page_table.shape[1])
        partial_rows_capacity = 0 if self.tmp_output is None else int(self.tmp_output.shape[0])

        needs_growth = (
            work_items_needed > work_items_capacity
            or total_q_needed > total_q_capacity
            or batch_needed > batch_capacity
            or page_table_width_needed > page_table_width_capacity
            or partial_rows_needed > partial_rows_capacity
        )
        if not needs_growth:
            return
        if self.use_cuda_graph and self.request_indices is not None:
            raise ValueError(
                "graph-mode paged workspace capacity exceeded; construct a larger workspace or capture a larger graph bucket"
            )

        work_items_capacity = max(work_items_capacity, work_items_needed)
        total_q_capacity = max(total_q_capacity, total_q_needed)
        batch_capacity = max(batch_capacity, batch_needed)
        page_table_width_capacity = max(page_table_width_capacity, page_table_width_needed)
        partial_rows_capacity = max(partial_rows_capacity, partial_rows_needed)

        self.request_indices = torch.empty(work_items_capacity, dtype=torch.int32, device=self.device)
        self.qo_tile_indices = torch.empty(work_items_capacity, dtype=torch.int32, device=self.device)
        self.kv_tile_indices = torch.empty(work_items_capacity, dtype=torch.int32, device=self.device)
        self.block_valid_mask = torch.empty(work_items_capacity, dtype=torch.int32, device=self.device)
        self.page_table = torch.empty(
            (batch_capacity, page_table_width_capacity), dtype=torch.int32, device=self.device
        )
        self.cache_seqlens = torch.empty(batch_capacity, dtype=torch.int32, device=self.device)
        self.cu_seqlens_q = torch.empty(batch_capacity + 1, dtype=torch.int32, device=self.device)
        self.merge_indptr = torch.empty(total_q_capacity + 1, dtype=torch.int32, device=self.device)
        self.o_indptr = torch.empty(batch_capacity + 1, dtype=torch.int32, device=self.device)
        self.kv_chunk_size_ptr = torch.empty(1, dtype=torch.int32, device=self.device)
        self.total_num_rows_ptr = torch.empty(1, dtype=torch.int32, device=self.device)
        self.lse = torch.empty(
            _paged_lse_storage_shape(total_q_capacity, self.num_q_heads),
            dtype=torch.float32,
            device=self.device,
        )
        if partial_rows_capacity > 0:
            self.tmp_output = torch.empty(
                (partial_rows_capacity, self.num_q_heads, self.head_dim_vo),
                dtype=self.dtype,
                device=self.device,
            )
            self.tmp_lse = torch.empty(
                (partial_rows_capacity, self.num_q_heads),
                dtype=torch.float32,
                device=self.device,
            )
        else:
            self.tmp_output = None
            self.tmp_lse = None

    def _copy_runtime_metadata(
        self,
        page_table: torch.Tensor,
        cache_seqlens: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
    ) -> None:
        assert self.page_table is not None
        assert self.cache_seqlens is not None
        assert self.cu_seqlens_q is not None

        page_table_i32 = page_table if page_table.dtype == torch.int32 else page_table.to(torch.int32)
        cache_seqlens_i32 = (
            cache_seqlens if cache_seqlens.dtype == torch.int32 else cache_seqlens.to(torch.int32)
        )
        cu_seqlens_q_i32 = (
            cu_seqlens_q if cu_seqlens_q.dtype == torch.int32 else cu_seqlens_q.to(torch.int32)
        )

        self.page_table[: page_table_i32.shape[0], : page_table_i32.shape[1]].copy_(page_table_i32)
        self.cache_seqlens[: cache_seqlens_i32.shape[0]].copy_(cache_seqlens_i32)
        self.cu_seqlens_q[: cu_seqlens_q_i32.shape[0]].copy_(cu_seqlens_q_i32)

    def _copy_plan_metadata(self, plan: PagedPlan) -> None:
        assert self.request_indices is not None
        assert self.qo_tile_indices is not None
        assert self.kv_tile_indices is not None
        assert self.merge_indptr is not None
        assert self.o_indptr is not None
        assert self.kv_chunk_size_ptr is not None
        assert self.total_num_rows_ptr is not None
        assert self.block_valid_mask is not None

        request_indices = _copy_int_metadata(plan.request_indices, device=self.device)
        qo_tile_indices = _copy_int_metadata(plan.qo_tile_indices, device=self.device)
        kv_tile_indices = _copy_int_metadata(plan.kv_tile_indices, device=self.device)
        merge_indptr = _copy_int_metadata(plan.merge_indptr, device=self.device)
        o_indptr = _copy_int_metadata(plan.o_indptr, device=self.device)
        block_valid_mask = torch.tensor(plan.block_valid_mask, dtype=torch.int32, device=self.device)

        self.request_indices.zero_()
        self.qo_tile_indices.zero_()
        self.kv_tile_indices.zero_()
        self.request_indices[: request_indices.shape[0]].copy_(request_indices)
        self.qo_tile_indices[: qo_tile_indices.shape[0]].copy_(qo_tile_indices)
        self.kv_tile_indices[: kv_tile_indices.shape[0]].copy_(kv_tile_indices)
        self.merge_indptr[: merge_indptr.shape[0]].copy_(merge_indptr)
        self.o_indptr[: o_indptr.shape[0]].copy_(o_indptr)
        self.block_valid_mask.zero_()
        self.block_valid_mask[: block_valid_mask.shape[0]].copy_(block_valid_mask)
        self.kv_chunk_size_ptr[0] = int(plan.kv_chunk_size if plan.split_kv else 0)
        self.total_num_rows_ptr[0] = int(plan.total_q)
