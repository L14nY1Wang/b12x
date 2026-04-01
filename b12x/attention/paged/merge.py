"""Standalone persistent merge kernel for the primary paged backend.

This module intentionally does not share implementation with the archived
`PagedAttentionCombineKernel`. It follows FlashInfer's split-state merge model:

- partial outputs are normalized attention outputs,
- partial scores are base-2 log-sum-exp values,
- CTAs walk `(row, head)` work persistently,
- each CTA cooperatively folds multiple partial states into one final state.

The first version here ports the control flow and state arithmetic faithfully
but keeps the partial-vector loads direct from global memory. The cp.async
staging layer can be added on top of this contract without changing the API.
"""

from __future__ import annotations

from typing import Type

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass._mlir.dialects import llvm

from cutlass import Float32, Int32, const_expr
from cutlass.cutlass_dsl import Int64, T, dsl_user_op

from b12x.attention import utils as attention_utils
from b12x.cute.fp4 import get_ptr_as_int64, shared_ptr_to_u32


@dsl_user_op
def _cp_async_load_128b(smem_addr: Int32, gmem_addr: Int64, *, loc=None, ip=None):
    llvm.inline_asm(
        None,
        [
            Int32(smem_addr).ir_value(loc=loc, ip=ip),
            Int64(gmem_addr).ir_value(loc=loc, ip=ip),
        ],
        "cp.async.cg.shared.global.L2::128B [$0], [$1], 16;",
        "r,l",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


@dsl_user_op
def _exp2_approx_ftz_f32(a: Float32, *, loc=None, ip=None) -> Float32:
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [Float32(a).ir_value(loc=loc, ip=ip)],
            "ex2.approx.ftz.f32 $0, $1;",
            "=f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@dsl_user_op
def _log2_approx_ftz_f32(a: Float32, *, loc=None, ip=None) -> Float32:
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [Float32(a).ir_value(loc=loc, ip=ip)],
            "lg2.approx.ftz.f32 $0, $1;",
            "=f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


def _ceil_div(x: int, y: int) -> int:
    return (x + y - 1) // y


def default_paged_persistent_ctas(
    *,
    total_rows: int,
    num_heads: int,
    device: torch.device | int | None = None,
) -> int:
    if device is None:
        device = torch.cuda.current_device()
    num_sms = int(torch.cuda.get_device_properties(device).multi_processor_count)
    total_work = max(int(total_rows) * int(num_heads), 1)
    # Match FlashInfer's persistent merge launch shape more closely: size the
    # resident grid by useful blocks-per-SM rather than pinning it to 1 CTA/SM.
    blocks_per_sm = min(3, _ceil_div(total_work, num_sms))
    return int(num_sms * max(blocks_per_sm, 1))


@cute.jit
def _state_init(
    state_o: cute.Tensor,
):
    state_o.fill(0.0)
    return Float32(-Float32.inf), Float32(1.0)


@cute.jit
def _state_get_lse_base2(
    state_m: Float32,
    state_d: Float32,
) -> Float32:
    return Float32(state_m + _log2_approx_ftz_f32(Float32(state_d)))


@cute.jit
def _state_merge(
    state_o: cute.Tensor,
    state_m: Float32,
    state_d: Float32,
    other_o: cute.Tensor,
    other_m: Float32,
    other_d: Float32,
) -> tuple[Float32, Float32]:
    prev_m = state_m
    prev_d = state_d
    if prev_m == -Float32.inf:
        if other_m == -Float32.inf:
            state_m = Float32(prev_m)
            state_d = Float32(prev_d)
        else:
            for vec_idx in cutlass.range_constexpr(cute.size(state_o.shape)):
                state_o[vec_idx] = other_o[vec_idx]
            state_m = Float32(other_m)
            state_d = Float32(other_d)
    elif other_m == -Float32.inf:
        state_m = Float32(prev_m)
        state_d = Float32(prev_d)
    else:
        state_m = attention_utils.fmax(prev_m, other_m)
        prev_scale = _exp2_approx_ftz_f32(prev_m - state_m)
        other_scale = _exp2_approx_ftz_f32(other_m - state_m)
        state_d = Float32(prev_d * prev_scale + other_d * other_scale)
        for vec_idx in cutlass.range_constexpr(cute.size(state_o.shape)):
            state_o[vec_idx] = state_o[vec_idx] * prev_scale + other_o[vec_idx] * other_scale
    return Float32(state_m), state_d


@cute.jit
def _state_merge_normalized_lse_base2(
    state_o: cute.Tensor,
    state_m: Float32,
    state_d: Float32,
    other_o: cute.Tensor,
    other_lse: Float32,
) -> tuple[Float32, Float32]:
    return _state_merge(
        state_o,
        state_m,
        state_d,
        other_o,
        other_lse,
        Float32(1.0),
    )


@cute.jit
def _state_normalize(
    state_o: cute.Tensor,
    state_d: Float32,
):
    inv_d = cute.arch.rcp_approx(Float32(state_d))
    for vec_idx in cutlass.range_constexpr(cute.size(state_o.shape)):
        state_o[vec_idx] = state_o[vec_idx] * inv_d


@cute.jit
def _threadblock_sync_state(
    state_o: cute.Tensor,
    state_m: Float32,
    state_d: Float32,
    s_partial: cute.Tensor,
    s_lse: cute.Tensor,
    *,
    vec_size: cutlass.Constexpr[int],
    bdy: cutlass.Constexpr[int],
) -> tuple[Float32, Float32]:
    tx, ty, _ = cute.arch.thread_idx()
    base_k = tx * vec_size
    _state_normalize(state_o, state_d)
    for vec_idx in cutlass.range_constexpr(vec_size):
        s_partial[ty, base_k + vec_idx] = state_o[vec_idx]
    s_lse[ty] = _state_get_lse_base2(state_m, state_d)
    state_m, state_d = _state_init(state_o)
    cute.arch.sync_threads()

    for iter_idx in cutlass.range_constexpr(bdy):
        other_o = cute.make_rmem_tensor(cute.make_layout((vec_size,), stride=(1,)), Float32)
        for vec_idx in cutlass.range_constexpr(vec_size):
            other_o[vec_idx] = s_partial[iter_idx, base_k + vec_idx]
        state_m, state_d = _state_merge_normalized_lse_base2(
            state_o,
            state_m,
            state_d,
            other_o,
            s_lse[iter_idx],
        )
    return state_m, state_d


@cute.jit
def _merge_async_slot(
    *,
    iter_idx: Int32,
    start_idx,
    num_heads,
    head_idx,
    num_index_sets,
    num_stage_iters,
    base_k,
    head_dim,
    vec_size: cutlass.Constexpr[int],
    bdx: cutlass.Constexpr[int],
    bdy: cutlass.Constexpr[int],
    num_smem_stages: cutlass.Constexpr[int],
    s_stage_partial: cute.Tensor,
    s_stage_lse: cute.Tensor,
    mV_partial: cute.Tensor,
    mLSE_partial: cute.Tensor,
    state_o: cute.Tensor,
    state_m: Float32,
    state_d: Float32,
    slot: cutlass.Constexpr[int],
) -> tuple[Float32, Float32]:
    tx, ty, _ = cute.arch.thread_idx()
    cur_iter = iter_idx + Int32(slot)
    if cur_iter < num_stage_iters:
        if cur_iter % bdx == 0:
            lse_linear_idx = cur_iter * bdy + ty * bdx + tx
            s_stage_lse[ty * bdx + tx] = (
                mLSE_partial[start_idx + lse_linear_idx, head_idx]
                if lse_linear_idx < num_index_sets
                else Float32(0.0)
            )
            cute.arch.sync_threads()

        cute.arch.cp_async_wait_group(num_smem_stages - 1)
        cute.arch.sync_threads()

        if cur_iter * bdy + ty < num_index_sets:
            other_o = cute.make_rmem_tensor(
                cute.make_layout((vec_size,), stride=(1,)),
                Float32,
            )
            for vec_idx in cutlass.range_constexpr(vec_size):
                other_o[vec_idx] = s_stage_partial[
                    cur_iter % num_smem_stages, ty, base_k + vec_idx
                ].to(Float32)
            state_m, state_d = _state_merge_normalized_lse_base2(
                state_o,
                state_m,
                state_d,
                other_o,
                s_stage_lse[(cur_iter % bdx) * bdy + ty],
            )

        cute.arch.sync_threads()
        next_linear_idx = (cur_iter + num_smem_stages) * bdy + ty
        smem_addr = shared_ptr_to_u32(
            s_stage_partial.iterator
            + Int32(((cur_iter % num_smem_stages) * bdy + ty) * head_dim + base_k)
        )
        if next_linear_idx < num_index_sets:
            partial_idx = start_idx + next_linear_idx
            gmem_addr = get_ptr_as_int64(
                mV_partial,
                (Int64(partial_idx) * Int64(num_heads) + Int64(head_idx)) * Int64(head_dim) + Int64(base_k),
            )
            _cp_async_load_128b(smem_addr, gmem_addr)
        cute.arch.cp_async_commit_group()
    return state_m, state_d


class PagedPersistentMergeKernel:
    """Faithful row/head-persistent merge for the current paged backend path.

    The kernel expects:
    - `mV_partial`: `(nnz_partial_rows, num_heads, head_dim)`
    - `mLSE_partial`: `(nnz_partial_rows, num_heads)` in base-2 log domain
    - `mMergeIndptr`: `(max_total_rows + 1,)`
    - `mO`: `(max_total_rows, num_heads, head_dim)`
    - `mLSE`: `(num_heads, max_total_rows)` in base-2 log domain
    - `mTotalRowsPtr`: optional `(1,)` dynamic sequence length for graph replay
    """

    def __init__(
        self,
        dtype: Type[cutlass.Numeric],
        dtype_partial: Type[cutlass.Numeric],
        *,
        head_dim: int,
        vec_size: int = 8,
        bdx: int = 32,
        bdy: int = 4,
        num_smem_stages: int = 4,
        persistent_ctas: int | None = None,
    ):
        self.dtype = dtype
        self.dtype_partial = dtype_partial
        self.head_dim = head_dim
        self.vec_size = vec_size
        self.bdx = bdx
        self.bdy = bdy
        self.num_smem_stages = num_smem_stages
        self.persistent_ctas = (
            int(persistent_ctas) if persistent_ctas is not None else 0
        )

    @staticmethod
    def can_implement(
        dtype: Type[cutlass.Numeric],
        dtype_partial: Type[cutlass.Numeric],
        *,
        head_dim: int,
        vec_size: int,
        bdx: int,
        bdy: int,
        num_smem_stages: int,
        persistent_ctas: int,
    ) -> bool:
        if dtype not in (cutlass.Float16, cutlass.BFloat16, cutlass.Float32):
            return False
        if dtype_partial not in (cutlass.Float16, cutlass.BFloat16, cutlass.Float32):
            return False
        if head_dim <= 0 or vec_size <= 0 or bdx <= 0 or bdy <= 0 or num_smem_stages <= 0:
            return False
        if head_dim != bdx * vec_size:
            return False
        if bdx % 32 != 0:
            return False
        if persistent_ctas <= 0:
            return False
        return True

    def _get_shared_storage_cls(self):
        stage_partial_storage = cute.struct.MemRange[
            self.dtype_partial, int(self.num_smem_stages * self.bdy * self.head_dim)
        ]
        stage_lse_storage = cute.struct.MemRange[cutlass.Float32, int(self.bdx * self.bdy)]
        partial_storage = cute.struct.MemRange[cutlass.Float32, int(self.bdy * self.head_dim)]
        lse_storage = cute.struct.MemRange[cutlass.Float32, int(self.bdy)]

        class SharedStorage:
            pass

        SharedStorage.__annotations__ = {
            "sStagePartial": stage_partial_storage,
            "sStageLSE": stage_lse_storage,
            "sPartial": partial_storage,
            "sLSE": lse_storage,
        }

        return cute.struct(SharedStorage)

    @cute.jit
    def __call__(
        self,
        mV_partial: cute.Tensor,
        mLSE_partial: cute.Tensor,
        mMergeIndptr: cute.Tensor,
        mO: cute.Tensor,
        mLSE: cute.Tensor,
        mTotalRowsPtr: cute.Tensor | None,
        stream: cuda.CUstream,
    ):
        if const_expr(len(mV_partial.shape) != 3):
            raise ValueError("mV_partial must have shape (nnz_partial_rows, num_heads, head_dim)")
        if const_expr(len(mLSE_partial.shape) != 2):
            raise ValueError("mLSE_partial must have shape (nnz_partial_rows, num_heads)")
        if const_expr(len(mMergeIndptr.shape) != 1):
            raise ValueError("mMergeIndptr must have shape (max_total_rows + 1,)")
        if const_expr(len(mO.shape) != 3):
            raise ValueError("mO must have shape (max_total_rows, num_heads, head_dim)")
        if const_expr(len(mLSE.shape) != 2):
            raise ValueError("mLSE must have shape (num_heads, max_total_rows)")
        if const_expr(mTotalRowsPtr is not None and len(mTotalRowsPtr.shape) != 1):
            raise ValueError("mTotalRowsPtr must have shape (1,)")
        if const_expr(mV_partial.element_type != self.dtype_partial):
            raise TypeError("mV_partial dtype must match dtype_partial")
        if const_expr(mO.element_type != self.dtype):
            raise TypeError("mO dtype must match dtype")
        if const_expr(mLSE_partial.element_type != Float32 or mLSE.element_type != Float32):
            raise TypeError("mLSE tensors must be Float32")
        if const_expr(
            not self.can_implement(
                self.dtype,
                self.dtype_partial,
                head_dim=self.head_dim,
                vec_size=self.vec_size,
                bdx=self.bdx,
                bdy=self.bdy,
                num_smem_stages=self.num_smem_stages,
                persistent_ctas=self.persistent_ctas,
            )
        ):
            raise TypeError("paged merge kernel configuration is not supported")

        SharedStorage = self._get_shared_storage_cls()
        self.kernel(
            mV_partial,
            mLSE_partial,
            mMergeIndptr,
            mO,
            mLSE,
            mTotalRowsPtr,
        ).launch(
            grid=(self.persistent_ctas, 1, 1),
            block=[self.bdx, self.bdy, 1],
            smem=SharedStorage.size_in_bytes(),
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        mV_partial: cute.Tensor,
        mLSE_partial: cute.Tensor,
        mMergeIndptr: cute.Tensor,
        mO: cute.Tensor,
        mLSE: cute.Tensor,
        mTotalRowsPtr: cute.Tensor | None,
    ):
        tx, ty, _ = cute.arch.thread_idx()
        cta_id, _, _ = cute.arch.block_idx()
        num_ctas, _, _ = cute.arch.grid_dim()
        max_total_rows = mO.shape[0]
        num_heads = mO.shape[1]
        head_dim = self.head_dim
        total_rows = mTotalRowsPtr[0] if const_expr(mTotalRowsPtr is not None) else max_total_rows

        smem = cutlass.utils.SmemAllocator()
        SharedStorage = self._get_shared_storage_cls()
        storage = smem.allocate(SharedStorage)
        s_stage_partial = storage.sStagePartial.get_tensor(
            cute.make_layout(
                (self.num_smem_stages, self.bdy, head_dim),
                stride=(self.bdy * head_dim, head_dim, 1),
            )
        )
        s_stage_lse = storage.sStageLSE.get_tensor(
            cute.make_layout((self.bdx * self.bdy,), stride=(1,))
        )
        s_partial = storage.sPartial.get_tensor(
            cute.make_layout((self.bdy, head_dim), stride=(head_dim, 1))
        )
        s_lse = storage.sLSE.get_tensor(cute.make_layout((self.bdy,), stride=(1,)))

        cute.arch.griddepcontrol_wait()

        total_work = total_rows * num_heads
        base_k = tx * self.vec_size
        work_linear_idx = cta_id
        while work_linear_idx < total_work:
            cute.arch.sync_threads()

            row_idx = work_linear_idx // num_heads
            head_idx = work_linear_idx % num_heads
            start_idx = mMergeIndptr[row_idx]
            end_idx = mMergeIndptr[row_idx + 1]
            num_index_sets = end_idx - start_idx

            if num_index_sets == 0:
                for vec_idx in cutlass.range_constexpr(self.vec_size):
                    mO[row_idx, head_idx, base_k + vec_idx] = self.dtype(0.0)
                if tx == 0 and ty == 0:
                    mLSE[head_idx, row_idx] = -Float32.inf
            elif num_index_sets == 1:
                for vec_idx in cutlass.range_constexpr(self.vec_size):
                    mO[row_idx, head_idx, base_k + vec_idx] = mV_partial[
                        start_idx, head_idx, base_k + vec_idx
                    ].to(self.dtype)
                if tx == 0 and ty == 0:
                    mLSE[head_idx, row_idx] = mLSE_partial[start_idx, head_idx]
            else:
                state_o = cute.make_rmem_tensor(
                    cute.make_layout((self.vec_size,), stride=(1,)),
                    Float32,
                )
                state_m, state_d = _state_init(state_o)
                bytes_per_vec = self.vec_size * self.dtype_partial.width // 8
                can_stage_async = bytes_per_vec == 16
                if can_stage_async:
                    for stage_idx in cutlass.range_constexpr(self.num_smem_stages):
                        staged_linear_idx = stage_idx * self.bdy + ty
                        smem_addr = shared_ptr_to_u32(
                            s_stage_partial.iterator
                            + Int32((stage_idx * self.bdy + ty) * head_dim + base_k)
                        )
                        if staged_linear_idx < num_index_sets:
                            partial_idx = start_idx + staged_linear_idx
                            gmem_addr = get_ptr_as_int64(
                                mV_partial,
                                (Int64(partial_idx) * Int64(num_heads) + Int64(head_idx)) * Int64(head_dim)
                                + Int64(base_k),
                            )
                            _cp_async_load_128b(smem_addr, gmem_addr)
                        cute.arch.cp_async_commit_group()

                    num_stage_iters = (num_index_sets + self.bdy - 1) // self.bdy
                    iter_idx = Int32(0)
                    while iter_idx < num_stage_iters:
                        state_m, state_d = _merge_async_slot(
                            iter_idx=iter_idx,
                            start_idx=start_idx,
                            num_heads=num_heads,
                            head_idx=head_idx,
                            num_index_sets=num_index_sets,
                            num_stage_iters=num_stage_iters,
                            base_k=base_k,
                            head_dim=head_dim,
                            vec_size=self.vec_size,
                            bdx=self.bdx,
                            bdy=self.bdy,
                            num_smem_stages=self.num_smem_stages,
                            s_stage_partial=s_stage_partial,
                            s_stage_lse=s_stage_lse,
                            mV_partial=mV_partial,
                            mLSE_partial=mLSE_partial,
                            state_o=state_o,
                            state_m=state_m,
                            state_d=state_d,
                            slot=0,
                        )
                        state_m, state_d = _merge_async_slot(
                            iter_idx=iter_idx,
                            start_idx=start_idx,
                            num_heads=num_heads,
                            head_idx=head_idx,
                            num_index_sets=num_index_sets,
                            num_stage_iters=num_stage_iters,
                            base_k=base_k,
                            head_dim=head_dim,
                            vec_size=self.vec_size,
                            bdx=self.bdx,
                            bdy=self.bdy,
                            num_smem_stages=self.num_smem_stages,
                            s_stage_partial=s_stage_partial,
                            s_stage_lse=s_stage_lse,
                            mV_partial=mV_partial,
                            mLSE_partial=mLSE_partial,
                            state_o=state_o,
                            state_m=state_m,
                            state_d=state_d,
                            slot=1,
                        )
                        state_m, state_d = _merge_async_slot(
                            iter_idx=iter_idx,
                            start_idx=start_idx,
                            num_heads=num_heads,
                            head_idx=head_idx,
                            num_index_sets=num_index_sets,
                            num_stage_iters=num_stage_iters,
                            base_k=base_k,
                            head_dim=head_dim,
                            vec_size=self.vec_size,
                            bdx=self.bdx,
                            bdy=self.bdy,
                            num_smem_stages=self.num_smem_stages,
                            s_stage_partial=s_stage_partial,
                            s_stage_lse=s_stage_lse,
                            mV_partial=mV_partial,
                            mLSE_partial=mLSE_partial,
                            state_o=state_o,
                            state_m=state_m,
                            state_d=state_d,
                            slot=2,
                        )
                        state_m, state_d = _merge_async_slot(
                            iter_idx=iter_idx,
                            start_idx=start_idx,
                            num_heads=num_heads,
                            head_idx=head_idx,
                            num_index_sets=num_index_sets,
                            num_stage_iters=num_stage_iters,
                            base_k=base_k,
                            head_dim=head_dim,
                            vec_size=self.vec_size,
                            bdx=self.bdx,
                            bdy=self.bdy,
                            num_smem_stages=self.num_smem_stages,
                            s_stage_partial=s_stage_partial,
                            s_stage_lse=s_stage_lse,
                            mV_partial=mV_partial,
                            mLSE_partial=mLSE_partial,
                            state_o=state_o,
                            state_m=state_m,
                            state_d=state_d,
                            slot=3,
                        )
                        iter_idx += Int32(4)

                    cute.arch.cp_async_wait_group(0)
                    cute.arch.sync_threads()
                else:
                    partial_idx = start_idx + ty
                    while partial_idx < end_idx:
                        other_o = cute.make_rmem_tensor(
                            cute.make_layout((self.vec_size,), stride=(1,)),
                            Float32,
                        )
                        for vec_idx in cutlass.range_constexpr(self.vec_size):
                            other_o[vec_idx] = mV_partial[partial_idx, head_idx, base_k + vec_idx].to(
                                Float32
                            )
                        state_m, state_d = _state_merge_normalized_lse_base2(
                            state_o,
                            state_m,
                            state_d,
                            other_o,
                            mLSE_partial[partial_idx, head_idx],
                        )
                        partial_idx += self.bdy

                state_m, state_d = _threadblock_sync_state(
                    state_o,
                    state_m,
                    state_d,
                    s_partial,
                    s_lse,
                    vec_size=self.vec_size,
                    bdy=self.bdy,
                )
                _state_normalize(state_o, state_d)
                for vec_idx in cutlass.range_constexpr(self.vec_size):
                    mO[row_idx, head_idx, base_k + vec_idx] = state_o[vec_idx].to(self.dtype)
                if tx == 0 and ty == 0:
                    mLSE[head_idx, row_idx] = _state_get_lse_base2(state_m, state_d)
            work_linear_idx += num_ctas

        cute.arch.griddepcontrol_launch_dependents()
