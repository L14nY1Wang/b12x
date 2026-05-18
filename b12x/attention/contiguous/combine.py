from __future__ import annotations

import operator
from typing import Type

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute

from cutlass import Float32, Int32, const_expr

from b12x.attention import utils


class PagedAttentionCombineKernel:
    """Minimal split-KV combine kernel for the paged varlen serving path.

    This kernel assumes the input tensors already use the `b12x` paged workspace
    storage layout:

    - `mO_partial`: `(num_splits, total_q, num_heads, head_dim)`
    - `mLSE_partial`: `(num_splits, num_heads, total_q)`
    - `mO`: `(total_q, num_heads, head_dim)`
    - `mLSE`: `(num_heads, total_q)`

    The implementation is intentionally simple: one warp handles one
    `(row_idx, head_idx, k_block)` tile and loops over splits directly from
    global memory. This is sufficient to move split reduction off the Python
    path while preserving exactness.
    """

    def __init__(
        self,
        dtype: Type[cutlass.Numeric],
        dtype_partial: Type[cutlass.Numeric],
        *,
        head_dim: int,
        num_splits: int,
        tile_k: int = 32,
        num_threads: int = 32,
    ):
        self.dtype = dtype
        self.dtype_partial = dtype_partial
        self.head_dim = head_dim
        self.num_splits = num_splits
        self.tile_k = tile_k
        self.num_threads = num_threads

    @staticmethod
    def can_implement(
        dtype,
        dtype_partial,
        *,
        head_dim: int,
        num_splits: int,
        tile_k: int,
        num_threads: int,
    ) -> bool:
        if dtype not in [cutlass.Float16, cutlass.BFloat16, cutlass.Float32]:
            return False
        if dtype_partial not in [cutlass.Float16, cutlass.BFloat16, cutlass.Float32]:
            return False
        if head_dim <= 0 or head_dim % 8 != 0:
            return False
        if num_splits <= 1 or num_splits > 32:
            return False
        if tile_k != 32:
            return False
        if num_threads <= 0 or num_threads > 128 or num_threads % 32 != 0:
            return False
        return True

    @cute.jit
    def __call__(
        self,
        mO_partial: cute.Tensor,
        mLSE_partial: cute.Tensor,
        mO: cute.Tensor,
        mLSE: cute.Tensor,
        stream: cuda.CUstream,
    ):
        if const_expr(len(mO_partial.shape) != 4):
            raise ValueError("mO_partial must have shape (num_splits, total_q, num_heads, head_dim)")
        if const_expr(len(mLSE_partial.shape) != 3):
            raise ValueError("mLSE_partial must have shape (num_splits, num_heads, total_q)")
        if const_expr(len(mO.shape) != 3):
            raise ValueError("mO must have shape (total_q, num_heads, head_dim)")
        if const_expr(len(mLSE.shape) != 2):
            raise ValueError("mLSE must have shape (num_heads, total_q)")
        if const_expr(mO_partial.element_type != self.dtype_partial):
            raise TypeError("mO_partial dtype must match dtype_partial")
        if const_expr(mO.element_type != self.dtype):
            raise TypeError("mO dtype must match dtype")
        if const_expr(mLSE_partial.element_type != Float32 or mLSE.element_type != Float32):
            raise TypeError("mLSE tensors must be Float32")
        if const_expr(mO_partial.shape[0] != self.num_splits):
            raise ValueError(f"mO_partial split dimension must be {self.num_splits}")
        if const_expr(mLSE_partial.shape[0] != self.num_splits):
            raise ValueError(f"mLSE_partial split dimension must be {self.num_splits}")
        if const_expr(mO_partial.shape[1] != mO.shape[0]):
            raise ValueError("mO_partial total_q must match mO total_q")
        if const_expr(mO_partial.shape[2] != mO.shape[1]):
            raise ValueError("mO_partial num_heads must match mO num_heads")
        if const_expr(mO_partial.shape[3] != mO.shape[2]):
            raise ValueError("mO_partial head_dim must match mO head_dim")
        if const_expr(mLSE_partial.shape[1] != mO.shape[1] or mLSE_partial.shape[2] != mO.shape[0]):
            raise ValueError("mLSE_partial layout must be (num_splits, num_heads, total_q)")
        if const_expr(mLSE.shape[0] != mO.shape[1] or mLSE.shape[1] != mO.shape[0]):
            raise ValueError("mLSE layout must be (num_heads, total_q)")
        if const_expr(mO.shape[2] != self.head_dim):
            raise ValueError(f"mO head_dim must be {self.head_dim}")
        if const_expr(not self.can_implement(
            self.dtype,
            self.dtype_partial,
            head_dim=self.head_dim,
            num_splits=self.num_splits,
            tile_k=self.tile_k,
            num_threads=self.num_threads,
        )):
            raise TypeError("combine kernel configuration is not supported")

        total_q = mO.shape[0]
        num_heads = mO.shape[1]
        grid = (total_q, num_heads, 1)
        self.kernel(mO_partial, mLSE_partial, mO, mLSE).launch(
            grid=grid,
            block=[self.num_threads, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        mO_partial: cute.Tensor,
        mLSE_partial: cute.Tensor,
        mO: cute.Tensor,
        mLSE: cute.Tensor,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        row_idx, head_idx, _ = cute.arch.block_idx()
        lane = tidx % cute.arch.WARP_SIZE
        warp_idx = tidx // cute.arch.WARP_SIZE

        lane_active = lane < self.num_splits
        split_lse = -Float32.inf
        if lane_active:
            split_lse = mLSE_partial[lane, head_idx, row_idx]

        lse_max = utils.warp_reduce(split_lse, utils.fmax)
        lse_max_cur = 0.0 if lse_max == -Float32.inf else lse_max
        split_weight = Float32.zero
        if lane_active:
            split_weight = cute.math.exp2(
                (split_lse - lse_max_cur) * utils.LOG2_E,
                fastmath=True,
            )
        lse_sum = utils.warp_reduce(split_weight, operator.add)
        final_lse = -Float32.inf
        if lse_sum != 0.0:
            split_weight *= 1.0 / lse_sum
            final_lse = cute.math.log(lse_sum, fastmath=True) + lse_max_cur

        final_lse = cute.arch.shuffle_sync(final_lse, Int32(0))
        if warp_idx == 0 and lane == 0:
            mLSE[head_idx, row_idx] = final_lse

        for idx_iter in cutlass.range_constexpr(cute.ceil_div(self.head_dim, self.num_threads)):
            k_idx = tidx + idx_iter * self.num_threads
            if k_idx < self.head_dim:
                accum = Float32.zero
                for split_idx in cutlass.range_constexpr(self.num_splits):
                    weight = cute.arch.shuffle_sync(split_weight, Int32(split_idx))
                    accum += weight * mO_partial[split_idx, row_idx, head_idx, k_idx].to(Float32)
                mO[row_idx, head_idx, k_idx] = accum.to(self.dtype)
