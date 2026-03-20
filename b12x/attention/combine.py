from __future__ import annotations

import math
from typing import Type

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute

from cutlass import Float32, const_expr

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
    `(row_idx, head_idx)` CTA stripes warps across head-dim tiles while sharing
    one split-weight reduction in shared memory. This keeps the split reduction
    off the Python path while avoiding redundant LSE work per head-dim tile.
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
        self.shared_storage = self._get_shared_storage_cls()

    def _get_shared_storage_cls(self):
        scratch_struct = cute.struct.Align[
            cute.struct.MemRange[Float32, self.num_splits + 1],
            16,
        ]

        @cute.struct
        class SharedStorage:
            scratch: scratch_struct

        return SharedStorage

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
        if num_splits <= 1 or num_splits > 16:
            return False
        if tile_k != 32:
            return False
        required_threads = max(32, math.ceil(head_dim / tile_k) * 32)
        if num_threads % 32 != 0 or num_threads < required_threads or num_threads > 256:
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
        self.kernel(mO_partial, mLSE_partial, mO, mLSE, self.shared_storage).launch(
            grid=grid,
            block=[self.num_threads, 1, 1],
            smem=self.shared_storage.size_in_bytes(),
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        mO_partial: cute.Tensor,
        mLSE_partial: cute.Tensor,
        mO: cute.Tensor,
        mLSE: cute.Tensor,
        SharedStorage: cutlass.Constexpr,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        lane = cute.arch.lane_idx()
        warp_idx = cute.arch.warp_idx()
        row_idx, head_idx, _ = cute.arch.block_idx()

        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(SharedStorage)
        scratch = storage.scratch.get_tensor(cute.make_layout((self.num_splits + 1,)))

        lse_max = -Float32.inf
        lse_sum = Float32.zero
        if warp_idx == 0 and lane == 0:
            for split_idx in cutlass.range_constexpr(self.num_splits):
                lse_val = mLSE_partial[split_idx, head_idx, row_idx]
                lse_max = max(lse_max, lse_val)
            lse_max_cur = 0.0 if lse_max == -Float32.inf else lse_max
            for split_idx in cutlass.range_constexpr(self.num_splits):
                lse_val = mLSE_partial[split_idx, head_idx, row_idx]
                weight = cute.math.exp2(
                    (lse_val - lse_max_cur) * utils.LOG2_E,
                    fastmath=True,
                )
                scratch[split_idx] = weight
                lse_sum += weight
            if lse_sum == 0.0:
                for split_idx in cutlass.range_constexpr(self.num_splits):
                    scratch[split_idx] = 0.0
                scratch[self.num_splits] = -Float32.inf
            else:
                inv_lse_sum = 1.0 / lse_sum
                for split_idx in cutlass.range_constexpr(self.num_splits):
                    scratch[split_idx] *= inv_lse_sum
                scratch[self.num_splits] = cute.math.log(lse_sum, fastmath=True) + lse_max_cur

        cute.arch.sync_threads()
        if tidx == 0:
            mLSE[head_idx, row_idx] = scratch[self.num_splits]

        k_idx = warp_idx * self.tile_k + lane
        if k_idx < self.head_dim:
            accum = Float32.zero
            for split_idx in cutlass.range_constexpr(self.num_splits):
                accum += scratch[split_idx] * mO_partial[split_idx, row_idx, head_idx, k_idx].to(Float32)
            mO[row_idx, head_idx, k_idx] = accum.to(self.dtype)
