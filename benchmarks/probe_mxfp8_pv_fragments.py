from __future__ import annotations

import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass import Int32
from cutlass.cute.runtime import from_dlpack

from b12x.attention import layout_utils
from b12x.attention.forward import SM120ForwardKernel


def _to_cute_tensor(x: torch.Tensor, dtype) -> cute.Tensor:
    tensor = from_dlpack(x, assumed_align=16)
    tensor.element_type = dtype
    return tensor


class ProbeMxFp8PvFragments:
    tile_m = 16
    tile_n = 64
    head_dim = 256
    num_compute_warps = 1
    num_threads = (num_compute_warps + 1) * 32

    def __init__(self):
        self.kernel_spec = SM120ForwardKernel(
            cutlass.BFloat16,
            self.head_dim,
            kv_dtype=cutlass.Float8E4M3FN,
            head_dim_v=self.head_dim,
            tile_m=self.tile_m,
            tile_n=self.tile_n,
            num_threads=self.num_threads,
            num_compute_warps=self.num_compute_warps,
            Q_in_regs=True,
        )

    @cute.jit
    def __call__(self, m_out: cute.Tensor, stream: cuda.CUstream):
        self.kernel_spec.num_mma_threads = self.kernel_spec.num_compute_warps * 32
        self.kernel_spec.num_producer_threads = 32
        self.kernel_spec.num_Q_load_threads = self.kernel_spec.num_mma_threads
        self.kernel_spec.num_epilogue_threads = self.kernel_spec.num_mma_threads
        self.kernel_spec._setup_attributes()
        _, tiled_mma_pv = self.kernel_spec._get_tiled_mma()
        self.kernel(m_out, self.kernel_spec.sV_layout, self.kernel_spec.sV_raw_layout, tiled_mma_pv).launch(
            grid=(1, 1, 1),
            block=[self.num_threads, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        m_out: cute.Tensor,
        sV_layout: cutlass.Constexpr,
        sV_raw_layout: cutlass.Constexpr,
        tiled_mma_pv: cutlass.Constexpr,
    ):
        tidx = cute.arch.thread_idx()[0]
        smem = cutlass.utils.SmemAllocator()
        sV = smem.allocate_tensor(
            element_type=cutlass.BFloat16,
            layout=sV_layout,
            byte_alignment=1024,
        )
        sVRaw = smem.allocate_tensor(
            element_type=cutlass.Float8E4M3FN,
            layout=sV_raw_layout,
            byte_alignment=1024,
        )
        thr_mma_pv = tiled_mma_pv.get_slice(tidx)
        tOrVt = thr_mma_pv.make_fragment_B(thr_mma_pv.partition_B(layout_utils.transpose_view(sV)[None, None, 0]))
        tOrVtRaw = cute.make_fragment_like(cute.recast_tensor(tOrVt, cutlass.Uint8), cutlass.Uint8)
        acc_shape = thr_mma_pv.partition_shape_C((self.tile_m, self.head_dim))
        acc = cute.make_fragment(acc_shape, cutlass.Float32)
        rP = cute.make_fragment_like(acc, cutlass.BFloat16)
        tOrP = layout_utils.reshape_acc_to_frgA(rP)
        if tidx == Int32(0):
            m_out[0] = Int32(cute.size(cute.flatten(cute.recast_tensor(tOrP[None, None, 0], cutlass.Uint32)).shape))
            m_out[1] = Int32(cute.size(cute.flatten(cute.recast_tensor(tOrVtRaw[None, None, 0], cutlass.Uint32)).shape))
            m_out[2] = Int32(cute.size(tOrP.shape[2]))
            m_out[3] = Int32(cute.size(tOrVtRaw.shape[2]))


def main() -> None:
    out = torch.zeros(4, device="cuda", dtype=torch.int32)
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    probe = ProbeMxFp8PvFragments()
    compiled = cute.compile(probe, _to_cute_tensor(out, cutlass.Int32), stream)
    compiled(_to_cute_tensor(out, cutlass.Int32), stream)
    torch.cuda.synchronize()
    vals = out.cpu().tolist()
    print(
        json.dumps(
            {
                "a_u32_per_k": vals[0],
                "b_u32_per_k": vals[1],
                "a_k_slices": vals[2],
                "b_k_slices": vals[3],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
