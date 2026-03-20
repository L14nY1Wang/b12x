from __future__ import annotations

import argparse
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

from b12x.attention.forward import SM120ForwardKernel
from b12x.cute.fp4 import bfloat2_to_float2_scaled, fp8x4_e4m3_to_bfloat2x2


def _to_cute_tensor(x: torch.Tensor, dtype) -> cute.Tensor:
    tensor = from_dlpack(x, assumed_align=16)
    tensor.element_type = dtype
    return tensor


class Fp8StageStoreProbe:
    tile_m = 48
    tile_n = 64
    head_dim = 256
    num_compute_warps = 3

    def __init__(self, *, swap_words: bool, swap_halves: bool, store_threads: int):
        self.swap_words = swap_words
        self.swap_halves = swap_halves
        self.store_threads = store_threads
        self.num_threads = max(self.num_compute_warps * 32, store_threads)
        self.kernel_spec = SM120ForwardKernel(
            cutlass.BFloat16,
            self.head_dim,
            kv_dtype=cutlass.Float8E4M3FN,
            head_dim_v=self.head_dim,
            tile_m=self.tile_m,
            tile_n=self.tile_n,
            num_threads=(self.num_compute_warps + 1) * 32,
            num_compute_warps=self.num_compute_warps,
            Q_in_regs=False,
        )

    @cute.jit
    def __call__(
        self,
        m_raw: cute.Tensor,
        m_count: cute.Tensor,
        stream: cuda.CUstream,
    ):
        self.kernel_spec.num_threads = (self.kernel_spec.num_compute_warps + 1) * 32
        self.kernel_spec.num_mma_threads = self.kernel_spec.num_compute_warps * 32
        self.kernel_spec.num_producer_threads = 32
        self.kernel_spec.num_Q_load_threads = self.kernel_spec.num_mma_threads
        self.kernel_spec.num_epilogue_threads = self.kernel_spec.num_mma_threads
        self.kernel_spec._setup_attributes()
        self.kernel(
            m_raw,
            m_count,
            self.kernel_spec.sV_layout,
            self.kernel_spec.sV_raw_layout,
        ).launch(
            grid=(1, 1, 1),
            block=[self.num_threads, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        m_raw: cute.Tensor,
        m_count: cute.Tensor,
        sV_layout: cutlass.Constexpr,
        sV_raw_layout: cutlass.Constexpr,
    ):
        tidx = cute.arch.thread_idx()[0]
        smem = cutlass.utils.SmemAllocator()
        s_ref = smem.allocate_tensor(
            element_type=cutlass.BFloat16,
            layout=sV_layout,
            byte_alignment=1024,
        )
        s_cand = smem.allocate_tensor(
            element_type=cutlass.BFloat16,
            layout=sV_layout,
            byte_alignment=1024,
        )
        s_raw = smem.allocate_tensor(
            element_type=cutlass.Float8E4M3FN,
            layout=sV_raw_layout,
            byte_alignment=1024,
        )
        s_ref_u16 = cute.recast_tensor(s_ref, cutlass.Uint16)
        s_cand_u16 = cute.recast_tensor(s_cand, cutlass.Uint16)
        total_elems = self.tile_n * self.head_dim
        total_vec4 = total_elems // 4
        for idx_iter in cutlass.range_constexpr(cute.ceil_div(total_elems, self.num_threads)):
            linear_idx = tidx + idx_iter * self.num_threads
            if linear_idx < total_elems:
                row = linear_idx // self.head_dim
                col = linear_idx - row * self.head_dim
                s_raw[row, col, 0] = m_raw[row, col]
                s_ref_u16[row, col, 0] = cutlass.Uint16(0)
                s_cand_u16[row, col, 0] = cutlass.Uint16(0)
        cute.arch.sync_threads()

        one = cutlass.Float32(1.0)
        s_cand_u32 = cute.flatten(cute.recast_tensor(s_cand[None, None, 0], cutlass.Uint32))
        if tidx < self.store_threads:
            for idx_iter in cutlass.range_constexpr(cute.ceil_div(total_vec4, self.store_threads)):
                vec_idx = tidx + idx_iter * self.store_threads
                if vec_idx < total_vec4:
                    linear_idx = vec_idx * 4
                    row = linear_idx // self.head_dim
                    col = linear_idx - row * self.head_dim
                    packed = (
                        cutlass.Uint32(cutlass.Uint8(s_raw[row, col + 0, 0]))
                        | (cutlass.Uint32(cutlass.Uint8(s_raw[row, col + 1, 0])) << cutlass.Uint32(8))
                        | (cutlass.Uint32(cutlass.Uint8(s_raw[row, col + 2, 0])) << cutlass.Uint32(16))
                        | (cutlass.Uint32(cutlass.Uint8(s_raw[row, col + 3, 0])) << cutlass.Uint32(24))
                    )
                    bf2_01, bf2_23 = fp8x4_e4m3_to_bfloat2x2(packed)
                    value0, value1 = bfloat2_to_float2_scaled(bf2_01, one)
                    value2, value3 = bfloat2_to_float2_scaled(bf2_23, one)
                    s_ref[row, col + 0, 0] = value0.to(cutlass.BFloat16)
                    s_ref[row, col + 1, 0] = value1.to(cutlass.BFloat16)
                    s_ref[row, col + 2, 0] = value2.to(cutlass.BFloat16)
                    s_ref[row, col + 3, 0] = value3.to(cutlass.BFloat16)

                    if self.swap_halves:
                        bf2_01 = (bf2_01 << cutlass.Uint32(16)) | (bf2_01 >> cutlass.Uint32(16))
                        bf2_23 = (bf2_23 << cutlass.Uint32(16)) | (bf2_23 >> cutlass.Uint32(16))
                    col_pair = col // 2
                    flat_word_idx = col_pair * self.tile_n + row
                    if self.swap_words:
                        s_cand_u32[flat_word_idx] = bf2_23
                        s_cand_u32[flat_word_idx + self.tile_n] = bf2_01
                    else:
                        s_cand_u32[flat_word_idx] = bf2_01
                        s_cand_u32[flat_word_idx + self.tile_n] = bf2_23
        cute.arch.sync_threads()

        if tidx == 0:
            mismatch_count = Int32(0)
            for linear_idx in cutlass.range(total_elems, unroll=1):
                row = linear_idx // self.head_dim
                col = linear_idx - row * self.head_dim
                ref_bits = Int32(s_ref_u16[row, col, 0])
                cand_bits = Int32(s_cand_u16[row, col, 0])
                if ref_bits != cand_bits:
                    mismatch_count += 1
            m_count[0] = mismatch_count


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare scalar vs packed FP8->BF16 shared stores.")
    parser.add_argument("--swap-words", action="store_true")
    parser.add_argument("--swap-halves", action="store_true")
    parser.add_argument("--store-threads", type=int, default=96)
    args = parser.parse_args()

    probe = Fp8StageStoreProbe(
        swap_words=args.swap_words,
        swap_halves=args.swap_halves,
        store_threads=args.store_threads,
    )
    device = torch.device("cuda")
    raw = (torch.arange(probe.tile_n * probe.head_dim, device=device, dtype=torch.int32) * 17 + 3)
    raw = (raw & 0xFF).to(torch.uint8).view(probe.tile_n, probe.head_dim).contiguous()
    count = torch.empty(1, device=device, dtype=torch.int32)
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    compiled = cute.compile(
        probe,
        _to_cute_tensor(raw.view(torch.float8_e4m3fn), cutlass.Float8E4M3FN),
        _to_cute_tensor(count, cutlass.Int32),
        stream,
    )
    compiled(
        _to_cute_tensor(raw.view(torch.float8_e4m3fn), cutlass.Float8E4M3FN),
        _to_cute_tensor(count, cutlass.Int32),
        stream,
    )
    torch.cuda.synchronize()
    result = {
        "swap_words": args.swap_words,
        "swap_halves": args.swap_halves,
        "store_threads": args.store_threads,
        "mismatch_count": int(count.cpu()[0].item()),
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
