from __future__ import annotations

import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass.cute.runtime import from_dlpack

from b12x.attention.forward import SM120ForwardKernel


def _to_cute_tensor(x: torch.Tensor, dtype) -> cute.Tensor:
    tensor = from_dlpack(x, assumed_align=16)
    tensor.element_type = dtype
    return tensor


class Bf16SmemLayoutProbe:
    tile_m = 48
    tile_n = 64
    head_dim = 256
    num_compute_warps = 3
    num_threads = num_compute_warps * 32

    def __init__(self):
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
    def __call__(self, m_words: cute.Tensor, stream: cuda.CUstream):
        self.kernel_spec.num_threads = (self.kernel_spec.num_compute_warps + 1) * 32
        self.kernel_spec.num_mma_threads = self.kernel_spec.num_compute_warps * 32
        self.kernel_spec.num_producer_threads = 32
        self.kernel_spec.num_Q_load_threads = self.kernel_spec.num_mma_threads
        self.kernel_spec.num_epilogue_threads = self.kernel_spec.num_mma_threads
        self.kernel_spec._setup_attributes()
        self.kernel(m_words, self.kernel_spec.sV_layout).launch(
            grid=(1, 1, 1),
            block=[self.num_threads, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        m_words: cute.Tensor,
        sV_layout: cutlass.Constexpr,
    ):
        tidx = cute.arch.thread_idx()[0]
        smem = cutlass.utils.SmemAllocator()
        sV = smem.allocate_tensor(
            element_type=cutlass.BFloat16,
            layout=sV_layout,
            byte_alignment=1024,
        )
        sV_u16 = cute.recast_tensor(sV, cutlass.Uint16)
        total_elems = self.tile_n * self.head_dim
        total_words = total_elems // 2
        for idx_iter in cutlass.range_constexpr(cute.ceil_div(total_elems, self.num_threads)):
            linear_idx = tidx + idx_iter * self.num_threads
            if linear_idx < total_elems:
                row = linear_idx // self.head_dim
                col = linear_idx - row * self.head_dim
                sV_u16[row, col, 0] = cutlass.Uint16(linear_idx)
        cute.arch.sync_threads()
        flat_words = cute.flatten(cute.recast_tensor(sV[None, None, 0], cutlass.Uint32))
        for idx_iter in cutlass.range_constexpr(cute.ceil_div(total_words, self.num_threads)):
            word_idx = tidx + idx_iter * self.num_threads
            if word_idx < total_words:
                m_words[word_idx] = flat_words[word_idx].to(cutlass.Int32)


def main() -> None:
    probe = Bf16SmemLayoutProbe()
    num_words = probe.tile_n * probe.head_dim // 2
    device = torch.device("cuda")
    words = torch.empty(num_words, device=device, dtype=torch.int32)
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    compiled = cute.compile(
        probe,
        _to_cute_tensor(words, cutlass.Int32),
        stream,
    )
    compiled(
        _to_cute_tensor(words, cutlass.Int32),
        stream,
    )
    torch.cuda.synchronize()

    flat_words = words.cpu().numpy().astype("uint32")
    pairs = []
    inverse = [-1] * num_words
    bad_pairs = 0
    for flat_idx, packed in enumerate(flat_words.tolist()):
        lo = int(packed & 0xFFFF)
        hi = int((packed >> 16) & 0xFFFF)
        pairs.append({"flat_idx": flat_idx, "lo": lo, "hi": hi})
        if hi != lo + 1 or (lo & 1):
            bad_pairs += 1
            continue
        inverse[lo // 2] = flat_idx

    result = {
        "num_words": num_words,
        "bad_pairs": bad_pairs,
        "first_pairs": pairs[:32],
        "first_inverse": inverse[:64],
        "last_inverse": inverse[-64:],
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
