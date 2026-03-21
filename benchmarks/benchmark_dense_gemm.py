#!/usr/bin/env python3
"""Benchmark: b12x dense_gemm vs FlashInfer mm_fp4 (cuDNN and CUTLASS backends).

Compares block-scaled FP4 dense GEMM performance on realistic Qwen3.5-397B
(TP=4) shapes: attention projections, shared expert MLP, at decode batch
sizes 1, 2, 4, 8.

Note: M and N must be multiples of 128 for our kernel's tile constraints,
so shapes are padded up accordingly.
"""

from __future__ import annotations

import argparse
import pathlib
import statistics
import sys
from typing import Callable, List

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import torch

from b12x.cute.fp4 import quantize_grouped_nvfp4_torch
from b12x.cute.utils import convert_sf_from_mma_layout
from b12x.gemm.dense import dense_gemm

from flashinfer.gemm.gemm_base import CUDNN_AVAILABLE
from flashinfer.gemm import mm_fp4


def _align128(x: int) -> int:
    return ((x + 127) // 128) * 128


# Qwen3.5-397B at TP=4 — all non-MoE GEMMs
# Shape: (M, N, K) where M=batch tokens, padded to 128-multiples
#
# Linear attention (45/60 layers):
#   QKV: [M, 4096] x [4096, 4608] -> Q=8x256 + K=4x128 + V=16x128
#   O:   [M, 2048] x [2048, 4096]
#
# Full attention (15/60 layers):
#   QKV: [M, 4096] x [4096, 3072] -> Q=8x256 + KV=2x2x256
#   O:   [M, 2048] x [2048, 4096]
#
# Shared expert MLP (60/60 layers):
#   FC1 gate+up: [M, 4096] x [4096, 512]
#   FC2 down:    [M, 256]  x [256, 4096]

GEMM_SPECS = [
    # (name, K, N, layers_per_60)
    ("Linear attn QKV", 4096, 4608, 45),
    ("Linear attn O",   2048, 4096, 45),
    ("Full attn QKV",   4096, 3072, 15),
    ("Full attn O",     2048, 4096, 15),
    ("Shared expert FC1 gate+up", 4096, 512, 60),
    ("Shared expert FC2 down",    256, 4096, 60),
]

BATCH_SIZES = [1, 2, 4, 8]


def bench_events(fn: Callable[[], None], *, warmup: int, iters: int) -> List[float]:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for i in range(iters):
        starts[i].record()
        fn()
        ends[i].record()
    torch.cuda.synchronize()
    return [s.elapsed_time(e) for s, e in zip(starts, ends)]


def fmt_us(times_ms: List[float]) -> str:
    med = statistics.median(times_ms) * 1000
    mn = min(times_ms) * 1000
    return f"{med:7.1f} us (min {mn:.1f})"


def make_quantized_operand(M: int, K: int):
    source = torch.randn(1, M, K, device="cuda", dtype=torch.bfloat16) / 4
    row_counts = torch.full((1,), M, dtype=torch.int32, device="cuda")
    tensor_amax = source.abs().max().to(torch.float32)
    global_scale = torch.tensor(
        [torch.finfo(torch.float8_e4m3fn).max * 6.0 / tensor_amax],
        dtype=torch.float32, device="cuda",
    )
    packed, scales = quantize_grouped_nvfp4_torch(source, row_counts, global_scale)
    return packed, scales, global_scale


def bench_one(M: int, N: int, K: int, *, warmup: int, iters: int):
    """Benchmark one (M,N,K) problem. Returns (b12x_med, cudnn_med, cutlass_med) in us."""
    torch.manual_seed(42)
    a_packed, a_sf, a_gs = make_quantized_operand(M, K)
    b_packed, b_sf, b_gs = make_quantized_operand(N, K)
    alpha = (1.0 / (a_gs[0] * b_gs[0])).view(1)

    a_fp4_2d = a_packed[:, :, 0].contiguous()
    b_fp4_2d = b_packed[:, :, 0].contiguous()
    a_sf_2d = convert_sf_from_mma_layout(a_sf, m=M, k=K, num_groups=1)
    b_sf_2d = convert_sf_from_mma_layout(b_sf, m=N, k=K, num_groups=1)

    results = {}

    # b12x
    try:
        b12x_out = [None]

        def b12x_launch():
            b12x_out[0] = dense_gemm(
                (a_packed, a_sf), (b_packed, b_sf), alpha=alpha,
                ab_dtype="float4_e2m1fn", sf_dtype="float8_e4m3fn",
                c_dtype="bfloat16", sf_vec_size=16,
            )
        b12x_launch()
        results["b12x"] = bench_events(b12x_launch, warmup=warmup, iters=iters)
    except Exception as exc:
        results["b12x"] = None
        print(f"      b12x FAILED: {exc}")

    # cuDNN
    try:
        def cudnn_launch():
            return mm_fp4(
                a_fp4_2d, b_fp4_2d.T, a_sf_2d, b_sf_2d.T,
                alpha, torch.bfloat16, block_size=16,
                use_8x4_sf_layout=False, backend="cudnn", use_nvfp4=True,
            )
        cudnn_launch()
        results["cuDNN"] = bench_events(cudnn_launch, warmup=warmup, iters=iters)
    except Exception as exc:
        results["cuDNN"] = None
        print(f"      cuDNN FAILED: {exc}")

    # CUTLASS
    try:
        def cutlass_launch():
            return mm_fp4(
                a_fp4_2d, b_fp4_2d.T, a_sf_2d, b_sf_2d.T,
                alpha, torch.bfloat16, block_size=16,
                use_8x4_sf_layout=False, backend="cutlass", use_nvfp4=True,
            )
        cutlass_launch()
        results["CUTLASS"] = bench_events(cutlass_launch, warmup=warmup, iters=iters)
    except Exception as exc:
        results["CUTLASS"] = None
        print(f"      CUTLASS FAILED: {exc}")

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=BATCH_SIZES)
    args = parser.parse_args()

    major, minor = torch.cuda.get_device_capability()
    if (major, minor) != (12, 0):
        raise RuntimeError(f"Requires sm_120, got sm_{major}{minor}")
    if not CUDNN_AVAILABLE:
        raise RuntimeError("cuDNN Python bindings not installed")
    torch.empty(1, device="cuda")

    print("Dense FP4 GEMM: b12x vs cuDNN vs CUTLASS")
    print("Qwen3.5-397B shapes at TP=4")
    print(f"warmup={args.warmup}, iters={args.iters}")
    print()

    # Collect all results for summary
    all_results = []  # (name, bs, M, N, K, b12x_med, cudnn_med, cutlass_med)

    for name, K, N_raw, layers in GEMM_SPECS:
        N = _align128(N_raw)
        pad_note = f" (padded from {N_raw})" if N != N_raw else ""
        print(f"{'=' * 75}")
        print(f"  {name}  K={K} N={N}{pad_note}  [{layers}/60 layers]")
        print(f"{'=' * 75}")

        for bs in args.batch_sizes:
            M = _align128(bs)
            results = bench_one(M, N, K, warmup=args.warmup, iters=args.iters)

            b12x_med = statistics.median(results["b12x"]) * 1000 if results.get("b12x") else None
            cudnn_med = statistics.median(results["cuDNN"]) * 1000 if results.get("cuDNN") else None
            cutlass_med = statistics.median(results["CUTLASS"]) * 1000 if results.get("CUTLASS") else None

            parts = [f"  bs={bs:<3} (M={M:>4})"]
            if b12x_med is not None:
                parts.append(f"b12x={b12x_med:6.1f}")
            if cudnn_med is not None:
                parts.append(f"cuDNN={cudnn_med:6.1f}")
            if cutlass_med is not None:
                parts.append(f"CUTLASS={cutlass_med:6.1f}")

            ratios = []
            if b12x_med and cudnn_med:
                r = b12x_med / cudnn_med
                ratios.append(f"b12x/cuDNN={r:.2f}x")
            if b12x_med and cutlass_med:
                r = b12x_med / cutlass_med
                ratios.append(f"b12x/CUTLASS={r:.2f}x")

            print("  ".join(parts) + "  " + "  ".join(ratios) + "  (us)")

            all_results.append((name, bs, M, N, K, b12x_med, cudnn_med, cutlass_med))

        print()

    # --- Summary tables ---
    print(f"\n{'=' * 75}")
    print("  SUMMARY: b12x / cuDNN (lower = b12x faster)")
    print(f"{'=' * 75}")
    header = f"  {'GEMM':<30}"
    for bs in args.batch_sizes:
        header += f"  bs={bs:<4}"
    print(header)
    print("  " + "-" * 70)

    cudnn_ratios = []
    for name, K, N_raw, layers in GEMM_SPECS:
        N = _align128(N_raw)
        row = f"  {name:<30}"
        for bs in args.batch_sizes:
            M = _align128(bs)
            match = [r for r in all_results if r[0] == name and r[1] == bs]
            if match and match[0][5] and match[0][6]:
                ratio = match[0][5] / match[0][6]
                row += f"  {ratio:.2f}x "
                cudnn_ratios.append(ratio)
            else:
                row += f"  {'n/a':>6}"
        print(row)

    if cudnn_ratios:
        geo = 1.0
        for r in cudnn_ratios:
            geo *= r
        geo **= 1.0 / len(cudnn_ratios)
        print(f"\n  geo mean: {geo:.2f}x")

    print(f"\n{'=' * 75}")
    print("  SUMMARY: b12x / CUTLASS (lower = b12x faster)")
    print(f"{'=' * 75}")
    print(header)
    print("  " + "-" * 70)

    cutlass_ratios = []
    for name, K, N_raw, layers in GEMM_SPECS:
        N = _align128(N_raw)
        row = f"  {name:<30}"
        for bs in args.batch_sizes:
            M = _align128(bs)
            match = [r for r in all_results if r[0] == name and r[1] == bs]
            if match and match[0][5] and match[0][7]:
                ratio = match[0][5] / match[0][7]
                row += f"  {ratio:.2f}x "
                cutlass_ratios.append(ratio)
            else:
                row += f"  {'n/a':>6}"
        print(row)

    if cutlass_ratios:
        geo = 1.0
        for r in cutlass_ratios:
            geo *= r
        geo **= 1.0 / len(cutlass_ratios)
        print(f"\n  geo mean: {geo:.2f}x")


if __name__ == "__main__":
    main()
