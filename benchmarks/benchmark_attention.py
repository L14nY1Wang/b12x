#!/usr/bin/env python3
"""Benchmark the transplanted b12x attention forward path on Qwen-like GQA shapes."""

from __future__ import annotations

import argparse
import pathlib
import statistics
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import torch

from b12x.attention.reference import attention_reference
from b12x.integration.attention import (
    allocate_attention_workspace,
    b12x_attention_forward,
    clear_attention_caches,
)


def require_sm120() -> None:
    major, minor = torch.cuda.get_device_capability()
    if (major, minor) != (12, 0):
        raise RuntimeError(f"Requires sm_120, got sm_{major}{minor}")


def bench_events(fn, *, warmup: int, iters: int) -> list[float]:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for idx in range(iters):
        starts[idx].record()
        fn()
        ends[idx].record()
    torch.cuda.synchronize()
    return [start.elapsed_time(end) for start, end in zip(starts, ends)]


def _dtype_from_name(name: str) -> torch.dtype:
    if name == "bf16":
        return torch.bfloat16
    if name == "fp16":
        return torch.float16
    raise ValueError(f"unsupported dtype {name}")


def _make_inputs(
    *,
    batch: int,
    seqlen: int,
    q_heads: int,
    kv_heads: int,
    head_dim: int,
    dtype: torch.dtype,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    torch.manual_seed(seed)
    q = torch.randn(batch, seqlen, q_heads, head_dim, device="cuda", dtype=dtype) / 4
    k = torch.randn(batch, seqlen, kv_heads, head_dim, device="cuda", dtype=dtype) / 4
    v = torch.randn(batch, seqlen, kv_heads, head_dim, device="cuda", dtype=dtype) / 4
    return q, k, v


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--seqlens", type=int, nargs="+", default=[48, 128, 512, 2048])
    parser.add_argument("--q-heads", type=int, default=8)
    parser.add_argument("--kv-heads", type=int, default=1)
    parser.add_argument("--head-dim", type=int, default=256)
    parser.add_argument("--dtype", choices=["bf16", "fp16"], default="bf16")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    require_sm120()
    clear_attention_caches()

    dtype = _dtype_from_name(args.dtype)
    print(
        "shape profile:",
        {
            "batch": args.batch,
            "seqlens": args.seqlens,
            "q_heads": args.q_heads,
            "kv_heads": args.kv_heads,
            "head_dim": args.head_dim,
            "dtype": str(dtype),
        },
    )

    for idx, seqlen in enumerate(args.seqlens):
        q, k, v = _make_inputs(
            batch=args.batch,
            seqlen=seqlen,
            q_heads=args.q_heads,
            kv_heads=args.kv_heads,
            head_dim=args.head_dim,
            dtype=dtype,
            seed=idx + 1,
        )
        workspace = allocate_attention_workspace(q, k, v, causal=True)

        if args.check:
            out, lse = b12x_attention_forward(q, k, v, workspace=workspace)
            ref_out, ref_lse = attention_reference(q, k, v, causal=True)
            max_abs = (out - ref_out).abs().max().item()
            max_lse_abs = (lse - ref_lse).abs().max().item()
            print(
                f"check s={seqlen}: "
                f"out_max_abs={max_abs:.5f} "
                f"lse_max_abs={max_lse_abs:.5f}"
            )

        times_ms = bench_events(
            lambda: b12x_attention_forward(q, k, v, workspace=workspace),
            warmup=args.warmup,
            iters=args.iters,
        )
        print(
            f"s={seqlen:5d} "
            f"median={statistics.median(times_ms) * 1000.0:8.1f} us "
            f"min={min(times_ms) * 1000.0:8.1f} us"
        )


if __name__ == "__main__":
    main()
