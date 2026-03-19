#!/usr/bin/env python3
"""Benchmark the SGLang-style paged b12x attention API on Qwen-like GQA shapes."""

from __future__ import annotations

import argparse
import pathlib
import statistics
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import torch

from b12x.attention.reference import paged_attention_reference
from b12x.integration.attention import (
    allocate_paged_attention_workspace,
    b12x_paged_attention_forward,
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


def _cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> float:
    a_f = a.to(torch.float32).reshape(-1)
    b_f = b.to(torch.float32).reshape(-1)
    return torch.nn.functional.cosine_similarity(a_f, b_f, dim=0).item()


def _make_uniform_paged_inputs(
    *,
    batch: int,
    q_seqlen: int,
    cache_seqlen: int,
    page_size: int,
    q_heads: int,
    kv_heads: int,
    head_dim: int,
    dtype: torch.dtype,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    torch.manual_seed(seed)
    device = "cuda"
    total_q = batch * q_seqlen
    q = torch.randn(total_q, q_heads, head_dim, device=device, dtype=dtype) / 4
    pages_per_request = (cache_seqlen + page_size - 1) // page_size
    max_pages = pages_per_request
    total_pages_needed = batch * pages_per_request
    num_pages = max(1, total_pages_needed * 2)
    k_cache = torch.randn(num_pages, page_size, kv_heads, head_dim, device=device, dtype=dtype) / 4
    v_cache = torch.randn(num_pages, page_size, kv_heads, head_dim, device=device, dtype=dtype) / 4
    page_table = torch.zeros(batch, max_pages, dtype=torch.int32, device=device)
    page_order = torch.randperm(num_pages, device=device)
    for request_idx in range(batch):
        start = request_idx * pages_per_request
        page_ids = page_order[start : start + pages_per_request].to(torch.int32)
        page_table[request_idx] = page_ids
    cache_seqlens = torch.full((batch,), cache_seqlen, dtype=torch.int32, device=device)
    cu_seqlens_q = torch.arange(0, total_q + 1, q_seqlen, dtype=torch.int32, device=device)
    return q, k_cache, v_cache, page_table, cache_seqlens, cu_seqlens_q


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--q-seqlen", type=int, default=6)
    parser.add_argument("--cache-seqlen", type=int, default=128)
    parser.add_argument("--page-size", type=int, default=64)
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
    q, k_cache, v_cache, page_table, cache_seqlens, cu_seqlens_q = _make_uniform_paged_inputs(
        batch=args.batch,
        q_seqlen=args.q_seqlen,
        cache_seqlen=args.cache_seqlen,
        page_size=args.page_size,
        q_heads=args.q_heads,
        kv_heads=args.kv_heads,
        head_dim=args.head_dim,
        dtype=dtype,
        seed=1,
    )
    workspace = allocate_paged_attention_workspace(
        q,
        k_cache,
        v_cache,
        page_table,
        cache_seqlens,
        cu_seqlens_q,
        causal=True,
    )

    print(
        "shape profile:",
        {
            "batch": args.batch,
            "q_seqlen": args.q_seqlen,
            "cache_seqlen": args.cache_seqlen,
            "page_size": args.page_size,
            "q_heads": args.q_heads,
            "kv_heads": args.kv_heads,
            "head_dim": args.head_dim,
            "dtype": str(dtype),
            "total_q": int(q.shape[0]),
        },
    )

    if args.check:
        out, lse = b12x_paged_attention_forward(
            q,
            k_cache,
            v_cache,
            page_table,
            cache_seqlens,
            cu_seqlens_q,
            workspace=workspace,
        )
        ref_out, ref_lse = paged_attention_reference(
            q,
            k_cache,
            v_cache,
            page_table,
            cache_seqlens,
            cu_seqlens_q,
            causal=True,
        )
        max_abs = (out - ref_out).abs().max().item()
        max_lse_abs = (lse - ref_lse).abs().max().item()
        cos = _cosine_similarity(out, ref_out)
        print(
            f"check batch={args.batch} q={args.q_seqlen} k={args.cache_seqlen}: "
            f"out_max_abs={max_abs:.5f} "
            f"lse_max_abs={max_lse_abs:.5f} "
            f"cos={cos:.8f}"
        )

    times_ms = bench_events(
        lambda: b12x_paged_attention_forward(
            q,
            k_cache,
            v_cache,
            page_table,
            cache_seqlens,
            cu_seqlens_q,
            workspace=workspace,
        ),
        warmup=args.warmup,
        iters=args.iters,
    )
    print(
        f"batch={args.batch:4d} "
        f"q={args.q_seqlen:4d} "
        f"k={args.cache_seqlen:5d} "
        f"median={statistics.median(times_ms) * 1000.0:8.1f} us "
        f"min={min(times_ms) * 1000.0:8.1f} us"
    )


if __name__ == "__main__":
    main()
