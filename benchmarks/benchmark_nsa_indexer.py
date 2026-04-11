#!/usr/bin/env python3
"""Benchmark realistic SGLang-like decode replay for NSA top-k selection."""

from __future__ import annotations

import argparse
import functools
import json
import pathlib
import statistics
import sys
from dataclasses import dataclass

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import torch

from b12x.attention.nsa_indexer.reference import sparse_nsa_index_reference
from b12x.integration.nsa_indexer import (
    NSAIndexerDecodeMetadata,
    NSAIndexerExtendMetadata,
    clear_nsa_indexer_caches,
    pack_nsa_index_k_cache_reference,
    sparse_nsa_index_decode_topk,
    sparse_nsa_index_extend_topk,
)

from benchmarks.common import make_sparse_pool_locs, require_sm120, scatter_rows_into_pool
from benchmarks.common import (
    bench_cuda_graph,
    capture_cuda_graph,
    make_dense_candidate_page_table,
)


MODEL_PATH = pathlib.Path("/data/models/GLM-5.1-NVFP4")
DEFAULT_POOL_FACTOR = 6
DEFAULT_GRAPH_WIDTH = 8192
DEFAULT_TOPK = 2048


@dataclass(frozen=True)
class GLMNSAConfig:
    num_heads: int
    head_dim: int = 128
    page_size: int = 64


@functools.lru_cache(maxsize=1)
def _load_glm_config() -> GLMNSAConfig:
    config_path = MODEL_PATH / "config.json"
    if not config_path.exists():
        raise SystemExit(f"GLM-5.1 config not found at {config_path}")
    config = json.loads(config_path.read_text())
    return GLMNSAConfig(num_heads=int(config["num_attention_heads"]))


def _parse_csv_ints(value: str) -> list[int]:
    return [int(part) for part in value.split(",") if part]


def _make_q_and_weights(
    *,
    rows: int,
    cfg: GLMNSAConfig,
    seed: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    del seed
    q_fp8 = torch.full(
        (rows, cfg.num_heads, cfg.head_dim),
        0.5,
        dtype=torch.float32,
        device=device,
    ).to(torch.float8_e4m3fn)
    weights = torch.ones((rows, cfg.num_heads, 1), dtype=torch.float32, device=device)
    return q_fp8, weights


def _make_index_k_cache(
    *,
    active_tokens: int,
    pool_locs: torch.Tensor,
    pool_tokens: int,
    seed: int,
    device: torch.device,
) -> torch.Tensor:
    del seed
    token_scores = torch.linspace(
        0.25,
        1.25,
        active_tokens,
        dtype=torch.float32,
        device=device,
    )
    k = token_scores.unsqueeze(1).expand(-1, 128).contiguous()
    k_pool = scatter_rows_into_pool(k, pool_locs=pool_locs, pool_tokens=pool_tokens)
    return pack_nsa_index_k_cache_reference(k_pool)


def _make_page_table(
    *,
    rows: int,
    width: int,
    valid_per_row: int,
    token_locs: torch.Tensor,
    seed: int,
    device: torch.device,
) -> torch.Tensor:
    if width <= 0:
        raise ValueError("width must be positive")
    if valid_per_row <= 0 or valid_per_row > width:
        raise ValueError("valid_per_row must be in [1, width]")
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    token_locs_cpu = token_locs.to("cpu")
    num_tokens = int(token_locs_cpu.numel())
    out = torch.full((rows, width), -1, dtype=torch.int32)
    for row in range(rows):
        perm = torch.randperm(num_tokens, generator=gen, dtype=torch.int64)[:valid_per_row]
        ids = token_locs_cpu[perm].to(torch.int32)
        out[row, :valid_per_row] = ids
    return out.to(device=device)


def _assert_exact_match(actual: torch.Tensor, expected: torch.Tensor) -> None:
    if torch.equal(actual, expected):
        return
    mismatch = int((actual != expected).sum().item())
    raise AssertionError(
        f"NSA indexer correctness mismatch: {mismatch} differing entries, "
        f"actual[0]={actual[0].tolist()} expected[0]={expected[0].tolist()}"
    )


def _assert_decode_contract_match(
    *,
    actual: torch.Tensor,
    expected: torch.Tensor,
    page_table_1: torch.Tensor,
    seqlens: torch.Tensor,
    topk: int,
) -> None:
    for row_idx in range(actual.shape[0]):
        seq_len = int(seqlens[row_idx].item())
        if seq_len <= topk:
            if not torch.equal(actual[row_idx, :seq_len], page_table_1[row_idx, :seq_len]):
                raise AssertionError(
                    f"decode trivial-row prefix mismatch at row {row_idx}: "
                    f"actual={actual[row_idx, :seq_len].tolist()} "
                    f"expected_prefix={page_table_1[row_idx, :seq_len].tolist()}"
                )
            if not torch.equal(
                actual[row_idx, seq_len:],
                torch.full((actual.shape[1] - seq_len,), -1, dtype=torch.int32, device=actual.device),
            ):
                raise AssertionError(f"decode trivial-row tail mismatch at row {row_idx}")
            actual_set = {int(token) for token in actual[row_idx].tolist() if int(token) >= 0}
            expected_set = {int(token) for token in expected[row_idx].tolist() if int(token) >= 0}
            if actual_set != expected_set:
                raise AssertionError(
                    f"decode trivial-row token-set mismatch at row {row_idx}: "
                    f"actual={sorted(actual_set)} expected={sorted(expected_set)}"
                )
        elif not torch.equal(actual[row_idx], expected[row_idx]):
            mismatch = int((actual[row_idx] != expected[row_idx]).sum().item())
            raise AssertionError(
                f"decode non-trivial row mismatch at row {row_idx}: {mismatch} differing entries, "
                f"actual={actual[row_idx].tolist()} expected={expected[row_idx].tolist()}"
            )


def _resolve_graph_width(*, cache_len: int, graph_width: int) -> int:
    if graph_width <= 0:
        raise ValueError(f"graph_width must be positive, got {graph_width}")
    return max(cache_len, graph_width)


def _run_decode_case(
    *,
    cfg: GLMNSAConfig,
    q_rows: int,
    cache_len: int,
    width: int,
    topk: int,
    warmup: int,
    replays: int,
    seed: int,
    device: torch.device,
    pool_factor: int,
) -> None:
    graph_width = _resolve_graph_width(cache_len=cache_len, graph_width=width)
    pool_tokens = max(cache_len, cache_len * pool_factor)
    pool_locs = make_sparse_pool_locs(
        active_tokens=cache_len,
        pool_tokens=pool_tokens,
        seed=seed + 10,
        device=device,
    )
    q_fp8, weights = _make_q_and_weights(rows=q_rows, cfg=cfg, seed=seed, device=device)
    index_k_cache = _make_index_k_cache(
        active_tokens=cache_len,
        pool_locs=pool_locs,
        pool_tokens=pool_tokens,
        seed=seed + 1,
        device=device,
    )
    live_page_table_1 = make_dense_candidate_page_table(
        batch_size=q_rows,
        token_locs=pool_locs,
        width=cache_len,
    )
    seqlens = torch.full((q_rows,), cache_len, dtype=torch.int32, device=device)
    graph_page_table_1 = torch.zeros(
        (q_rows, graph_width),
        dtype=torch.int32,
        device=device,
    )
    graph_seqlens = torch.empty_like(seqlens)

    def prepare_decode_graph() -> None:
        graph_page_table_1[:, :cache_len].copy_(live_page_table_1)
        graph_seqlens.copy_(seqlens)

    prepare_decode_graph()
    metadata = NSAIndexerDecodeMetadata(
        page_table_1=graph_page_table_1,
        cache_seqlens_int32=graph_seqlens,
    )

    def run():
        return sparse_nsa_index_decode_topk(
            q_fp8=q_fp8,
            weights=weights,
            index_k_cache=index_k_cache,
            metadata=metadata,
            topk=topk,
            page_size=cfg.page_size,
        )

    clear_nsa_indexer_caches()
    actual = run()
    expected = sparse_nsa_index_reference(
        q_fp8=q_fp8,
        weights=weights,
        index_k_cache=index_k_cache,
        page_table_1=graph_page_table_1,
        query_row_to_batch=torch.arange(q_rows, dtype=torch.int32, device=device),
        seqlens_per_query=graph_seqlens,
        topk=topk,
        page_size=cfg.page_size,
    )
    torch.cuda.synchronize()
    _assert_decode_contract_match(
        actual=actual,
        expected=expected,
        page_table_1=graph_page_table_1,
        seqlens=graph_seqlens,
        topk=topk,
    )

    graph = capture_cuda_graph(
        run,
        warmup=warmup,
        prepare=prepare_decode_graph,
    )
    stats = bench_cuda_graph(
        graph,
        replays=replays,
        prepare=prepare_decode_graph,
    )
    print(
        json.dumps(
            {
                "contract": "sglang_decode_graph",
                "mode": "decode",
                "q_rows": q_rows,
                "cache_len": cache_len,
                "graph_width": graph_width,
                "topk": topk,
                "pool_tokens": pool_tokens,
                "metadata_median_us": statistics.median(stats["metadata_us"]),
                "replay_median_us": statistics.median(stats["replay_us"]),
                "step_median_us": statistics.median(stats["step_us"]),
                "replay_mean_us": statistics.fmean(stats["replay_us"]),
                "replay_min_us": min(stats["replay_us"]),
                "replay_max_us": max(stats["replay_us"]),
                "replays": replays,
            }
        )
    )


def _run_extend_case(
    *,
    cfg: GLMNSAConfig,
    batch: int,
    q_len: int,
    cache_len: int,
    width: int,
    topk: int,
    warmup: int,
    replays: int,
    seed: int,
    device: torch.device,
    pool_factor: int,
) -> None:
    total_q = batch * q_len
    pool_tokens = max(cache_len, cache_len * pool_factor)
    pool_locs = make_sparse_pool_locs(
        active_tokens=cache_len,
        pool_tokens=pool_tokens,
        seed=seed + 10,
        device=device,
    )
    q_fp8, weights = _make_q_and_weights(rows=total_q, cfg=cfg, seed=seed, device=device)
    index_k_cache = _make_index_k_cache(
        active_tokens=cache_len,
        pool_locs=pool_locs,
        pool_tokens=pool_tokens,
        seed=seed + 1,
        device=device,
    )
    valid_per_row = min(width, cache_len)
    page_table_1 = _make_page_table(
        rows=batch,
        width=width,
        valid_per_row=valid_per_row,
        token_locs=pool_locs,
        seed=seed + 2,
        device=device,
    )
    extend_lengths = [q_len] * batch
    seqlens_expanded = torch.full((total_q,), valid_per_row, dtype=torch.int32, device=device)
    metadata = NSAIndexerExtendMetadata(
        page_table_1=page_table_1,
        nsa_seqlens_expanded=seqlens_expanded,
        nsa_extend_seq_lens_list=extend_lengths,
    )

    def run():
        return sparse_nsa_index_extend_topk(
            q_fp8=q_fp8,
            weights=weights,
            index_k_cache=index_k_cache,
            metadata=metadata,
            topk=topk,
            page_size=cfg.page_size,
        )

    clear_nsa_indexer_caches()
    actual = run()
    expected = sparse_nsa_index_reference(
        q_fp8=q_fp8,
        weights=weights,
        index_k_cache=index_k_cache,
        page_table_1=page_table_1,
        query_row_to_batch=torch.repeat_interleave(
            torch.arange(batch, dtype=torch.int32, device=device),
            torch.tensor(extend_lengths, dtype=torch.int32, device=device),
        ),
        seqlens_per_query=seqlens_expanded,
        topk=topk,
        page_size=cfg.page_size,
    )
    torch.cuda.synchronize()
    _assert_exact_match(actual[: expected.shape[0]], expected)

    graph = _capture_graph(run, warmup=warmup)
    replay_us = _bench_graph(graph, replays=replays)
    print(
        json.dumps(
            {
                "mode": "extend",
                "batch": batch,
                "q_len": q_len,
                "cache_len": cache_len,
                "width": width,
                "topk": topk,
                "pool_tokens": pool_tokens,
                "median_us": statistics.median(replay_us),
                "mean_us": statistics.fmean(replay_us),
                "min_us": min(replay_us),
                "max_us": max(replay_us),
                "replays": replays,
            }
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("decode", "extend", "both"), default="decode")
    parser.add_argument("--decode-rows", default="1,16")
    parser.add_argument("--extend-batches", default="8")
    parser.add_argument("--extend-q-lens", default="4")
    parser.add_argument("--cache-lens", default="2048,8192")
    parser.add_argument(
        "--width",
        type=int,
        default=DEFAULT_GRAPH_WIDTH,
        help="decode graph candidate-table width; actual width is max(cache_len, width)",
    )
    parser.add_argument("--topk", type=int, default=DEFAULT_TOPK)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--replays", type=int, default=50)
    parser.add_argument("--seed", type=int, default=88_000)
    parser.add_argument("--pool-factor", type=int, default=DEFAULT_POOL_FACTOR)
    args = parser.parse_args()

    device = require_sm120()
    cfg = _load_glm_config()
    cache_lens = _parse_csv_ints(args.cache_lens)
    decode_rows = _parse_csv_ints(args.decode_rows)
    extend_batches = _parse_csv_ints(args.extend_batches)
    extend_q_lens = _parse_csv_ints(args.extend_q_lens)

    case_seed = args.seed
    if args.mode in ("decode", "both"):
        for cache_len in cache_lens:
            for q_rows in decode_rows:
                _run_decode_case(
                    cfg=cfg,
                    q_rows=q_rows,
                    cache_len=cache_len,
                    width=args.width,
                    topk=args.topk,
                    warmup=args.warmup,
                    replays=args.replays,
                    seed=case_seed,
                    device=device,
                    pool_factor=args.pool_factor,
                )
                case_seed += 17
    if args.mode in ("extend", "both"):
        for cache_len in cache_lens:
            for batch in extend_batches:
                for q_len in extend_q_lens:
                    _run_extend_case(
                        cfg=cfg,
                        batch=batch,
                        q_len=q_len,
                        cache_len=cache_len,
                        width=args.width,
                        topk=args.topk,
                        warmup=args.warmup,
                        replays=args.replays,
                        seed=case_seed,
                        device=device,
                        pool_factor=args.pool_factor,
                    )
                    case_seed += 17


if __name__ == "__main__":
    main()
