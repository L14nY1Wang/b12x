#!/usr/bin/env python3
"""Benchmark realistic SGLang-like GLM-5.1 TP8 decode replay: NSA indexer + sparse MLA."""

from __future__ import annotations

import argparse
import json
import pathlib
import statistics
import sys
from dataclasses import dataclass

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import torch

from b12x.attention.mla.split import select_sparse_mla_split_decode_config
from b12x.attention.nsa_indexer.reference import sparse_nsa_index_reference
from b12x.integration.mla import (
    MLASparseDecodeMetadata,
    MLAWorkspace,
    clear_mla_caches,
    dense_mla_reference,
    pack_mla_kv_cache_reference,
    sparse_mla_decode_forward,
)
from b12x.integration.nsa_indexer import (
    NSAIndexerDecodeMetadata,
    clear_nsa_indexer_caches,
    pack_nsa_index_k_cache_reference,
    sparse_nsa_index_decode_topk,
)

from benchmarks.common import (
    bench_cuda_graph,
    capture_cuda_graph,
    make_dense_candidate_page_table,
    make_sparse_pool_locs,
    require_sm120,
    scatter_rows_into_pool,
)


MODEL_PATH = pathlib.Path("/data/models/GLM-5.1-NVFP4")
DEFAULT_BATCH_SIZES = (1, 2, 4, 8)
DEFAULT_CACHE_LENS = (1024, 8192, 16384, 32768, 65536)
DEFAULT_TP_SIZE = 8
DEFAULT_TP_RANK = 0
DEFAULT_POOL_FACTOR = 6
DEFAULT_GRAPH_WIDTH = 8192
MLA_MAX_ABS_TOL = 0.10
MLA_RMSE_TOL = 0.005
MLA_COS_TOL = 0.9995


@dataclass(frozen=True)
class GLMDecodeContractConfig:
    num_attention_heads: int
    index_n_heads: int
    index_head_dim: int
    index_topk: int
    qk_nope_head_dim: int
    qk_rope_head_dim: int
    kv_lora_rank: int
    tp_size: int
    tp_rank: int
    page_size: int = 64

    @property
    def num_local_heads(self) -> int:
        return self.num_attention_heads // self.tp_size

    @property
    def q_head_dim(self) -> int:
        return self.kv_lora_rank + self.qk_rope_head_dim

    @property
    def sm_scale(self) -> float:
        return (self.qk_nope_head_dim + self.qk_rope_head_dim) ** -0.5


@dataclass(frozen=True)
class DecodeCase:
    batch_size: int
    cache_len: int
    topk: int


@dataclass(frozen=True)
class SanityMetrics:
    max_abs: float
    rmse: float
    cos: float


@dataclass(frozen=True)
class CaseReport:
    case: DecodeCase
    graph_width: int
    metadata_us: float
    replay_us: float
    indexer_us: float
    mla_us: float
    split_enabled: bool
    chunk_size: int
    num_chunks: int
    mla_sanity: SanityMetrics

    @property
    def total_us(self) -> float:
        return self.metadata_us + self.replay_us


class BenchmarkFailure(RuntimeError):
    def __init__(self, case: DecodeCase, message: str):
        super().__init__(f"bs={case.batch_size} ctx={case.cache_len} topk={case.topk}: {message}")
        self.case = case


def _require_glm_config() -> pathlib.Path:
    config_path = MODEL_PATH / "config.json"
    if not config_path.exists():
        raise SystemExit(f"GLM-5.1 config not found at {config_path}")
    return config_path


def _load_glm_contract_config(*, tp_size: int, tp_rank: int) -> GLMDecodeContractConfig:
    config = json.loads(_require_glm_config().read_text())
    num_attention_heads = int(config["num_attention_heads"])
    if num_attention_heads % tp_size != 0:
        raise SystemExit(
            f"num_attention_heads={num_attention_heads} is not divisible by tp_size={tp_size}"
        )
    if tp_rank < 0 or tp_rank >= tp_size:
        raise SystemExit(f"tp_rank must be in [0, {tp_size}), got {tp_rank}")
    return GLMDecodeContractConfig(
        num_attention_heads=num_attention_heads,
        index_n_heads=int(config["index_n_heads"]),
        index_head_dim=int(config["index_head_dim"]),
        index_topk=int(config["index_topk"]),
        qk_nope_head_dim=int(config["qk_nope_head_dim"]),
        qk_rope_head_dim=int(config["qk_rope_head_dim"]),
        kv_lora_rank=int(config["kv_lora_rank"]),
        tp_size=tp_size,
        tp_rank=tp_rank,
    )


def _parse_csv_ints(value: str) -> list[int]:
    return [int(part) for part in value.split(",") if part]


def _resolve_topk(*, cache_len: int, topk_cap: int) -> int:
    if cache_len <= 0:
        raise ValueError(f"cache_len must be positive, got {cache_len}")
    if topk_cap <= 0:
        raise ValueError(f"topk_cap must be positive, got {topk_cap}")
    return min(cache_len, topk_cap)


def _build_decode_cases(
    *,
    batch_sizes: list[int],
    cache_lens: list[int],
    topk_cap: int,
) -> list[DecodeCase]:
    cases: list[DecodeCase] = []
    for batch_size in batch_sizes:
        if batch_size <= 0:
            raise ValueError(f"batch sizes must be positive, got {batch_size}")
        for cache_len in cache_lens:
            cases.append(
                DecodeCase(
                    batch_size=batch_size,
                    cache_len=cache_len,
                    topk=_resolve_topk(cache_len=cache_len, topk_cap=topk_cap),
                )
            )
    return cases


def _geomean(values: list[float]) -> float:
    if not values:
        raise ValueError("geomean requires at least one value")
    return statistics.geometric_mean(values)


def _compare(a: torch.Tensor, b: torch.Tensor) -> SanityMetrics:
    diff = (a - b).to(torch.float32)
    a_f = a.to(torch.float32).reshape(-1)
    b_f = b.to(torch.float32).reshape(-1)
    cos = torch.nn.functional.cosine_similarity(a_f, b_f, dim=0).item()
    return SanityMetrics(
        max_abs=diff.abs().max().item(),
        rmse=torch.sqrt(diff.square().mean()).item(),
        cos=cos,
    )


def _resolve_graph_width(*, cache_len: int, graph_width: int) -> int:
    if graph_width <= 0:
        raise ValueError(f"graph_width must be positive, got {graph_width}")
    return max(cache_len, graph_width)


def _assert_decode_contract_match(
    *,
    case: DecodeCase,
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
                raise BenchmarkFailure(case, f"trivial-row prefix mismatch at row {row_idx}")
            if not torch.equal(
                actual[row_idx, seq_len:],
                torch.full((actual.shape[1] - seq_len,), -1, dtype=torch.int32, device=actual.device),
            ):
                raise BenchmarkFailure(case, f"trivial-row tail mismatch at row {row_idx}")
            actual_set = {int(token) for token in actual[row_idx].tolist() if int(token) >= 0}
            expected_set = {int(token) for token in expected[row_idx].tolist() if int(token) >= 0}
            if actual_set != expected_set:
                raise BenchmarkFailure(case, f"trivial-row token-set mismatch at row {row_idx}")
        elif not torch.equal(actual[row_idx], expected[row_idx]):
            mismatch = int((actual[row_idx] != expected[row_idx]).sum().item())
            raise BenchmarkFailure(case, f"non-trivial row mismatch at row {row_idx}: {mismatch} differing entries")


def _make_decode_graph_prepare(
    *,
    live_page_table_1: torch.Tensor,
    cache_seqlens_int32: torch.Tensor,
    nsa_cache_seqlens_int32: torch.Tensor,
    graph_page_table_1: torch.Tensor,
    graph_cache_seqlens_int32: torch.Tensor,
    graph_nsa_cache_seqlens_int32: torch.Tensor,
):
    live_width = live_page_table_1.shape[1]

    def prepare() -> None:
        graph_page_table_1[:, :live_width].copy_(live_page_table_1)
        graph_cache_seqlens_int32.copy_(cache_seqlens_int32)
        graph_nsa_cache_seqlens_int32.copy_(nsa_cache_seqlens_int32)

    return prepare


def _make_indexer_inputs(
    *,
    case: DecodeCase,
    cfg: GLMDecodeContractConfig,
    seed: int,
    device: torch.device,
    pool_locs: torch.Tensor,
    pool_tokens: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    del seed
    q_fp8 = torch.full(
        (case.batch_size, cfg.index_n_heads, cfg.index_head_dim),
        0.5,
        dtype=torch.float32,
        device=device,
    ).to(torch.float8_e4m3fn)
    weights = torch.ones(
        (case.batch_size, cfg.index_n_heads, 1),
        dtype=torch.float32,
        device=device,
    )
    token_scores = torch.linspace(
        0.25,
        1.25,
        case.cache_len,
        dtype=torch.float32,
        device=device,
    )
    k = token_scores.unsqueeze(1).expand(-1, cfg.index_head_dim).contiguous()
    k_pool = scatter_rows_into_pool(k, pool_locs=pool_locs, pool_tokens=pool_tokens)
    return q_fp8, weights, pack_nsa_index_k_cache_reference(k_pool, page_size=cfg.page_size)


def _make_mla_inputs(
    *,
    case: DecodeCase,
    cfg: GLMDecodeContractConfig,
    seed: int,
    device: torch.device,
    pool_locs: torch.Tensor,
    pool_tokens: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    q_all = (
        torch.randn(
            (case.batch_size, cfg.num_local_heads, cfg.q_head_dim),
            generator=gen,
            dtype=torch.float32,
        )
        .to(device=device)
        .div_(4.0)
        .to(torch.bfloat16)
    )
    k_nope = (
        torch.randn(
            (case.cache_len, 1, cfg.kv_lora_rank),
            generator=gen,
            dtype=torch.float32,
        )
        .to(device=device)
        .div_(4.0)
        .to(torch.bfloat16)
    )
    k_rope = (
        torch.randn(
            (case.cache_len, 1, cfg.qk_rope_head_dim),
            generator=gen,
            dtype=torch.float32,
        )
        .to(device=device)
        .div_(4.0)
        .to(torch.bfloat16)
    )
    k_nope_pool = scatter_rows_into_pool(k_nope, pool_locs=pool_locs, pool_tokens=pool_tokens)
    k_rope_pool = scatter_rows_into_pool(k_rope, pool_locs=pool_locs, pool_tokens=pool_tokens)
    kv_cache = pack_mla_kv_cache_reference(k_nope_pool, k_rope_pool)
    return q_all, k_nope_pool, k_rope_pool, kv_cache


def _make_mla_workspace(
    *,
    cfg: GLMDecodeContractConfig,
    case: DecodeCase,
    device: torch.device,
) -> MLAWorkspace:
    return MLAWorkspace.for_fixed_capacity(
        mode="decode",
        device=device,
        dtype=torch.bfloat16,
        kv_dtype=torch.uint8,
        num_q_heads=cfg.num_local_heads,
        head_dim=cfg.q_head_dim,
        v_head_dim=cfg.kv_lora_rank,
        topk=case.topk,
        max_total_q=case.batch_size,
        max_batch=case.batch_size,
    )


def _run_decode_case(
    *,
    case: DecodeCase,
    cfg: GLMDecodeContractConfig,
    warmup: int,
    replays: int,
    seed: int,
    device: torch.device,
    pool_factor: int,
    graph_width: int,
) -> CaseReport:
    if pool_factor <= 0:
        raise ValueError(f"pool_factor must be positive, got {pool_factor}")
    graph_width = _resolve_graph_width(cache_len=case.cache_len, graph_width=graph_width)
    pool_tokens = max(case.cache_len, case.cache_len * pool_factor)
    pool_locs = make_sparse_pool_locs(
        active_tokens=case.cache_len,
        pool_tokens=pool_tokens,
        seed=seed + 2,
        device=device,
    )
    q_fp8, weights, index_k_cache = _make_indexer_inputs(
        case=case,
        cfg=cfg,
        seed=seed,
        device=device,
        pool_locs=pool_locs,
        pool_tokens=pool_tokens,
    )
    q_all, k_nope, k_rope, kv_cache = _make_mla_inputs(
        case=case,
        cfg=cfg,
        seed=seed + 1,
        device=device,
        pool_locs=pool_locs,
        pool_tokens=pool_tokens,
    )
    live_candidate_page_table = make_dense_candidate_page_table(
        batch_size=case.batch_size,
        token_locs=pool_locs,
        width=case.cache_len,
    )
    full_cache_seqlens = torch.full(
        (case.batch_size,),
        case.cache_len,
        dtype=torch.int32,
        device=device,
    )
    nsa_cache_seqlens = torch.full(
        (case.batch_size,),
        case.topk,
        dtype=torch.int32,
        device=device,
    )
    graph_candidate_page_table = torch.zeros(
        (case.batch_size, graph_width),
        dtype=torch.int32,
        device=device,
    )
    graph_cache_seqlens = torch.empty_like(full_cache_seqlens)
    graph_nsa_cache_seqlens = torch.empty_like(nsa_cache_seqlens)
    prepare_decode_graph = _make_decode_graph_prepare(
        live_page_table_1=live_candidate_page_table,
        cache_seqlens_int32=full_cache_seqlens,
        nsa_cache_seqlens_int32=nsa_cache_seqlens,
        graph_page_table_1=graph_candidate_page_table,
        graph_cache_seqlens_int32=graph_cache_seqlens,
        graph_nsa_cache_seqlens_int32=graph_nsa_cache_seqlens,
    )
    prepare_decode_graph()
    indexer_metadata = NSAIndexerDecodeMetadata(
        page_table_1=graph_candidate_page_table,
        cache_seqlens_int32=graph_cache_seqlens,
    )

    def run_indexer():
        return sparse_nsa_index_decode_topk(
            q_fp8=q_fp8,
            weights=weights,
            index_k_cache=index_k_cache,
            metadata=indexer_metadata,
            topk=case.topk,
            page_size=cfg.page_size,
        )

    clear_nsa_indexer_caches()
    actual_topk = run_indexer()
    expected_topk = sparse_nsa_index_reference(
        q_fp8=q_fp8,
        weights=weights,
        index_k_cache=index_k_cache,
        page_table_1=graph_candidate_page_table,
        query_row_to_batch=torch.arange(case.batch_size, dtype=torch.int32, device=device),
        seqlens_per_query=graph_cache_seqlens,
        topk=case.topk,
        page_size=cfg.page_size,
    )
    torch.cuda.synchronize()
    _assert_decode_contract_match(
        case=case,
        actual=actual_topk,
        expected=expected_topk,
        page_table_1=graph_candidate_page_table,
        seqlens=graph_cache_seqlens,
        topk=case.topk,
    )

    mla_metadata = MLASparseDecodeMetadata(
        page_table_1=actual_topk,
        cache_seqlens_int32=graph_cache_seqlens,
        nsa_cache_seqlens_int32=graph_nsa_cache_seqlens,
        max_seq_len_k=graph_width,
    )
    mla_workspace = _make_mla_workspace(cfg=cfg, case=case, device=device)
    split_cfg = select_sparse_mla_split_decode_config(
        q_all=q_all,
        kv_cache=kv_cache,
        page_table_1=actual_topk,
        output_dtype=q_all.dtype,
        v_head_dim=cfg.kv_lora_rank,
    )

    def run_mla():
        return sparse_mla_decode_forward(
            q_all=q_all,
            kv_cache=kv_cache,
            metadata=mla_metadata,
            workspace=mla_workspace,
            sm_scale=cfg.sm_scale,
            v_head_dim=cfg.kv_lora_rank,
        )

    def run_step():
        topk_indices = sparse_nsa_index_decode_topk(
            q_fp8=q_fp8,
            weights=weights,
            index_k_cache=index_k_cache,
            metadata=indexer_metadata,
            topk=case.topk,
            page_size=cfg.page_size,
        )
        return sparse_mla_decode_forward(
            q_all=q_all,
            kv_cache=kv_cache,
            metadata=MLASparseDecodeMetadata(
                page_table_1=topk_indices,
                cache_seqlens_int32=graph_cache_seqlens,
                nsa_cache_seqlens_int32=graph_nsa_cache_seqlens,
                max_seq_len_k=graph_width,
            ),
            workspace=mla_workspace,
            sm_scale=cfg.sm_scale,
            v_head_dim=cfg.kv_lora_rank,
        )

    clear_mla_caches()
    actual_output = run_mla()
    expected_output = dense_mla_reference(
        q_all=q_all,
        k_nope=k_nope,
        k_rope=k_rope,
        page_table_1=actual_topk,
        sm_scale=cfg.sm_scale,
        v_head_dim=cfg.kv_lora_rank,
    )
    torch.cuda.synchronize()
    mla_sanity = _compare(actual_output, expected_output)
    if mla_sanity.max_abs > MLA_MAX_ABS_TOL:
        raise BenchmarkFailure(
            case,
            f"MLA max_abs {mla_sanity.max_abs:.6f} exceeded {MLA_MAX_ABS_TOL:.6f}",
        )
    if mla_sanity.rmse > MLA_RMSE_TOL:
        raise BenchmarkFailure(
            case,
            f"MLA rmse {mla_sanity.rmse:.6f} exceeded {MLA_RMSE_TOL:.6f}",
        )
    if mla_sanity.cos < MLA_COS_TOL:
        raise BenchmarkFailure(
            case,
            f"MLA cos {mla_sanity.cos:.6f} fell below {MLA_COS_TOL:.6f}",
        )

    clear_nsa_indexer_caches()
    indexer_graph = capture_cuda_graph(
        run_indexer,
        warmup=warmup,
        prepare=prepare_decode_graph,
    )
    indexer_stats = bench_cuda_graph(indexer_graph, replays=replays)
    indexer_us = statistics.median(indexer_stats["replay_us"])

    clear_mla_caches()
    prepare_decode_graph()
    mla_graph = capture_cuda_graph(run_mla, warmup=warmup)
    mla_stats = bench_cuda_graph(mla_graph, replays=replays)
    mla_us = statistics.median(mla_stats["replay_us"])

    clear_nsa_indexer_caches()
    clear_mla_caches()
    step_graph = capture_cuda_graph(
        run_step,
        warmup=warmup,
        prepare=prepare_decode_graph,
    )
    step_stats = bench_cuda_graph(
        step_graph,
        replays=replays,
        prepare=prepare_decode_graph,
    )

    return CaseReport(
        case=case,
        graph_width=graph_width,
        metadata_us=statistics.median(step_stats["metadata_us"]),
        replay_us=statistics.median(step_stats["replay_us"]),
        indexer_us=indexer_us,
        mla_us=mla_us,
        split_enabled=split_cfg is not None,
        chunk_size=0 if split_cfg is None else split_cfg.chunk_size,
        num_chunks=0 if split_cfg is None else split_cfg.num_chunks,
        mla_sanity=mla_sanity,
    )


def collect_case_reports(
    args: argparse.Namespace,
    *,
    device: torch.device | None = None,
) -> list[CaseReport]:
    cfg = _load_glm_contract_config(tp_size=args.tp_size, tp_rank=args.tp_rank)
    device = require_sm120() if device is None else device
    cases = _build_decode_cases(
        batch_sizes=_parse_csv_ints(args.batch_sizes),
        cache_lens=_parse_csv_ints(args.cache_lens),
        topk_cap=min(args.topk_cap, cfg.index_topk),
    )
    reports: list[CaseReport] = []
    case_seed = args.seed
    for case in cases:
        reports.append(
            _run_decode_case(
                case=case,
                cfg=cfg,
                warmup=args.warmup,
                replays=args.replays,
                seed=case_seed,
                device=device,
                pool_factor=args.pool_factor,
                graph_width=args.graph_width,
            )
        )
        case_seed += 17
    return reports


def _render_case_line(report: CaseReport) -> str:
    split_flag = "on" if report.split_enabled else "off"
    return (
        f"glm51-decode tp8 bs={report.case.batch_size:2d} ctx={report.case.cache_len:6d} "
        f"graphw={report.graph_width:6d} topk={report.case.topk:4d} split={split_flag:>3s} "
        f"chunk={report.chunk_size:3d} nchunks={report.num_chunks:d} | "
        f"step={report.total_us:8.2f} us | "
        f"meta={report.metadata_us:8.2f} us | "
        f"replay={report.replay_us:8.2f} us | "
        f"indexer={report.indexer_us:8.2f} us | "
        f"mla={report.mla_us:8.2f} us"
    )


def _render_summary_lines(reports: list[CaseReport]) -> list[str]:
    total_geo = _geomean([report.total_us for report in reports])
    metadata_geo = _geomean([report.metadata_us for report in reports])
    replay_geo = _geomean([report.replay_us for report in reports])
    indexer_geo = _geomean([report.indexer_us for report in reports])
    mla_geo = _geomean([report.mla_us for report in reports])
    return [
        "Summary",
        f"  cases: {len(reports)}",
        f"  step geo:    {total_geo:.2f} us",
        f"  meta geo:    {metadata_geo:.2f} us",
        f"  replay geo:  {replay_geo:.2f} us",
        f"  indexer geo: {indexer_geo:.2f} us",
        f"  mla geo:     {mla_geo:.2f} us",
    ]


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--batch-sizes",
        default="1,2,4,8",
        help=f"decode batch sizes, default {','.join(str(v) for v in DEFAULT_BATCH_SIZES)}",
    )
    parser.add_argument(
        "--cache-lens",
        default="1024,8192,16384,32768,65536",
        help=f"decode cache lengths, default {','.join(str(v) for v in DEFAULT_CACHE_LENS)}",
    )
    parser.add_argument("--topk-cap", type=int, default=2048)
    parser.add_argument("--tp-size", type=int, default=DEFAULT_TP_SIZE)
    parser.add_argument("--tp-rank", type=int, default=DEFAULT_TP_RANK)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--replays", type=int, default=200)
    parser.add_argument("--seed", type=int, default=70_000)
    parser.add_argument("--pool-factor", type=int, default=DEFAULT_POOL_FACTOR)
    parser.add_argument(
        "--graph-width",
        type=int,
        default=DEFAULT_GRAPH_WIDTH,
        help="decode graph candidate-table width; actual width is max(cache_len, graph_width)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        reports = collect_case_reports(args)
    except BenchmarkFailure as exc:
        print(str(exc), file=sys.stderr)
        return 1

    for report in reports:
        print(_render_case_line(report))
    for line in _render_summary_lines(reports):
        print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
