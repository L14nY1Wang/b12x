#!/usr/bin/env python3
"""Optuna search over BF16 decode-2048 scheduler knobs.

This script searches the live host-side scheduling surface for the single hot
point:

- batch=8
- q=1
- k=2048
- q/kv dtype=bf16
- page_size=64

It intentionally patches planner/dispatch functions at runtime per trial
instead of rewriting source files or recompiling a custom benchmark binary for
each candidate.
"""

from __future__ import annotations

import argparse
import contextlib
import math
import pathlib
import statistics
import sys
import traceback
from dataclasses import dataclass
from typing import Any

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

LOCAL_OPTUNA = ROOT / ".deps" / "optuna"
if LOCAL_OPTUNA.exists():
    sys.path.insert(0, str(LOCAL_OPTUNA))

try:
    import optuna
except Exception as exc:  # pragma: no cover - env-time dependency
    raise ImportError(
        "optuna is required; install it with "
        "`pip install --target .deps/optuna optuna` from the repo root."
    ) from exc

import torch

import benchmarks.benchmark_paged_attention as bench
from b12x.attention.paged import api as paged_api
from b12x.attention.paged import merge as paged_merge
from b12x.attention.paged import planner as paged_planner
from b12x.attention.paged import workspace as paged_workspace
from b12x.integration.attention import clear_attention_caches

CHUNK_PAGE_LADDER = [1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256]
BF16_DECODE_LEGACY_BREAKPOINTS = [1, 2, 16, 32, 64, 128, 256, 320, 448, 640, 960, 2048]
BF16_DECODE_EXACT_BREAKPOINTS = [1, 2, 16, 32, 64, 128, 256, 320, 512, 640, 960, 2048]
MAX_BATCH_SCALE_LADDER = [0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0]
DEFAULT_STUDY_NAME = "bf16_decode2048_scheduler"


@dataclass(frozen=True)
class TrialConfig:
    cta_policy: str
    q64_threshold: int
    decode_policy: str
    decode_chunk_pages: tuple[int, ...]
    split_mode: str
    fixed_split_pages: int
    graph_chunk_policy: bool
    max_batch_size_scale: float
    merge_cta_policy: str
    merge_blocks_per_sm_cap: int
    merge_ctas_per_sm: int


@dataclass(frozen=True)
class TrialResult:
    ratio: float
    ratio_ci_low: float
    ratio_ci_high: float
    b12x_mean_us: float
    fa2_mean_us: float
    plan_desc: str
    cta_tile_q: int
    kv_chunk_size: int
    split_kv: bool
    max_abs: float
    cos: float


def _build_decode_table(*, exact_plane: bool, chunk_pages: tuple[int, ...]) -> tuple[tuple[int, int], ...]:
    breakpoints = BF16_DECODE_EXACT_BREAKPOINTS if exact_plane else BF16_DECODE_LEGACY_BREAKPOINTS
    if len(chunk_pages) != len(breakpoints):
        raise ValueError("chunk_pages length does not match decode breakpoints")
    return tuple(zip(breakpoints, chunk_pages, strict=True))


def _sample_config(trial: optuna.Trial) -> TrialConfig:
    decode_chunk_pages = tuple(
        trial.suggest_categorical(f"decode_chunk_pages_le_{bp}", CHUNK_PAGE_LADDER)
        for bp in BF16_DECODE_EXACT_BREAKPOINTS
    )
    split_mode = trial.suggest_categorical("split_mode", ["planner", "fixed", "disabled"])
    fixed_split_pages = (
        int(trial.suggest_categorical("fixed_split_pages", CHUNK_PAGE_LADDER)) if split_mode == "fixed" else 0
    )
    merge_cta_policy = trial.suggest_categorical("merge_cta_policy", ["formula", "absolute"])
    return TrialConfig(
        cta_policy=trial.suggest_categorical("cta_policy", ["planner", "force16", "force64", "force128"]),
        q64_threshold=trial.suggest_int("q64_threshold", 1, 64),
        decode_policy=trial.suggest_categorical("decode_policy", ["exact_table", "legacy_table", "binary_search"]),
        decode_chunk_pages=decode_chunk_pages,
        split_mode=split_mode,
        fixed_split_pages=fixed_split_pages,
        graph_chunk_policy=trial.suggest_categorical("graph_chunk_policy", [False, True]),
        max_batch_size_scale=float(trial.suggest_categorical("max_batch_size_scale", MAX_BATCH_SCALE_LADDER)),
        merge_cta_policy=merge_cta_policy,
        merge_blocks_per_sm_cap=trial.suggest_int("merge_blocks_per_sm_cap", 1, 6),
        merge_ctas_per_sm=trial.suggest_int("merge_ctas_per_sm", 1, 6),
    )


def _baseline_params() -> dict[str, Any]:
    return {
        "cta_policy": "planner",
        "q64_threshold": 16,
        "decode_policy": "exact_table",
        "decode_chunk_pages_le_1": 1,
        "decode_chunk_pages_le_2": 2,
        "decode_chunk_pages_le_16": 1,
        "decode_chunk_pages_le_32": 2,
        "decode_chunk_pages_le_64": 3,
        "decode_chunk_pages_le_128": 6,
        "decode_chunk_pages_le_256": 12,
        "decode_chunk_pages_le_320": 16,
        "decode_chunk_pages_le_512": 48,
        "decode_chunk_pages_le_640": 64,
        "decode_chunk_pages_le_960": 96,
        "decode_chunk_pages_le_2048": 128,
        "split_mode": "planner",
        "graph_chunk_policy": True,
        "max_batch_size_scale": 1.0,
        "merge_cta_policy": "formula",
        "merge_blocks_per_sm_cap": 3,
        "merge_ctas_per_sm": 3,
    }


@contextlib.contextmanager
def _scheduler_overrides(config: TrialConfig):
    orig_fa2_determine = paged_planner._fa2_determine_cta_tile_q
    orig_exact_fn = paged_planner._use_paged_bf16_tma_exact_plane_chunk_tables
    orig_exact_table = paged_planner._PAGED_DECODE_BF16_TMA_EXACT_PLANE_CHUNK_TABLE_PAGES
    orig_legacy_table = paged_planner._PAGED_DECODE_BF16_CHUNK_TABLE_PAGES
    orig_chunk_table_pages = paged_planner._paged_chunk_table_pages
    orig_prefill_binary_search = paged_planner._prefill_binary_search_kv_chunk_size
    orig_workspace_create_plan = paged_workspace.create_paged_plan
    orig_api_default_merge = paged_api.default_paged_persistent_ctas
    orig_merge_default_merge = paged_merge.default_paged_persistent_ctas

    def patched_fa2_determine(avg_packed_qo_len: int, head_dim: int) -> int:
        if config.cta_policy == "force16":
            return 16
        if config.cta_policy == "force64":
            return 64
        if config.cta_policy == "force128":
            return 128
        if avg_packed_qo_len > 64 and head_dim < 256:
            return 128
        if avg_packed_qo_len > config.q64_threshold:
            return 64
        return 16

    def patched_exact_plane() -> bool:
        return config.decode_policy == "exact_table"

    decode_exact_table = _build_decode_table(exact_plane=True, chunk_pages=config.decode_chunk_pages)
    decode_legacy_table = _build_decode_table(exact_plane=False, chunk_pages=config.decode_chunk_pages)

    def patched_chunk_table_pages(*, mode, q_dtype, kv_dtype, page_size, head_dim_qk, head_dim_vo, gqa_group_size,
                                  max_effective_kv_pages, graph_chunk_policy):
        if mode == "decode" and kv_dtype == torch.bfloat16 and config.decode_policy == "binary_search":
            return None
        return orig_chunk_table_pages(
            mode=mode,
            q_dtype=q_dtype,
            kv_dtype=kv_dtype,
            page_size=page_size,
            head_dim_qk=head_dim_qk,
            head_dim_vo=head_dim_vo,
            gqa_group_size=gqa_group_size,
            max_effective_kv_pages=max_effective_kv_pages,
            graph_chunk_policy=graph_chunk_policy,
        )

    def patched_prefill_binary_search_kv_chunk_size(*, enable_cuda_graph, max_batch_size_if_split,
                                                    packed_qo_len_arr, kv_len_arr, qo_chunk_size,
                                                    min_kv_chunk_size=1):
        scaled_budget = max(1, int(round(max_batch_size_if_split * config.max_batch_size_scale)))
        return orig_prefill_binary_search(
            enable_cuda_graph=enable_cuda_graph,
            max_batch_size_if_split=scaled_budget,
            packed_qo_len_arr=packed_qo_len_arr,
            kv_len_arr=kv_len_arr,
            qo_chunk_size=qo_chunk_size,
            min_kv_chunk_size=min_kv_chunk_size,
        )

    def patched_workspace_create_plan(*args, **kwargs):
        if config.split_mode == "disabled":
            kwargs["disable_split_kv"] = True
            kwargs["fixed_split_size"] = -1
        elif config.split_mode == "fixed":
            kwargs["disable_split_kv"] = False
            kwargs["fixed_split_size"] = config.fixed_split_pages
        else:
            kwargs["disable_split_kv"] = False
            kwargs["fixed_split_size"] = -1
        kwargs["graph_chunk_policy"] = config.graph_chunk_policy
        return orig_workspace_create_plan(*args, **kwargs)

    def patched_default_persistent_ctas(*, total_rows: int, num_heads: int, device=None) -> int:
        if device is None:
            device = torch.cuda.current_device()
        num_sms = int(torch.cuda.get_device_properties(device).multi_processor_count)
        if config.merge_cta_policy == "absolute":
            return int(num_sms * max(config.merge_ctas_per_sm, 1))
        total_work = max(int(total_rows) * int(num_heads), 1)
        blocks_per_sm = min(config.merge_blocks_per_sm_cap, math.ceil(total_work / num_sms))
        return int(num_sms * max(blocks_per_sm, 1))

    paged_planner._fa2_determine_cta_tile_q = patched_fa2_determine
    paged_planner._use_paged_bf16_tma_exact_plane_chunk_tables = patched_exact_plane
    paged_planner._PAGED_DECODE_BF16_TMA_EXACT_PLANE_CHUNK_TABLE_PAGES = decode_exact_table
    paged_planner._PAGED_DECODE_BF16_CHUNK_TABLE_PAGES = decode_legacy_table
    paged_planner._paged_chunk_table_pages = patched_chunk_table_pages
    paged_planner._prefill_binary_search_kv_chunk_size = patched_prefill_binary_search_kv_chunk_size
    paged_workspace.create_paged_plan = patched_workspace_create_plan
    paged_api.default_paged_persistent_ctas = patched_default_persistent_ctas
    paged_merge.default_paged_persistent_ctas = patched_default_persistent_ctas
    try:
        yield
    finally:
        paged_planner._fa2_determine_cta_tile_q = orig_fa2_determine
        paged_planner._use_paged_bf16_tma_exact_plane_chunk_tables = orig_exact_fn
        paged_planner._PAGED_DECODE_BF16_TMA_EXACT_PLANE_CHUNK_TABLE_PAGES = orig_exact_table
        paged_planner._PAGED_DECODE_BF16_CHUNK_TABLE_PAGES = orig_legacy_table
        paged_planner._paged_chunk_table_pages = orig_chunk_table_pages
        paged_planner._prefill_binary_search_kv_chunk_size = orig_prefill_binary_search
        paged_workspace.create_paged_plan = orig_workspace_create_plan
        paged_api.default_paged_persistent_ctas = orig_api_default_merge
        paged_merge.default_paged_persistent_ctas = orig_merge_default_merge


def _build_trial_inputs(seed: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return bench._make_uniform_paged_inputs(
        batch=8,
        q_seqlen=1,
        cache_seqlen=2048,
        page_size=64,
        q_heads=8,
        kv_heads=1,
        head_dim=256,
        dtype=torch.bfloat16,
        seed=seed,
    )


def _capture_b12x_graph(
    *,
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    page_table: torch.Tensor,
    cache_seqlens: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    warmup: int,
) -> tuple[torch.cuda.CUDAGraph, torch.Tensor, Any]:
    output = torch.empty_like(q)
    workspace = paged_workspace.PagedAttentionWorkspace.for_tensors(
        mode="decode",
        q=q,
        k_cache=k_cache,
        v_cache=v_cache,
        use_cuda_graph=True,
        attn_mode="default",
    )
    workspace.prepare(page_table, cache_seqlens, cu_seqlens_q)

    def run() -> None:
        workspace.run(q, k_cache, v_cache, output=output)

    graph = bench._capture_graph(run, warmup=warmup)
    return graph, output, workspace.plan


def _bench_backend_mean_us(graph: torch.cuda.CUDAGraph, *, replays: int) -> tuple[float, float, float, float]:
    times_ms = bench._bench_graph(graph, replays=replays)
    ci_low_ms, ci_high_ms, sem_ms = bench._mean_ci(times_ms, ci_level=0.95)
    return (
        statistics.fmean(times_ms) * 1000.0,
        ci_low_ms * 1000.0,
        ci_high_ms * 1000.0,
        sem_ms * 1000.0,
    )


def _run_trial(
    *,
    config: TrialConfig,
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    page_table: torch.Tensor,
    cache_seqlens: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    fa_output: torch.Tensor,
    fa_mean_us: float,
    fa_sem_us: float,
    warmup: int,
    replays: int,
) -> TrialResult:
    with _scheduler_overrides(config):
        clear_attention_caches()
        b12x_graph, b12x_output, plan = _capture_b12x_graph(
            q=q,
            k_cache=k_cache,
            v_cache=v_cache,
            page_table=page_table,
            cache_seqlens=cache_seqlens,
            cu_seqlens_q=cu_seqlens_q,
            warmup=warmup,
        )
        b12x_mean_us, _, _, b12x_sem_us = _bench_backend_mean_us(b12x_graph, replays=replays)
        max_abs = float((b12x_output - fa_output).abs().max().item())
        cos = float(bench._cosine_similarity(b12x_output, fa_output))
        ratio = fa_mean_us / b12x_mean_us
        ratio_ci_low, ratio_ci_high = bench._ratio_mean_ci(
            fa_mean_us,
            fa_sem_us,
            b12x_mean_us,
            b12x_sem_us,
            ci_level=0.95,
        )
        return TrialResult(
            ratio=ratio,
            ratio_ci_low=ratio_ci_low,
            ratio_ci_high=ratio_ci_high,
            b12x_mean_us=b12x_mean_us,
            fa2_mean_us=fa_mean_us,
            plan_desc=f"chunk={plan.kv_chunk_size},{'split' if plan.split_kv else 'nosplit'}",
            cta_tile_q=int(plan.cta_tile_q),
            kv_chunk_size=int(plan.kv_chunk_size),
            split_kv=bool(plan.split_kv),
            max_abs=max_abs,
            cos=cos,
        )


def _make_objective(
    *,
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    page_table: torch.Tensor,
    cache_seqlens: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    fa_output: torch.Tensor,
    fa_mean_us: float,
    fa_sem_us: float,
    warmup: int,
    replays: int,
    cos_threshold: float,
) -> Any:
    def objective(trial: optuna.Trial) -> float:
        config = _sample_config(trial)
        try:
            result = _run_trial(
                config=config,
                q=q,
                k_cache=k_cache,
                v_cache=v_cache,
                page_table=page_table,
                cache_seqlens=cache_seqlens,
                cu_seqlens_q=cu_seqlens_q,
                fa_output=fa_output,
                fa_mean_us=fa_mean_us,
                fa_sem_us=fa_sem_us,
                warmup=warmup,
                replays=replays,
            )
        except Exception as exc:
            trial.set_user_attr("status", "crash")
            trial.set_user_attr("error", f"{type(exc).__name__}: {exc}")
            trial.set_user_attr("traceback", traceback.format_exc(limit=20))
            raise

        trial.set_user_attr("status", "ok")
        trial.set_user_attr("plan_desc", result.plan_desc)
        trial.set_user_attr("cta_tile_q", result.cta_tile_q)
        trial.set_user_attr("kv_chunk_size", result.kv_chunk_size)
        trial.set_user_attr("split_kv", result.split_kv)
        trial.set_user_attr("b12x_mean_us", result.b12x_mean_us)
        trial.set_user_attr("fa2_mean_us", result.fa2_mean_us)
        trial.set_user_attr("ratio", result.ratio)
        trial.set_user_attr("ratio_ci_low", result.ratio_ci_low)
        trial.set_user_attr("ratio_ci_high", result.ratio_ci_high)
        trial.set_user_attr("max_abs", result.max_abs)
        trial.set_user_attr("cos", result.cos)
        if result.cos < cos_threshold:
            trial.set_user_attr("status", "bad_cos")
            return 0.0
        return result.ratio

    return objective


def _print_top_trials(study: optuna.Study, *, limit: int) -> None:
    complete = [trial for trial in study.trials if trial.state == optuna.trial.TrialState.COMPLETE]
    complete.sort(key=lambda trial: float(trial.value), reverse=True)
    print(f"top {min(limit, len(complete))} trials:")
    for trial in complete[:limit]:
        print(
            {
                "trial": trial.number,
                "value": round(float(trial.value), 6),
                "plan": trial.user_attrs.get("plan_desc"),
                "cta_tile_q": trial.user_attrs.get("cta_tile_q"),
                "kv_chunk_size": trial.user_attrs.get("kv_chunk_size"),
                "split_kv": trial.user_attrs.get("split_kv"),
                "cos": trial.user_attrs.get("cos"),
                "params": trial.params,
            }
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=200)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--replays", type=int, default=200)
    parser.add_argument("--cos-threshold", type=float, default=0.999)
    parser.add_argument("--study-name", type=str, default=DEFAULT_STUDY_NAME)
    parser.add_argument(
        "--journal-path",
        type=str,
        default=str(ROOT / ".optuna" / (DEFAULT_STUDY_NAME + ".journal")),
    )
    parser.add_argument("--enqueue-baseline", action="store_true", default=True)
    parser.add_argument("--no-enqueue-baseline", action="store_false", dest="enqueue_baseline")
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--timeout", type=int, default=0)
    args = parser.parse_args()

    bench.require_sm120()
    if args.replays < 100:
        raise ValueError("--replays must be at least 100")

    journal_path = pathlib.Path(args.journal_path)
    journal_path.parent.mkdir(parents=True, exist_ok=True)

    q, k_cache, v_cache, page_table, cache_seqlens, cu_seqlens_q = _build_trial_inputs(args.seed)
    clear_attention_caches()
    fa_graph, fa_output = bench._capture_flashinfer_fa2_graph(
        q=q,
        k_cache=k_cache,
        v_cache=v_cache,
        page_table=page_table,
        cache_seqlens=cache_seqlens,
        q_seqlen=1,
        page_size=64,
        q_heads=8,
        kv_heads=1,
        head_dim=256,
        q_dtype=torch.bfloat16,
        kv_dtype=torch.bfloat16,
        k_scale=None,
        v_scale=None,
        workspace_bytes=512 * 1024 * 1024,
        warmup=args.warmup,
    )
    fa_mean_us, _, _, fa_sem_us = _bench_backend_mean_us(fa_graph, replays=args.replays)
    print({"fa2_mean_us": round(fa_mean_us, 3), "fa2_sem_us": round(fa_sem_us, 6)})

    sampler = optuna.samplers.TPESampler(
        seed=args.seed,
        multivariate=True,
        group=True,
        n_startup_trials=20,
        constant_liar=True,
    )
    journal_backend = optuna.storages.JournalFileBackend(str(journal_path))
    storage = optuna.storages.JournalStorage(journal_backend)
    study = optuna.create_study(
        study_name=args.study_name,
        storage=storage,
        direction="maximize",
        sampler=sampler,
        load_if_exists=True,
    )
    if args.enqueue_baseline:
        study.enqueue_trial(_baseline_params())

    objective = _make_objective(
        q=q,
        k_cache=k_cache,
        v_cache=v_cache,
        page_table=page_table,
        cache_seqlens=cache_seqlens,
        cu_seqlens_q=cu_seqlens_q,
        fa_output=fa_output,
        fa_mean_us=fa_mean_us,
        fa_sem_us=fa_sem_us,
        warmup=args.warmup,
        replays=args.replays,
        cos_threshold=args.cos_threshold,
    )
    study.optimize(objective, n_trials=args.trials, timeout=None if args.timeout <= 0 else args.timeout)

    best = study.best_trial
    print("best trial:")
    print(
        {
            "trial": best.number,
            "value": round(float(best.value), 6),
            "plan": best.user_attrs.get("plan_desc"),
            "cta_tile_q": best.user_attrs.get("cta_tile_q"),
            "kv_chunk_size": best.user_attrs.get("kv_chunk_size"),
            "split_kv": best.user_attrs.get("split_kv"),
            "cos": best.user_attrs.get("cos"),
            "params": best.params,
        }
    )
    _print_top_trials(study, limit=args.topk)


if __name__ == "__main__":
    main()
