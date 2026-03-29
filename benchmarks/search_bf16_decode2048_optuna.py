#!/usr/bin/env python3
"""Optuna search over paged-attention scheduler knobs for one target point.

The search is point-targeted and family-aware:

- BF16 decode uses decode-focused scheduler dimensions
- BF16 extend uses BF16-extend-focused dimensions
- FP8 extend uses FP8-extend-focused dimensions

Each trial:
- benchmarks only b12x
- uses one fixed FlashInfer FA2 reference output for correctness
- returns score=0 on compile/runtime failure
- returns score=0 on cosine below the threshold
- otherwise maximizes 1 / mean_us
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
from typing import Any, Literal

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
        "`pip install --target .deps/optuna --no-deps optuna colorlog` from the repo root."
    ) from exc

import torch

import benchmarks.benchmark_paged_attention as bench
from b12x.attention.paged import api as paged_api
from b12x.attention.paged import merge as paged_merge
from b12x.attention.paged import planner as paged_planner
from b12x.attention.paged import workspace as paged_workspace
from b12x.integration.attention import clear_attention_caches

CHUNK_PAGE_LADDER = [1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256]
MAX_BATCH_SCALE_LADDER = [0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0]
LONG_FORM_CUTOFF_TOKENS_LADDER = [128, 256, 384, 512, 768, 1024, 1536, 2048, 3072, 4096, 8192, 16384, 32768]

BF16_DECODE_LEGACY_BREAKPOINTS = [1, 2, 16, 32, 64, 128, 256, 320, 448, 640, 960, 2048]
BF16_DECODE_EXACT_BREAKPOINTS = [1, 2, 16, 32, 64, 128, 256, 320, 512, 640, 960, 2048]
BF16_EXTEND_BREAKPOINTS = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]
FP8_EXTEND_BREAKPOINTS = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]

DEFAULT_TRIALS = 200
DEFAULT_REPLAYS = 1000
DEFAULT_WARMUP = 3
DEFAULT_COS_THRESHOLD = 0.999


Family = Literal["bf16_decode", "bf16_extend", "fp8_extend"]


@dataclass(frozen=True)
class TargetSpec:
    mode: Literal["decode", "extend"]
    kv_dtype: torch.dtype
    batch: int
    q_seqlen: int
    cache_seqlen: int
    page_size: int
    q_heads: int
    kv_heads: int
    head_dim: int
    q_dtype: torch.dtype

    @property
    def family(self) -> Family:
        if self.mode == "decode" and self.kv_dtype == torch.bfloat16:
            return "bf16_decode"
        if self.mode == "extend" and self.kv_dtype == torch.bfloat16:
            return "bf16_extend"
        if self.mode == "extend" and self.kv_dtype == torch.float8_e4m3fn:
            return "fp8_extend"
        raise ValueError(f"unsupported target family for mode={self.mode} kv_dtype={self.kv_dtype}")

    @property
    def name(self) -> str:
        kv = "fp8" if self.kv_dtype == torch.float8_e4m3fn else "bf16"
        return f"{kv}_{self.mode}_q{self.q_seqlen}_k{self.cache_seqlen}"


@dataclass(frozen=True)
class TrialResult:
    score: float
    b12x_mean_us: float
    b12x_ci_low_us: float
    b12x_ci_high_us: float
    b12x_sem_us: float
    plan_desc: str
    cta_tile_q: int
    kv_chunk_size: int
    split_kv: bool
    max_abs: float
    cos: float


def _dtype_name(dtype: torch.dtype) -> str:
    if dtype == torch.bfloat16:
        return "bf16"
    if dtype == torch.float8_e4m3fn:
        return "fp8_e4m3fn"
    raise TypeError(f"unsupported dtype {dtype}")


def _study_name_for_target(spec: TargetSpec) -> str:
    return f"{spec.name}_scheduler"


def _journal_path_for_target(spec: TargetSpec) -> pathlib.Path:
    return ROOT / ".optuna" / f"{_study_name_for_target(spec)}.journal"


def _build_decode_table(*, exact_plane: bool, chunk_pages: tuple[int, ...]) -> tuple[tuple[int, int], ...]:
    breakpoints = BF16_DECODE_EXACT_BREAKPOINTS if exact_plane else BF16_DECODE_LEGACY_BREAKPOINTS
    if len(chunk_pages) != len(breakpoints):
        raise ValueError("decode chunk_pages length does not match decode breakpoints")
    return tuple(zip(breakpoints, chunk_pages, strict=True))


def _build_extend_table(*, chunk_pages: tuple[int, ...]) -> tuple[tuple[int, int], ...]:
    if len(chunk_pages) != len(BF16_EXTEND_BREAKPOINTS):
        raise ValueError("extend chunk_pages length does not match extend breakpoints")
    return tuple(zip(BF16_EXTEND_BREAKPOINTS, chunk_pages, strict=True))


def _parse_kv_dtype(name: str) -> torch.dtype:
    if name == "bf16":
        return torch.bfloat16
    if name == "fp8_e4m3fn":
        return torch.float8_e4m3fn
    raise ValueError(f"unsupported kv dtype {name}")


def _sample_common_config(trial: optuna.Trial) -> dict[str, Any]:
    split_mode = trial.suggest_categorical("split_mode", ["planner", "fixed", "disabled"])
    fixed_split_pages = (
        int(trial.suggest_categorical("fixed_split_pages", CHUNK_PAGE_LADDER)) if split_mode == "fixed" else 0
    )
    merge_cta_policy = trial.suggest_categorical("merge_cta_policy", ["formula", "absolute"])
    return {
        "split_mode": split_mode,
        "fixed_split_pages": fixed_split_pages,
        "graph_chunk_policy": trial.suggest_categorical("graph_chunk_policy", [False, True]),
        "max_batch_size_scale": float(trial.suggest_categorical("max_batch_size_scale", MAX_BATCH_SCALE_LADDER)),
        "merge_cta_policy": merge_cta_policy,
        "merge_blocks_per_sm_cap": trial.suggest_int("merge_blocks_per_sm_cap", 1, 6),
        "merge_ctas_per_sm": trial.suggest_int("merge_ctas_per_sm", 1, 6),
    }


def _sample_bf16_decode_config(trial: optuna.Trial) -> dict[str, Any]:
    config = _sample_common_config(trial)
    config.update(
        {
            "cta_policy": trial.suggest_categorical("cta_policy", ["planner", "force16", "force64", "force128"]),
            "q64_threshold": trial.suggest_int("q64_threshold", 1, 64),
            "decode_policy": trial.suggest_categorical(
                "decode_policy", ["exact_table", "legacy_table", "binary_search"]
            ),
            "decode_chunk_pages": tuple(
                int(trial.suggest_categorical(f"decode_chunk_pages_le_{bp}", CHUNK_PAGE_LADDER))
                for bp in BF16_DECODE_EXACT_BREAKPOINTS
            ),
        }
    )
    return config


def _sample_bf16_extend_config(trial: optuna.Trial) -> dict[str, Any]:
    config = _sample_common_config(trial)
    config.update(
        {
            "cta_policy": trial.suggest_categorical("cta_policy", ["planner", "force16", "force64", "force128"]),
            "q64_threshold": trial.suggest_int("q64_threshold", 1, 64),
            "extend_policy": trial.suggest_categorical(
                "extend_policy", ["exact_table", "legacy_table", "binary_search"]
            ),
            "extend_chunk_pages": tuple(
                int(trial.suggest_categorical(f"extend_chunk_pages_le_{bp}", CHUNK_PAGE_LADDER))
                for bp in BF16_EXTEND_BREAKPOINTS
            ),
            "bf16_long_form_mode": trial.suggest_categorical(
                "bf16_long_form_mode", ["planner", "force_on", "force_off"]
            ),
            "bf16_long_form_cutoff_tokens": int(
                trial.suggest_categorical("bf16_long_form_cutoff_tokens", LONG_FORM_CUTOFF_TOKENS_LADDER)
            ),
        }
    )
    return config


def _sample_fp8_extend_config(trial: optuna.Trial) -> dict[str, Any]:
    config = _sample_common_config(trial)
    config.update(
        {
            "cta_policy": trial.suggest_categorical("cta_policy", ["planner", "force16", "force64", "force128"]),
            "q64_threshold": trial.suggest_int("q64_threshold", 1, 64),
            "extend_policy": trial.suggest_categorical("extend_policy", ["table", "binary_search"]),
            "extend_chunk_pages": tuple(
                int(trial.suggest_categorical(f"extend_chunk_pages_le_{bp}", CHUNK_PAGE_LADDER))
                for bp in FP8_EXTEND_BREAKPOINTS
            ),
            "fp8_q64_mode": trial.suggest_categorical("fp8_q64_mode", ["planner", "force32", "force48"]),
            "fp8_q32_cutoff_pages": trial.suggest_int("fp8_q32_cutoff_pages", 1, 64),
        }
    )
    return config


def _sample_config(trial: optuna.Trial, spec: TargetSpec) -> dict[str, Any]:
    if spec.family == "bf16_decode":
        return _sample_bf16_decode_config(trial)
    if spec.family == "bf16_extend":
        return _sample_bf16_extend_config(trial)
    if spec.family == "fp8_extend":
        return _sample_fp8_extend_config(trial)
    raise AssertionError(f"unsupported family {spec.family}")


def _baseline_params(spec: TargetSpec) -> dict[str, Any]:
    common = {
        "split_mode": "planner",
        "graph_chunk_policy": True,
        "max_batch_size_scale": 1.0,
        "merge_cta_policy": "formula",
        "merge_blocks_per_sm_cap": 3,
        "merge_ctas_per_sm": 3,
        "cta_policy": "planner",
        "q64_threshold": 16,
    }
    if spec.family == "bf16_decode":
        return {
            **common,
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
        }
    if spec.family == "bf16_extend":
        return {
            **common,
            "extend_policy": "exact_table",
            "extend_chunk_pages_le_1": 1,
            "extend_chunk_pages_le_2": 1,
            "extend_chunk_pages_le_4": 1,
            "extend_chunk_pages_le_8": 1,
            "extend_chunk_pages_le_16": 1,
            "extend_chunk_pages_le_32": 2,
            "extend_chunk_pages_le_64": 3,
            "extend_chunk_pages_le_128": 6,
            "extend_chunk_pages_le_256": 6,
            "extend_chunk_pages_le_512": 32,
            "extend_chunk_pages_le_1024": 32,
            "extend_chunk_pages_le_2048": 32,
            "bf16_long_form_mode": "planner",
            "bf16_long_form_cutoff_tokens": 2048,
        }
    if spec.family == "fp8_extend":
        return {
            **common,
            "extend_policy": "table",
            "extend_chunk_pages_le_1": 1,
            "extend_chunk_pages_le_2": 1,
            "extend_chunk_pages_le_4": 1,
            "extend_chunk_pages_le_8": 1,
            "extend_chunk_pages_le_16": 1,
            "extend_chunk_pages_le_32": 2,
            "extend_chunk_pages_le_64": 3,
            "extend_chunk_pages_le_128": 6,
            "extend_chunk_pages_le_256": 6,
            "extend_chunk_pages_le_512": 24,
            "extend_chunk_pages_le_1024": 24,
            "extend_chunk_pages_le_2048": 24,
            "fp8_q64_mode": "planner",
            "fp8_q32_cutoff_pages": 8,
        }
    raise AssertionError(f"unsupported family {spec.family}")


@contextlib.contextmanager
def _scheduler_overrides(spec: TargetSpec, config: dict[str, Any]):
    orig_fa2_determine = paged_planner._fa2_determine_cta_tile_q
    orig_paged_determine = paged_planner._paged_determine_cta_tile_q
    orig_exact_fn = paged_planner._use_paged_bf16_tma_exact_plane_chunk_tables
    orig_decode_exact = paged_planner._PAGED_DECODE_BF16_TMA_EXACT_PLANE_CHUNK_TABLE_PAGES
    orig_decode_legacy = paged_planner._PAGED_DECODE_BF16_CHUNK_TABLE_PAGES
    orig_extend_bf16_exact = paged_planner._PAGED_EXTEND_BF16_TMA_EXACT_PLANE_CHUNK_TABLE_PAGES
    orig_extend_bf16_legacy = paged_planner._PAGED_EXTEND_BF16_CHUNK_TABLE_PAGES
    orig_extend_fp8 = paged_planner._PAGED_EXTEND_FP8_CHUNK_TABLE_PAGES
    orig_extend_fp8_graph = paged_planner._PAGED_EXTEND_FP8_GRAPH_CHUNK_TABLE_PAGES
    orig_chunk_table_pages = paged_planner._paged_chunk_table_pages
    orig_prefill_binary_search = paged_planner._prefill_binary_search_kv_chunk_size
    orig_workspace_create_plan = paged_workspace.create_paged_plan
    orig_api_default_merge = paged_api.default_paged_persistent_ctas
    orig_merge_default_merge = paged_merge.default_paged_persistent_ctas
    orig_bf16_long_form = paged_api._use_bf16_extend_raw_long_form

    def patched_fa2_determine(avg_packed_qo_len: int, head_dim: int) -> int:
        policy = config["cta_policy"]
        if policy == "force16":
            return 16
        if policy == "force64":
            return 64
        if policy == "force128":
            return 128
        if avg_packed_qo_len > 64 and head_dim < 256:
            return 128
        if avg_packed_qo_len > config["q64_threshold"]:
            return 64
        return 16

    def patched_paged_determine_cta_tile_q(*, mode, kv_dtype, packed_qo_len, head_dim, max_effective_kv_pages):
        cta_tile_q = patched_fa2_determine(packed_qo_len, head_dim)
        if spec.family == "fp8_extend" and mode == "extend" and kv_dtype == torch.float8_e4m3fn and cta_tile_q == 64:
            q64_mode = config["fp8_q64_mode"]
            if q64_mode == "force32":
                return 32
            if q64_mode == "force48":
                return 48
            return 32 if max_effective_kv_pages <= config["fp8_q32_cutoff_pages"] else 48
        return cta_tile_q

    def patched_exact_plane() -> bool:
        if spec.family == "bf16_decode":
            return config["decode_policy"] == "exact_table"
        if spec.family == "bf16_extend":
            return config["extend_policy"] == "exact_table"
        return True

    decode_exact_table = None
    decode_legacy_table = None
    if spec.family == "bf16_decode":
        decode_exact_table = _build_decode_table(exact_plane=True, chunk_pages=config["decode_chunk_pages"])
        decode_legacy_table = _build_decode_table(exact_plane=False, chunk_pages=config["decode_chunk_pages"])

    extend_bf16_table = None
    if spec.family == "bf16_extend":
        extend_bf16_table = _build_extend_table(chunk_pages=config["extend_chunk_pages"])

    extend_fp8_table = None
    if spec.family == "fp8_extend":
        extend_fp8_table = _build_extend_table(chunk_pages=config["extend_chunk_pages"])

    def patched_chunk_table_pages(*, mode, q_dtype, kv_dtype, page_size, head_dim_qk, head_dim_vo, gqa_group_size,
                                  max_effective_kv_pages, graph_chunk_policy):
        if spec.family == "bf16_decode" and mode == "decode" and kv_dtype == torch.bfloat16:
            if config["decode_policy"] == "binary_search":
                return None
        if spec.family == "bf16_extend" and mode == "extend" and kv_dtype == torch.bfloat16:
            if config["extend_policy"] == "binary_search":
                return None
        if spec.family == "fp8_extend" and mode == "extend" and kv_dtype == torch.float8_e4m3fn:
            if config["extend_policy"] == "binary_search":
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
        scaled_budget = max(1, int(round(max_batch_size_if_split * config["max_batch_size_scale"])))
        return orig_prefill_binary_search(
            enable_cuda_graph=enable_cuda_graph,
            max_batch_size_if_split=scaled_budget,
            packed_qo_len_arr=packed_qo_len_arr,
            kv_len_arr=kv_len_arr,
            qo_chunk_size=qo_chunk_size,
            min_kv_chunk_size=min_kv_chunk_size,
        )

    def patched_workspace_create_plan(*args, **kwargs):
        split_mode = config["split_mode"]
        if split_mode == "disabled":
            kwargs["disable_split_kv"] = True
            kwargs["fixed_split_size"] = -1
        elif split_mode == "fixed":
            kwargs["disable_split_kv"] = False
            kwargs["fixed_split_size"] = config["fixed_split_pages"]
        else:
            kwargs["disable_split_kv"] = False
            kwargs["fixed_split_size"] = -1
        kwargs["graph_chunk_policy"] = config["graph_chunk_policy"]
        return orig_workspace_create_plan(*args, **kwargs)

    def patched_default_persistent_ctas(*, total_rows: int, num_heads: int, device=None) -> int:
        if device is None:
            device = torch.cuda.current_device()
        num_sms = int(torch.cuda.get_device_properties(device).multi_processor_count)
        if config["merge_cta_policy"] == "absolute":
            return int(num_sms * max(config["merge_ctas_per_sm"], 1))
        total_work = max(int(total_rows) * int(num_heads), 1)
        blocks_per_sm = min(config["merge_blocks_per_sm_cap"], math.ceil(total_work / num_sms))
        return int(num_sms * max(blocks_per_sm, 1))

    def patched_bf16_long_form(kv_chunk_size: int) -> bool:
        mode = config.get("bf16_long_form_mode", "planner")
        if mode == "force_on":
            return True
        if mode == "force_off":
            return False
        return kv_chunk_size < int(config.get("bf16_long_form_cutoff_tokens", 2048))

    paged_planner._fa2_determine_cta_tile_q = patched_fa2_determine
    paged_planner._paged_determine_cta_tile_q = patched_paged_determine_cta_tile_q
    paged_planner._use_paged_bf16_tma_exact_plane_chunk_tables = patched_exact_plane
    if decode_exact_table is not None:
        paged_planner._PAGED_DECODE_BF16_TMA_EXACT_PLANE_CHUNK_TABLE_PAGES = decode_exact_table
        paged_planner._PAGED_DECODE_BF16_CHUNK_TABLE_PAGES = decode_legacy_table
    if extend_bf16_table is not None:
        paged_planner._PAGED_EXTEND_BF16_TMA_EXACT_PLANE_CHUNK_TABLE_PAGES = extend_bf16_table
        paged_planner._PAGED_EXTEND_BF16_CHUNK_TABLE_PAGES = extend_bf16_table
        paged_api._use_bf16_extend_raw_long_form = patched_bf16_long_form
    if extend_fp8_table is not None:
        paged_planner._PAGED_EXTEND_FP8_CHUNK_TABLE_PAGES = extend_fp8_table
        paged_planner._PAGED_EXTEND_FP8_GRAPH_CHUNK_TABLE_PAGES = extend_fp8_table
    paged_planner._paged_chunk_table_pages = patched_chunk_table_pages
    paged_planner._prefill_binary_search_kv_chunk_size = patched_prefill_binary_search_kv_chunk_size
    paged_workspace.create_paged_plan = patched_workspace_create_plan
    paged_api.default_paged_persistent_ctas = patched_default_persistent_ctas
    paged_merge.default_paged_persistent_ctas = patched_default_persistent_ctas
    try:
        yield
    finally:
        paged_planner._fa2_determine_cta_tile_q = orig_fa2_determine
        paged_planner._paged_determine_cta_tile_q = orig_paged_determine
        paged_planner._use_paged_bf16_tma_exact_plane_chunk_tables = orig_exact_fn
        paged_planner._PAGED_DECODE_BF16_TMA_EXACT_PLANE_CHUNK_TABLE_PAGES = orig_decode_exact
        paged_planner._PAGED_DECODE_BF16_CHUNK_TABLE_PAGES = orig_decode_legacy
        paged_planner._PAGED_EXTEND_BF16_TMA_EXACT_PLANE_CHUNK_TABLE_PAGES = orig_extend_bf16_exact
        paged_planner._PAGED_EXTEND_BF16_CHUNK_TABLE_PAGES = orig_extend_bf16_legacy
        paged_planner._PAGED_EXTEND_FP8_CHUNK_TABLE_PAGES = orig_extend_fp8
        paged_planner._PAGED_EXTEND_FP8_GRAPH_CHUNK_TABLE_PAGES = orig_extend_fp8_graph
        paged_planner._paged_chunk_table_pages = orig_chunk_table_pages
        paged_planner._prefill_binary_search_kv_chunk_size = orig_prefill_binary_search
        paged_workspace.create_paged_plan = orig_workspace_create_plan
        paged_api.default_paged_persistent_ctas = orig_api_default_merge
        paged_merge.default_paged_persistent_ctas = orig_merge_default_merge
        paged_api._use_bf16_extend_raw_long_form = orig_bf16_long_form


def _build_trial_inputs(spec: TargetSpec, seed: int):
    q, k_cache, v_cache, page_table, cache_seqlens, cu_seqlens_q = bench._make_uniform_paged_inputs(
        batch=spec.batch,
        q_seqlen=spec.q_seqlen,
        cache_seqlen=spec.cache_seqlen,
        page_size=spec.page_size,
        q_heads=spec.q_heads,
        kv_heads=spec.kv_heads,
        head_dim=spec.head_dim,
        dtype=spec.q_dtype,
        seed=seed,
    )
    k_descale = None
    v_descale = None
    k_scale = None
    v_scale = None
    if spec.kv_dtype == torch.float8_e4m3fn:
        k_cache, v_cache, k_descale, v_descale, k_scale, v_scale = bench._quantize_paged_kv_cache_global_e4m3(
            k_cache,
            v_cache,
            batch=spec.batch,
            kv_heads=spec.kv_heads,
        )
    return q, k_cache, v_cache, page_table, cache_seqlens, cu_seqlens_q, k_descale, v_descale, k_scale, v_scale


def _capture_b12x_graph(
    *,
    spec: TargetSpec,
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    page_table: torch.Tensor,
    cache_seqlens: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    k_descale: torch.Tensor | None,
    v_descale: torch.Tensor | None,
    warmup: int,
):
    output = torch.empty_like(q)
    workspace = paged_workspace.PagedAttentionWorkspace.for_tensors(
        mode=spec.mode,
        q=q,
        k_cache=k_cache,
        v_cache=v_cache,
        use_cuda_graph=True,
        attn_mode="default",
    )
    workspace.prepare(page_table, cache_seqlens, cu_seqlens_q)

    def run() -> None:
        workspace.run(q, k_cache, v_cache, output=output, k_descale=k_descale, v_descale=v_descale)

    graph = bench._capture_graph(run, warmup=warmup)
    return graph, output, workspace.plan


def _capture_reference_output(
    *,
    spec: TargetSpec,
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    page_table: torch.Tensor,
    cache_seqlens: torch.Tensor,
    k_scale: float | None,
    v_scale: float | None,
):
    fa_graph, fa_output = bench._capture_flashinfer_fa2_graph(
        q=q,
        k_cache=k_cache,
        v_cache=v_cache,
        page_table=page_table,
        cache_seqlens=cache_seqlens,
        q_seqlen=spec.q_seqlen,
        page_size=spec.page_size,
        q_heads=spec.q_heads,
        kv_heads=spec.kv_heads,
        head_dim=spec.head_dim,
        q_dtype=spec.q_dtype,
        kv_dtype=spec.kv_dtype,
        k_scale=k_scale,
        v_scale=v_scale,
        workspace_bytes=512 * 1024 * 1024,
        warmup=1,
    )
    fa_graph.replay()
    torch.cuda.synchronize()
    return fa_output


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
    spec: TargetSpec,
    config: dict[str, Any],
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    page_table: torch.Tensor,
    cache_seqlens: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    k_descale: torch.Tensor | None,
    v_descale: torch.Tensor | None,
    fa_output: torch.Tensor,
    warmup: int,
    replays: int,
    cos_threshold: float,
) -> TrialResult:
    with _scheduler_overrides(spec, config):
        clear_attention_caches()
        b12x_graph, b12x_output, plan = _capture_b12x_graph(
            spec=spec,
            q=q,
            k_cache=k_cache,
            v_cache=v_cache,
            page_table=page_table,
            cache_seqlens=cache_seqlens,
            cu_seqlens_q=cu_seqlens_q,
            k_descale=k_descale,
            v_descale=v_descale,
            warmup=warmup,
        )
        b12x_mean_us, b12x_ci_low_us, b12x_ci_high_us, b12x_sem_us = _bench_backend_mean_us(
            b12x_graph,
            replays=replays,
        )
        max_abs = float((b12x_output - fa_output).abs().max().item())
        cos = float(bench._cosine_similarity(b12x_output, fa_output))
        score = 0.0 if cos < cos_threshold else (1.0 / b12x_mean_us)
        return TrialResult(
            score=score,
            b12x_mean_us=b12x_mean_us,
            b12x_ci_low_us=b12x_ci_low_us,
            b12x_ci_high_us=b12x_ci_high_us,
            b12x_sem_us=b12x_sem_us,
            plan_desc=f"chunk={plan.kv_chunk_size},{'split' if plan.split_kv else 'nosplit'}",
            cta_tile_q=int(plan.cta_tile_q),
            kv_chunk_size=int(plan.kv_chunk_size),
            split_kv=bool(plan.split_kv),
            max_abs=max_abs,
            cos=cos,
        )


def _make_objective(
    *,
    spec: TargetSpec,
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    page_table: torch.Tensor,
    cache_seqlens: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    k_descale: torch.Tensor | None,
    v_descale: torch.Tensor | None,
    fa_output: torch.Tensor,
    warmup: int,
    replays: int,
    cos_threshold: float,
):
    def objective(trial: optuna.Trial) -> float:
        config = _sample_config(trial, spec)
        try:
            result = _run_trial(
                spec=spec,
                config=config,
                q=q,
                k_cache=k_cache,
                v_cache=v_cache,
                page_table=page_table,
                cache_seqlens=cache_seqlens,
                cu_seqlens_q=cu_seqlens_q,
                k_descale=k_descale,
                v_descale=v_descale,
                fa_output=fa_output,
                warmup=warmup,
                replays=replays,
                cos_threshold=cos_threshold,
            )
        except Exception as exc:
            trial.set_user_attr("status", "crash")
            trial.set_user_attr("error", f"{type(exc).__name__}: {exc}")
            trial.set_user_attr("traceback", traceback.format_exc(limit=20))
            trial.set_user_attr("score", 0.0)
            return 0.0

        trial.set_user_attr("status", "ok" if result.cos >= cos_threshold else "bad_cos")
        trial.set_user_attr("score", result.score)
        trial.set_user_attr("plan_desc", result.plan_desc)
        trial.set_user_attr("cta_tile_q", result.cta_tile_q)
        trial.set_user_attr("kv_chunk_size", result.kv_chunk_size)
        trial.set_user_attr("split_kv", result.split_kv)
        trial.set_user_attr("b12x_mean_us", result.b12x_mean_us)
        trial.set_user_attr("b12x_ci_low_us", result.b12x_ci_low_us)
        trial.set_user_attr("b12x_ci_high_us", result.b12x_ci_high_us)
        trial.set_user_attr("b12x_sem_us", result.b12x_sem_us)
        trial.set_user_attr("max_abs", result.max_abs)
        trial.set_user_attr("cos", result.cos)
        return result.score

    return objective


def _print_top_trials(study: optuna.Study, *, limit: int) -> None:
    complete = [trial for trial in study.trials if trial.state == optuna.trial.TrialState.COMPLETE]
    complete.sort(key=lambda trial: float(trial.user_attrs.get("b12x_mean_us", float("inf"))))
    print(f"top {min(limit, len(complete))} trials:")
    for trial in complete[:limit]:
        print(
            {
                "trial": trial.number,
                "score": round(float(trial.value), 9),
                "b12x_mean_us": trial.user_attrs.get("b12x_mean_us"),
                "b12x_ci_low_us": trial.user_attrs.get("b12x_ci_low_us"),
                "b12x_ci_high_us": trial.user_attrs.get("b12x_ci_high_us"),
                "plan": trial.user_attrs.get("plan_desc"),
                "cta_tile_q": trial.user_attrs.get("cta_tile_q"),
                "kv_chunk_size": trial.user_attrs.get("kv_chunk_size"),
                "split_kv": trial.user_attrs.get("split_kv"),
                "cos": trial.user_attrs.get("cos"),
                "params": trial.params,
            }
        )


def _build_target_from_args(args: argparse.Namespace) -> TargetSpec:
    return TargetSpec(
        mode=args.mode,
        kv_dtype=_parse_kv_dtype(args.kv_dtype),
        batch=args.batch,
        q_seqlen=args.q_seqlen,
        cache_seqlen=args.cache_seqlen,
        page_size=args.page_size,
        q_heads=args.q_heads,
        kv_heads=args.kv_heads,
        head_dim=args.head_dim,
        q_dtype=torch.bfloat16,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["decode", "extend"], default="decode")
    parser.add_argument("--kv-dtype", choices=["bf16", "fp8_e4m3fn"], default="bf16")
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--q-seqlen", type=int, default=1)
    parser.add_argument("--cache-seqlen", type=int, default=2048)
    parser.add_argument("--page-size", type=int, default=64)
    parser.add_argument("--q-heads", type=int, default=8)
    parser.add_argument("--kv-heads", type=int, default=1)
    parser.add_argument("--head-dim", type=int, default=256)
    parser.add_argument("--trials", type=int, default=DEFAULT_TRIALS)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--warmup", type=int, default=DEFAULT_WARMUP)
    parser.add_argument("--replays", type=int, default=DEFAULT_REPLAYS)
    parser.add_argument("--cos-threshold", type=float, default=DEFAULT_COS_THRESHOLD)
    parser.add_argument("--study-name", type=str, default=None)
    parser.add_argument("--journal-path", type=str, default=None)
    parser.add_argument("--enqueue-baseline", action="store_true", default=True)
    parser.add_argument("--no-enqueue-baseline", action="store_false", dest="enqueue_baseline")
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--timeout", type=int, default=0)
    args = parser.parse_args()

    bench.require_sm120()
    if args.replays < 100:
        raise ValueError("--replays must be at least 100")

    spec = _build_target_from_args(args)
    if spec.mode == "decode" and spec.q_seqlen != 1:
        raise ValueError("decode target must use q_seqlen=1")
    if spec.mode == "extend" and spec.q_seqlen <= 1:
        raise ValueError("extend target must use q_seqlen > 1")

    study_name = args.study_name or _study_name_for_target(spec)
    journal_path = pathlib.Path(args.journal_path) if args.journal_path else _journal_path_for_target(spec)
    journal_path.parent.mkdir(parents=True, exist_ok=True)

    print(
        {
            "target": spec.name,
            "family": spec.family,
            "study_name": study_name,
            "journal_path": str(journal_path),
            "replays": args.replays,
            "trials": args.trials,
        }
    )

    q, k_cache, v_cache, page_table, cache_seqlens, cu_seqlens_q, k_descale, v_descale, k_scale, v_scale = (
        _build_trial_inputs(spec, args.seed)
    )
    clear_attention_caches()
    fa_output = _capture_reference_output(
        spec=spec,
        q=q,
        k_cache=k_cache,
        v_cache=v_cache,
        page_table=page_table,
        cache_seqlens=cache_seqlens,
        k_scale=k_scale,
        v_scale=v_scale,
    )

    sampler = optuna.samplers.TPESampler(
        seed=args.seed,
        multivariate=True,
        group=True,
        n_startup_trials=20,
        constant_liar=True,
    )
    journal_backend = optuna.storages.journal.JournalFileBackend(str(journal_path))
    storage = optuna.storages.JournalStorage(journal_backend)
    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        direction="maximize",
        sampler=sampler,
        load_if_exists=True,
    )
    if args.enqueue_baseline:
        study.enqueue_trial(_baseline_params(spec))

    objective = _make_objective(
        spec=spec,
        q=q,
        k_cache=k_cache,
        v_cache=v_cache,
        page_table=page_table,
        cache_seqlens=cache_seqlens,
        cu_seqlens_q=cu_seqlens_q,
        k_descale=k_descale,
        v_descale=v_descale,
        fa_output=fa_output,
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
            "score": round(float(best.value), 9),
            "b12x_mean_us": best.user_attrs.get("b12x_mean_us"),
            "b12x_ci_low_us": best.user_attrs.get("b12x_ci_low_us"),
            "b12x_ci_high_us": best.user_attrs.get("b12x_ci_high_us"),
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
