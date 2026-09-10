#!/usr/bin/env python3
"""NVFP4 split-materialized vs monolithic MoE prefill A/B receipt.

Both arms drive ONE traced call through the production ``@cute.jit`` host
adapter ``_DynamicMoELaunch`` (the same adapter ``b12x_moe_fp4`` reaches for
the dynamic recipe), so the measured path is the serving path, not a packed
or reference fallback.  The monolithic arm compiles the cooperative fused
kernel; the split arm compiles the route/pack front-end plus the external
``Nvfp4MaterializedPhase1Kernel`` / ``Nvfp4MaterializedPhase2Kernel``.  Both
arms share identical expert payloads, routed inputs, and oracle, and differ
only by ``materialize_intermediate``.

The split specialization is engaged only for the ``mma_tiler_mn == (128, 128)``
regime (validated by ``MoEDynamicKernelBackend.nvfp4_split_materialized``);
small-M tiles that fall back to the monolithic kernel are not benchmarked here.

Evidence contract (AGENTS.md): capture the command, commit, worktree, GPU mode,
correctness state, raw timing samples, and ratio direction.  A shape whose
correctness gate fails is recorded with the gate outcome and its timings are
kept OUT of the headline speedup.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from benchmarks.common import nvidia_smi_gpu_mode_snapshot
from tests.moe.test_nvfp4_phase_kernels import (
    _bf16_output_bound,
    _build_domain,
)
from tests.moe.test_nvfp4_split_backend import (
    _compile_launch,
    _make_launcher,
    _split_workspace,
)
from b12x.moe._shared.kernels.reference import compare_to_reference


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args],
        capture_output=True,
        text=True,
        cwd=ROOT,
    ).stdout.strip()


def _source_hashes() -> dict[str, str]:
    tracked = [
        "b12x/moe/_shared/kernels/dynamic.py",
        "b12x/moe/_shared/kernels/nvfp4_phase1.py",
        "b12x/moe/_shared/kernels/nvfp4_phase2.py",
        "b12x/moe/fused_moe/_impl.py",
        "b12x/_lib/intrinsics.py",
        "tests/moe/test_nvfp4_phase_kernels.py",
        "tests/moe/test_nvfp4_split_backend.py",
        "benchmarks/benchmark_nvfp4_split_materialized.py",
    ]
    out = {}
    for rel in tracked:
        path = ROOT / rel
        out[rel] = (
            hashlib.sha256(path.read_bytes()).hexdigest()
            if path.exists()
            else None
        )
    return out


def _package_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {"torch": torch.__version__}
    try:
        import cutlass

        versions["nvidia-cutlass-dsl"] = getattr(cutlass, "__version__", None)
    except Exception:
        versions["nvidia-cutlass-dsl"] = None
    return versions


def _time_launch(launch, *, warmup: int, iterations: int) -> list[float]:
    for _ in range(warmup):
        launch()
    torch.cuda.synchronize()
    samples_us: list[float] = []
    for _ in range(iterations):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        launch()
        end.record()
        torch.cuda.synchronize()
        samples_us.append(start.elapsed_time(end) * 1e3)
    return samples_us


def _sample_under_load(launch) -> dict:
    """Read the GPU operating state while a launch burst keeps clocks up."""
    import threading

    holder: dict = {}

    def _snap():
        holder["snap"] = nvidia_smi_gpu_mode_snapshot()

    # Burn launches, start the blocking nvidia-smi query mid-burst so the
    # returned pstate/SM clock reflects sustained active prefill, not idle.
    for _ in range(20):
        launch()
    t = threading.Thread(target=_snap)
    t.start()
    for _ in range(20):
        launch()
    t.join()
    torch.cuda.synchronize()
    return holder.get("snap", {"available": False})


def _build_arm(*, materialize: bool, domain):
    """Compile one arm once and return its reusable launch closure + workspace."""
    ws = _split_workspace(domain)
    compiled = _compile_launch(
        domain,
        materialize=materialize,
        spec_name=f"bench.nvfp4_split.{'split' if materialize else 'mono'}",
    )
    launch = _make_launcher(compiled, domain, ws)
    return launch, ws


def _time_arm(launch, ws) -> tuple[torch.Tensor, list[float]]:
    launch()
    torch.cuda.synchronize()
    out = ws["scatter_output"].clone()
    samples = _time_launch(launch, warmup=ARGS.warmup, iterations=ARGS.iters)
    return out, samples


def _engaged(domain) -> bool:
    """Confirm this shape resolves to the split specialization (M128 regime)."""
    from b12x.moe._shared.kernels.dynamic import MoEDynamicKernelBackend

    kernel = MoEDynamicKernelBackend(
        16,
        (128, 128),
        quant_recipe="nvfp4",
        activation="silu",
        share_input_across_experts=True,
        num_topk=domain["top_k"],
        materialize_intermediate=True,
        deterministic_output=False,
    )
    return bool(kernel.nvfp4_split_materialized)


ARGS: argparse.Namespace


def main() -> None:
    global ARGS
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--shape",
        action="append",
        type=str,
        default=None,
        help="E:K:n:topk:M (repeatable); defaults cover NVFP4 prefill bands",
    )
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=99)
    parser.add_argument("--output", type=Path, required=True)
    ARGS = parser.parse_args()

    shapes = ARGS.shape or [
        "8:4096:2048:2:2048",
        "8:4096:2048:2:4096",
        "8:4096:2048:2:8192",
        "64:4096:1024:8:2048",
        "64:4096:1024:8:4096",
        "64:4096:1024:8:8192",
    ]

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")

    report = {
        "schema": "b12x.moe.nvfp4_split_materialized.benchmark",
        "version": 1,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "command": " ".join(
            [Path(sys.argv[0]).name, *sys.argv[1:]]
        ),
        "commit": _git("rev-parse", "HEAD"),
        "commit_short": _git("rev-parse", "--short", "HEAD"),
        "worktree_status": _git("status", "--porcelain") or "clean",
        "branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "gpu_name": torch.cuda.get_device_name(0),
        "gpu_snapshot": nvidia_smi_gpu_mode_snapshot(),
        "package_versions": _package_versions(),
        "source_sha256": _source_hashes(),
        "env": {
            "B12X_NVFP4_DYNAMIC_MATERIALIZED": os.environ.get(
                "B12X_NVFP4_DYNAMIC_MATERIALIZED"
            ),
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
        },
        "iters": ARGS.iters,
        "warmup": ARGS.warmup,
        "rounds": ARGS.rounds,
        "cases": [],
    }

    def save() -> None:
        ARGS.output.parent.mkdir(parents=True, exist_ok=True)
        tmp = ARGS.output.with_suffix(ARGS.output.suffix + ".tmp")
        tmp.write_text(json.dumps(report, indent=2))
        os.replace(tmp, ARGS.output)

    ratios = []
    for spec in shapes:
        E, K, n, top_k, m = (int(v) for v in spec.split(":"))
        domain = _build_domain(E=E, K=K, n=n, m=m, top_k=top_k, seed=ARGS.seed)
        engaged = _engaged(domain)
        case = {
            "shape": {"E": E, "K": K, "n": n, "top_k": top_k, "M": m},
            "routed_rows": m * top_k,
            "split_engaged": engaged,
            "correctness": {},
            "samples_us": {"monolithic": [], "split": []},
        }
        report["cases"].append(case)
        if not engaged:
            print(f"{spec}: split NOT engaged (falls back to monolithic), skipped")
            save()
            continue

        # Compile each arm once, then interleave timed rounds so thermal/clock
        # drift hits both arms equally.  The cooperative front-end re-zeros its
        # own volatile launch state, so repeated launches on the cached closure
        # are valid measurements.
        mono_launch, mono_ws = _build_arm(materialize=False, domain=domain)
        split_launch, split_ws = _build_arm(materialize=True, domain=domain)
        split_out = mono_out = None
        for _r in range(ARGS.rounds):
            mono_out, mono_samples = _time_arm(mono_launch, mono_ws)
            split_out, split_samples = _time_arm(split_launch, split_ws)
            case["samples_us"]["monolithic"].append(mono_samples)
            case["samples_us"]["split"].append(split_samples)
        # Capture the operating state under sustained load: the nvidia-smi
        # query itself takes ~100ms, during which clocks decay if we sample
        # after synchronize.  Run the query in a thread while continuing to
        # launch, so the reported pstate/SM clock reflect active prefill.
        case["gpu_mode_active"] = _sample_under_load(split_launch)

        # Correctness gate on the final outputs (kept separate from timing).
        bound = _bf16_output_bound(domain["oracle"])
        mono_metrics = compare_to_reference(mono_out.float(), domain["oracle"])
        split_metrics = compare_to_reference(split_out.float(), domain["oracle"])
        split_vs_mono = compare_to_reference(split_out.float(), mono_out.float())
        case["correctness"] = {
            "bound_abs": bound,
            "monolithic": asdict(mono_metrics),
            "split": asdict(split_metrics),
            "split_vs_monolithic": asdict(split_vs_mono),
            "nonzero_split": int(split_out.count_nonzero()) > 0,
            "nonzero_monolithic": int(mono_out.count_nonzero()) > 0,
        }
        # Qualification uses the global gates the repo relies on as primary for
        # NVFP4: cosine direction and global rmse.  The per-element max_abs is
        # recorded as a diagnostic only: FP4 intermediate requantization noise
        # grows with routed-row count on the single worst BF16 element, so a
        # hard max_abs gate would reject valid large-shape prefill (the
        # monolithic production arm trips it too).  A genuinely broken path
        # still fails via cos, rmse, or the split-vs-monolithic agreement.
        case["correctness"]["split_passed"] = bool(
            split_metrics.cos > 0.9999 and split_metrics.rmse <= bound
        )
        case["correctness"]["monolithic_passed"] = bool(
            mono_metrics.cos > 0.9999 and mono_metrics.rmse <= bound
        )
        case["correctness"]["passed"] = bool(
            case["correctness"]["split_passed"]
            and case["correctness"]["monolithic_passed"]
            and split_vs_mono.cos > 0.9999
            and split_vs_mono.rmse <= bound
        )

        # Per-case median over iterations; ratio uses medians.  A failed
        # correctness case contributes NO ratio to the headline.
        mono_med = statistics.median(
            statistics.median(s) for s in case["samples_us"]["monolithic"]
        )
        split_med = statistics.median(
            statistics.median(s) for s in case["samples_us"]["split"]
        )
        case["median_us"] = {
            "monolithic": mono_med,
            "split": split_med,
        }
        if case["correctness"]["passed"]:
            ratio = mono_med / split_med
            case["speedup_split_over_mono"] = ratio
            ratios.append(ratio)
        else:
            case["speedup_split_over_mono"] = None
        save()
        print(
            f"{spec}: split={split_med:.1f}us mono={mono_med:.1f}us "
            f"speedup={case['speedup_split_over_mono'] and round(case['speedup_split_over_mono'], 3)} "
            f"correct={case['correctness']['passed']}",
            flush=True,
        )

    report["summary"] = {
        "qualified_speedups": len(ratios),
        "geomean_speedup_split_over_mono": (
            statistics.geometric_mean(ratios) if ratios else None
        ),
        "ratio_direction": "monolithic_us / split_us (>1.0 means split is faster)",
        "note": "Cases whose correctness gate failed are excluded from the "
        "headline speedup and retained only as diagnostics.",
    }
    save()
    print(f"\ngeomean split-over-mono speedup = "
          f"{report['summary']['geomean_speedup_split_over_mono']} "
          f"across {report['summary']['qualified_speedups']} qualified shapes")
    print(f"wrote {ARGS.output}")


if __name__ == "__main__":
    main()
