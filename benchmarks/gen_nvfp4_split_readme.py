#!/usr/bin/env python3
"""Render the NVFP4 split-materialized evidence README from the JSON receipt."""
from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path


def main() -> None:
    src = Path(sys.argv[1])
    dst = Path(sys.argv[2])
    r = json.loads(src.read_text())
    lines: list[str] = []
    a = lines.append

    snap = r.get("gpu_snapshot", {})
    # Prefer the first engaged case that actually carries an active-snapshot
    # reading; cases[0] may be a skipped shape or a correctness-failed case
    # whose timings (and therefore the under-load snapshot) were withheld.
    active = next(
        (
            c["gpu_mode_active"]
            for c in r["cases"]
            if c.get("split_engaged") and c.get("gpu_mode_active", {}).get("fields")
        ),
        {},
    )
    af = active.get("fields", {})
    a("# NVFP4 split-materialized prefill — evidence")
    a("")
    a(f"- **Date (UTC):** {r['generated_utc']}")
    a(f"- **Commit:** `{r['commit']}` (short `{r['commit_short']}`, branch `{r['branch']}`)")
    a(f"- **Worktree:** `{r['worktree_status']}`")
    a(f"- **GPU:** {r['gpu_name']}")
    a(f"- **Active GPU mode (P-state / SM / mem clock / temp):** "
      f"{af.get('pstate')} / {af.get('clocks.current.sm')} MHz / "
      f"{af.get('clocks.current.memory')} MHz / {af.get('temperature.gpu')} °C "
      f"(sampled under sustained launch)")
    a(f"- **Idle snapshot:** pstate={snap.get('fields', {}).get('pstate')} "
      f"(clocks not locked; not root)")
    a(f"- **Package versions:** {json.dumps(r['package_versions'])}")
    a(f"- **`B12X_NVFP4_DYNAMIC_MATERIALIZED`:** {r['env'].get('B12X_NVFP4_DYNAMIC_MATERIALIZED')!r} "
      f"(unset → auto-enable for matching shapes)")
    a(f"- **Per-arm samples:** {r['iters']} iterations × {r['rounds']} interleaved rounds, "
      f"{r['warmup']} warmup; CUDA-event timing; medians reported.")
    a(f"- **Command:** `{r['command']}`")
    a("")
    a("## Path")
    a("")
    a("Both arms drive one traced call through the production `@cute.jit` host "
      "adapter `_DynamicMoELaunch` (the adapter `b12x_moe_fp4` reaches for the "
      "dynamic recipe). The monolithic arm compiles the cooperative fused kernel; "
      "the split arm compiles the route/pack front-end plus the external "
      "`Nvfp4MaterializedPhase1Kernel` / `Nvfp4MaterializedPhase2Kernel`. Identical "
      "expert payloads, routed inputs, and `moe_reference_nvfp4` oracle; they differ "
      "only by `materialize_intermediate`. The split specialization engages only in "
      "the `mma_tiler_mn == (128, 128)` regime.")
    a("")
    a("## Correctness gate")
    a("")
    a("Qualification requires, per shape: split and monolithic cosine > 0.9999 vs "
      "the oracle, global RMSE within the BF16 bound, and split-vs-monolithic cosine "
      "> 0.9999 with RMSE within bound. The per-element `max_abs` is recorded as a "
      "diagnostic only: FP4 intermediate requantization noise grows with routed-row "
      "count on the single worst BF16 element, so a hard `max_abs` gate would reject "
      "valid large-M prefill (the monolithic production arm trips it too). Shapes "
      "that fail the gate are excluded from the headline speedup.")
    a("")
    a("## Results (ratio = monolithic_us / split_us; >1.0 means split is faster)")
    a("")
    a("| E | K | n | top_k | M | routed rows | split med (us) | mono med (us) | speedup | correct |")
    a("|---|---|---|-------|---|-------------|----------------|---------------|---------|---------|")
    for c in r["cases"]:
        s = c["shape"]
        med = c.get("median_us", {})
        sp = c.get("speedup_split_over_mono")
        if c.get("split_engaged") and med:
            a(f"| {s['E']} | {s['K']} | {s['n']} | {s['top_k']} | {s['M']} | "
              f"{c['routed_rows']} | {med['split']:.1f} | {med['monolithic']:.1f} | "
              f"{(f'{sp:.2f}x' if sp else '—')} | "
              f"{'PASS' if c['correctness']['passed'] else 'FAIL'} |")
        elif not c.get("split_engaged"):
            a(f"| {s['E']} | {s['K']} | {s['n']} | {s['top_k']} | {s['M']} | "
              f"{c['routed_rows']} | — | — | — | split not engaged |")
    a("")
    summ = r["summary"]
    a(f"**Geomean split-over-monolithic speedup over {summ['qualified_speedups']} "
      f"qualified shapes: "
      f"{summ['geomean_speedup_split_over_mono'] and round(summ['geomean_speedup_split_over_mono'], 3)}x**")
    a("")
    a("## Scope")
    a("")
    a("This receipt covers the split's target regime only: the large-M "
      "M128-tile prefill band (routed rows 4096–65536) where the split "
      "specialization engages. Small-M tiles fall back to the monolithic kernel "
      "and are intentionally not measured here, so the geomean is a "
      "target-regime figure, not a whole-workload average. The oracle is the "
      "repo's `moe_reference_nvfp4` on synthetic quantized weights, not a "
      "checkpoint decode; treat this as kernel-path evidence for the dispatch "
      "decision and re-measure on the target checkpoint before quoting a "
      "serving number. No other receipt's figures are cited or compared here.")
    a("")
    a("## Source artifact hashes (SHA-256)")
    a("")
    for path, h in sorted(r["source_sha256"].items()):
        a(f"- `{path}`: {h}")
    a("")
    a("## Raw data")
    a("")
    a(f"Per-shape raw timing samples (all {r['rounds']} rounds × {r['iters']} iters) "
      "and full correctness metrics are in the companion JSON next to this file.")
    dst.write_text("\n".join(lines) + "\n")
    print(f"wrote {dst}")


if __name__ == "__main__":
    main()
