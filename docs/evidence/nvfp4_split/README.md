# NVFP4 split-materialized prefill — evidence

- **Date (UTC):** 2026-09-10T12:36:15.873407+00:00
- **Commit:** `9a03a9a9093ddd876228a92e1a4535cda7aa80f5` (short `9a03a9a9`, branch `feat/nvfp4-split-prefill`)
- **Worktree (at launch):** `M benchmarks/benchmark_nvfp4_split_materialized.py; M benchmarks/gen_nvfp4_split_readme.py; M docs/evidence/nvfp4_split/20260910-rtx5090-nvfp4-split-prefill.json; M docs/evidence/nvfp4_split/README.md; M tests/moe/test_nvfp4_phase_kernels.py; M tests/moe/test_nvfp4_split_dispatch_policy.py; ?? .zcode/`
- **GPU:** NVIDIA GeForce RTX 5090
- **Active GPU mode (P-state / SM / mem clock / temp):** P1 / 2917 MHz / 13801 MHz / 32 °C (sampled under sustained launch)
- **Idle snapshot:** pstate=P8 (clocks not locked; not root)
- **Package versions:** {"torch": "2.13.0+cu130", "nvidia-cutlass-dsl": "4.6.2"}
- **`B12X_NVFP4_DYNAMIC_MATERIALIZED`:** None (unset → auto-enable for matching shapes)
- **Per-arm samples:** 100 iterations × 5 interleaved rounds, 20 warmup; CUDA-event timing; medians reported.
- **Command:** `python benchmarks/benchmark_nvfp4_split_materialized.py --iters 100 --warmup 20 --rounds 5 --output docs/evidence/nvfp4_split/20260910-rtx5090-nvfp4-split-prefill.json`
- **argv:** `["python", "benchmarks/benchmark_nvfp4_split_materialized.py", "--iters", "100", "--warmup", "20", "--rounds", "5", "--output", "docs/evidence/nvfp4_split/20260910-rtx5090-nvfp4-split-prefill.json"]`

## Path

Both arms drive one traced call through the production `@cute.jit` host adapter `_DynamicMoELaunch` (the adapter `b12x_moe_fp4` reaches for the dynamic recipe). The monolithic arm compiles the cooperative fused kernel; the split arm compiles the route/pack front-end plus the external `Nvfp4MaterializedPhase1Kernel` / `Nvfp4MaterializedPhase2Kernel`. Identical expert payloads, routed inputs, and `moe_reference_nvfp4` oracle; they differ only by `materialize_intermediate`. The split specialization engages only in the `mma_tiler_mn == (128, 128)` regime.

## Correctness gate

Per shape, all of the following must hold before any timing is recorded: split and monolithic outputs are finite and nonzero; split and monolithic cosine > 0.9999 versus the `moe_reference_nvfp4` oracle; split-vs-monolithic cosine > 0.9999 and split-vs-monolithic RMSE within the bound; and each arm's global RMSE ≤ `max(8e-4, 3 · 2⁻⁸ · max|oracle|)`. The bound is computed by `_bf16_output_bound` in `tests/moe/test_nvfp4_phase_kernels.py`; the per-shape value is recorded as `bound_abs` in the companion JSON. The per-element `max_abs` is recorded as a diagnostic only: FP4 intermediate requantization noise grows with routed-row count on the single worst BF16 element, so a hard `max_abs` gate would reject valid large-M prefill (the monolithic production arm trips it too). A shape that fails this gate is shown as FAIL with no timings and is excluded from the headline speedup.

## Results (ratio = monolithic_us / split_us; >1.0 means split is faster)

| E | K | n | top_k | M | routed rows | split med (us) | mono med (us) | speedup | correct |
|---|---|---|-------|---|-------------|----------------|---------------|---------|---------|
| 8 | 4096 | 2048 | 2 | 2048 | 4096 | 8837.3 | 20023.4 | 2.27x | PASS |
| 8 | 4096 | 2048 | 2 | 4096 | 8192 | 17129.5 | 38447.4 | 2.24x | PASS |
| 8 | 4096 | 2048 | 2 | 8192 | 16384 | 33800.7 | 75134.8 | 2.22x | PASS |
| 64 | 4096 | 1024 | 8 | 2048 | 16384 | 19270.0 | 42651.7 | 2.21x | PASS |
| 64 | 4096 | 1024 | 8 | 4096 | 32768 | 36359.0 | 78447.1 | 2.16x | PASS |
| 64 | 4096 | 1024 | 8 | 8192 | 65536 | 70757.6 | 151509.2 | 2.14x | PASS |

**Geomean split-over-monolithic speedup over 6 qualified shapes: 2.207x**

## Scope and reconciliation with the PR objective

This receipt measures only the split's target regime: the large-M M128-tile prefill band (routed rows 4096–65536, M 2048–8192) where the split specialization engages. Small-M tiles fall back to the monolithic kernel and are intentionally not measured, so the geomean is a target-regime figure, not a whole-workload average.

The pull-request objective reports roughly 1.24–1.41x per shape and a ~1.31x geomean for the same feature on the RTX 5090. The two results do not conflict: the PR-objective band averages a wider engaged-shape mix that includes smaller-M shapes, where the split's advantage over the monolithic kernel is smaller, while this receipt isolates the large-M end of that band. The measurements that support the dispatch decision for the covered band are this receipt's per-shape, correctness-gated ratios; the PR-objective figure is not reproducible from this receipt and is quoted only as the objective it reconciles against.

The oracle is the repo's `moe_reference_nvfp4` on synthetic quantized weights, not a checkpoint decode; treat this as kernel-path evidence and re-measure on the target checkpoint before quoting a serving number.

## Source artifact hashes (SHA-256)

Measured kernels, host adapter, and this benchmark orchestrator. Validation tests are intentionally excluded: refining a test does not invalidate a recorded measurement.

- `b12x/_lib/intrinsics.py`: 7e0c8e42bcdf609f2aae29a27c9f9013e2a8d3da91f8e032af234554d206ef7a
- `b12x/moe/_shared/kernels/dynamic.py`: d286abbf387a3f314e924058ec7a640736816dc7ca44174e0ee989e4abf61599
- `b12x/moe/_shared/kernels/nvfp4_phase1.py`: 1464775ebbd06f089109c3a183faa8c559b1a704e09fe02b85be84bb00aae280
- `b12x/moe/_shared/kernels/nvfp4_phase2.py`: 653122f1695ec98513277e22b5f26535f591ce8947800c080d2512d91fbae519
- `b12x/moe/fused_moe/_impl.py`: bb5b9c5b77ada800f6c383133616879bd339a94e755832c0c946fc79b61295e8
- `benchmarks/benchmark_nvfp4_split_materialized.py`: 8e8b9ef37b78e0736ef3b4f9368b36b91e4813cd5db95e6dae16fab2512b09fb

## Raw data

Per-shape raw timing samples (all 5 rounds × 100 iters) and full correctness metrics are in the companion JSON next to this file.
