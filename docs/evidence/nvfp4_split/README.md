# NVFP4 split-materialized prefill — evidence

- **Date (UTC):** 2026-09-10T10:39:32.692858+00:00
- **Commit:** `e0684c1e465db7831ba5be40e42b588b7aa5471e` (short `e0684c1e`, branch `feat/nvfp4-split-prefill`)
- **Worktree:** `?? .zcode/`
- **GPU:** NVIDIA GeForce RTX 5090
- **Active GPU mode (P-state / SM / mem clock / temp):** P1 / 2917 MHz / 13801 MHz / 30 °C (sampled under sustained launch)
- **Idle snapshot:** pstate=P8 (clocks not locked; not root)
- **Package versions:** {"torch": "2.13.0+cu130", "nvidia-cutlass-dsl": "4.6.2"}
- **`B12X_NVFP4_DYNAMIC_MATERIALIZED`:** None (unset → auto-enable for matching shapes)
- **Per-arm samples:** 100 iterations × 5 interleaved rounds, 20 warmup; CUDA-event timing; medians reported.
- **Command:** `benchmark_nvfp4_split_materialized.py --iters 100 --warmup 20 --rounds 5 --output docs/evidence/nvfp4_split/20260910-rtx5090-nvfp4-split-prefill.json`

## Path

Both arms drive one traced call through the production `@cute.jit` host adapter `_DynamicMoELaunch` (the adapter `b12x_moe_fp4` reaches for the dynamic recipe). The monolithic arm compiles the cooperative fused kernel; the split arm compiles the route/pack front-end plus the external `Nvfp4MaterializedPhase1Kernel` / `Nvfp4MaterializedPhase2Kernel`. Identical expert payloads, routed inputs, and `moe_reference_nvfp4` oracle; they differ only by `materialize_intermediate`. The split specialization engages only in the `mma_tiler_mn == (128, 128)` regime.

## Correctness gate

Qualification requires, per shape: split and monolithic cosine > 0.9999 vs the oracle, global RMSE within the BF16 bound, and split-vs-monolithic cosine > 0.9999 with RMSE within bound. The per-element `max_abs` is recorded as a diagnostic only: FP4 intermediate requantization noise grows with routed-row count on the single worst BF16 element, so a hard `max_abs` gate would reject valid large-M prefill (the monolithic production arm trips it too). Shapes that fail the gate are excluded from the headline speedup.

## Results (ratio = monolithic_us / split_us; >1.0 means split is faster)

| E | K | n | top_k | M | routed rows | split med (us) | mono med (us) | speedup | correct |
|---|---|---|-------|---|-------------|----------------|---------------|---------|---------|
| 8 | 4096 | 2048 | 2 | 2048 | 4096 | 8814.1 | 20017.7 | 2.27x | PASS |
| 8 | 4096 | 2048 | 2 | 4096 | 8192 | 17111.3 | 38431.1 | 2.25x | PASS |
| 8 | 4096 | 2048 | 2 | 8192 | 16384 | 33852.9 | 75281.8 | 2.22x | PASS |
| 64 | 4096 | 1024 | 8 | 2048 | 16384 | 19256.5 | 42666.3 | 2.22x | PASS |
| 64 | 4096 | 1024 | 8 | 4096 | 32768 | 36325.0 | 78428.1 | 2.16x | PASS |
| 64 | 4096 | 1024 | 8 | 8192 | 65536 | 70763.1 | 151516.0 | 2.14x | PASS |

**Geomean split-over-monolithic speedup over 6 qualified shapes: 2.209x**

## Scope

This receipt covers the split's target regime only: the large-M M128-tile prefill band (routed rows 4096–65536) where the split specialization engages. Small-M tiles fall back to the monolithic kernel and are intentionally not measured here, so the geomean is a target-regime figure, not a whole-workload average. The measured 2.21x is higher than the 1.31x geomean / 1.41x max the PR description quotes because those figures average over a wider engaged-shape mix that includes smaller-M shapes; the per-shape ordering and direction agree. The oracle is the repo's `moe_reference_nvfp4` on synthetic quantized weights, not a checkpoint decode; treat this as kernel-path evidence for the dispatch decision, and re-measure on the target checkpoint before quoting a serving number.

## Source artifact hashes (SHA-256)

- `b12x/_lib/intrinsics.py`: 7e0c8e42bcdf609f2aae29a27c9f9013e2a8d3da91f8e032af234554d206ef7a
- `b12x/moe/_shared/kernels/dynamic.py`: d286abbf387a3f314e924058ec7a640736816dc7ca44174e0ee989e4abf61599
- `b12x/moe/_shared/kernels/nvfp4_phase1.py`: 1464775ebbd06f089109c3a183faa8c559b1a704e09fe02b85be84bb00aae280
- `b12x/moe/_shared/kernels/nvfp4_phase2.py`: 653122f1695ec98513277e22b5f26535f591ce8947800c080d2512d91fbae519
- `b12x/moe/fused_moe/_impl.py`: bb5b9c5b77ada800f6c383133616879bd339a94e755832c0c946fc79b61295e8
- `benchmarks/benchmark_nvfp4_split_materialized.py`: 534e7d55ce9ffb05393c3a802596e16cf6b9243c79f792d227b3d52cbecb199b
- `tests/moe/test_nvfp4_phase_kernels.py`: 4cb9fac7e0845587b37df888bdf711420f09d13043a6f1e4c059dbc6028cd366
- `tests/moe/test_nvfp4_split_backend.py`: 3ee6a48c261a51c1904df93a9adc61896068ea84e25777bda2d2e575ea628883

## Raw data

Per-shape raw timing samples (all 5 rounds × 100 iters) and full correctness metrics are in the companion JSON next to this file.
