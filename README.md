# b12x

`b12x` is an SM120-only CuTe DSL kernel library for Blackwell NVFP4 dense GEMM
and routed Mixture-of-Experts inference.

It is intentionally narrow. This is not a generic CUDA kernel collection or a
full model-serving stack. It does not intend to target any other GPU architectures,
including SM100. It is a focused package for a small number of hand-tuned, high-performance
SM120 kernels plus the runtime glue needed to launch them cleanly from PyTorch and `sglang`.

## Installation

### Runtime install

```bash
python -m pip install b12x
```

### Development install from source

```bash
git clone <repo-url>
cd b12x
python -m pip install -e '.[dev]'
```

## Requirements

- Blackwell SM120 GPU
- CUDA 13 toolchain
- Python `>=3.10,<4.0`
- CUDA 13 PyTorch, `torch>=2.10.0`
- `nvidia-cutlass-dsl[cu13]==4.4.1`
- FlashInfer available if you want reference and benchmark comparisons, but it's not a runtime dependency
- Qwen3.5-397B A17B NVFP4 checkpoint available through `B12X_MODEL_PATH` for the end-to-end MoE benchmark

## Package layout

- `b12x.cute`
  - Low-level CuTe and FP4 helpers
- `b12x.gemm`
  - Standalone dense NVFP4 GEMM
- `b12x.integration`
  - Public runtime entrypoints such as `b12x_moe_fp4`
- `b12x.moe.fused`
  - Static and dynamic fused MoE kernels, scheduler, and reference paths
- `b12x.quant`
  - Torch-side NVFP4 packing and quantization helpers
- `b12x.sglang`
  - Thin `sglang` integration shims

## Benchmarks and tests

### Benchmarks

- `benchmarks/benchmark_moe.py`
  - End-to-end Qwen3.5-397B TP=4 MoE benchmark
  - `micro` batch profile: `[1, 2, 4, 8]`
  - `sglang-single-request` batch profile: `[1, 23, 80]`
- `benchmarks/benchmark_dense_gemm.py`
  - Dense FP4 GEMM vs FlashInfer/cuDNN/CUTLASS

### Tests

- `tests/test_tp_moe_reference.py`
  - Independent oracle-backed MoE correctness test
- `tests/test_moe_equivalence.py`
  - Real-weight smoke and CUDA-graph replay routing-safety checks
- `tests/test_gemm_stack.py`
  - Dense GEMM exactness vs FlashInfer/cuDNN

## Common commands

```bash
# Static backend, graph-first benchmark defaults
B12X_MODEL_PATH=/path/to/Qwen3.5-397B-A17B-NVFP4 python benchmarks/benchmark_moe.py --backend static

# Dynamic backend, same benchmark harness
B12X_MODEL_PATH=/path/to/Qwen3.5-397B-A17B-NVFP4 python benchmarks/benchmark_moe.py --backend dynamic

# Measure eager launches instead of CUDA graph replay
B12X_MODEL_PATH=/path/to/Qwen3.5-397B-A17B-NVFP4 python benchmarks/benchmark_moe.py --backend static --no-cuda-graph

# Include routing in the timed region
B12X_MODEL_PATH=/path/to/Qwen3.5-397B-A17B-NVFP4 python benchmarks/benchmark_moe.py --backend static --include-routing

# Use the recorded single-request sglang profile
B12X_MODEL_PATH=/path/to/Qwen3.5-397B-A17B-NVFP4 python benchmarks/benchmark_moe.py --backend static --batch-size-profile sglang-single-request

# Multi-layer CUDA-graph replay validation with real consecutive MoE layers
B12X_MODEL_PATH=/path/to/Qwen3.5-397B-A17B-NVFP4 python benchmarks/benchmark_moe.py --backend static --graph-mode multi-layer --reference none --validate none

# Dense GEMM microbenchmark
python benchmarks/benchmark_dense_gemm.py

# Oracle-backed MoE correctness
python tests/test_tp_moe_reference.py --impls static dynamic --scale-contract per-expert

# Real-weight CUDA-graph smoke
pytest tests/test_moe_equivalence.py
```
