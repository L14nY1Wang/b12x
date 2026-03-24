# b12x

`b12x` is an SM120-only CuTe DSL kernel library for Blackwell NVFP4 dense GEMM,
routed Mixture-of-Experts, and paged attention inference.

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

- `b12x.attention`
  - Primary SM120 paged attention backend (split-KV, BF16/FP8 KV, exact host planning)
- `b12x.cute`
  - Low-level CuTe and FP4 helpers
- `b12x.gemm`
  - Standalone dense NVFP4 GEMM
- `b12x.integration`
  - Public runtime entrypoints: `b12x_moe_fp4`, `b12x_paged_attention_forward`, `create_paged_attention_plan`
- `b12x.moe.fused`
  - Static, micro, and dynamic fused MoE kernels and reference paths
- `b12x.quant`
  - Torch-side NVFP4 packing and quantization helpers
- `b12x.sglang`
  - Thin `sglang` integration shims

## Attention runtime contract

Paged attention now routes through the primary `b12x.attention.paged` backend.
It is narrow by design and tuned for the Blackwell serving matrix this repo
cares about.

- `b12x.integration.attention.create_paged_attention_plan` builds an exact-shape launch plan for one paged attention configuration.
- `allocate_paged_attention_workspace_for_plan` allocates reusable scratch buffers for a plan.
- `allocate_paged_attention_workspace_pool` provides a caller-owned pool that partitions scratch by CUDA stream for variable-shape workloads.
- `b12x_paged_attention_forward` executes the kernel given a plan and workspace.
- Page size is fixed at 64.
- Supported KV dtypes: BF16, FP16, FP8 E4M3.
- FP8 KV uses raw-byte staging with in-kernel descale. Per-head descale tensors are required for FP8 KV.
- Split-KV chunking is automatic by default. `fixed_split_size` pins chunk size in pages for exact-shape benchmarking or graph replay.
- GQA with arbitrary ratios is supported.
- During CUDA graph capture, `output=` must be caller-owned and stable across replays.

## Acknowledgement

The paged attention planner, split/merge structure, and benchmark methodology
were developed by studying FlashInfer's paged attention kernels. `b12x` ships
its own SM120-first implementation and does not depend on FlashInfer at runtime.

## MoE runtime contract

- `b12x.integration.tp_moe.b12x_moe_fp4` requires a caller-owned workspace.
- `b12x` selects its fused MoE backend from shape alone:
  - compact routed workloads use the static or micro backend
  - all larger routed workloads use dynamic
- Use `allocate_tp_moe_workspace(...)` for one exact unchunked launch shape.
- Use `allocate_tp_moe_workspace_pool()` for variable-size or chunked workloads.
- Keep one workspace pool per process/device, and let the pool partition scratch by CUDA stream internally.
- During CUDA graph capture, `output=` must also be caller-owned and stable across replays.

## Environment variables

| Variable | Default | Description |
|---|---|---|
| `B12X_ATTN=TURBO` | off | Enable MXFP8 PV accumulation for FP8 KV configs (higher throughput, slight accuracy trade-off). |
| `B12X_FAST_MATH` | `1` | Enable fast-math MoE paths. |
| `B12X_MODEL_PATH` | — | Path to Qwen3.5 NVFP4 checkpoint for end-to-end MoE benchmarks. |
| `B12X_STATIC_COMPACT_CUTOVER_PAIRS` | auto | Override static→dynamic cutover threshold (routed pairs). |
| `B12X_MICRO_CUTOVER_TOKENS` | auto | Override micro→static cutover threshold (tokens). |
| `B12X_DYNAMIC_ENABLE_MULTICTA` | `1` | Enable multi-CTA dynamic launches. |
| `B12X_DYNAMIC_CHUNK_MULTIPLIER` | `1` | Dynamic backend chunk size multiplier. |
| `B12X_{STATIC,MICRO,DYNAMIC}_MAX_ACTIVE_CLUSTERS` | auto | Override max active clusters per backend. |
| `B12X_{STATIC,MICRO,DYNAMIC}_REUSE_COMPILED` | `1` | Reuse compiled kernels across shapes within a backend. |

## Benchmarks and tests

### Benchmarks

- `benchmarks/benchmark_moe.py`
  - End-to-end Qwen3.5-397B TP=4 MoE benchmark
  - `micro` batch profile: `[1, 2, 4, 8]`
  - `sglang-single-request` batch profile: `[1, 23, 80]`
  - `chunked-prefill` batch profile: `[8192, 16384, 24576, 32768]`
- `benchmarks/benchmark_paged_attention.py`
  - Paged attention vs FlashInfer across decode and extend shapes
- `benchmarks/benchmark_dense_gemm.py`
  - Dense FP4 GEMM vs FlashInfer/cuDNN/CUTLASS
- `benchmarks/benchmark_mxfp8_pv.py`
  - MXFP8 PV microbenchmark (turbo mode throughput)

### Tests

- `tests/test_paged_attention_workspace_api.py`
  - Public paged attention plan, workspace, and wrapper correctness
- `tests/test_attention_cuda_graphs.py`
  - CUDA graph capture and replay for paged attention, including FP8 KV and small GQA ratios
- `tests/test_attention_paged_forward.py`
  - Primary paged forward kernel exactness against the reference path
- `tests/test_attention_paged_merge.py`
  - Persistent split-merge exactness
- `tests/test_attention_paged_planner.py`
  - Exact host plan metadata and explicit chunk-table coverage
- `tests/test_attention_paged_traits.py`
  - Forward-trait selection for the supported serving families
- `tests/test_tp_moe_reference.py`
  - Independent oracle-backed MoE correctness test
- `tests/test_moe_equivalence.py`
  - Real-weight smoke and CUDA-graph replay routing-safety checks
- `tests/test_gemm_stack.py`
  - Dense GEMM exactness vs FlashInfer/cuDNN

## Common commands

```bash
# Graph-first benchmark defaults with auto-dispatch
B12X_MODEL_PATH=/path/to/Qwen3.5-397B-A17B-NVFP4 python benchmarks/benchmark_moe.py

# Measure eager launches instead of CUDA graph replay
B12X_MODEL_PATH=/path/to/Qwen3.5-397B-A17B-NVFP4 python benchmarks/benchmark_moe.py --no-cuda-graph

# Include routing in the timed region
B12X_MODEL_PATH=/path/to/Qwen3.5-397B-A17B-NVFP4 python benchmarks/benchmark_moe.py --include-routing

# Use the recorded single-request sglang profile
B12X_MODEL_PATH=/path/to/Qwen3.5-397B-A17B-NVFP4 python benchmarks/benchmark_moe.py --batch-size-profile sglang-single-request

# Graph-first prefill-scale sweep aligned with chunked-prefill serving
B12X_MODEL_PATH=/path/to/Qwen3.5-397B-A17B-NVFP4 python benchmarks/benchmark_moe.py --batch-size-profile chunked-prefill

# Multi-layer CUDA-graph replay validation with real consecutive MoE layers
B12X_MODEL_PATH=/path/to/Qwen3.5-397B-A17B-NVFP4 python benchmarks/benchmark_moe.py --graph-mode multi-layer --reference none --validate none

# Paged attention benchmark vs FlashInfer
python benchmarks/benchmark_paged_attention.py

# Dense GEMM microbenchmark
python benchmarks/benchmark_dense_gemm.py

# Attention correctness
pytest tests/test_attention_cuda_graphs.py tests/test_paged_attention_workspace_api.py

# Oracle-backed MoE correctness
python tests/test_tp_moe_reference.py --impls b12x --scale-contract per-expert

# Real-weight CUDA-graph smoke
pytest tests/test_moe_equivalence.py
```
