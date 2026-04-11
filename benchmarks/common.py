from __future__ import annotations

from collections.abc import Callable
import statistics

import torch

from b12x.cute.fp4 import quantize_grouped_nvfp4_torch


FLOAT4_E2M1_MAX = 6.0
FLOAT8_E4M3_MAX = float(torch.finfo(torch.float8_e4m3fn).max)


def require_sm120() -> torch.device:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required to run b12x benchmarks")
    major, minor = torch.cuda.get_device_capability()
    if (major, minor) != (12, 0):
        raise SystemExit(f"SM120 is required to run b12x benchmarks, got sm_{major}{minor}")
    return torch.device("cuda")


def make_sparse_pool_locs(
    *,
    active_tokens: int,
    pool_tokens: int,
    seed: int,
    device: torch.device,
) -> torch.Tensor:
    if pool_tokens < active_tokens:
        raise ValueError(
            f"pool_tokens {pool_tokens} must be at least active_tokens {active_tokens}"
        )
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    locs = torch.randperm(pool_tokens, generator=gen)[:active_tokens]
    return locs.to(device=device, dtype=torch.int32)


def scatter_rows_into_pool(
    rows: torch.Tensor,
    *,
    pool_locs: torch.Tensor,
    pool_tokens: int,
) -> torch.Tensor:
    pool_shape = (pool_tokens, *rows.shape[1:])
    pool = torch.zeros(pool_shape, dtype=rows.dtype, device=rows.device)
    pool[pool_locs.to(torch.long)] = rows
    return pool


def make_dense_candidate_page_table(
    *,
    batch_size: int,
    token_locs: torch.Tensor,
    width: int,
    fill_value: int = 0,
) -> torch.Tensor:
    if token_locs.ndim != 1:
        raise ValueError(f"token_locs must be rank-1, got {tuple(token_locs.shape)}")
    if width <= 0:
        raise ValueError(f"width must be positive, got {width}")
    if token_locs.shape[0] > width:
        raise ValueError(
            f"width {width} must be at least token_locs length {token_locs.shape[0]}"
        )
    page_table = torch.full(
        (batch_size, width),
        int(fill_value),
        dtype=torch.int32,
        device=token_locs.device,
    )
    if token_locs.numel():
        page_table[:, : token_locs.shape[0]] = token_locs.unsqueeze(0).expand(batch_size, -1)
    return page_table


def capture_cuda_graph(
    fn: Callable[[], object],
    *,
    warmup: int,
    prepare: Callable[[], None] | None = None,
) -> torch.cuda.CUDAGraph:
    for _ in range(warmup):
        if prepare is not None:
            prepare()
        fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    if prepare is not None:
        prepare()
    with torch.cuda.graph(graph):
        fn()
    graph.replay()
    torch.cuda.synchronize()
    return graph


def bench_cuda_graph(
    graph: torch.cuda.CUDAGraph,
    *,
    replays: int,
    prepare: Callable[[], None] | None = None,
) -> dict[str, list[float]]:
    if prepare is None:
        starts = [torch.cuda.Event(enable_timing=True) for _ in range(replays)]
        ends = [torch.cuda.Event(enable_timing=True) for _ in range(replays)]
        for idx in range(replays):
            starts[idx].record()
            graph.replay()
            ends[idx].record()
        torch.cuda.synchronize()
        replay_us = [start.elapsed_time(end) * 1000.0 for start, end in zip(starts, ends)]
        return {
            "metadata_us": [0.0] * replays,
            "replay_us": replay_us,
            "step_us": replay_us,
        }

    starts = [torch.cuda.Event(enable_timing=True) for _ in range(replays)]
    mids = [torch.cuda.Event(enable_timing=True) for _ in range(replays)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(replays)]
    for idx in range(replays):
        starts[idx].record()
        prepare()
        mids[idx].record()
        graph.replay()
        ends[idx].record()
    torch.cuda.synchronize()
    metadata_us = [start.elapsed_time(mid) * 1000.0 for start, mid in zip(starts, mids)]
    replay_us = [mid.elapsed_time(end) * 1000.0 for mid, end in zip(mids, ends)]
    step_us = [start.elapsed_time(end) * 1000.0 for start, end in zip(starts, ends)]
    return {
        "metadata_us": metadata_us,
        "replay_us": replay_us,
        "step_us": step_us,
    }


def bench_gpu_ms(fn, *, warmup: int, iters: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    times_ms: list[float] = []
    for _ in range(iters):
        start.record()
        fn()
        end.record()
        end.synchronize()
        times_ms.append(start.elapsed_time(end))
    return statistics.median(times_ms)


def compute_global_scale(x: torch.Tensor) -> torch.Tensor:
    amax = x.abs().max().to(torch.float32)
    value = FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX / amax
    return torch.tensor([value], dtype=torch.float32, device=x.device)


def compute_per_group_global_scale(x: torch.Tensor) -> torch.Tensor:
    amax = x.abs().amax(dim=(1, 2)).to(torch.float32)
    numerator = torch.full_like(amax, FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX)
    return torch.where(amax > 0, numerator / amax, torch.ones_like(amax))


def make_quantized_operand(
    shape: tuple[int, int, int],
    *,
    dtype: torch.dtype,
    scale: float = 0.25,
) -> tuple[tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
    source = torch.randn(shape, device="cuda", dtype=dtype) * scale
    row_counts = torch.full((shape[0],), shape[1], dtype=torch.int32, device=source.device)
    tensor_amax = source.abs().max().to(torch.float32)
    global_scale = torch.tensor(
        [FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX / tensor_amax],
        dtype=torch.float32,
        device=source.device,
    )
    packed, scales = quantize_grouped_nvfp4_torch(source, row_counts, global_scale)
    return (packed, scales), global_scale
