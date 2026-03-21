#!/usr/bin/env python3
"""Sweep full-rank MoE experts across the model and compare b12x static to a reference."""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benchmarks.benchmark_moe import (
    MODEL_PATH,
    TP_RANK,
    TP_SIZE,
    ModelSpec,
    get_scale_contract_params,
    load_expert_weights,
)
from experiments.verify_static_tile_tp1 import _run_flashinfer, _run_static
from b12x.integration.tp_moe import _STATIC_KERNEL_CACHE, _STATE_CACHE, _WEIGHT_CACHE
from b12x.moe.fused.reference import compare_to_reference, moe_reference_f32, moe_reference_nvfp4


def _parse_layer_indices(model_path: pathlib.Path, layer_indices: list[int] | None) -> list[int]:
    cfg = json.loads((model_path / "config.json").read_text())["text_config"]
    num_layers = cfg["num_hidden_layers"]
    if layer_indices is None:
        return list(range(num_layers))
    for layer_idx in layer_indices:
        if layer_idx < 0 or layer_idx >= num_layers:
            raise ValueError(f"layer_idx must be in [0, {num_layers}), got {layer_idx}")
    return layer_indices


def _slice_scale_param(param: torch.Tensor, expert_idx: int) -> torch.Tensor:
    return param.narrow(0, expert_idx, 1) if param.numel() > 1 else param


def _reference_output(
    reference: str,
    x: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    w13_weight: torch.Tensor,
    w13_blockscale_swizzled: torch.Tensor,
    g1_alphas: torch.Tensor,
    a1_gscale: torch.Tensor,
    w2_weight: torch.Tensor,
    w2_blockscale_swizzled: torch.Tensor,
    g2_alphas: torch.Tensor,
    a2_gscale: torch.Tensor,
    spec: ModelSpec,
) -> torch.Tensor:
    if reference == "flashinfer":
        return _run_flashinfer(
            x,
            topk_ids,
            topk_weights,
            w13_weight,
            w13_blockscale_swizzled,
            g1_alphas,
            a1_gscale,
            w2_weight,
            w2_blockscale_swizzled,
            g2_alphas,
            a2_gscale,
        )

    oracle_fn = moe_reference_nvfp4 if reference == "oracle-nvfp4" else moe_reference_f32
    out = oracle_fn(
        x,
        w13_weight,
        w13_blockscale_swizzled,
        g1_alphas,
        w2_weight,
        w2_blockscale_swizzled,
        g2_alphas,
        a1_gscale,
        a2_gscale,
        topk_ids,
        topk_weights,
        1,
        spec.hidden_size,
        spec.I_tp,
    )
    torch.cuda.synchronize()
    return out.detach().clone()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--reference",
        choices=["flashinfer", "oracle-nvfp4", "oracle-f32"],
        default="flashinfer",
    )
    parser.add_argument("--scale-contract", choices=["shared", "per-expert"], default="shared")
    parser.add_argument("--layer-indices", type=int, nargs="+", default=None)
    parser.add_argument("--expert-indices", type=int, nargs="+", default=None)
    parser.add_argument("--activation-scale", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--results-path",
        type=pathlib.Path,
        default=ROOT / "full_model_expert_sweep.tsv",
    )
    args = parser.parse_args()

    if args.reference == "flashinfer" and args.scale_contract != "shared":
        raise ValueError("flashinfer only supports --scale-contract shared")

    layer_indices = _parse_layer_indices(MODEL_PATH, args.layer_indices)
    spec = ModelSpec(
        hidden_size=4096,
        intermediate_size=1024,
        num_experts=512,
        top_k=1,
        tp_size=TP_SIZE,
        tp_rank=TP_RANK,
    )
    expert_indices = args.expert_indices or list(range(spec.num_experts))
    for expert_idx in expert_indices:
        if expert_idx < 0 or expert_idx >= spec.num_experts:
            raise ValueError(f"expert_idx must be in [0, {spec.num_experts}), got {expert_idx}")

    device = torch.device("cuda")
    torch.manual_seed(args.seed)
    x = (torch.randn(1, spec.hidden_size, device=device, dtype=torch.float32) * args.activation_scale).to(
        torch.bfloat16
    )
    topk_ids = torch.zeros(1, 1, dtype=torch.int32, device=device)
    topk_weights = torch.ones(1, 1, dtype=torch.float32, device=device)

    _STATE_CACHE.clear()
    _WEIGHT_CACHE.clear()
    _STATIC_KERNEL_CACHE.clear()
    torch.cuda.empty_cache()

    args.results_path.write_text("layer\texpert\tmax_abs\trmse\tmean_abs\tcos\texact\n")

    total = 0
    exact = 0
    worst = None
    start = time.time()

    print(
        f"Sweeping layers={len(layer_indices)} experts_per_layer={len(expert_indices)} "
        f"reference={args.reference} scale_contract={args.scale_contract} "
        f"K={spec.hidden_size} I_tp={spec.I_tp}"
    )

    with torch.inference_mode():
        for layer_idx in layer_indices:
            layer_start = time.time()
            weights = load_expert_weights(MODEL_PATH, spec, layer_idx=layer_idx)
            scale_params = get_scale_contract_params(weights, args.scale_contract)
            _WEIGHT_CACHE.clear()

            layer_worst = None
            layer_exact = 0
            for expert_idx in expert_indices:
                w13_weight = weights.w13_weight.narrow(0, expert_idx, 1)
                w13_blockscale_swizzled = weights.w13_blockscale_swizzled.narrow(0, expert_idx, 1)
                g1_alphas = _slice_scale_param(scale_params.g1_alphas, expert_idx)
                a1_gscale = _slice_scale_param(scale_params.a1_gscale, expert_idx)
                w2_weight = weights.w2_weight.narrow(0, expert_idx, 1)
                w2_blockscale_swizzled = weights.w2_blockscale_swizzled.narrow(0, expert_idx, 1)
                g2_alphas = _slice_scale_param(scale_params.g2_alphas, expert_idx)
                a2_gscale = _slice_scale_param(scale_params.a2_gscale, expert_idx)

                ref = _reference_output(
                    args.reference,
                    x,
                    topk_ids,
                    topk_weights,
                    w13_weight,
                    w13_blockscale_swizzled,
                    g1_alphas,
                    a1_gscale,
                    w2_weight,
                    w2_blockscale_swizzled,
                    g2_alphas,
                    a2_gscale,
                    spec,
                )
                out = _run_static(
                    x,
                    topk_ids,
                    topk_weights,
                    w13_weight,
                    w13_blockscale_swizzled,
                    g1_alphas,
                    a1_gscale,
                    w2_weight,
                    w2_blockscale_swizzled,
                    g2_alphas,
                    a2_gscale,
                )
                metrics = compare_to_reference(out, ref)
                is_exact = torch.equal(out, ref)
                total += 1
                exact += int(is_exact)
                layer_exact += int(is_exact)
                row = (
                    f"{layer_idx}\t{expert_idx}\t{metrics.max_abs:.9f}\t{metrics.rmse:.9f}\t"
                    f"{metrics.mean_abs:.9f}\t{metrics.cos:.9f}\t{int(is_exact)}\n"
                )
                with args.results_path.open("a") as f:
                    f.write(row)

                key = (metrics.max_abs, metrics.rmse, -metrics.cos, layer_idx, expert_idx)
                if layer_worst is None or key > layer_worst[0]:
                    layer_worst = (key, metrics, expert_idx, is_exact)
                if worst is None or key > worst[0]:
                    worst = (key, metrics, layer_idx, expert_idx, is_exact)

            print(
                f"layer={layer_idx:02d} exact={layer_exact}/{len(expert_indices)} "
                f"worst_expert={layer_worst[2]} max_abs={layer_worst[1].max_abs:.6f} "
                f"rmse={layer_worst[1].rmse:.6f} cos={layer_worst[1].cos:.6f} "
                f"elapsed={time.time() - layer_start:.1f}s"
            )

    assert worst is not None
    print(
        f"done total={total} exact={exact}/{total} "
        f"worst_layer={worst[2]} worst_expert={worst[3]} "
        f"max_abs={worst[1].max_abs:.6f} rmse={worst[1].rmse:.6f} "
        f"mean_abs={worst[1].mean_abs:.6f} cos={worst[1].cos:.6f} "
        f"elapsed={time.time() - start:.1f}s results={args.results_path}"
    )


if __name__ == "__main__":
    main()
