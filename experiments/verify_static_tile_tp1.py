#!/usr/bin/env python3
"""Reduced-case verifier for b12x static vs FlashInfer vs oracle.

This trims the problem to one token and a windowed subset of the real
checkpoint weights so we can isolate correctness issues at the 128x128 tile
level without touching the production kernel path.
"""

from __future__ import annotations

import argparse
import itertools
import json
import pathlib
import sys

import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benchmarks.benchmark_moe import MODEL_PATH
from benchmarks.checkpoint_loader import IndexedSafetensorLoader
from b12x.cute.fp4 import fp4_quantize_values_torch, swizzle_block_scale
from flashinfer.fused_moe import cutlass_fused_moe as flashinfer_cutlass_fused_moe
from b12x.integration.tp_moe import _STATIC_KERNEL_CACHE, _STATE_CACHE, _WEIGHT_CACHE, b12x_moe_fp4
from b12x.moe.fused.reference import (
    compare_to_reference,
    moe_reference_f32,
    moe_reference_nvfp4,
    unswizzle_block_scale,
)


def _clear_codegen_artifacts() -> None:
    for pattern in (
        "cutlass___call___b12xmoefusedstatic*.ptx",
        "cutlass___call___b12xmoefusedstatic*.cubin",
        "cutlass___call___b12xmoefusedstatic*.sass",
    ):
        for path in ROOT.glob(pattern):
            path.unlink(missing_ok=True)


def _run_flashinfer(
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
) -> torch.Tensor:
    out = torch.empty(x.shape[0], x.shape[1], dtype=torch.bfloat16, device=x.device)
    quant_scales = [
        (1.0 / a1_gscale).reshape(()),
        w13_blockscale_swizzled.view(torch.int32),
        g1_alphas,
        (1.0 / a2_gscale).reshape(()),
        w2_blockscale_swizzled.view(torch.int32),
        g2_alphas,
    ]
    flashinfer_cutlass_fused_moe(
        output=out,
        input=x,
        token_selected_experts=topk_ids.to(torch.int32),
        token_final_scales=topk_weights,
        fc1_expert_weights=w13_weight.view(torch.long),
        fc2_expert_weights=w2_weight.view(torch.long),
        output_dtype=torch.bfloat16,
        quant_scales=quant_scales,
        input_sf=None,
        tp_size=1,
        tp_rank=0,
        ep_size=1,
        ep_rank=0,
        tune_max_num_tokens=16,
    )
    torch.cuda.synchronize()
    return out.detach().clone()


def _run_static(
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
) -> torch.Tensor:
    out = b12x_moe_fp4(
        x,
        a1_gscale,
        w13_weight,
        w13_blockscale_swizzled,
        g1_alphas,
        a2_gscale,
        w2_weight,
        w2_blockscale_swizzled,
        g2_alphas,
        topk_weights,
        topk_ids,
    )
    torch.cuda.synchronize()
    return out.detach().clone()


def _metric_line(name: str, out: torch.Tensor, ref: torch.Tensor) -> str:
    metrics = compare_to_reference(out, ref)
    return (
        f"{name:<10} max_abs={metrics.max_abs:.6f} "
        f"rmse={metrics.rmse:.6f} "
        f"mean_abs={metrics.mean_abs:.6f} "
        f"cos={metrics.cos:.6f}"
    )


def _dequant_fp4(packed_u8: torch.Tensor, rows: int, cols: int) -> torch.Tensor:
    fp4_lut = torch.tensor(
        [
            0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
            -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
        ],
        dtype=torch.float32,
        device=packed_u8.device,
    )
    lo = (packed_u8 & 0x0F).to(torch.int64)
    hi = ((packed_u8 >> 4) & 0x0F).to(torch.int64)
    return torch.stack([fp4_lut[lo], fp4_lut[hi]], dim=-1).reshape(rows, cols)


def _apply_block_scales(raw: torch.Tensor, sf_f32: torch.Tensor, rows: int, cols: int) -> torch.Tensor:
    block_size = 16
    n_blocks = cols // block_size
    sf = sf_f32[:rows, :n_blocks]
    return raw * sf.unsqueeze(-1).expand(rows, n_blocks, block_size).reshape(rows, cols)


def _quantize_vec_to_fp4_dequant(vals_f32: torch.Tensor, global_scale: float) -> torch.Tensor:
    block_size = 16
    fp8_e4m3_max = float(torch.finfo(torch.float8_e4m3fn).max)
    cols = vals_f32.shape[0]
    n_blocks = cols // block_size
    blocked = vals_f32.reshape(n_blocks, block_size)
    block_max = blocked.abs().amax(dim=-1)

    raw_scale = (block_max / (6.0 * global_scale)).clamp(max=fp8_e4m3_max)
    sf_e4m3 = raw_scale.to(torch.float8_e4m3fn).to(torch.float32)

    sf_times_gs = sf_e4m3.unsqueeze(-1).expand(n_blocks, block_size).reshape(cols) * global_scale
    scaled = vals_f32 / sf_times_gs.clamp(min=1e-30)
    quant = fp4_quantize_values_torch(scaled)
    sf_only = sf_e4m3.unsqueeze(-1).expand(n_blocks, block_size).reshape(cols)
    return quant * sf_only


def _quantize_vec_to_fp4_packed(vals_f32: torch.Tensor, global_scale: float) -> tuple[torch.Tensor, torch.Tensor]:
    block_size = 16
    fp8_e4m3_max = float(torch.finfo(torch.float8_e4m3fn).max)
    cols = vals_f32.shape[0]
    n_blocks = cols // block_size
    blocked = vals_f32.reshape(n_blocks, block_size)
    block_max = blocked.abs().amax(dim=-1)

    raw_scale = (block_max / (6.0 * global_scale)).clamp(max=fp8_e4m3_max)
    sf_e4m3 = raw_scale.to(torch.float8_e4m3fn)
    scale_bytes = sf_e4m3.view(torch.uint8).to(torch.uint8)
    sf_f32 = sf_e4m3.to(torch.float32)

    sf_times_gs = sf_f32.unsqueeze(-1) * global_scale
    scaled = blocked / sf_times_gs.clamp(min=1e-30)
    quant = fp4_quantize_values_torch(scaled)
    mags = quant.abs()
    idx = torch.zeros_like(quant, dtype=torch.uint8)
    fp4_mags = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
    for code, mag in enumerate(fp4_mags):
        idx = torch.where(mags == mag, torch.full_like(idx, code), idx)
    nibbles = idx | ((quant < 0).to(torch.uint8) << 3)
    packed = (nibbles[:, 0::2] | (nibbles[:, 1::2] << 4)).reshape(-1)
    return packed, scale_bytes


def _single_expert_stage_reference(
    x: torch.Tensor,
    w13_weight: torch.Tensor,
    w13_blockscale_swizzled: torch.Tensor,
    g1_alphas: torch.Tensor,
    w2_weight: torch.Tensor,
    w2_blockscale_swizzled: torch.Tensor,
    g2_alphas: torch.Tensor,
    a1_gscale: torch.Tensor,
    a2_gscale: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    hidden_size = x.shape[1]
    intermediate_size = w2_weight.shape[2] * 2

    x_dequant = _quantize_vec_to_fp4_dequant(x[0].float(), float(a1_gscale.item()))
    w13_sf = unswizzle_block_scale(w13_blockscale_swizzled[0], 2 * intermediate_size, hidden_size // 16)
    w2_sf = unswizzle_block_scale(w2_blockscale_swizzled[0], hidden_size, intermediate_size // 16)

    up_dequant = _apply_block_scales(
        _dequant_fp4(w13_weight[0, :intermediate_size], intermediate_size, hidden_size),
        w13_sf[:intermediate_size],
        intermediate_size,
        hidden_size,
    )
    gate_dequant = _apply_block_scales(
        _dequant_fp4(w13_weight[0, intermediate_size:], intermediate_size, hidden_size),
        w13_sf[intermediate_size:],
        intermediate_size,
        hidden_size,
    )

    alpha_fc1 = float(g1_alphas[0].item())
    alpha_fc2 = float(g2_alphas[0].item())

    gate_out = ((gate_dequant @ x_dequant) * alpha_fc1).to(torch.bfloat16)
    up_out = ((up_dequant @ x_dequant) * alpha_fc1).to(torch.bfloat16)
    stage1 = (torch.sigmoid(gate_out.float()) * gate_out.float() * up_out.float()).to(torch.bfloat16)
    stage1_packed, stage1_scale = _quantize_vec_to_fp4_packed(stage1.float(), float(a2_gscale.item()))
    int_dequant = _quantize_vec_to_fp4_dequant(stage1.float(), float(a2_gscale.item()))
    down_dequant = _apply_block_scales(
        _dequant_fp4(w2_weight[0], hidden_size, intermediate_size),
        w2_sf,
        hidden_size,
        intermediate_size,
    )
    stage2 = ((down_dequant @ int_dequant) * alpha_fc2).to(torch.bfloat16)
    return gate_out, up_out, stage1, stage1_packed, stage1_scale, stage2


def _bf16_serial_sum(parts: list[torch.Tensor], order: tuple[int, ...]) -> torch.Tensor:
    acc = torch.zeros_like(parts[0])
    for idx in order:
        acc = (acc.float() + parts[idx].float()).to(torch.bfloat16)
    return acc


def _permutation_summary(full_out: torch.Tensor, parts: list[torch.Tensor]) -> tuple[int, tuple[int, ...] | None]:
    orders = list(itertools.permutations(range(len(parts))))
    perm_sums = [_bf16_serial_sum(parts, order) for order in orders]
    match_order = next((order for order, perm_sum in zip(orders, perm_sums) if torch.equal(perm_sum, full_out)), None)
    legal = torch.stack([perm_sum.squeeze(0) for perm_sum in perm_sums], dim=0)
    explainable = (legal == full_out.squeeze(0).unsqueeze(0)).any(dim=0)
    return int((~explainable).sum().item()), match_order


def _validate_window(
    *,
    layer_idx: int,
    hidden_size: int,
    intermediate_size: int,
    hidden_offset: int,
    intermediate_offset: int,
    cfg: dict,
) -> None:
    if hidden_size % 128 != 0 or intermediate_size % 128 != 0:
        raise ValueError("hidden_size and intermediate_size must be multiples of 128")
    if hidden_offset % 128 != 0 or intermediate_offset % 128 != 0:
        raise ValueError("hidden_offset and intermediate_offset must be multiples of 128")
    if layer_idx < 0 or layer_idx >= cfg["num_hidden_layers"]:
        raise ValueError(f"layer_idx must be in [0, {cfg['num_hidden_layers']})")
    if hidden_offset + hidden_size > cfg["hidden_size"]:
        raise ValueError(
            f"hidden window [{hidden_offset}, {hidden_offset + hidden_size}) exceeds hidden_size={cfg['hidden_size']}"
        )
    if intermediate_offset + intermediate_size > cfg["moe_intermediate_size"]:
        raise ValueError(
            "intermediate window "
            f"[{intermediate_offset}, {intermediate_offset + intermediate_size}) exceeds "
            f"moe_intermediate_size={cfg['moe_intermediate_size']}"
        )


def _load_windowed_real_weights(
    *,
    layer_idx: int,
    device: torch.device,
    hidden_size: int,
    intermediate_size: int,
    hidden_offset: int = 0,
    intermediate_offset: int = 0,
    expert_idx: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    cfg = json.loads((MODEL_PATH / "config.json").read_text())["text_config"]
    _validate_window(
        layer_idx=layer_idx,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        hidden_offset=hidden_offset,
        intermediate_offset=intermediate_offset,
        cfg=cfg,
    )

    prefix = f"model.language_model.layers.{layer_idx}.mlp.experts"
    loader = IndexedSafetensorLoader(MODEL_PATH)
    hidden_packed_offset = hidden_offset // 2
    hidden_block_offset = hidden_offset // 16
    intermediate_packed_offset = intermediate_offset // 2
    intermediate_block_offset = intermediate_offset // 16

    ep = f"{prefix}.{expert_idx}"
    gate_w = (
        loader.get_tensor(f"{ep}.gate_proj.weight")
        .narrow(0, intermediate_offset, intermediate_size)
        .narrow(1, hidden_packed_offset, hidden_size // 2)
    )
    gate_sf = (
        loader.get_tensor(f"{ep}.gate_proj.weight_scale")
        .narrow(0, intermediate_offset, intermediate_size)
        .narrow(1, hidden_block_offset, hidden_size // 16)
    )
    gate_gs = loader.get_tensor(f"{ep}.gate_proj.weight_scale_2")
    gate_is = loader.get_tensor(f"{ep}.gate_proj.input_scale")

    up_w = (
        loader.get_tensor(f"{ep}.up_proj.weight")
        .narrow(0, intermediate_offset, intermediate_size)
        .narrow(1, hidden_packed_offset, hidden_size // 2)
    )
    up_sf = (
        loader.get_tensor(f"{ep}.up_proj.weight_scale")
        .narrow(0, intermediate_offset, intermediate_size)
        .narrow(1, hidden_block_offset, hidden_size // 16)
    )

    down_w = (
        loader.get_tensor(f"{ep}.down_proj.weight")
        .narrow(0, hidden_offset, hidden_size)
        .narrow(1, intermediate_packed_offset, intermediate_size // 2)
    )
    down_sf = (
        loader.get_tensor(f"{ep}.down_proj.weight_scale")
        .narrow(0, hidden_offset, hidden_size)
        .narrow(1, intermediate_block_offset, intermediate_size // 16)
    )
    down_gs = loader.get_tensor(f"{ep}.down_proj.weight_scale_2")
    down_is = loader.get_tensor(f"{ep}.down_proj.input_scale")

    w13_weight = torch.cat([up_w, gate_w], dim=0).unsqueeze(0).contiguous().to(device)
    w13_blockscale_swizzled = swizzle_block_scale(
        torch.cat([up_sf, gate_sf], dim=0).unsqueeze(0).contiguous().to(device)
    )
    w2_weight = down_w.unsqueeze(0).contiguous().to(device)
    w2_blockscale_swizzled = swizzle_block_scale(down_sf.unsqueeze(0).contiguous().to(device))

    a1_gscale = gate_is.to(device=device, dtype=torch.float32).reshape(())
    a2_gscale = down_is.to(device=device, dtype=torch.float32).reshape(())
    g1_alphas = (a1_gscale * gate_gs.to(device=device, dtype=torch.float32).reshape(1)).contiguous()
    g2_alphas = (a2_gscale * down_gs.to(device=device, dtype=torch.float32).reshape(1)).contiguous()

    return (
        w13_weight,
        w13_blockscale_swizzled,
        g1_alphas,
        w2_weight,
        w2_blockscale_swizzled,
        g2_alphas,
        a1_gscale,
        a2_gscale,
    )


def _load_windowed_real_weights_multi(
    *,
    layer_idx: int,
    device: torch.device,
    hidden_size: int,
    intermediate_size: int,
    hidden_offset: int = 0,
    intermediate_offset: int = 0,
    expert_indices: list[int],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if not expert_indices:
        raise ValueError("expert_indices must not be empty")

    per_expert = [
        _load_windowed_real_weights(
            layer_idx=layer_idx,
            device=device,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            hidden_offset=hidden_offset,
            intermediate_offset=intermediate_offset,
            expert_idx=expert_idx,
        )
        for expert_idx in expert_indices
    ]

    (
        w13_list,
        w13_sf_list,
        g1_list,
        w2_list,
        w2_sf_list,
        g2_list,
        a1_list,
        a2_list,
    ) = zip(*per_expert, strict=True)

    a1_gscale = torch.stack([a.reshape(()) for a in a1_list]).amax().reshape(())
    a2_gscale = torch.stack([a.reshape(()) for a in a2_list]).amax().reshape(())
    g1_list = tuple((a1_gscale * (g1 / a1)).contiguous() for g1, a1 in zip(g1_list, a1_list, strict=True))
    g2_list = tuple((a2_gscale * (g2 / a2)).contiguous() for g2, a2 in zip(g2_list, a2_list, strict=True))

    return (
        torch.cat(w13_list, dim=0).contiguous(),
        torch.cat(w13_sf_list, dim=0).contiguous(),
        torch.cat(g1_list, dim=0).contiguous(),
        torch.cat(w2_list, dim=0).contiguous(),
        torch.cat(w2_sf_list, dim=0).contiguous(),
        torch.cat(g2_list, dim=0).contiguous(),
        a1_gscale,
        a2_gscale,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--layer-index", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--intermediate-size", type=int, default=128)
    parser.add_argument("--hidden-offset", type=int, default=0)
    parser.add_argument("--intermediate-offset", type=int, default=0)
    parser.add_argument("--activation-source-hidden-size", type=int, default=None)
    parser.add_argument("--expert", type=int, default=0)
    parser.add_argument("--experts", type=str, default=None)
    parser.add_argument("--activation-scale", type=float, default=10.0)
    parser.add_argument("--oracle-mode", choices=["nvfp4", "f32"], default="nvfp4")
    parser.add_argument("--prelaunch", choices=["none", "flashinfer", "static"], default="none")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--permutation-sum-check", action="store_true")
    parser.add_argument("--print-stage-reference", action="store_true")
    args = parser.parse_args()

    activation_source_hidden_size = args.activation_source_hidden_size or (args.hidden_offset + args.hidden_size)
    if activation_source_hidden_size < args.hidden_offset + args.hidden_size:
        raise ValueError("activation_source_hidden_size must cover hidden_offset + hidden_size")

    device = torch.device("cuda")
    torch.empty(1, device=device)

    torch.manual_seed(args.seed)
    full_x = (
        torch.randn(1, activation_source_hidden_size, device=device, dtype=torch.float32) * args.activation_scale
    ).to(torch.bfloat16)
    x = full_x[:, args.hidden_offset:args.hidden_offset + args.hidden_size].contiguous()

    expert_indices = [int(tok) for tok in args.experts.split(",")] if args.experts else [args.expert]
    num_selected_experts = len(expert_indices)
    topk_ids = torch.arange(num_selected_experts, dtype=torch.int32, device=device).view(1, num_selected_experts)
    topk_weights = torch.full((1, num_selected_experts), 1.0 / num_selected_experts, dtype=torch.float32, device=device)

    (
        w13_weight,
        w13_blockscale_swizzled,
        g1_alphas,
        w2_weight,
        w2_blockscale_swizzled,
        g2_alphas,
        a1_gscale,
        a2_gscale,
    ) = _load_windowed_real_weights_multi(
        layer_idx=args.layer_index,
        device=device,
        hidden_size=args.hidden_size,
        intermediate_size=args.intermediate_size,
        hidden_offset=args.hidden_offset,
        intermediate_offset=args.intermediate_offset,
        expert_indices=expert_indices,
    )

    _clear_codegen_artifacts()
    _STATE_CACHE.clear()
    _WEIGHT_CACHE.clear()
    _STATIC_KERNEL_CACHE.clear()
    torch.cuda.empty_cache()

    oracle_fn = moe_reference_nvfp4 if args.oracle_mode == "nvfp4" else moe_reference_f32
    ref = oracle_fn(
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
        num_selected_experts,
        args.hidden_size,
        args.intermediate_size,
    )

    if args.prelaunch == "flashinfer":
        _run_flashinfer(
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
    elif args.prelaunch == "static":
        _run_static(
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

    fi_out = _run_flashinfer(
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
    static_out = _run_static(
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

    print("Single-tile verifier")
    print(
        f"layer={args.layer_index} E={num_selected_experts} "
        f"K={args.hidden_size} I_tp={args.intermediate_size} "
        f"hidden_offset={args.hidden_offset} intermediate_offset={args.intermediate_offset}"
    )
    print(
        f"experts={expert_indices} activation_source_hidden_size={activation_source_hidden_size} "
        f"oracle={args.oracle_mode} prelaunch={args.prelaunch} repeats={args.repeats}"
    )
    print(
        f"a1_gscale={float(a1_gscale.item()):.8f} "
        f"a2_gscale={float(a2_gscale.item()):.8f} "
        f"ref_norm={ref.float().norm().item():.6f}"
    )
    print(_metric_line("flashinfer", fi_out, ref))
    print(_metric_line("static", static_out, ref))
    print("ref[0,0:8]=", [float(v) for v in ref[0, :8].float().tolist()])
    print("fi[0,0:8]=", [float(v) for v in fi_out[0, :8].float().tolist()])
    print("static[0,0:8]=", [float(v) for v in static_out[0, :8].float().tolist()])
    if args.print_stage_reference:
        if num_selected_experts != 1:
            raise ValueError("--print-stage-reference requires exactly one expert")
        gate_ref, up_ref, stage1_ref, stage1_packed, stage1_scale, stage2_ref = _single_expert_stage_reference(
            x,
            w13_weight,
            w13_blockscale_swizzled,
            g1_alphas,
            w2_weight,
            w2_blockscale_swizzled,
            g2_alphas,
            a1_gscale,
            a2_gscale,
        )
        x_packed, x_scale = _quantize_vec_to_fp4_packed(x[0].float(), float(a1_gscale.item()))
        print("x_pack_ref[0:8]=", [int(v) for v in x_packed[:8].tolist()])
        print("x_pack_ref[8:16]=", [int(v) for v in x_packed[8:16].tolist()])
        print("x_scale_ref[0:8]=", [int(v) for v in x_scale[:8].tolist()])
        print("gate_ref[0:8]=", [float(v) for v in gate_ref[:8].float().tolist()])
        print("up_ref[0:8]=", [float(v) for v in up_ref[:8].float().tolist()])
        print("stage1_ref[0:8]=", [float(v) for v in stage1_ref[:8].float().tolist()])
        print("stage1_ref[8:16]=", [float(v) for v in stage1_ref[8:16].float().tolist()])
        print("stage1_pack_ref[0:8]=", [int(v) for v in stage1_packed[:8].tolist()])
        print("stage1_pack_ref[8:16]=", [int(v) for v in stage1_packed[8:16].tolist()])
        print("stage1_scale_ref[0:8]=", [int(v) for v in stage1_scale[:8].tolist()])
        print("stage2_ref[0:8]=", [float(v) for v in stage2_ref[:8].float().tolist()])

    if args.permutation_sum_check:
        if args.intermediate_size % 128 != 0:
            raise ValueError("--permutation-sum-check requires intermediate_size to be a multiple of 128")
        num_slices = args.intermediate_size // 128
        if num_slices > 6:
            raise ValueError("--permutation-sum-check is limited to at most 6 slices")

        static_parts: list[torch.Tensor] = []
        fi_parts: list[torch.Tensor] = []
        for slice_idx in range(num_slices):
            slice_offset = args.intermediate_offset + slice_idx * 128
            print(f"self_check_slice idx={slice_idx} intermediate_offset={slice_offset} size=128")
            (
                s_w13_weight,
                s_w13_blockscale_swizzled,
                s_g1_alphas,
                s_w2_weight,
                s_w2_blockscale_swizzled,
                s_g2_alphas,
                s_a1_gscale,
                s_a2_gscale,
            ) = _load_windowed_real_weights_multi(
                layer_idx=args.layer_index,
                device=device,
                hidden_size=args.hidden_size,
                intermediate_size=128,
                hidden_offset=args.hidden_offset,
                intermediate_offset=slice_offset,
                expert_indices=expert_indices,
            )
            fi_parts.append(_run_flashinfer(
                x,
                topk_ids,
                topk_weights,
                s_w13_weight,
                s_w13_blockscale_swizzled,
                s_g1_alphas,
                s_a1_gscale,
                s_w2_weight,
                s_w2_blockscale_swizzled,
                s_g2_alphas,
                s_a2_gscale,
            ))
            static_parts.append(_run_static(
                x,
                topk_ids,
                topk_weights,
                s_w13_weight,
                s_w13_blockscale_swizzled,
                s_g1_alphas,
                s_a1_gscale,
                s_w2_weight,
                s_w2_blockscale_swizzled,
                s_g2_alphas,
                s_a2_gscale,
            ))

        static_bad, static_match_order = _permutation_summary(static_out, static_parts)
        fi_part_exact = sum(int(torch.equal(fi_part, static_part)) for fi_part, static_part in zip(fi_parts, static_parts, strict=True))
        fi_bad, fi_match_order = _permutation_summary(fi_out, fi_parts)
        print(f"permutation_sum_check slices={num_slices}")
        print(f"fi_static_exact_parts={fi_part_exact}/{num_slices}")
        print(
            f"flashinfer illegal_indices={fi_bad} "
            f"global_perm_match={fi_match_order if fi_match_order is not None else 'none'}"
        )
        print(
            f"static     illegal_indices={static_bad} "
            f"global_perm_match={static_match_order if static_match_order is not None else 'none'}"
        )

    if args.repeats > 1:
        fi_equal_to_base = 0
        fi_equal_to_prev = 0
        fi_prev = fi_out
        static_equal_to_base = 0
        static_equal_to_prev = 0
        static_prev = static_out
        for _ in range(args.repeats - 1):
            fi_next = _run_flashinfer(
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
            static_next = _run_static(
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
            fi_equal_to_base += int(torch.equal(fi_next, fi_out))
            fi_equal_to_prev += int(torch.equal(fi_next, fi_prev))
            fi_prev = fi_next
            static_equal_to_base += int(torch.equal(static_next, static_out))
            static_equal_to_prev += int(torch.equal(static_next, static_prev))
            static_prev = static_next
        print(
            f"flashinfer exact_equal_to_base={fi_equal_to_base}/{args.repeats - 1} "
            f"exact_equal_to_prev={fi_equal_to_prev}/{args.repeats - 1}"
        )
        print(
            f"static     exact_equal_to_base={static_equal_to_base}/{args.repeats - 1} "
            f"exact_equal_to_prev={static_equal_to_prev}/{args.repeats - 1}"
        )


if __name__ == "__main__":
    main()
