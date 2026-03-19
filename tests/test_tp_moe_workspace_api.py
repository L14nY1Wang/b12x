from __future__ import annotations

import pytest
import torch

import b12x.integration.tp_moe as tp_moe
from benchmarks.benchmark_moe import MODEL_PATH, TP_RANK, TP_SIZE, ModelSpec, load_expert_weights, make_routed_inputs
from b12x.integration.tp_moe import (
    allocate_tp_moe_workspace,
    allocate_tp_moe_workspace_pool,
    b12x_moe_fp4,
    clear_tp_moe_caches,
)
from b12x.moe.fused.reference import compare_to_reference

from .helpers import require_sm120


def _require_model_weights() -> None:
    if not MODEL_PATH.exists():
        pytest.skip(f"Model not found at {MODEL_PATH}")


def _make_spec() -> ModelSpec:
    return ModelSpec(
        hidden_size=4096,
        intermediate_size=1024,
        num_experts=512,
        top_k=10,
        tp_size=TP_SIZE,
        tp_rank=TP_RANK,
    )


def test_workspace_pool_handles_chunked_calls(monkeypatch: pytest.MonkeyPatch) -> None:
    require_sm120()
    _require_model_weights()

    clear_tp_moe_caches()

    device = torch.device("cuda")
    spec = _make_spec()
    weights = load_expert_weights(MODEL_PATH, spec, layer_idx=0)
    x, topk_ids, topk_weights = make_routed_inputs(spec, 8, seed=321, device=device)

    exact_workspace = allocate_tp_moe_workspace(
        x,
        weights.w13_input_scale_per_expert,
        weights.w13_weight,
        weights.w2_input_scale_per_expert,
        weights.w2_weight,
        topk_ids,
        implementation="static",
        input_scales_static=True,
    )
    expected = b12x_moe_fp4(
        x,
        weights.w13_input_scale_per_expert,
        weights.w13_weight,
        weights.w13_blockscale_swizzled,
        weights.g1_alphas_per_expert,
        weights.w2_input_scale_per_expert,
        weights.w2_weight,
        weights.w2_blockscale_swizzled,
        weights.g2_alphas_per_expert,
        topk_weights,
        topk_ids,
        implementation="static",
        workspace=exact_workspace,
        input_scales_static=True,
    ).clone()
    torch.cuda.synchronize(device)

    pool = allocate_tp_moe_workspace_pool()
    monkeypatch.setattr(tp_moe, "_safe_token_chunk", lambda *_args: 2)
    actual = b12x_moe_fp4(
        x,
        weights.w13_input_scale_per_expert,
        weights.w13_weight,
        weights.w13_blockscale_swizzled,
        weights.g1_alphas_per_expert,
        weights.w2_input_scale_per_expert,
        weights.w2_weight,
        weights.w2_blockscale_swizzled,
        weights.g2_alphas_per_expert,
        topk_weights,
        topk_ids,
        implementation="static",
        workspace=pool,
        input_scales_static=True,
    ).clone()
    torch.cuda.synchronize(device)

    with pytest.raises(ValueError, match="chunked requests require a TPMoEWorkspacePool"):
        b12x_moe_fp4(
            x,
            weights.w13_input_scale_per_expert,
            weights.w13_weight,
            weights.w13_blockscale_swizzled,
            weights.g1_alphas_per_expert,
            weights.w2_input_scale_per_expert,
            weights.w2_weight,
            weights.w2_blockscale_swizzled,
            weights.g2_alphas_per_expert,
            topk_weights,
            topk_ids,
            implementation="static",
            workspace=exact_workspace,
            input_scales_static=True,
        )

    assert len(pool.workspaces) == 1
    metrics = compare_to_reference(actual, expected)
    assert metrics.max_abs <= 5e-4
    assert metrics.cos > 0.9999


def test_cuda_graph_capture_requires_output_buffer() -> None:
    require_sm120()
    _require_model_weights()

    clear_tp_moe_caches()

    device = torch.device("cuda")
    spec = _make_spec()
    weights = load_expert_weights(MODEL_PATH, spec, layer_idx=0)
    x, topk_ids, topk_weights = make_routed_inputs(spec, 1, seed=654, device=device)
    workspace = allocate_tp_moe_workspace(
        x,
        weights.w13_input_scale_per_expert,
        weights.w13_weight,
        weights.w2_input_scale_per_expert,
        weights.w2_weight,
        topk_ids,
        implementation="static",
        input_scales_static=True,
    )

    graph = torch.cuda.CUDAGraph()
    with pytest.raises(ValueError, match="caller-owned output buffer"):
        with torch.cuda.graph(graph):
            b12x_moe_fp4(
                x,
                weights.w13_input_scale_per_expert,
                weights.w13_weight,
                weights.w13_blockscale_swizzled,
                weights.g1_alphas_per_expert,
                weights.w2_input_scale_per_expert,
                weights.w2_weight,
                weights.w2_blockscale_swizzled,
                weights.g2_alphas_per_expert,
                topk_weights,
                topk_ids,
                implementation="static",
                workspace=workspace,
                input_scales_static=True,
            )


def test_dynamic_chunk_limit_uses_compact_layout() -> None:
    old_limit = tp_moe._safe_token_chunk(512, 4096, 256, 10)
    compact_limit = tp_moe._safe_dynamic_token_chunk(512, 4096, 256, 10)

    assert old_limit == 192
    assert compact_limit == 98304
    assert compact_limit > old_limit


def test_dynamic_workspace_uses_compact_storage() -> None:
    require_sm120()
    _require_model_weights()

    clear_tp_moe_caches()

    device = torch.device("cuda")
    spec = _make_spec()
    weights = load_expert_weights(MODEL_PATH, spec, layer_idx=0)
    x, topk_ids, _topk_weights = make_routed_inputs(spec, 1, seed=777, device=device)

    workspace = allocate_tp_moe_workspace(
        x,
        weights.w13_input_scale_per_expert,
        weights.w13_weight,
        weights.w2_input_scale_per_expert,
        weights.w2_weight,
        topk_ids,
        implementation="dynamic",
        input_scales_static=True,
    )

    n = spec.intermediate_size // spec.tp_size
    max_phys_tiles, _, max_tasks = tp_moe._dynamic_task_geometry(
        spec.num_experts,
        n,
        spec.top_k,
    )
    rows_padded = max_phys_tiles * tp_moe._LEVEL_TILE_M
    cols_pad_k = tp_moe.align_up(spec.hidden_size // tp_moe._NVFP4_BLOCK_SIZE, 4)

    assert workspace.scheduler_out is None
    assert workspace.expert_write_rows is not None
    assert workspace.expert_tile_base is not None
    assert tuple(workspace.token_map.shape) == (rows_padded,)
    assert tuple(workspace.token_weights_map.shape) == (rows_padded,)
    assert tuple(workspace.packed_input.shape) == (1, rows_padded, spec.hidden_size // 2)
    assert tuple(workspace.packed_input_scale.shape) == (rows_padded, cols_pad_k)
    assert tuple(workspace.tile_write_count.shape) == (max_phys_tiles,)
    assert tuple(workspace.task_ready.shape) == (max_tasks,)
