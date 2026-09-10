"""Lockstep tests for the NVFP4 split-materialized dispatch policy.

Covers the structural predicate (_nvfp4_dynamic_dense_candidate), the env gate
(_nvfp4_dynamic_materialized_enabled), workspace sizing, and one e2e smoke
test through the production dispatch.
"""

from __future__ import annotations

import os
import pytest
import torch

from b12x.moe.fused_moe._impl import (
    _nvfp4_dynamic_dense_candidate,
    _nvfp4_dynamic_materialized_enabled,
    _nvfp4_materialized_env_refresh,
    _plan_core_workspace,
    _DYNAMIC_NVFP4_MATERIALIZED_ENV,
    _DYNAMIC_WORK_SOURCE_ENV,
)


def _dense_args(**overrides):
    """Canonical dense-prefill arguments that should match the predicate.
    Only pass fields that the structural predicate accepts (no share_input)."""
    args = dict(
        quant_mode="nvfp4",
        activation="silu",
        routed_rows=4096,
        num_experts=16,
        k=4096,
        n=2048,
        deterministic_output=False,
    )
    args.update(overrides)
    return args


def _enabled_args(**overrides):
    """Arguments for the enabled gate (structural + share_input_across_experts)."""
    args = dict(
        **_dense_args(),
        share_input_across_experts=True,
    )
    args.update(overrides)
    return args


class TestNvfp4SplitPredicate:
    @pytest.fixture(autouse=True)
    def _env_backup(self):
        saved = os.environ.get(_DYNAMIC_NVFP4_MATERIALIZED_ENV)
        yield
        if saved is None:
            os.environ.pop(_DYNAMIC_NVFP4_MATERIALIZED_ENV, None)
        else:
            os.environ[_DYNAMIC_NVFP4_MATERIALIZED_ENV] = saved

    def test_accepted_dense(self):
        """Reference dense prefill satisfies both candidate and enabled
        (auto-on for matching shapes after measured win)."""
        assert _nvfp4_dynamic_dense_candidate(**_dense_args()) is True
        # Default auto-on: predicate + share_input → enabled without env.
        assert _nvfp4_dynamic_materialized_enabled(**_enabled_args()) is True
        # Explicit toggle-off works.
        os.environ[_DYNAMIC_NVFP4_MATERIALIZED_ENV] = "0"
        assert _nvfp4_dynamic_materialized_enabled(**_enabled_args()) is False
        # Explicit toggle-on works.
        os.environ[_DYNAMIC_NVFP4_MATERIALIZED_ENV] = "1"
        assert _nvfp4_dynamic_materialized_enabled(**_enabled_args()) is True

    def test_rejects_non_nvfp4(self):
        # w4a8_mx is a valid quant_mode that is not nvfp4 → predicate False.
        assert _nvfp4_dynamic_dense_candidate(**_dense_args(quant_mode="w4a8_mx")) is False
        # w6a8_mx is also a valid non-nvfp4 mode.
        assert _nvfp4_dynamic_dense_candidate(**_dense_args(quant_mode="w6a8_mx")) is False

    def test_rejects_bad_activation(self):
        assert _nvfp4_dynamic_dense_candidate(**_dense_args(activation="relu2")) is False

    def test_rejects_k_not_divisible_by_128(self):
        assert _nvfp4_dynamic_dense_candidate(**_dense_args(k=2047)) is False
        assert _nvfp4_dynamic_dense_candidate(**_dense_args(k=2048)) is True

    def test_rejects_n_not_divisible_by_128(self):
        assert _nvfp4_dynamic_dense_candidate(**_dense_args(n=700)) is False
        assert _nvfp4_dynamic_dense_candidate(**_dense_args(n=512)) is True

    def test_rejects_deterministic_output(self):
        assert _nvfp4_dynamic_dense_candidate(**_dense_args(deterministic_output=True)) is False

    def test_rejects_tile_64_and_16(self):
        """E=288 domains that drive the tile planner to 64 and 16 are both
        rejected: the split phase kernels are validated only for the M128
        source tile (the 64-row multi-tile path is known-wrong)."""
        big_e = dict(_dense_args(), num_experts=288)
        # routed_rows >= 96*E -> tile 128 (accepted)
        assert _nvfp4_dynamic_dense_candidate(**{**big_e, "routed_rows": 128 * 288}) is True
        # routed_rows in [48*E, 96*E) -> tile 64 (now rejected, not validated)
        assert _nvfp4_dynamic_dense_candidate(**{**big_e, "routed_rows": 80 * 288}) is False
        # routed_rows < 48*E -> tile 16 (rejected)
        assert _nvfp4_dynamic_dense_candidate(**{**big_e, "routed_rows": 10 * 288}) is False

    def test_rejects_ready_queue_work_source(self):
        """Streaming (ready_queue) work source is not supported by the split."""
        os.environ[_DYNAMIC_WORK_SOURCE_ENV] = "ready_queue"
        try:
            assert _nvfp4_dynamic_dense_candidate(**_dense_args()) is False
        finally:
            os.environ.pop(_DYNAMIC_WORK_SOURCE_ENV, None)

    def test_rejects_non_shared_input_even_with_env(self):
        """Without share_input_across_experts the enabled gate rejects even with env."""
        os.environ[_DYNAMIC_NVFP4_MATERIALIZED_ENV] = "1"
        _nvfp4_materialized_env_refresh()
        # Predicate is satisfied (share_input is not a predicate check).
        assert _nvfp4_dynamic_dense_candidate(**_dense_args()) is True
        # But the enabled gate demands share_input.
        args_on = _dense_args(share_input_across_experts=False)
        assert _nvfp4_dynamic_materialized_enabled(**args_on) is False

    def test_env_auto_on_for_matching(self):
        """Without any env set, predicate+share_input → auto-on (matching shapes)."""
        os.environ.pop(_DYNAMIC_NVFP4_MATERIALIZED_ENV, None)
        _nvfp4_materialized_env_refresh()
        assert _nvfp4_dynamic_materialized_enabled(**_enabled_args()) is True

    def test_env_off_explicitly(self):
        os.environ[_DYNAMIC_NVFP4_MATERIALIZED_ENV] = "0"
        _nvfp4_materialized_env_refresh()
        assert _nvfp4_dynamic_materialized_enabled(**_enabled_args()) is False

    def test_non_matching_defaults_false(self):
        """Non-matching shapes (no share_input) default False even without env."""
        args = _dense_args()
        os.environ.pop(_DYNAMIC_NVFP4_MATERIALIZED_ENV, None)
        _nvfp4_materialized_env_refresh()
        assert _nvfp4_dynamic_materialized_enabled(**_enabled_args(share_input_across_experts=False)) is False


class TestNvfp4SplitWorkspace:
    """The plan-time workspace sizing for NVFP4 intermediate must cover
    payload + scale planes: rows_padded * (n//128) * 72 bytes."""

    @pytest.fixture(autouse=True)
    def _env_backup(self):
        saved = os.environ.get(_DYNAMIC_NVFP4_MATERIALIZED_ENV)
        yield
        if saved is None:
            os.environ.pop(_DYNAMIC_NVFP4_MATERIALIZED_ENV, None)
        else:
            os.environ[_DYNAMIC_NVFP4_MATERIALIZED_ENV] = saved
        _nvfp4_materialized_env_refresh()

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
    def test_intermediate_bytes_sufficient(self):
        # Domain large enough that the tile planner selects (128,128), matching
        # the split predicate (real prefill shapes are far above this).  Env
        # unset → auto-on (default True), so the plan allocates the NVFP4
        # intermediate scratch.
        os.environ.pop(_DYNAMIC_NVFP4_MATERIALIZED_ENV, None)
        _nvfp4_materialized_env_refresh()
        plan = _plan_core_workspace(
            implementation="b12x",
            quant_mode="nvfp4",
            state_E=8,
            weight_E=8,
            k=4096,
            n=2048,
            num_topk=2,
            device=torch.device("cuda"),
            dtype=torch.bfloat16,
            routed_rows=2048,
            max_rows=8192,
        )
        # The plan sets dynamic_physical_tiles and dynamic_tile_m.
        tile_m = plan.dynamic_tile_m
        phys_tiles = plan.dynamic_physical_tiles
        assert phys_tiles is not None and tile_m is not None
        assert tile_m == 128, tile_m  # split predicate requires the validated M128 tile
        rows_padded = phys_tiles * tile_m
        needed = rows_padded * (plan.n // 128) * 72
        for spec in plan.tensor_specs:
            if spec.name == "materialized_intermediate":
                elt = torch.tensor([], dtype=plan.dtype).element_size()
                allocated = spec.shape[0] * spec.shape[1] * elt
                assert allocated >= needed, f"NVFP4 intermediate allocated {allocated}B < needed {needed}B"
                return
        pytest.fail("materialized_intermediate tensor spec not found in plan")