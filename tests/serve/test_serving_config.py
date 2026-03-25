"""Tests for serving-engine configuration policy helpers."""

from serve.engine.serving import _should_enable_prefix_cache


def test_prefix_cache_enabled_for_non_hybrid_models():
    assert _should_enable_prefix_cache(
        is_hybrid=False,
        world_size=4,
        has_state_snapshot_slots=False,
    )


def test_prefix_cache_enabled_for_single_gpu_hybrid_with_snapshots():
    assert _should_enable_prefix_cache(
        is_hybrid=True,
        world_size=1,
        has_state_snapshot_slots=True,
    )


def test_prefix_cache_disabled_for_single_gpu_hybrid_without_snapshots():
    assert not _should_enable_prefix_cache(
        is_hybrid=True,
        world_size=1,
        has_state_snapshot_slots=False,
    )


def test_prefix_cache_disabled_for_tp_hybrid_even_with_snapshots():
    assert not _should_enable_prefix_cache(
        is_hybrid=True,
        world_size=4,
        has_state_snapshot_slots=True,
    )
