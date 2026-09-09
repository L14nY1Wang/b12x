"""Batch-bounded io_uring reads and mapped-host row staging for PLE."""

from __future__ import annotations

import operator
import os
import threading
from typing import TYPE_CHECKING

import torch

from ._storage import _MappedHostAllocation

if TYPE_CHECKING:
    from ._contracts import Binding, Plan


class DiskTable:
    """Own immutable O_DIRECT file sources and a reusable compact batch cache.

    ``run`` performs host I/O outside CUDA graphs/torch.compile. It produces
    ordinary fixed-address GPU outputs which downstream graphs may consume.
    Files must remain immutable while this owner is alive. Cache allocations
    and native I/O buffers depend on batch capacity, never table payload size.
    Calls are serialized, including across CUDA streams; consumers of ``out``
    must follow ordinary caller-stream ordering before the next preparation.
    """

    def __init__(
        self,
        plan: Plan,
        shard_rows: int,
        *,
        queue_depth: int = 64,
    ) -> None:
        from b12x.loader._native import load

        from ._contracts import Plan

        if not isinstance(plan, Plan):
            raise TypeError("plan must be Plan")
        if plan.caps.table_memory != "io_uring":
            raise ValueError("DiskTable requires table_memory='io_uring'")
        shard_rows = operator.index(shard_rows)
        if not 0 < shard_rows <= (1 << 63) - 1:
            raise ValueError("shard_rows must be a positive signed int64")
        queue_depth = operator.index(queue_depth)
        if queue_depth <= 0:
            raise ValueError("queue_depth must be positive")
        self.plan = plan
        self.shard_rows = shard_rows
        self.shard_count = (plan.padded_vocab_size + shard_rows - 1) // shard_rows
        self.max_lookups = plan.caps.max_tokens * plan.head_count
        self.weight_row_bytes = plan.weight_shape[1] * plan.weight_dtype.itemsize
        self.scale_row_bytes = (
            plan.head_dim // 16 if plan.caps.quant_mode == "nvfp4_group16" else 0
        )
        self._native = load()
        self._reader = self._native.ple_reader(
            shard_rows,
            plan.padded_vocab_size,
            plan.shard_start,
            plan.shard_end,
            self.weight_row_bytes,
            self.scale_row_bytes,
            self.max_lookups,
            queue_depth,
        )
        self.ids_host = torch.empty(
            (plan.caps.max_tokens, plan.head_count),
            dtype=torch.int64,
            device="cpu",
            pin_memory=True,
        )
        self._ids_buffer = memoryview(self.ids_host.numpy())
        self._weight_allocation = _MappedHostAllocation(
            (self.max_lookups, plan.weight_shape[1]),
            plan.weight_dtype,
            plan.caps.device,
        )
        self.weight = self._weight_allocation.device_view
        self.weight_host = self._weight_allocation.host_view
        self._weight_buffer = memoryview(self.weight_host.view(torch.uint8).numpy())
        self._scale_allocation = None
        self.weight_scale = None
        self.weight_scale_host = None
        self._scale_buffer = None
        if self.scale_row_bytes:
            self._scale_allocation = _MappedHostAllocation(
                (self.max_lookups, self.scale_row_bytes),
                torch.float8_e4m3fn,
                plan.caps.device,
            )
            self.weight_scale = self._scale_allocation.device_view
            self.weight_scale_host = self._scale_allocation.host_view
            self._scale_buffer = memoryview(
                self.weight_scale_host.view(torch.uint8).numpy()
            )
        self._sources: set[tuple[bool, int]] = set()
        self._frozen = False
        self._lock = threading.Lock()
        self._ids_ready = torch.cuda.Event()
        self._cache_done = torch.cuda.Event()
        self._cache_used = False

    def add_shard(
        self, shard_index: int, path: str, offset: int, *, scale: bool = False
    ) -> None:
        """Register one immutable complete checkpoint shard, without reading it."""
        if self._frozen:
            raise RuntimeError("cannot change disk shards after binding")
        shard_index = operator.index(shard_index)
        offset = operator.index(offset)
        if not 0 <= shard_index < self.shard_count:
            raise ValueError("checkpoint shard index is out of range")
        if offset < 0:
            raise ValueError("checkpoint file offset must be nonnegative")
        if scale and not self.scale_row_bytes:
            raise ValueError("only NVFP4 has disk row scales")
        key = (scale, shard_index)
        if key in self._sources:
            raise ValueError("checkpoint shard is already registered")
        start = shard_index * self.shard_rows
        end = min(start + self.shard_rows, self.plan.padded_vocab_size)
        if end <= self.plan.shard_start or start >= self.plan.shard_end:
            return
        self._native.ple_reader_add(
            self._reader, shard_index, os.fspath(path), offset, scale
        )
        self._sources.add(key)

    def _require_complete(self) -> None:
        first = self.plan.shard_start // self.shard_rows
        last = (self.plan.shard_end + self.shard_rows - 1) // self.shard_rows
        for shard_index in range(first, last):
            if (False, shard_index) not in self._sources:
                raise ValueError(f"missing disk weight shard {shard_index}")
            if self.scale_row_bytes and (True, shard_index) not in self._sources:
                raise ValueError(f"missing disk scale shard {shard_index}")

    def stats(self) -> dict[str, int | float]:
        """Last native call counters plus persistent Python-owned staging bytes.

        ``staging_bytes`` and ``metadata_bytes`` are native allocations.
        ``ids_host_bytes`` and ``cache_bytes`` are additional owned pinned bytes;
        GPU aliases share cache storage and allocate no second payload copy.
        Source registry/descriptor metadata scales with the local shard count.
        """
        with self._lock:
            result = dict(self._native.ple_reader_stats(self._reader))
            result["ids_host_bytes"] = (
                self.ids_host.numel() * self.ids_host.element_size()
            )
            result["weight_cache_bytes"] = self._weight_allocation.nbytes
            result["scale_cache_bytes"] = (
                self._scale_allocation.nbytes if self._scale_allocation else 0
            )
            result["cache_bytes"] = (
                result["weight_cache_bytes"] + result["scale_cache_bytes"]
            )
            result["owned_staging_bytes"] = (
                result["staging_bytes"]
                + result["ids_host_bytes"]
                + result["cache_bytes"]
            )
            return result

    def _run(self, binding: Binding, *, token_count: int) -> None:
        from ._kernels import (
            _launch_bf16_lookup,
            _launch_fp8_lookup,
            _launch_hash,
            _launch_nvfp4_lookup,
        )

        if torch.compiler.is_compiling():
            raise RuntimeError("disk PLE preparation cannot run under torch.compile")
        plan = self.plan
        caps = plan.caps
        with torch.cuda.device(caps.device):
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError(
                    "disk PLE preparation must run outside CUDA graph capture"
                )
            with self._lock:
                stream = torch.cuda.current_stream(caps.device)
                # This dependency also covers reuse from a different caller stream.
                # The host event wait below completes all previous cache readers
                # before native I/O overwrites any mapped row bytes.
                if self._cache_used:
                    stream.wait_event(self._cache_done)
                _launch_hash(
                    binding.token_ids,
                    binding.query_start_loc,
                    binding.committed_history,
                    binding.num_seqs,
                    binding.num_tokens,
                    plan.multipliers,
                    plan.prime_sizes,
                    plan.table_offsets,
                    binding._ids,
                    binding._hash_binding.request_ids,
                    binding.error_code,
                    caps.eos_token_id,
                    caps.vocab_size,
                    caps.max_order,
                    caps.heads_per_order,
                    caps.max_seqs,
                    caps.max_tokens,
                    token_count,
                )
                self.ids_host[:token_count].copy_(
                    binding._ids[:token_count], non_blocking=True
                )
                self._ids_ready.record(stream)
                self._ids_ready.synchronize()
                self._native.ple_reader_run(
                    self._reader,
                    self._ids_buffer,
                    self._weight_buffer,
                    self._scale_buffer,
                    token_count * plan.head_count,
                )
                if token_count == 0:
                    return
                args = (
                    binding._ids,
                    binding.num_tokens,
                    binding.out[:token_count],
                    caps.max_tokens,
                    plan.head_count,
                    plan.head_dim,
                    caps.embedding_dim,
                    plan.table_vocab_size,
                    plan.shard_start,
                    plan.shard_end,
                )
                try:
                    if caps.quant_mode == "bf16":
                        _launch_bf16_lookup(self.weight, *args, compact_rows=True)
                    elif caps.quant_mode == "fp8_e4m3_per_tensor":
                        _launch_fp8_lookup(
                            self.weight, binding.weight_scale, *args, compact_rows=True
                        )
                    else:
                        _launch_nvfp4_lookup(
                            self.weight,
                            self.weight_scale,
                            binding.weight_scale_2,
                            *args,
                            compact_rows=True,
                        )
                finally:
                    self._cache_done.record(stream)
                    self._cache_used = True
