"""Persistent PLE table storage allocation."""

from __future__ import annotations

import math
import operator
import sys
from contextlib import suppress
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from cuda.bindings import runtime as cudart

if TYPE_CHECKING:
    from ._contracts import Plan


def _check_cuda(error: cudart.cudaError_t, operation: str) -> None:
    if error != cudart.cudaError_t.cudaSuccess:
        raise RuntimeError(f"{operation} failed: {error}")


def _contiguous_strides(shape: tuple[int, ...]) -> tuple[int, ...]:
    stride = 1
    result = []
    for extent in reversed(shape):
        result.append(stride)
        stride *= int(extent)
    return tuple(reversed(result))


def _tensor_from_pointer(
    pointer: int,
    *,
    shape: tuple[int, ...],
    dtype: torch.dtype,
    device: torch.device,
    nbytes: int,
) -> torch.Tensor:
    constructor = getattr(torch._C, "_construct_storage_from_data_pointer", None)
    if constructor is None:
        raise RuntimeError(
            "mapped-host PLE storage requires "
            "torch._C._construct_storage_from_data_pointer"
        )
    storage = constructor(int(pointer), device, int(nbytes))
    return torch.empty(0, dtype=dtype, device=device).set_(
        storage,
        0,
        shape,
        _contiguous_strides(shape),
    )


class _MappedHostAllocation:
    """Own one mapped page-locked allocation and its two tensor aliases."""

    def __init__(
        self,
        shape: tuple[int, ...],
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        if device.type != "cuda" or device.index is None:
            raise ValueError(
                f"mapped-host PLE storage requires an indexed CUDA device, got {device}"
            )
        element_size = int(torch.empty((), dtype=dtype).element_size())
        nbytes = math.prod(shape) * element_size
        if nbytes <= 0:
            raise ValueError(
                f"mapped-host allocation size must be positive, got {nbytes}"
            )

        self.device = device
        self.nbytes = nbytes
        self._host_pointer = 0
        self._closed = False

        with torch.cuda.device(device):
            error, host_pointer = cudart.cudaHostAlloc(
                nbytes,
                cudart.cudaHostAllocMapped | cudart.cudaHostAllocWriteCombined,
            )
            _check_cuda(error, "cudaHostAlloc")
            self._host_pointer = int(host_pointer)
            try:
                error, device_pointer = cudart.cudaHostGetDevicePointer(host_pointer, 0)
                _check_cuda(error, "cudaHostGetDevicePointer")
                self.host_view = _tensor_from_pointer(
                    self._host_pointer,
                    shape=shape,
                    dtype=dtype,
                    device=torch.device("cpu"),
                    nbytes=nbytes,
                )
                self.device_view = _tensor_from_pointer(
                    int(device_pointer),
                    shape=shape,
                    dtype=dtype,
                    device=device,
                    nbytes=nbytes,
                )
            except Exception:
                cudart.cudaFreeHost(host_pointer)
                self._host_pointer = 0
                self._closed = True
                raise

    def close(self) -> None:
        """Release the allocation after all GPU access has completed."""
        if self._closed:
            return
        torch.cuda.synchronize(self.device)
        error = cudart.cudaFreeHost(self._host_pointer)[0]
        _check_cuda(error, "cudaFreeHost")
        self._host_pointer = 0
        self._closed = True

    def __del__(self) -> None:
        if self._closed or self._host_pointer == 0 or sys.is_finalizing():
            return
        with suppress(Exception):
            self.close()


class MMapTable:
    """Own immutable checkpoint mappings and fixed device shard-pointer arrays.

    Only pointer metadata is allocated on the GPU. Mappings use read-only,
    private, demand-paged file storage, including on CUDA HMM devices. Keep
    this owner (normally through a Binding) alive for all graph replays and
    keep the checkpoint files immutable for its entire lifetime.
    """

    def __init__(self, plan: Plan, shard_rows: int) -> None:
        from ._contracts import Plan

        if not isinstance(plan, Plan):
            raise TypeError("plan must be Plan")
        if plan.caps.table_memory != "mmap":
            raise ValueError("MMapTable requires table_memory='mmap'")
        shard_rows = operator.index(shard_rows)
        if shard_rows <= 0 or shard_rows > (1 << 63) - 1:
            raise ValueError("shard_rows must be a positive signed int64")
        device = plan.caps.device
        if device.type != "cuda" or device.index is None:
            raise ValueError("mmap PLE storage requires an indexed CUDA device")
        error, pageable = cudart.cudaDeviceGetAttribute(
            cudart.cudaDeviceAttr.cudaDevAttrPageableMemoryAccess, device.index
        )
        _check_cuda(error, "cudaDeviceGetAttribute")
        if not pageable:
            raise RuntimeError("mmap PLE storage requires GPU pageable memory access")
        self.plan = plan
        self.shard_rows = shard_rows
        self.shard_count = (plan.padded_vocab_size + shard_rows - 1) // shard_rows
        self.weight_pointers = torch.zeros(
            self.shard_count, dtype=torch.int64, device=device
        )
        self.scale_pointers = (
            torch.zeros(self.shard_count, dtype=torch.int64, device=device)
            if plan.caps.quant_mode == "nvfp4_group16"
            else None
        )
        self._mappings: dict[tuple[bool, int], torch.Tensor] = {}
        self._frozen = False
        self.mapped_file_nbytes = 0

    def map_shard(
        self, shard_index: int, path: str, offset: int, *, scale: bool = False
    ) -> None:
        """Map a complete checkpoint shard when it overlaps this TP rank.

        Geometry comes from the plan, not from untrusted file metadata. File
        range and dtype alignment are checked by the owning native reader
        before mmap. This call never reads or copies tensor payload bytes.
        """
        from b12x.loader import read_tensor

        if self._frozen:
            raise RuntimeError("cannot change mmap shards after binding")
        shard_index = operator.index(shard_index)
        offset = operator.index(offset)
        if shard_index < 0 or shard_index >= self.shard_count:
            raise ValueError("checkpoint shard index is out of range")
        if offset < 0:
            raise ValueError("checkpoint file offset must be nonnegative")
        if scale and self.scale_pointers is None:
            raise ValueError("only NVFP4 has mapped row scales")
        key = (scale, shard_index)
        if key in self._mappings:
            raise ValueError("checkpoint shard is already mapped")
        start = shard_index * self.shard_rows
        end = min(start + self.shard_rows, self.plan.padded_vocab_size)
        if end <= self.plan.shard_start or start >= self.plan.shard_end:
            return
        if scale:
            assert self.plan.weight_scale_shape is not None
            assert self.plan.weight_scale_dtype is not None
            width = self.plan.weight_scale_shape[1]
            dtype = self.plan.weight_scale_dtype
            pointers = self.scale_pointers
        else:
            width = self.plan.weight_shape[1]
            dtype = self.plan.weight_dtype
            pointers = self.weight_pointers
        tensor = read_tensor(
            path,
            shape=(end - start, width),
            dtype=dtype,
            offset=offset,
            allocation="file_readonly",
            device=self.plan.caps.device.index,
        )
        assert pointers is not None
        # Publish only the address; never pass the unregistered mapping as a
        # Triton argument (its launcher may reject pageable host pointers).
        pointers[shard_index].fill_(tensor.data_ptr())
        self._mappings[key] = tensor
        self.mapped_file_nbytes += tensor.numel() * tensor.element_size()

    def _require_complete(self) -> None:
        first = self.plan.shard_start // self.shard_rows
        last = (self.plan.shard_end + self.shard_rows - 1) // self.shard_rows
        for shard_index in range(first, last):
            if (False, shard_index) not in self._mappings:
                raise ValueError(f"missing mmap weight shard {shard_index}")
            if (
                self.scale_pointers is not None
                and (True, shard_index) not in self._mappings
            ):
                raise ValueError(f"missing mmap scale shard {shard_index}")


@dataclass(kw_only=True)
class TableStorage:
    """Owning persistent table tensors and their checkpoint loading views.

    Kernel-visible tensors are always CUDA tensors. For mapped-host table
    storage, a loading view is a CPU tensor over the same page-locked bytes.
    The owner must outlive every binding that references its tensors.
    """

    weight: torch.Tensor
    weight_scale: torch.Tensor | None
    weight_scale_2: torch.Tensor | None
    weight_load_view: torch.Tensor
    weight_scale_load_view: torch.Tensor | None
    weight_scale_2_load_view: torch.Tensor | None
    mapped_host_nbytes: int
    _mapped_allocations: tuple[_MappedHostAllocation, ...]

    def close(self) -> None:
        """Synchronize and release any mapped-host allocations."""
        for allocation in reversed(self._mapped_allocations):
            allocation.close()


def _device_tensor(
    shape: tuple[int, ...], dtype: torch.dtype, device: torch.device
) -> torch.Tensor:
    return torch.empty(shape, dtype=dtype, device=device)


def allocate_storage(plan: Plan) -> TableStorage:
    """Allocate persistent table storage according to the planned policy."""
    from ._contracts import Plan

    if not isinstance(plan, Plan):
        raise TypeError(f"plan must be Plan, got {type(plan)!r}")
    caps = plan.caps
    if caps.table_memory == "mmap":
        raise ValueError("mmap tables must be loaded with MMapTable(plan, shard_rows)")

    allocations: list[_MappedHostAllocation] = []

    def table_tensor(
        shape: tuple[int, ...], dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if caps.table_memory == "device":
            tensor = _device_tensor(shape, dtype, caps.device)
            return tensor, tensor
        allocation = _MappedHostAllocation(shape, dtype, caps.device)
        allocations.append(allocation)
        return allocation.device_view, allocation.host_view

    weight, weight_load_view = table_tensor(plan.weight_shape, plan.weight_dtype)

    weight_scale: torch.Tensor | None = None
    weight_scale_load_view: torch.Tensor | None = None
    if plan.weight_scale_shape is not None:
        assert plan.weight_scale_dtype is not None
        if caps.quant_mode == "nvfp4_group16":
            weight_scale, weight_scale_load_view = table_tensor(
                plan.weight_scale_shape, plan.weight_scale_dtype
            )
        else:
            weight_scale = _device_tensor(
                plan.weight_scale_shape, plan.weight_scale_dtype, caps.device
            )
            weight_scale_load_view = weight_scale

    weight_scale_2: torch.Tensor | None = None
    weight_scale_2_load_view: torch.Tensor | None = None
    if plan.weight_scale_2_shape is not None:
        assert plan.weight_scale_2_dtype is not None
        weight_scale_2 = _device_tensor(
            plan.weight_scale_2_shape, plan.weight_scale_2_dtype, caps.device
        )
        weight_scale_2_load_view = weight_scale_2

    return TableStorage(
        weight=weight,
        weight_scale=weight_scale,
        weight_scale_2=weight_scale_2,
        weight_load_view=weight_load_view,
        weight_scale_load_view=weight_scale_load_view,
        weight_scale_2_load_view=weight_scale_2_load_view,
        mapped_host_nbytes=sum(allocation.nbytes for allocation in allocations),
        _mapped_allocations=tuple(allocations),
    )


__all__ = ["MMapTable", "TableStorage", "allocate_storage"]
