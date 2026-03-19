"""Planner-backed public attention entrypoints for the transplanted SM120 kernel."""

from __future__ import annotations

import functools
from dataclasses import dataclass, field

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch

from b12x.attention.combine import PagedAttentionCombineKernel
from b12x.attention.forward import SM120ForwardKernel
from b12x.cute.utils import current_cuda_stream, make_ptr

_DEFAULT_PAGED_SPLIT_BUCKETS = (1, 2, 4, 8)
_DEFAULT_MIN_PAGES_PER_SPLIT = 8


def _torch_to_cutlass_dtype(dtype: torch.dtype) -> type[cutlass.Numeric]:
    if dtype == torch.bfloat16:
        return cutlass.BFloat16
    if dtype == torch.float16:
        return cutlass.Float16
    raise TypeError(f"unsupported dtype {dtype}; expected torch.bfloat16 or torch.float16")


def _contiguous_stride(shape: tuple[int, ...]) -> tuple[int, ...]:
    if not shape:
        return ()
    stride = [1] * len(shape)
    running = 1
    for idx in range(len(shape) - 1, -1, -1):
        stride[idx] = running
        running *= shape[idx]
    return tuple(stride)


def _lse_shape(q_shape: tuple[int, ...]) -> tuple[int, ...]:
    if len(q_shape) == 3:
        seqlen_q, q_heads, _ = q_shape
        return (q_heads, seqlen_q)
    batch, seqlen_q, q_heads, _ = q_shape
    return (batch, q_heads, seqlen_q)


def _token_major_lse_shape(q_shape: tuple[int, ...]) -> tuple[int, int]:
    if len(q_shape) != 3:
        raise ValueError(f"expected rank-3 q shape for paged attention, got {q_shape}")
    total_q, q_heads, _ = q_shape
    return (total_q, q_heads)


def _paged_lse_storage_shape(q_shape: tuple[int, ...]) -> tuple[int, int]:
    if len(q_shape) != 3:
        raise ValueError(f"expected rank-3 q shape for paged attention, got {q_shape}")
    total_q, q_heads, _ = q_shape
    return (q_heads, total_q)


def _split_paged_output_shape(q_shape: tuple[int, ...], *, num_splits: int) -> tuple[int, ...]:
    if num_splits == 1:
        return q_shape
    return (num_splits, *q_shape)


def _split_paged_lse_storage_shape(
    q_shape: tuple[int, ...], *, num_splits: int
) -> tuple[int, ...]:
    base = _paged_lse_storage_shape(q_shape)
    if num_splits == 1:
        return base
    return (num_splits, *base)


def _seq_dims(shape: tuple[int, ...]) -> tuple[tuple[int, ...], int, int, int]:
    if len(shape) == 3:
        seqlen, num_heads, head_dim = shape
        return (), seqlen, num_heads, head_dim
    if len(shape) == 4:
        batch, seqlen, num_heads, head_dim = shape
        return (batch,), seqlen, num_heads, head_dim
    raise ValueError(f"expected rank-3 or rank-4 tensor shape, got {shape}")


def _select_tile_shape(head_dim: int, *, causal: bool) -> tuple[int, int]:
    if head_dim <= 64:
        return (128, 128)
    if head_dim <= 128:
        return (128, 64)
    if head_dim == 256:
        return (64, 32 if causal else 48)
    raise ValueError(f"unsupported head_dim={head_dim} for the current b12x attention path")


def _select_paged_tile_shape(head_dim: int, *, causal: bool, page_size: int) -> tuple[int, int]:
    if not causal:
        raise ValueError("b12x paged attention currently supports causal mode only")
    if page_size != 64:
        raise ValueError(
            f"b12x paged attention currently requires page_size=64 for the TMA path, got {page_size}"
        )
    if head_dim <= 128:
        return (128, 64)
    if head_dim == 256:
        return (64, 64)
    raise ValueError(
        f"unsupported head_dim={head_dim} for the current b12x paged attention path"
    )


def _normalize_tensor_shape(t: torch.Tensor) -> tuple[int, ...]:
    return tuple(int(dim) for dim in t.shape)


def _metadata_to_cpu_int_list(t: torch.Tensor, *, name: str) -> list[int]:
    if t.dtype not in (torch.int32, torch.int64):
        raise TypeError(f"{name} must be torch.int32 or torch.int64")
    if not t.is_contiguous():
        raise ValueError(f"{name} must be contiguous")
    return [int(v) for v in t.detach().cpu().tolist()]


def _cuda_device_index(device: torch.device) -> int:
    if device.type != "cuda":
        raise ValueError(f"expected CUDA device, got {device}")
    return torch.cuda.current_device() if device.index is None else int(device.index)


def _normalize_split_buckets(split_buckets: tuple[int, ...]) -> tuple[int, ...]:
    if not split_buckets:
        raise ValueError("split_buckets must be non-empty")
    normalized = tuple(sorted(set(int(bucket) for bucket in split_buckets)))
    if normalized[0] != 1:
        raise ValueError(f"split_buckets must include 1, got {normalized}")
    if any(bucket < 1 for bucket in normalized):
        raise ValueError(f"split_buckets must be positive, got {normalized}")
    return normalized


def _validate_forward_inputs(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...], torch.device, torch.dtype]:
    if q.ndim not in (3, 4):
        raise ValueError(f"q must be rank-3 or rank-4, got rank {q.ndim}")
    if q.ndim != k.ndim or q.ndim != v.ndim:
        raise ValueError("q, k, and v must have the same rank")
    if q.device.type != "cuda" or k.device != q.device or v.device != q.device:
        raise ValueError("q, k, and v must all be CUDA tensors on the same device")
    if q.dtype != k.dtype or q.dtype != v.dtype:
        raise ValueError("q, k, and v must have the same dtype")
    if q.dtype not in (torch.bfloat16, torch.float16):
        raise TypeError(f"unsupported dtype {q.dtype}; expected torch.bfloat16 or torch.float16")
    if not q.is_contiguous() or not k.is_contiguous() or not v.is_contiguous():
        raise ValueError("q, k, and v must all be contiguous")

    q_shape = _normalize_tensor_shape(q)
    k_shape = _normalize_tensor_shape(k)
    v_shape = _normalize_tensor_shape(v)
    batch_q, _, q_heads, q_head_dim = _seq_dims(q_shape)
    batch_k, _, kv_heads, k_head_dim = _seq_dims(k_shape)
    batch_v, _, v_heads, v_head_dim = _seq_dims(v_shape)
    if batch_q != batch_k or batch_q != batch_v:
        raise ValueError("q, k, and v must have matching batch dimensions")
    if q_head_dim != k_head_dim or q_head_dim != v_head_dim:
        raise ValueError("q, k, and v must have matching head dimensions in the initial path")
    if kv_heads != v_heads:
        raise ValueError("k and v must have the same number of KV heads")
    if q_heads % kv_heads != 0:
        raise ValueError(f"q_heads={q_heads} must be divisible by kv_heads={kv_heads}")
    return q_shape, k_shape, v_shape, q.device, q.dtype


def _inspect_paged_forward_inputs(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    page_table: torch.Tensor,
    cache_seqlens: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
) -> tuple[
    tuple[int, ...],
    tuple[int, ...],
    tuple[int, ...],
    tuple[int, ...],
    torch.device,
    torch.dtype,
    int,
]:
    if q.ndim != 3:
        raise ValueError(f"paged q must be rank-3 [total_q, q_heads, head_dim], got rank {q.ndim}")
    if k_cache.ndim != 4 or v_cache.ndim != 4:
        raise ValueError(
            "paged k_cache and v_cache must be rank-4 [num_pages, page_size, kv_heads, head_dim]"
        )
    if page_table.ndim != 2:
        raise ValueError(f"page_table must be rank-2 [batch, max_pages], got rank {page_table.ndim}")
    if cache_seqlens.ndim != 1:
        raise ValueError(f"cache_seqlens must be rank-1 [batch], got rank {cache_seqlens.ndim}")
    if cu_seqlens_q.ndim != 1:
        raise ValueError(f"cu_seqlens_q must be rank-1 [batch + 1], got rank {cu_seqlens_q.ndim}")
    if (
        q.device.type != "cuda"
        or k_cache.device != q.device
        or v_cache.device != q.device
        or page_table.device != q.device
        or cache_seqlens.device != q.device
        or cu_seqlens_q.device != q.device
    ):
        raise ValueError("paged attention tensors and metadata must all be CUDA tensors on the same device")
    if q.dtype != k_cache.dtype or q.dtype != v_cache.dtype:
        raise ValueError("paged attention currently requires q, k_cache, and v_cache to share one dtype")
    if q.dtype not in (torch.bfloat16, torch.float16):
        raise TypeError(f"unsupported dtype {q.dtype}; expected torch.bfloat16 or torch.float16")
    if not q.is_contiguous() or not k_cache.is_contiguous() or not v_cache.is_contiguous():
        raise ValueError("paged q, k_cache, and v_cache must be contiguous")
    if page_table.dtype not in (torch.int32, torch.int64):
        raise TypeError(f"page_table must be torch.int32 or torch.int64, got {page_table.dtype}")

    q_shape = _normalize_tensor_shape(q)
    k_cache_shape = _normalize_tensor_shape(k_cache)
    v_cache_shape = _normalize_tensor_shape(v_cache)
    page_table_shape = _normalize_tensor_shape(page_table)
    total_q, q_heads, head_dim = q_shape
    num_pages, page_size, kv_heads, k_head_dim = k_cache_shape
    num_pages_v, page_size_v, v_heads, v_head_dim = v_cache_shape
    if num_pages != num_pages_v or page_size != page_size_v:
        raise ValueError("k_cache and v_cache must have matching [num_pages, page_size]")
    if head_dim != k_head_dim or head_dim != v_head_dim:
        raise ValueError("paged attention currently requires matching Q/K/V head dimensions")
    if kv_heads != v_heads:
        raise ValueError("k_cache and v_cache must have the same number of KV heads")
    if q_heads % kv_heads != 0:
        raise ValueError(f"q_heads={q_heads} must be divisible by kv_heads={kv_heads}")

    batch, _max_pages_per_request = page_table_shape
    if tuple(cache_seqlens.shape) != (batch,):
        raise ValueError(
            f"cache_seqlens shape mismatch: expected {(batch,)}, got {tuple(cache_seqlens.shape)}"
        )
    if tuple(cu_seqlens_q.shape) != (batch + 1,):
        raise ValueError(
            f"cu_seqlens_q shape mismatch: expected {(batch + 1,)}, got {tuple(cu_seqlens_q.shape)}"
        )
    if q_shape[0] == 0:
        raise ValueError("paged attention requires total_q > 0")
    return q_shape, k_cache_shape, v_cache_shape, page_table_shape, q.device, q.dtype, page_size


def _validate_paged_lengths(
    *,
    total_q: int,
    page_size: int,
    max_pages_per_request: int,
    cache_seqlens: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    causal: bool,
) -> None:
    cache_seqlens_list = _metadata_to_cpu_int_list(cache_seqlens, name="cache_seqlens")
    cu_seqlens_q_list = _metadata_to_cpu_int_list(cu_seqlens_q, name="cu_seqlens_q")
    if cu_seqlens_q_list[0] != 0:
        raise ValueError(f"cu_seqlens_q must start at 0, got {cu_seqlens_q_list[0]}")
    if cu_seqlens_q_list[-1] != total_q:
        raise ValueError(f"cu_seqlens_q must end at total_q={total_q}, got {cu_seqlens_q_list[-1]}")

    q_lengths: list[int] = []
    for start, end in zip(cu_seqlens_q_list[:-1], cu_seqlens_q_list[1:]):
        if end < start:
            raise ValueError("cu_seqlens_q must be non-decreasing")
        q_lengths.append(end - start)

    for request_idx, (q_len, cache_len) in enumerate(zip(q_lengths, cache_seqlens_list)):
        if cache_len < 0:
            raise ValueError(f"cache_seqlens[{request_idx}] must be non-negative, got {cache_len}")
        if cache_len == 0:
            raise ValueError("b12x paged attention currently requires cache_seqlens > 0")
        if cache_len > max_pages_per_request * page_size:
            raise ValueError(
                f"cache_seqlens[{request_idx}]={cache_len} exceeds page_table capacity "
                f"{max_pages_per_request * page_size}"
            )
        if q_len == 1 and cache_len == 1:
            raise ValueError(
                "b12x paged attention does not currently support the single-token single-key corner "
                "(q_len=1, cache_len=1)"
            )
        if causal and q_len > cache_len:
            raise ValueError(
                f"causal paged attention requires q_len <= cache_len; got q_len={q_len}, "
                f"cache_len={cache_len} for request {request_idx}"
            )


def choose_paged_attention_num_splits(
    cache_seqlens: torch.Tensor,
    *,
    page_size: int,
    split_buckets: tuple[int, ...] = _DEFAULT_PAGED_SPLIT_BUCKETS,
    min_pages_per_split: int = _DEFAULT_MIN_PAGES_PER_SPLIT,
) -> int:
    """Choose a split bucket deterministically from paged KV lengths."""
    if min_pages_per_split < 1:
        raise ValueError(f"min_pages_per_split must be >= 1, got {min_pages_per_split}")
    buckets = _normalize_split_buckets(split_buckets)
    cache_seqlens_list = _metadata_to_cpu_int_list(cache_seqlens, name="cache_seqlens")
    max_pages = 0
    for cache_len in cache_seqlens_list:
        max_pages = max(max_pages, (cache_len + page_size - 1) // page_size)
    chosen = 1
    for bucket in buckets[1:]:
        if max_pages >= bucket * min_pages_per_split:
            chosen = bucket
    return chosen


def _validate_attention_inputs_against_plan(
    *,
    q_shape: tuple[int, ...],
    k_shape: tuple[int, ...],
    v_shape: tuple[int, ...],
    device: torch.device,
    dtype: torch.dtype,
    plan: AttentionPlan,
) -> None:
    expected = (
        plan.q_shape,
        plan.k_shape,
        plan.v_shape,
        plan.device,
        plan.dtype,
    )
    actual = (q_shape, k_shape, v_shape, device, dtype)
    if expected != actual:
        raise ValueError(
            "attention plan mismatch: "
            f"expected q/k/v/device/dtype={expected}, got {actual}"
        )


def _validate_paged_inputs_against_plan(
    *,
    q_shape: tuple[int, ...],
    k_cache_shape: tuple[int, ...],
    v_cache_shape: tuple[int, ...],
    page_table_shape: tuple[int, ...],
    cache_seqlens_shape: tuple[int, ...],
    cu_seqlens_q_shape: tuple[int, ...],
    device: torch.device,
    dtype: torch.dtype,
    plan: PagedAttentionPlan,
) -> None:
    expected = (
        plan.q_shape,
        plan.k_cache_shape,
        plan.v_cache_shape,
        plan.page_table_shape,
        plan.cache_seqlens_shape,
        plan.cu_seqlens_q_shape,
        plan.device,
        plan.dtype,
    )
    actual = (
        q_shape,
        k_cache_shape,
        v_cache_shape,
        page_table_shape,
        cache_seqlens_shape,
        cu_seqlens_q_shape,
        device,
        dtype,
    )
    if expected != actual:
        raise ValueError(
            "paged attention plan mismatch: "
            f"expected q/k_cache/v_cache/page_table/cache_seqlens/cu_seqlens_q/device/dtype={expected}, "
            f"got {actual}"
        )


@dataclass(frozen=True, kw_only=True)
class AttentionPlanKey:
    q_shape: tuple[int, ...]
    k_shape: tuple[int, ...]
    v_shape: tuple[int, ...]
    device_index: int
    dtype: torch.dtype
    causal: bool
    tile_m: int
    tile_n: int


@dataclass(frozen=True, kw_only=True)
class PagedAttentionPlanKey:
    q_shape: tuple[int, ...]
    k_cache_shape: tuple[int, ...]
    v_cache_shape: tuple[int, ...]
    page_table_shape: tuple[int, ...]
    cache_seqlens_shape: tuple[int, ...]
    cu_seqlens_q_shape: tuple[int, ...]
    device_index: int
    dtype: torch.dtype
    causal: bool
    tile_m: int
    tile_n: int
    num_splits: int


@dataclass(frozen=True, kw_only=True)
class AttentionPlan:
    """Exact-shape launch contract for one contiguous attention shape."""

    key: AttentionPlanKey
    compiled: object = field(repr=False, compare=False)
    cutlass_dtype: type[cutlass.Numeric] = field(repr=False, compare=False)

    def __getattr__(self, name: str):
        return getattr(self.key, name)

    @property
    def device(self) -> torch.device:
        return torch.device("cuda", self.key.device_index)


@dataclass(frozen=True, kw_only=True)
class PagedAttentionPlan:
    """Exact-shape launch contract for one paged attention shape."""

    key: PagedAttentionPlanKey
    compiled: object = field(repr=False, compare=False)
    compiled_combine: object | None = field(default=None, repr=False, compare=False)
    cutlass_dtype: type[cutlass.Numeric] = field(repr=False, compare=False)

    def __getattr__(self, name: str):
        return getattr(self.key, name)

    @property
    def device(self) -> torch.device:
        return torch.device("cuda", self.key.device_index)


@dataclass(kw_only=True)
class AttentionWorkspace:
    """Reusable exact-shape output buffers for one contiguous attention plan."""

    q_shape: tuple[int, ...]
    k_shape: tuple[int, ...]
    v_shape: tuple[int, ...]
    device: torch.device
    dtype: torch.dtype
    causal: bool
    tile_m: int
    tile_n: int
    output: torch.Tensor
    lse: torch.Tensor
    plan_key: AttentionPlanKey | None = None


@dataclass(kw_only=True)
class PagedAttentionWorkspace:
    """Reusable output buffers for one paged attention plan."""

    q_shape: tuple[int, ...]
    k_cache_shape: tuple[int, ...]
    v_cache_shape: tuple[int, ...]
    page_table_shape: tuple[int, ...]
    device: torch.device
    dtype: torch.dtype
    causal: bool
    tile_m: int
    tile_n: int
    num_splits: int
    output: torch.Tensor
    lse: torch.Tensor
    split_output: torch.Tensor | None = None
    split_lse: torch.Tensor | None = None
    plan_key: PagedAttentionPlanKey | None = None


class _AttentionForwardLaunch:
    def __init__(
        self,
        *,
        q_shape: tuple[int, ...],
        k_shape: tuple[int, ...],
        v_shape: tuple[int, ...],
        dtype: torch.dtype,
        causal: bool,
        tile_m: int,
        tile_n: int,
    ):
        self._q_shape = q_shape
        self._k_shape = k_shape
        self._v_shape = v_shape
        self._o_shape = q_shape
        self._lse_shape = _lse_shape(q_shape)
        self._q_stride = _contiguous_stride(q_shape)
        self._k_stride = _contiguous_stride(k_shape)
        self._v_stride = _contiguous_stride(v_shape)
        self._o_stride = _contiguous_stride(q_shape)
        self._lse_stride = _contiguous_stride(self._lse_shape)
        self._dtype = _torch_to_cutlass_dtype(dtype)
        _, _, q_heads, head_dim = _seq_dims(q_shape)
        _, _, kv_heads, head_dim_k = _seq_dims(k_shape)
        _, _, _, head_dim_v = _seq_dims(v_shape)
        qhead_per_kvhead = q_heads // kv_heads
        if not SM120ForwardKernel.can_implement(
            self._dtype,
            head_dim,
            head_dim_v,
            tile_m,
            tile_n,
            1,
            160,
            causal,
            False,
        ):
            raise TypeError(
                "b12x attention launch is unsupported with "
                f"dtype={dtype}, q_shape={q_shape}, k_shape={k_shape}, v_shape={v_shape}, "
                f"causal={causal}, tile=({tile_m}, {tile_n})"
            )
        self._kernel = SM120ForwardKernel(
            self._dtype,
            head_dim,
            head_dim_v=head_dim_v,
            qhead_per_kvhead=qhead_per_kvhead,
            is_causal=causal,
            pack_gqa=qhead_per_kvhead != 1,
            tile_m=tile_m,
            tile_n=tile_n,
        )
        assert head_dim == head_dim_k

    @cute.jit
    def __call__(
        self,
        q_ptr: cute.Pointer,
        k_ptr: cute.Pointer,
        v_ptr: cute.Pointer,
        o_ptr: cute.Pointer,
        lse_ptr: cute.Pointer,
        softmax_scale: float,
        current_stream: cuda.CUstream,
    ):
        q_tensor = cute.make_tensor(q_ptr, layout=cute.make_layout(self._q_shape, stride=self._q_stride))
        k_tensor = cute.make_tensor(k_ptr, layout=cute.make_layout(self._k_shape, stride=self._k_stride))
        v_tensor = cute.make_tensor(v_ptr, layout=cute.make_layout(self._v_shape, stride=self._v_stride))
        o_tensor = cute.make_tensor(o_ptr, layout=cute.make_layout(self._o_shape, stride=self._o_stride))
        lse_tensor = cute.make_tensor(
            lse_ptr,
            layout=cute.make_layout(self._lse_shape, stride=self._lse_stride),
        )
        self._kernel(
            q_tensor,
            k_tensor,
            v_tensor,
            o_tensor,
            lse_tensor,
            softmax_scale,
            stream=current_stream,
        )


class _PagedAttentionForwardLaunch:
    def __init__(
        self,
        *,
        q_shape: tuple[int, ...],
        k_cache_shape: tuple[int, ...],
        v_cache_shape: tuple[int, ...],
        page_table_shape: tuple[int, ...],
        cache_seqlens_shape: tuple[int, ...],
        cu_seqlens_q_shape: tuple[int, ...],
        dtype: torch.dtype,
        causal: bool,
        tile_m: int,
        tile_n: int,
        num_splits: int,
    ):
        self._q_shape = q_shape
        self._k_cache_shape = k_cache_shape
        self._v_cache_shape = v_cache_shape
        self._page_table_shape = page_table_shape
        self._cache_seqlens_shape = cache_seqlens_shape
        self._cu_seqlens_q_shape = cu_seqlens_q_shape
        self._num_splits = num_splits
        self._o_shape = _split_paged_output_shape(q_shape, num_splits=num_splits)
        self._lse_shape = _split_paged_lse_storage_shape(q_shape, num_splits=num_splits)
        self._q_stride = _contiguous_stride(q_shape)
        self._k_cache_stride = _contiguous_stride(k_cache_shape)
        self._v_cache_stride = _contiguous_stride(v_cache_shape)
        self._page_table_stride = _contiguous_stride(page_table_shape)
        self._cache_seqlens_stride = _contiguous_stride(cache_seqlens_shape)
        self._cu_seqlens_q_stride = _contiguous_stride(cu_seqlens_q_shape)
        self._o_stride = _contiguous_stride(self._o_shape)
        self._lse_stride = _contiguous_stride(self._lse_shape)
        self._dtype = _torch_to_cutlass_dtype(dtype)
        _, q_heads, head_dim = q_shape
        _, page_size, kv_heads, head_dim_k = k_cache_shape
        _, _, v_heads, head_dim_v = v_cache_shape
        qhead_per_kvhead = q_heads // kv_heads
        if page_size != tile_n:
            raise TypeError(
                f"b12x paged attention requires page_size == tile_n, got page_size={page_size}, tile_n={tile_n}"
            )
        if kv_heads != v_heads:
            raise TypeError("paged k_cache and v_cache must have matching KV head counts")
        if not SM120ForwardKernel.can_implement(
            self._dtype,
            head_dim,
            head_dim_v,
            tile_m,
            tile_n,
            1,
            160,
            causal,
            False,
        ):
            raise TypeError(
                "b12x paged attention launch is unsupported with "
                f"dtype={dtype}, q_shape={q_shape}, k_cache_shape={k_cache_shape}, "
                f"v_cache_shape={v_cache_shape}, causal={causal}, tile=({tile_m}, {tile_n})"
            )
        self._kernel = SM120ForwardKernel(
            self._dtype,
            head_dim,
            head_dim_v=head_dim_v,
            qhead_per_kvhead=qhead_per_kvhead,
            is_causal=causal,
            pack_gqa=qhead_per_kvhead != 1,
            tile_m=tile_m,
            tile_n=tile_n,
            num_splits=num_splits,
        )
        assert head_dim == head_dim_k

    @cute.jit
    def __call__(
        self,
        q_ptr: cute.Pointer,
        k_cache_ptr: cute.Pointer,
        v_cache_ptr: cute.Pointer,
        o_ptr: cute.Pointer,
        lse_ptr: cute.Pointer,
        cu_seqlens_q_ptr: cute.Pointer,
        cache_seqlens_ptr: cute.Pointer,
        page_table_ptr: cute.Pointer,
        softmax_scale: float,
        current_stream: cuda.CUstream,
    ):
        q_tensor = cute.make_tensor(q_ptr, layout=cute.make_layout(self._q_shape, stride=self._q_stride))
        k_cache_tensor = cute.make_tensor(
            k_cache_ptr,
            layout=cute.make_layout(self._k_cache_shape, stride=self._k_cache_stride),
        )
        v_cache_tensor = cute.make_tensor(
            v_cache_ptr,
            layout=cute.make_layout(self._v_cache_shape, stride=self._v_cache_stride),
        )
        o_tensor = cute.make_tensor(o_ptr, layout=cute.make_layout(self._o_shape, stride=self._o_stride))
        lse_tensor = cute.make_tensor(
            lse_ptr,
            layout=cute.make_layout(self._lse_shape, stride=self._lse_stride),
        )
        cu_seqlens_q_tensor = cute.make_tensor(
            cu_seqlens_q_ptr,
            layout=cute.make_layout(self._cu_seqlens_q_shape, stride=self._cu_seqlens_q_stride),
        )
        cache_seqlens_tensor = cute.make_tensor(
            cache_seqlens_ptr,
            layout=cute.make_layout(self._cache_seqlens_shape, stride=self._cache_seqlens_stride),
        )
        page_table_tensor = cute.make_tensor(
            page_table_ptr,
            layout=cute.make_layout(self._page_table_shape, stride=self._page_table_stride),
        )
        self._kernel(
            q_tensor,
            k_cache_tensor,
            v_cache_tensor,
            o_tensor,
            lse_tensor,
            softmax_scale,
            mCuSeqlensQ=cu_seqlens_q_tensor,
            mSeqUsedK=cache_seqlens_tensor,
            mPageTable=page_table_tensor,
            stream=current_stream,
        )


class _PagedAttentionCombineLaunch:
    def __init__(
        self,
        *,
        split_output_shape: tuple[int, ...],
        split_lse_shape: tuple[int, ...],
        output_shape: tuple[int, ...],
        lse_shape: tuple[int, ...],
        dtype: torch.dtype,
        num_splits: int,
        tile_k: int = 32,
    ):
        self._split_output_shape = split_output_shape
        self._split_lse_shape = split_lse_shape
        self._output_shape = output_shape
        self._lse_shape = lse_shape
        self._split_output_stride = _contiguous_stride(split_output_shape)
        self._split_lse_stride = _contiguous_stride(split_lse_shape)
        self._output_stride = _contiguous_stride(output_shape)
        self._lse_stride = _contiguous_stride(lse_shape)
        self._dtype = _torch_to_cutlass_dtype(dtype)
        _, _, _, head_dim = split_output_shape
        if not PagedAttentionCombineKernel.can_implement(
            self._dtype,
            self._dtype,
            head_dim=head_dim,
            num_splits=num_splits,
            tile_k=tile_k,
            num_threads=32,
        ):
            raise TypeError(
                "b12x paged attention combine launch is unsupported with "
                f"dtype={dtype}, split_output_shape={split_output_shape}, "
                f"split_lse_shape={split_lse_shape}, output_shape={output_shape}, "
                f"lse_shape={lse_shape}, num_splits={num_splits}, tile_k={tile_k}"
            )
        self._kernel = PagedAttentionCombineKernel(
            self._dtype,
            self._dtype,
            head_dim=head_dim,
            num_splits=num_splits,
            tile_k=tile_k,
        )

    @cute.jit
    def __call__(
        self,
        split_output_ptr: cute.Pointer,
        split_lse_ptr: cute.Pointer,
        output_ptr: cute.Pointer,
        lse_ptr: cute.Pointer,
        current_stream: cuda.CUstream,
    ):
        split_output_tensor = cute.make_tensor(
            split_output_ptr,
            layout=cute.make_layout(self._split_output_shape, stride=self._split_output_stride),
        )
        split_lse_tensor = cute.make_tensor(
            split_lse_ptr,
            layout=cute.make_layout(self._split_lse_shape, stride=self._split_lse_stride),
        )
        output_tensor = cute.make_tensor(
            output_ptr,
            layout=cute.make_layout(self._output_shape, stride=self._output_stride),
        )
        lse_tensor = cute.make_tensor(
            lse_ptr,
            layout=cute.make_layout(self._lse_shape, stride=self._lse_stride),
        )
        self._kernel(
            split_output_tensor,
            split_lse_tensor,
            output_tensor,
            lse_tensor,
            stream=current_stream,
        )


@functools.cache
def _compile_attention(
    q_shape: tuple[int, ...],
    k_shape: tuple[int, ...],
    v_shape: tuple[int, ...],
    dtype: torch.dtype,
    causal: bool,
    tile_m: int,
    tile_n: int,
):
    cutlass_dtype = _torch_to_cutlass_dtype(dtype)
    launch = _AttentionForwardLaunch(
        q_shape=q_shape,
        k_shape=k_shape,
        v_shape=v_shape,
        dtype=dtype,
        causal=causal,
        tile_m=tile_m,
        tile_n=tile_n,
    )
    return cute.compile(
        launch,
        make_ptr(cutlass_dtype, 16, cute.AddressSpace.gmem, assumed_align=16),
        make_ptr(cutlass_dtype, 16, cute.AddressSpace.gmem, assumed_align=16),
        make_ptr(cutlass_dtype, 16, cute.AddressSpace.gmem, assumed_align=16),
        make_ptr(cutlass_dtype, 16, cute.AddressSpace.gmem, assumed_align=16),
        make_ptr(cutlass.Float32, 16, cute.AddressSpace.gmem, assumed_align=4),
        1.0,
        current_cuda_stream(),
    )


@functools.cache
def _compile_paged_attention(
    q_shape: tuple[int, ...],
    k_cache_shape: tuple[int, ...],
    v_cache_shape: tuple[int, ...],
    page_table_shape: tuple[int, ...],
    cache_seqlens_shape: tuple[int, ...],
    cu_seqlens_q_shape: tuple[int, ...],
    dtype: torch.dtype,
    causal: bool,
    tile_m: int,
    tile_n: int,
    num_splits: int,
):
    cutlass_dtype = _torch_to_cutlass_dtype(dtype)
    launch = _PagedAttentionForwardLaunch(
        q_shape=q_shape,
        k_cache_shape=k_cache_shape,
        v_cache_shape=v_cache_shape,
        page_table_shape=page_table_shape,
        cache_seqlens_shape=cache_seqlens_shape,
        cu_seqlens_q_shape=cu_seqlens_q_shape,
        dtype=dtype,
        causal=causal,
        tile_m=tile_m,
        tile_n=tile_n,
        num_splits=num_splits,
    )
    return cute.compile(
        launch,
        make_ptr(cutlass_dtype, 16, cute.AddressSpace.gmem, assumed_align=16),
        make_ptr(cutlass_dtype, 16, cute.AddressSpace.gmem, assumed_align=16),
        make_ptr(cutlass_dtype, 16, cute.AddressSpace.gmem, assumed_align=16),
        make_ptr(cutlass_dtype, 16, cute.AddressSpace.gmem, assumed_align=16),
        make_ptr(cutlass.Float32, 16, cute.AddressSpace.gmem, assumed_align=4),
        make_ptr(cutlass.Int32, 16, cute.AddressSpace.gmem, assumed_align=4),
        make_ptr(cutlass.Int32, 16, cute.AddressSpace.gmem, assumed_align=4),
        make_ptr(cutlass.Int32, 16, cute.AddressSpace.gmem, assumed_align=4),
        1.0,
        current_cuda_stream(),
    )


@functools.cache
def _compile_paged_attention_combine(
    split_output_shape: tuple[int, ...],
    split_lse_shape: tuple[int, ...],
    output_shape: tuple[int, ...],
    lse_shape: tuple[int, ...],
    dtype: torch.dtype,
    num_splits: int,
):
    cutlass_dtype = _torch_to_cutlass_dtype(dtype)
    launch = _PagedAttentionCombineLaunch(
        split_output_shape=split_output_shape,
        split_lse_shape=split_lse_shape,
        output_shape=output_shape,
        lse_shape=lse_shape,
        dtype=dtype,
        num_splits=num_splits,
    )
    return cute.compile(
        launch,
        make_ptr(cutlass_dtype, 16, cute.AddressSpace.gmem, assumed_align=16),
        make_ptr(cutlass.Float32, 16, cute.AddressSpace.gmem, assumed_align=4),
        make_ptr(cutlass_dtype, 16, cute.AddressSpace.gmem, assumed_align=16),
        make_ptr(cutlass.Float32, 16, cute.AddressSpace.gmem, assumed_align=4),
        current_cuda_stream(),
    )


@functools.cache
def _get_attention_plan(
    q_shape: tuple[int, ...],
    k_shape: tuple[int, ...],
    v_shape: tuple[int, ...],
    device_index: int,
    dtype: torch.dtype,
    causal: bool,
    tile_m: int,
    tile_n: int,
) -> AttentionPlan:
    return AttentionPlan(
        key=AttentionPlanKey(
            q_shape=q_shape,
            k_shape=k_shape,
            v_shape=v_shape,
            device_index=device_index,
            dtype=dtype,
            causal=causal,
            tile_m=tile_m,
            tile_n=tile_n,
        ),
        compiled=_compile_attention(
            q_shape,
            k_shape,
            v_shape,
            dtype,
            causal,
            tile_m,
            tile_n,
        ),
        cutlass_dtype=_torch_to_cutlass_dtype(dtype),
    )


@functools.cache
def _get_paged_attention_plan(
    q_shape: tuple[int, ...],
    k_cache_shape: tuple[int, ...],
    v_cache_shape: tuple[int, ...],
    page_table_shape: tuple[int, ...],
    cache_seqlens_shape: tuple[int, ...],
    cu_seqlens_q_shape: tuple[int, ...],
    device_index: int,
    dtype: torch.dtype,
    causal: bool,
    tile_m: int,
    tile_n: int,
    num_splits: int,
) -> PagedAttentionPlan:
    return PagedAttentionPlan(
        key=PagedAttentionPlanKey(
            q_shape=q_shape,
            k_cache_shape=k_cache_shape,
            v_cache_shape=v_cache_shape,
            page_table_shape=page_table_shape,
            cache_seqlens_shape=cache_seqlens_shape,
            cu_seqlens_q_shape=cu_seqlens_q_shape,
            device_index=device_index,
            dtype=dtype,
            causal=causal,
            tile_m=tile_m,
            tile_n=tile_n,
            num_splits=num_splits,
        ),
        compiled=_compile_paged_attention(
            q_shape,
            k_cache_shape,
            v_cache_shape,
            page_table_shape,
            cache_seqlens_shape,
            cu_seqlens_q_shape,
            dtype,
            causal,
            tile_m,
            tile_n,
            num_splits,
        ),
        compiled_combine=(
            _compile_paged_attention_combine(
                _split_paged_output_shape(q_shape, num_splits=num_splits),
                _split_paged_lse_storage_shape(q_shape, num_splits=num_splits),
                q_shape,
                _paged_lse_storage_shape(q_shape),
                dtype,
                num_splits,
            )
            if num_splits > 1
            else None
        ),
        cutlass_dtype=_torch_to_cutlass_dtype(dtype),
    )


def clear_attention_caches() -> None:
    """Clear global compile caches owned by the b12x attention integration."""
    _compile_attention.cache_clear()
    _compile_paged_attention.cache_clear()
    _compile_paged_attention_combine.cache_clear()
    _get_attention_plan.cache_clear()
    _get_paged_attention_plan.cache_clear()


def _validate_workspace(
    workspace: AttentionWorkspace,
    *,
    plan: AttentionPlan,
) -> None:
    expected = (
        plan.q_shape,
        plan.k_shape,
        plan.v_shape,
        plan.device,
        plan.dtype,
        plan.causal,
        plan.tile_m,
        plan.tile_n,
    )
    actual = (
        workspace.q_shape,
        workspace.k_shape,
        workspace.v_shape,
        workspace.device,
        workspace.dtype,
        workspace.causal,
        workspace.tile_m,
        workspace.tile_n,
    )
    if expected != actual:
        raise ValueError(
            "workspace shape mismatch: "
            f"expected q/k/v/device/dtype/causal/tile={expected}, got {actual}"
        )
    if workspace.plan_key is not None and workspace.plan_key != plan.key:
        raise ValueError(
            "workspace plan mismatch: "
            f"expected {workspace.plan_key}, got {plan.key}"
        )


def _validate_paged_workspace(
    workspace: PagedAttentionWorkspace,
    *,
    plan: PagedAttentionPlan,
) -> None:
    expected = (
        plan.q_shape,
        plan.k_cache_shape,
        plan.v_cache_shape,
        plan.page_table_shape,
        plan.device,
        plan.dtype,
        plan.causal,
        plan.tile_m,
        plan.tile_n,
        plan.num_splits,
    )
    actual = (
        workspace.q_shape,
        workspace.k_cache_shape,
        workspace.v_cache_shape,
        workspace.page_table_shape,
        workspace.device,
        workspace.dtype,
        workspace.causal,
        workspace.tile_m,
        workspace.tile_n,
        workspace.num_splits,
    )
    if expected != actual:
        raise ValueError(
            "paged workspace shape mismatch: "
            f"expected q/k_cache/v_cache/page_table/device/dtype/causal/tile/splits={expected}, got {actual}"
        )
    if workspace.num_splits < 1:
        raise ValueError(f"paged workspace num_splits must be >= 1, got {workspace.num_splits}")
    if workspace.plan_key is not None and workspace.plan_key != plan.key:
        raise ValueError(
            "paged workspace plan mismatch: "
            f"expected {workspace.plan_key}, got {plan.key}"
        )
    if tuple(workspace.output.shape) != plan.q_shape:
        raise ValueError(
            f"paged workspace output shape mismatch: expected {plan.q_shape}, got {tuple(workspace.output.shape)}"
        )
    expected_lse_shape = _paged_lse_storage_shape(plan.q_shape)
    if tuple(workspace.lse.shape) != expected_lse_shape:
        raise ValueError(
            "paged workspace lse shape mismatch: "
            f"expected {expected_lse_shape}, got {tuple(workspace.lse.shape)}"
        )
    if workspace.num_splits == 1:
        if workspace.split_output is not None or workspace.split_lse is not None:
            raise ValueError("paged workspace with num_splits=1 must not carry split scratch buffers")
    else:
        expected_split_output_shape = _split_paged_output_shape(
            plan.q_shape,
            num_splits=workspace.num_splits,
        )
        expected_split_lse_shape = _split_paged_lse_storage_shape(
            plan.q_shape,
            num_splits=workspace.num_splits,
        )
        if workspace.split_output is None or workspace.split_lse is None:
            raise ValueError("paged workspace with num_splits>1 requires split scratch buffers")
        if tuple(workspace.split_output.shape) != expected_split_output_shape:
            raise ValueError(
                "paged workspace split_output shape mismatch: "
                f"expected {expected_split_output_shape}, got {tuple(workspace.split_output.shape)}"
            )
        if tuple(workspace.split_lse.shape) != expected_split_lse_shape:
            raise ValueError(
                "paged workspace split_lse shape mismatch: "
                f"expected {expected_split_lse_shape}, got {tuple(workspace.split_lse.shape)}"
            )


def allocate_attention_workspace_for_plan(plan: AttentionPlan) -> AttentionWorkspace:
    """Allocate reusable scratch for one exact contiguous attention plan."""
    output = torch.empty(plan.q_shape, dtype=plan.dtype, device=plan.device)
    lse = torch.empty(_lse_shape(plan.q_shape), dtype=torch.float32, device=plan.device)
    return AttentionWorkspace(
        q_shape=plan.q_shape,
        k_shape=plan.k_shape,
        v_shape=plan.v_shape,
        device=plan.device,
        dtype=plan.dtype,
        causal=plan.causal,
        tile_m=plan.tile_m,
        tile_n=plan.tile_n,
        output=output,
        lse=lse,
        plan_key=plan.key,
    )


def allocate_paged_attention_workspace_for_plan(plan: PagedAttentionPlan) -> PagedAttentionWorkspace:
    """Allocate reusable scratch for one exact paged attention plan."""
    output = torch.empty(plan.q_shape, dtype=plan.dtype, device=plan.device)
    lse = torch.empty(_paged_lse_storage_shape(plan.q_shape), dtype=torch.float32, device=plan.device)
    split_output = None
    split_lse = None
    if plan.num_splits > 1:
        split_output = torch.empty(
            _split_paged_output_shape(plan.q_shape, num_splits=plan.num_splits),
            dtype=plan.dtype,
            device=plan.device,
        )
        split_lse = torch.empty(
            _split_paged_lse_storage_shape(plan.q_shape, num_splits=plan.num_splits),
            dtype=torch.float32,
            device=plan.device,
        )
    return PagedAttentionWorkspace(
        q_shape=plan.q_shape,
        k_cache_shape=plan.k_cache_shape,
        v_cache_shape=plan.v_cache_shape,
        page_table_shape=plan.page_table_shape,
        device=plan.device,
        dtype=plan.dtype,
        causal=plan.causal,
        tile_m=plan.tile_m,
        tile_n=plan.tile_n,
        num_splits=plan.num_splits,
        output=output,
        lse=lse,
        split_output=split_output,
        split_lse=split_lse,
        plan_key=plan.key,
    )


def create_attention_plan(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    causal: bool = True,
    tile_shape: tuple[int, int] | None = None,
) -> AttentionPlan:
    """Create one exact contiguous attention launch plan."""
    q_shape, k_shape, v_shape, device, dtype = _validate_forward_inputs(q, k, v)
    _, _, _, head_dim = _seq_dims(q_shape)
    tile_m, tile_n = tile_shape or _select_tile_shape(head_dim, causal=causal)
    return _get_attention_plan(
        q_shape,
        k_shape,
        v_shape,
        _cuda_device_index(device),
        dtype,
        causal,
        tile_m,
        tile_n,
    )


def create_paged_attention_plan(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    page_table: torch.Tensor,
    cache_seqlens: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    *,
    causal: bool = True,
    tile_shape: tuple[int, int] | None = None,
    num_splits: int | None = None,
    split_buckets: tuple[int, ...] = _DEFAULT_PAGED_SPLIT_BUCKETS,
    min_pages_per_split: int = _DEFAULT_MIN_PAGES_PER_SPLIT,
) -> PagedAttentionPlan:
    """Create one exact paged attention launch plan."""
    buckets = _normalize_split_buckets(split_buckets)
    (
        q_shape,
        k_cache_shape,
        v_cache_shape,
        page_table_shape,
        device,
        dtype,
        page_size,
    ) = _inspect_paged_forward_inputs(
        q,
        k_cache,
        v_cache,
        page_table,
        cache_seqlens,
        cu_seqlens_q,
    )
    _validate_paged_lengths(
        total_q=q_shape[0],
        page_size=page_size,
        max_pages_per_request=page_table_shape[1],
        cache_seqlens=cache_seqlens,
        cu_seqlens_q=cu_seqlens_q,
        causal=causal,
    )
    if num_splits in (None, 0):
        num_splits = choose_paged_attention_num_splits(
            cache_seqlens,
            page_size=page_size,
            split_buckets=buckets,
            min_pages_per_split=min_pages_per_split,
        )
    elif num_splits not in buckets:
        raise ValueError(f"num_splits must be one of {buckets}, got {num_splits}")
    _, _, head_dim = q_shape
    tile_m, tile_n = tile_shape or _select_paged_tile_shape(
        head_dim,
        causal=causal,
        page_size=page_size,
    )
    return _get_paged_attention_plan(
        q_shape,
        k_cache_shape,
        v_cache_shape,
        page_table_shape,
        tuple(int(dim) for dim in cache_seqlens.shape),
        tuple(int(dim) for dim in cu_seqlens_q.shape),
        _cuda_device_index(device),
        dtype,
        causal,
        tile_m,
        tile_n,
        num_splits,
    )


def allocate_attention_workspace(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    causal: bool = True,
    tile_shape: tuple[int, int] | None = None,
) -> AttentionWorkspace:
    """Allocate one exact-shape workspace for `b12x_attention_forward`."""
    plan = create_attention_plan(
        q,
        k,
        v,
        causal=causal,
        tile_shape=tile_shape,
    )
    return allocate_attention_workspace_for_plan(plan)


def allocate_paged_attention_workspace(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    page_table: torch.Tensor,
    cache_seqlens: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    *,
    causal: bool = True,
    tile_shape: tuple[int, int] | None = None,
    num_splits: int | None = None,
    split_buckets: tuple[int, ...] = _DEFAULT_PAGED_SPLIT_BUCKETS,
    min_pages_per_split: int = _DEFAULT_MIN_PAGES_PER_SPLIT,
) -> PagedAttentionWorkspace:
    """Allocate one exact workspace for the page-size-64 SGLang paged path."""
    plan = create_paged_attention_plan(
        q,
        k_cache,
        v_cache,
        page_table,
        cache_seqlens,
        cu_seqlens_q,
        causal=causal,
        tile_shape=tile_shape,
        num_splits=num_splits,
        split_buckets=split_buckets,
        min_pages_per_split=min_pages_per_split,
    )
    return allocate_paged_attention_workspace_for_plan(plan)


def _combine_split_partials(
    workspace: PagedAttentionWorkspace,
    *,
    plan: PagedAttentionPlan,
) -> None:
    assert workspace.split_output is not None
    assert workspace.split_lse is not None
    assert plan.compiled_combine is not None
    plan.compiled_combine(
        make_ptr(
            plan.cutlass_dtype,
            workspace.split_output.data_ptr(),
            cute.AddressSpace.gmem,
            assumed_align=16,
        ),
        make_ptr(
            cutlass.Float32,
            workspace.split_lse.data_ptr(),
            cute.AddressSpace.gmem,
            assumed_align=4,
        ),
        make_ptr(
            plan.cutlass_dtype,
            workspace.output.data_ptr(),
            cute.AddressSpace.gmem,
            assumed_align=16,
        ),
        make_ptr(
            cutlass.Float32,
            workspace.lse.data_ptr(),
            cute.AddressSpace.gmem,
            assumed_align=4,
        ),
        current_cuda_stream(),
    )


def b12x_attention_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    workspace: AttentionWorkspace,
    plan: AttentionPlan | None = None,
    softmax_scale: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Execute contiguous self-attention using the transplanted SM120 kernel."""
    q_shape, k_shape, v_shape, device, dtype = _validate_forward_inputs(q, k, v)
    resolved_plan = plan or _get_attention_plan(
        q_shape,
        k_shape,
        v_shape,
        _cuda_device_index(workspace.device),
        workspace.dtype,
        workspace.causal,
        workspace.tile_m,
        workspace.tile_n,
    )
    _validate_attention_inputs_against_plan(
        q_shape=q_shape,
        k_shape=k_shape,
        v_shape=v_shape,
        device=device,
        dtype=dtype,
        plan=resolved_plan,
    )
    _validate_workspace(
        workspace,
        plan=resolved_plan,
    )
    _, seqlen_q, _, head_dim = _seq_dims(q_shape)
    _, seqlen_k, _, _ = _seq_dims(k_shape)
    if seqlen_q == 1 and seqlen_k == 1:
        raise ValueError(
            "b12x attention does not currently support the single-token single-key corner "
            "(seqlen_q=1, seqlen_k=1)"
    )
    if softmax_scale is None:
        softmax_scale = head_dim ** -0.5

    resolved_plan.compiled(
        make_ptr(resolved_plan.cutlass_dtype, q.data_ptr(), cute.AddressSpace.gmem, assumed_align=16),
        make_ptr(resolved_plan.cutlass_dtype, k.data_ptr(), cute.AddressSpace.gmem, assumed_align=16),
        make_ptr(resolved_plan.cutlass_dtype, v.data_ptr(), cute.AddressSpace.gmem, assumed_align=16),
        make_ptr(
            resolved_plan.cutlass_dtype,
            workspace.output.data_ptr(),
            cute.AddressSpace.gmem,
            assumed_align=16,
        ),
        make_ptr(cutlass.Float32, workspace.lse.data_ptr(), cute.AddressSpace.gmem, assumed_align=4),
        float(softmax_scale),
        current_cuda_stream(),
    )
    return workspace.output, workspace.lse


def b12x_paged_attention_forward(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    page_table: torch.Tensor,
    cache_seqlens: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    *,
    workspace: PagedAttentionWorkspace,
    plan: PagedAttentionPlan | None = None,
    softmax_scale: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Execute the page-size-64 SGLang paged path with true in-kernel paged loads."""
    (
        q_shape,
        k_cache_shape,
        v_cache_shape,
        page_table_shape,
        device,
        dtype,
        page_size,
    ) = _inspect_paged_forward_inputs(
        q,
        k_cache,
        v_cache,
        page_table,
        cache_seqlens,
        cu_seqlens_q,
    )
    resolved_plan = plan or _get_paged_attention_plan(
        q_shape,
        workspace.k_cache_shape,
        workspace.v_cache_shape,
        workspace.page_table_shape,
        tuple(int(dim) for dim in cache_seqlens.shape),
        tuple(int(dim) for dim in cu_seqlens_q.shape),
        _cuda_device_index(workspace.device),
        workspace.dtype,
        workspace.causal,
        workspace.tile_m,
        workspace.tile_n,
        workspace.num_splits,
    )
    _validate_paged_inputs_against_plan(
        q_shape=q_shape,
        k_cache_shape=k_cache_shape,
        v_cache_shape=v_cache_shape,
        page_table_shape=page_table_shape,
        cache_seqlens_shape=tuple(int(dim) for dim in cache_seqlens.shape),
        cu_seqlens_q_shape=tuple(int(dim) for dim in cu_seqlens_q.shape),
        device=device,
        dtype=dtype,
        plan=resolved_plan,
    )
    _validate_paged_workspace(
        workspace,
        plan=resolved_plan,
    )
    if workspace.tile_n != page_size:
        raise ValueError(
            f"paged workspace tile_n must match page_size, got tile_n={workspace.tile_n}, page_size={page_size}"
        )
    if softmax_scale is None:
        softmax_scale = q_shape[2] ** -0.5

    kernel_output = workspace.output if workspace.num_splits == 1 else workspace.split_output
    kernel_lse = workspace.lse if workspace.num_splits == 1 else workspace.split_lse
    assert kernel_output is not None
    assert kernel_lse is not None
    resolved_plan.compiled(
        make_ptr(resolved_plan.cutlass_dtype, q.data_ptr(), cute.AddressSpace.gmem, assumed_align=16),
        make_ptr(resolved_plan.cutlass_dtype, k_cache.data_ptr(), cute.AddressSpace.gmem, assumed_align=16),
        make_ptr(resolved_plan.cutlass_dtype, v_cache.data_ptr(), cute.AddressSpace.gmem, assumed_align=16),
        make_ptr(
            resolved_plan.cutlass_dtype,
            kernel_output.data_ptr(),
            cute.AddressSpace.gmem,
            assumed_align=16,
        ),
        make_ptr(cutlass.Float32, kernel_lse.data_ptr(), cute.AddressSpace.gmem, assumed_align=4),
        make_ptr(cutlass.Int32, cu_seqlens_q.data_ptr(), cute.AddressSpace.gmem, assumed_align=4),
        make_ptr(cutlass.Int32, cache_seqlens.data_ptr(), cute.AddressSpace.gmem, assumed_align=4),
        make_ptr(cutlass.Int32, page_table.data_ptr(), cute.AddressSpace.gmem, assumed_align=4),
        float(softmax_scale),
        current_cuda_stream(),
    )
    if workspace.num_splits > 1:
        _combine_split_partials(workspace, plan=resolved_plan)
    return workspace.output, workspace.lse.transpose(0, 1)


__all__ = [
    "AttentionPlan",
    "AttentionPlanKey",
    "AttentionWorkspace",
    "PagedAttentionPlan",
    "PagedAttentionPlanKey",
    "PagedAttentionWorkspace",
    "allocate_attention_workspace",
    "allocate_attention_workspace_for_plan",
    "allocate_paged_attention_workspace",
    "allocate_paged_attention_workspace_for_plan",
    "b12x_attention_forward",
    "b12x_paged_attention_forward",
    "choose_paged_attention_num_splits",
    "clear_attention_caches",
    "create_attention_plan",
    "create_paged_attention_plan",
]
