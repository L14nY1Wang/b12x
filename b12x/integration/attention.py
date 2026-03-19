"""Public attention entrypoints backed by the transplanted SM120 forward kernel."""

from __future__ import annotations

import functools
from dataclasses import dataclass

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch

from b12x.attention.forward import SM120ForwardKernel
from b12x.cute.utils import current_cuda_stream, make_ptr


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


@dataclass(kw_only=True)
class AttentionWorkspace:
    """Reusable exact-shape output buffers for `b12x_attention_forward`."""

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


@dataclass(kw_only=True)
class PagedAttentionWorkspace:
    """Reusable output buffers for one SGLang-style paged-attention launch shape."""

    q_shape: tuple[int, ...]
    k_cache_shape: tuple[int, ...]
    v_cache_shape: tuple[int, ...]
    page_table_shape: tuple[int, ...]
    device: torch.device
    dtype: torch.dtype
    causal: bool
    tile_m: int
    tile_n: int
    output: torch.Tensor
    lse: torch.Tensor


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
    ):
        self._q_shape = q_shape
        self._k_cache_shape = k_cache_shape
        self._v_cache_shape = v_cache_shape
        self._page_table_shape = page_table_shape
        self._cache_seqlens_shape = cache_seqlens_shape
        self._cu_seqlens_q_shape = cu_seqlens_q_shape
        self._o_shape = q_shape
        self._lse_shape = _paged_lse_storage_shape(q_shape)
        self._q_stride = _contiguous_stride(q_shape)
        self._k_cache_stride = _contiguous_stride(k_cache_shape)
        self._v_cache_stride = _contiguous_stride(v_cache_shape)
        self._page_table_stride = _contiguous_stride(page_table_shape)
        self._cache_seqlens_stride = _contiguous_stride(cache_seqlens_shape)
        self._cu_seqlens_q_stride = _contiguous_stride(cu_seqlens_q_shape)
        self._o_stride = _contiguous_stride(q_shape)
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


@functools.cache
def _get_compiled_attention(
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
def _get_compiled_paged_attention(
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


def clear_attention_caches() -> None:
    """Clear global compile caches owned by the b12x attention integration."""
    _get_compiled_attention.cache_clear()
    _get_compiled_paged_attention.cache_clear()


def _validate_workspace(
    workspace: AttentionWorkspace,
    *,
    q_shape: tuple[int, ...],
    k_shape: tuple[int, ...],
    v_shape: tuple[int, ...],
    device: torch.device,
    dtype: torch.dtype,
    causal: bool,
) -> None:
    expected = (
        workspace.q_shape,
        workspace.k_shape,
        workspace.v_shape,
        workspace.device,
        workspace.dtype,
        workspace.causal,
    )
    actual = (q_shape, k_shape, v_shape, device, dtype, causal)
    if expected != actual:
        raise ValueError(
            "workspace shape mismatch: "
            f"expected q/k/v/device/dtype/causal={expected}, got {actual}"
        )


def _validate_paged_workspace(
    workspace: PagedAttentionWorkspace,
    *,
    q_shape: tuple[int, ...],
    k_cache_shape: tuple[int, ...],
    v_cache_shape: tuple[int, ...],
    page_table_shape: tuple[int, ...],
    device: torch.device,
    dtype: torch.dtype,
    causal: bool,
) -> None:
    expected = (
        workspace.q_shape,
        workspace.k_cache_shape,
        workspace.v_cache_shape,
        workspace.page_table_shape,
        workspace.device,
        workspace.dtype,
        workspace.causal,
    )
    actual = (q_shape, k_cache_shape, v_cache_shape, page_table_shape, device, dtype, causal)
    if expected != actual:
        raise ValueError(
            "paged workspace shape mismatch: "
            f"expected q/k_cache/v_cache/page_table/device/dtype/causal={expected}, got {actual}"
        )


def _allocate_workspace(
    *,
    q_shape: tuple[int, ...],
    k_shape: tuple[int, ...],
    v_shape: tuple[int, ...],
    device: torch.device,
    dtype: torch.dtype,
    causal: bool,
    tile_m: int,
    tile_n: int,
) -> AttentionWorkspace:
    output = torch.empty(q_shape, dtype=dtype, device=device)
    lse = torch.empty(_lse_shape(q_shape), dtype=torch.float32, device=device)
    return AttentionWorkspace(
        q_shape=q_shape,
        k_shape=k_shape,
        v_shape=v_shape,
        device=device,
        dtype=dtype,
        causal=causal,
        tile_m=tile_m,
        tile_n=tile_n,
        output=output,
        lse=lse,
    )


def _allocate_paged_workspace(
    *,
    q_shape: tuple[int, ...],
    k_cache_shape: tuple[int, ...],
    v_cache_shape: tuple[int, ...],
    page_table_shape: tuple[int, ...],
    device: torch.device,
    dtype: torch.dtype,
    causal: bool,
    tile_m: int,
    tile_n: int,
) -> PagedAttentionWorkspace:
    output = torch.empty(q_shape, dtype=dtype, device=device)
    lse = torch.empty(_paged_lse_storage_shape(q_shape), dtype=torch.float32, device=device)
    return PagedAttentionWorkspace(
        q_shape=q_shape,
        k_cache_shape=k_cache_shape,
        v_cache_shape=v_cache_shape,
        page_table_shape=page_table_shape,
        device=device,
        dtype=dtype,
        causal=causal,
        tile_m=tile_m,
        tile_n=tile_n,
        output=output,
        lse=lse,
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
    q_shape, k_shape, v_shape, device, dtype = _validate_forward_inputs(q, k, v)
    _, _, _, head_dim = _seq_dims(q_shape)
    tile_m, tile_n = tile_shape or _select_tile_shape(head_dim, causal=causal)
    return _allocate_workspace(
        q_shape=q_shape,
        k_shape=k_shape,
        v_shape=v_shape,
        device=device,
        dtype=dtype,
        causal=causal,
        tile_m=tile_m,
        tile_n=tile_n,
    )


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
) -> PagedAttentionWorkspace:
    """Allocate one exact workspace for the page-size-64 SGLang paged path."""
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
    _, _, head_dim = q_shape
    tile_m, tile_n = tile_shape or _select_paged_tile_shape(
        head_dim,
        causal=causal,
        page_size=page_size,
    )
    return _allocate_paged_workspace(
        q_shape=q_shape,
        k_cache_shape=k_cache_shape,
        v_cache_shape=v_cache_shape,
        page_table_shape=page_table_shape,
        device=device,
        dtype=dtype,
        causal=causal,
        tile_m=tile_m,
        tile_n=tile_n,
    )


def b12x_attention_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    workspace: AttentionWorkspace,
    softmax_scale: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Execute contiguous self-attention using the transplanted SM120 kernel."""
    q_shape, k_shape, v_shape, device, dtype = _validate_forward_inputs(q, k, v)
    _validate_workspace(
        workspace,
        q_shape=q_shape,
        k_shape=k_shape,
        v_shape=v_shape,
        device=device,
        dtype=dtype,
        causal=workspace.causal,
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

    compiled = _get_compiled_attention(
        workspace.q_shape,
        workspace.k_shape,
        workspace.v_shape,
        workspace.dtype,
        workspace.causal,
        workspace.tile_m,
        workspace.tile_n,
    )
    cutlass_dtype = _torch_to_cutlass_dtype(dtype)
    compiled(
        make_ptr(cutlass_dtype, q.data_ptr(), cute.AddressSpace.gmem, assumed_align=16),
        make_ptr(cutlass_dtype, k.data_ptr(), cute.AddressSpace.gmem, assumed_align=16),
        make_ptr(cutlass_dtype, v.data_ptr(), cute.AddressSpace.gmem, assumed_align=16),
        make_ptr(cutlass_dtype, workspace.output.data_ptr(), cute.AddressSpace.gmem, assumed_align=16),
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
    _validate_paged_workspace(
        workspace,
        q_shape=q_shape,
        k_cache_shape=k_cache_shape,
        v_cache_shape=v_cache_shape,
        page_table_shape=page_table_shape,
        device=device,
        dtype=dtype,
        causal=workspace.causal,
    )
    if workspace.tile_n != page_size:
        raise ValueError(
            f"paged workspace tile_n must match page_size, got tile_n={workspace.tile_n}, page_size={page_size}"
        )
    if softmax_scale is None:
        softmax_scale = q_shape[2] ** -0.5

    compiled = _get_compiled_paged_attention(
        workspace.q_shape,
        workspace.k_cache_shape,
        workspace.v_cache_shape,
        workspace.page_table_shape,
        tuple(int(dim) for dim in cache_seqlens.shape),
        tuple(int(dim) for dim in cu_seqlens_q.shape),
        workspace.dtype,
        workspace.causal,
        workspace.tile_m,
        workspace.tile_n,
    )
    cutlass_dtype = _torch_to_cutlass_dtype(dtype)
    compiled(
        make_ptr(cutlass_dtype, q.data_ptr(), cute.AddressSpace.gmem, assumed_align=16),
        make_ptr(cutlass_dtype, k_cache.data_ptr(), cute.AddressSpace.gmem, assumed_align=16),
        make_ptr(cutlass_dtype, v_cache.data_ptr(), cute.AddressSpace.gmem, assumed_align=16),
        make_ptr(cutlass_dtype, workspace.output.data_ptr(), cute.AddressSpace.gmem, assumed_align=16),
        make_ptr(cutlass.Float32, workspace.lse.data_ptr(), cute.AddressSpace.gmem, assumed_align=4),
        make_ptr(cutlass.Int32, cu_seqlens_q.data_ptr(), cute.AddressSpace.gmem, assumed_align=4),
        make_ptr(cutlass.Int32, cache_seqlens.data_ptr(), cute.AddressSpace.gmem, assumed_align=4),
        make_ptr(cutlass.Int32, page_table.data_ptr(), cute.AddressSpace.gmem, assumed_align=4),
        float(softmax_scale),
        current_cuda_stream(),
    )
    return workspace.output, workspace.lse.transpose(0, 1)


__all__ = [
    "AttentionWorkspace",
    "PagedAttentionWorkspace",
    "allocate_attention_workspace",
    "allocate_paged_attention_workspace",
    "b12x_attention_forward",
    "b12x_paged_attention_forward",
    "clear_attention_caches",
]
