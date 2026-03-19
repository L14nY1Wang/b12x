"""Public attention entrypoints backed by the transplanted SM120 forward kernel."""

from __future__ import annotations

import functools
from dataclasses import dataclass, field
from typing import Dict, Tuple

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


def _normalize_tensor_shape(t: torch.Tensor) -> tuple[int, ...]:
    return tuple(int(dim) for dim in t.shape)


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


@dataclass
class AttentionWorkspacePool:
    """Caller-owned exact-shape workspace cache partitioned by CUDA stream."""

    workspaces: Dict[Tuple, AttentionWorkspace] = field(default_factory=dict)

    def clear(self) -> None:
        self.workspaces.clear()


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


def clear_attention_caches() -> None:
    """Clear global compile caches owned by the b12x attention integration."""
    _get_compiled_attention.cache_clear()


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


def allocate_attention_workspace_pool() -> AttentionWorkspacePool:
    """Allocate an explicit caller-owned attention workspace pool."""
    return AttentionWorkspacePool()


def _workspace_pool_key(
    *,
    q_shape: tuple[int, ...],
    k_shape: tuple[int, ...],
    v_shape: tuple[int, ...],
    device: torch.device,
    dtype: torch.dtype,
    causal: bool,
    tile_m: int,
    tile_n: int,
) -> tuple:
    stream_id = int(torch.cuda.current_stream(device=device).cuda_stream)
    return (q_shape, k_shape, v_shape, device, dtype, causal, tile_m, tile_n, stream_id)


def _resolve_workspace(
    workspace: AttentionWorkspace | AttentionWorkspacePool,
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
    if isinstance(workspace, AttentionWorkspace):
        _validate_workspace(
            workspace,
            q_shape=q_shape,
            k_shape=k_shape,
            v_shape=v_shape,
            device=device,
            dtype=dtype,
            causal=causal,
        )
        return workspace
    if not isinstance(workspace, AttentionWorkspacePool):
        raise TypeError("workspace must be an AttentionWorkspace or AttentionWorkspacePool")

    key = _workspace_pool_key(
        q_shape=q_shape,
        k_shape=k_shape,
        v_shape=v_shape,
        device=device,
        dtype=dtype,
        causal=causal,
        tile_m=tile_m,
        tile_n=tile_n,
    )
    resolved = workspace.workspaces.get(key)
    if resolved is None:
        resolved = _allocate_workspace(
            q_shape=q_shape,
            k_shape=k_shape,
            v_shape=v_shape,
            device=device,
            dtype=dtype,
            causal=causal,
            tile_m=tile_m,
            tile_n=tile_n,
        )
        workspace.workspaces[key] = resolved
    return resolved


def b12x_attention_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    workspace: AttentionWorkspace | AttentionWorkspacePool,
    causal: bool = True,
    tile_shape: tuple[int, int] | None = None,
    softmax_scale: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Execute contiguous self-attention using the transplanted SM120 kernel.

    The current public slice is intentionally narrow:
    - contiguous rank-3 `[seqlen, heads, dim]` or rank-4 `[batch, seqlen, heads, dim]`
    - fp16/bf16 Q/K/V
    - exact-shape caller-owned workspace or workspace pool
    - output and LSE buffers are owned by the workspace
    """
    q_shape, k_shape, v_shape, device, dtype = _validate_forward_inputs(q, k, v)
    _, _, _, head_dim = _seq_dims(q_shape)
    if isinstance(workspace, AttentionWorkspace):
        effective_causal = workspace.causal
        tile_m, tile_n = workspace.tile_m, workspace.tile_n
    else:
        effective_causal = causal
        tile_m, tile_n = tile_shape or _select_tile_shape(head_dim, causal=effective_causal)
    resolved = _resolve_workspace(
        workspace,
        q_shape=q_shape,
        k_shape=k_shape,
        v_shape=v_shape,
        device=device,
        dtype=dtype,
        causal=effective_causal,
        tile_m=tile_m,
        tile_n=tile_n,
    )
    if softmax_scale is None:
        softmax_scale = head_dim ** -0.5

    compiled = _get_compiled_attention(
        resolved.q_shape,
        resolved.k_shape,
        resolved.v_shape,
        resolved.dtype,
        resolved.causal,
        resolved.tile_m,
        resolved.tile_n,
    )
    cutlass_dtype = _torch_to_cutlass_dtype(dtype)
    compiled(
        make_ptr(cutlass_dtype, q.data_ptr(), cute.AddressSpace.gmem, assumed_align=16),
        make_ptr(cutlass_dtype, k.data_ptr(), cute.AddressSpace.gmem, assumed_align=16),
        make_ptr(cutlass_dtype, v.data_ptr(), cute.AddressSpace.gmem, assumed_align=16),
        make_ptr(cutlass_dtype, resolved.output.data_ptr(), cute.AddressSpace.gmem, assumed_align=16),
        make_ptr(cutlass.Float32, resolved.lse.data_ptr(), cute.AddressSpace.gmem, assumed_align=4),
        float(softmax_scale),
        current_cuda_stream(),
    )
    return resolved.output, resolved.lse


__all__ = [
    "AttentionWorkspace",
    "AttentionWorkspacePool",
    "allocate_attention_workspace",
    "allocate_attention_workspace_pool",
    "b12x_attention_forward",
    "clear_attention_caches",
]
