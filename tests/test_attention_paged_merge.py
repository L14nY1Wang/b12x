from __future__ import annotations

import torch

import cutlass

from b12x.attention.paged.merge import PagedPersistentMergeKernel
from b12x.cute.utils import current_cuda_stream

from .helpers import require_sm120


def _merge_reference_base2(
    partial_o: torch.Tensor,
    partial_lse: torch.Tensor,
    merge_indptr: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    total_rows = merge_indptr.numel() - 1
    num_heads = partial_o.shape[1]
    head_dim = partial_o.shape[2]
    out = torch.empty(total_rows, num_heads, head_dim, dtype=torch.float32, device=partial_o.device)
    lse = torch.empty(num_heads, total_rows, dtype=torch.float32, device=partial_o.device)

    for row_idx in range(total_rows):
        start_idx = int(merge_indptr[row_idx].item())
        end_idx = int(merge_indptr[row_idx + 1].item())
        if start_idx == end_idx:
            out[row_idx].zero_()
            lse[:, row_idx] = -torch.inf
            continue
        if end_idx == start_idx + 1:
            out[row_idx] = partial_o[start_idx].to(torch.float32)
            lse[:, row_idx] = partial_lse[start_idx]
            continue

        row_lse = partial_lse[start_idx:end_idx].to(torch.float32)
        row_out = partial_o[start_idx:end_idx].to(torch.float32)
        lse_max = row_lse.max(dim=0).values
        weights = torch.pow(2.0, row_lse - lse_max)
        weight_sum = weights.sum(dim=0)
        out[row_idx] = (row_out * weights[:, :, None]).sum(dim=0) / weight_sum[:, None]
        lse[:, row_idx] = torch.log2(weight_sum) + lse_max
    return out, lse


def _make_merge_problem(
    *,
    dtype: torch.dtype = torch.bfloat16,
    counts: list[int] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    device = "cuda"
    counts = [0, 1, 3, 1, 4] if counts is None else counts
    num_heads = 3
    head_dim = 256
    nnz = sum(counts)
    partial_o = (torch.randn(nnz, num_heads, head_dim, device=device) / 4).to(dtype)
    partial_lse = torch.randn(nnz, num_heads, dtype=torch.float32, device=device) * 2 - 1
    merge_indptr_list = [0]
    for count in counts:
        merge_indptr_list.append(merge_indptr_list[-1] + count)
    merge_indptr = torch.tensor(merge_indptr_list, dtype=torch.int32, device=device)
    output = torch.full((len(counts), num_heads, head_dim), -77.0, dtype=dtype, device=device)
    lse = torch.full((num_heads, len(counts)), -99.0, dtype=torch.float32, device=device)
    total_rows_ptr = torch.tensor([len(counts)], dtype=torch.int32, device=device)
    return partial_o, partial_lse, merge_indptr, output, lse, total_rows_ptr


@torch.inference_mode()
def test_paged_persistent_merge_matches_reference() -> None:
    require_sm120()
    partial_o, partial_lse, merge_indptr, output, lse, total_rows_ptr = _make_merge_problem()
    kernel = PagedPersistentMergeKernel(
        cutlass.BFloat16,
        cutlass.BFloat16,
        head_dim=256,
        persistent_ctas=2,
    )

    kernel(
        partial_o,
        partial_lse,
        merge_indptr,
        output,
        lse,
        total_rows_ptr,
        stream=current_cuda_stream(),
    )
    torch.cuda.synchronize()

    ref_out, ref_lse = _merge_reference_base2(partial_o, partial_lse, merge_indptr)
    assert torch.allclose(output.to(torch.float32), ref_out, atol=2e-2, rtol=2e-2)
    assert torch.allclose(lse, ref_lse, atol=2e-3, rtol=2e-3)


@torch.inference_mode()
def test_paged_persistent_merge_respects_dynamic_total_rows() -> None:
    require_sm120()
    partial_o, partial_lse, merge_indptr, output, lse, total_rows_ptr = _make_merge_problem()
    total_rows_ptr[0] = 3
    kernel = PagedPersistentMergeKernel(
        cutlass.BFloat16,
        cutlass.BFloat16,
        head_dim=256,
        persistent_ctas=2,
    )

    output_before = output.clone()
    lse_before = lse.clone()
    kernel(
        partial_o,
        partial_lse,
        merge_indptr,
        output,
        lse,
        total_rows_ptr,
        stream=current_cuda_stream(),
    )
    torch.cuda.synchronize()

    ref_out, ref_lse = _merge_reference_base2(partial_o, partial_lse, merge_indptr)
    assert torch.allclose(output[:3].to(torch.float32), ref_out[:3], atol=2e-2, rtol=2e-2)
    assert torch.allclose(lse[:, :3], ref_lse[:, :3], atol=2e-3, rtol=2e-3)
    assert torch.equal(output[3:], output_before[3:])
    assert torch.equal(lse[:, 3:], lse_before[:, 3:])


@torch.inference_mode()
def test_paged_persistent_merge_handles_more_than_one_partial_per_ty() -> None:
    require_sm120()
    partial_o, partial_lse, merge_indptr, output, lse, total_rows_ptr = _make_merge_problem(
        counts=[8, 7, 5]
    )
    kernel = PagedPersistentMergeKernel(
        cutlass.BFloat16,
        cutlass.BFloat16,
        head_dim=256,
        persistent_ctas=2,
    )

    kernel(
        partial_o,
        partial_lse,
        merge_indptr,
        output,
        lse,
        total_rows_ptr,
        stream=current_cuda_stream(),
    )
    torch.cuda.synchronize()

    ref_out, ref_lse = _merge_reference_base2(partial_o, partial_lse, merge_indptr)
    assert torch.allclose(output.to(torch.float32), ref_out, atol=2e-2, rtol=2e-2)
    assert torch.allclose(lse, ref_lse, atol=2e-3, rtol=2e-3)
