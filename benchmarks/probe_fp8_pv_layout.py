from __future__ import annotations

import argparse
import json
import pathlib
import sys
from dataclasses import dataclass

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass import Int32, const_expr
from cutlass.cute.runtime import from_dlpack

from b12x.attention import layout_utils
from b12x.attention import utils
from b12x.attention.forward import SM120ForwardKernel, copy_flattened


def _to_cute_tensor(x: torch.Tensor, dtype) -> cute.Tensor:
    tensor = from_dlpack(x, assumed_align=16)
    tensor.element_type = dtype
    return tensor


@dataclass(frozen=True)
class ProbeConfig:
    source_mode: str
    copy_tiling: str


class Fp8PvLayoutDumpKernel:
    tile_m = 48
    tile_n = 64
    head_dim = 256
    num_compute_warps = 3
    num_threads = num_compute_warps * 32
    max_dump_bytes = 256

    def __init__(self, *, source_mode: str, copy_tiling: str):
        if source_mode not in {"word_direct", "word_transpose"}:
            raise ValueError(f"unsupported source_mode={source_mode}")
        if copy_tiling not in {"A", "B"}:
            raise ValueError(f"unsupported copy_tiling={copy_tiling}")
        self.source_mode = source_mode
        self.copy_tiling = copy_tiling
        self.kernel_spec = SM120ForwardKernel(
            cutlass.BFloat16,
            self.head_dim,
            kv_dtype=cutlass.Float8E4M3FN,
            head_dim_v=self.head_dim,
            tile_m=self.tile_m,
            tile_n=self.tile_n,
            num_threads=(self.num_compute_warps + 1) * 32,
            num_compute_warps=self.num_compute_warps,
            Q_in_regs=False,
        )

    @cute.jit
    def __call__(
        self,
        mVRaw: cute.Tensor,
        mRefBytes: cute.Tensor,
        mCandBytes: cute.Tensor,
        mNumBytes: cute.Tensor,
        stream: cuda.CUstream,
    ):
        self.kernel_spec.num_threads = (self.kernel_spec.num_compute_warps + 1) * 32
        self.kernel_spec.num_mma_threads = self.kernel_spec.num_compute_warps * 32
        self.kernel_spec.num_producer_threads = 32
        self.kernel_spec.num_Q_load_threads = self.kernel_spec.num_mma_threads
        self.kernel_spec.num_epilogue_threads = self.kernel_spec.num_mma_threads
        self.kernel_spec._setup_attributes()
        _, tiled_mma_pv = self.kernel_spec._get_tiled_mma()
        self.kernel(
            mVRaw,
            mRefBytes,
            mCandBytes,
            mNumBytes,
            self.kernel_spec.sV_layout,
            self.kernel_spec.sV_raw_layout,
            tiled_mma_pv,
        ).launch(
            grid=(1, 1, 1),
            block=[self.num_threads, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        mVRaw: cute.Tensor,
        mRefBytes: cute.Tensor,
        mCandBytes: cute.Tensor,
        mNumBytes: cute.Tensor,
        sV_layout: cutlass.Constexpr,
        sV_raw_layout: cutlass.Constexpr,
        tiled_mma_pv: cutlass.Constexpr,
    ):
        tidx = cute.arch.thread_idx()[0]
        smem = cutlass.utils.SmemAllocator()
        sV = smem.allocate_tensor(
            element_type=cutlass.BFloat16,
            layout=sV_layout,
            byte_alignment=1024,
        )
        sVRaw = smem.allocate_tensor(
            element_type=cutlass.Float8E4M3FN,
            layout=sV_raw_layout,
            byte_alignment=1024,
        )
        total_elems = self.tile_n * self.head_dim
        for idx_iter in cutlass.range_constexpr(cute.ceil_div(total_elems, self.num_threads)):
            linear_idx = tidx + idx_iter * self.num_threads
            if linear_idx < total_elems:
                row = linear_idx // self.head_dim
                col = linear_idx - row * self.head_dim
                sVRaw[row, col, 0] = mVRaw[row, col]
        cute.arch.sync_threads()

        for idx in cutlass.range_constexpr(self.max_dump_bytes):
            mRefBytes[tidx, idx] = Int32(-1)
            mCandBytes[tidx, idx] = Int32(-1)
        mNumBytes[tidx] = Int32(0)

        thr_mma_pv = tiled_mma_pv.get_slice(tidx)
        sVt = layout_utils.transpose_view(sV)
        tOrVt = thr_mma_pv.make_fragment_B(thr_mma_pv.partition_B(sVt[None, None, 0]))

        sVtRaw = layout_utils.transpose_view(sVRaw)
        sVtRawU8 = cute.recast_tensor(sVtRaw, cutlass.Uint8)
        ref_copy_atom = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), cutlass.Uint8)
        ref_copy = utils.make_tiled_copy_B(ref_copy_atom, tiled_mma_pv).get_slice(tidx)
        tRefRaw = cute.make_fragment_like(cute.recast_tensor(tOrVt, cutlass.Uint8), cutlass.Uint8)
        tRefCopyView = ref_copy.retile(tRefRaw)
        tRefSrc = ref_copy.partition_S(sVtRawU8)
        copy_flattened(tRefSrc[None, None, None, 0], tRefCopyView[None, None, 0])

        sVRawU32 = cute.recast_tensor(sVRaw, cutlass.Uint32)
        if const_expr(self.source_mode == "word_transpose"):
            sCand = layout_utils.transpose_view(sVRawU32)
        else:
            sCand = sVRawU32
        cand_copy_atom = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), cutlass.Uint32)
        if const_expr(self.copy_tiling == "A"):
            cand_copy = utils.make_tiled_copy_A(cand_copy_atom, tiled_mma_pv).get_slice(tidx)
        else:
            cand_copy = utils.make_tiled_copy_B(cand_copy_atom, tiled_mma_pv).get_slice(tidx)
        tCandWords = cute.make_fragment_like(cute.recast_tensor(tOrVt, cutlass.Uint32), cutlass.Uint32)
        tCandCopyView = cand_copy.retile(tCandWords)
        tCandSrc = cand_copy.partition_S(sCand)
        copy_flattened(tCandSrc[None, None, None, 0], tCandCopyView[None, None, 0])

        ref_bytes = cute.flatten(tRefRaw[None, None, 0])
        cand_bytes = cute.flatten(
            cute.recast_tensor(cute.recast_tensor(tRefCopyView[None, None, 0], cutlass.Uint32), cutlass.Uint8)
        )
        num_bytes = cute.size(ref_bytes.shape)
        mNumBytes[tidx] = Int32(num_bytes)
        for idx in cutlass.range_constexpr(self.max_dump_bytes):
            if idx < num_bytes:
                mRefBytes[tidx, idx] = Int32(ref_bytes[idx])
                mCandBytes[tidx, idx] = Int32(cand_bytes[idx])


def run_probe(config: ProbeConfig) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    device = torch.device("cuda")
    linear_ids = torch.arange(64 * 256, device=device, dtype=torch.int32).view(64, 256)
    kernel = Fp8PvLayoutDumpKernel(
        source_mode=config.source_mode,
        copy_tiling=config.copy_tiling,
    )
    ref_planes = []
    cand_planes = []
    ref_bytes = torch.empty(
        kernel.num_threads,
        kernel.max_dump_bytes,
        device=device,
        dtype=torch.int32,
    )
    cand_bytes = torch.empty_like(ref_bytes)
    num_bytes = torch.empty(kernel.num_threads, device=device, dtype=torch.int32)
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    raw0 = ((linear_ids >> 0) & 0xFF).to(torch.uint8).contiguous()
    compiled = cute.compile(
        kernel,
        _to_cute_tensor(raw0.view(torch.float8_e4m3fn), cutlass.Float8E4M3FN),
        _to_cute_tensor(ref_bytes, cutlass.Int32),
        _to_cute_tensor(cand_bytes, cutlass.Int32),
        _to_cute_tensor(num_bytes, cutlass.Int32),
        stream,
    )
    for plane in range(4):
        raw_plane = ((linear_ids >> (8 * plane)) & 0xFF).to(torch.uint8).contiguous()
        compiled(
            _to_cute_tensor(raw_plane.view(torch.float8_e4m3fn), cutlass.Float8E4M3FN),
            _to_cute_tensor(ref_bytes, cutlass.Int32),
            _to_cute_tensor(cand_bytes, cutlass.Int32),
            _to_cute_tensor(num_bytes, cutlass.Int32),
            stream,
        )
        torch.cuda.synchronize()
        ref_planes.append(ref_bytes.cpu())
        cand_planes.append(cand_bytes.cpu())

    ref_ids = torch.zeros_like(ref_planes[0], dtype=torch.int64)
    cand_ids = torch.zeros_like(cand_planes[0], dtype=torch.int64)
    for plane in range(4):
        ref_ids |= (ref_planes[plane].to(torch.int64) & 0xFF) << (8 * plane)
        cand_ids |= (cand_planes[plane].to(torch.int64) & 0xFF) << (8 * plane)
    return ref_ids, cand_ids, num_bytes.cpu()


def _thread_report(
    *,
    thread_idx: int,
    ref_row: torch.Tensor,
    cand_row: torch.Tensor,
    num_bytes: int,
) -> dict[str, object]:
    ref = [int(v) for v in ref_row[:num_bytes].tolist()]
    cand = [int(v) for v in cand_row[:num_bytes].tolist()]
    cand_pos = {value: idx for idx, value in enumerate(cand)}
    cand_pos_for_ref = [cand_pos.get(value, -1) for value in ref]
    return {
        "thread": thread_idx,
        "num_bytes": num_bytes,
        "ref_ids": ref,
        "cand_ids": cand,
        "cand_pos_for_ref": cand_pos_for_ref,
    }


def _summarize_reports(reports: list[dict[str, object]]) -> dict[str, object]:
    active = [report for report in reports if int(report["num_bytes"]) > 0]
    perfect = [
        int(report["thread"])
        for report in active
        if report["cand_pos_for_ref"] == list(range(int(report["num_bytes"])))
    ]
    first_bad = None
    for report in active:
        identity = list(range(int(report["num_bytes"])))
        if report["cand_pos_for_ref"] != identity:
            first_bad = int(report["thread"])
            break
    return {
        "active_threads": len(active),
        "perfect_threads": perfect,
        "first_mismatched_thread": first_bad,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Dump FP8 PV raw-fragment byte IDs per thread.")
    parser.add_argument(
        "--source-mode",
        choices=["word_direct", "word_transpose", "all"],
        default="all",
    )
    parser.add_argument(
        "--copy-tiling",
        choices=["A", "B", "all"],
        default="all",
    )
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        default=pathlib.Path("probe_fp8_pv_layout.json"),
    )
    args = parser.parse_args()

    torch.cuda.init()
    configs: list[ProbeConfig] = []
    source_modes = ["word_direct", "word_transpose"] if args.source_mode == "all" else [args.source_mode]
    copy_tilings = ["A", "B"] if args.copy_tiling == "all" else [args.copy_tiling]
    for source_mode in source_modes:
        for copy_tiling in copy_tilings:
            configs.append(ProbeConfig(source_mode=source_mode, copy_tiling=copy_tiling))

    payload: dict[str, object] = {}
    for config in configs:
        ref_bytes, cand_bytes, num_bytes = run_probe(config)
        reports = [
            _thread_report(
                thread_idx=thread_idx,
                ref_row=ref_bytes[thread_idx],
                cand_row=cand_bytes[thread_idx],
                num_bytes=int(num_bytes[thread_idx].item()),
            )
            for thread_idx in range(ref_bytes.shape[0])
        ]
        key = f"source={config.source_mode},copy={config.copy_tiling}"
        payload[key] = {
            "summary": _summarize_reports(reports),
            "threads": reports,
        }
        summary = payload[key]["summary"]
        print(
            f"{key} "
            f"active_threads={summary['active_threads']} "
            f"first_mismatched_thread={summary['first_mismatched_thread']}"
        )

    args.output.write_text(json.dumps(payload, indent=2))
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
