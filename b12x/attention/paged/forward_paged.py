"""Standalone paged forward kernel for the primary paged backend.

This uses the exact host planner worklists and split scratch layout with the
literal tensor-core inner path that we actually ship:
- staged paged K/V ingress,
- literal QK/PV MMA for BF16 and FP8 KV,
- base-2 LSE storage compatible with the paged merge kernel.
"""

from __future__ import annotations
from typing import Type

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass._mlir.dialects import llvm

from cutlass import Float32, Int32, Uint32, const_expr
from cutlass.cutlass_dsl import Int64, T, dsl_user_op
from b12x.attention import utils as attention_utils
from b12x.cute.fp4 import get_ptr_as_int64, shared_ptr_to_u32
from b12x.cute.fp4 import (
    bf16_mma_m16n16k16_f32,
    bf16_rowsum_m16k16_f32,
    bfloat2_mul,
    broadcast_f32_to_bfloat2,
    cvt_bf16x2_to_e4m3x2,
    fp8x4_e4m3_to_bfloat2x2,
    ldmatrix_m8n8x4_b16,
    ldmatrix_m8n8x4_left_half_b16,
    ldmatrix_m8n8x4_right_half_b16,
    ldmatrix_m8n8x4_trans_b16,
    ldmatrix_m8n8x4_trans_left_half_b16,
    ldmatrix_m8n8x4_trans_right_half_b16,
    mxfp8_mma_m16n8k32_f32_e4m3,
    pack_f32x2_to_bfloat2,
    frag_layout_swizzle_16b_to_8b,
    frag_layout_swizzle_16b_to_8b_trans,
    st_global_v4_u32,
)

from .traits import PagedForwardTraits


@dsl_user_op
def _cp_async_load_128b_pred(
    smem_addr: Int32,
    gmem_addr: Int64,
    predicate: Int32,
    *,
    loc=None,
    ip=None,
):
    llvm.inline_asm(
        None,
        [
            Int32(predicate).ir_value(loc=loc, ip=ip),
            Int32(smem_addr).ir_value(loc=loc, ip=ip),
            Int64(gmem_addr).ir_value(loc=loc, ip=ip),
        ],
        "{\n"
        " .reg .pred p;\n"
        " setp.ne.b32 p, $0, 0;\n"
        " @p cp.async.cg.shared.global.L2::128B [$1], [$2], 16;\n"
        "}",
        "r,r,l",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


@dsl_user_op
def _cp_async_load_128b_zfill(
    smem_addr: Int32,
    gmem_addr: Int64,
    src_bytes: Int32,
    *,
    loc=None,
    ip=None,
):
    llvm.inline_asm(
        None,
        [
            Int32(smem_addr).ir_value(loc=loc, ip=ip),
            Int64(gmem_addr).ir_value(loc=loc, ip=ip),
            Int32(src_bytes).ir_value(loc=loc, ip=ip),
        ],
        "cp.async.cg.shared.global.L2::128B [$0], [$1], 16, $2;",
        "r,l,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


@dsl_user_op
def _exp2_approx_ftz_f32(a: Float32, *, loc=None, ip=None) -> Float32:
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [Float32(a).ir_value(loc=loc, ip=ip)],
            "ex2.approx.ftz.f32 $0, $1;",
            "=f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@dsl_user_op
def _exit_thread(
    *,
    loc=None,
    ip=None,
):
    llvm.inline_asm(
        None,
        [],
        "exit;",
        "",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


@cute.jit
def _permuted_offset_128b(row_idx, vec_idx, stride_128b):
    return row_idx * stride_128b + (vec_idx ^ (row_idx % 8))


@cute.jit
def _smem_addr_from_b128_offset(base_addr: Int32, offset_128b):
    return base_addr + Int32(offset_128b * 16)


@cute.jit
def _advance_offset_by_row_128b(offset_128b, step_size, row_stride_128b):
    return offset_128b + step_size * row_stride_128b


@cute.jit
def _advance_offset_by_column_128b_2(offset_128b, step_idx):
    xor_term = Int32(0x2) + (Int32(0x4) if const_expr(step_idx % 2 == 1) else Int32(0))
    extra = Int32(8) if const_expr(step_idx % 4 == 3) else Int32(0)
    return (offset_128b ^ xor_term) + extra


@cute.jit
def _literal_qk_mma_into_sfrag(
    s_frag: cute.Tensor,
    q_base_addr: Int32,
    k_base_addr: Int32,
    lane,
    warp_q_idx,
    warp_kv_idx,
    row_base,
    num_mma_q,
    num_mma_kv,
    num_mma_d_qk,
    upcast_stride_q,
    upcast_stride_k,
):
    for mma_d in cutlass.range_constexpr(num_mma_d_qk):
        a_regs = cute.make_rmem_tensor(
            cute.make_layout((num_mma_q, 4), stride=(4, 1)),
            Uint32,
        )
        for mma_q in cutlass.range_constexpr(num_mma_q):
            q_row = warp_q_idx * num_mma_q * 16 + mma_q * 16 + lane % 16
            q_col = mma_d * 2 + lane // 16
            q_offset = _permuted_offset_128b(q_row, q_col, upcast_stride_q)
            a0, a1, a2, a3 = ldmatrix_m8n8x4_b16(_smem_addr_from_b128_offset(q_base_addr, q_offset))
            a_regs[mma_q, 0] = a0
            a_regs[mma_q, 1] = a1
            a_regs[mma_q, 2] = a2
            a_regs[mma_q, 3] = a3

        for mma_kv in cutlass.range_constexpr(num_mma_kv):
            k_row = row_base + warp_kv_idx * num_mma_kv * 16 + mma_kv * 16 + 8 * (lane // 16) + lane % 8
            k_col = mma_d * 2 + (lane % 16) // 8
            k_offset = _permuted_offset_128b(k_row, k_col, upcast_stride_k)
            b0, b1, b2, b3 = ldmatrix_m8n8x4_b16(_smem_addr_from_b128_offset(k_base_addr, k_offset))

            for mma_q in cutlass.range_constexpr(num_mma_q):
                d0, d1, d2, d3, d4, d5, d6, d7 = bf16_mma_m16n16k16_f32(
                    s_frag[mma_q, mma_kv, 0],
                    s_frag[mma_q, mma_kv, 1],
                    s_frag[mma_q, mma_kv, 2],
                    s_frag[mma_q, mma_kv, 3],
                    s_frag[mma_q, mma_kv, 4],
                    s_frag[mma_q, mma_kv, 5],
                    s_frag[mma_q, mma_kv, 6],
                    s_frag[mma_q, mma_kv, 7],
                    a_regs[mma_q, 0],
                    a_regs[mma_q, 1],
                    a_regs[mma_q, 2],
                    a_regs[mma_q, 3],
                    b0,
                    b1,
                    b2,
                    b3,
                )
                s_frag[mma_q, mma_kv, 0] = d0
                s_frag[mma_q, mma_kv, 1] = d1
                s_frag[mma_q, mma_kv, 2] = d2
                s_frag[mma_q, mma_kv, 3] = d3
                s_frag[mma_q, mma_kv, 4] = d4
                s_frag[mma_q, mma_kv, 5] = d5
                s_frag[mma_q, mma_kv, 6] = d6
                s_frag[mma_q, mma_kv, 7] = d7


@cute.jit
def _literal_qk_mma_into_sfrag_fp8_raw(
    s_frag: cute.Tensor,
    q_base_addr: Int32,
    k_base_addr: Int32,
    lane,
    warp_q_idx,
    warp_kv_idx,
    row_base,
    num_mma_q,
    num_mma_kv,
    num_mma_d_qk,
    upcast_stride_q,
    upcast_stride_k,
):
    q_offset = _permuted_offset_128b(
        warp_q_idx * num_mma_q * 16 + lane % 16,
        lane // 16,
        upcast_stride_q,
    )
    k_offset = _permuted_offset_128b(
        row_base + warp_kv_idx * num_mma_kv * 16 + 8 * (lane // 16) + lane % 8,
        (lane % 16) // 8,
        upcast_stride_k,
    )
    for mma_d in cutlass.range_constexpr(num_mma_d_qk):
        a_regs = cute.make_rmem_tensor(
            cute.make_layout((num_mma_q, 4), stride=(4, 1)),
            Uint32,
        )
        q_offset_cur = q_offset
        for mma_q in cutlass.range_constexpr(num_mma_q):
            a0, a1, a2, a3 = ldmatrix_m8n8x4_b16(_smem_addr_from_b128_offset(q_base_addr, q_offset_cur))
            a_regs[mma_q, 0] = a0
            a_regs[mma_q, 1] = a1
            a_regs[mma_q, 2] = a2
            a_regs[mma_q, 3] = a3
            q_offset_cur = _advance_offset_by_row_128b(q_offset_cur, 16, upcast_stride_q)
        q_offset = _advance_offset_by_column_128b_2(q_offset_cur, mma_d) - Int32(num_mma_q * 16 * upcast_stride_q)

        k_offset_cur = k_offset
        for mma_kv in cutlass.range_constexpr(num_mma_kv):
            if const_expr(mma_d % 2 == 0):
                b_f8_0, b_f8_1 = ldmatrix_m8n8x4_left_half_b16(_smem_addr_from_b128_offset(k_base_addr, k_offset_cur))
            else:
                b_f8_0, b_f8_1 = ldmatrix_m8n8x4_right_half_b16(_smem_addr_from_b128_offset(k_base_addr, k_offset_cur))
            b_f8_0 = frag_layout_swizzle_16b_to_8b(b_f8_0)
            b_f8_1 = frag_layout_swizzle_16b_to_8b(b_f8_1)
            b0, b1 = fp8x4_e4m3_to_bfloat2x2(b_f8_0)
            b2, b3 = fp8x4_e4m3_to_bfloat2x2(b_f8_1)
            k_offset_cur = _advance_offset_by_row_128b(k_offset_cur, 16, upcast_stride_k)

            for mma_q in cutlass.range_constexpr(num_mma_q):
                d0, d1, d2, d3, d4, d5, d6, d7 = bf16_mma_m16n16k16_f32(
                    s_frag[mma_q, mma_kv, 0],
                    s_frag[mma_q, mma_kv, 1],
                    s_frag[mma_q, mma_kv, 2],
                    s_frag[mma_q, mma_kv, 3],
                    s_frag[mma_q, mma_kv, 4],
                    s_frag[mma_q, mma_kv, 5],
                    s_frag[mma_q, mma_kv, 6],
                    s_frag[mma_q, mma_kv, 7],
                    a_regs[mma_q, 0],
                    a_regs[mma_q, 1],
                    a_regs[mma_q, 2],
                    a_regs[mma_q, 3],
                    b0,
                    b1,
                    b2,
                    b3,
                )
                s_frag[mma_q, mma_kv, 0] = d0
                s_frag[mma_q, mma_kv, 1] = d1
                s_frag[mma_q, mma_kv, 2] = d2
                s_frag[mma_q, mma_kv, 3] = d3
                s_frag[mma_q, mma_kv, 4] = d4
                s_frag[mma_q, mma_kv, 5] = d5
                s_frag[mma_q, mma_kv, 6] = d6
                s_frag[mma_q, mma_kv, 7] = d7

        if const_expr(mma_d % 2 == 1):
            k_offset = _advance_offset_by_column_128b_2(k_offset_cur, mma_d // 2) - Int32(
                num_mma_kv * 16 * upcast_stride_k
            )
        else:
            k_offset = k_offset_cur - Int32(num_mma_kv * 16 * upcast_stride_k)


@cute.jit
def _literal_qk_mma_into_sfrag_mxfp8_raw(
    s_frag: cute.Tensor,
    q_base_addr: Int32,
    k_base_addr: Int32,
    lane,
    warp_q_idx,
    warp_kv_idx,
    row_base,
    num_mma_q,
    num_mma_kv,
    num_mma_d_qk,
    upcast_stride_q,
    upcast_stride_k,
):
    unit_scale = Uint32(0x7F7F7F7F)
    mask16 = Uint32(0xFFFF)
    shift16 = Uint32(16)
    q_offset = _permuted_offset_128b(
        warp_q_idx * num_mma_q * 16 + lane % 16,
        lane // 16,
        upcast_stride_q,
    )
    k_offset = _permuted_offset_128b(
        row_base + warp_kv_idx * num_mma_kv * 16 + 8 * (lane // 16) + lane % 8,
        (lane % 16) // 8,
        upcast_stride_k,
    )
    for mma_pair in cutlass.range_constexpr(num_mma_d_qk // 2):
        a_regs_k0 = cute.make_rmem_tensor(
            cute.make_layout((num_mma_q, 4), stride=(4, 1)),
            Uint32,
        )
        a_regs_k1 = cute.make_rmem_tensor(
            cute.make_layout((num_mma_q, 4), stride=(4, 1)),
            Uint32,
        )

        q_offset_cur = q_offset
        for mma_q in cutlass.range_constexpr(num_mma_q):
            a0, a1, a2, a3 = ldmatrix_m8n8x4_b16(_smem_addr_from_b128_offset(q_base_addr, q_offset_cur))
            a_regs_k0[mma_q, 0] = a0
            a_regs_k0[mma_q, 1] = a1
            a_regs_k0[mma_q, 2] = a2
            a_regs_k0[mma_q, 3] = a3
            q_offset_cur = _advance_offset_by_row_128b(q_offset_cur, 16, upcast_stride_q)

        mma_d0 = mma_pair * 2
        q_offset_mid = _advance_offset_by_column_128b_2(q_offset_cur, mma_d0) - Int32(
            num_mma_q * 16 * upcast_stride_q
        )
        q_offset_cur = q_offset_mid
        for mma_q in cutlass.range_constexpr(num_mma_q):
            a0, a1, a2, a3 = ldmatrix_m8n8x4_b16(_smem_addr_from_b128_offset(q_base_addr, q_offset_cur))
            a_regs_k1[mma_q, 0] = a0
            a_regs_k1[mma_q, 1] = a1
            a_regs_k1[mma_q, 2] = a2
            a_regs_k1[mma_q, 3] = a3
            q_offset_cur = _advance_offset_by_row_128b(q_offset_cur, 16, upcast_stride_q)
        q_offset = _advance_offset_by_column_128b_2(q_offset_cur, mma_d0 + 1) - Int32(
            num_mma_q * 16 * upcast_stride_q
        )

        k_offset_cur = k_offset
        for mma_kv in cutlass.range_constexpr(num_mma_kv):
            b0_k0, b1_k0 = ldmatrix_m8n8x4_left_half_b16(
                _smem_addr_from_b128_offset(k_base_addr, k_offset_cur)
            )
            b0_k1, b1_k1 = ldmatrix_m8n8x4_right_half_b16(
                _smem_addr_from_b128_offset(k_base_addr, k_offset_cur)
            )
            b0_k0 = frag_layout_swizzle_16b_to_8b(b0_k0)
            b1_k0 = frag_layout_swizzle_16b_to_8b(b1_k0)
            b0_k1 = frag_layout_swizzle_16b_to_8b(b0_k1)
            b1_k1 = frag_layout_swizzle_16b_to_8b(b1_k1)
            k_offset_cur = _advance_offset_by_row_128b(k_offset_cur, 16, upcast_stride_k)

            for mma_q in cutlass.range_constexpr(num_mma_q):
                qa0 = (cvt_bf16x2_to_e4m3x2(a_regs_k0[mma_q, 0]) & mask16) | (
                    (cvt_bf16x2_to_e4m3x2(a_regs_k1[mma_q, 0]) & mask16) << shift16
                )
                qa1 = (cvt_bf16x2_to_e4m3x2(a_regs_k0[mma_q, 1]) & mask16) | (
                    (cvt_bf16x2_to_e4m3x2(a_regs_k1[mma_q, 1]) & mask16) << shift16
                )
                qa2 = (cvt_bf16x2_to_e4m3x2(a_regs_k0[mma_q, 2]) & mask16) | (
                    (cvt_bf16x2_to_e4m3x2(a_regs_k1[mma_q, 2]) & mask16) << shift16
                )
                qa3 = (cvt_bf16x2_to_e4m3x2(a_regs_k0[mma_q, 3]) & mask16) | (
                    (cvt_bf16x2_to_e4m3x2(a_regs_k1[mma_q, 3]) & mask16) << shift16
                )

                d0, d1, d2, d3 = mxfp8_mma_m16n8k32_f32_e4m3(
                    s_frag[mma_q, mma_kv, 0],
                    s_frag[mma_q, mma_kv, 1],
                    s_frag[mma_q, mma_kv, 2],
                    s_frag[mma_q, mma_kv, 3],
                    qa0,
                    qa1,
                    qa2,
                    qa3,
                    b0_k0,
                    b0_k1,
                    unit_scale,
                    unit_scale,
                )
                d4, d5, d6, d7 = mxfp8_mma_m16n8k32_f32_e4m3(
                    s_frag[mma_q, mma_kv, 4],
                    s_frag[mma_q, mma_kv, 5],
                    s_frag[mma_q, mma_kv, 6],
                    s_frag[mma_q, mma_kv, 7],
                    qa0,
                    qa1,
                    qa2,
                    qa3,
                    b1_k0,
                    b1_k1,
                    unit_scale,
                    unit_scale,
                )
                s_frag[mma_q, mma_kv, 0] = d0
                s_frag[mma_q, mma_kv, 1] = d1
                s_frag[mma_q, mma_kv, 2] = d2
                s_frag[mma_q, mma_kv, 3] = d3
                s_frag[mma_q, mma_kv, 4] = d4
                s_frag[mma_q, mma_kv, 5] = d5
                s_frag[mma_q, mma_kv, 6] = d6
                s_frag[mma_q, mma_kv, 7] = d7

        k_offset = _advance_offset_by_column_128b_2(k_offset_cur, mma_pair) - Int32(
            num_mma_kv * 16 * upcast_stride_k
        )


@cute.jit
def _literal_pv_mma_into_ofrag_bf16_packed(
    o_frag: cute.Tensor,
    p_frag: cute.Tensor,
    v_base_addr: Int32,
    lane,
    warp_kv_idx,
    row_base,
    num_mma_q,
    num_mma_kv,
    num_mma_d_vo,
    upcast_stride_v,
    v_scale,
):
    v_scale_bf2 = broadcast_f32_to_bfloat2(v_scale)
    v_offset = _permuted_offset_128b(
        row_base + warp_kv_idx * num_mma_kv * 16 + lane % 16,
        lane // 16,
        upcast_stride_v,
    )
    for mma_kv in cutlass.range_constexpr(num_mma_kv):
        a_regs = cute.make_rmem_tensor(
            cute.make_layout((num_mma_q, 4), stride=(4, 1)),
            Uint32,
        )
        for mma_q in cutlass.range_constexpr(num_mma_q):
            a_regs[mma_q, 0] = bfloat2_mul(p_frag[mma_q, mma_kv, 0], v_scale_bf2)
            a_regs[mma_q, 1] = bfloat2_mul(p_frag[mma_q, mma_kv, 1], v_scale_bf2)
            a_regs[mma_q, 2] = bfloat2_mul(p_frag[mma_q, mma_kv, 2], v_scale_bf2)
            a_regs[mma_q, 3] = bfloat2_mul(p_frag[mma_q, mma_kv, 3], v_scale_bf2)

        v_offset_cur = v_offset
        for mma_d in cutlass.range_constexpr(num_mma_d_vo):
            b0, b1, b2, b3 = ldmatrix_m8n8x4_trans_b16(
                _smem_addr_from_b128_offset(v_base_addr, v_offset_cur)
            )
            for mma_q in cutlass.range_constexpr(num_mma_q):
                d0, d1, d2, d3, d4, d5, d6, d7 = bf16_mma_m16n16k16_f32(
                    o_frag[mma_q, mma_d, 0],
                    o_frag[mma_q, mma_d, 1],
                    o_frag[mma_q, mma_d, 2],
                    o_frag[mma_q, mma_d, 3],
                    o_frag[mma_q, mma_d, 4],
                    o_frag[mma_q, mma_d, 5],
                    o_frag[mma_q, mma_d, 6],
                    o_frag[mma_q, mma_d, 7],
                    a_regs[mma_q, 0],
                    a_regs[mma_q, 1],
                    a_regs[mma_q, 2],
                    a_regs[mma_q, 3],
                    b0,
                    b1,
                    b2,
                    b3,
                )
                o_frag[mma_q, mma_d, 0] = d0
                o_frag[mma_q, mma_d, 1] = d1
                o_frag[mma_q, mma_d, 2] = d2
                o_frag[mma_q, mma_d, 3] = d3
                o_frag[mma_q, mma_d, 4] = d4
                o_frag[mma_q, mma_d, 5] = d5
                o_frag[mma_q, mma_d, 6] = d6
                o_frag[mma_q, mma_d, 7] = d7
            v_offset_cur = _advance_offset_by_column_128b_2(v_offset_cur, mma_d)
        v_offset = _advance_offset_by_row_128b(v_offset_cur, 16, upcast_stride_v) - Int32(2 * num_mma_d_vo)
    v_offset -= Int32(16 * num_mma_kv * upcast_stride_v)


@cute.jit
def _literal_pv_mma_into_ofrag_fp8_raw(
    o_frag: cute.Tensor,
    p_frag: cute.Tensor,
    v_base_addr: Int32,
    lane,
    warp_kv_idx,
    row_base,
    num_mma_q,
    num_mma_kv,
    num_mma_d_vo,
    upcast_stride_v,
    v_scale,
):
    v_scale_bf2 = broadcast_f32_to_bfloat2(v_scale)
    v_offset = _permuted_offset_128b(
        row_base + warp_kv_idx * num_mma_kv * 16 + lane % 16,
        lane // 16,
        upcast_stride_v,
    )
    for mma_kv in cutlass.range_constexpr(num_mma_kv):
        a_regs = cute.make_rmem_tensor(
            cute.make_layout((num_mma_q, 4), stride=(4, 1)),
            Uint32,
        )
        for mma_q in cutlass.range_constexpr(num_mma_q):
            a_regs[mma_q, 0] = bfloat2_mul(p_frag[mma_q, mma_kv, 0], v_scale_bf2)
            a_regs[mma_q, 1] = bfloat2_mul(p_frag[mma_q, mma_kv, 1], v_scale_bf2)
            a_regs[mma_q, 2] = bfloat2_mul(p_frag[mma_q, mma_kv, 2], v_scale_bf2)
            a_regs[mma_q, 3] = bfloat2_mul(p_frag[mma_q, mma_kv, 3], v_scale_bf2)

        v_offset_cur = v_offset
        for mma_d in cutlass.range_constexpr(num_mma_d_vo):
            if const_expr(mma_d % 2 == 0):
                b_f8_0, b_f8_1 = ldmatrix_m8n8x4_trans_left_half_b16(
                    _smem_addr_from_b128_offset(v_base_addr, v_offset_cur)
                )
            else:
                b_f8_0, b_f8_1 = ldmatrix_m8n8x4_trans_right_half_b16(
                    _smem_addr_from_b128_offset(v_base_addr, v_offset_cur)
                )
            b_f8_0 = frag_layout_swizzle_16b_to_8b_trans(b_f8_0)
            b_f8_1 = frag_layout_swizzle_16b_to_8b_trans(b_f8_1)
            b0, b1 = fp8x4_e4m3_to_bfloat2x2(b_f8_0)
            b2, b3 = fp8x4_e4m3_to_bfloat2x2(b_f8_1)
            tmp = b1
            b1 = b2
            b2 = tmp
            for mma_q in cutlass.range_constexpr(num_mma_q):
                d0, d1, d2, d3, d4, d5, d6, d7 = bf16_mma_m16n16k16_f32(
                    o_frag[mma_q, mma_d, 0],
                    o_frag[mma_q, mma_d, 1],
                    o_frag[mma_q, mma_d, 2],
                    o_frag[mma_q, mma_d, 3],
                    o_frag[mma_q, mma_d, 4],
                    o_frag[mma_q, mma_d, 5],
                    o_frag[mma_q, mma_d, 6],
                    o_frag[mma_q, mma_d, 7],
                    a_regs[mma_q, 0],
                    a_regs[mma_q, 1],
                    a_regs[mma_q, 2],
                    a_regs[mma_q, 3],
                    b0,
                    b1,
                    b2,
                    b3,
                )
                o_frag[mma_q, mma_d, 0] = d0
                o_frag[mma_q, mma_d, 1] = d1
                o_frag[mma_q, mma_d, 2] = d2
                o_frag[mma_q, mma_d, 3] = d3
                o_frag[mma_q, mma_d, 4] = d4
                o_frag[mma_q, mma_d, 5] = d5
                o_frag[mma_q, mma_d, 6] = d6
                o_frag[mma_q, mma_d, 7] = d7
            if const_expr(mma_d % 2 == 1):
                v_offset_cur = _advance_offset_by_column_128b_2(v_offset_cur, mma_d // 2)
        v_offset = _advance_offset_by_row_128b(v_offset_cur, 16, upcast_stride_v) - Int32(num_mma_d_vo)
    v_offset -= Int32(16 * num_mma_kv * upcast_stride_v)


@cute.jit
def _literal_pv_mma_into_ofrag_mxfp8_raw(
    o_frag: cute.Tensor,
    p_frag: cute.Tensor,
    v_base_addr: Int32,
    lane,
    warp_kv_idx,
    row_base,
    num_mma_q,
    num_mma_kv,
    num_mma_d_vo,
    upcast_stride_v,
    v_scale,
):
    unit_scale = Uint32(0x7F7F7F7F)
    mask16 = Uint32(0xFFFF)
    shift16 = Uint32(16)
    v_scale_bf2 = broadcast_f32_to_bfloat2(v_scale)
    v_offset = _permuted_offset_128b(
        row_base + warp_kv_idx * num_mma_kv * 16 + lane % 16,
        lane // 16,
        upcast_stride_v,
    )
    for mma_pair in cutlass.range_constexpr(num_mma_kv // 2):
        a_regs = cute.make_rmem_tensor(
            cute.make_layout((num_mma_q, 4), stride=(4, 1)),
            Uint32,
        )
        mma_kv0 = mma_pair * 2
        mma_kv1 = mma_kv0 + 1
        for mma_q in cutlass.range_constexpr(num_mma_q):
            a_regs[mma_q, 0] = (cvt_bf16x2_to_e4m3x2(bfloat2_mul(p_frag[mma_q, mma_kv0, 0], v_scale_bf2)) & mask16) | (
                (cvt_bf16x2_to_e4m3x2(bfloat2_mul(p_frag[mma_q, mma_kv1, 0], v_scale_bf2)) & mask16) << shift16
            )
            a_regs[mma_q, 1] = (cvt_bf16x2_to_e4m3x2(bfloat2_mul(p_frag[mma_q, mma_kv0, 1], v_scale_bf2)) & mask16) | (
                (cvt_bf16x2_to_e4m3x2(bfloat2_mul(p_frag[mma_q, mma_kv1, 1], v_scale_bf2)) & mask16) << shift16
            )
            a_regs[mma_q, 2] = (cvt_bf16x2_to_e4m3x2(bfloat2_mul(p_frag[mma_q, mma_kv0, 2], v_scale_bf2)) & mask16) | (
                (cvt_bf16x2_to_e4m3x2(bfloat2_mul(p_frag[mma_q, mma_kv1, 2], v_scale_bf2)) & mask16) << shift16
            )
            a_regs[mma_q, 3] = (cvt_bf16x2_to_e4m3x2(bfloat2_mul(p_frag[mma_q, mma_kv0, 3], v_scale_bf2)) & mask16) | (
                (cvt_bf16x2_to_e4m3x2(bfloat2_mul(p_frag[mma_q, mma_kv1, 3], v_scale_bf2)) & mask16) << shift16
            )

        v_offset_k0 = v_offset
        v_offset_k1 = _advance_offset_by_row_128b(v_offset, 16, upcast_stride_v)
        for mma_d in cutlass.range_constexpr(num_mma_d_vo):
            if const_expr(mma_d % 2 == 0):
                b0_k0, b1_k0 = ldmatrix_m8n8x4_trans_left_half_b16(
                    _smem_addr_from_b128_offset(v_base_addr, v_offset_k0)
                )
                b0_k1, b1_k1 = ldmatrix_m8n8x4_trans_left_half_b16(
                    _smem_addr_from_b128_offset(v_base_addr, v_offset_k1)
                )
            else:
                b0_k0, b1_k0 = ldmatrix_m8n8x4_trans_right_half_b16(
                    _smem_addr_from_b128_offset(v_base_addr, v_offset_k0)
                )
                b0_k1, b1_k1 = ldmatrix_m8n8x4_trans_right_half_b16(
                    _smem_addr_from_b128_offset(v_base_addr, v_offset_k1)
                )
            b0_k0 = frag_layout_swizzle_16b_to_8b_trans(b0_k0)
            b1_k0 = frag_layout_swizzle_16b_to_8b_trans(b1_k0)
            b0_k1 = frag_layout_swizzle_16b_to_8b_trans(b0_k1)
            b1_k1 = frag_layout_swizzle_16b_to_8b_trans(b1_k1)

            for mma_q in cutlass.range_constexpr(num_mma_q):
                d0, d1, d2, d3 = mxfp8_mma_m16n8k32_f32_e4m3(
                    o_frag[mma_q, mma_d, 0],
                    o_frag[mma_q, mma_d, 1],
                    o_frag[mma_q, mma_d, 2],
                    o_frag[mma_q, mma_d, 3],
                    a_regs[mma_q, 0],
                    a_regs[mma_q, 1],
                    a_regs[mma_q, 2],
                    a_regs[mma_q, 3],
                    b0_k0,
                    b0_k1,
                    unit_scale,
                    unit_scale,
                )
                d4, d5, d6, d7 = mxfp8_mma_m16n8k32_f32_e4m3(
                    o_frag[mma_q, mma_d, 4],
                    o_frag[mma_q, mma_d, 5],
                    o_frag[mma_q, mma_d, 6],
                    o_frag[mma_q, mma_d, 7],
                    a_regs[mma_q, 0],
                    a_regs[mma_q, 1],
                    a_regs[mma_q, 2],
                    a_regs[mma_q, 3],
                    b1_k0,
                    b1_k1,
                    unit_scale,
                    unit_scale,
                )
                o_frag[mma_q, mma_d, 0] = d0
                o_frag[mma_q, mma_d, 1] = d1
                o_frag[mma_q, mma_d, 2] = d2
                o_frag[mma_q, mma_d, 3] = d3
                o_frag[mma_q, mma_d, 4] = d4
                o_frag[mma_q, mma_d, 5] = d5
                o_frag[mma_q, mma_d, 6] = d6
                o_frag[mma_q, mma_d, 7] = d7
            if const_expr(mma_d % 2 == 1):
                v_offset_k0 = _advance_offset_by_column_128b_2(v_offset_k0, mma_d // 2)
                v_offset_k1 = _advance_offset_by_column_128b_2(v_offset_k1, mma_d // 2)

        v_offset = _advance_offset_by_row_128b(v_offset, 32, upcast_stride_v)


@cute.jit
def _literal_update_mdo_states_fp32_pack_p(
    s_frag: cute.Tensor,
    o_frag: cute.Tensor,
    m_frag: cute.Tensor,
    d_frag: cute.Tensor,
    p_frag: cute.Tensor,
    sm_scale_log2: Float32,
    num_mma_q,
    num_mma_kv,
    num_mma_d_vo,
):
    for mma_q in cutlass.range_constexpr(num_mma_q):
        for row_slot in cutlass.range_constexpr(2):
            m_prev = Float32(m_frag[mma_q, row_slot])
            m_new = Float32(m_prev)
            for mma_kv in cutlass.range_constexpr(num_mma_kv):
                m_local = attention_utils.fmax(
                    attention_utils.fmax(
                        s_frag[mma_q, mma_kv, row_slot * 2 + 0],
                        s_frag[mma_q, mma_kv, row_slot * 2 + 1],
                    ),
                    attention_utils.fmax(
                        s_frag[mma_q, mma_kv, row_slot * 2 + 4],
                        s_frag[mma_q, mma_kv, row_slot * 2 + 5],
                    ),
                )
                m_new = attention_utils.fmax(m_new, m_local)
            m_new = attention_utils.fmax(m_new, cute.arch.shuffle_sync_bfly(m_new, offset=2))
            m_new = attention_utils.fmax(m_new, cute.arch.shuffle_sync_bfly(m_new, offset=1))

            scale_term = (
                Float32(1.0)
                if m_new == -Float32.inf
                else _exp2_approx_ftz_f32(m_prev * sm_scale_log2 - m_new * sm_scale_log2)
            )
            d_frag[mma_q, row_slot] = Float32(d_frag[mma_q, row_slot] * scale_term)
            for mma_d in cutlass.range_constexpr(num_mma_d_vo):
                o_frag[mma_q, mma_d, row_slot * 2 + 0] *= scale_term
                o_frag[mma_q, mma_d, row_slot * 2 + 1] *= scale_term
                o_frag[mma_q, mma_d, row_slot * 2 + 4] *= scale_term
                o_frag[mma_q, mma_d, row_slot * 2 + 5] *= scale_term

            m_scaled = Float32(m_new * sm_scale_log2)
            for mma_kv in cutlass.range_constexpr(num_mma_kv):
                p0 = (
                    Float32(0.0)
                    if m_new == -Float32.inf
                    else _exp2_approx_ftz_f32(
                        s_frag[mma_q, mma_kv, row_slot * 2 + 0] * sm_scale_log2 - m_scaled
                    )
                )
                p1 = (
                    Float32(0.0)
                    if m_new == -Float32.inf
                    else _exp2_approx_ftz_f32(
                        s_frag[mma_q, mma_kv, row_slot * 2 + 1] * sm_scale_log2 - m_scaled
                    )
                )
                p2 = (
                    Float32(0.0)
                    if m_new == -Float32.inf
                    else _exp2_approx_ftz_f32(
                        s_frag[mma_q, mma_kv, row_slot * 2 + 4] * sm_scale_log2 - m_scaled
                    )
                )
                p3 = (
                    Float32(0.0)
                    if m_new == -Float32.inf
                    else _exp2_approx_ftz_f32(
                        s_frag[mma_q, mma_kv, row_slot * 2 + 5] * sm_scale_log2 - m_scaled
                    )
                )
                p_frag[mma_q, mma_kv, row_slot + 0] = pack_f32x2_to_bfloat2(p0, p1)
                p_frag[mma_q, mma_kv, row_slot + 2] = pack_f32x2_to_bfloat2(p2, p3)

            m_frag[mma_q, row_slot] = Float32(m_new)


class PagedForwardKernel:
    def __init__(
        self,
        dtype_q: Type[cutlass.Numeric],
        dtype_kv: Type[cutlass.Numeric],
        dtype_kv_storage: Type[cutlass.Numeric],
        dtype_o: Type[cutlass.Numeric],
        *,
        traits: PagedForwardTraits,
        split_kv: bool,
        mxfp8_turbo: bool = False,
        enable_mxfp8_pv: bool = False,
    ):
        self.dtype_q = dtype_q
        self.dtype_kv = dtype_kv
        self.dtype_kv_storage = dtype_kv_storage
        self.dtype_o = dtype_o
        self.traits = traits
        self.split_kv = split_kv
        self.kv_is_fp8 = dtype_kv == cutlass.Float8E4M3FN
        self.vec_size = traits.head_dim_vo // 32
        self.total_warps = traits.num_warps_q * traits.num_warps_kv
        self.stage_tile_rows = traits.cta_tile_kv
        q_stage_bytes = traits.cta_tile_q * traits.head_dim_qk * (dtype_q.width // 8)
        kv_stage_bytes = self.stage_tile_rows * (
            traits.head_dim_qk + traits.head_dim_vo
        ) * (dtype_kv_storage.width // 8)
        self.num_stages = (
            1
            if traits.num_warps_kv > 1 or self.kv_is_fp8
            else (2 if q_stage_bytes + 2 * kv_stage_bytes <= traits.max_smem_per_threadblock else 1)
        )
        self.use_mxfp8_qk = (
            mxfp8_turbo
            and self.kv_is_fp8
            and dtype_q == cutlass.BFloat16
            and traits.head_dim_qk % 32 == 0
            and traits.num_mma_d_qk % 2 == 0
        )
        self.use_mxfp8_pv = (
            enable_mxfp8_pv
            and self.kv_is_fp8
            and dtype_q == cutlass.BFloat16
            and traits.num_warps_kv == 1
            and traits.num_mma_kv % 2 == 0
        )
        self.softmax_scale_log2 = Float32((traits.head_dim_qk ** -0.5) * attention_utils.LOG2_E)

    def _get_shared_storage_cls(self):
        class SharedStorage:
            pass

        if self.traits.num_warps_kv > 1:
            payload_struct = cute.struct.Align[
                cute.struct.MemRange[
                    cutlass.Uint8,
                    int(self.traits.shared_storage_bytes),
                ],
                128,
            ]
            SharedStorage.__annotations__ = {
                "payload": payload_struct,
            }
        else:
            q_struct = cute.struct.Align[
                cute.struct.MemRange[
                    self.dtype_q,
                    int(self.traits.cta_tile_q * self.traits.head_dim_qk),
                ],
                128,
            ]
            k_struct = cute.struct.Align[
                cute.struct.MemRange[
                    self.dtype_kv_storage,
                    int(self.num_stages * self.stage_tile_rows * self.traits.head_dim_qk),
                ],
                128,
            ]
            v_struct = cute.struct.Align[
                cute.struct.MemRange[
                    self.dtype_kv_storage,
                    int(self.num_stages * self.stage_tile_rows * self.traits.head_dim_vo),
                ],
                128,
            ]
            SharedStorage.__annotations__ = {
                "sQ": q_struct,
                "sK": k_struct,
                "sV": v_struct,
            }

        return cute.struct(SharedStorage)

    @cute.jit
    def _async_copy_paged_tile_permuted_128b(
        self,
        mCacheBytes: cute.Tensor,
        mPageTable: cute.Tensor,
        request_idx,
        tile_token_base,
        kv_head_idx,
        num_kv_heads,
        row_bytes,
        sStageBytes: cute.Tensor,
        stage_byte_offset,
        lane,
        warp_linear_idx,
        valid_rows,
        upcast_stride,
        fill_zero: cutlass.Constexpr,
    ):
        page_size = Int32(mPageTable.shape[1] * 0 + 64)
        lane_row = lane // 8
        lane_col = lane % 8
        for tile_iter in cutlass.range_constexpr(self.traits.num_mma_kv * 4 // self.traits.num_warps_q):
            row_idx = Int32(warp_linear_idx * 4 + lane_row + tile_iter * self.total_warps * 4)
            token_idx = Int32(tile_token_base + row_idx)
            page_iter = token_idx // page_size
            entry_idx = token_idx - page_iter * page_size
            page_id = mPageTable[request_idx, page_iter]
            row_valid = row_idx < valid_rows
            row_byte_base = (((page_id * page_size + entry_idx) * num_kv_heads) + kv_head_idx) * row_bytes
            for vec_iter in cutlass.range_constexpr(row_bytes // 128):
                vec_idx = Int32(lane_col + vec_iter * 8)
                src_byte_idx = row_byte_base + vec_idx * 16
                dst_byte_idx = stage_byte_offset + _permuted_offset_128b(row_idx, vec_idx, upcast_stride) * 16
                if const_expr(fill_zero):
                    _cp_async_load_128b_zfill(
                        shared_ptr_to_u32(sStageBytes.iterator + dst_byte_idx),
                        get_ptr_as_int64(mCacheBytes, src_byte_idx),
                        cutlass.select_(row_valid, Int32(16), Int32(0)),
                    )
                else:
                    _cp_async_load_128b_pred(
                        shared_ptr_to_u32(sStageBytes.iterator + dst_byte_idx),
                        get_ptr_as_int64(mCacheBytes, src_byte_idx),
                        Int32(row_valid),
                    )

    @cute.jit
    def _async_copy_q_tile_permuted_128b(
        self,
        mQBytes: cute.Tensor,
        q_start,
        packed_tile_start,
        packed_tile_rows,
        kv_head_idx,
        group_size,
        num_q_heads,
        row_bytes,
        sQBytes: cute.Tensor,
        lane,
        warp_q_idx,
    ):
        lane_row = lane // 8
        lane_col = lane % 8
        warp_row_base = Int32(warp_q_idx * self.traits.num_mma_q * 16)
        for mma_q in cutlass.range_constexpr(self.traits.num_mma_q):
            for row_iter in cutlass.range_constexpr(4):
                packed_q_idx = Int32(packed_tile_start + warp_row_base + mma_q * 16 + lane_row + row_iter * 4)
                row_valid = packed_q_idx < (packed_tile_start + packed_tile_rows)
                q_row_local = packed_q_idx // group_size
                q_group_lane = packed_q_idx - q_row_local * group_size
                q_head_idx = Int32(kv_head_idx * group_size + q_group_lane)
                q_row_idx = Int32(q_start + q_row_local)
                row_byte_base = ((q_row_idx * num_q_heads) + q_head_idx) * row_bytes
                row_idx = Int32(warp_row_base + mma_q * 16 + lane_row + row_iter * 4)
                for mma_do in cutlass.range_constexpr(self.traits.num_mma_d_qk // 4):
                    vec_idx = Int32(lane_col + mma_do * 8)
                    src_byte_idx = row_byte_base + vec_idx * 16
                    dst_byte_idx = _permuted_offset_128b(row_idx, vec_idx, self.traits.upcast_stride_q) * 16
                    _cp_async_load_128b_pred(
                        shared_ptr_to_u32(sQBytes.iterator + dst_byte_idx),
                        get_ptr_as_int64(mQBytes, src_byte_idx),
                        Int32(row_valid),
                    )

    @staticmethod
    def can_implement(
        dtype_q: Type[cutlass.Numeric],
        dtype_kv: Type[cutlass.Numeric],
        dtype_kv_storage: Type[cutlass.Numeric],
        dtype_o: Type[cutlass.Numeric],
        *,
        traits: PagedForwardTraits,
        split_kv: bool,
    ) -> bool:
        del split_kv
        if dtype_q not in (cutlass.Float16, cutlass.BFloat16):
            return False
        if dtype_kv not in (cutlass.Float16, cutlass.BFloat16, cutlass.Float8E4M3FN):
            return False
        if dtype_kv_storage not in (cutlass.Float16, cutlass.BFloat16, cutlass.Uint8):
            return False
        if dtype_o not in (cutlass.Float16, cutlass.BFloat16):
            return False
        if traits.head_dim_qk != traits.head_dim_vo:
            return False
        if traits.head_dim_qk % 32 != 0:
            return False
        if traits.num_threads != 128:
            return False
        if traits.cta_tile_q not in (16, 64, 128):
            return False
        return True

    @cute.jit
    def __call__(
        self,
        mQ: cute.Tensor,
        mKCache: cute.Tensor,
        mVCache: cute.Tensor,
        mPageTable: cute.Tensor,
        mCacheSeqlens: cute.Tensor,
        mCuSeqlensQ: cute.Tensor,
        mRequestIndices: cute.Tensor,
        mQoTileIndices: cute.Tensor,
        mKvTileIndices: cute.Tensor,
        mOIndptr: cute.Tensor,
        mKvChunkSizePtr: cute.Tensor,
        mBlockValidMask: cute.Tensor,
        mO: cute.Tensor,
        mLSE: cute.Tensor,
        mKDescale: cute.Tensor | None,
        mVDescale: cute.Tensor | None,
        stream: cuda.CUstream,
    ):
        if const_expr(len(mQ.shape) != 3):
            raise ValueError("mQ must have shape (total_q, q_heads, head_dim)")
        if const_expr(len(mKCache.shape) != 4 or len(mVCache.shape) != 4):
            raise ValueError("mKCache and mVCache must have shape (num_pages, page_size, kv_heads, head_dim)")
        if const_expr(len(mPageTable.shape) != 2):
            raise ValueError("mPageTable must have shape (batch, max_pages)")
        if const_expr(len(mCacheSeqlens.shape) != 1 or len(mCuSeqlensQ.shape) != 1):
            raise ValueError("mCacheSeqlens and mCuSeqlensQ must be rank-1")
        if const_expr(len(mRequestIndices.shape) != 1 or len(mQoTileIndices.shape) != 1 or len(mKvTileIndices.shape) != 1):
            raise ValueError("worklist tensors must be rank-1")
        if const_expr(len(mOIndptr.shape) != 1 or len(mKvChunkSizePtr.shape) != 1 or len(mBlockValidMask.shape) != 1):
            raise ValueError("mOIndptr, mKvChunkSizePtr, and mBlockValidMask must be rank-1")
        if const_expr(len(mO.shape) != 3 or len(mLSE.shape) != 2):
            raise ValueError("mO must be rank-3 and mLSE must be rank-2")
        if const_expr(mKDescale is not None and len(mKDescale.shape) not in (1, 2)):
            raise ValueError("mKDescale must have shape (batch,) or (batch, kv_heads)")
        if const_expr(mVDescale is not None and len(mVDescale.shape) not in (1, 2)):
            raise ValueError("mVDescale must have shape (batch,) or (batch, kv_heads)")
        if const_expr(mQ.element_type != self.dtype_q):
            raise TypeError("mQ dtype must match dtype_q")
        if const_expr(mKCache.element_type != self.dtype_kv_storage or mVCache.element_type != self.dtype_kv_storage):
            raise TypeError("mKCache/mVCache dtype must match dtype_kv_storage")
        if const_expr(mO.element_type != self.dtype_o):
            raise TypeError("mO dtype must match dtype_o")
        if const_expr(mLSE.element_type != Float32):
            raise TypeError("mLSE must be Float32")
        if const_expr(
            not self.can_implement(
                self.dtype_q,
                self.dtype_kv,
                self.dtype_kv_storage,
                self.dtype_o,
                traits=self.traits,
                split_kv=self.split_kv,
            )
        ):
            raise TypeError("paged forward kernel configuration is not supported")

        self.kernel(
            mQ,
            mKCache,
            mVCache,
            mPageTable,
            mCacheSeqlens,
            mCuSeqlensQ,
            mRequestIndices,
            mQoTileIndices,
            mKvTileIndices,
            mOIndptr,
            mKvChunkSizePtr,
            mBlockValidMask,
            mO,
            mLSE,
            mKDescale,
            mVDescale,
        ).launch(
            grid=(mBlockValidMask.shape[0], mKCache.shape[2], 1),
            block=[32, self.traits.num_warps_q, self.traits.num_warps_kv],
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        mQ: cute.Tensor,
        mKCache: cute.Tensor,
        mVCache: cute.Tensor,
        mPageTable: cute.Tensor,
        mCacheSeqlens: cute.Tensor,
        mCuSeqlensQ: cute.Tensor,
        mRequestIndices: cute.Tensor,
        mQoTileIndices: cute.Tensor,
        mKvTileIndices: cute.Tensor,
        mOIndptr: cute.Tensor,
        mKvChunkSizePtr: cute.Tensor,
        mBlockValidMask: cute.Tensor,
        mO: cute.Tensor,
        mLSE: cute.Tensor,
        mKDescale: cute.Tensor | None,
        mVDescale: cute.Tensor | None,
    ):
        lane, warp_q_idx, warp_kv_idx = cute.arch.thread_idx()
        work_idx, kv_head_idx, _ = cute.arch.block_idx()
        block_valid = mBlockValidMask[work_idx]
        if block_valid == Int32(0):
            _exit_thread()
        valid_work = True
        request_idx = mRequestIndices[work_idx]
        qo_tile_idx = mQoTileIndices[work_idx]
        kv_tile_idx = mKvTileIndices[work_idx]
        q_start = mCuSeqlensQ[request_idx]
        q_end = mCuSeqlensQ[request_idx + 1]
        qo_len = q_end - q_start
        cache_len = mCacheSeqlens[request_idx]
        group_size = mQ.shape[1] // mKCache.shape[2]
        packed_qo_len = qo_len * group_size
        packed_tile_start = qo_tile_idx * self.traits.cta_tile_q
        packed_tile_limit = packed_tile_start + self.traits.cta_tile_q
        packed_tile_end = cutlass.select_(packed_tile_limit < packed_qo_len, packed_tile_limit, packed_qo_len)
        kv_chunk_size = mKvChunkSizePtr[0]

        chunk_start = kv_tile_idx * kv_chunk_size if const_expr(self.split_kv) else 0
        chunk_end = (
            cutlass.select_(
                (kv_tile_idx + 1) * kv_chunk_size < cache_len,
                (kv_tile_idx + 1) * kv_chunk_size,
                cache_len,
            )
            if const_expr(self.split_kv)
            else cache_len
        )
        request_partial_start = mOIndptr[request_idx]
        request_partial_end = mOIndptr[request_idx + 1]
        num_chunks_kv = (
            (request_partial_end - request_partial_start) // qo_len
            if const_expr(self.split_kv)
            else 1
        )
        page_size = mKCache.shape[1]
        stage_tile_rows = self.stage_tile_rows
        q_bytes = self.traits.q_smem_bytes
        k_bytes = self.num_stages * stage_tile_rows * self.traits.head_dim_qk * (self.dtype_kv_storage.width // 8)
        v_bytes = self.num_stages * stage_tile_rows * self.traits.head_dim_vo * (self.dtype_kv_storage.width // 8)
        warp_linear_idx = warp_kv_idx * self.traits.num_warps_q + warp_q_idx
        tidx = lane + 32 * (warp_q_idx + self.traits.num_warps_q * warp_kv_idx)
        packed_tile_rows = packed_tile_end - packed_tile_start

        smem = cutlass.utils.SmemAllocator()
        SharedStorage = self._get_shared_storage_cls()
        storage = smem.allocate(SharedStorage)
        if const_expr(self.traits.num_warps_kv > 1):
            payload_u8 = storage.payload.get_tensor(
                cute.make_layout((self.traits.shared_storage_bytes,), stride=(1,))
            )
            sQ = cute.make_tensor(
                cute.recast_tensor(
                    cute.make_tensor(
                        payload_u8.iterator,
                        cute.make_layout((q_bytes,), stride=(1,)),
                    ),
                    self.dtype_q,
                ).iterator,
                cute.make_layout((self.traits.cta_tile_q, self.traits.head_dim_qk), stride=(self.traits.head_dim_qk, 1)),
            )
            sK = cute.make_tensor(
                cute.recast_tensor(
                    cute.make_tensor(
                        payload_u8.iterator + Int32(q_bytes),
                        cute.make_layout((k_bytes,), stride=(1,)),
                    ),
                    self.dtype_kv_storage,
                ).iterator,
                cute.make_layout(
                    (stage_tile_rows, self.traits.head_dim_qk, self.num_stages),
                    stride=(self.traits.head_dim_qk, 1, stage_tile_rows * self.traits.head_dim_qk),
                )
            )
            sKStageBytes = cute.make_tensor(
                payload_u8.iterator + Int32(q_bytes),
                cute.make_layout((k_bytes,), stride=(1,)),
            )
            sV = cute.make_tensor(
                cute.recast_tensor(
                    cute.make_tensor(
                        payload_u8.iterator + Int32(q_bytes + k_bytes),
                        cute.make_layout((v_bytes,), stride=(1,)),
                    ),
                    self.dtype_kv_storage,
                ).iterator,
                cute.make_layout(
                    (stage_tile_rows, self.traits.head_dim_vo, self.num_stages),
                    stride=(self.traits.head_dim_vo, 1, stage_tile_rows * self.traits.head_dim_vo),
                )
            )
            sVStageBytes = cute.make_tensor(
                payload_u8.iterator + Int32(q_bytes + k_bytes),
                cute.make_layout((v_bytes,), stride=(1,)),
            )
        else:
            sK = storage.sK.get_tensor(
                cute.make_layout(
                    (stage_tile_rows, self.traits.head_dim_qk, self.num_stages),
                    stride=(self.traits.head_dim_qk, 1, stage_tile_rows * self.traits.head_dim_qk),
                )
            )
            sV = storage.sV.get_tensor(
                cute.make_layout(
                    (stage_tile_rows, self.traits.head_dim_vo, self.num_stages),
                    stride=(self.traits.head_dim_vo, 1, stage_tile_rows * self.traits.head_dim_vo),
                )
            )
            sKStageBytes = cute.make_tensor(
                cute.recast_tensor(sK, cutlass.Uint8).iterator,
                cute.make_layout((k_bytes,), stride=(1,)),
            )
            sVStageBytes = cute.make_tensor(
                cute.recast_tensor(sV, cutlass.Uint8).iterator,
                cute.make_layout((v_bytes,), stride=(1,)),
            )
        if const_expr(self.traits.num_warps_kv > 1):
            sQ = cute.make_tensor(
                cute.recast_tensor(
                    cute.make_tensor(
                        payload_u8.iterator,
                        cute.make_layout((q_bytes,), stride=(1,)),
                    ),
                    self.dtype_q,
                ).iterator,
                cute.make_layout((self.traits.cta_tile_q * self.traits.head_dim_qk,), stride=(1,)),
            )
            sKTC = None
            sVTC = None
        else:
            sQ = storage.sQ.get_tensor(
                cute.make_layout((self.traits.cta_tile_q * self.traits.head_dim_qk,), stride=(1,))
            )
            sKTC = None
            sVTC = None
        k_row_bytes = self.traits.head_dim_qk * (self.dtype_kv_storage.width // 8)
        v_row_bytes = self.traits.head_dim_vo * (self.dtype_kv_storage.width // 8)
        k_stage_bytes = stage_tile_rows * k_row_bytes
        v_stage_bytes = stage_tile_rows * v_row_bytes
        mQBytes = cute.flatten(cute.recast_tensor(mQ, cutlass.Uint8))
        mKBytes = cute.flatten(cute.recast_tensor(mKCache, cutlass.Uint8))
        mVBytes = cute.flatten(cute.recast_tensor(mVCache, cutlass.Uint8))
        sKU8 = (
            sK
            if const_expr(self.kv_is_fp8 and self.dtype_kv_storage == cutlass.Uint8)
            else (cute.recast_tensor(sK, cutlass.Uint8) if const_expr(self.kv_is_fp8) else None)
        )
        sVU8 = (
            sV
            if const_expr(self.kv_is_fp8 and self.dtype_kv_storage == cutlass.Uint8)
            else (cute.recast_tensor(sV, cutlass.Uint8) if const_expr(self.kv_is_fp8) else None)
        )
        if const_expr(self.traits.num_warps_kv > 1):
            sync_payload = cute.recast_tensor(
                payload_u8,
                Float32,
            )
            sync_o_elems = self.traits.num_warps_kv * self.traits.cta_tile_q * self.traits.head_dim_vo
            sSyncO = cute.make_tensor(
                sync_payload.iterator,
                cute.make_layout(
                    (self.traits.num_warps_kv, self.traits.cta_tile_q, self.traits.head_dim_vo),
                    stride=(
                        self.traits.cta_tile_q * self.traits.head_dim_vo,
                        self.traits.head_dim_vo,
                        1,
                    ),
                ),
            )
            sSyncMD = cute.make_tensor(
                sync_payload.iterator + Int32(sync_o_elems),
                cute.make_layout(
                    (self.traits.num_warps_kv, self.traits.cta_tile_q, 2),
                    stride=(self.traits.cta_tile_q * 2, 2, 1),
                ),
            )
            sDecodeStage = cute.make_tensor(
                cute.recast_tensor(sync_payload, self.dtype_o).iterator,
                cute.make_layout(
                    (self.traits.num_warps_kv, self.traits.cta_tile_q, self.traits.head_dim_vo * 2),
                    stride=(
                        self.traits.cta_tile_q * self.traits.head_dim_vo * 2,
                        self.traits.head_dim_vo * 2,
                        1,
                    ),
                ),
            )
            sDecodeStageU32 = cute.recast_tensor(sDecodeStage, cutlass.Uint32)
        else:
            sync_payload = cute.make_tensor(
                cute.recast_tensor(cute.flatten(sQ), Float32).iterator,
                cute.make_layout((4,), stride=(1,)),
            )
            sSyncO = cute.make_tensor(
                sync_payload.iterator,
                cute.make_layout((1, 1, 1), stride=(1, 1, 1)),
            )
            sSyncMD = cute.make_tensor(
                sync_payload.iterator,
                cute.make_layout((1, 1, 2), stride=(2, 2, 1)),
            )
        decode_store_v128 = const_expr(
            self.traits.num_warps_kv > 1 and self.dtype_o == cutlass.BFloat16
        )
        split_store_v128 = const_expr(
            self.split_kv and self.traits.num_warps_kv == 1 and self.dtype_o == cutlass.BFloat16
        )
        final_store_v128 = const_expr(
            not self.split_kv and self.traits.num_warps_kv == 1 and self.dtype_o == cutlass.BFloat16
        )
        sOStage = cute.make_tensor(
            sQ.iterator,
            cute.make_layout(
                (self.traits.cta_tile_q, self.traits.head_dim_vo),
                stride=(self.traits.head_dim_vo, 1),
            ),
        )
        sOStageU32 = cute.recast_tensor(sOStage, cutlass.Uint32)
        mOFlat = cute.flatten(mO)

        tc_upcast_elems_qk = 16 // (self.dtype_q.width // 8)
        tc_upcast_stride_qk = self.traits.head_dim_qk // tc_upcast_elems_qk
        tc_upcast_elems_vo = 16 // (self.dtype_q.width // 8)
        tc_upcast_stride_vo = self.traits.head_dim_vo // tc_upcast_elems_vo
        if const_expr(self.traits.num_warps_kv > 1):
            sQBytes = cute.flatten(cute.recast_tensor(sQ, cutlass.Uint8))
            if warp_kv_idx == Int32(0):
                self._async_copy_q_tile_permuted_128b(
                    mQBytes,
                    q_start,
                    packed_tile_start,
                    packed_tile_rows,
                    kv_head_idx,
                    group_size,
                    mQ.shape[1],
                    self.traits.head_dim_qk * (self.dtype_q.width // 8),
                    sQBytes,
                    lane,
                    warp_q_idx,
                )
                cute.arch.cp_async_commit_group()
                cute.arch.cp_async_wait_group(0)
            cute.arch.sync_threads()
        else:
            sQBytes = cute.flatten(cute.recast_tensor(sQ, cutlass.Uint8))
            self._async_copy_q_tile_permuted_128b(
                mQBytes,
                q_start,
                packed_tile_start,
                packed_tile_rows,
                kv_head_idx,
                group_size,
                mQ.shape[1],
                self.traits.head_dim_qk * (self.dtype_q.width // 8),
                sQBytes,
                lane,
                warp_q_idx,
            )
            cute.arch.cp_async_commit_group()
            cute.arch.cp_async_wait_group(0)
            cute.arch.sync_threads()

        k_scale = (
            mKDescale[request_idx]
            if const_expr(mKDescale is not None and len(mKDescale.shape) == 1)
            else (
                mKDescale[request_idx, kv_head_idx]
                if const_expr(mKDescale is not None)
                else Float32(1.0)
            )
        )
        v_scale = (
            mVDescale[request_idx]
            if const_expr(mVDescale is not None and len(mVDescale.shape) == 1)
            else (
                mVDescale[request_idx, kv_head_idx]
                if const_expr(mVDescale is not None)
                else Float32(1.0)
            )
        )
        num_mma_q = self.traits.num_mma_q
        num_mma_kv = self.traits.num_mma_kv
        num_mma_d_vo = self.traits.num_mma_d_vo
        warp_row_base = warp_q_idx * num_mma_q * 16
        warp_kv_base = warp_kv_idx * num_mma_kv * 16
        lane_group = lane // 4
        lane_pair_base = 2 * (lane % 4)
        row_local_idx = cute.make_rmem_tensor(cute.make_layout((num_mma_q, 2), stride=(2, 1)), Int32)
        row_valid = cute.make_rmem_tensor(cute.make_layout((num_mma_q, 2), stride=(2, 1)), Int32)
        q_token_local = cute.make_rmem_tensor(cute.make_layout((num_mma_q, 2), stride=(2, 1)), Int32)
        q_head_idx_frag = cute.make_rmem_tensor(cute.make_layout((num_mma_q, 2), stride=(2, 1)), Int32)
        q_row_idx_frag = cute.make_rmem_tensor(cute.make_layout((num_mma_q, 2), stride=(2, 1)), Int32)
        causal_k_limit = cute.make_rmem_tensor(cute.make_layout((num_mma_q, 2), stride=(2, 1)), Int32)
        s_frag = cute.make_rmem_tensor(
            cute.make_layout((num_mma_q, num_mma_kv, 8), stride=(num_mma_kv * 8, 8, 1)),
            Float32,
        )
        o_frag = cute.make_rmem_tensor(
            cute.make_layout((num_mma_q, num_mma_d_vo, 8), stride=(num_mma_d_vo * 8, 8, 1)),
            Float32,
        )
        m_frag = cute.make_rmem_tensor(cute.make_layout((num_mma_q, 2), stride=(2, 1)), Float32)
        d_frag = cute.make_rmem_tensor(cute.make_layout((num_mma_q, 2), stride=(2, 1)), Float32)
        q_smem_base_addr = shared_ptr_to_u32(sQ.iterator)

        for mma_q in cutlass.range_constexpr(num_mma_q):
            for row_slot in cutlass.range_constexpr(2):
                packed_row_local = warp_row_base + mma_q * 16 + lane_group + 8 * row_slot
                row_local_idx[mma_q, row_slot] = Int32(packed_row_local)
                valid_row = packed_row_local < packed_tile_rows
                row_valid[mma_q, row_slot] = Int32(valid_row)
                if valid_row:
                    packed_q_idx = packed_tile_start + packed_row_local
                    token_local = packed_q_idx // group_size
                    q_group_lane = packed_q_idx - token_local * group_size
                    q_token_local[mma_q, row_slot] = Int32(token_local)
                    q_head_idx_frag[mma_q, row_slot] = Int32(kv_head_idx * group_size + q_group_lane)
                    q_row_idx_frag[mma_q, row_slot] = Int32(q_start + token_local)
                    causal_k_limit[mma_q, row_slot] = Int32(token_local + cache_len - qo_len)
                else:
                    q_token_local[mma_q, row_slot] = Int32(0)
                    q_head_idx_frag[mma_q, row_slot] = Int32(0)
                    q_row_idx_frag[mma_q, row_slot] = Int32(0)
                    causal_k_limit[mma_q, row_slot] = Int32(-1)

        for mma_q in cutlass.range_constexpr(num_mma_q):
            for mma_d in cutlass.range_constexpr(num_mma_d_vo):
                for reg_id in cutlass.range_constexpr(8):
                    o_frag[mma_q, mma_d, reg_id] = Float32(0.0)
            for row_slot in cutlass.range_constexpr(2):
                m_frag[mma_q, row_slot] = Float32(-Float32.inf)
                d_frag[mma_q, row_slot] = Float32(1.0)

        prefetch_base = chunk_start
        preload_count = 0
        preload_stage_idx = Int32(0)
        while preload_count < self.num_stages and prefetch_base < chunk_end:
            tile_limit = cutlass.select_(
                prefetch_base + stage_tile_rows < chunk_end,
                prefetch_base + stage_tile_rows,
                chunk_end,
            )
            tile_tokens = tile_limit - prefetch_base
            self._async_copy_paged_tile_permuted_128b(
                mKBytes,
                mPageTable,
                request_idx,
                prefetch_base,
                kv_head_idx,
                mKCache.shape[2],
                k_row_bytes,
                sKStageBytes,
                Int32(preload_stage_idx * k_stage_bytes),
                lane,
                warp_linear_idx,
                tile_tokens,
                self.traits.upcast_stride_k,
                False,
            )
            cute.arch.cp_async_commit_group()
            self._async_copy_paged_tile_permuted_128b(
                mVBytes,
                mPageTable,
                request_idx,
                prefetch_base,
                kv_head_idx,
                mVCache.shape[2],
                v_row_bytes,
                sVStageBytes,
                Int32(preload_stage_idx * v_stage_bytes),
                lane,
                warp_linear_idx,
                tile_tokens,
                self.traits.upcast_stride_v,
                True,
            )
            cute.arch.cp_async_commit_group()
            prefetch_base += stage_tile_rows
            preload_count += 1
            if const_expr(self.num_stages == 2):
                preload_stage_idx = Int32(1) - preload_stage_idx

        consume_stage_idx = Int32(0)
        tile_base = chunk_start
        while tile_base < chunk_end:
            tile_limit = cutlass.select_(tile_base + stage_tile_rows < chunk_end, tile_base + stage_tile_rows, chunk_end)
            tile_tokens = tile_limit - tile_base
            if const_expr(self.traits.num_warps_kv > 1):
                cute.arch.cp_async_wait_group(1 if self.kv_is_fp8 else 0)
            else:
                cute.arch.cp_async_wait_group(1)
            cute.arch.sync_threads()

            subtile_base = Int32(0) if const_expr(self.traits.num_warps_kv == 1) else warp_kv_base
            for _ in cutlass.range_constexpr(1):
                if const_expr(self.use_mxfp8_qk):
                    k_smem_base_addr = shared_ptr_to_u32(sKStageBytes.iterator + Int32(consume_stage_idx * k_stage_bytes))
                    frag_S = cute.make_rmem_tensor(
                        cute.make_layout(
                            (num_mma_q, num_mma_kv, 8),
                            stride=(num_mma_kv * 8, 8, 1),
                        ),
                        Float32,
                    )
                    frag_S.fill(0.0)
                    _literal_qk_mma_into_sfrag_mxfp8_raw(
                        frag_S,
                        q_smem_base_addr,
                        k_smem_base_addr,
                        lane,
                        warp_q_idx,
                        warp_kv_idx,
                        Int32(0) if const_expr(self.traits.num_warps_kv > 1) else subtile_base,
                        num_mma_q,
                        num_mma_kv,
                        self.traits.num_mma_d_qk,
                        tc_upcast_stride_qk,
                        self.traits.upcast_stride_k,
                    )
                    for mma_q in cutlass.range_constexpr(num_mma_q):
                        for mma_kv in cutlass.range_constexpr(num_mma_kv):
                            for reg_id in cutlass.range_constexpr(8):
                                row_slot = (reg_id % 4) // 2
                                key_local = (
                                    warp_kv_base + mma_kv * 16 + lane_pair_base + 8 * (reg_id // 4) + (reg_id % 2)
                                )
                                valid = row_valid[mma_q, row_slot] != 0
                                if valid:
                                    valid = valid and key_local < tile_tokens
                                if valid:
                                    valid = valid and (tile_base + key_local) <= causal_k_limit[mma_q, row_slot]
                                if valid:
                                    frag_S[mma_q, mma_kv, reg_id] = frag_S[mma_q, mma_kv, reg_id] * k_scale
                                else:
                                    frag_S[mma_q, mma_kv, reg_id] = Float32(-Float32.inf)
                elif const_expr(self.kv_is_fp8):
                    k_smem_base_addr = shared_ptr_to_u32(sKStageBytes.iterator + Int32(consume_stage_idx * k_stage_bytes))
                    frag_S = cute.make_rmem_tensor(
                        cute.make_layout(
                            (num_mma_q, num_mma_kv, 8),
                            stride=(num_mma_kv * 8, 8, 1),
                        ),
                        Float32,
                    )
                    frag_S.fill(0.0)
                    _literal_qk_mma_into_sfrag_fp8_raw(
                        frag_S,
                        q_smem_base_addr,
                        k_smem_base_addr,
                        lane,
                        warp_q_idx,
                        warp_kv_idx,
                        Int32(0) if const_expr(self.traits.num_warps_kv > 1) else subtile_base,
                        num_mma_q,
                        num_mma_kv,
                        self.traits.num_mma_d_qk,
                        tc_upcast_stride_qk,
                        self.traits.upcast_stride_k,
                    )
                    for mma_q in cutlass.range_constexpr(num_mma_q):
                        for mma_kv in cutlass.range_constexpr(num_mma_kv):
                            for reg_id in cutlass.range_constexpr(8):
                                row_slot = (reg_id % 4) // 2
                                key_local = (
                                    warp_kv_base + mma_kv * 16 + lane_pair_base + 8 * (reg_id // 4) + (reg_id % 2)
                                )
                                valid = row_valid[mma_q, row_slot] != 0
                                if valid:
                                    valid = valid and key_local < tile_tokens
                                if valid:
                                    valid = valid and (tile_base + key_local) <= causal_k_limit[mma_q, row_slot]
                                if valid:
                                    frag_S[mma_q, mma_kv, reg_id] = frag_S[mma_q, mma_kv, reg_id] * k_scale
                                else:
                                    frag_S[mma_q, mma_kv, reg_id] = Float32(-Float32.inf)
                else:
                    literal_key_base = Int32(0) if const_expr(self.traits.num_warps_kv > 1) else subtile_base
                    k_smem_base_addr = shared_ptr_to_u32(
                        sKStageBytes.iterator + Int32(consume_stage_idx * k_stage_bytes)
                    )
                    frag_S = cute.make_rmem_tensor(
                        cute.make_layout(
                            (num_mma_q, num_mma_kv, 8),
                            stride=(num_mma_kv * 8, 8, 1),
                        ),
                        Float32,
                    )
                    frag_S.fill(0.0)
                    _literal_qk_mma_into_sfrag(
                        frag_S,
                        q_smem_base_addr,
                        k_smem_base_addr,
                        lane,
                        warp_q_idx,
                        warp_kv_idx,
                        literal_key_base,
                        num_mma_q,
                        num_mma_kv,
                        self.traits.num_mma_d_qk,
                        tc_upcast_stride_qk,
                        tc_upcast_stride_qk,
                    )
                    for mma_q in cutlass.range_constexpr(num_mma_q):
                        for mma_kv in cutlass.range_constexpr(num_mma_kv):
                            for reg_id in cutlass.range_constexpr(8):
                                row_slot = (reg_id % 4) // 2
                                key_local = (
                                    literal_key_base + mma_kv * 16 + lane_pair_base + 8 * (reg_id // 4) + (reg_id % 2)
                                )
                                valid = row_valid[mma_q, row_slot] != 0
                                if valid:
                                    valid = valid and key_local < tile_tokens
                                if valid:
                                    valid = valid and (tile_base + key_local) <= causal_k_limit[mma_q, row_slot]
                                if not valid:
                                    frag_S[mma_q, mma_kv, reg_id] = Float32(-Float32.inf)

                p_frag = cute.make_rmem_tensor(
                    cute.make_layout(
                        (num_mma_q, num_mma_kv, 4),
                        stride=(num_mma_kv * 4, 4, 1),
                    ),
                    Uint32,
                )
                _literal_update_mdo_states_fp32_pack_p(
                    frag_S,
                    o_frag,
                    m_frag,
                    d_frag,
                    p_frag,
                    self.softmax_scale_log2,
                    num_mma_q,
                    num_mma_kv,
                    num_mma_d_vo,
                )
                for mma_q in cutlass.range_constexpr(num_mma_q):
                    for mma_kv in cutlass.range_constexpr(num_mma_kv):
                        d0, d1 = bf16_rowsum_m16k16_f32(
                            d_frag[mma_q, 0],
                            d_frag[mma_q, 1],
                            p_frag[mma_q, mma_kv, 0],
                            p_frag[mma_q, mma_kv, 1],
                            p_frag[mma_q, mma_kv, 2],
                            p_frag[mma_q, mma_kv, 3],
                        )
                        d_frag[mma_q, 0] = d0
                        d_frag[mma_q, 1] = d1

                next_tile_base = prefetch_base
                next_tile_tokens = Int32(0)
                if const_expr(self.traits.num_warps_kv > 1):
                    if next_tile_base < chunk_end:
                        next_tile_limit = cutlass.select_(
                            next_tile_base + stage_tile_rows < chunk_end,
                            next_tile_base + stage_tile_rows,
                            chunk_end,
                        )
                        next_tile_tokens = next_tile_limit - next_tile_base
                        self._async_copy_paged_tile_permuted_128b(
                            mKBytes,
                            mPageTable,
                            request_idx,
                            next_tile_base,
                            kv_head_idx,
                            mKCache.shape[2],
                            k_row_bytes,
                            sKStageBytes,
                            Int32(consume_stage_idx * k_stage_bytes),
                            lane,
                            warp_linear_idx,
                            next_tile_tokens,
                            self.traits.upcast_stride_k,
                            False,
                        )
                        cute.arch.cp_async_commit_group()
                    cute.arch.cp_async_wait_group(1)
                    cute.arch.sync_threads()
                elif const_expr(self.traits.num_warps_kv == 1):
                    if next_tile_base < chunk_end:
                        next_tile_limit = cutlass.select_(
                            next_tile_base + stage_tile_rows < chunk_end,
                            next_tile_base + stage_tile_rows,
                            chunk_end,
                        )
                        next_tile_tokens = next_tile_limit - next_tile_base
                        self._async_copy_paged_tile_permuted_128b(
                            mKBytes,
                            mPageTable,
                            request_idx,
                            next_tile_base,
                            kv_head_idx,
                            mKCache.shape[2],
                            k_row_bytes,
                            sKStageBytes,
                            Int32(consume_stage_idx * k_stage_bytes),
                            lane,
                            warp_linear_idx,
                            next_tile_tokens,
                            self.traits.upcast_stride_k,
                            False,
                        )
                        cute.arch.cp_async_commit_group()
                    cute.arch.cp_async_wait_group(1)
                    cute.arch.sync_threads()

                if const_expr(self.use_mxfp8_pv):
                    v_smem_base_addr = shared_ptr_to_u32(sVStageBytes.iterator + Int32(consume_stage_idx * v_stage_bytes))
                    _literal_pv_mma_into_ofrag_mxfp8_raw(
                        o_frag,
                        p_frag,
                        v_smem_base_addr,
                        lane,
                        warp_kv_idx,
                        Int32(0) if const_expr(self.traits.num_warps_kv > 1) else subtile_base,
                        num_mma_q,
                        num_mma_kv,
                        num_mma_d_vo,
                        self.traits.upcast_stride_v,
                        v_scale,
                    )
                elif const_expr(self.kv_is_fp8):
                    v_smem_base_addr = shared_ptr_to_u32(sVStageBytes.iterator + Int32(consume_stage_idx * v_stage_bytes))
                    _literal_pv_mma_into_ofrag_fp8_raw(
                        o_frag,
                        p_frag,
                        v_smem_base_addr,
                        lane,
                        warp_kv_idx,
                        Int32(0) if const_expr(self.traits.num_warps_kv > 1) else subtile_base,
                        num_mma_q,
                        num_mma_kv,
                        num_mma_d_vo,
                        self.traits.upcast_stride_v,
                        v_scale,
                    )
                else:
                    v_smem_base_addr = shared_ptr_to_u32(
                        sVStageBytes.iterator + Int32(consume_stage_idx * v_stage_bytes)
                    )
                    _literal_pv_mma_into_ofrag_bf16_packed(
                        o_frag,
                        p_frag,
                        v_smem_base_addr,
                        lane,
                        warp_kv_idx,
                        Int32(0) if const_expr(self.traits.num_warps_kv > 1) else subtile_base,
                        num_mma_q,
                        num_mma_kv,
                        num_mma_d_vo,
                        tc_upcast_stride_vo,
                        v_scale,
                    )

                if const_expr(self.traits.num_warps_kv > 1):
                    if next_tile_base < chunk_end:
                        self._async_copy_paged_tile_permuted_128b(
                            mVBytes,
                            mPageTable,
                            request_idx,
                            next_tile_base,
                            kv_head_idx,
                            mVCache.shape[2],
                            v_row_bytes,
                            sVStageBytes,
                            Int32(consume_stage_idx * v_stage_bytes),
                            lane,
                            warp_linear_idx,
                            next_tile_tokens,
                            self.traits.upcast_stride_v,
                            True,
                        )
                        cute.arch.cp_async_commit_group()
                        prefetch_base += stage_tile_rows
                elif const_expr(self.traits.num_warps_kv == 1):
                    if next_tile_base < chunk_end:
                        self._async_copy_paged_tile_permuted_128b(
                            mVBytes,
                            mPageTable,
                            request_idx,
                            next_tile_base,
                            kv_head_idx,
                            mVCache.shape[2],
                            v_row_bytes,
                            sVStageBytes,
                            Int32(consume_stage_idx * v_stage_bytes),
                            lane,
                            warp_linear_idx,
                            next_tile_tokens,
                            self.traits.upcast_stride_v,
                            True,
                        )
                        cute.arch.cp_async_commit_group()
                        prefetch_base += stage_tile_rows

            cute.arch.sync_threads()
            if const_expr(self.num_stages == 2):
                consume_stage_idx = Int32(1) - consume_stage_idx
            tile_base += stage_tile_rows

        for mma_q in cutlass.range_constexpr(num_mma_q):
            for row_slot in cutlass.range_constexpr(2):
                if m_frag[mma_q, row_slot] != -Float32.inf:
                    m_frag[mma_q, row_slot] = Float32(m_frag[mma_q, row_slot] * self.softmax_scale_log2)

        if const_expr(self.traits.num_warps_kv > 1):
            for mma_q in cutlass.range_constexpr(num_mma_q):
                for row_slot in cutlass.range_constexpr(2):
                    packed_row_local = row_local_idx[mma_q, row_slot]
                    if row_valid[mma_q, row_slot] != 0 and lane_pair_base == 0:
                        sSyncMD[warp_kv_idx, packed_row_local, 0] = m_frag[mma_q, row_slot]
                        sSyncMD[warp_kv_idx, packed_row_local, 1] = d_frag[mma_q, row_slot]
                    for mma_d in cutlass.range_constexpr(num_mma_d_vo):
                        dim_low = mma_d * 16 + lane_pair_base
                        dim_high = dim_low + 8
                        reg_base = row_slot * 2
                        if row_valid[mma_q, row_slot] != 0:
                            sSyncO[warp_kv_idx, packed_row_local, dim_low + 0] = o_frag[mma_q, mma_d, reg_base + 0]
                            sSyncO[warp_kv_idx, packed_row_local, dim_low + 1] = o_frag[mma_q, mma_d, reg_base + 1]
                            sSyncO[warp_kv_idx, packed_row_local, dim_high + 0] = o_frag[mma_q, mma_d, reg_base + 4]
                            sSyncO[warp_kv_idx, packed_row_local, dim_high + 1] = o_frag[mma_q, mma_d, reg_base + 5]
            cute.arch.sync_threads()

        store_enabled = valid_work and warp_kv_idx == 0
        packed_row_local = Int32(0)
        q_head_idx = Int32(0)
        q_row_idx = Int32(0)
        token_local = Int32(0)
        partial_row_idx = Int32(0)
        if const_expr(self.traits.num_warps_kv > 1):
            if store_enabled:
                for mma_q in cutlass.range_constexpr(num_mma_q):
                    for row_slot in cutlass.range_constexpr(2):
                        packed_row_local = row_local_idx[mma_q, row_slot]
                        q_head_idx = q_head_idx_frag[mma_q, row_slot]
                        q_row_idx = q_row_idx_frag[mma_q, row_slot]
                        token_local = q_token_local[mma_q, row_slot]
                        valid_row_store = row_valid[mma_q, row_slot] != 0
                        merged_m = Float32(-Float32.inf)
                        merged_d = Float32(1.0)
                        inv_d = Float32(0.0)
                        merge_scale = cute.make_rmem_tensor(
                            cute.make_layout((self.traits.num_warps_kv,), stride=(1,)),
                            Float32,
                        )
                        merge_scale.fill(0.0)
                        if valid_row_store:
                            for kv_warp in cutlass.range_constexpr(self.traits.num_warps_kv):
                                part_m = sSyncMD[kv_warp, packed_row_local, 0]
                                part_d = sSyncMD[kv_warp, packed_row_local, 1]
                                if merged_m == -Float32.inf:
                                    merged_m = part_m
                                    merged_d = part_d
                                elif part_m != -Float32.inf:
                                    new_m = attention_utils.fmax(merged_m, part_m)
                                    merged_d = Float32(
                                        merged_d * _exp2_approx_ftz_f32(merged_m - new_m)
                                        + part_d * _exp2_approx_ftz_f32(part_m - new_m)
                                    )
                                    merged_m = new_m
                            if merged_m != -Float32.inf:
                                inv_d = cute.arch.rcp_approx(merged_d)
                                for kv_warp in cutlass.range_constexpr(self.traits.num_warps_kv):
                                    part_m = sSyncMD[kv_warp, packed_row_local, 0]
                                    merge_scale[kv_warp] = (
                                        Float32(0.0)
                                        if part_m == -Float32.inf
                                        else _exp2_approx_ftz_f32(part_m - merged_m)
                                    )

                        for mma_d in cutlass.range_constexpr(num_mma_d_vo):
                            dim_low = mma_d * 16 + lane_pair_base
                            dim_high = dim_low + 8
                            out_low0 = Float32(0.0)
                            out_low1 = Float32(0.0)
                            out_high0 = Float32(0.0)
                            out_high1 = Float32(0.0)
                            if valid_row_store and merged_m != -Float32.inf:
                                acc_low0 = Float32(0.0)
                                acc_low1 = Float32(0.0)
                                acc_high0 = Float32(0.0)
                                acc_high1 = Float32(0.0)
                                for kv_warp in cutlass.range_constexpr(self.traits.num_warps_kv):
                                    scale = merge_scale[kv_warp]
                                    acc_low0 += sSyncO[kv_warp, packed_row_local, dim_low + 0] * scale
                                    acc_low1 += sSyncO[kv_warp, packed_row_local, dim_low + 1] * scale
                                    acc_high0 += sSyncO[kv_warp, packed_row_local, dim_high + 0] * scale
                                    acc_high1 += sSyncO[kv_warp, packed_row_local, dim_high + 1] * scale
                                out_low0 = acc_low0 * inv_d
                                out_low1 = acc_low1 * inv_d
                                out_high0 = acc_high0 * inv_d
                                out_high1 = acc_high1 * inv_d

                            if valid_row_store:
                                if const_expr(self.dtype_o == cutlass.BFloat16):
                                    sDecodeStageU32[0, packed_row_local, dim_low // 2] = pack_f32x2_to_bfloat2(
                                        out_low0, out_low1
                                    )
                                    sDecodeStageU32[0, packed_row_local, dim_high // 2] = pack_f32x2_to_bfloat2(
                                        out_high0, out_high1
                                    )
                                elif split_store_v128:
                                    sOStageU32[packed_row_local, dim_low // 2] = pack_f32x2_to_bfloat2(out_low0, out_low1)
                                    sOStageU32[packed_row_local, dim_high // 2] = pack_f32x2_to_bfloat2(
                                        out_high0, out_high1
                                    )
                                elif final_store_v128:
                                    sOStageU32[packed_row_local, dim_low // 2] = pack_f32x2_to_bfloat2(out_low0, out_low1)
                                    sOStageU32[packed_row_local, dim_high // 2] = pack_f32x2_to_bfloat2(
                                        out_high0, out_high1
                                    )
                                elif const_expr(self.split_kv):
                                    partial_row_idx = request_partial_start + token_local * num_chunks_kv + kv_tile_idx
                                    mO[partial_row_idx, q_head_idx, dim_low + 0] = out_low0.to(self.dtype_o)
                                    mO[partial_row_idx, q_head_idx, dim_low + 1] = out_low1.to(self.dtype_o)
                                    mO[partial_row_idx, q_head_idx, dim_high + 0] = out_high0.to(self.dtype_o)
                                    mO[partial_row_idx, q_head_idx, dim_high + 1] = out_high1.to(self.dtype_o)
                                else:
                                    mO[q_row_idx, q_head_idx, dim_low + 0] = out_low0.to(self.dtype_o)
                                    mO[q_row_idx, q_head_idx, dim_low + 1] = out_low1.to(self.dtype_o)
                                    mO[q_row_idx, q_head_idx, dim_high + 0] = out_high0.to(self.dtype_o)
                                    mO[q_row_idx, q_head_idx, dim_high + 1] = out_high1.to(self.dtype_o)
                        if valid_row_store and lane_pair_base == 0:
                            row_lse = (
                                Float32(-Float32.inf)
                                if merged_m == -Float32.inf
                                else Float32(merged_m + cute.math.log2(merged_d, fastmath=True))
                            )
                            if const_expr(self.split_kv):
                                partial_row_idx = request_partial_start + token_local * num_chunks_kv + kv_tile_idx
                                mLSE[partial_row_idx, q_head_idx] = row_lse
                            else:
                                mLSE[q_head_idx, q_row_idx] = row_lse
        else:
            for mma_q in cutlass.range_constexpr(num_mma_q):
                for row_slot in cutlass.range_constexpr(2):
                    packed_row_local = row_local_idx[mma_q, row_slot]
                    q_head_idx = q_head_idx_frag[mma_q, row_slot]
                    q_row_idx = q_row_idx_frag[mma_q, row_slot]
                    token_local = q_token_local[mma_q, row_slot]
                    valid_row_store = row_valid[mma_q, row_slot] != 0
                    merged_m = Float32(-Float32.inf)
                    merged_d = Float32(1.0)
                    inv_d = Float32(0.0)
                    if store_enabled and valid_row_store:
                        merged_m = m_frag[mma_q, row_slot]
                        merged_d = d_frag[mma_q, row_slot]
                        if merged_m != -Float32.inf:
                            inv_d = cute.arch.rcp_approx(merged_d)

                    for mma_d in cutlass.range_constexpr(num_mma_d_vo):
                        dim_low = mma_d * 16 + lane_pair_base
                        dim_high = dim_low + 8
                        reg_base = row_slot * 2
                        out_low0 = Float32(0.0)
                        out_low1 = Float32(0.0)
                        out_high0 = Float32(0.0)
                        out_high1 = Float32(0.0)
                        if store_enabled and valid_row_store and merged_m != -Float32.inf:
                            out_low0 = o_frag[mma_q, mma_d, reg_base + 0] * inv_d
                            out_low1 = o_frag[mma_q, mma_d, reg_base + 1] * inv_d
                            out_high0 = o_frag[mma_q, mma_d, reg_base + 4] * inv_d
                            out_high1 = o_frag[mma_q, mma_d, reg_base + 5] * inv_d

                        if store_enabled and valid_row_store:
                            if split_store_v128:
                                sOStageU32[packed_row_local, dim_low // 2] = pack_f32x2_to_bfloat2(out_low0, out_low1)
                                sOStageU32[packed_row_local, dim_high // 2] = pack_f32x2_to_bfloat2(
                                    out_high0, out_high1
                                )
                            elif final_store_v128:
                                sOStageU32[packed_row_local, dim_low // 2] = pack_f32x2_to_bfloat2(out_low0, out_low1)
                                sOStageU32[packed_row_local, dim_high // 2] = pack_f32x2_to_bfloat2(
                                    out_high0, out_high1
                                )
                            elif const_expr(self.split_kv):
                                partial_row_idx = request_partial_start + token_local * num_chunks_kv + kv_tile_idx
                                mO[partial_row_idx, q_head_idx, dim_low + 0] = out_low0.to(self.dtype_o)
                                mO[partial_row_idx, q_head_idx, dim_low + 1] = out_low1.to(self.dtype_o)
                                mO[partial_row_idx, q_head_idx, dim_high + 0] = out_high0.to(self.dtype_o)
                                mO[partial_row_idx, q_head_idx, dim_high + 1] = out_high1.to(self.dtype_o)
                            else:
                                mO[q_row_idx, q_head_idx, dim_low + 0] = out_low0.to(self.dtype_o)
                                mO[q_row_idx, q_head_idx, dim_low + 1] = out_low1.to(self.dtype_o)
                                mO[q_row_idx, q_head_idx, dim_high + 0] = out_high0.to(self.dtype_o)
                                mO[q_row_idx, q_head_idx, dim_high + 1] = out_high1.to(self.dtype_o)
                    if store_enabled and valid_row_store and lane_pair_base == 0:
                        row_lse = (
                            Float32(-Float32.inf)
                            if merged_m == -Float32.inf
                            else Float32(merged_m + cute.math.log2(merged_d, fastmath=True))
                        )
                        if const_expr(self.split_kv):
                            partial_row_idx = request_partial_start + token_local * num_chunks_kv + kv_tile_idx
                            mLSE[partial_row_idx, q_head_idx] = row_lse
                        else:
                            mLSE[q_head_idx, q_row_idx] = row_lse

        if const_expr(decode_store_v128):
            if valid_work:
                cute.arch.sync_threads()
                decode_chunks_per_row = self.traits.head_dim_vo // 8
                decode_chunk_linear_idx = tidx
                decode_total_chunks = packed_tile_rows * decode_chunks_per_row
                while decode_chunk_linear_idx < decode_total_chunks:
                    packed_row_local = decode_chunk_linear_idx // decode_chunks_per_row
                    chunk_idx = decode_chunk_linear_idx - packed_row_local * decode_chunks_per_row
                    packed_q_idx = packed_tile_start + packed_row_local
                    token_local = packed_q_idx // group_size
                    q_group_lane = packed_q_idx - token_local * group_size
                    q_head_idx = kv_head_idx * group_size + q_group_lane
                    q_row_idx = q_start + token_local
                    u32_idx = chunk_idx * 4
                    if const_expr(self.split_kv):
                        partial_row_idx = request_partial_start + token_local * num_chunks_kv + kv_tile_idx
                        gmem_elem_offset = (
                            ((partial_row_idx * mO.shape[1] + q_head_idx) * self.traits.head_dim_vo)
                            + chunk_idx * 8
                        )
                    else:
                        gmem_elem_offset = (
                            ((q_row_idx * mO.shape[1] + q_head_idx) * self.traits.head_dim_vo)
                            + chunk_idx * 8
                        )
                    st_global_v4_u32(
                        get_ptr_as_int64(mOFlat, gmem_elem_offset),
                        sDecodeStageU32[0, packed_row_local, u32_idx + 0],
                        sDecodeStageU32[0, packed_row_local, u32_idx + 1],
                        sDecodeStageU32[0, packed_row_local, u32_idx + 2],
                        sDecodeStageU32[0, packed_row_local, u32_idx + 3],
                    )
                    decode_chunk_linear_idx += self.traits.num_threads

        if split_store_v128:
            if valid_work:
                cute.arch.sync_threads()
                split_chunks_per_row = self.traits.head_dim_vo // 8
                split_chunk_linear_idx = tidx
                split_total_chunks = packed_tile_rows * split_chunks_per_row
                while split_chunk_linear_idx < split_total_chunks:
                    packed_row_local = split_chunk_linear_idx // split_chunks_per_row
                    chunk_idx = split_chunk_linear_idx - packed_row_local * split_chunks_per_row
                    packed_q_idx = packed_tile_start + packed_row_local
                    token_local = packed_q_idx // group_size
                    q_group_lane = packed_q_idx - token_local * group_size
                    q_head_idx = kv_head_idx * group_size + q_group_lane
                    partial_row_idx = request_partial_start + token_local * num_chunks_kv + kv_tile_idx
                    u32_idx = chunk_idx * 4
                    gmem_elem_offset = (
                        ((partial_row_idx * mO.shape[1] + q_head_idx) * self.traits.head_dim_vo)
                        + chunk_idx * 8
                    )
                    st_global_v4_u32(
                        get_ptr_as_int64(mOFlat, gmem_elem_offset),
                        sOStageU32[packed_row_local, u32_idx + 0],
                        sOStageU32[packed_row_local, u32_idx + 1],
                        sOStageU32[packed_row_local, u32_idx + 2],
                        sOStageU32[packed_row_local, u32_idx + 3],
                    )
                    split_chunk_linear_idx += self.traits.num_threads

        if final_store_v128:
            if valid_work:
                cute.arch.sync_threads()
                final_chunks_per_row = self.traits.head_dim_vo // 8
                final_chunk_linear_idx = tidx
                final_total_chunks = packed_tile_rows * final_chunks_per_row
                while final_chunk_linear_idx < final_total_chunks:
                    packed_row_local = final_chunk_linear_idx // final_chunks_per_row
                    chunk_idx = final_chunk_linear_idx - packed_row_local * final_chunks_per_row
                    packed_q_idx = packed_tile_start + packed_row_local
                    token_local = packed_q_idx // group_size
                    q_group_lane = packed_q_idx - token_local * group_size
                    q_head_idx = kv_head_idx * group_size + q_group_lane
                    q_row_idx = q_start + token_local
                    u32_idx = chunk_idx * 4
                    gmem_elem_offset = (
                        ((q_row_idx * mO.shape[1] + q_head_idx) * self.traits.head_dim_vo)
                        + chunk_idx * 8
                    )
                    st_global_v4_u32(
                        get_ptr_as_int64(mOFlat, gmem_elem_offset),
                        sOStageU32[packed_row_local, u32_idx + 0],
                        sOStageU32[packed_row_local, u32_idx + 1],
                        sOStageU32[packed_row_local, u32_idx + 2],
                        sOStageU32[packed_row_local, u32_idx + 3],
                    )
                    final_chunk_linear_idx += self.traits.num_threads
