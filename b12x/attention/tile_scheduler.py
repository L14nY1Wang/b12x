from dataclasses import dataclass
from typing import Tuple

import cutlass
import cutlass.cute as cute
from cutlass._mlir import ir
from cutlass import Int32
from cutlass.cute import FastDivmodDivisor

from b12x.attention.cute_dsl_utils import ParamsBase


class WorkTileInfo(cutlass.utils.WorkTileInfo):
    def __new_from_mlir_values__(self, values: list[ir.Value]) -> "WorkTileInfo":
        assert len(values) == 5
        new_tile_idx = cutlass.new_from_mlir_values(self._tile_idx, values[:-1])
        new_is_valid_tile = cutlass.new_from_mlir_values(self._is_valid_tile, [values[-1]])
        return WorkTileInfo(new_tile_idx, new_is_valid_tile)


@dataclass
class TileSchedulerArguments(ParamsBase):
    num_block: Int32
    num_head: Int32
    num_batch: Int32
    num_splits: Int32
    seqlen_k: Int32
    headdim: Int32
    headdim_v: Int32
    total_q: Int32
    tile_shape_mn: cutlass.Constexpr[Tuple[int, int]]
    cluster_shape_mn: cutlass.Constexpr[Tuple[int, int]] = (1, 1)
    mCuSeqlensQ: cute.Tensor | None = None
    mSeqUsedQ: cute.Tensor | None = None
    qhead_per_kvhead_packgqa: cutlass.Constexpr[int] = 1
    element_size: cutlass.Constexpr[int] = 2
    is_persistent: cutlass.Constexpr[bool] = False
    lpt: cutlass.Constexpr[bool] = False
    is_split_kv: cutlass.Constexpr[bool] = False
    head_swizzle: cutlass.Constexpr[bool] = False


class SingleTileScheduler:
    @dataclass
    class Params(ParamsBase):
        num_block: Int32
        num_head: Int32
        num_batch: Int32
        num_splits: Int32
        num_splits_divmod: FastDivmodDivisor
        is_split_kv: cutlass.Constexpr[bool] = False
        cluster_shape_mn: cutlass.Constexpr[Tuple[int, int]] = (1, 1)

        @staticmethod
        def create(args: TileSchedulerArguments, *, loc=None, ip=None):
            del loc, ip
            return SingleTileScheduler.Params(
                args.num_block,
                args.num_head,
                args.num_batch,
                args.num_splits,
                FastDivmodDivisor(args.num_splits),
                args.is_split_kv,
                args.cluster_shape_mn,
            )

    def __init__(self, params: "SingleTileScheduler.Params", blk_coord: cute.Coord, *, loc=None, ip=None):
        self.params = params
        self._blk_coord = blk_coord
        self._is_first_block = True
        self._loc = loc
        self._ip = ip

    @staticmethod
    def to_underlying_arguments(args: TileSchedulerArguments, *, loc=None, ip=None):
        return SingleTileScheduler.Params.create(args, loc=loc, ip=ip)

    @staticmethod
    def create(params: "SingleTileScheduler.Params", *, loc=None, ip=None):
        return SingleTileScheduler(params, cute.arch.block_idx(), loc=loc, ip=ip)

    @staticmethod
    def get_grid_shape(params: "SingleTileScheduler.Params", *, loc=None, ip=None):
        del loc, ip
        assert params.cluster_shape_mn[1] == 1
        return (
            cute.round_up(params.num_block, params.cluster_shape_mn[0]),
            params.num_head * params.num_splits,
            params.num_batch,
        )

    def get_current_work(self, *, loc=None, ip=None) -> WorkTileInfo:
        del loc, ip
        block_idx, head_idx, batch_idx = self._blk_coord
        if cutlass.const_expr(self.params.is_split_kv):
            head_idx, split_idx = divmod(head_idx, self.params.num_splits_divmod)
        else:
            split_idx = Int32(0)
        return WorkTileInfo((block_idx, head_idx, batch_idx, split_idx), self._is_first_block)

    def initial_work_tile_info(self, *, loc=None, ip=None):
        return self.get_current_work(loc=loc, ip=ip)

    def prefetch_next_work(self, *, loc=None, ip=None):
        del loc, ip

    def advance_to_next_work(self, *, loc=None, ip=None):
        del loc, ip
        self._is_first_block = False

    def __extract_mlir_values__(self):
        values, self._values_pos = [], []
        for obj in [self.params, self._blk_coord]:
            obj_values = cutlass.extract_mlir_values(obj)
            values += obj_values
            self._values_pos.append(len(obj_values))
        return values

    def __new_from_mlir_values__(self, values):
        obj_list = []
        for obj, n_items in zip([self.params, self._blk_coord], self._values_pos):
            obj_list.append(cutlass.new_from_mlir_values(obj, values[:n_items]))
            values = values[n_items:]
        return SingleTileScheduler(*(tuple(obj_list)), loc=self._loc)
