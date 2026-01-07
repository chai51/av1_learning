"""
运动向量预测过程模块
实现规范文档7.10节"Motion vector prediction processes"中描述的所有过程函数
"""

from copy import deepcopy
from typing import List
# 参考帧相关常量
from constants import (
    GM_TYPE,
    MV_BORDER,
    NONE,
    PARTITION,
    REF_FRAME,
    MAX_REF_MV_STACK_SIZE, REF_CAT_LEVEL, LEAST_SQUARES_SAMPLES_MAX,
    WARPEDMODEL_PREC_BITS,
    Y_MODE,
)
# 块尺寸相关常量
from constants import (
    Num_4x4_Blocks_Wide, Num_4x4_Blocks_High,
    Block_Width, Block_Height, SUB_SIZE
)
# MI尺寸相关常量
from constants import MI_SIZE
from obu.decoder import AV1Decoder
from utils.tile_utils import is_inside, clamp_mv_col, clamp_mv_row
from utils.math_utils import Array, Clip3, Round2Signed


class FindMvStack:
    def __init__(self):
        self._RefStackMv: List[List[List[int]]] = Array(
            None, (MAX_REF_MV_STACK_SIZE, 2))
        self._NewMvCount = 0

        self._FoundMatch = 0
        self._CloseMatches = 0
        self._WeightStack: List[int] = [NONE] * MAX_REF_MV_STACK_SIZE
        self._TotalMatches = 0
        self._RefIdCount: List[int] = [0, 0]
        self._RefDiffCount: List[int] = [0, 0]
        self._RefIdMvs: List[List[List[int]]] = [
            [NONE, NONE], [NONE, NONE]]  # [2][max 2][2]
        self._RefDiffMvs: List[List[List[int]]] = [
            [NONE, NONE], [NONE, NONE]]  # [2][max 2][2]

    def find_mv_stack(self, av1: AV1Decoder, isCompound: int) -> List[List[List[int]]]:
        """
        查找MV栈过程
        规范文档 7.10.2 Find MV stack process

        此过程由函数调用find_mv_stack触发。
        输入是变量isCompound，包含0表示单预测，或1表示复合预测。
        此过程构建包含运动向量候选的数组RefStackMv。

        Args:
            isCompound: 0表示单预测，1表示复合预测
        """
        frame_header = av1.frame_header
        tile_group = av1.tile_group
        MiSize = tile_group.MiSize

        bw4 = Num_4x4_Blocks_Wide[MiSize]

        bh4 = Num_4x4_Blocks_High[MiSize]

        # 1.
        tile_group.NumMvFound = 0

        # 2.
        self._NewMvCount = 0

        # 3.
        tile_group.GlobalMvs[0] = self._setup_global_mv_process(av1, 0)

        # 4.
        if isCompound == 1:
            tile_group.GlobalMvs[1] = self._setup_global_mv_process(av1, 1)

        # 5.
        self._FoundMatch = 0

        # 6.
        self._scan_row_process(av1, -1, isCompound)

        # 7.
        foundAboveMatch = self._FoundMatch
        self._FoundMatch = 0

        # 8.
        self._scan_col_process(av1, -1, isCompound)

        # 9.
        foundLeftMatch = self._FoundMatch
        self._FoundMatch = 0

        # 10.
        if max(bw4, bh4) <= 16 and has_top_right(av1):
            self._scan_point_process(av1, -1, bw4, isCompound)

        # 11.
        if self._FoundMatch == 1:
            foundAboveMatch = 1

        # 12.
        self._CloseMatches = foundAboveMatch + foundLeftMatch

        # 13.
        numNearest = tile_group.NumMvFound

        # 14.
        numNew = self._NewMvCount

        # 15.
        if numNearest > 0:
            for idx in range(numNearest):
                self._WeightStack[idx] += REF_CAT_LEVEL

        # 16.
        tile_group.ZeroMvContext = 0

        # 17.
        if frame_header.use_ref_frame_mvs == 1:
            self._temporal_scan_process(av1, isCompound)

        # 18.
        self._scan_point_process(av1, -1, -1, isCompound)

        # 19.
        if self._FoundMatch == 1:
            foundAboveMatch = 1

        # 20.
        self._FoundMatch = 0

        # 21.
        self._scan_row_process(av1, -3, isCompound)

        # 22.
        if self._FoundMatch == 1:
            foundAboveMatch = 1

        # 23.
        self._FoundMatch = 0

        # 24.
        self._scan_col_process(av1, -3, isCompound)

        # 25.
        if self._FoundMatch == 1:
            foundLeftMatch = 1

        # 26.
        self._FoundMatch = 0

        # 27.
        if bh4 > 1:
            self._scan_row_process(av1, -5, isCompound)

        # 28.
        if self._FoundMatch == 1:
            foundAboveMatch = 1

        # 29.
        self._FoundMatch = 0

        # 30.
        if bw4 > 1:
            self._scan_col_process(av1, -5, isCompound)

        # 31.
        if self._FoundMatch == 1:
            foundLeftMatch = 1

        # 32.
        self._TotalMatches = foundAboveMatch + foundLeftMatch

        # 33.
        self._sorting_process(av1, 0, numNearest, isCompound)

        # 34.
        self._sorting_process(
            av1, numNearest, tile_group.NumMvFound, isCompound)

        # 35.
        if tile_group.NumMvFound < 2:
            self._extra_search_process(av1, isCompound)

        # 36.
        self._context_and_clamping_process(av1, isCompound, numNew)

        return self._RefStackMv

    def _setup_global_mv_process(self, av1: AV1Decoder, refList: int) -> List[int]:
        """
        设置全局MV过程
        规范文档 7.10.2.1 Setup global MV process

        输入是变量refList，指定要预测的运动向量集合。
        输出是运动向量mv，表示此块的全局运动。

        Args:
            refList: 指定要预测的运动向量集合（0或1）

        Returns:
            mv - 全局运动向量 [row, col]
        """
        frame_header = av1.frame_header
        tile_group = av1.tile_group
        MiRow = tile_group.MiRow
        MiCol = tile_group.MiCol
        MiSize = tile_group.MiSize

        ref = tile_group.RefFrame[refList]

        typ: int = NONE
        if ref != REF_FRAME.INTRA_FRAME:
            typ = frame_header.GmType[ref]

        bw = Block_Width[MiSize]

        bh = Block_Height[MiSize]

        mv = [0, 0]
        if ref == REF_FRAME.INTRA_FRAME or typ == GM_TYPE.IDENTITY:
            mv[0] = 0
            mv[1] = 0
        elif typ == GM_TYPE.TRANSLATION:
            mv[0] = frame_header.gm_params[ref][0] >> (
                WARPEDMODEL_PREC_BITS - 3)
            mv[1] = frame_header.gm_params[ref][1] >> (
                WARPEDMODEL_PREC_BITS - 3)
        else:
            x = MiCol * MI_SIZE + bw // 2 - 1
            y = MiRow * MI_SIZE + bh // 2 - 1

            xc = ((frame_header.gm_params[ref][2] - (1 << WARPEDMODEL_PREC_BITS)) * x +
                  frame_header.gm_params[ref][3] * y +
                  frame_header.gm_params[ref][0])
            yc = (frame_header.gm_params[ref][4] * x +
                  (frame_header.gm_params[ref][5] - (1 << WARPEDMODEL_PREC_BITS)) * y +
                  frame_header.gm_params[ref][1])

            if frame_header.allow_high_precision_mv:
                mv[0] = Round2Signed(yc, WARPEDMODEL_PREC_BITS - 3)
                mv[1] = Round2Signed(xc, WARPEDMODEL_PREC_BITS - 3)
            else:
                mv[0] = Round2Signed(yc, WARPEDMODEL_PREC_BITS - 2) * 2
                mv[1] = Round2Signed(xc, WARPEDMODEL_PREC_BITS - 2) * 2

        self._lower_mv_precision(av1, mv)

        return mv

    def _scan_row_process(self, av1: AV1Decoder, deltaRow: int, isCompound: int):
        """
        扫描行过程
        规范文档 7.10.2.2 Scan row process

        Args:
            deltaRow: 行偏移（4x4单位）
            isCompound: 0表示单预测，1表示复合预测
        """
        frame_header = av1.frame_header
        tile_group = av1.tile_group
        MiCols = frame_header.MiCols
        MiRow = tile_group.MiRow
        MiCol = tile_group.MiCol
        MiSize = tile_group.MiSize

        bw4 = Num_4x4_Blocks_Wide[MiSize]

        end4 = min(min(bw4, MiCols - MiCol), 16)

        deltaCol = 0

        useStep16 = (bw4 >= 16)

        if abs(deltaRow) > 1:
            deltaRow += MiRow & 1
            deltaCol = 1 - (MiCol & 1)

        i = 0
        while i < end4:
            mvRow = MiRow + deltaRow
            mvCol = MiCol + deltaCol + i

            if not is_inside(av1, mvRow, mvCol):
                break

            len_val = min(
                bw4, Num_4x4_Blocks_Wide[tile_group.MiSizes[mvRow][mvCol]])

            if abs(deltaRow) > 1:
                len_val = max(2, len_val)

            if useStep16:
                len_val = max(4, len_val)

            weight = len_val * 2

            self._add_ref_mv_candidate(av1, mvRow, mvCol, isCompound, weight)

            i += len_val

    def _scan_col_process(self, av1: AV1Decoder, deltaCol: int, isCompound: int):
        """
        扫描列过程
        规范文档 7.10.2.3 Scan col process

        Args:
            deltaCol: 列偏移（4x4单位）
            isCompound: 0表示单预测，1表示复合预测
        """
        frame_header = av1.frame_header
        tile_group = av1.tile_group
        MiRows = frame_header.MiRows
        MiRow = tile_group.MiRow
        MiCol = tile_group.MiCol
        MiSize = tile_group.MiSize

        bh4 = Num_4x4_Blocks_High[MiSize]

        end4 = min(min(bh4, MiRows - MiRow), 16)

        deltaRow = 0

        useStep16 = (bh4 >= 16)

        if abs(deltaCol) > 1:
            deltaRow = 1 - (MiRow & 1)
            deltaCol += MiCol & 1

        i = 0
        while i < end4:
            mvRow = MiRow + deltaRow + i
            mvCol = MiCol + deltaCol

            if not is_inside(av1, mvRow, mvCol):
                break

            len_val = min(
                bh4, Num_4x4_Blocks_High[tile_group.MiSizes[mvRow][mvCol]])

            if abs(deltaCol) > 1:
                len_val = max(2, len_val)

            if useStep16:
                len_val = max(4, len_val)

            weight = len_val * 2

            self._add_ref_mv_candidate(av1, mvRow, mvCol, isCompound, weight)

            i += len_val

    def _scan_point_process(self, av1: AV1Decoder, deltaRow: int, deltaCol: int, isCompound: int):
        """
        扫描点过程
        规范文档 7.10.2.4 Scan point process
            deltaRow: 行偏移（4x4单位）
            deltaCol: 列偏移（4x4单位）
            isCompound: 0表示单预测，1表示复合预测
        """
        tile_group = av1.tile_group
        MiRow = tile_group.MiRow
        MiCol = tile_group.MiCol

        mvRow = MiRow + deltaRow

        mvCol = MiCol + deltaCol

        weight = 4

        if (is_inside(av1, mvRow, mvCol) == 1 and
                tile_group.RefFrames[mvRow][mvCol][0]):
            self._add_ref_mv_candidate(av1, mvRow, mvCol, isCompound, weight)

    def _temporal_scan_process(self, av1: AV1Decoder, isCompound: int):
        """
        时间扫描过程
        规范文档 7.10.2.5 Temporal scan process
        Args:
            isCompound: 0表示单预测，1表示复合预测
        """
        tile_group = av1.tile_group
        MiRow = tile_group.MiRow
        MiCol = tile_group.MiCol
        MiSize = tile_group.MiSize

        bw4 = Num_4x4_Blocks_Wide[MiSize]

        bh4 = Num_4x4_Blocks_High[MiSize]

        stepW4 = 4 if (bw4 >= 16) else 2

        stepH4 = 4 if (bh4 >= 16) else 2

        for deltaRow in range(0, min(bh4, 16), stepH4):
            for deltaCol in range(0, min(bw4, 16), stepW4):
                self._add_tpl_ref_mv(av1, deltaRow, deltaCol, isCompound)

        def check_sb_border(deltaRow: int, deltaCol: int) -> bool:
            """
            检查SB边界
            规范文档 7.10.2.5 Temporal scan process
            check_sb_border检查位置是否在同一64x64块内

            Args:
                deltaRow: 行偏移
                deltaCol: 列偏移

            Returns:
                如果位置在同一64x64块内返回True
            """
            MiRow = tile_group.MiRow
            MiCol = tile_group.MiCol

            row = (MiRow & 15) + deltaRow

            col = (MiCol & 15) + deltaCol

            return (row >= 0 and row < 16 and col >= 0 and col < 16)

        allowExtension = ((bh4 >= Num_4x4_Blocks_High[SUB_SIZE.BLOCK_8X8]) and
                          (bh4 < Num_4x4_Blocks_High[SUB_SIZE.BLOCK_64X64]) and
                          (bw4 >= Num_4x4_Blocks_Wide[SUB_SIZE.BLOCK_8X8]) and
                          (bw4 < Num_4x4_Blocks_Wide[SUB_SIZE.BLOCK_64X64]))

        if allowExtension:
            tplSamplePos = [
                [bh4, -2],
                [bh4, bw4],
                [bh4 - 2, bw4]
            ]

            for i in range(3):
                deltaRow = tplSamplePos[i][0]
                deltaCol = tplSamplePos[i][1]

                if check_sb_border(deltaRow, deltaCol):
                    self._add_tpl_ref_mv(av1, deltaRow, deltaCol, isCompound)

    def _add_tpl_ref_mv(self, av1: AV1Decoder, deltaRow: int, deltaCol: int, isCompound: int):
        """
        时间样本过程
        规范文档 7.10.2.6 Temporal sample process
        Args:
            deltaRow: 行偏移（4x4单位）
            deltaCol: 列偏移（4x4单位）
            isCompound: 0表示单预测，1表示复合预测
        """
        frame_header = av1.frame_header
        tile_group = av1.tile_group
        MiRow = tile_group.MiRow
        MiCol = tile_group.MiCol

        mvRow = (MiRow + deltaRow) | 1

        mvCol = (MiCol + deltaCol) | 1

        if is_inside(av1, mvRow, mvCol) == 0:
            return

        x8 = mvCol >> 1

        y8 = mvRow >> 1

        if deltaRow == 0 and deltaCol == 0:
            tile_group.ZeroMvContext = 1

        if not isCompound:
            candMv = deepcopy(
                frame_header.MotionFieldMvs[tile_group.RefFrame[0]][y8][x8])

            if candMv[0] == -1 << 15:
                return

            self._lower_mv_precision(av1, candMv)

            if deltaRow == 0 and deltaCol == 0:
                if (abs(candMv[0] - tile_group.GlobalMvs[0][0]) >= 16 or
                        abs(candMv[1] - tile_group.GlobalMvs[0][1]) >= 16):
                    tile_group.ZeroMvContext = 1
                else:
                    tile_group.ZeroMvContext = 0

            idx = 0
            while idx < tile_group.NumMvFound:
                if (candMv[0] == self._RefStackMv[idx][0][0] and
                        candMv[1] == self._RefStackMv[idx][0][1]):
                    break
                idx += 1

            if idx < tile_group.NumMvFound:
                self._WeightStack[idx] += 2
            elif tile_group.NumMvFound < MAX_REF_MV_STACK_SIZE:
                self._RefStackMv[tile_group.NumMvFound][0] = candMv
                self._WeightStack[tile_group.NumMvFound] = 2
                tile_group.NumMvFound += 1
        else:
            candMv0 = deepcopy(
                frame_header.MotionFieldMvs[tile_group.RefFrame[0]][y8][x8])

            if candMv0[0] == -1 << 15:
                return

            candMv1 = deepcopy(
                frame_header.MotionFieldMvs[tile_group.RefFrame[1]][y8][x8])

            if candMv1[0] == -1 << 15:
                return

            self._lower_mv_precision(av1, candMv0)
            self._lower_mv_precision(av1, candMv1)

            if deltaRow == 0 and deltaCol == 0:
                if (abs(candMv0[0] - tile_group.GlobalMvs[0][0]) >= 16 or
                    abs(candMv0[1] - tile_group.GlobalMvs[0][1]) >= 16 or
                    abs(candMv1[0] - tile_group.GlobalMvs[1][0]) >= 16 or
                        abs(candMv1[1] - tile_group.GlobalMvs[1][1]) >= 16):
                    tile_group.ZeroMvContext = 1
                else:
                    tile_group.ZeroMvContext = 0

            idx = 0
            while idx < tile_group.NumMvFound:
                if (candMv0[0] == self._RefStackMv[idx][0][0] and
                    candMv0[1] == self._RefStackMv[idx][0][1] and
                    candMv1[0] == self._RefStackMv[idx][1][0] and
                        candMv1[1] == self._RefStackMv[idx][1][1]):
                    break
                idx += 1
            if idx < tile_group.NumMvFound:
                self._WeightStack[idx] += 2
            elif tile_group.NumMvFound < MAX_REF_MV_STACK_SIZE:
                self._RefStackMv[tile_group.NumMvFound][0] = deepcopy(candMv0)
                self._RefStackMv[tile_group.NumMvFound][1] = deepcopy(candMv1)
                self._WeightStack[tile_group.NumMvFound] = 2
                tile_group.NumMvFound += 1

    def _add_ref_mv_candidate(self, av1: AV1Decoder, mvRow: int, mvCol: int, isCompound: int, weight: int):
        """
        添加参考运动向量过程
        规范文档 7.10.2.7 Add reference motion vector process

        Args:
            mvRow: 候选行位置
            mvCol: 候选列位置
            isCompound: 0表示单预测，1表示复合预测
            weight: 权重
        """
        tile_group = av1.tile_group

        if tile_group.IsInters[mvRow][mvCol] == 0:
            return

        if isCompound == 0:
            for candList in range(2):
                if (tile_group.RefFrames[mvRow][mvCol][candList] == tile_group.RefFrame[0]):
                    self._search_stack_process(
                        av1, mvRow, mvCol, candList, weight)
        else:
            if (tile_group.RefFrames[mvRow][mvCol][0] == tile_group.RefFrame[0] and
                    tile_group.RefFrames[mvRow][mvCol][1] == tile_group.RefFrame[1]):
                self._compound_search_stack_process(av1, mvRow, mvCol, weight)

    def _search_stack_process(self, av1: AV1Decoder, mvRow: int, mvCol: int, candList: int, weight: int):
        """
        搜索栈过程
        规范文档 7.10.2.8 Search stack process
        Args:
            mvRow: 候选行位置
            mvCol: 候选列位置
            candList: 候选列表索引
            weight: 权重
        """
        frame_header = av1.frame_header
        tile_group = av1.tile_group

        candMode = tile_group.YModes[mvRow][mvCol]
        candSize = tile_group.MiSizes[mvRow][mvCol]
        large = (min(Block_Width[candSize], Block_Height[candSize]) >= 8)

        candMv = [0, 0]
        if ((candMode == Y_MODE.GLOBALMV or candMode == Y_MODE.GLOBAL_GLOBALMV) and
            frame_header.GmType[tile_group.RefFrame[0]] > GM_TYPE.TRANSLATION and
                large == 1):
            candMv = deepcopy(tile_group.GlobalMvs[0])
        else:
            candMv = deepcopy(tile_group.Mvs[mvRow][mvCol][candList])

        self._lower_mv_precision(av1, candMv)

        if self._has_newmv(candMode) == 1:
            self._NewMvCount += 1

        self._FoundMatch = 1

        idx = 0
        while idx < tile_group.NumMvFound:
            if (candMv == self._RefStackMv[idx][0]):
                break
            idx += 1

        if idx < tile_group.NumMvFound:
            self._WeightStack[idx] += weight
        elif tile_group.NumMvFound < MAX_REF_MV_STACK_SIZE:
            self._RefStackMv[tile_group.NumMvFound][0] = deepcopy(candMv)
            self._WeightStack[tile_group.NumMvFound] = weight
            tile_group.NumMvFound += 1
        else:
            pass

    def _compound_search_stack_process(self, av1: AV1Decoder, mvRow: int, mvCol: int, weight: int):
        """
        复合搜索栈过程
        规范文档 7.10.2.9 Compound search stack process

        Args:
            mvRow: 候选行位置
            mvCol: 候选列位置
            weight: 权重

        此过程在栈中搜索与候选运动向量对的精确匹配。
        如果存在，候选运动向量对的权重被添加到栈中对应项的权重中，
        否则过程将运动向量添加到栈中。
        """
        frame_header = av1.frame_header
        tile_group = av1.tile_group

        candMvs = deepcopy(tile_group.Mvs[mvRow][mvCol])

        candMode = tile_group.YModes[mvRow][mvCol]

        candSize = tile_group.MiSizes[mvRow][mvCol]

        if candMode == Y_MODE.GLOBAL_GLOBALMV:
            for refList in range(2):
                if (frame_header.GmType[tile_group.RefFrame[refList]] > GM_TYPE.TRANSLATION):
                    candMvs[refList] = deepcopy(tile_group.GlobalMvs[refList])

        for i in range(2):
            self._lower_mv_precision(av1, candMvs[i])

        self._FoundMatch = 1

        idx = 0
        while idx < tile_group.NumMvFound:
            if (candMvs[0] == self._RefStackMv[idx][0] and
                    candMvs[1] == self._RefStackMv[idx][1]):
                break
            idx += 1

        if idx < tile_group.NumMvFound:
            self._WeightStack[idx] += weight
        elif tile_group.NumMvFound < MAX_REF_MV_STACK_SIZE:
            # a.
            for i in range(2):
                self._RefStackMv[tile_group.NumMvFound][i] = deepcopy(
                    candMvs[i])
            # b.
            self._WeightStack[tile_group.NumMvFound] = weight
            # c.
            tile_group.NumMvFound += 1
        else:
            pass

        if self._has_newmv(candMode) == 1:
            self._NewMvCount += 1

    def _has_newmv(self, mode: int) -> bool:
        """
        检查模式是否使用NEWMV编码
        规范文档 7.10.2.9 Compound search stack process

        Args:
            mode: 预测模式

        Returns:
            如果模式使用NEWMV编码返回True
        """
        return (mode == Y_MODE.NEWMV or
                mode == Y_MODE.NEW_NEWMV or
                mode == Y_MODE.NEAR_NEWMV or
                mode == Y_MODE.NEW_NEARMV or
                mode == Y_MODE.NEAREST_NEWMV or
                mode == Y_MODE.NEW_NEARESTMV)

    def _lower_mv_precision(self, av1: AV1Decoder, candMv: List[int]):
        """
        降低MV精度过程
        规范文档 7.10.2.10 Lower precision process

        输入是参考candMv的运动向量数组。
        此过程修改输入运动向量的内容，当不允许高精度时移除最低有效位，
        当force_integer_mv等于1时移除所有三个分数位。

        Args:
            candMv: 候选运动向量数组（会被修改）[row, col]
        """
        frame_header = av1.frame_header

        if frame_header.allow_high_precision_mv == 1:
            return

        for i in range(2):
            if frame_header.force_integer_mv:
                a = abs(candMv[i])
                aInt = (a + 3) >> 3
                if candMv[i] > 0:
                    candMv[i] = aInt << 3
                else:
                    candMv[i] = -(aInt << 3)
            else:
                if candMv[i] & 1:
                    if candMv[i] > 0:
                        candMv[i] -= 1
                    else:
                        candMv[i] += 1

    def _sorting_process(self, av1: AV1Decoder, start: int, end: int, isCompound: int):
        """
        排序过程
        规范文档 7.10.2.11 Sorting process

        此过程根据相应权重对运动向量栈的一部分执行稳定排序。
        从start（包含）到end（不包含）的RefStackMv中的条目被排序。

        Args:
            start: 起始位置
            end: 结束位置
            isCompound: 0表示单预测，1表示复合预测
        """
        def swap_stack(i: int, j: int):
            """
            交换栈条目
            规范文档 7.10.2.11 Sorting process

            当调用swap_stack函数时，位置idx和idx - 1的条目应该在WeightStack和RefStackMv中交换。

            Args:
                i: 第一个索引
                j: 第二个索引
            """
            temp = self._WeightStack[i]
            self._WeightStack[i] = self._WeightStack[j]
            self._WeightStack[j] = temp

            for list_idx in range(1 + isCompound):
                for comp in range(2):
                    temp = self._RefStackMv[i][list_idx][comp]
                    self._RefStackMv[i][list_idx][comp] = self._RefStackMv[j][list_idx][comp]
                    self._RefStackMv[j][list_idx][comp] = temp

        while end > start:
            newEnd = start
            for idx in range(start + 1, end):
                if self._WeightStack[idx - 1] < self._WeightStack[idx]:
                    swap_stack(idx - 1, idx)
                    newEnd = idx
            end = newEnd

    def _extra_search_process(self, av1: AV1Decoder, isCompound: int):
        """
        额外搜索过程
        规范文档 7.10.2.12 Extra search process

        输入是变量isCompound，包含0表示单预测，或1表示复合预测。
        此过程将额外的运动向量添加到RefStackMv，直到它有2个运动向量选择，
        首先搜索左侧和上方邻居以查找部分匹配的候选，然后添加全局运动候选。

        Args:
            isCompound: 0表示单预测，1表示复合预测
        """
        frame_header = av1.frame_header
        tile_group = av1.tile_group
        MiRows = frame_header.MiRows
        MiCols = frame_header.MiCols
        MiRow = tile_group.MiRow
        MiCol = tile_group.MiCol
        MiSize = tile_group.MiSize

        self._RefIdCount = [0, 0]
        self._RefDiffCount = [0, 0]

        w4 = min(16, Num_4x4_Blocks_Wide[MiSize])
        h4 = min(16, Num_4x4_Blocks_High[MiSize])

        w4 = min(w4, MiCols - MiCol)
        h4 = min(h4, MiRows - MiRow)

        num4x4 = min(w4, h4)

        for pass_idx in range(2):
            idx = 0
            while idx < num4x4 and tile_group.NumMvFound < 2:
                if pass_idx == 0:
                    mvRow = MiRow - 1
                    mvCol = MiCol + idx
                else:
                    mvRow = MiRow + idx
                    mvCol = MiCol - 1

                if not is_inside(av1, mvRow, mvCol):
                    break

                self._add_extra_mv_candidate_process(
                    av1, mvRow, mvCol, isCompound)

                if pass_idx == 0:
                    idx += Num_4x4_Blocks_Wide[tile_group.MiSizes[mvRow][mvCol]]
                else:
                    idx += Num_4x4_Blocks_High[tile_group.MiSizes[mvRow][mvCol]]

        if isCompound == 1:
            combinedMvs: List[List[List[int]]] = Array(None, (2, 2))

            for list_idx in range(2):
                compCount = 0

                for idx in range(self._RefIdCount[list_idx]):
                    combinedMvs[compCount][list_idx] = deepcopy(
                        self._RefIdMvs[list_idx][idx])
                    compCount += 1

                for idx in range(self._RefDiffCount[list_idx]):
                    if compCount >= 2:
                        break
                    combinedMvs[compCount][list_idx] = deepcopy(
                        self._RefDiffMvs[list_idx][idx])
                    compCount += 1

                while compCount < 2:
                    combinedMvs[compCount][list_idx] = deepcopy(
                        tile_group.GlobalMvs[list_idx])
                    compCount += 1

            if tile_group.NumMvFound == 1:
                if (combinedMvs[0][0] == self._RefStackMv[0][0] and
                        combinedMvs[0][1] == self._RefStackMv[0][1]):
                    self._RefStackMv[tile_group.NumMvFound][0] = deepcopy(
                        combinedMvs[1][0])
                    self._RefStackMv[tile_group.NumMvFound][1] = deepcopy(
                        combinedMvs[1][1])
                else:
                    self._RefStackMv[tile_group.NumMvFound][0] = deepcopy(
                        combinedMvs[0][0])
                    self._RefStackMv[tile_group.NumMvFound][1] = deepcopy(
                        combinedMvs[0][1])

                self._WeightStack[tile_group.NumMvFound] = 2
                tile_group.NumMvFound += 1
            else:
                for idx in range(2):
                    self._RefStackMv[tile_group.NumMvFound][0] = deepcopy(
                        combinedMvs[idx][0])
                    self._RefStackMv[tile_group.NumMvFound][1] = deepcopy(
                        combinedMvs[idx][1])
                    self._WeightStack[tile_group.NumMvFound] = 2
                    tile_group.NumMvFound += 1
        else:
            for idx in range(tile_group.NumMvFound, 2):
                self._RefStackMv[idx][0] = deepcopy(tile_group.GlobalMvs[0])

    def _add_extra_mv_candidate_process(self, av1: AV1Decoder, mvRow: int, mvCol: int, isCompound: int):
        """
        添加额外MV候选过程
        规范文档 7.10.2.13 Add extra MV candidate process

        此过程可能修改RefIdMvs、RefIdCount、RefDiffMvs、RefDiffCount、RefStackMv、WeightStack和NumMvFound的内容。

        Args:
            mvRow: 候选行位置
            mvCol: 候选列位置
            isCompound: 0表示单预测，1表示复合预测
        """
        frame_header = av1.frame_header
        tile_group = av1.tile_group
        ref_frame_store = av1.ref_frame_store

        if isCompound:
            for candList in range(2):
                candRef = tile_group.RefFrames[mvRow][mvCol][candList]

                if candRef > REF_FRAME.INTRA_FRAME:
                    for list_idx in range(2):
                        candMv = deepcopy(
                            tile_group.Mvs[mvRow][mvCol][candList])

                        if candRef == tile_group.RefFrame[list_idx] and self._RefIdCount[list_idx] < 2:
                            self._RefIdMvs[list_idx][self._RefIdCount[list_idx]] = deepcopy(
                                candMv)
                            self._RefIdCount[list_idx] += 1
                        elif self._RefDiffCount[list_idx] < 2:
                            if (ref_frame_store.RefFrameSignBias[candRef] != ref_frame_store.RefFrameSignBias[tile_group.RefFrame[list_idx]]):
                                candMv[0] *= -1
                                candMv[1] *= -1

                            self._RefDiffMvs[list_idx][self._RefDiffCount[list_idx]] = deepcopy(
                                candMv)
                            self._RefDiffCount[list_idx] += 1
        else:
            for candList in range(2):
                candRef = tile_group.RefFrames[mvRow][mvCol][candList]

                if candRef > REF_FRAME.INTRA_FRAME:
                    candMv = deepcopy(tile_group.Mvs[mvRow][mvCol][candList])

                    if (ref_frame_store.RefFrameSignBias[candRef] != ref_frame_store.RefFrameSignBias[tile_group.RefFrame[0]]):
                        candMv[0] *= -1
                        candMv[1] *= -1

                    # 检查candMv是否已在栈中
                    idx = 0
                    while idx < tile_group.NumMvFound:
                        if candMv == self._RefStackMv[idx][0]:
                            break
                        idx += 1

                    if idx == tile_group.NumMvFound:
                        self._RefStackMv[idx][0] = deepcopy(candMv)
                        self._WeightStack[idx] = 2
                        tile_group.NumMvFound += 1

    def _context_and_clamping_process(self, av1: AV1Decoder, isCompound: int, numNew: int):
        """
        上下文和裁剪过程
        规范文档 7.10.2.14 Context and clamping process

        Args:
            isCompound: 0表示单预测，1表示复合预测
            numNew: NEWMV候选数量

        """
        tile_group = av1.tile_group
        MiSize = tile_group.MiSize

        bw = Block_Width[MiSize]
        bh = Block_Height[MiSize]

        numLists = 2 if isCompound else 1

        for idx in range(tile_group.NumMvFound):
            z = 0

            if idx + 1 < tile_group.NumMvFound:
                w0 = self._WeightStack[idx]
                w1 = self._WeightStack[idx + 1]
                if w0 >= REF_CAT_LEVEL:
                    if w1 < REF_CAT_LEVEL:
                        z = 1
                else:
                    z = 2

            tile_group.DrlCtxStack[idx] = z

        for list_idx in range(numLists):
            for idx in range(tile_group.NumMvFound):
                refMv = self._RefStackMv[idx][list_idx]

                refMv[0] = clamp_mv_row(av1, refMv[0], MV_BORDER + bh * 8)
                refMv[1] = clamp_mv_col(av1, refMv[1], MV_BORDER + bw * 8)

                self._RefStackMv[idx][list_idx] = refMv

        if self._CloseMatches == 0:
            tile_group.NewMvContext = min(self._TotalMatches, 1)
            tile_group.RefMvContext = self._TotalMatches
        elif self._CloseMatches == 1:
            tile_group.NewMvContext = 3 - min(numNew, 1)
            tile_group.RefMvContext = 2 + self._TotalMatches
        else:
            tile_group.NewMvContext = 5 - min(numNew, 1)
            tile_group.RefMvContext = 5

    def find_warp_samples_process(self, av1: AV1Decoder):
        """
        查找warp样本过程
        规范文档 7.10.4 Find warp samples process

        当调用find_warp_samples函数时触发此过程。
        此过程检查相邻的帧间预测块，并根据运动向量估计局部warp变换。
        此过程产生变量NumSamples，包含找到的有效候选数量，以及数组CandList，包含排序的候选。

        """
        frame_header = av1.frame_header
        tile_group = av1.tile_group
        MiRows = frame_header.MiRows
        MiCols = frame_header.MiCols
        MiRow = tile_group.MiRow
        MiCol = tile_group.MiCol
        MiSize = tile_group.MiSize
        AvailU = tile_group.AvailU
        AvailL = tile_group.AvailL

        tile_group.NumSamples = 0

        tile_group.NumSamplesScanned = 0

        w4 = Num_4x4_Blocks_Wide[MiSize]

        h4 = Num_4x4_Blocks_High[MiSize]

        doTopLeft = 1

        doTopRight = 1

        if AvailU:
            srcSize = tile_group.MiSizes[MiRow - 1][MiCol]

            srcW = Num_4x4_Blocks_Wide[srcSize]

            if w4 <= srcW:
                colOffset = -(MiCol & (srcW - 1))
                if colOffset < 0:
                    doTopLeft = 0
                if colOffset + srcW > w4:
                    doTopRight = 0
                self._add_sample(av1, -1, 0)
            else:
                i = 0
                while i < min(w4, MiCols - MiCol):
                    srcSize = tile_group.MiSizes[MiRow - 1][MiCol + i]
                    srcW = Num_4x4_Blocks_Wide[srcSize]
                    miStep = min(w4, srcW)
                    self._add_sample(av1, -1, i)
                    i += miStep

        if AvailL:
            srcSize = tile_group.MiSizes[MiRow][MiCol - 1]
            srcH = Num_4x4_Blocks_High[srcSize]
            if h4 <= srcH:
                rowOffset = -(MiRow & (srcH - 1))
                if rowOffset < 0:
                    doTopLeft = 0
                self._add_sample(av1, 0, -1)
            else:
                i = 0
                while i < min(h4, MiRows - MiRow):
                    srcSize = tile_group.MiSizes[MiRow + i][MiCol - 1]
                    srcH = Num_4x4_Blocks_High[srcSize]
                    miStep = min(h4, srcH)
                    self._add_sample(av1, i, -1)
                    i += miStep

        if doTopLeft:
            self._add_sample(av1, -1, -1)

        if doTopRight:
            if max(w4, h4) <= 16 and has_top_right(av1):
                self._add_sample(av1, -1, w4)

        if tile_group.NumSamples == 0 and tile_group.NumSamplesScanned > 0:
            tile_group.NumSamples = 1

    def _add_sample(self, av1: AV1Decoder, deltaRow: int, deltaCol: int):
        """
        添加样本过程
        规范文档 7.10.4.2 Add sample process

        输入：
        - deltaRow: 指定（以4x4亮度样本为单位）在上方多远查找运动向量
        - deltaCol: 指定（以4x4亮度样本为单位）在左侧多远查找运动向量

        此过程的输出是将新样本添加到候选列表（如果它是有效候选且之前未见过）。

        Args:
            deltaRow: 行偏移（4x4单位）
            deltaCol: 列偏移（4x4单位）
        """
        tile_group = av1.tile_group
        MiRow = tile_group.MiRow
        MiCol = tile_group.MiCol
        MiSize = tile_group.MiSize

        if tile_group.NumSamplesScanned >= LEAST_SQUARES_SAMPLES_MAX:
            return

        mvRow = MiRow + deltaRow
        mvCol = MiCol + deltaCol

        if not is_inside(av1, mvRow, mvCol):
            return

        if not tile_group.RefFrames[mvRow][mvCol][0]:
            return

        if tile_group.RefFrames[mvRow][mvCol][0] != tile_group.RefFrame[0]:
            return

        if tile_group.RefFrames[mvRow][mvCol][1] != REF_FRAME.NONE:
            return

        candSz = tile_group.MiSizes[mvRow][mvCol]

        candW4 = Num_4x4_Blocks_Wide[candSz]

        candH4 = Num_4x4_Blocks_High[candSz]

        candRow = mvRow & ~(candH4 - 1)

        candCol = mvCol & ~(candW4 - 1)

        midY = candRow * 4 + candH4 * 2 - 1

        midX = candCol * 4 + candW4 * 2 - 1

        threshold = Clip3(16, 112, max(
            Block_Width[MiSize], Block_Height[MiSize]))

        mvDiffRow = abs(tile_group.Mvs[candRow]
                        [candCol][0][0] - tile_group.Mv[0][0])

        mvDiffCol = abs(tile_group.Mvs[candRow]
                        [candCol][0][1] - tile_group.Mv[0][1])

        valid = ((mvDiffRow + mvDiffCol) <= threshold)

        cand = [0] * 4
        cand[0] = midY * 8
        cand[1] = midX * 8
        cand[2] = midY * 8 + tile_group.Mvs[candRow][candCol][0][0]
        cand[3] = midX * 8 + tile_group.Mvs[candRow][candCol][0][1]

        # 1.
        tile_group.NumSamplesScanned += 1

        # 2.
        if not valid and tile_group.NumSamplesScanned > 1:
            return

        # 3.
        for j in range(4):
            tile_group.CandList[tile_group.NumSamples][j] = cand[j]

        # 4.
        if valid:
            tile_group.NumSamples += 1


def has_top_right(av1: AV1Decoder) -> bool:
    """
    !!! 这个方法待验证是否有复用
    检查右上角位置是否可用
    规范文档 7.10.4 Find warp samples process

    此函数检查位置 (MiRow - 1, MiCol + w4) 是否在边界内，用于判断是否可以添加右上角样本。

    Returns:
        如果右上角位置可用返回True，否则返回False
    """
    reader = av1.reader
    header = av1.obu.header
    seq_header = av1.seq_header
    frame_header = av1.frame_header
    tile_group = av1.tile_group
    decoder = av1.decoder
    mono_chrome = seq_header.color_config.mono_chrome
    use_128x128_superblock = seq_header.use_128x128_superblock
    subsampling_x = seq_header.color_config.subsampling_x
    subsampling_y = seq_header.color_config.subsampling_y
    BitDepth = seq_header.color_config.BitDepth
    NumPlanes = seq_header.color_config.NumPlanes
    frame_to_show_map_idx = frame_header.frame_to_show_map_idx
    OrderHint = frame_header.OrderHint
    OrderHints = frame_header.OrderHints
    ref_frame_idx = frame_header.ref_frame_idx
    MiRows = frame_header.MiRows
    MiCols = frame_header.MiCols
    film_grain_params = frame_header.film_grain_params
    gm_params = frame_header.gm_params
    MiRow = tile_group.MiRow
    MiCol = tile_group.MiCol
    MiSize = tile_group.MiSize
    motion_mode = tile_group.motion_mode
    compound_type = tile_group.compound_type
    HasChroma = tile_group.HasChroma
    AvailU = tile_group.AvailU
    AvailL = tile_group.AvailL
    YMode = tile_group.YMode

    bw4 = Num_4x4_Blocks_Wide[MiSize]
    bh4 = Num_4x4_Blocks_High[MiSize]
    bs = max(bw4, bh4)

    if bs > Num_4x4_Blocks_Wide[SUB_SIZE.BLOCK_64X64]:
        return False

    sb_size = SUB_SIZE.BLOCK_128X128 if use_128x128_superblock else SUB_SIZE.BLOCK_64X64
    sb_mi_size = Num_4x4_Blocks_Wide[sb_size]
    mask_row = MiRow & (sb_mi_size - 1)
    mask_col = MiCol & (sb_mi_size - 1)
    has_tr = not (mask_row & bs and mask_col & bs)

    assert (bs > 0 and not (bs & (bs - 1)))

    while bs < sb_mi_size:
        if mask_col & bs:
            if mask_col & (2 * bs) and mask_row & (2 * bs):
                has_tr = False
                break
        else:
            break
        bs <<= 1

    if bw4 < bh4:
        is_last_vertical_rect = 0
        if not ((MiCol + bw4) & (bh4 - 1)):
            is_last_vertical_rect = 1

        if not is_last_vertical_rect:
            has_tr = True

    if bw4 > bh4:
        is_first_horizontal_rect = 0
        if not (MiRow & (bw4 - 1)):
            is_first_horizontal_rect = 1

        if not is_first_horizontal_rect:
            has_tr = False

    if tile_group.partition == PARTITION.PARTITION_VERT_A:
        if bw4 == bh4:
            if mask_row & bs:
                has_tr = False
    return has_tr


def has_overlappable_candidates(av1: AV1Decoder) -> int:
    """
    检查是否存在可重叠的候选
    规范文档 7.10.3 Has overlappable candidates
    """
    frame_header = av1.frame_header
    tile_group = av1.tile_group
    MiRows = frame_header.MiRows
    MiCols = frame_header.MiCols
    MiRow = tile_group.MiRow
    MiCol = tile_group.MiCol
    MiSize = tile_group.MiSize
    AvailU = tile_group.AvailU
    AvailL = tile_group.AvailL

    if AvailU:
        w4 = Num_4x4_Blocks_Wide[MiSize]
        for x4 in range(MiCol, min(MiCols, MiCol + w4), 2):
            if tile_group.RefFrames[MiRow - 1][x4 | 1][0] > REF_FRAME.INTRA_FRAME:
                return 1

    if AvailL:
        h4 = Num_4x4_Blocks_High[MiSize]
        for y4 in range(MiRow, min(MiRows, MiRow + h4), 2):
            if tile_group.RefFrames[y4 | 1][MiCol - 1][0] > REF_FRAME.INTRA_FRAME:
                return 1

    return 0
