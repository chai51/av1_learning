

from copy import deepcopy
from constants import FRAME_TYPE, MAX_FRAME_DISTANCE, MAX_OFFSET_HEIGHT, MAX_OFFSET_WIDTH, MAX_TILE_COLS, MAX_TILE_ROWS, MFMV_STACK_SIZE, MI_SIZE_LOG2, NUM_REF_FRAMES, REF_FRAME, Div_Mult
from obu.decoder import AV1Decoder
from typing import List
from utils.math_utils import Array, Clip3, Round2Signed


class MotionFieldEstimation:
    def __init__(self, av1: AV1Decoder):
        frame_header = av1.frame_header
        MiCols = frame_header.MiCols
        MiRows = frame_header.MiRows

        self._MotionFieldMvs: List[List[List[List[int]]]] = Array(
            None, (NUM_REF_FRAMES, MiRows >> 1, MiCols >> 1, 2), 0)
        self._PosY8 = 0
        self._PosX8 = 0

    def motion_field_estimation(self, av1: AV1Decoder) -> List[List[List[List[int]]]]:
        """
        运动场估计过程
        规范文档 7.9 Motion field estimation process
        """
        from utils.frame_utils import get_relative_dist
        frame_header = av1.frame_header
        ref_frame_store = av1.ref_frame_store
        OrderHint = frame_header.OrderHint
        OrderHints = frame_header.OrderHints
        ref_frame_idx = frame_header.ref_frame_idx
        MiRows = frame_header.MiRows
        MiCols = frame_header.MiCols

        w8 = MiCols >> 1
        h8 = MiRows >> 1

        for ref in range(REF_FRAME.LAST_FRAME, REF_FRAME.ALTREF_FRAME + 1):
            for y in range(h8):
                for x in range(w8):
                    for j in range(2):
                        self._MotionFieldMvs[ref][y][x][j] = -1 << 15

        lastIdx = ref_frame_idx[0]
        curGoldOrderHint = OrderHints[REF_FRAME.GOLDEN_FRAME]
        lastAltOrderHint = ref_frame_store.SavedOrderHints[lastIdx][REF_FRAME.ALTREF_FRAME]
        useLast = lastAltOrderHint != curGoldOrderHint

        if useLast == 1:
            self.projection(av1, REF_FRAME.LAST_FRAME, -1)

        refStamp = MFMV_STACK_SIZE - 2

        useBwd = get_relative_dist(
            av1, OrderHints[REF_FRAME.BWDREF_FRAME], OrderHint) > 0
        if useBwd == 1:
            projOutput = self.projection(av1, REF_FRAME.BWDREF_FRAME, 1)
            if projOutput == 1:
                refStamp = refStamp - 1

        useAlt2 = get_relative_dist(
            av1, OrderHints[REF_FRAME.ALTREF2_FRAME], OrderHint) > 0
        if useAlt2 == 1:
            projOutput = self.projection(av1, REF_FRAME.ALTREF2_FRAME, 1)
            if projOutput == 1:
                refStamp = refStamp - 1

        useAlt = get_relative_dist(
            av1, OrderHints[REF_FRAME.ALTREF_FRAME], OrderHint) > 0
        if useAlt == 1 and refStamp >= 0:
            projOutput = self.projection(av1, REF_FRAME.ALTREF_FRAME, 1)
            if projOutput == 1:
                refStamp = refStamp - 1

        if refStamp >= 0:
            self.projection(av1, REF_FRAME.LAST2_FRAME, -1)

        return self._MotionFieldMvs

    def projection(self, av1: AV1Decoder, src: REF_FRAME, dstSign: int) -> int:
        """
        投影过程
        规范文档 7.9.2 Projection process

        Args:
            src: 源帧索引
            dstSign: 目标帧符号

        Returns:
            projOutput - 投影输出标志
        """
        frame_header = av1.frame_header
        tile_group = av1.tile_group
        ref_frame_store = av1.ref_frame_store
        OrderHint = frame_header.OrderHint
        OrderHints = frame_header.OrderHints
        ref_frame_idx = frame_header.ref_frame_idx
        MiRows = frame_header.MiRows
        MiCols = frame_header.MiCols

        srcIdx = ref_frame_idx[src - REF_FRAME.LAST_FRAME]
        w8 = MiCols >> 1
        h8 = MiRows >> 1

        if ((ref_frame_store.RefMiRows[srcIdx] != MiRows and ref_frame_store.RefMiCols[srcIdx] != MiCols) or
                ref_frame_store.RefFrameType[srcIdx] == FRAME_TYPE.INTRA_ONLY_FRAME or ref_frame_store.RefFrameType[srcIdx] == FRAME_TYPE.KEY_FRAME):
            return 0

        for y8 in range(h8):
            for x8 in range(w8):
                row = 2 * y8 + 1
                col = 2 * x8 + 1

                srcRef = ref_frame_store.SavedRefFrames[srcIdx][row][col]
                if srcRef > REF_FRAME.INTRA_FRAME:
                    from utils.frame_utils import get_relative_dist
                    refToCur = get_relative_dist(
                        av1, OrderHints[src], OrderHint)
                    refOffset = get_relative_dist(
                        av1, OrderHints[src], ref_frame_store.SavedOrderHints[srcIdx][srcRef])
                    posValid = abs(refToCur) <= MAX_FRAME_DISTANCE and abs(
                        refOffset) <= MAX_FRAME_DISTANCE and refOffset > 0
                    if posValid:
                        mv = deepcopy(
                            ref_frame_store.SavedMvs[srcIdx][row][col])
                        projMv = self.get_mv_projection(
                            av1, mv, refToCur * dstSign, refOffset)
                        posValid = self.get_block_position(
                            av1, x8, y8, dstSign, projMv)
                        if posValid:
                            for dst in range(REF_FRAME.LAST_FRAME, REF_FRAME.ALTREF_FRAME + 1):
                                refToDst = get_relative_dist(
                                    av1, OrderHint, OrderHints[dst])
                                projMv = self.get_mv_projection(
                                    av1, mv, refToDst, refOffset)
                                self._MotionFieldMvs[dst][self._PosY8][self._PosX8] = deepcopy(
                                    projMv)

        return 1

    def get_mv_projection(self, av1: AV1Decoder, mv: List[int], numerator: int, denominator: int) -> List[int]:
        """
        获取MV投影过程
        规范文档 7.9.3 Get MV projection process

        Args:
            mv: 运动向量 [mv[0], mv[1]]
            refToCur: 参考帧到当前帧的距离
            refOffset: 参考偏移

        Returns:
            投影后的运动向量 [projMv[0], projMv[1]]
        """
        clippedDenominator = min(MAX_FRAME_DISTANCE, denominator)
        clippedNumerator = Clip3(-MAX_FRAME_DISTANCE,
                                 MAX_FRAME_DISTANCE, numerator)

        projMv = [0, 0]
        for i in range(2):
            scaled = Round2Signed(
                mv[i] * clippedNumerator * Div_Mult[clippedDenominator], 14)
            projMv[i] = Clip3(-(1 << 14) + 1, (1 << 14) - 1, scaled)
        return projMv

    def get_block_position(self, av1: AV1Decoder, x8: int, y8: int, dstSign: int, projMv: List[int]) -> int:
        """
        获取块位置过程
        规范文档 7.9.4 Get block position process

        Args:
            x8: X坐标（8x8单位）
            y8: Y坐标（8x8单位）
            dstSign: 目标帧符号
            projMv: 投影后的运动向量

        Returns:
            posValid - 位置是否有效
        """
        frame_header = av1.frame_header
        MiRows = frame_header.MiRows
        MiCols = frame_header.MiCols

        posValid = 1

        def project(v8: int, delta: int, dstSign: int, max8: int, maxOff8: int) -> int:
            nonlocal posValid
            base8 = (v8 >> 3) << 3
            if delta >= 0:
                offset8 = delta >> (3 + 1 + MI_SIZE_LOG2)
            else:
                offset8 = -((-delta) >> (3 + 1 + MI_SIZE_LOG2))
            v8 += dstSign * offset8
            if v8 < 0 or v8 >= max8 or v8 < base8 - maxOff8 or v8 >= base8 + 8 + maxOff8:
                posValid = 0
            return v8

        self._PosY8 = project(
            y8, projMv[0], dstSign, MiRows >> 1, MAX_OFFSET_HEIGHT)
        self._PosX8 = project(
            x8, projMv[1], dstSign, MiCols >> 1, MAX_OFFSET_WIDTH)

        return posValid
