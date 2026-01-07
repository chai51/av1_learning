"""
环路滤波过程模块
实现规范文档7.14节"Loop filter process"中描述的所有环路滤波过程函数
包括：
- 7.14 Loop filter process
- 7.14.1 Loop filter edge process
- 7.14.2 Filter level process
- 7.14.3 Filter select process
- 7.14.4 Filter strength process
- 7.14.5 Normal filter process
- 7.14.6 Wide filter process
  - 7.14.6.1 Wide filter tap process
  - 7.14.6.2 Wide filter tap 2 process
  - 7.14.6.3 Wide filter tap 3 process
  - 7.14.6.4 Wide filter process
"""

from typing import List
from constants import (
    MI_SIZE, MAX_LOOP_FILTER, NONE,
    SEG_LVL_ALT_LF_Y_V, REF_FRAME, Y_MODE,
    Block_Width, Block_Height, Tx_Width, Tx_Height
)
from obu.decoder import AV1Decoder
from utils.math_utils import Clip3, Round2
from utils.tile_utils import get_plane_residual_size
from utils.tile_utils import seg_feature_active_idx


class LoopFilter:
    def __init__(self):
        self._F: List[int] = [NONE] * 12
        pass

    def loop_filter_process(self, av1: AV1Decoder):
        """
        环路滤波过程
        规范文档 7.14 Loop filter process

        此过程对CurrFrame应用去块效应滤波。
        """
        seq_header = av1.seq_header
        frame_header = av1.frame_header
        subsampling_x = seq_header.color_config.subsampling_x
        subsampling_y = seq_header.color_config.subsampling_y
        NumPlanes = seq_header.color_config.NumPlanes
        MiRows = frame_header.MiRows
        MiCols = frame_header.MiCols

        for plane in range(NumPlanes):
            if plane == 0 or frame_header.loop_filter_level[1 + plane]:
                for pass_idx in range(2):
                    rowStep = 1 if plane == 0 else (1 << subsampling_y)
                    colStep = 1 if plane == 0 else (1 << subsampling_x)

                    for row in range(0, MiRows, rowStep):
                        for col in range(0, MiCols, colStep):
                            self.loop_filter_edge(
                                av1, plane, pass_idx, row, col)

    def loop_filter_edge(self, av1: AV1Decoder, plane: int, pass_idx: int, row: int, col: int):
        """
        环路滤波边缘过程
        规范文档 7.14.2 Edge loop filter process

        此过程对指定边缘应用去块效应滤波。

        Args:
            plane: 平面索引（0=Y, 1=U, 2=V）
            pass_idx: 通道索引（0=垂直边缘，1=水平边缘）
            row: MI行索引
            col: MI列索引
        """
        seq_header = av1.seq_header
        frame_header = av1.frame_header
        tile_group = av1.tile_group
        subsampling_x = seq_header.color_config.subsampling_x
        subsampling_y = seq_header.color_config.subsampling_y

        if plane == 0:
            subX = 0
            subY = 0
        else:
            subX = subsampling_x
            subY = subsampling_y

        dx = 0
        dy = 1
        if pass_idx == 0:
            dx = 1
            dy = 0
        else:
            dy = 1
            dx = 0

        x = col * MI_SIZE
        y = row * MI_SIZE
        row = row | subY
        col = col | subX

        if x >= frame_header.FrameWidth:
            onScreen = 0
        elif y >= frame_header.FrameHeight:
            onScreen = 0
        elif pass_idx == 0 and x == 0:
            onScreen = 0
        elif pass_idx == 1 and y == 0:
            onScreen = 0
        else:
            onScreen = 1

        if onScreen == 0:
            return

        xP = x >> subX
        yP = y >> subY

        prevRow = row - (dy << subY)
        prevCol = col - (dx << subX)

        tile_group.MiSize = tile_group.MiSizes[row][col]

        txSz = tile_group.LoopfilterTxSizes[plane][row >> subY][col >> subX]

        planeSize = get_plane_residual_size(av1, tile_group.MiSize, plane)

        tile_group.skip = tile_group.Skips[row][col]

        isIntra = tile_group.RefFrames[row][col][0] <= REF_FRAME.INTRA_FRAME

        prevTxSz = tile_group.LoopfilterTxSizes[plane][prevRow >>
                                                       subY][prevCol >> subX]

        if pass_idx == 0 and xP % Block_Width[planeSize] == 0:
            isBlockEdge = 1
        elif pass_idx == 1 and yP % Block_Height[planeSize] == 0:
            isBlockEdge = 1
        else:
            isBlockEdge = 0

        if pass_idx == 0 and xP % Tx_Width[txSz] == 0:
            isTxEdge = 1
        elif pass_idx == 1 and yP % Tx_Height[txSz] == 0:
            isTxEdge = 1
        else:
            isTxEdge = 0

        if isTxEdge == 0:
            applyFilter = 0
        elif isBlockEdge == 1 or tile_group.skip == 0 or isIntra == 1:
            applyFilter = 1
        else:
            applyFilter = 0

        filterSize = self.filter_size(txSz, prevTxSz, pass_idx, plane)

        lvl, limit, blimit, thresh = self.adaptive_filter_strength(
            av1, row, col, plane, pass_idx)

        if lvl == 0:
            lvl, limit, blimit, thresh = self.adaptive_filter_strength(
                av1, prevRow, prevCol, plane, pass_idx)

        for i in range(MI_SIZE):
            if applyFilter == 1 and lvl > 0:
                self.sample_filtering(
                    av1, xP + dy * i, yP + dx * i, plane, limit, blimit, thresh, dx, dy, filterSize)

    def filter_size(self, txSz: int, prevTxSz: int, pass_idx: int, plane: int) -> int:
        """
        滤波尺寸过程
        规范文档 7.14.3 Filter size process

        Args:
            txSz: 当前变换尺寸
            prevTxSz: 前一个变换尺寸
            pass_idx: 通道索引
            plane: 平面索引

        Returns:
            滤波尺寸
        """

        if pass_idx == 0:
            baseSize = min(Tx_Width[prevTxSz], Tx_Width[txSz])
        else:
            baseSize = min(Tx_Height[prevTxSz], Tx_Height[txSz])

        if plane == 0:
            filterSize = min(16, baseSize)
        else:
            filterSize = min(8, baseSize)

        return filterSize

    def adaptive_filter_strength(self, av1: AV1Decoder, row: int, col: int, plane: int, pass_idx: int) -> tuple[int, int, int, int]:
        """
        自适应滤波强度过程
        规范文档 7.14.4 Adaptive filter strength process

        Args:
            row: MI行索引
            col: MI列索引
            plane: 平面索引
            pass_idx: 通道索引

        Returns:
            包含lvl, limit, blimit, thresh的字典
        """
        frame_header = av1.frame_header
        tile_group = av1.tile_group

        segment = tile_group.SegmentIds[row][col]

        ref = tile_group.RefFrames[row][col][0]

        mode = tile_group.YModes[row][col]

        # 1.
        if mode >= Y_MODE.NEARESTMV and mode != Y_MODE.GLOBALMV and mode != Y_MODE.GLOBAL_GLOBALMV:
            modeType = 1
        # 2.
        else:
            modeType = 0

        # 1.
        if frame_header.delta_lf_multi == 0:
            deltaLF = tile_group.DeltaLFs[row][col][0]
        else:
            deltaLF = tile_group.DeltaLFs[row][col][pass_idx if plane ==
                                                    0 else plane + 1]

        lvl = self.adaptive_filter_strength_selection(
            av1, segment, ref, modeType, deltaLF, plane, pass_idx)

        if frame_header.loop_filter_sharpness > 4:
            shift = 2
        elif frame_header.loop_filter_sharpness > 0:
            shift = 1
        else:
            shift = 0

        if frame_header.loop_filter_sharpness > 0:
            limit = Clip3(
                1, 9 - frame_header.loop_filter_sharpness, lvl >> shift)
        else:
            limit = max(1, lvl >> shift)

        blimit = 2 * (lvl + 2) + limit
        thresh = lvl >> 4

        return lvl, limit, blimit, thresh

    def adaptive_filter_strength_selection(self, av1: AV1Decoder, segment: int, ref: REF_FRAME, modeType: int,
                                           deltaLF: int, plane: int, pass_idx: int) -> int:
        """
        自适应滤波强度选择过程
        规范文档 7.14.5 Adaptive filter strength selection process

        Args:
            segment: Segment ID
            ref: 参考帧
            modeType: 模式类型
            deltaLF: Delta LF值
            plane: 平面索引
            pass_idx: 通道索引

        Returns:
            滤波级别
        """
        frame_header = av1.frame_header

        i = pass_idx if plane == 0 else plane + 1
        baseFilterLevel = Clip3(
            0, MAX_LOOP_FILTER, deltaLF + frame_header.loop_filter_level[i])

        # 1.
        lvlSeg = baseFilterLevel

        # 2.
        feature = SEG_LVL_ALT_LF_Y_V + i

        # 3.
        if seg_feature_active_idx(av1, segment, feature):
            # a.
            lvlSeg = frame_header.FeatureData[segment][feature] + lvlSeg

            # b.
            lvlSeg = Clip3(0, MAX_LOOP_FILTER, lvlSeg)

        # 4.
        if frame_header.loop_filter_delta_enabled == 1:
            # a.
            nShift = lvlSeg >> 5

            # b.
            if ref == REF_FRAME.INTRA_FRAME:
                lvlSeg = (lvlSeg +
                          (frame_header.loop_filter_ref_deltas[REF_FRAME.INTRA_FRAME] << nShift))

            # c.
            elif ref != REF_FRAME.INTRA_FRAME:
                lvlSeg = lvlSeg + (frame_header.loop_filter_ref_deltas[ref] << nShift) + (
                    frame_header.loop_filter_mode_deltas[modeType] << nShift)

            # d.
            lvlSeg = Clip3(0, MAX_LOOP_FILTER, lvlSeg)

        # 5.
        return lvlSeg

    def sample_filtering(self, av1: AV1Decoder, x: int, y: int, plane: int, limit: int, blimit: int,
                         thresh: int, dx: int, dy: int, filterSize: int):
        """
        样本滤波过程
        规范文档 7.14.6 Sample filtering process

        Args:
            x: X坐标（样本单位）
            y: Y坐标（样本单位）
            plane: 平面索引
            limit: limit值
            blimit: blimit值
            thresh: thresh值
            dx: X方向增量
            dy: Y方向增量
            filterSize: 滤波尺寸
        """
        hevMask, filterMask, flatMask, flatMask2 = self.filter_mask(
            av1, x, y, plane, limit, blimit, thresh, dx, dy, filterSize)

        if filterMask == 0:
            pass
        elif filterSize == 4 or flatMask == 0:
            self.narrow_filter(av1, hevMask, x, y, plane, dx, dy)
        elif filterSize == 8 or flatMask2 == 0:
            self.wide_filter(av1, x, y, plane, dx, dy, 3)
        else:
            self.wide_filter(av1, x, y, plane, dx, dy, 4)

    def filter_mask(self, av1: AV1Decoder, x: int, y: int, plane: int, limit: int, blimit: int,
                    thresh: int, dx: int, dy: int, filterSize: int) -> tuple[int, int, int, int]:
        """
        滤波掩码过程
        规范文档 7.14.6.2 Filter mask process

        Args:
            x: X坐标
            y: Y坐标
            plane: 平面索引
            limit: limit值
            blimit: blimit值
            thresh: thresh值
            dx: X方向增量
            dy: Y方向增量
            filterSize: 滤波尺寸

        Returns:
            包含hevMask, filterMask, flatMask, flatMask2的字典
        """
        seq_header = av1.seq_header
        BitDepth = seq_header.color_config.BitDepth

        q0 = av1.CurrFrame[plane][y][x]
        q1 = av1.CurrFrame[plane][y + dy][x + dx]
        q2 = av1.CurrFrame[plane][y + dy * 2][x + dx * 2]
        q3 = av1.CurrFrame[plane][y + dy * 3][x + dx * 3]
        p0 = av1.CurrFrame[plane][y - dy][x - dx]
        p1 = av1.CurrFrame[plane][y - dy * 2][x - dx * 2]
        p2 = av1.CurrFrame[plane][y - dy * 3][x - dx * 3]
        p3 = av1.CurrFrame[plane][y - dy * 4][x - dx * 4]

        hevMask = 0
        threshBd = thresh << (BitDepth - 8)
        hevMask |= abs(p1 - p0) > threshBd
        hevMask |= abs(q1 - q0) > threshBd

        if filterSize == 4:
            filterLen = 4
        elif plane != 0:
            filterLen = 6
        elif filterSize == 8:
            filterLen = 8
        else:
            filterLen = 16

        limitBd = limit << (BitDepth - 8)
        blimitBd = blimit << (BitDepth - 8)
        mask = 0
        mask |= abs(p1 - p0) > limitBd
        mask |= abs(q1 - q0) > limitBd
        mask |= abs(p0 - q0) * 2 + (abs(p1 - q1) // 2) > blimitBd
        if filterLen >= 6:
            mask |= abs(p2 - p1) > limitBd
            mask |= abs(q2 - q1) > limitBd
        if filterLen >= 8:
            mask |= abs(p3 - p2) > limitBd
            mask |= abs(q3 - q2) > limitBd
        filterMask = mask == 0

        flatMask: int = NONE
        thresholdBd = 1 << (BitDepth - 8)
        if filterSize >= 8:
            mask = 0
            mask |= abs(p1 - p0) > thresholdBd
            mask |= abs(q1 - q0) > thresholdBd
            mask |= abs(p2 - p0) > thresholdBd
            mask |= abs(q2 - q0) > thresholdBd
            if filterLen >= 8:
                mask |= abs(p3 - p0) > thresholdBd
                mask |= abs(q3 - q0) > thresholdBd
            flatMask = mask == 0

        flatMask2: int = NONE
        thresholdBd = 1 << (BitDepth - 8)
        if filterSize >= 16:
            q4 = av1.CurrFrame[plane][y + dy * 4][x + dx * 4]
            q5 = av1.CurrFrame[plane][y + dy * 5][x + dx * 5]
            q6 = av1.CurrFrame[plane][y + dy * 6][x + dx * 6]
            p4 = av1.CurrFrame[plane][y - dy * 5][x - dx * 5]
            p5 = av1.CurrFrame[plane][y - dy * 6][x - dx * 6]
            p6 = av1.CurrFrame[plane][y - dy * 7][x - dx * 7]
            mask = 0
            mask |= abs(p6 - p0) > thresholdBd
            mask |= abs(q6 - q0) > thresholdBd
            mask |= abs(p5 - p0) > thresholdBd
            mask |= abs(q5 - q0) > thresholdBd
            mask |= abs(p4 - p0) > thresholdBd
            mask |= abs(q4 - q0) > thresholdBd
            flatMask2 = mask == 0

        return hevMask, filterMask, flatMask, flatMask2

    def narrow_filter(self, av1: AV1Decoder, hevMask: int, x: int, y: int, plane: int, dx: int, dy: int):
        """
        窄滤波过程
        规范文档 7.14.6.3 Narrow filter process

        Args:
            hevMask: HEV掩码
            x: X坐标
            y: Y坐标
            plane: 平面索引
            dx: X方向增量
            dy: Y方向增量
        """
        seq_header = av1.seq_header
        BitDepth = seq_header.color_config.BitDepth

        q0 = av1.CurrFrame[plane][y][x]
        q1 = av1.CurrFrame[plane][y + dy][x + dx]
        p0 = av1.CurrFrame[plane][y - dy][x - dx]
        p1 = av1.CurrFrame[plane][y - dy * 2][x - dx * 2]
        ps1 = p1 - (0x80 << (BitDepth - 8))
        ps0 = p0 - (0x80 << (BitDepth - 8))
        qs0 = q0 - (0x80 << (BitDepth - 8))
        qs1 = q1 - (0x80 << (BitDepth - 8))

        def filter4_clamp(value: int) -> int:
            """
            滤波4裁剪函数
            规范文档 7.14.6.3 Narrow filter process - filter4_clamp()

            Args:
                value: 要裁剪的值

            Returns:
                裁剪后的值
            """
            BitDepth = seq_header.color_config.BitDepth

            return Clip3(-(1 << (BitDepth - 1)), (1 << (BitDepth - 1)) - 1, value)

        filter = filter4_clamp(ps1 - qs1) if hevMask else 0
        filter = filter4_clamp(filter + 3 * (qs0 - ps0))
        filter1 = filter4_clamp(filter + 4) >> 3
        filter2 = filter4_clamp(filter + 3) >> 3
        oq0 = filter4_clamp(qs0 - filter1) + (0x80 << (BitDepth - 8))
        op0 = filter4_clamp(ps0 + filter2) + (0x80 << (BitDepth - 8))
        av1.CurrFrame[plane][y][x] = oq0
        av1.CurrFrame[plane][y - dy][x - dx] = op0
        if not hevMask:
            filter = Round2(filter1, 1)
            oq1 = filter4_clamp(qs1 - filter) + (0x80 << (BitDepth - 8))
            op1 = filter4_clamp(ps1 + filter) + (0x80 << (BitDepth - 8))
            av1.CurrFrame[plane][y + dy][x + dx] = oq1
            av1.CurrFrame[plane][y - dy * 2][x - dx * 2] = op1

    def wide_filter(self, av1: AV1Decoder, x: int, y: int, plane: int, dx: int, dy: int, log2Size: int):
        """
        宽滤波过程
        规范文档 7.14.6.4 Wide filter process

        Args:
            x: X坐标
            y: Y坐标
            plane: 平面索引
            dx: X方向增量
            dy: Y方向增量
            log2Size: 对数尺寸
        """

        if log2Size == 4:
            n = 6
        elif plane == 0:
            n = 3
        else:
            n = 2

        if log2Size == 3 and plane == 0:
            n2 = 0
        else:
            n2 = 1

        for i in range(-n, n):
            t = 0
            for j in range(-n, n + 1):
                p = Clip3(-(n + 1), n, i + j)
                tap = 2 if abs(j) <= n2 else 1
                t += av1.CurrFrame[plane][y + p * dy][x + p * dx] * tap
            self._F[i] = Round2(t, log2Size)

        for i in range(-n, n):
            av1.CurrFrame[plane][y + i * dy][x + i * dx] = self._F[i]
