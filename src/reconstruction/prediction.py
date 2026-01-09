"""
预测过程模块
实现规范文档7.11节"Prediction processes"中描述的所有预测过程函数
包括：
- 7.11.1 Intra prediction process
- 7.11.2 Inter prediction process
- 7.11.3 Compound prediction process
- 7.11.4 Inter-intra prediction process
- 7.11.5 Predict chroma from luma process
"""

from copy import deepcopy
from typing import List, TYPE_CHECKING, Optional
from math import ceil
from constants import (
    COMPOUND_TYPE,
    DIV_LUT_BITS,
    DIV_LUT_PREC_BITS,
    FILTER_BITS,
    GM_TYPE,
    INTERPOLATION_FILTER,
    LS_MV_MAX,
    MASK_MASTER_SIZE,
    MAX_FRAME_DISTANCE,
    MAX_SB_SIZE,
    MAX_TILE_COLS,
    MAX_TILE_ROWS,
    MOTION_MODE,
    NONE,
    NUM_REF_FRAMES,
    PLANE_MAX,
    REF_FRAME,
    REF_SCALE_SHIFT,
    Y_MODE,
    SCALE_SUBPEL_BITS,
    SUB_SIZE,
    SUBPEL_BITS,
    SUBPEL_MASK,
    WARP_PARAM_REDUCE_BITS,
    WARPEDDIFF_PREC_BITS,
    WARPEDMODEL_NONDIAGAFFINE_CLAMP,
    WARPEDMODEL_PREC_BITS,
    WARPEDMODEL_TRANS_CLAMP,
    WARPEDPIXEL_PREC_SHIFTS,
    WEDGE_HORIZONTAL,
    WEDGE_OBLIQUE117,
    WEDGE_OBLIQUE153,
    WEDGE_OBLIQUE27,
    WEDGE_OBLIQUE63,
    WEDGE_TYPES,
    WEDGE_VERTICAL,
    Y_MODE,
    INTERINTRA_MODE,
    ANGLE_STEP, INTRA_FILTER_SCALE_BITS, INTRA_EDGE_TAPS,
    Block_Height,
    Block_Width,
    Div_Lut,
    Ii_Weights_1d,
    Intra_Edge_Kernel,
    Mi_Height_Log2,
    Mi_Width_Log2,
    Mode_To_Angle, Dr_Intra_Derivative,
    Num_4x4_Blocks_High,
    Num_4x4_Blocks_Wide,
    Obmc_Mask_16,
    Obmc_Mask_2,
    Obmc_Mask_32,
    Obmc_Mask_4,
    Obmc_Mask_8,
    Quant_Dist_Lookup,
    Quant_Dist_Weight, Sm_Weights_Tx_4x4,
    Sm_Weights_Tx_8x8, Sm_Weights_Tx_16x16, Sm_Weights_Tx_32x32,
    Sm_Weights_Tx_64x64, Intra_Filter_Taps, MI_SIZE,
    Subpel_Filters,
    Tx_Width, Tx_Height, Tx_Width_Log2, Tx_Height_Log2,
    Warped_Filters,
    Wedge_Bits,
    Wedge_Codebook,
    Wedge_Master_Oblique_Even,
    Wedge_Master_Oblique_Odd,
    Wedge_Master_Vertical
)
from obu.decoder import AV1Decoder
from utils.math_utils import Array
from utils.tile_utils import is_directional_mode
from utils.tile_utils import get_plane_residual_size
from utils.tile_utils import is_scaled
from utils.math_utils import bits_signed, Clip1, Clip3, FloorLog2, Round2, Round2Signed
if TYPE_CHECKING:
    from frame.frame_header import FrameHeader


class Prediction:
    def __init__(self, av1: AV1Decoder):
        width = av1.seq_header.max_frame_width_minus_1 + 1
        height = av1.seq_header.max_frame_height_minus_1 + 1

        self._AboveRow: List[int] = NONE
        self._LeftCol: List[int] = NONE
        self._LocalValid = 0
        self._LocalWarpParams: List[int] = [0] * 6
        self._FwdWeight = 0
        self._BckWeight = 0

        self._Mask: List[List[int]] = Array(
            None, (MAX_TILE_ROWS, MAX_TILE_COLS), 0)
        self._WedgeMasks: List[List[List[List[List[int]]]]] = Array(
            None, (SUB_SIZE.BLOCK_SIZES, 2, WEDGE_TYPES, MASK_MASTER_SIZE, 128), 0)

    def predict_intra(self, av1: AV1Decoder, plane: int, x: int, y: int,
                      haveLeft: Optional[int], haveAbove: Optional[int],
                      haveAboveRight: int, haveBelowLeft: int,
                      mode: Y_MODE, log2W: int, log2H: int):
        """
        帧内预测过程
        规范文档 7.11.2 Intra prediction process

        此过程生成帧内预测块。

        Args:
            plane: 平面索引
            x: X坐标
            y: Y坐标
            haveLeft: 左侧可用标志
            haveAbove: 上方可用标志
            haveAboveRight: 上方右可用标志
            haveBelowLeft: 下方左可用标志
            mode: 预测模式
            log2W: 宽度对数
            log2H: 高度对数
        """
        seq_header = av1.seq_header
        frame_header = av1.frame_header
        tile_group = av1.tile_group
        subsampling_x = seq_header.color_config.subsampling_x
        subsampling_y = seq_header.color_config.subsampling_y
        BitDepth = seq_header.color_config.BitDepth
        MiRows = frame_header.MiRows
        MiCols = frame_header.MiCols

        w = 1 << log2W

        h = 1 << log2H

        maxX = (MiCols * MI_SIZE) - 1
        maxY = (MiRows * MI_SIZE) - 1
        if plane > 0:
            maxX = ((MiCols * MI_SIZE) >> subsampling_x) - 1
            maxY = ((MiRows * MI_SIZE) >> subsampling_y) - 1

        self._AboveRow = Array(None, (w + h) * 2)
        self._LeftCol = Array(None, (w + h) * 2)
        for i in range(w + h):
            if haveAbove == 0 and haveLeft == 1:
                self._AboveRow[i] = av1.CurrFrame[plane][y][x - 1]
            elif haveAbove == 0 and haveLeft == 0:
                self._AboveRow[i] = (1 << (BitDepth - 1)) - 1
            else:
                aboveLimit = min(
                    maxX, x + (2 * w if haveAboveRight else w) - 1)
                self._AboveRow[i] = av1.CurrFrame[plane][y -
                                                         1][min(aboveLimit, x + i)]

        for i in range(w + h):
            if haveLeft == 0 and haveAbove == 1:
                self._LeftCol[i] = av1.CurrFrame[plane][y - 1][x]
            elif haveLeft == 0 and haveAbove == 0:
                self._LeftCol[i] = (1 << (BitDepth - 1)) + 1
            else:
                leftLimit = min(maxY, y + (2 * h if haveBelowLeft else h) - 1)
                self._LeftCol[i] = av1.CurrFrame[plane][min(
                    leftLimit, y + i)][x - 1]

        if haveAbove == 1 and haveLeft == 1:
            self._AboveRow[-1] = av1.CurrFrame[plane][y - 1][x - 1]
        elif haveAbove == 1:
            self._AboveRow[-1] = av1.CurrFrame[plane][y - 1][x]
        elif haveLeft == 1:
            self._AboveRow[-1] = av1.CurrFrame[plane][y][x - 1]
        else:
            self._AboveRow[-1] = 1 << (BitDepth - 1)
        self._LeftCol[-1] = self._AboveRow[-1]

        if plane == 0 and tile_group.use_filter_intra:
            pred = self._recursive_intra_prediction_process(av1, w, h)
        elif is_directional_mode(mode):
            pred = self._directional_intra_prediction_process(
                av1, plane, x, y, haveLeft, haveAbove, mode, w, h, maxX, maxY)
        elif mode in [Y_MODE.SMOOTH_PRED, Y_MODE.SMOOTH_V_PRED, Y_MODE.SMOOTH_H_PRED]:
            pred = self._smooth_intra_prediction_process(
                av1, mode, log2W, log2H, w, h)
        elif mode == Y_MODE.DC_PRED:
            pred = self._dc_intra_prediction_process(
                av1, haveLeft, haveAbove, log2W, log2H, w, h)
        else:
            pred = self._basic_intra_prediction_process(av1, w, h)

        if av1.on_pred is not None:
            av1.on_pred(plane, x, y, pred)

        av1.CurrFrame = Array(av1.CurrFrame, (PLANE_MAX, y + h, x + w))
        for i in range(h):
            for j in range(w):
                av1.CurrFrame[plane][y + i][x + j] = pred[i][j]

    def _basic_intra_prediction_process(self, av1: AV1Decoder, w: int, h: int) -> List[List[int]]:
        """
        基本帧内预测过程
        规范文档 7.11.2.2 Basic intra prediction process

        Args:
            w: 宽度
            h: 高度
        """
        pred = Array(None, (h, w), 0)
        for i in range(h):
            for j in range(w):
                base = (self._AboveRow[j] +
                        self._LeftCol[i] - self._AboveRow[-1])
                pLeft = abs(base - self._LeftCol[i])
                pTop = abs(base - self._AboveRow[j])
                pTopLeft = abs(base - self._AboveRow[-1])
                if pLeft <= pTop and pLeft <= pTopLeft:
                    pred[i][j] = self._LeftCol[i]
                elif pTop <= pTopLeft:
                    pred[i][j] = self._AboveRow[j]
                else:
                    pred[i][j] = self._AboveRow[-1]
        return pred


    def _recursive_intra_prediction_process(self, av1: AV1Decoder, w: int, h: int) -> List[List[int]]:
        """
        递归帧内预测过程
        规范文档 7.11.2.3 Recursive intra prediction process

        Args:
            w: 宽度
            h: 高度
        """
        seq_header = av1.seq_header
        BitDepth = seq_header.color_config.BitDepth

        pred = Array(None, (h, w), 0)
        w4 = w >> 2
        h2 = h >> 1
        for i2 in range(h2):
            for j4 in range(w4):
                p = [0] * 7
                for i in range(7):
                    if i < 5:
                        if i2 == 0:
                            p[i] = self._AboveRow[(j4 << 2) + i - 1]
                        elif j4 == 0 and i == 0:
                            p[i] = self._LeftCol[(i2 << 1) - 1]
                        else:
                            p[i] = pred[(i2 << 1) - 1][(j4 << 2) + i - 1]
                    else:
                        if j4 == 0:
                            p[i] = self._LeftCol[(i2 << 1) + i - 5]
                        else:
                            p[i] = pred[(i2 << 1) + i - 5][(j4 << 2) - 1]

                for i1 in range(2):
                    for j1 in range(4):
                        pr = 0
                        for i in range(7):
                            pr += Intra_Filter_Taps[av1.tile_group.filter_intra_mode][(
                                i1 << 2) + j1][i] * p[i]
                        pred[(i2 << 1) + i1][(j4 << 2) + j1] = Clip1(Round2Signed(pr,
                                                                                  INTRA_FILTER_SCALE_BITS), BitDepth)
        return pred

    def _directional_intra_prediction_process(self, av1: AV1Decoder, plane: int, x: int, y: int, haveLeft: Optional[int], haveAbove: Optional[int], mode: Y_MODE, w: int, h: int, maxX: int, maxY: int) -> List[List[int]]:
        """
        方向帧内预测过程
        规范文档 7.11.2.4 Directional intra prediction process

        Args:
            plane: 平面索引
            x: X坐标
            y: Y坐标
            haveLeft: 左侧可用标志
            haveAbove: 上方可用标志
            mode: 预测模式
            w: 宽度
            h: 高度
            maxX: 最大X坐标
            maxY: 最大Y坐标
        """
        seq_header = av1.seq_header
        tile_group = av1.tile_group

        # 1.
        if plane == 0:
            angleDelta = tile_group.AngleDeltaY
        else:
            angleDelta = tile_group.AngleDeltaUV

        # 2.
        pAngle = (Mode_To_Angle[mode] + angleDelta * ANGLE_STEP)

        # 3.
        upsampleAbove = 0
        upsampleLeft = 0

        # 4.
        filterType = 0
        if seq_header.enable_intra_edge_filter == 1:
            if pAngle != 90 and pAngle != 180:
                if pAngle > 90 and pAngle < 180 and w + h >= 24:
                    filter = self._filter_corner_process(av1)
                    self._LeftCol[-1] = filter
                    self._AboveRow[-1] = filter
                filterType = self._get_filter_type(av1, plane)
                if haveAbove == 1:
                    strength = self._intra_edge_filter_strength_selection_process(
                        av1, w, h, filterType, pAngle - 90)
                    numPx = (min(w, maxX - x + 1) +
                             (h if pAngle < 90 else 0) + 1)
                    self._intra_edge_filter_process(av1, numPx, strength, 0)

                if haveLeft == 1:
                    strength = self._intra_edge_filter_strength_selection_process(
                        av1, w, h, filterType, pAngle - 180)
                    numPx = (min(h, maxY - y + 1) +
                             (w if pAngle > 180 else 0) + 1)
                    self._intra_edge_filter_process(av1, numPx, strength, 1)

            upsampleAbove = self._intra_edge_upsample_selection_process(
                av1, w, h, filterType, pAngle - 90)
            numPx = w + (h if pAngle < 90 else 0)
            if upsampleAbove == 1:
                self._intra_edge_upsample_process(av1, numPx, 0)

            upsampleLeft = self._intra_edge_upsample_selection_process(
                av1, w, h, filterType, pAngle - 180)
            numPx = h + (w if pAngle > 180 else 0)
            if upsampleLeft == 1:
                self._intra_edge_upsample_process(av1, numPx, 1)

        # 5.
        if pAngle < 90:
            dx = Dr_Intra_Derivative[pAngle]
        elif pAngle > 90 and pAngle < 180:
            dx = Dr_Intra_Derivative[180 - pAngle]
        else:
            dx = NONE

        # 6.
        if pAngle > 90 and pAngle < 180:
            dy = Dr_Intra_Derivative[pAngle - 90]
        elif pAngle > 180:
            dy = Dr_Intra_Derivative[270 - pAngle]
        else:
            dy = NONE

        pred = Array(None, (h, w), 0)
        # 7.
        if pAngle < 90:
            for i in range(h):
                for j in range(w):
                    idx = (i + 1) * dx
                    base = (idx >> (6 - upsampleAbove)) + (j << upsampleAbove)
                    shift = ((idx << upsampleAbove) >> 1) & 0x1F
                    maxBaseX = (w + h - 1) << upsampleAbove
                    if base < maxBaseX:
                        pred[i][j] = Round2(
                            self._AboveRow[base] * (32 - shift) + self._AboveRow[base + 1] * shift, 5)
                    else:
                        pred[i][j] = self._AboveRow[maxBaseX]
        # 8.
        elif pAngle > 90 and pAngle < 180:
            for i in range(h):
                for j in range(w):
                    idx = (j << 6) - (i + 1) * dx
                    base = idx >> (6 - upsampleAbove)
                    if base >= -(1 << upsampleAbove):
                        shift = ((idx << upsampleAbove) >> 1) & 0x1F
                        pred[i][j] = Round2(
                            self._AboveRow[base] * (32 - shift) + self._AboveRow[base + 1] * shift, 5)
                    else:
                        idx = (i << 6) - (j + 1) * dy
                        base = idx >> (6 - upsampleLeft)
                        shift = ((idx << upsampleLeft) >> 1) & 0x1F
                        pred[i][j] = Round2(
                            self._LeftCol[base] * (32 - shift) + self._LeftCol[base + 1] * shift, 5)

        # 9.
        elif pAngle > 180:
            for i in range(h):
                for j in range(w):
                    idx = (j + 1) * dy
                    base = (idx >> (6 - upsampleLeft)) + (i << upsampleLeft)
                    shift = ((idx << upsampleLeft) >> 1) & 0x1F
                    pred[i][j] = Round2(
                        self._LeftCol[base] * (32 - shift) + self._LeftCol[base + 1] * shift, 5)

        # 10.
        elif pAngle == 90:
            for i in range(h):
                for j in range(w):
                    pred[i][j] = self._AboveRow[j]

        # 11.
        elif pAngle == 180:
            for i in range(h):
                for j in range(w):
                    pred[i][j] = self._LeftCol[i]
        return pred

    def _dc_intra_prediction_process(self, av1: AV1Decoder, haveLeft: Optional[int], haveAbove: Optional[int], log2W: int, log2H: int, w: int, h: int) -> List[List[int]]:
        """
        DC帧内预测过程
        规范文档 7.11.2.5 DC intra prediction process
        """
        seq_header = av1.seq_header
        BitDepth = seq_header.color_config.BitDepth

        pred = Array(None, (h, w), 0)

        if haveLeft == 1 and haveAbove == 1:
            sum = 0
            for k in range(h):
                sum += self._LeftCol[k]
            for k in range(w):
                sum += self._AboveRow[k]
            sum += (w + h) >> 1
            avg = sum // (w + h)

            for i in range(h):
                for j in range(w):
                    pred[i][j] = avg

        elif haveLeft == 1 and haveAbove == 0:
            sum = 0
            for k in range(h):
                sum += self._LeftCol[k]
            sum += (h >> 1)
            leftAvg = Clip1(sum >> log2H, BitDepth)
            for i in range(h):
                for j in range(w):
                    pred[i][j] = leftAvg

        elif haveLeft == 0 and haveAbove == 1:
            sum = 0
            for k in range(w):
                sum += self._AboveRow[k]
            sum += (w >> 1)
            aboveAvg = Clip1(sum >> log2W, BitDepth)
            for i in range(h):
                for j in range(w):
                    pred[i][j] = aboveAvg

        elif haveLeft == 0 and haveAbove == 0:
            for i in range(h):
                for j in range(w):
                    pred[i][j] = 1 << (BitDepth - 1)

        return pred

    def _smooth_intra_prediction_process(self, av1: AV1Decoder, mode: int, log2W: int, log2H: int, w: int, h: int) -> List[List[int]]:
        """
        Smooth帧内预测过程
        规范文档 7.11.2.6 Smooth intra prediction process
        """
        pred = Array(None, (h, w), 0)

        smWeights = [None, None, Sm_Weights_Tx_4x4, Sm_Weights_Tx_8x8,
                     Sm_Weights_Tx_16x16, Sm_Weights_Tx_32x32, Sm_Weights_Tx_64x64]

        if mode == Y_MODE.SMOOTH_PRED:
            # 1.
            smWeightsX = smWeights[log2W]

            # 2.
            smWeightsY = smWeights[log2H]

            for i in range(h):
                for j in range(w):
                    # 3.
                    smoothPred = smWeightsY[i] * self._AboveRow[j] + (
                        256 - smWeightsY[i]) * self._LeftCol[h - 1] + smWeightsX[j] * self._LeftCol[i] + (256 - smWeightsX[j]) * self._AboveRow[w - 1]

                    # 4.
                    pred[i][j] = Round2(smoothPred, 9)

        elif mode == Y_MODE.SMOOTH_V_PRED:
            # 1.
            smWeights = smWeights[log2H]

            for i in range(h):
                for j in range(w):
                    # 2.
                    smoothPred = (smWeights[i] * self._AboveRow[j] +
                                  (256 - smWeights[i]) * self._LeftCol[h - 1])

                    # 3.
                    pred[i][j] = Round2(smoothPred, 8)

        elif mode == Y_MODE.SMOOTH_H_PRED:
            # 1.
            smWeights = smWeights[log2W]

            for i in range(h):
                for j in range(w):
                    # 2.
                    smoothPred = (smWeights[j] * self._LeftCol[i] +
                                  (256 - smWeights[j]) * self._AboveRow[w - 1])

                    # 3.
                    pred[i][j] = Round2(smoothPred, 8)

        return pred

    def _filter_corner_process(self, av1: AV1Decoder) -> int:
        """
        Filter corner过程
        规范文档 7.11.2.7 Filter corner process
        """
        s = (self._LeftCol[0] * 5 + self._AboveRow[-1] *
             6 + self._AboveRow[0] * 5)

        return Round2(s, 4)

    def _get_filter_type(self, av1: AV1Decoder, plane: int) -> int:
        """
        Intra filter type过程
        规范文档 7.11.2.8 Intra filter type process
        """
        seq_header = av1.seq_header
        tile_group = av1.tile_group
        subsampling_x = seq_header.color_config.subsampling_x
        subsampling_y = seq_header.color_config.subsampling_y
        MiRow = tile_group.MiRow
        MiCol = tile_group.MiCol
        AvailU = tile_group.AvailU
        AvailL = tile_group.AvailL

        aboveSmooth = 0
        leftSmooth = 0

        def is_smooth(row: int, col: int, plane: int) -> int:
            if plane == 0:
                mode = tile_group.YModes[row][col]
            else:
                if tile_group.RefFrames[row][col][0] > REF_FRAME.INTRA_FRAME:
                    return 0
                mode = tile_group.UVModes[row][col]
            return (mode == Y_MODE.SMOOTH_PRED or mode == Y_MODE.SMOOTH_V_PRED or mode == Y_MODE.SMOOTH_H_PRED)

        if AvailU if (plane == 0) else tile_group.AvailUChroma:
            r = MiRow - 1
            c = MiCol
            if plane > 0:
                if subsampling_x and not (MiCol & 1):
                    c += 1
                if subsampling_y and (MiRow & 1):
                    r -= 1
            aboveSmooth = is_smooth(r, c, plane)

        if AvailL if (plane == 0) else tile_group.AvailLChroma:
            r = MiRow
            c = MiCol - 1
            if plane > 0:
                if subsampling_x and (MiCol & 1):
                    c -= 1
                if subsampling_y and not (MiRow & 1):
                    r += 1
            leftSmooth = is_smooth(r, c, plane)

        return aboveSmooth or leftSmooth

    def _intra_edge_filter_strength_selection_process(self, av1: AV1Decoder, w: int, h: int, filterType: int, delta: int) -> int:
        """
        Intra edge filter strength selection过程
        规范文档 7.11.2.9 Intra edge filter strength selection process
        """
        d = abs(delta)
        blkWh = w + h

        strength = 0
        if filterType == 0:
            if blkWh <= 8:
                if d >= 56:
                    strength = 1
            elif blkWh <= 12:
                if d >= 40:
                    strength = 1
            elif blkWh <= 16:
                if d >= 40:
                    strength = 1
            elif blkWh <= 24:
                if d >= 8:
                    strength = 1
                if d >= 16:
                    strength = 2
                if d >= 32:
                    strength = 3
            elif blkWh <= 32:
                strength = 1
                if d >= 4:
                    strength = 2
                if d >= 32:
                    strength = 3
            else:
                strength = 3
        else:
            if blkWh <= 8:
                if d >= 40:
                    strength = 1
                if d >= 64:
                    strength = 2
            elif blkWh <= 16:
                if d >= 20:
                    strength = 1
                if d >= 48:
                    strength = 2
            elif blkWh <= 24:
                if d >= 4:
                    strength = 3
            else:
                strength = 3

        return strength

    def _intra_edge_upsample_selection_process(self, av1: AV1Decoder, w: int, h: int, filterType: int, delta: int) -> int:
        """
        Intra edge upsample selection过程
        规范文档 7.11.2.10 Intra edge upsample selection process
        """
        d = abs(delta)
        blkWh = w + h

        useUpsample = 0
        if d <= 0 or d >= 40:
            useUpsample = 0
        elif filterType == 0:
            useUpsample = (blkWh <= 16)
        else:
            useUpsample = (blkWh <= 8)

        return useUpsample

    def _intra_edge_upsample_process(self, av1: AV1Decoder, numPx: int, dir_val: int) -> None:
        """
        Intra edge upsample过程
        规范文档 7.11.2.11 Intra edge upsample process
        """
        seq_header = av1.seq_header
        BitDepth = seq_header.color_config.BitDepth

        buf = Array(None, (numPx * 2,))
        if dir_val == 0:
            buf = self._AboveRow
        else:
            buf = self._LeftCol

        dup = [0] * (numPx + 3)
        dup[0] = buf[-1]
        for i in range(-1, numPx):
            dup[i + 2] = buf[i]
        dup[numPx + 2] = buf[numPx - 1]

        buf[-2] = dup[0]
        for i in range(numPx):
            s = -dup[i] + (9 * dup[i + 1]) + (9 * dup[i + 2]) - dup[i + 3]
            s = Clip1(Round2(s, 4), BitDepth)
            buf[2 * i - 1] = s
            buf[2 * i] = dup[i + 2]

    def _intra_edge_filter_process(self, av1: AV1Decoder, sz: int, strength: int, left: int) -> None:
        """
        Intra edge filter过程
        规范文档 7.11.2.12 Intra edge filter process
        """
        if strength == 0:
            return
        edge = [0] * sz
        for i in range(sz):
            edge[i] = self._LeftCol[i - 1] if left else self._AboveRow[i - 1]

        for i in range(1, sz):
            # 1.
            s = 0

            # 2.
            for j in range(INTRA_EDGE_TAPS):
                # a.
                k = Clip3(0, sz - 1, i - 2 + j)
                # b.
                s += Intra_Edge_Kernel[strength - 1][j] * edge[k]

            # 3.
            if left == 1:
                self._LeftCol[i - 1] = (s + 8) >> 4
            # 4.
            elif left == 0:
                self._AboveRow[i - 1] = (s + 8) >> 4

    def predict_inter(self, av1: AV1Decoder, plane: int, x: int, y: int,
                      w: int, h: int, candRow: int, candCol: int):
        """
        帧间预测过程
        规范文档 7.11.3 Inter prediction process

        此过程生成帧间预测块。

        Args:
            plane: 平面索引
            x: X坐标
            y: Y坐标
            w: 预测宽度
            h: 预测高度
            candRow: 候选行位置
            candCol: 候选列位置
        """
        seq_header = av1.seq_header
        frame_header = av1.frame_header
        tile_group = av1.tile_group
        ref_frame_store = av1.ref_frame_store
        BitDepth = seq_header.color_config.BitDepth
        ref_frame_idx = frame_header.ref_frame_idx
        MiRows = frame_header.MiRows
        MiCols = frame_header.MiCols
        motion_mode = tile_group.motion_mode
        compound_type = tile_group.compound_type
        YMode = tile_group.YMode

        isCompound = tile_group.RefFrames[candRow][candCol][1] > REF_FRAME.INTRA_FRAME

        preds: List[List[List[int]]] = [NONE] * 2

        # 1.
        rounding_variables_derivation(av1, isCompound)

        # 2.
        if plane == 0 and motion_mode == MOTION_MODE.LOCALWARP:
            self._warp_estimation(av1)

        # 3.
        if plane == 0 and motion_mode == MOTION_MODE.LOCALWARP and self._LocalValid == 1:
            warpValid, _, _, _, _ = self._setup_shear(
                av1, self._LocalWarpParams)
            self._LocalValid = warpValid

        # 4.
        refList = 0

        globalValid = 0

        while 1:
            # 5.
            refFrame = tile_group.RefFrames[candRow][candCol][refList]

            # 6.
            if ((YMode in [Y_MODE.GLOBALMV, Y_MODE.GLOBAL_GLOBALMV]) and
                    frame_header.GmType[refFrame] > GM_TYPE.TRANSLATION):
                warpValid, _, _, _, _ = self._setup_shear(
                    av1, frame_header.gm_params[refFrame])
                globalValid = warpValid

            # 7.
            useWarp = 0
            if w < 8 or h < 8:
                useWarp = 0
            elif frame_header.force_integer_mv == 1:
                useWarp = 0
            elif motion_mode == MOTION_MODE.LOCALWARP and self._LocalValid == 1:
                useWarp = 1
            elif (YMode in [Y_MODE.GLOBALMV, Y_MODE.GLOBAL_GLOBALMV] and
                  frame_header.GmType[refFrame] > GM_TYPE.TRANSLATION and
                  is_scaled(av1, refFrame) == 0 and
                  globalValid == 1):
                useWarp = 2
            else:
                useWarp = 0

            # 8.
            mv = deepcopy(tile_group.Mvs[candRow][candCol][refList])

            # 9.
            use_intrabc = tile_group.use_intrabc
            if use_intrabc == 0:
                refIdx = ref_frame_idx[refFrame - REF_FRAME.LAST_FRAME]
            else:
                refIdx = -1
                ref_frame_store.RefFrameWidth[-1] = frame_header.FrameWidth
                ref_frame_store.RefFrameHeight[-1] = frame_header.FrameHeight
                ref_frame_store.RefUpscaledWidth[-1] = frame_header.UpscaledWidth

            # 10.
            startX, startY, stepX, stepY = self._motion_vector_scaling(
                av1, plane, refIdx, x, y, mv)

            # 11.
            if use_intrabc == 1:
                ref_frame_store.RefFrameWidth[-1] = MiCols * MI_SIZE
                ref_frame_store.RefFrameHeight[-1] = MiRows * MI_SIZE
                ref_frame_store.RefUpscaledWidth[-1] = MiCols * MI_SIZE

            # 12.
            if useWarp != 0:
                pred: List[List[int]] = Array(
                    None, (ceil(h / 8)*8, ceil(w / 8)*8))
                for i8 in range(((h - 1) >> 3) + 1):
                    for j8 in range(((w - 1) >> 3) + 1):
                        self._block_warp(av1, useWarp, plane,
                                         refList, x, y, i8, j8, w, h, pred)
                preds[refList] = pred

            # 13.
            else:
                preds[refList] = self.block_inter_prediction(
                    av1, plane, refIdx, startX, startY, stepX, stepY, w, h, candRow, candCol)

            # 14.
            if refList == 1:
                break
            elif isCompound == 1:
                refList = 1
            else:
                break

        if av1.on_pred is not None:
            for pred in preds:
                if pred is not None:
                    av1.on_pred(plane, x, y, pred)

        if compound_type == COMPOUND_TYPE.COMPOUND_WEDGE and plane == 0:
            self._wedge_mask(av1, w, h)
        elif compound_type == COMPOUND_TYPE.COMPOUND_INTRA:
            self._intra_mode_variant_mask(av1, w, h)
        elif compound_type == COMPOUND_TYPE.COMPOUND_DIFFWTD and plane == 0:
            self._difference_weight_mask(av1, preds, w, h)

        if compound_type == COMPOUND_TYPE.COMPOUND_DISTANCE:
            self._distance_weights(av1, candRow, candCol)

        av1.CurrFrame = Array(av1.CurrFrame, (plane, y + h, x + w))
        if isCompound == 0 and tile_group.IsInterIntra == 0:
            for i in range(h):
                for j in range(w):
                    av1.CurrFrame[plane][y + i][x +
                                                j] = Clip1(preds[0][i][j], BitDepth)
        elif compound_type == COMPOUND_TYPE.COMPOUND_AVERAGE:
            for i in range(h):
                for j in range(w):
                    av1.CurrFrame[plane][y + i][x + j] = Clip1(Round2(
                        preds[0][i][j] + preds[1][i][j], 1 + tile_group.InterPostRound), BitDepth)
        elif compound_type == COMPOUND_TYPE.COMPOUND_DISTANCE:
            for i in range(h):
                for j in range(w):
                    av1.CurrFrame[plane][y + i][x + j] = Clip1(Round2(
                        self._FwdWeight * preds[0][i][j] + self._BckWeight * preds[1][i][j], 4 + tile_group.InterPostRound), BitDepth)
        else:
            self._mask_blend(av1, preds, plane, x, y, w, h)

        if motion_mode == MOTION_MODE.OBMC:
            self._overlapped_motion_compensation(av1, plane, w, h)

    def _motion_vector_scaling(self, av1: AV1Decoder, plane: int, refIdx: int, x: int, y: int, mv: List[int]) -> tuple:
        """
        运动向量缩放过程
        规范文档 7.11.3.3 Motion vector scaling process

        Args:
            plane: 平面索引
            refIdx: 参考帧索引
            x: X坐标
            y: Y坐标
            mv: 运动向量 [row, col]

        Returns:
            dict包含 startX, startY, stepX, stepY
        """
        seq_header = av1.seq_header
        frame_header = av1.frame_header
        ref_frame_store = av1.ref_frame_store
        subsampling_x = seq_header.color_config.subsampling_x
        subsampling_y = seq_header.color_config.subsampling_y

        # It is a requirement of bitstream conformance that all the following conditions are satisfied:
        # - 2 * FrameWidth >= RefUpscaledWidth[ refIdx ]
        # - 2 * FrameHeight >= RefFrameHeight[ refIdx ]
        # - FrameWidth <= 16 * RefUpscaledWidth[ refIdx ]
        # - FrameHeight <= 16 * RefFrameHeight[ refIdx ]
        assert (2 *
                frame_header.FrameWidth >= ref_frame_store.RefUpscaledWidth[refIdx])
        assert (2 *
                frame_header.FrameHeight >= ref_frame_store.RefFrameHeight[refIdx])
        assert (frame_header.FrameWidth <= 16 *
                ref_frame_store.RefUpscaledWidth[refIdx])
        assert (frame_header.FrameHeight <= 16 *
                ref_frame_store.RefFrameHeight[refIdx])

        xScale = ((ref_frame_store.RefUpscaledWidth[refIdx] << REF_SCALE_SHIFT) + (
            frame_header.FrameWidth // 2)) // frame_header.FrameWidth
        yScale = ((ref_frame_store.RefFrameHeight[refIdx] << REF_SCALE_SHIFT) + (
            frame_header.FrameHeight // 2)) // frame_header.FrameHeight

        if plane == 0:
            subX = 0
            subY = 0
        else:
            subX = subsampling_x
            subY = subsampling_y

        halfSample = 1 << (SUBPEL_BITS - 1)
        origX = (x << SUBPEL_BITS) + ((2 * mv[1]) >> subX) + halfSample
        origY = (y << SUBPEL_BITS) + ((2 * mv[0]) >> subY) + halfSample
        baseX = origX * xScale - (halfSample << REF_SCALE_SHIFT)
        baseY = origY * yScale - (halfSample << REF_SCALE_SHIFT)
        off = (1 << (SCALE_SUBPEL_BITS - SUBPEL_BITS)) // 2
        startX = Round2Signed(baseX, REF_SCALE_SHIFT +
                              SUBPEL_BITS - SCALE_SUBPEL_BITS) + off
        startY = Round2Signed(baseY, REF_SCALE_SHIFT +
                              SUBPEL_BITS - SCALE_SUBPEL_BITS) + off
        stepX = Round2Signed(xScale, REF_SCALE_SHIFT - SCALE_SUBPEL_BITS)
        stepY = Round2Signed(yScale, REF_SCALE_SHIFT - SCALE_SUBPEL_BITS)

        return startX, startY, stepX, stepY

    def block_inter_prediction(self, av1: AV1Decoder, plane: int, refIdx: int, x: int, y: int,
                               xStep: int, yStep: int, w: int, h: int,
                               candRow: int, candCol: int) -> List[List[int]]:
        """
        块间预测过程
        规范文档 7.11.3.4 Block inter prediction process

        Args:
            plane: 平面索引
            refIdx: 参考帧索引
            x: X坐标
            y: Y坐标
            xStep: X步长
            yStep: Y步长
            w: 宽度
            h: 高度
            candRow: 候选行
            candCol: 候选列

        Returns:
            预测块数组
        """
        seq_header = av1.seq_header
        frame_header = av1.frame_header
        tile_group = av1.tile_group
        ref_frame_store = av1.ref_frame_store
        subsampling_x = seq_header.color_config.subsampling_x
        subsampling_y = seq_header.color_config.subsampling_y

        if refIdx == -1:
            ref = av1.CurrFrame
        else:
            ref = ref_frame_store.FrameStore[refIdx]

        if plane == 0:
            subX = 0
            subY = 0
        else:
            subX = subsampling_x
            subY = subsampling_y

        lastX = ((ref_frame_store.RefUpscaledWidth[refIdx] + subX) >> subX) - 1
        lastY = ((ref_frame_store.RefFrameHeight[refIdx] + subY) >> subY) - 1

        intermediateHeight = (
            ((h - 1) * yStep + (1 << SCALE_SUBPEL_BITS) - 1) >> SCALE_SUBPEL_BITS) + 8

        interpFilter = tile_group.InterpFilters[candRow][candCol][1]
        if w <= 4:
            if interpFilter == INTERPOLATION_FILTER.EIGHTTAP or interpFilter == INTERPOLATION_FILTER.EIGHTTAP_SHARP:
                interpFilter = 4
            elif interpFilter == INTERPOLATION_FILTER.EIGHTTAP_SMOOTH:
                interpFilter = 5

        intermediate = Array(None, (intermediateHeight, w), 0)

        for r in range(intermediateHeight):
            for c in range(w):
                s = 0
                p = x + xStep * c
                for t in range(8):
                    s += Subpel_Filters[interpFilter][(p >> 6) & SUBPEL_MASK][t] * ref[plane][Clip3(
                        0, lastY, (y >> 10) + r - 3)][Clip3(0, lastX, (p >> 10) + t - 3)]
                intermediate[r][c] = Round2(s, tile_group.InterRound0)

        interpFilter = tile_group.InterpFilters[candRow][candCol][0]
        if h <= 4:
            if interpFilter == INTERPOLATION_FILTER.EIGHTTAP or interpFilter == INTERPOLATION_FILTER.EIGHTTAP_SHARP:
                interpFilter = 4
            elif interpFilter == INTERPOLATION_FILTER.EIGHTTAP_SMOOTH:
                interpFilter = 5

        pred = Array(None, (h, w), 0)
        for r in range(h):
            for c in range(w):
                s = 0
                p = (y & 1023) + yStep * r
                for t in range(8):
                    s += Subpel_Filters[interpFilter][(
                        p >> 6) & SUBPEL_MASK][t] * intermediate[(p >> 10) + t][c]
                pred[r][c] = Round2(s, tile_group.InterRound1)

        return pred

    def _block_warp(self, av1: AV1Decoder, useWarp: int, plane: int, refList: int,
                    x: int, y: int, i8: int, j8: int, w: int, h: int,
                    pred: List[List[int]]):
        """
        块warp过程
        规范文档 7.11.3.5 Block warp process

        Args:
            useWarp: warp类型
            plane: 平面索引
            refList: 参考列表索引
            x: X坐标
            y: Y坐标
            i8: 8x8块行索引
            j8: 8x8块列索引
            w: 宽度
            h: 高度
            pred: 预测数组（会被修改）
        """
        seq_header = av1.seq_header
        frame_header = av1.frame_header
        tile_group = av1.tile_group
        ref_frame_store = av1.ref_frame_store
        subsampling_x = seq_header.color_config.subsampling_x
        subsampling_y = seq_header.color_config.subsampling_y
        ref_frame_idx = frame_header.ref_frame_idx
        gm_params = frame_header.gm_params

        refIdx = ref_frame_idx[tile_group.RefFrame[refList] -
                               REF_FRAME.LAST_FRAME]
        ref = ref_frame_store.FrameStore[refIdx]

        if plane == 0:
            subX = 0
            subY = 0
        else:
            subX = subsampling_x
            subY = subsampling_y

        lastX = ((ref_frame_store.RefUpscaledWidth[refIdx] + subX) >> subX) - 1
        lastY = ((ref_frame_store.RefFrameHeight[refIdx] + subY) >> subY) - 1
        srcX = (x + j8 * 8 + 4) << subX
        srcY = (y + i8 * 8 + 4) << subY

        if useWarp == 1:
            warpParams = self._LocalWarpParams
        else:
            warpParams = gm_params[tile_group.RefFrame[refList]]

        dstX = warpParams[2] * srcX + warpParams[3] * srcY + warpParams[0]
        dstY = warpParams[4] * srcX + warpParams[5] * srcY + warpParams[1]

        warpValid, alpha, beta, gamma, delta = self._setup_shear(
            av1, warpParams)
        # warpValid will always be equal to 1 at this point.
        assert warpValid == 1

        x4 = dstX >> subX
        y4 = dstY >> subY
        ix4 = x4 >> WARPEDMODEL_PREC_BITS
        sx4 = x4 & ((1 << WARPEDMODEL_PREC_BITS) - 1)
        iy4 = y4 >> WARPEDMODEL_PREC_BITS
        sy4 = y4 & ((1 << WARPEDMODEL_PREC_BITS) - 1)

        intermediate = Array(None, (15, 8), 0)
        for i1 in range(-7, 8):
            for i2 in range(-4, 4):
                sx = sx4 + alpha * i2 + beta * i1
                offs = (Round2(sx, WARPEDDIFF_PREC_BITS) +
                        WARPEDPIXEL_PREC_SHIFTS)
                s = 0
                for i3 in range(8):
                    s += Warped_Filters[offs][i3] * ref[plane][Clip3(
                        0, lastY, iy4 + i1)][Clip3(0, lastX, ix4 + i2 - 3 + i3)]
                intermediate[i1 + 7][i2 +
                                     4] = Round2(s, tile_group.InterRound0)

        for i1 in range(-4, min(4, h - i8 * 8 - 4)):
            for i2 in range(-4, min(4, w - j8 * 8 - 4)):
                sy = sy4 + gamma * i2 + delta * i1
                offs = (Round2(sy, WARPEDDIFF_PREC_BITS) +
                        WARPEDPIXEL_PREC_SHIFTS)
                s = 0
                for i3 in range(8):
                    s += (Warped_Filters[offs][i3] *
                          intermediate[(i1 + i3 + 4)][(i2 + 4)])
                pred[i8 * 8 + i1 + 4][j8 * 8 + i2 +
                                      4] = Round2(s, tile_group.InterRound1)

        return pred

    def _setup_shear(self, av1: AV1Decoder, warpParams: List[int]) -> tuple:
        """
        Setup shear process
        规范文档 7.11.3.6 Setup shear process
        """
        alpha0 = Clip3(-32768, 32767,
                       warpParams[2] - (1 << WARPEDMODEL_PREC_BITS))
        beta0 = Clip3(-32768, 32767, warpParams[3])
        divShift, divFactor = self.resolve_divisor(warpParams[2])
        v = warpParams[4] << WARPEDMODEL_PREC_BITS
        gamma0 = Clip3(-32768, 32767, Round2Signed(v * divFactor, divShift))
        w = warpParams[3] * warpParams[4]
        delta0 = Clip3(-32768, 32767, warpParams[5] - Round2Signed(
            w * divFactor, divShift) - (1 << WARPEDMODEL_PREC_BITS))

        alpha = Round2Signed(
            alpha0, WARP_PARAM_REDUCE_BITS) << WARP_PARAM_REDUCE_BITS
        beta = Round2Signed(
            beta0, WARP_PARAM_REDUCE_BITS) << WARP_PARAM_REDUCE_BITS
        gamma = Round2Signed(
            gamma0, WARP_PARAM_REDUCE_BITS) << WARP_PARAM_REDUCE_BITS
        delta = Round2Signed(
            delta0, WARP_PARAM_REDUCE_BITS) << WARP_PARAM_REDUCE_BITS

        warpValid = 1
        if 4 * abs(alpha) + 7 * abs(beta) >= 1 << WARPEDMODEL_PREC_BITS:
            warpValid = 0
        if 4 * abs(gamma) + 4 * abs(delta) >= 1 << WARPEDMODEL_PREC_BITS:
            warpValid = 0
        else:
            warpValid = 1

        return warpValid, alpha, beta, gamma, delta

    def resolve_divisor(self, d: int) -> tuple:
        """
        Resolve divisor process
        规范文档 7.11.3.7 Resolve divisor process
        """

        n = FloorLog2(abs(d))
        e = abs(d) - (1 << n)

        if n > DIV_LUT_BITS:
            f = Round2(e, n - DIV_LUT_BITS)
        else:
            f = e << (DIV_LUT_BITS - n)

        divShift = n + DIV_LUT_PREC_BITS

        if d < 0:
            divFactor = -Div_Lut[f]
        else:
            divFactor = Div_Lut[f]

        return divShift, divFactor

    def _warp_estimation(self, av1: AV1Decoder):
        """
        Warp estimation过程
        规范文档 7.11.3.8 Warp estimation process
        """
        tile_group = av1.tile_group
        MiRow = tile_group.MiRow
        MiCol = tile_group.MiCol
        MiSize = tile_group.MiSize

        A = Array(None, (2, 2), 0)
        Bx = [0] * 2
        By = [0] * 2

        w4 = Num_4x4_Blocks_Wide[MiSize]
        h4 = Num_4x4_Blocks_High[MiSize]
        midY = MiRow * 4 + h4 * 2 - 1
        midX = MiCol * 4 + w4 * 2 - 1
        suy = midY * 8
        sux = midX * 8
        duy = suy + tile_group.Mv[0][0]
        dux = sux + tile_group.Mv[0][1]

        def ls_product(a: int, b: int) -> int:
            return ((a * b) >> 2) + (a + b)

        for i in range(tile_group.NumSamples):
            sy = tile_group.CandList[i][0] - suy
            sx = tile_group.CandList[i][1] - sux
            dy = tile_group.CandList[i][2] - duy
            dx = tile_group.CandList[i][3] - dux

            if abs(sx - dx) < LS_MV_MAX and abs(sy - dy) < LS_MV_MAX:
                A[0][0] += ls_product(sx, sx) + 8
                A[0][1] += ls_product(sx, sy) + 4
                A[1][1] += ls_product(sy, sy) + 8
                Bx[0] += ls_product(sx, dx) + 8
                Bx[1] += ls_product(sy, dx) + 4
                By[0] += ls_product(sx, dy) + 4
                By[1] += ls_product(sy, dy) + 8

        det = A[0][0] * A[1][1] - A[0][1] * A[0][1]

        if det == 0:
            self._LocalValid = 0
        else:
            self._LocalValid = 1

        if det == 0:
            return

        divShift, divFactor = self.resolve_divisor(det)

        divShift -= WARPEDMODEL_PREC_BITS
        if divShift < 0:
            divFactor = divFactor << (-divShift)
            divShift = 0

        def nondiag(v: int) -> int:
            return Clip3(-WARPEDMODEL_NONDIAGAFFINE_CLAMP + 1,
                         WARPEDMODEL_NONDIAGAFFINE_CLAMP - 1,
                         Round2Signed(v * divFactor, divShift))

        def diag(v: int) -> int:
            return Clip3((1 << WARPEDMODEL_PREC_BITS) - WARPEDMODEL_NONDIAGAFFINE_CLAMP + 1,
                         (1 << WARPEDMODEL_PREC_BITS) +
                         WARPEDMODEL_NONDIAGAFFINE_CLAMP - 1,
                         Round2Signed(v * divFactor, divShift))

        self._LocalWarpParams[2] = diag(A[1][1] * Bx[0] - A[0][1] * Bx[1])
        self._LocalWarpParams[3] = nondiag(-A[0][1] * Bx[0] + A[0][0] * Bx[1])
        self._LocalWarpParams[4] = nondiag(A[1][1] * By[0] - A[0][1] * By[1])
        self._LocalWarpParams[5] = diag(-A[0][1] * By[0] + A[0][0] * By[1])

        mvx = tile_group.Mv[0][1]
        mvy = tile_group.Mv[0][0]

        vx = mvx * (1 << (WARPEDMODEL_PREC_BITS - 3)) - (midX * (self._LocalWarpParams[2] - (
            1 << WARPEDMODEL_PREC_BITS)) + midY * self._LocalWarpParams[3])
        vy = mvy * (1 << (WARPEDMODEL_PREC_BITS - 3)) - (midX * self._LocalWarpParams[4] + midY * (
            self._LocalWarpParams[5] - (1 << WARPEDMODEL_PREC_BITS)))
        self._LocalWarpParams[0] = Clip3(-WARPEDMODEL_TRANS_CLAMP,
                                         WARPEDMODEL_TRANS_CLAMP - 1, vx)
        self._LocalWarpParams[1] = Clip3(-WARPEDMODEL_TRANS_CLAMP,
                                         WARPEDMODEL_TRANS_CLAMP - 1, vy)

        if av1.on_wmmat is not None:
            av1.on_wmmat(self._LocalWarpParams)

    def _overlapped_motion_compensation(self, av1: AV1Decoder, plane: int, w: int, h: int):
        """
        重叠运动补偿过程
        规范文档 7.11.3.9 Overlapped motion compensation process

        Args:
            plane: 平面索引
            w: 宽度
            h: 高度
        """
        seq_header = av1.seq_header
        frame_header = av1.frame_header
        tile_group = av1.tile_group
        subsampling_x = seq_header.color_config.subsampling_x
        subsampling_y = seq_header.color_config.subsampling_y
        BitDepth = seq_header.color_config.BitDepth
        ref_frame_idx = frame_header.ref_frame_idx
        MiRows = frame_header.MiRows
        MiCols = frame_header.MiCols
        MiRow = tile_group.MiRow
        MiCol = tile_group.MiCol
        MiSize = tile_group.MiSize
        AvailU = tile_group.AvailU
        AvailL = tile_group.AvailL

        if plane == 0:
            subX = 0
            subY = 0
        else:
            subX = subsampling_x
            subY = subsampling_y

        def get_obmc_mask(length: int) -> List[int]:
            if length == 2:
                return Obmc_Mask_2
            elif length == 4:
                return Obmc_Mask_4
            elif length == 8:
                return Obmc_Mask_8
            elif length == 16:
                return Obmc_Mask_16
            else:
                return Obmc_Mask_32

        def predict_overlap(candRow: int, candCol: int, x4: int, y4: int, subX: int, subY: int, plane: int, predW: int, predH: int, pass_val: int, mask: List[int]):
            """
            预测重叠
            规范文档 7.11.3.9 predict_overlap
            """
            BitDepth = seq_header.color_config.BitDepth
            ref_frame_idx = frame_header.ref_frame_idx

            mv = deepcopy(tile_group.Mvs[candRow][candCol][0])

            refIdx = ref_frame_idx[tile_group.RefFrames[candRow]
                                   [candCol][0] - REF_FRAME.LAST_FRAME]

            predX = (x4 * 4) >> subX
            predY = (y4 * 4) >> subY

            startX, startY, stepX, stepY = self._motion_vector_scaling(
                av1, plane, refIdx, predX, predY, mv)

            obmcPred = self.block_inter_prediction(
                av1, plane, refIdx, startX, startY, stepX, stepY, predW, predH, candRow, candCol)

            for i in range(predH):
                for j in range(predW):
                    obmcPred[i][j] = Clip1(obmcPred[i][j], BitDepth)

            self._overlap_blending(
                av1, plane, predX, predY, predW, predH, pass_val, obmcPred, mask)

        if AvailU:
            if get_plane_residual_size(av1, MiSize, plane) >= SUB_SIZE.BLOCK_8X8:
                pass_val = 0
                w4 = Num_4x4_Blocks_Wide[MiSize]
                x4 = MiCol
                y4 = MiRow
                nCount = 0
                nLimit = min(4, Mi_Width_Log2[MiSize])
                while nCount < nLimit and x4 < min(MiCols, MiCol + w4):
                    candRow = MiRow - 1
                    candCol = x4 | 1
                    candSz = tile_group.MiSizes[candRow][candCol]
                    step4 = Clip3(2, 16, Num_4x4_Blocks_Wide[candSz])
                    if tile_group.RefFrames[candRow][candCol][0] > REF_FRAME.INTRA_FRAME:
                        nCount += 1
                        predW = min(w, (step4 * MI_SIZE) >> subX)
                        predH = min(h >> 1, 32 >> subY)
                        mask = get_obmc_mask(predH)
                        predict_overlap(
                            candRow, candCol, x4, y4, subX, subY, plane, predW, predH, pass_val, mask)
                    x4 += step4

        if AvailL:
            pass_val = 1
            h4 = Num_4x4_Blocks_High[MiSize]
            x4 = MiCol
            y4 = MiRow
            nCount = 0
            nLimit = min(4, Mi_Height_Log2[MiSize])
            while nCount < nLimit and y4 < min(MiRows, MiRow + h4):
                candCol = MiCol - 1
                candRow = y4 | 1
                candSz = tile_group.MiSizes[candRow][candCol]
                step4 = Clip3(2, 16, Num_4x4_Blocks_High[candSz])
                if tile_group.RefFrames[candRow][candCol][0] > REF_FRAME.INTRA_FRAME:
                    nCount += 1
                    predW = min(w >> 1, 32 >> subX)
                    predH = min(h, (step4 * MI_SIZE) >> subY)
                    mask = get_obmc_mask(predW)
                    predict_overlap(candRow, candCol, x4, y4, subX,
                                    subY, plane, predW, predH, pass_val, mask)
                y4 += step4

    def _overlap_blending(self, av1: AV1Decoder, plane: int, predX: int, predY: int, predW: int, predH: int, pass_val: int, obmcPred: List[List[int]], mask: List[int]):
        """
        重叠混合
        规范文档 7.11.3.10 Overlap blending process
        """
        for i in range(predH):
            for j in range(predW):
                # 1.
                if pass_val == 0:
                    m = mask[i]
                else:
                    m = mask[j]

                # 2.
                av1.CurrFrame[plane][predY + i][predX + j] = Round2(
                    m * av1.CurrFrame[plane][predY + i][predX + j] + (64 - m) * obmcPred[i][j], 6)

    def _wedge_mask(self, av1: AV1Decoder, w: int, h: int):
        """
        Wedge mask过程
        规范文档 7.11.3.11 Wedge mask process

        Args:
            w: 宽度
            h: 高度
        """
        tile_group = av1.tile_group
        MiSize = tile_group.MiSize

        self._Mask = Array(None, (h, w))
        self._initialise_wedge_mask_table()
        for i in range(h):
            for j in range(w):
                self._Mask[i][j] = self._WedgeMasks[MiSize][tile_group.wedge_sign][tile_group.wedge_index][i][j]

    def _initialise_wedge_mask_table(self):
        """
        Initialise wedge mask table
        规范文档 7.11.3.11 Wedge mask process
        """
        w = MASK_MASTER_SIZE
        h = MASK_MASTER_SIZE

        def block_shape(bsize: int) -> int:
            w4 = Num_4x4_Blocks_Wide[bsize]
            h4 = Num_4x4_Blocks_High[bsize]
            if h4 > w4:
                return 0
            elif h4 < w4:
                return 1
            else:
                return 2

        def get_wedge_direction(bsize: SUB_SIZE, wedge: int) -> int:
            return Wedge_Codebook[block_shape(bsize)][wedge][0]

        def get_wedge_xoff(bsize: SUB_SIZE, wedge: int) -> int:
            return Wedge_Codebook[block_shape(bsize)][wedge][1]

        def get_wedge_yoff(bsize: SUB_SIZE, wedge: int) -> int:
            return Wedge_Codebook[block_shape(bsize)][wedge][2]

        MasterMask = Array(None, (6, max(h + 1, w), max(h, w)))
        for j in range(w):
            shift = MASK_MASTER_SIZE // 4
            for i in range(0, h, 2):
                MasterMask[WEDGE_OBLIQUE63][i][j] = Wedge_Master_Oblique_Even[Clip3(
                    0, MASK_MASTER_SIZE - 1, j - shift)]
                shift -= 1
                MasterMask[WEDGE_OBLIQUE63][i + 1][j] = Wedge_Master_Oblique_Odd[Clip3(
                    0, MASK_MASTER_SIZE - 1, j - shift)]
                MasterMask[WEDGE_VERTICAL][i][j] = Wedge_Master_Vertical[j]
                MasterMask[WEDGE_VERTICAL][i + 1][j] = Wedge_Master_Vertical[j]
        for i in range(h):
            for j in range(w):
                msk = MasterMask[WEDGE_OBLIQUE63][i][j]
                MasterMask[WEDGE_OBLIQUE27][j][i] = msk
                MasterMask[WEDGE_OBLIQUE117][i][w - 1 - j] = 64 - msk
                MasterMask[WEDGE_OBLIQUE153][w - 1 - j][i] = 64 - msk
                MasterMask[WEDGE_HORIZONTAL][j][i] = MasterMask[WEDGE_VERTICAL][i][j]

        for bsize in range(SUB_SIZE.BLOCK_8X8, SUB_SIZE.BLOCK_SIZES):
            if Wedge_Bits[bsize] > 0:
                w = Block_Width[bsize]
                h = Block_Height[bsize]
                for wedge in range(WEDGE_TYPES):
                    dir = get_wedge_direction(SUB_SIZE(bsize), wedge)
                    xoff = (MASK_MASTER_SIZE // 2 -
                            ((get_wedge_xoff(SUB_SIZE(bsize), wedge) * w) >> 3))
                    yoff = (MASK_MASTER_SIZE // 2 -
                            ((get_wedge_yoff(SUB_SIZE(bsize), wedge) * h) >> 3))
                    sum = 0
                    for i in range(w):
                        sum += MasterMask[dir][yoff][xoff + i]
                    for i in range(1, h):
                        sum += MasterMask[dir][yoff + i][xoff]
                    avg = (sum + (w + h - 1) // 2) // (w + h - 1)
                    flipSign = (avg < 32)
                    for i in range(h):
                        for j in range(w):
                            self._WedgeMasks[bsize][flipSign][wedge][i][j] = MasterMask[dir][yoff + i][xoff + j]
                            self._WedgeMasks[bsize][not flipSign][wedge][i][j] = (64 -
                                                                                  MasterMask[dir][yoff + i][xoff + j])

    def _difference_weight_mask(self, av1: AV1Decoder, preds: List[List[List[int]]], w: int, h: int):
        """
        Difference weight mask过程
        规范文档 7.11.3.12 Difference weight mask process

        Args:
            preds: 预测数组
            w: 宽度
            h: 高度
        """
        seq_header = av1.seq_header
        tile_group = av1.tile_group
        BitDepth = seq_header.color_config.BitDepth

        self._Mask = Array(None, (h, w))
        for i in range(h):
            for j in range(w):
                diff = abs(preds[0][i][j] - preds[1][i][j])
                diff = Round2(diff, (BitDepth - 8) + tile_group.InterPostRound)
                m = Clip3(0, 64, 38 + diff // 16)
                if tile_group.mask_type:
                    self._Mask[i][j] = 64 - m
                else:
                    self._Mask[i][j] = m

    def _intra_mode_variant_mask(self, av1: AV1Decoder, w: int, h: int):
        """
        Intra mode variant mask过程
        规范文档 7.11.3.13 Intra mode variant mask process

        Args:
            w: 宽度
            h: 高度
        """
        tile_group = av1.tile_group

        self._Mask = Array(None, (h, w))
        sizeScale = MAX_SB_SIZE // max(h, w)
        for i in range(h):
            for j in range(w):
                if tile_group.interintra_mode == INTERINTRA_MODE.II_V_PRED:
                    self._Mask[i][j] = Ii_Weights_1d[i * sizeScale]
                elif tile_group.interintra_mode == INTERINTRA_MODE.II_H_PRED:
                    self._Mask[i][j] = Ii_Weights_1d[j * sizeScale]
                elif tile_group.interintra_mode == INTERINTRA_MODE.II_SMOOTH_PRED:
                    self._Mask[i][j] = Ii_Weights_1d[min(i, j) * sizeScale]
                else:
                    self._Mask[i][j] = 32

    def _mask_blend(self, av1: AV1Decoder, preds: List[List[List[int]]], plane: int,
                    dstX: int, dstY: int, w: int, h: int):
        """
        Mask blend过程
        规范文档 7.11.3.14 Mask blend process

        Args:
            preds: 预测数组
            plane: 平面索引
            dstX: 目标X坐标
            dstY: 目标Y坐标
            w: 宽度
            h: 高度
        """
        seq_header = av1.seq_header
        tile_group = av1.tile_group
        subsampling_x = seq_header.color_config.subsampling_x
        subsampling_y = seq_header.color_config.subsampling_y
        BitDepth = seq_header.color_config.BitDepth

        if plane == 0:
            subX = 0
            subY = 0
        else:
            subX = subsampling_x
            subY = subsampling_y

        for y in range(h):
            for x in range(w):
                if (not subX and not subY) or (tile_group.interintra and not tile_group.wedge_interintra):
                    m = self._Mask[y][x]
                elif subX and not subY:
                    m = Round2(self._Mask[y][2 * x] +
                               self._Mask[y][2 * x + 1], 1)
                else:
                    m = Round2(self._Mask[2 * y][2 * x] + self._Mask[2 * y][2 * x + 1] +
                               self._Mask[2 * y + 1][2 * x] + self._Mask[2 * y + 1][2 * x + 1], 2)
                if tile_group.interintra:
                    pred0 = Clip1(
                        Round2(preds[0][y][x], tile_group.InterPostRound), BitDepth)
                    pred1 = av1.CurrFrame[plane][y + dstY][x + dstX]
                    av1.CurrFrame[plane][y + dstY][x +
                                                   dstX] = Round2(m * pred1 + (64 - m) * pred0, 6)
                else:
                    pred0 = preds[0][y][x]
                    pred1 = preds[1][y][x]
                    av1.CurrFrame[plane][y + dstY][x + dstX] = Clip1(
                        Round2(m * pred0 + (64 - m) * pred1, 6 + tile_group.InterPostRound), BitDepth)

    def _distance_weights(self, av1: AV1Decoder, candRow: int, candCol: int):
        """
        Distance weights过程
        规范文档 7.11.3.15 Distance weights process

        Args:
            candRow: 候选行
            candCol: 候选列
        """
        from utils.frame_utils import get_relative_dist
        frame_header = av1.frame_header
        tile_group = av1.tile_group
        OrderHint = frame_header.OrderHint
        OrderHints = frame_header.OrderHints

        dist = [0, 0]
        for refList in range(2):
            h = OrderHints[tile_group.RefFrames[candRow][candCol][refList]]
            dist[refList] = Clip3(0, MAX_FRAME_DISTANCE, abs(
                get_relative_dist(av1, h, OrderHint)))
        d0 = dist[1]
        d1 = dist[0]
        order = d0 <= d1
        if d0 == 0 or d1 == 0:
            self._FwdWeight = Quant_Dist_Lookup[3][order]
            self._BckWeight = Quant_Dist_Lookup[3][1 - order]
        else:
            i = 0
            while i < 3:
                c0 = Quant_Dist_Weight[i][order]
                c1 = Quant_Dist_Weight[i][1 - order]
                if order:
                    if d0 * c0 > d1 * c1:
                        break
                else:
                    if d0 * c0 < d1 * c1:
                        break
                i += 1
            self._FwdWeight = Quant_Dist_Lookup[i][order]
            self._BckWeight = Quant_Dist_Lookup[i][1 - order]

    def predict_palette(self, av1: AV1Decoder, plane: int, startX: int, startY: int, x: int, y: int, txSz: int):
        """
        调色板预测过程
        规范文档 7.11.4 Palette prediction process

        Args:
            plane: 平面索引
            startX: 起始X坐标
            startY: 起始Y坐标
            x: X坐标
            y: Y坐标
            txSz: 变换块大小
        """
        tile_group = av1.tile_group

        w = Tx_Width[txSz]
        h = Tx_Height[txSz]
        if plane == 0:
            palette = tile_group.palette_colors_y
        elif plane == 1:
            palette = tile_group.palette_colors_u
        else:
            palette = tile_group.palette_colors_v

        if plane == 0:
            map = tile_group.ColorMapY
        else:
            map = tile_group.ColorMapUV

        av1.CurrFrame = Array(
            av1.CurrFrame, (PLANE_MAX, startY + h, startX + w))
        for i in range(h):
            for j in range(w):
                idx = map[y * 4 + i][x * 4 + j]
                av1.CurrFrame[plane][startY + i][startX + j] = palette[idx]

    def predict_chroma_from_luma(self, av1: AV1Decoder, plane: int, startX: int, startY: int,
                                 txSz: int):
        """
        从亮度预测色度过程
        规范文档 7.11.5 Predict chroma from luma process

        输入：
        - plane: 平面索引（1=U, 2=V）
        - baseX, baseY: 块左上角位置（以样本为单位）
        - log2W, log2H: 宽度和高度的对数（以2为底）

        此过程根据亮度平面生成色度预测块。

        Args:
            plane: 平面索引（1=U, 2=V）
            baseX: 基准X坐标
            baseY: 基准Y坐标
            log2W: 宽度对数
            log2H: 高度对数
        """
        seq_header = av1.seq_header
        tile_group = av1.tile_group
        subsampling_x = seq_header.color_config.subsampling_x
        subsampling_y = seq_header.color_config.subsampling_y
        BitDepth = seq_header.color_config.BitDepth

        w = Tx_Width[txSz]
        h = Tx_Height[txSz]
        subX = subsampling_x
        subY = subsampling_y

        if plane == 1:
            alpha = tile_group.CflAlphaU
        else:
            alpha = tile_group.CflAlphaV

        L = Array(None, (h, w), 0)
        lumaAvg = 0
        for i in range(h):
            lumaY = (startY + i) << subY
            lumaY = min(lumaY, tile_group.MaxLumaH - (1 << subY))
            for j in range(w):
                lumaX = (startX + j) << subX
                lumaX = min(lumaX, tile_group.MaxLumaW - (1 << subX))
                t = 0
                for dy in range(subY + 1):
                    for dx in range(subX + 1):
                        t += av1.CurrFrame[0][lumaY + dy][lumaX + dx]
                v = t << (3 - subX - subY)
                L[i][j] = v
                lumaAvg += v
        lumaAvg = Round2(lumaAvg, Tx_Width_Log2[txSz] + Tx_Height_Log2[txSz])

        for i in range(h):
            for j in range(w):
                dc = av1.CurrFrame[plane][startY + i][startX + j]
                scaledLuma = Round2Signed(alpha * (L[i][j] - lumaAvg), 6)
                av1.CurrFrame[plane][startY + i][startX +
                                                 j] = Clip1(dc + scaledLuma, BitDepth)


def rounding_variables_derivation(av1: AV1Decoder, isCompound: int):
    """
    舍入变量推导过程
    规范文档 7.11.3.2 Rounding variables derivation process

    Args:
        isCompound: 是否为复合预测（0=单预测，1=复合预测）
    """
    seq_header = av1.seq_header
    tile_group = av1.tile_group
    BitDepth = seq_header.color_config.BitDepth
    tile_group.InterRound0 = 3
    tile_group.InterRound1 = 7 if isCompound else 11
    if BitDepth == 12:
        tile_group.InterRound0 = tile_group.InterRound0 + 2
    if BitDepth == 12 and isCompound == 0:
        tile_group.InterRound1 = tile_group.InterRound1 - 2
    tile_group.InterPostRound = (2 * FILTER_BITS -
                                 (tile_group.InterRound0 + tile_group.InterRound1))
