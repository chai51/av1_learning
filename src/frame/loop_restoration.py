"""
环路恢复模块
实现规范文档7.17节"Loop restoration process"中的具体实现函数
"""

from typing import List
from constants import (
    FILTER_BITS, FRAME_RESTORATION_TYPE, NONE, SGRPROJ_MTABLE_BITS, SGRPROJ_RST_BITS,
    SGRPROJ_PRJ_BITS, SGRPROJ_SGR_BITS, SGRPROJ_RECIP_BITS, Sgr_Params,
    MI_SIZE
)
from obu.decoder import AV1Decoder
from reconstruction.prediction import rounding_variables_derivation
from utils.math_utils import Array, Clip1, Clip3, Round2
from utils.tile_utils import count_units_in_frame


class LoopRestoration:
    """
    环路恢复实现类
    包含除入口函数外的所有实现方法
    """

    def __init__(self, LrFrame: List[List[List[int]]]):
        self.LrFrame: List[List[List[int]]] = LrFrame
        self._StripeStartY = 0
        self._StripeEndY = 0
        self._PlaneEndX = 0
        self._PlaneEndY = 0

    def loop_restore_block(self, av1: AV1Decoder, plane: int, row: int, col: int):
        """
        环路恢复块过程
        规范文档 7.17.1 Loop restore block process

        Args:
            plane: 平面索引
            row: MI行索引
            col: MI列索引
        """
        seq_header = av1.seq_header
        frame_header = av1.frame_header
        tile_group = av1.tile_group
        subsampling_x = seq_header.color_config.subsampling_x
        subsampling_y = seq_header.color_config.subsampling_y

        lumaY = row * MI_SIZE
        stripeNum = (lumaY + 8) // 64

        if plane == 0:
            subX = 0
            subY = 0
        else:
            subX = subsampling_x
            subY = subsampling_y

        self._StripeStartY = (-8 + stripeNum * 64) >> subY
        self._StripeEndY = self._StripeStartY + (64 >> subY) - 1

        unitSize = frame_header.LoopRestorationSize[plane] if hasattr(
            frame_header, 'LoopRestorationSize') and plane < len(frame_header.LoopRestorationSize) else 0
        unitRows = count_units_in_frame(
            unitSize, Round2(frame_header.FrameHeight, subY))
        unitCols = count_units_in_frame(
            unitSize, Round2(frame_header.UpscaledWidth, subX))

        unitRow = min(unitRows - 1, ((row * MI_SIZE + 8) >> subY) // unitSize)
        unitCol = min(unitCols - 1, ((col * MI_SIZE) >> subX) // unitSize)

        self._PlaneEndX = Round2(frame_header.UpscaledWidth, subX) - 1
        self._PlaneEndY = Round2(frame_header.FrameHeight, subY) - 1
        x = (col * MI_SIZE) >> subX
        y = (row * MI_SIZE) >> subY

        w = min(MI_SIZE >> subX, self._PlaneEndX - x + 1)
        h = min(MI_SIZE >> subY, self._PlaneEndY - y + 1)

        rType = tile_group.LrType[plane][unitRow][unitCol]

        if rType == FRAME_RESTORATION_TYPE.RESTORE_WIENER:
            self.wiener_filter(av1, plane, unitRow, unitCol, x, y, w, h)
        elif rType == FRAME_RESTORATION_TYPE.RESTORE_SGRPROJ:
            self.self_guided_restoration(
                av1, plane, unitRow, unitCol, x, y, w, h)

    def self_guided_restoration(self, av1: AV1Decoder, plane: int, unitRow: int, unitCol: int, x: int, y: int, w: int, h: int):
        """
        自引导恢复过程
        规范文档 7.17.2 Self-guided restoration process

        Args:
            plane: 平面索引
            unitRow: 单元行索引
            unitCol: 单元列索引
            x: 起始X坐标
            y: 起始Y坐标
            w: 宽度
            h: 高度
        """
        seq_header = av1.seq_header
        tile_group = av1.tile_group
        BitDepth = seq_header.color_config.BitDepth

        # 1.
        set_val = tile_group.LrSgrSet[plane][unitRow][unitCol]
        # 2.
        pass_val = 0
        # 3.
        flt0 = self._box_filter(av1, plane, x, y, w, h, set_val, pass_val)
        # 4.
        pass_val = 1
        # 5.
        flt1 = self._box_filter(av1, plane, x, y, w, h, set_val, pass_val)

        w0 = tile_group.LrSgrXqd[plane][unitRow][unitCol][0]
        w1 = tile_group.LrSgrXqd[plane][unitRow][unitCol][1]
        w2 = (1 << SGRPROJ_PRJ_BITS) - w0 - w1
        r0 = Sgr_Params[set_val][0]
        r1 = Sgr_Params[set_val][2]

        for i in range(h):
            for j in range(w):
                u = av1.UpscaledCdefFrame[plane][y +
                                                 i][x + j] << SGRPROJ_RST_BITS
                v = w1 * u
                if r0:
                    v += w0 * flt0[i][j]
                else:
                    v += w0 * u
                if r1:
                    v += w2 * flt1[i][j]
                else:
                    v += w2 * u
                s = Round2(v, SGRPROJ_RST_BITS + SGRPROJ_PRJ_BITS)
                self.LrFrame[plane][y + i][x + j] = Clip1(s, BitDepth)

    def _box_filter(self, av1: AV1Decoder, plane: int, x: int, y: int, w: int, h: int, set_val: int, pass_val: int) -> List[List[int]]:
        """
        盒滤波过程
        规范文档 7.17.3 Box filter process

        Args:
            plane: 平面索引
            x: 起始X坐标
            y: 起始Y坐标
            w: 宽度
            h: 高度
            set_val: SGR参数集索引
            pass_val: pass索引（0或1）

        Returns:
            滤波后的结果（二维数组）
        """
        seq_header = av1.seq_header
        BitDepth = seq_header.color_config.BitDepth

        r = Sgr_Params[set_val][pass_val * 2 + 0]
        if r == 0:
            return NONE

        eps = Sgr_Params[set_val][pass_val * 2 + 1]

        A = Array(None, (h + 2, w + 2), 0)
        B = Array(None, (h + 2, w + 2), 0)
        n = (2 * r + 1) * (2 * r + 1)
        n2e = n * n * eps
        s = ((1 << SGRPROJ_MTABLE_BITS) + (n2e // 2)) // n2e
        for i in range(-1, h + 1):
            for j in range(-1, w + 1):
                a = 0
                b = 0
                for dy in range(-r, r + 1):
                    for dx in range(-r, r + 1):
                        c = self._get_source_sample(
                            av1, plane, x + j + dx, y + i + dy)
                        a += c * c
                        b += c
                a = Round2(a, 2 * (BitDepth - 8))
                d = Round2(b, BitDepth - 8)
                p = max(0, a * n - d * d)
                z = Round2(p * s, SGRPROJ_MTABLE_BITS)
                if z >= 255:
                    a2 = 256
                elif z == 0:
                    a2 = 1
                else:
                    a2 = ((z << SGRPROJ_SGR_BITS) + (z // 2)) // (z + 1)

                oneOverN = ((1 << SGRPROJ_RECIP_BITS) + (n // 2)) // n
                b2 = ((1 << SGRPROJ_SGR_BITS) - a2) * b * oneOverN
                A[i][j] = a2
                B[i][j] = Round2(b2, SGRPROJ_RECIP_BITS)

        F = Array(None, (h, w), 0)
        for i in range(h):
            shift = 5
            if pass_val == 0 and (i & 1):
                shift = 4

            for j in range(w):
                a = 0
                b = 0
                for dy in range(-1, 2):
                    for dx in range(-1, 2):
                        if pass_val == 0:
                            if (i + dy) & 1:
                                weight = 6 if dx == 0 else 5
                            else:
                                weight = 0
                        else:
                            weight = 4 if (dx == 0 or dy == 0) else 3
                        a += weight * A[i + dy][j + dx]
                        b += weight * B[i + dy][j + dx]

                v = a * av1.UpscaledCdefFrame[plane][y + i][x + j] + b
                F[i][j] = Round2(v, SGRPROJ_SGR_BITS +
                                 shift - SGRPROJ_RST_BITS)

        return F

    def wiener_filter(self, av1: AV1Decoder, plane: int, unitRow: int, unitCol: int, x: int, y: int, w: int, h: int):
        """
        Wiener滤波过程
        规范文档 7.17.4 Wiener filter process

        Args:
            plane: 平面索引
            unitRow: 单元行索引
            unitCol: 单元列索引
            x: 起始X坐标
            y: 起始Y坐标
            w: 宽度
            h: 高度
        """
        seq_header = av1.seq_header
        tile_group = av1.tile_group
        BitDepth = seq_header.color_config.BitDepth

        rounding_variables_derivation(av1, 0)

        vfilter = self._wiener_coefficient(
            tile_group.LrWiener[plane][unitRow][unitCol][0])
        hfilter = self._wiener_coefficient(
            tile_group.LrWiener[plane][unitRow][unitCol][1])

        offset = 1 << (BitDepth + FILTER_BITS - tile_group.InterRound0 - 1)
        limit = (1 << (BitDepth + 1 + FILTER_BITS - tile_group.InterRound0)) - 1
        intermediate = Array(None, (h + 6, w), 0)
        for r in range(h + 6):
            for c in range(w):
                s = 0
                for t in range(7):
                    s += hfilter[t] * self._get_source_sample(
                        av1, plane, x + c + t - 3, y + r - 3)
                v = Round2(s, tile_group.InterRound0)
                intermediate[r][c] = Clip3(-offset, limit - offset, v)

        for r in range(h):
            for c in range(w):
                s = 0
                for t in range(7):
                    s += vfilter[t] * intermediate[r + t][c]
                v = Round2(s, tile_group.InterRound1)
                self.LrFrame[plane][y + r][x + c] = Clip1(v, BitDepth)

    def _wiener_coefficient(self, coeff: List[int]) -> List[int]:
        """
        Wiener滤波系数计算
        规范文档 7.17.5 Wiener coefficient process

        Args:
            coeff: Wiener滤波系数

        Returns:
            filter - 7抽头滤波器系数
        """
        filter = [0] * 7
        filter[3] = 128
        for i in range(3):
            c = coeff[i]
            filter[i] = c
            filter[6 - i] = c
            filter[3] -= 2 * c
        return filter

    def _get_source_sample(self, av1: AV1Decoder, plane: int, x: int, y: int) -> int:
        """
        获取源样本过程
        规范文档 7.17.6 Get source sample process

        Args:
            plane: 平面索引
            x: X坐标
            y: Y坐标

        Returns:
            源样本值
        """
        x = min(self._PlaneEndX, x)
        x = max(0, x)
        y = min(self._PlaneEndY, y)
        y = max(0, y)
        if y < self._StripeStartY:
            y = max(self._StripeStartY - 2, y)
            return av1.UpscaledCurrFrame[plane][y][x]
        elif y > self._StripeEndY:
            y = min(self._StripeEndY + 2, y)
            return av1.UpscaledCurrFrame[plane][y][x]
        else:
            return av1.UpscaledCdefFrame[plane][y][x]
