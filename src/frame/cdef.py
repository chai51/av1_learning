"""
CDEF过程模块
实现规范文档7.15节"CDEF process"中描述的所有CDEF过程函数
包括：
- 7.15 CDEF process
- 7.15.1 CDEF block process
- 7.15.2 CDEF direction process
- 7.15.3 CDEF filter process
"""

from turtle import width
from typing import List
from constants import (
    MI_SIZE, MI_SIZE_LOG2, PLANE_MAX, SUB_SIZE,
    Num_4x4_Blocks_Wide,
    Cdef_Directions, Cdef_Pri_Taps, Cdef_Sec_Taps,
    Div_Table
)
from obu.decoder import AV1Decoder
from utils.tile_utils import is_inside_filter_region
from utils.math_utils import Array, Clip3, FloorLog2


class CdefProcess:
    def __init__(self, av1: AV1Decoder):
        seq_header = av1.seq_header
        height = seq_header.max_frame_height_minus_1 + 1
        width = seq_header.max_frame_width_minus_1 + 1

        self._CdefFrame: List[List[List[int]]] = Array(
            None, (PLANE_MAX, height, width))

    def cdef_process(self, av1: AV1Decoder) -> List[List[List[int]]]:
        """
        CDEF过程
        规范文档 7.15 CDEF process

        此过程对CurrFrame应用CDEF（Constrained Directional Enhancement Filter）滤波。

        Returns:
            CdefFrame - CDEF处理后的帧
        """
        frame_header = av1.frame_header
        tile_group = av1.tile_group
        MiRows = frame_header.MiRows
        MiCols = frame_header.MiCols

        step4 = Num_4x4_Blocks_Wide[SUB_SIZE.BLOCK_8X8]
        cdefSize4 = Num_4x4_Blocks_Wide[SUB_SIZE.BLOCK_64X64]
        cdefMask4 = ~(cdefSize4 - 1)

        for r in range(0, MiRows, step4):
            for c in range(0, MiCols, step4):
                baseR = r & cdefMask4
                baseC = c & cdefMask4

                idx = tile_group.cdef_idx[baseR][baseC]

                self.__cdef_block(av1, r, c, idx)

        return self._CdefFrame

    def __cdef_block(self, av1: AV1Decoder, r: int, c: int, idx: int):
        """
        CDEF块过程
        规范文档 7.15.1 CDEF block process

        此过程对指定的8x8块应用CDEF滤波。

        Args:
            r: MI行索引
            c: MI列索引
            idx: CDEF索引
            CdefFrame: CDEF帧（会被修改）
        """
        seq_header = av1.seq_header
        frame_header = av1.frame_header
        tile_group = av1.tile_group
        subsampling_x = seq_header.color_config.subsampling_x
        subsampling_y = seq_header.color_config.subsampling_y
        BitDepth = seq_header.color_config.BitDepth
        NumPlanes = seq_header.color_config.NumPlanes

        startY = r * MI_SIZE
        endY = startY + MI_SIZE * 2
        startX = c * MI_SIZE
        endX = startX + MI_SIZE * 2
        for y in range(startY, endY):
            for x in range(startX, endX):
                self._CdefFrame[0][y][x] = av1.CurrFrame[0][y][x]

        if NumPlanes > 1:
            startY = startY >> subsampling_y
            endY = endY >> subsampling_y
            startX = startX >> subsampling_x
            endX = endX >> subsampling_x

            for y in range(startY, endY):
                for x in range(startX, endX):
                    self._CdefFrame[1][y][x] = av1.CurrFrame[1][y][x]
                    self._CdefFrame[2][y][x] = av1.CurrFrame[2][y][x]

        if idx == -1:
            return

        coeffShift = BitDepth - 8
        tile_group.skip = tile_group.Skips[r][c] and tile_group.Skips[r +
                                                                      1][c] and tile_group.Skips[r][c + 1] and tile_group.Skips[r + 1][c + 1]

        if tile_group.skip == 0:
            yDir, var = self.cdef_direction_process(av1, r, c)
            # 1.
            priStr = frame_header.cdef_y_pri_strength[idx] << coeffShift
            # 2.
            secStr = frame_header.cdef_y_sec_strength[idx] << coeffShift
            # 3.
            dir_val = 0 if priStr == 0 else yDir
            # 4.
            varStr = min(FloorLog2(var >> 6), 12) if var >> 6 else 0
            # 5.
            priStr = ((priStr * (4 + varStr) + 8) >> 4) if var else 0
            # 6.
            damping = frame_header.CdefDamping + coeffShift
            # 7.
            self.cdef_filter_process(
                av1, 0, r, c, priStr, secStr, damping, dir_val)

            # 8.
            if NumPlanes == 1:
                return

            # 9.
            priStr = frame_header.cdef_uv_pri_strength[idx] << coeffShift
            # 10.
            secStr = frame_header.cdef_uv_sec_strength[idx] << coeffShift
            # 11.
            dir_val = 0 if priStr == 0 else Cdef_Uv_Dir[subsampling_x][subsampling_y][yDir]
            # 12.
            damping = frame_header.CdefDamping + coeffShift - 1
            # 13.
            self.cdef_filter_process(
                av1, 1, r, c, priStr, secStr, damping, dir_val)
            # 14.
            self.cdef_filter_process(
                av1, 2, r, c, priStr, secStr, damping, dir_val)

    def cdef_direction_process(self, av1: AV1Decoder, r: int, c: int) -> tuple:
        """
        CDEF方向过程
        规范文档 7.15.2 CDEF direction process

        此过程计算CDEF滤波的方向。

        Args:
            r: MI行索引
            c: MI列索引

        Returns:
            (yDir, var1) - 方向值和方差
        """
        seq_header = av1.seq_header
        BitDepth = seq_header.color_config.BitDepth

        cost = [0] * 8
        partial: List[List[int]] = Array(None, (8, 15), 0)
        bestCost = 0
        yDir = 0
        x0 = c << MI_SIZE_LOG2
        y0 = r << MI_SIZE_LOG2

        for i in range(8):
            for j in range(8):
                x = (av1.CurrFrame[0][y0 + i][x0 + j] >> (BitDepth - 8)) - 128
                partial[0][i + j] += x
                partial[1][i + j // 2] += x
                partial[2][i] += x
                partial[3][3 + i - j // 2] += x
                partial[4][7 + i - j] += x
                partial[5][3 - i // 2 + j] += x
                partial[6][j] += x
                partial[7][i // 2 + j] += x

        for i in range(8):
            cost[2] += partial[2][i] * partial[2][i]
            cost[6] += partial[6][i] * partial[6][i]

        cost[2] *= Div_Table[8]
        cost[6] *= Div_Table[8]
        for i in range(7):
            cost[0] += (partial[0][i] * partial[0][i] +
                        partial[0][14 - i] * partial[0][14 - i]) * Div_Table[i + 1]
            cost[4] += (partial[4][i] * partial[4][i] +
                        partial[4][14 - i] * partial[4][14 - i]) * Div_Table[i + 1]

        cost[0] += partial[0][7] * partial[0][7] * Div_Table[8]
        cost[4] += partial[4][7] * partial[4][7] * Div_Table[8]
        for i in range(1, 8, 2):
            for j in range(4 + 1):
                cost[i] += partial[i][3 + j] * partial[i][3 + j]
            cost[i] *= Div_Table[8]
            for j in range(4 - 1):
                cost[i] += (partial[i][j] * partial[i][j] +
                            partial[i][10 - j] * partial[i][10 - j]) * Div_Table[2 * j + 2]

        for i in range(8):
            if cost[i] > bestCost:
                bestCost = cost[i]
                yDir = i

        var = (bestCost - cost[(yDir + 4) & 7]) >> 10
        return (yDir, var)

    def cdef_filter_process(self, av1: AV1Decoder, plane: int, r: int, c: int, priStr: int, secStr: int,
                            damping: int, dir_val: int):
        """
        CDEF滤波过程
        规范文档 7.15.3 CDEF filter process

        此过程对指定的8x8块应用CDEF滤波。

        Args:
            plane: 平面索引
            r: MI行索引
            c: MI列索引
            priStr: 主强度
            secStr: 次强度
            damping: CDEF阻尼值
            dir_val: 方向值
        """
        seq_header = av1.seq_header
        subsampling_x = seq_header.color_config.subsampling_x
        subsampling_y = seq_header.color_config.subsampling_y
        BitDepth = seq_header.color_config.BitDepth

        coeffShift = BitDepth - 8

        subX = subsampling_x if (plane > 0) else 0
        subY = subsampling_y if (plane > 0) else 0

        x0 = (c * MI_SIZE) >> subX
        y0 = (r * MI_SIZE) >> subY
        w = 8 >> subX
        h = 8 >> subY

        CdefAvailable = 0

        def constrain(diff: int, threshold: int, damping: int) -> int:
            """
            约束函数
            规范文档 7.15.3 CDEF filter process

            此函数约束差值diff。

            Args:
                diff: 差值
                threshold: 阈值
                damping: 阻尼值

            Returns:
                约束后的值
            """
            if not threshold:
                return 0

            dampingAdj = max(0, damping - FloorLog2(threshold))
            sign = -1 if diff < 0 else 1
            return sign * Clip3(0, abs(diff), threshold - (abs(diff) >> dampingAdj))

        def cdef_get_at(plane: int, x0: int, y0: int, i: int, j: int,
                        dir_val: int, k: int, sign: int, subX: int, subY: int) -> int:
            """
            获取CDEF样本
            规范文档 7.15.3 CDEF filter process

            此函数根据方向获取相邻样本。

            Args:
                plane: 平面索引
                x0: X起始位置
                y0: Y起始位置
                i: 行偏移
                j: 列偏移
                dir_val: 方向值
                k: 索引
                sign: 符号（-1或1）
                subX: X子采样
                subY: Y子采样

            Returns:
                (sample, CdefAvailable) - 样本值和可用性标志
            """
            nonlocal CdefAvailable
            y = y0 + i + sign * Cdef_Directions[dir_val][k][0]
            x = x0 + j + sign * Cdef_Directions[dir_val][k][1]

            candidateR = (y << subY) >> MI_SIZE_LOG2
            candidateC = (x << subX) >> MI_SIZE_LOG2

            if is_inside_filter_region(av1, candidateR, candidateC):
                CdefAvailable = 1
                return av1.CurrFrame[plane][y][x]
            else:
                CdefAvailable = 0
                return 0

        for i in range(h):
            for j in range(w):
                sum_val = 0
                x = av1.CurrFrame[plane][y0 + i][x0 + j]
                max_val = x
                min_val = x

                for k in range(2):
                    for sign in range(-1, 2, 2):
                        p = cdef_get_at(plane, x0, y0, i, j,
                                        dir_val, k, sign, subX, subY)
                        if CdefAvailable:
                            sum_val += Cdef_Pri_Taps[(priStr >> coeffShift)
                                                     & 1][k] * constrain(p - x, priStr, damping)
                            max_val = max(p, max_val)
                            min_val = min(p, min_val)

                        for dirOff in range(-2, 3, 4):
                            s = cdef_get_at(
                                plane, x0, y0, i, j, (dir_val + dirOff) & 7, k, sign, subX, subY)
                            if CdefAvailable:
                                sum_val += Cdef_Sec_Taps[(priStr >> coeffShift) & 1][k] * constrain(
                                    s - x, secStr, damping)
                                max_val = max(s, max_val)
                                min_val = min(s, min_val)

                self._CdefFrame[plane][y0 + i][x0 + j] = Clip3(
                    min_val, max_val, x + ((8 + sum_val - (sum_val < 0)) >> 4))


"""
规范文档 7.15.1 CDEF block process
"""
Cdef_Uv_Dir = [
    [
        [0, 1, 2, 3, 4, 5, 6, 7],
        [1, 2, 2, 2, 3, 4, 6, 0],
    ],
    [
        [7, 0, 2, 4, 5, 6, 6, 6],
        [0, 1, 2, 3, 4, 5, 6, 7],
    ],
]
