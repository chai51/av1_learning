"""
重建过程模块
实现规范文档7.12节"Reconstruction processes"中描述的所有重建过程函数
包括：
- 7.12.1 Dequantization process
- 7.12.2 Inverse transform process
- 7.12.3 Reconstruct process
"""

from copy import deepcopy
from typing import List, Optional, TYPE_CHECKING
# 变换类型相关常量
from constants import (
    DCT_DCT, ADST_DCT, DCT_ADST, ADST_ADST, MAX_TILE_COLS, MAX_TILE_ROWS, SUB_SIZE, TX_SIZE,
    V_DCT, H_DCT, V_ADST, H_ADST, IDTX,
    Dc_Qlookup, Ac_Qlookup, Transform_Row_Shift,
    Tx_Width_Log2, Tx_Height_Log2, Quantizer_Matrix, Qm_Offset,
    Cos128_Lookup,
    FLIPADST_DCT, DCT_FLIPADST, FLIPADST_ADST, ADST_FLIPADST,
    FLIPADST_FLIPADST, V_FLIPADST, H_FLIPADST,
    SEG_LVL_ALT_Q
)
from obu.decoder import AV1Decoder
from utils.math_utils import Array, bits_signed, Clip1, Clip3, Round2


def dc_q(av1: AV1Decoder, b: int) -> int:
    """
    DC量化查找函数
    规范文档 7.12.2 Dequantization functions - dc_q()

    Args:
        b: 量化系数

    Returns:
        反量化后的DC量化系数
    """
    seq_header = av1.seq_header
    BitDepth = seq_header.color_config.BitDepth
    return Dc_Qlookup[(BitDepth-8) >> 1][Clip3(0, 255, b)]


def ac_q(av1: AV1Decoder, b: int) -> int:
    """
    AC量化查找函数
    规范文档 7.12.2 Dequantization functions - ac_q()

    Args:
        b: 量化系数

    Returns:
        反量化后的AC量化系数
    """
    seq_header = av1.seq_header
    BitDepth = seq_header.color_config.BitDepth
    return Ac_Qlookup[(BitDepth-8) >> 1][Clip3(0, 255, b)]


def get_dc_quant(av1: AV1Decoder, plane: int) -> int:
    """
    获取DC量化系数
    规范文档 7.12.2 Dequantization functions - get_dc_quant()

    Args:
        plane: 平面索引

    Returns:
        DC量化系数
    """
    from utils.frame_utils import get_qindex
    frame_header = av1.frame_header
    tile_group = av1.tile_group

    if plane == 0:
        return dc_q(av1, get_qindex(av1, 0, tile_group.segment_id) + frame_header.DeltaQYDc)
    elif plane == 1:
        return dc_q(av1, get_qindex(av1, 0, tile_group.segment_id) + frame_header.DeltaQUDc)
    else:
        return dc_q(av1, get_qindex(av1, 0, tile_group.segment_id) + frame_header.DeltaQVDc)


def get_ac_quant(av1: AV1Decoder, plane: int) -> int:
    """
    获取AC量化系数
    规范文档 7.12.2 Dequantization functions - get_ac_quant()

    Args:
        plane: 平面索引

    Returns:
        AC量化系数
    """
    from utils.frame_utils import get_qindex
    frame_header = av1.frame_header
    tile_group = av1.tile_group

    if plane == 0:
        return ac_q(av1, get_qindex(av1, 0, tile_group.segment_id))
    elif plane == 1:
        return ac_q(av1, get_qindex(av1, 0, tile_group.segment_id) + frame_header.DeltaQUAc)
    else:
        return ac_q(av1, get_qindex(av1, 0, tile_group.segment_id) + frame_header.DeltaQVAc)


def reconstruct_process(av1: AV1Decoder, plane: int, x: int, y: int, txSz: int) -> None:
    """
    重建过程
    规范文档 7.12.3 Reconstruct process

    Args:
        plane: 平面索引
        x: X坐标
        y: Y坐标
        txSz: 变换块大小

    Returns:
        重建后的样本数组
    """
    seq_header = av1.seq_header
    frame_header = av1.frame_header
    tile_group = av1.tile_group
    BitDepth = seq_header.color_config.BitDepth
    inverse_transform = InverseTransform(av1)

    if txSz in [TX_SIZE.TX_32X32, TX_SIZE.TX_16X32, TX_SIZE.TX_32X16, TX_SIZE.TX_16X64, TX_SIZE.TX_64X16]:
        dqDenom = 2
    elif txSz in [TX_SIZE.TX_64X64, TX_SIZE.TX_32X64, TX_SIZE.TX_64X32]:
        dqDenom = 4
    else:
        dqDenom = 1

    log2W = Tx_Width_Log2[txSz]
    log2H = Tx_Height_Log2[txSz]
    w = 1 << log2W
    h = 1 << log2H
    tw = min(32, w)
    th = min(32, h)

    if tile_group.PlaneTxType in [FLIPADST_DCT, FLIPADST_ADST, V_FLIPADST, FLIPADST_FLIPADST]:
        flipUD = 1
    else:
        flipUD = 0

    if tile_group.PlaneTxType in [DCT_FLIPADST, ADST_FLIPADST, H_FLIPADST, FLIPADST_FLIPADST]:
        flipLR = 1
    else:
        flipLR = 0

    dq = 0
    sign = 0
    dq2 = 0
    # 1. 解码DC 和 AC系数
    for i in range(th):
        for j in range(tw):
            # a.
            if i == 0 and j == 0:
                q = get_dc_quant(av1, plane)
            else:
                q = get_ac_quant(av1, plane)

            # b.
            if frame_header.using_qmatrix == 1 and tile_group.PlaneTxType < IDTX and frame_header.SegQMLevel[plane][tile_group.segment_id] < 15:
                q2 = Round2(q * Quantizer_Matrix[frame_header.SegQMLevel[plane]
                            [tile_group.segment_id]][plane > 0][Qm_Offset[txSz] + i * tw + j], 5)
            else:
                q2 = q

            # c.
            dq = tile_group.Quant[i * tw + j] * q2

            # d.
            sign = -1 if dq < 0 else 1

            # e.
            dq2 = sign * ((abs(dq) & 0xFFFFFF) // dqDenom)

            # f.
            tile_group.Dequant[i][j] = Clip3(- (1 << (7 + BitDepth)),
                                             (1 << (7 + BitDepth)) - 1, dq2)

    # 2.
    if tile_group.Lossless == 1:
        tile_group.Residual = Array(None, (h, w), 1 + BitDepth)
    else:
        inverse_transform.inverse_2d_transform_process(av1, txSz)

    # 3.
    for i in range(h):
        for j in range(w):
            xx = (w - j - 1) if flipLR else j
            yy = (h - i - 1) if flipUD else i
            av1.CurrFrame[plane][y + yy][x + xx] = Clip1(
                av1.CurrFrame[plane][y + yy][x + xx] + tile_group.Residual[i][j], BitDepth)


class InverseTransform:
    def __init__(self, av1: AV1Decoder):
        self.T: List[int] = [0] * MAX_TILE_COLS

    def _brev(self, numBits: int, x: int) -> int:
        """
        位反转函数
        规范文档 7.13.2.1 Butterfly functions - brev()

        Args:
            numBits: 位数
            x: 要反转的值
        """
        t = 0
        for i in range(numBits):
            bit = (x >> i) & 1
            t += bit << (numBits - 1 - i)

        return t

    def _cos128(self, angle: int) -> int:
        """
        余弦查找函数
        规范文档 7.13.2.1 Butterfly functions - cos128()

        Args:
            angle: 角度值
        """
        # 1.
        angle2 = angle & 255

        # 2.
        if 0 <= angle2 <= 64:
            return Cos128_Lookup[angle2]

        # 3.
        elif 64 < angle2 <= 128:
            return Cos128_Lookup[128 - angle2] * -1

        # 4.
        elif 128 < angle2 <= 192:
            return Cos128_Lookup[angle2 - 128] * -1

        # 5.
        else:
            return Cos128_Lookup[256 - angle2]

    def _sin128(self, angle: int) -> int:
        """
        正弦查找函数
        规范文档 7.13.2.1 Butterfly functions - sin128()

        Args:
            angle: 角度值
        """
        return self._cos128(angle - 64)

    def _B(self, a: int, b: int, angle: int, _1: int, r: int):
        """
        Butterfly函数B0
        规范文档 7.13.2.1 Butterfly functions - B0()

        Args:
            a: 索引a
            b: 索引b
            angle: 角度
            _1: 交换标志
            r: 精度位数
        """
        def B0(a: int, b: int, angle: int, r: int):
            # 1.
            x = (self.T[a] * self._cos128(angle) -
                 self.T[b] * self._sin128(angle))
            # 2.
            y = (self.T[a] * self._sin128(angle) +
                 self.T[b] * self._cos128(angle))
            # 3.
            self.T[a] = Round2(x, 12)
            # 4.
            self.T[b] = Round2(y, 12)

            # It is a requirement of bitstream conformance that the values saved into the array T by this function are representable by a signed integer using r bits of precision.
            self.T[a] = bits_signed(self.T[a], r)
            self.T[b] = bits_signed(self.T[b], r)

        B0(a, b, angle, r)

        if _1:
            self.T[a], self.T[b] = self.T[b], self.T[a]

    def _H(self, a: int, b: int, _1: int, r: int):
        """
        Butterfly函数H
        规范文档 7.13.2.1 Butterfly functions - H()

        Args:
            a: 索引a
            b: 索引b
            _1: 交换标志
            r: 精度位数
        """
        def H0(a: int, b: int, r: int):
            # 1.
            x = self.T[a]
            # 2.
            y = self.T[b]
            # 3.
            self.T[a] = Clip3(-(1 << (r - 1)), (1 << (r - 1)) - 1, x + y)
            # 4.
            self.T[b] = Clip3(-(1 << (r - 1)), (1 << (r - 1)) - 1, x - y)

        if _1 == 1:
            return H0(b, a, r)
        else:
            return H0(a, b, r)

    def inverse_dct_array_permutation(self, n: int):
        """
        Inverse DCT array permutation process
        规范文档 7.13.2.2 Inverse DCT array permutation process

        Args:
            n: 长度
        """
        copyT = deepcopy(self.T)
        for i in range(1 << n):
            self.T[i] = copyT[self._brev(n, i)]

    def inverse_dct_process(self, n: int, r: int):
        """
        Inverse DCT process
        规范文档 7.13.2.3 Inverse DCT process

        Args:
            n: 长度
            r: 精度位数
        """

        # 1.
        self.inverse_dct_array_permutation(n)

        # 2.
        if n == 6:
            for i in range(16):
                self._B(32 + i, 63 - i, 63 - 4 * self._brev(4, i), 0, r)

        # 3.
        if n >= 5:
            for i in range(8):
                self._B(16 + i, 31 - i, 6 + (self._brev(3, 7 - i) << 3), 0, r)

        # 4.
        if n == 6:
            for i in range(16):
                self._H(32 + i * 2, 33 + i * 2, i & 1, r)

        # 5.
        if n >= 4:
            for i in range(4):
                self._B(8 + i, 15 - i, 12 + (self._brev(2, 3 - i) << 4), 0, r)

        # 6.
        if n >= 5:
            for i in range(8):
                self._H(16 + 2 * i, 17 + 2 * i, i & 1, r)

        # 7.
        if n == 6:
            for i in range(4):
                for j in range(2):
                    self._B(62 - i * 4 - j, 33 + i * 4 + j, 60 -
                            16 * self._brev(2, i) + 64 * j, 1, r)

        # 8.
        if n >= 3:
            for i in range(2):
                self._B(4 + i, 7 - i, 56 - 32 * i, 0, r)

        # 9.
        if n >= 4:
            for i in range(4):
                self._H(8 + 2 * i, 9 + 2 * i, i & 1, r)

        # 10.
        if n >= 5:
            for i in range(2):
                for j in range(2):
                    self._B(30 - 4 * i - j, 17 + 4 * i + j,
                            24 + (j << 6) + ((1 - i) << 5), 1, r)

        # 11.
        if n == 6:
            for i in range(8):
                for j in range(2):
                    self._H(32 + i * 4 + j, 35 + i * 4 - j, i & 1, r)

        # 12.
        for i in range(2):
            self._B(2 * i, 2 * i + 1, 32 + 16 * i, 1 - i, r)

        # 13.
        if n >= 3:
            for i in range(2):
                self._H(4 + 2 * i, 5 + 2 * i, i, r)

        # 14.
        if n >= 4:
            for i in range(2):
                self._B(14 - i, 9 + i, 48 + 64 * i, 1, r)

        # 15.
        if n >= 5:
            for i in range(4):
                for j in range(2):
                    self._H(16 + 4 * i + j, 19 + 4 * i - j, i & 1, r)

        # 16.
        if n == 6:
            for i in range(2):
                for j in range(4):
                    self._B(61 - i * 8 - j, 34 + i * 8 + j,
                            56 - i * 32 + (j >> 1) * 64, 1, r)

        # 17.
        for i in range(2):
            self._H(i, 3 - i, 0, r)

        # 18.
        if n >= 3:
            self._B(6, 5, 32, 1, r)

        # 19.
        if n >= 4:
            for i in range(2):
                for j in range(2):
                    self._H(8 + 4 * i + j, 11 + 4 * i - j, i, r)

        # 20.
        if n >= 5:
            for i in range(4):
                self._B(29 - i, 18 + i, 48 + (i >> 1) * 64, 1, r)

        # 21.
        if n == 6:
            for i in range(4):
                for j in range(4):
                    self._H(32 + 8 * i + j, 39 + 8 * i - j, i & 1, r)

        # 22.
        if n >= 3:
            for i in range(4):
                self._H(i, 7 - i, 0, r)

        # 23.
        if n >= 4:
            for i in range(2):
                self._B(13 - i, 10 + i, 32, 1, r)

        # 24.
        if n >= 5:
            for i in range(2):
                for j in range(4):
                    self._H(16 + i * 8 + j, 23 + i * 8 - j, i, r)

        # 25.
        if n == 6:
            for i in range(8):
                self._B(59 - i, 36 + i, 48 if i < 4 else 112, 1, r)

        # 26.
        if n >= 4:
            for i in range(8):
                self._H(i, 15 - i, 0, r)

        # 27.
        if n >= 5:
            for i in range(4):
                self._B(27 - i, 20 + i, 32, 1, r)

        # 28.
        if n == 6:
            for i in range(8):
                self._H(32 + i, 47 - i, 0, r)
                self._H(48 + i, 63 - i, 1, r)

        # 29.
        if n >= 5:
            for i in range(16):
                self._H(i, 31 - i, 0, r)

        # 30.
        if n == 6:
            for i in range(8):
                self._B(55 - i, 40 + i, 32, 1, r)

        # 31.
        if n == 6:
            for i in range(32):
                self._H(i, 63 - i, 0, r)

    def inverse_adst_input_array_permutation(self, n: int):
        """
        Inverse ADST input array permutation process
        规范文档 7.13.2.4 Inverse ADST input array permutation process

        Args:
            n: 长度
        """
        copyT = deepcopy(self.T)
        n0 = 1 << n
        for i in range(n0):
            idx = (i - 1) if (i & 1) else (n0 - i - 1)
            self.T[i] = copyT[idx]

    def inverse_adst_output_array_permutation(self, n: int):
        """
        Inverse ADST output array permutation process
        规范文档 7.13.2.5 Inverse ADST output array permutation process

        Args:
            n: 长度
        """
        n0 = 1 << n
        copyT = deepcopy(self.T)
        for i in range(n0):
            a = (i >> 3) & 1
            b = ((i >> 2) & 1) ^ ((i >> 3) & 1)
            c = ((i >> 1) & 1) ^ ((i >> 2) & 1)
            d = (i & 1) ^ ((i >> 1) & 1)
            idx = ((d << 3) | (c << 2) | (b << 1) | a) >> (4 - n)
            self.T[i] = -copyT[idx] if (i & 1) else copyT[idx]

    def inverse_adst4_process(self, r: int):
        """
        Inverse ADST4 process
        规范文档 7.13.2.6 Inverse ADST4 process

        Args:
            r: 精度位数
        """
        s = [0] * 7
        x = [0] * 4

        SINPI_1_9 = 1321
        SINPI_2_9 = 2482
        SINPI_3_9 = 3344
        SINPI_4_9 = 3803

        s[0] = SINPI_1_9 * self.T[0]
        s[1] = SINPI_2_9 * self.T[0]
        s[2] = SINPI_3_9 * self.T[1]
        s[3] = SINPI_4_9 * self.T[2]
        s[4] = SINPI_1_9 * self.T[2]
        s[5] = SINPI_2_9 * self.T[3]
        s[6] = SINPI_4_9 * self.T[3]

        # It is a requirement of bitstream conformance that values stored in the variable a7 by this process are representable by a signed integer using r + 1 bits of precision.
        a7 = bits_signed(self.T[0] - self.T[2], r + 1)
        # It is a requirement of bitstream conformance that values stored in the variable b7 by this process are representable by a signed integer using r bits of precision.
        b7 = bits_signed(a7 + self.T[3], r)

        s[0] = s[0] + s[3]
        s[1] = s[1] - s[4]
        s[3] = s[2]
        s[2] = SINPI_3_9 * b7

        s[0] = s[0] + s[5]
        s[1] = s[1] - s[6]
        # It is a requirement of bitstream conformance that all values stored in the s and x arrays by this process are representable by a signed integer using r + 12 bits of precision.
        s = [bits_signed(v, r + 12) for v in s]

        x[0] = s[0] + s[3]
        x[1] = s[1] + s[3]
        x[2] = s[2]
        x[3] = s[0] + s[1]

        x[3] = x[3] - s[3]
        # It is a requirement of bitstream conformance that all values stored in the s and x arrays by this process are representable by a signed integer using r + 12 bits of precision.
        x = [bits_signed(v, r + 12) for v in x]

        self.T[0] = Round2(x[0], 12)
        self.T[1] = Round2(x[1], 12)
        self.T[2] = Round2(x[2], 12)
        self.T[3] = Round2(x[3], 12)

    def inverse_adst8_process(self, r: int):
        """
        Inverse ADST8 process
        规范文档 7.13.2.7 Inverse ADST8 process

        Args:
            r: 精度位数
        """
        # 1.
        self.inverse_adst_input_array_permutation(3)

        # 2.
        for i in range(4):
            self._B(2 * i, 2 * i + 1, 60 - 16 * i, 1, r)

        # 3.
        for i in range(4):
            self._H(i, 4 + i, 0, r)

        # 4.
        for i in range(2):
            self._B(4 + 3 * i, 5 + i, 48 - 32 * i, 1, r)

        # 5.
        for i in range(2):
            for j in range(2):
                self._H(4 * j + i, 2 + 4 * j + i, 0, r)

        # 6.
        for i in range(2):
            self._B(2 + 4 * i, 3 + 4 * i, 32, 1, r)

        # 7.
        self.inverse_adst_output_array_permutation(3)

    def inverse_adst16_process(self, r: int):
        """
        Inverse ADST16 process
        规范文档 7.13.2.8 Inverse ADST16 process

        Args:
            r: 精度位数
        """
        # 1.
        self.inverse_adst_input_array_permutation(4)

        # 2.
        for i in range(8):
            self._B(2 * i, 2 * i + 1, 62 - 8 * i, 1, r)

        # 3.
        for i in range(8):
            self._H(i, 8 + i, 0, r)

        # 4.
        for i in range(2):
            self._B(8 + 2 * i, 9 + 2 * i, 56 - 32 * i, 1, r)
            self._B(13 + 2 * i, 12 + 2 * i, 8 + 32 * i, 1, r)

        # 5.
        for i in range(4):
            for j in range(2):
                self._H(8 * j + i, 4 + 8 * j + i, 0, r)

        # 6.
        for i in range(2):
            for j in range(2):
                self._B(4 + 8 * j + 3 * i, 5 + 8 * j + i, 48 - 32 * i, 1, r)

        # 7.
        for i in range(2):
            for j in range(4):
                self._H(4 * j + i, 2 + 4 * j + i, 0, r)

        # 8.
        for i in range(4):
            self._B(2 + 4 * i, 3 + 4 * i, 32, 1, r)

        # 9.
        self.inverse_adst_output_array_permutation(4)

    def inverse_adst_process(self, n: int, r: int):
        """
        Inverse ADST process
        规范文档 7.13.2.9 Inverse ADST process

        Args:
            n: 长度
            r: 精度位数
        """
        if n == 2:
            self.inverse_adst4_process(r)
        elif n == 3:
            self.inverse_adst8_process(r)
        else:
            self.inverse_adst16_process(r)

    def inverse_walsh_hadamard_transform_process(self, shift: int):
        """
        Inverse Walsh-Hadamard transform process
        规范文档 7.13.2.10 Inverse Walsh-Hadamard transform process

        Args:
            shift: 移位量
        """
        a = self.T[0] >> shift
        c = self.T[1] >> shift
        d = self.T[2] >> shift
        b = self.T[3] >> shift
        a += c
        d -= b
        e = (a - d) >> 1
        b = e - b
        c = e - c
        a -= b
        d += c
        self.T[0] = a
        self.T[1] = b
        self.T[2] = c
        self.T[3] = d

    def inverse_identity_transform_4_process(self):
        """
        Inverse identity transform 4 process
        规范文档 7.13.2.11 Inverse identity transform 4 process
        """
        for i in range(4):
            self.T[i] = Round2(self.T[i] * 5793, 12)

    def inverse_identity_transform_8_process(self):
        """
        Inverse identity transform 8 process
        规范文档 7.13.2.12 Inverse identity transform 8 process
        """
        for i in range(8):
            self.T[i] = self.T[i] * 2

    def inverse_identity_transform_16_process(self):
        """
        Inverse identity transform 16 process
        规范文档 7.13.2.13 Inverse identity transform 16 process
        """
        for i in range(16):
            self.T[i] = Round2(self.T[i] * 11586, 12)

    def inverse_identity_transform_32_process(self):
        """
        Inverse identity transform 32 process
        规范文档 7.13.2.14 Inverse identity transform 32 process
        """
        for i in range(32):
            self.T[i] = self.T[i] * 4

    def inverse_identity_transform_process(self, n: int):
        """
        Inverse identity transform process
        规范文档 7.13.2.15 Inverse identity transform process
        """
        if n == 2:
            self.inverse_identity_transform_4_process()
        elif n == 3:
            self.inverse_identity_transform_8_process()
        elif n == 4:
            self.inverse_identity_transform_16_process()
        else:
            self.inverse_identity_transform_32_process()

    def inverse_2d_transform_process(self, av1: AV1Decoder, txSz: int):
        """
        规范文档 7.13.3 2D inverse transform process
        """
        seq_header = av1.seq_header
        tile_group = av1.tile_group
        BitDepth = seq_header.color_config.BitDepth
        log2W = Tx_Width_Log2[txSz]
        log2H = Tx_Height_Log2[txSz]
        w = 1 << log2W
        h = 1 << log2H
        rowShift = 0 if tile_group.Lossless else Transform_Row_Shift[txSz]
        colShift = 0 if tile_group.Lossless else 4
        rowClampRange = BitDepth + 8
        colClampRange = max(BitDepth + 6, 16)

        # 1.
        for i in range(h):
            for j in range(w):
                if i < 32 and j < 32:
                    self.T[j] = tile_group.Dequant[i][j]
                else:
                    self.T[j] = 0

            if abs(log2W - log2H) == 1:
                for j in range(w):
                    self.T[j] = Round2(self.T[j] * 2896, 12)

            if tile_group.Lossless == 1:
                self.inverse_walsh_hadamard_transform_process(2)

            elif tile_group.PlaneTxType in [DCT_DCT, ADST_DCT, FLIPADST_DCT, H_DCT]:
                self.inverse_dct_process(log2W, rowClampRange)

            elif tile_group.PlaneTxType in [DCT_ADST, ADST_ADST, DCT_FLIPADST, FLIPADST_FLIPADST, ADST_FLIPADST, FLIPADST_ADST, H_ADST, H_FLIPADST]:
                self.inverse_adst_process(log2W, rowClampRange)

            else:
                self.inverse_identity_transform_process(log2W)

            for j in range(w):
                tile_group.Residual[i][j] = Round2(self.T[j], rowShift)

        # 2.
        for i in range(h):
            for j in range(w):
                tile_group.Residual[i][j] = Clip3(-(1 << (colClampRange - 1)), (1 << (
                    colClampRange - 1)) - 1, tile_group.Residual[i][j])

        # 3.
        for j in range(w):
            for i in range(h):
                self.T[i] = tile_group.Residual[i][j]

            if tile_group.Lossless == 1:
                self.inverse_walsh_hadamard_transform_process(0)

            elif tile_group.PlaneTxType in [DCT_DCT, DCT_ADST, DCT_FLIPADST, V_DCT]:
                self.inverse_dct_process(log2H, colClampRange)

            elif tile_group.PlaneTxType in [ADST_DCT, ADST_ADST, FLIPADST_DCT, FLIPADST_FLIPADST, ADST_FLIPADST, FLIPADST_ADST, V_ADST, V_FLIPADST]:
                self.inverse_adst_process(log2H, colClampRange)

            else:
                self.inverse_identity_transform_process(log2H)

            for i in range(h):
                tile_group.Residual[i][j] = Round2(self.T[i], colShift)

        return tile_group.Residual
