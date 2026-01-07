"""
符号解码器（Symbol Decoder）
按照规范文档8.2节实现熵解码功能
"""

from typing import List, Dict, Any, Optional
from copy import deepcopy
from bitstream.descriptors import read_f
from constants import EC_MIN_PROB, EC_PROB_SHIFT, FRAME_LF_COUNT, INT16_MAX, MV_CONTEXTS, NONE, NUM_REF_FRAMES
import constants
from entropy import default_cdfs
from obu.decoder import AV1Decoder
from bitstream.bit_reader import BitReader
from utils.math_utils import FloorLog2


class SymbolDecoder:
    """
    符号解码器
    实现规范文档8.2节描述的符号解码功能
    """

    def __init__(self):
        """
        初始化符号解码器
        """
        self.SymbolValue = 0
        self.SymbolRange = 0
        self.SymbolMaxBits = 0

        self.tile_cdfs: Dict[str, Any] = {}
        self.saved_cdfs: Dict[str, Any] = {}
        self.reader: BitReader = NONE

    def init_symbol(self, av1: AV1Decoder, sz: int):
        """
        初始化符号解码器
        规范文档 8.2.2 Initialization process for symbol decoder

        Args:
            sz: 要读取的字节数
        """
        self.reader = BitReader(av1.reader.read_bytes(sz))
        reader = self.reader
        ref_frame_store = av1.ref_frame_store

        # Note: The bit position will always be byte aligned when init_symbol is invoked because the uncompressed header and the data partitions are always a whole number of bytes long.
        assert reader.get_position() % 8 == 0

        numBits = min(sz * 8, 15)
        buf = read_f(reader, numBits)
        paddedBuf = buf << (15 - numBits)
        self.SymbolValue = INT16_MAX ^ paddedBuf
        self.SymbolRange = 1 << 15
        self.SymbolMaxBits = 8 * sz - 15

        # Note: Implementations may prefer to store the inverse cdf to move the subtraction out of this loop.
        cdf = deepcopy(default_cdfs.Default_Intra_Frame_Y_Mode_Cdf)
        self.tile_cdfs['TileIntraFrameYModeCdf'] = inverseCdf(cdf)

        for name, cdf in ref_frame_store.cdfs.items():
            self.tile_cdfs[f'Tile{name}'] = deepcopy(cdf)

    def read_bool(self, av1: AV1Decoder) -> int:
        """
        读取布尔值
        规范文档 8.2.3 Boolean decoding process

        Returns:
            解码的布尔值（0或1）
        """

        # Note: Implementations may prefer to store the inverse cdf to move the subtraction out of this loop.
        cdf = [1 << 14, 1 << 15, 0]
        inverseCdf(cdf)

        symbol = self.read_symbol(av1, cdf, update_cdf=False)
        return symbol

    def exit_symbol(self, av1: AV1Decoder):
        """
        退出符号解码器
        规范文档 8.2.4 Exit process for symbol decoder

        """
        reader = self.reader
        frame_header = av1.frame_header
        ref_frame_store = av1.ref_frame_store

        # It is a requirement of bitstream conformance that SymbolMaxBits is greater than or equal to -14 whenever this process is invoked.
        assert self.SymbolMaxBits >= -14

        trailingBitPosition = reader.get_position() - min(15, self.SymbolMaxBits + 15)
        reader.set_offset(max(0, self.SymbolMaxBits))
        paddingEndPosition = reader.get_position()

        # Note: paddingEndPosition will always be a multiple of 8 indicating that the bit position is byte aligned.
        assert paddingEndPosition % 8 == 0

        # It is a requirement of bitstream conformance that the bit at position trailingBitPosition is equal to 1.
        reader.set_position(trailingBitPosition)
        assert read_f(reader, 1) == 1

        # It is a requirement of bitstream conformance that the bit at position x is equal to 0 for values of x strictly between trailingBitPosition and paddingEndPosition.
        if reader.get_position() < paddingEndPosition:
            assert read_f(reader, 1) == 0

        if frame_header.disable_frame_end_update_cdf == 0 and frame_header.TileNum == frame_header.context_update_tile_id:
            for name, cdf in self.tile_cdfs.items():
                # 将Tile前缀名称转换为Saved前缀名称
                saved_name = 'Saved' + name[4:]
                self.saved_cdfs[saved_name] = cdf

    def read_literal(self, av1: AV1Decoder, n: int) -> int:
        """
        读取字面量
        规范文档 8.2.5 parsing_process_for_read_literal()

        Args:
            n: 要读取的位数

        Returns:
            解码的n位无符号整数
        """
        x = 0
        for i in range(n):
            x = 2 * x + self.read_bool(av1)
        return x

    def read_symbol(self, av1: AV1Decoder, cdf: List[int], update_cdf: bool = True) -> int:
        """
        读取符号
        规范文档 8.2.6 Symbol decoding process

        Args:
            cdf: 累积分布函数数组（长度为N+1，N为符号数量）
            update_cdf: 是否更新CDF（默认True，但会被disable_cdf_update覆盖）

        Returns:
            解码的符号值
        """
        reader = self.reader
        frame_header = av1.frame_header
        N = len(cdf) - 1

        """
        Note: When this process is invoked, N will be greater than 1 and cdf[ N-1 ] will be equal to 1 << 15.
        Note: Implementations may prefer to store the inverse cdf to move the subtraction out of this loop.
        """
        # assert N > 1 and cdf[N - 1] == 1 << 15
        assert N > 1 and cdf[N - 1] == inverseCdf(1 << 15)

        cur = self.SymbolRange
        symbol = -1

        while True:
            symbol += 1
            prev = cur

            # Note: Implementations may prefer to store the inverse cdf to move the subtraction out of this loop.
            if cdf[N - 1] == 1 << 15:
                f = (1 << 15) - cdf[symbol]
            else:
                f = cdf[symbol]

            cur = ((self.SymbolRange >> 8) *
                   (f >> EC_PROB_SHIFT)) >> (7 - EC_PROB_SHIFT)
            cur += EC_MIN_PROB * (N - symbol - 1)

            if not (self.SymbolValue < cur):
                break

        self.SymbolRange = prev - cur
        self.SymbolValue -= cur

        # 1.
        bits = 15 - FloorLog2(self.SymbolRange)
        if bits != 0:
            # 2.
            self.SymbolRange <<= bits

            # 3.
            numBits = min(bits, max(0, self.SymbolMaxBits))

            # 4.
            newData = read_f(reader, numBits)

            # 5.
            paddedData = newData << (bits - numBits)

            # 6.
            self.SymbolValue = paddedData ^ (
                ((self.SymbolValue + 1) << bits) - 1)

            # 7.
            self.SymbolMaxBits -= bits

        if av1.on_symbol is not None:
            av1.on_symbol([self.SymbolRange])

        if update_cdf and frame_header.disable_cdf_update == 0:
            rate = (3 +
                    (cdf[N] > 15) +
                    (cdf[N] > 31) +
                    min(FloorLog2(N), 2))

            tmp = inverseCdf(0)
            for i in range(N - 1):
                # Note: Implementations may prefer to store the inverse cdf to move the subtraction out of this loop.
                if i == symbol:
                    tmp = inverseCdf(1 << 15)

                if tmp < cdf[i]:
                    cdf[i] -= ((cdf[i] - tmp) >> rate)
                else:
                    cdf[i] += ((tmp - cdf[i]) >> rate)

            cdf[N] += cdf[N] < 32

            if av1.on_cdf is not None:
                av1.on_cdf(cdf)

        return symbol


def inverseCdf(cdfs: Any) -> Any:
    # return cdfs
    if type(cdfs) == int:
        return 32768 - cdfs
    if type(cdfs[-1]) == int:
        for i in range(len(cdfs)):
            if i != len(cdfs) - 1:
                cdfs[i] = 32768 - cdfs[i]
        return cdfs
    for cdf in cdfs:
        inverseCdf(cdf)
    return cdfs


def save_cdfs(av1: AV1Decoder, ctx: int):
    """
    保存CDF数组
    规范文档 7.20 save_cdfs()

    将当前的所有CDF数组（在init_coeff_cdfs和init_non_coeff_cdfs中提到的）
    保存到参考帧上下文ctx的存储区域。

    Args:
        ctx: 上下文索引（参考帧索引，范围0到NUM_REF_FRAMES-1）
    """
    ref_frame_store = av1.ref_frame_store

    ref_frame_store.SavedCdfs[ctx] = deepcopy(ref_frame_store.cdfs)
