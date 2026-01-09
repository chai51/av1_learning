"""
Tile组OBU解析器
按照规范文档5.11节实现tile_group_obu()
"""

from constants import FILTER_INTRA_MODE, FRAME_LF_COUNT, FRAME_RESTORATION_TYPE, GM_TYPE, INTERPOLATION_FILTER, MAX_REF_MV_STACK_SIZE, MAX_TILE_COLS, MAX_TILE_ROWS, NONE, NUM_REF_FRAMES, PALETTE_COLORS, PLANE_MAX, REF_FRAME, SUB_SIZE, WIENER_COEFFS, Y_MODE
from typing import List, Optional
from copy import deepcopy
from typing import List
from bitstream.descriptors import read_f, read_le
from frame.decoding_process import decode_frame_wrapup, find_mv_stack
from reconstruction.prediction import Prediction
from utils.math_utils import Array
from utils.math_utils import Clip3, Clip1, CeilLog2, FloorLog2, Round2
from constants import COMP_MODE, COMP_REF_TYPE, MV_CLASS, OBU_HEADER_TYPE
from constants import BR_CDF_SIZE, CFL_SIGN, CLASS0_SIZE, COEFF_BASE_RANGE, COMPOUND_TYPE, H_ADST, H_DCT, H_FLIPADST, IDTX, INTERPOLATION_FILTER, INTRABC_DELAY_PIXELS, INTRABC_DELAY_SB64, MAX_VARTX_DEPTH, MOTION_MODE, MV_INTRABC_CONTEXT, MV_JOINT, NUM_BASE_LEVELS, PALETTE_COLORS, PALETTE_NUM_NEIGHBORS, PARTITION, REF_SCALE_SHIFT, SUB_SIZE, TX_MODE, TX_SET, V_ADST, V_DCT, V_FLIPADST, Max_Tx_Size_Rect, Palette_Color_Hash_Multipliers, Split_Tx_Size, Tx_Height_Log2, Tx_Size_Sqr, Tx_Width_Log2
from utils.frame_utils import inverse_recenter
from utils.tile_utils import (
    is_directional_mode, is_inside, is_scaled, get_plane_residual_size,
    count_units_in_frame, get_tx_set)
# OBU类型相关常量
from constants import (
    IDTX, DCT_DCT, V_DCT, H_DCT, ADST_ADST, ADST_DCT, DCT_ADST,
    V_ADST, H_ADST, V_FLIPADST, H_FLIPADST,
    FLIPADST_DCT, DCT_FLIPADST, ADST_ADST,
    FLIPADST_FLIPADST, ADST_FLIPADST, FLIPADST_ADST
)
from constants import (
    DELTA_Q_SMALL, DELTA_LF_SMALL, MAX_ANGLE_DELTA, MAX_LOOP_FILTER,
    PLANE_MAX, SEG_LVL_ALT_Q, SEG_LVL_GLOBALMV, SEG_LVL_REF_FRAME,
    SEG_LVL_SKIP,
    Tx_Height, Tx_Size_Sqr_Up, Tx_Width
)
# 块尺寸相关常量
from constants import SUB_SIZE
# 划分模式相关常量
from constants import PARTITION
# 帧内预测模式相关常量
from constants import Y_MODE
# Inter-intra模式相关常量
from constants import INTERINTRA_MODE
# 参考帧相关常量
from constants import REF_FRAME
# Compound类型相关常量
from constants import COMPOUND_TYPE
# 查找表
from constants import Wedge_Bits
# MI尺寸相关常量
from constants import MI_SIZE, MI_SIZE_LOG2
# Loop Restoration相关常量
from constants import FRAME_RESTORATION_TYPE
# Loop Restoration参数相关常量
from constants import (
    WIENER_COEFFS, SGRPROJ_PARAMS_BITS, SGRPROJ_PRJ_BITS, SGRPROJ_PRJ_SUBEXP_K
)
# Superres相关常量
from constants import SUPERRES_NUM
# 变换模式相关常量
from constants import TX_SIZE
# 帧级Loop Filter相关常量
from constants import FRAME_LF_COUNT
# 查找表
from constants import (
    Num_4x4_Blocks_Wide, Num_4x4_Blocks_High,
    Sgr_Params,
    Partition_Subsize, Mi_Width_Log2,
    Mi_Height_Log2, Block_Width, Block_Height
)
from obu.decoder import AV1Decoder
from utils.tile_utils import compute_tx_type, is_directional_mode, is_inside, is_scaled, get_plane_residual_size


"""
Tile Group OBU数据结构
按照规范文档5.11.1 tile_group_obu()定义的数据结构
"""


class TileGroup:
    """
    Tile Group OBU数据结构
    规范文档 5.11.1 tile_group_obu()

    保存Tile group OBU解析后的所有数据，方便后续学习和数据解析
    """

    def __init__(self, av1: AV1Decoder):
        frame_header = av1.frame_header
        MiRows = frame_header.MiRows
        MiCols = frame_header.MiCols

        self.ReadDeltas: Optional[int] = None

        # BlockDecoded[plane][row][col] - 块解码标志
        # -1 <= row <= 32, -1 <= col <= 32
        self.BlockDecoded: List[List[List[int]]] = Array(
            None, (PLANE_MAX, MAX_TILE_ROWS, MAX_TILE_COLS))

        # partition - 当前块的划分模式
        self.partition: Optional[int] = None

        # MiRow, MiCol, MiSize - 当前块的MI行、列和尺寸
        self.MiRow: int = 0
        self.MiCol: int = 0
        self.MiSize: SUB_SIZE = NONE

        # HasChroma - 是否有色度
        self.HasChroma: int = NONE

        # use_intrabc - 是否使用帧内BC
        self.use_intrabc: Optional[int] = None

        # Lossless - 是否无损
        self.Lossless: Optional[int] = None

        self.segment_id: int = NONE

        self.skip_mode: int = NONE
        self.skip: int = NONE

        self.TxSize: TX_SIZE = NONE

        self.InterTxSizes: List[List[TX_SIZE]] = NONE

        self.is_inter: int = NONE

        self.interp_filter: List[INTERPOLATION_FILTER] = [NONE] * 2

        self.use_filter_intra: Optional[int] = None

        self.filter_intra_mode: FILTER_INTRA_MODE = NONE

        self.RefFrame: List[REF_FRAME] = [NONE, NONE]

        self.motion_mode: Optional[int] = None

        self.interintra: Optional[int] = None

        self.interintra_mode: Optional[int] = None

        self.wedge_interintra: Optional[int] = None

        self.comp_group_idx: int = NONE
        self.compound_idx: int = NONE
        self.compound_type: Optional[int] = None

        self.wedge_index: int = NONE
        self.wedge_sign: int = NONE
        self.mask_type: Optional[int] = None

        self.MvCtx: Optional[int] = None

        self.MaxLumaW: int = NONE
        self.MaxLumaH: int = NONE

        # Loop Filter变换尺寸
        self.LoopfilterTxSizes: List[List[List[int]]] = NONE

        self.TxTypes: List[List[int]] = NONE

        self.Quant: List[int] = [NONE] * (MAX_TILE_ROWS * MAX_TILE_COLS)

        self.AboveLevelContext: List[List[int]] = NONE
        self.LeftLevelContext: List[List[int]] = NONE
        self.AboveDcContext: List[List[int]] = NONE
        self.LeftDcContext: List[List[int]] = NONE

        self.AngleDeltaY: int = NONE
        self.AngleDeltaUV: int = NONE

        self.CflAlphaU: int = NONE
        self.CflAlphaV: int = NONE

        self.PaletteSizeY: int = NONE
        self.PaletteSizeUV: int = NONE

        self.palette_colors_y: List[int] = [
            NONE] * PALETTE_COLORS
        self.palette_colors_u: List[int] = [
            NONE] * PALETTE_COLORS
        self.palette_colors_v: List[int] = [
            NONE] * PALETTE_COLORS

        self.ColorContextHash: int = NONE

        self.cdef_idx: List[List[int]] = NONE

        self.LrType: List[List[List[FRAME_RESTORATION_TYPE]]] = Array(
            None, (PLANE_MAX, MAX_TILE_ROWS, MAX_TILE_COLS))
        self.LrSgrSet: List[List[List[int]]] = Array(
            None, (PLANE_MAX, MAX_TILE_ROWS, MAX_TILE_COLS))

        self.LrSgrXqd: List[List[List[List[int]]]] = Array(
            None, (PLANE_MAX, MAX_TILE_ROWS, MAX_TILE_COLS, 2))

        self.LrWiener: List[List[List[List[List[int]]]]] = Array(
            None, (PLANE_MAX, MAX_TILE_ROWS, MAX_TILE_COLS, 2, 3))

        self.DeltaLF: List[int] = NONE

        self.RefSgrXqd: List[List[int]] = Array(None, (PLANE_MAX, 2))

        self.MiRowStart = 0
        self.MiRowEnd = 0
        self.MiColStart = 0
        self.MiColEnd = 0

        self.CurrentQIndex = 0

        self.RefLrWiener: List[List[List[int]]] = Array(
            None, (PLANE_MAX, 2, WIENER_COEFFS))

        self.AvailU: int = NONE
        self.AvailL: int = NONE
        self.AvailUChroma: int = NONE
        self.AvailLChroma: int = NONE

        self.YMode: Y_MODE = NONE
        self.UVMode: Y_MODE = NONE
        self.YModes: List[List[Y_MODE]] = NONE
        self.UVModes: List[List[Y_MODE]] = NONE

        self.RefFrames: List[List[List[REF_FRAME]]] = NONE

        self.CompGroupIdxs: List[List[int]] = NONE
        self.CompoundIdxs: List[List[int]] = NONE

        self.InterpFilters: List[List[List[INTERPOLATION_FILTER]]] = NONE

        self.Mv: List[List[int]] = Array(None, (2, 2))
        self.Mvs: List[List[List[List[int]]]] = NONE

        self.IsInters: List[List[int]] = NONE
        self.SkipModes: List[List[int]] = NONE
        self.Skips: List[List[int]] = NONE
        self.MiSizes: List[List[SUB_SIZE]] = NONE
        self.SegmentIds: List[List[int]] = NONE
        self.PaletteSizes: List[List[List[int]]] = NONE

        self.PaletteColors: List[List[List[List[int]]]] = NONE

        self.DeltaLFs: List[List[List[int]]] = NONE

        self.LeftRefFrame: List[REF_FRAME] = [NONE, NONE]
        self.AboveRefFrame: List[REF_FRAME] = [NONE, NONE]

        self.LeftIntra: int = NONE
        self.AboveIntra: int = NONE
        self.LeftSingle: int = NONE
        self.AboveSingle: int = NONE

        self.AboveSegPredContext: List[int] = NONE
        self.LeftSegPredContext: List[int] = NONE

        self.RefMvIdx: int = NONE

        self.NumMvFound: int = NONE

        self.PredMv: List[List[int]] = [NONE] * 2
        self.RefStackMv: List[List[List[int]]] = NONE
        self.GlobalMvs: List[List[int]] = [NONE] * 2

        self.NumSamples: int = NONE

        self.Dequant: List[List[int]] = NONE

        self.PlaneTxType: int = NONE

        self.ColorMapY: List[List[int]] = Array(
            None, (MAX_TILE_ROWS, MAX_TILE_COLS))
        self.ColorMapUV: List[List[int]] = Array(
            None, (MAX_TILE_ROWS, MAX_TILE_COLS))
        self.ColorOrder: List[int] = [NONE] * PALETTE_COLORS

        self.ZeroMvContext: Optional[int] = NONE

        self.DrlCtxStack: List[int] = [NONE] * MAX_REF_MV_STACK_SIZE

        self.NewMvContext: int = NONE
        self.RefMvContext: int = NONE

        self.CandList: List[List[int]] = Array(
            None, (MAX_REF_MV_STACK_SIZE, 4))

        self.InterRound0: int = NONE
        self.InterRound1: int = NONE
        self.InterPostRound: int = NONE

        self.IsInterIntra: Optional[int] = None

        self.Residual: List[List[int]] = Array(
            None, (MAX_TILE_ROWS, MAX_TILE_COLS))

        self.MfRefFrames: List[List[REF_FRAME]] = Array(None, (MiRows, MiCols))

        self.MfMvs: List[List[List[int]]] = Array(None, (MiRows, MiCols, 2))

        self.NumSamplesScanned: int = NONE

        self.TxType: int = NONE

        self.prediction = Prediction(av1)


def read_L(av1: AV1Decoder, n: int) -> int:
    """
    该语法元素等于read_literal(n)的返回值
    规范文档 4.10.8

    Args:
        n: 要读取的位数

    Returns:
        无符号整数
    """
    decoder = av1.decoder
    return decoder.read_literal(av1, n)


def read_S(av1: AV1Decoder, syntax_element_name: str, *args, **kwargs) -> int:
    """
    该符号基于上下文敏感的CDF进行解码
    规范文档 4.10.9

    Args:
        syntax_element_name: 语法元素名称

    Returns:
        解码的符号值
    """
    decoder = av1.decoder

    from entropy.cdf_selection import cdf_selection_process
    cdf = cdf_selection_process(av1, syntax_element_name, *args, **kwargs)
    update_cdf = True
    if syntax_element_name in ["split_or_horz", "split_or_vert"]:
        update_cdf = False
    return decoder.read_symbol(av1, cdf, update_cdf)


def read_NS(av1: AV1Decoder, n: int) -> int:
    """
    非对称数值编码
    规范文档 4.10.10

    Args:
        n: 要读取的位数

    Returns:
        解码的值
    """
    w = FloorLog2(n) + 1
    m = (1 << w) - n
    v = read_L(av1, w - 1)

    if v < m:
        return v

    extra_bit = read_L(av1, 1)
    return (v << 1) - m + extra_bit


class TileGroupParser:
    """
    Tile组解析器
    实现规范文档中描述的tile_group_obu()函数
    """

    def __init__(self, av1: AV1Decoder):
        self.tile_group = TileGroup(av1)

    def tile_group_obu(self, av1: AV1Decoder, sz: int) -> TileGroup:
        """
        规范文档 5.11.1 General tile group OBU syntax
        """
        reader = av1.reader
        header = av1.obu.header
        frame_header = av1.frame_header
        tile_group = self.tile_group
        decoder = av1.decoder

        NumTiles = frame_header.TileCols * frame_header.TileRows
        startBitPos = reader.get_position()
        tile_start_and_end_present_flag = 0
        if NumTiles > 1:
            tile_start_and_end_present_flag = read_f(reader, 1)

        # If obu_type is equal to OBU_FRAME, it is a requirement of bitstream conformance that the value of tile_start_and_end_present_flag is equal to 0.
        if header.obu_type == OBU_HEADER_TYPE.OBU_FRAME:
            assert tile_start_and_end_present_flag == 0

        if NumTiles == 1 or not tile_start_and_end_present_flag:
            tg_start = 0
            tg_end = NumTiles - 1
        else:
            tileBits = frame_header.TileColsLog2 + frame_header.TileRowsLog2
            tg_start = read_f(reader, tileBits)
            tg_end = read_f(reader, tileBits)

        # It is a requirement of bitstream conformance that the value of tg_start is equal to the value of TileNum at the point that tile_group_obu is invoked.
        assert tg_start == frame_header.TileNum
        # It is a requirement of bitstream conformance that the value of tg_end is greater than or equal to tg_start.
        assert tg_end >= tg_start

        reader.byte_alignment()
        endBitPos = reader.get_position()
        headerBytes = (endBitPos - startBitPos) // 8
        sz -= headerBytes

        frame_header.TileNum = tg_start
        while frame_header.TileNum <= tg_end:
            tileRow = frame_header.TileNum // frame_header.TileCols
            tileCol = frame_header.TileNum % frame_header.TileCols

            lastTile = (frame_header.TileNum == tg_end)
            if lastTile:
                tileSize = sz
            else:
                tile_size_minus_1 = read_le(reader, frame_header.TileSizeBytes)
                tileSize = tile_size_minus_1 + 1
                sz -= tileSize + frame_header.TileSizeBytes

            tile_group.MiRowStart = frame_header.MiRowStarts[tileRow]
            tile_group.MiRowEnd = frame_header.MiRowStarts[tileRow + 1]
            tile_group.MiColStart = frame_header.MiColStarts[tileCol]
            tile_group.MiColEnd = frame_header.MiColStarts[tileCol + 1]

            tile_group.CurrentQIndex = frame_header.base_q_idx
            decoder.init_symbol(av1, tileSize)
            self.__decode_tile(av1)
            decoder.exit_symbol(av1)

            frame_header.TileNum += 1

        if tg_end == NumTiles - 1:
            if not frame_header.disable_frame_end_update_cdf:
                from frame.decoding_process import frame_end_update_cdf
                frame_end_update_cdf(av1)

            decode_frame_wrapup(av1)
            av1.SeenFrameHeader = 0

        return tile_group

    def __decode_tile(self, av1: AV1Decoder):
        """
        规范文档 5.11.2 Decode tile syntax
        """
        seq_header = av1.seq_header
        frame_header = av1.frame_header
        tile_group = self.tile_group
        MiRows = frame_header.MiRows
        MiCols = frame_header.MiCols
        use_128x128_superblock = seq_header.use_128x128_superblock
        NumPlanes = seq_header.color_config.NumPlanes

        clear_above_context(av1)
        tile_group.DeltaLF = [0] * FRAME_LF_COUNT
        for plane in range(NumPlanes):
            for pass_val in range(2):
                tile_group.RefSgrXqd[plane][pass_val] = Sgrproj_Xqd_Mid[pass_val]
                for i in range(WIENER_COEFFS):
                    tile_group.RefLrWiener[plane][pass_val][i] = Wiener_Taps_Mid[i]

        sbSize = SUB_SIZE.BLOCK_128X128 if use_128x128_superblock else SUB_SIZE.BLOCK_64X64
        sbSize4 = Num_4x4_Blocks_Wide[sbSize]

        tile_group.cdef_idx = Array(
            tile_group.cdef_idx, (MiRows + 32, MiCols + 32))
        for r in range(tile_group.MiRowStart, tile_group.MiRowEnd, sbSize4):
            clear_left_context(av1)
            for c in range(tile_group.MiColStart, tile_group.MiColEnd, sbSize4):
                tile_group.ReadDeltas = frame_header.delta_q_present
                self.__clear_cdef(av1, r, c)
                self.__clear_block_decoded_flags(av1, r, c, sbSize4)
                self.__read_lr(av1, r, c, sbSize)
                self.__decode_partition(av1, r, c, sbSize)

    def __clear_block_decoded_flags(self, av1: AV1Decoder, r: int, c: int, sbSize4: int):
        """
        清除块解码标志
        规范文档 5.11.3 Clear block decoded flags function
        """
        seq_header = av1.seq_header
        tile_group = self.tile_group
        subsampling_x = seq_header.color_config.subsampling_x
        subsampling_y = seq_header.color_config.subsampling_y
        NumPlanes = seq_header.color_config.NumPlanes

        for plane in range(NumPlanes):
            subX = subsampling_x if plane > 0 else 0
            subY = subsampling_y if plane > 0 else 0
            sbWidth4 = (tile_group.MiColEnd - c) >> subX
            sbHeight4 = (tile_group.MiRowEnd - r) >> subY

            for y in range(-1, (sbSize4 >> subY) + 1):
                for x in range(-1, (sbSize4 >> subX) + 1):
                    if y < 0 and x < sbWidth4:
                        tile_group.BlockDecoded[plane][y][x] = 1
                    elif x < 0 and y < sbHeight4:
                        tile_group.BlockDecoded[plane][y][x] = 1
                    else:
                        tile_group.BlockDecoded[plane][y][x] = 0

            tile_group.BlockDecoded[plane][sbSize4 >> subY][-1] = 0

    def __decode_partition(self, av1: AV1Decoder, r: int, c: int, bSize: SUB_SIZE):
        """
        解码划分
        规范文档 5.11.4 Decode partition syntax

        Args:
            r: Mi行位置
            c: Mi列位置
            bSize: 块尺寸
        """
        frame_header = av1.frame_header
        tile_group = self.tile_group
        MiRows = frame_header.MiRows
        MiCols = frame_header.MiCols

        if r >= MiRows or c >= MiCols:
            return 0

        tile_group.AvailU = is_inside(av1, r - 1, c)
        tile_group.AvailL = is_inside(av1, r, c - 1)
        num4x4 = Num_4x4_Blocks_Wide[bSize]
        halfBlock4x4 = num4x4 >> 1
        quarterBlock4x4 = halfBlock4x4 >> 1
        hasRows = (r + halfBlock4x4) < MiRows
        hasCols = (c + halfBlock4x4) < MiCols

        if bSize < SUB_SIZE.BLOCK_8X8:
            tile_group.partition = PARTITION.PARTITION_NONE
        elif hasRows and hasCols:
            tile_group.partition = read_S(
                av1, "partition", r=r, c=c, bSize=bSize)
        elif hasCols:
            split_or_horz = read_S(av1, "split_or_horz", r=r, c=c, bSize=bSize)
            tile_group.partition = PARTITION.PARTITION_SPLIT if split_or_horz else PARTITION.PARTITION_HORZ
        elif hasRows:
            split_or_vert = read_S(av1, "split_or_vert", r=r, c=c, bSize=bSize)
            tile_group.partition = PARTITION.PARTITION_SPLIT if split_or_vert else PARTITION.PARTITION_VERT
        else:
            tile_group.partition = PARTITION.PARTITION_SPLIT

        subSize: SUB_SIZE = Partition_Subsize[tile_group.partition][bSize]
        # It is a requirement of bitstream conformance that get_plane_residual_size( subSize, 1 ) is not equal to BLOCK_INVALID every time subSize is computed.
        assert get_plane_residual_size(
            av1, subSize, 1) != SUB_SIZE.BLOCK_INVALID

        splitSize = Partition_Subsize[PARTITION.PARTITION_SPLIT][bSize]

        if tile_group.partition == PARTITION.PARTITION_NONE:
            self.__decode_block(av1, r, c, subSize)
        elif tile_group.partition == PARTITION.PARTITION_HORZ:
            self.__decode_block(av1, r, c, subSize)
            if hasRows:
                self.__decode_block(av1, r + halfBlock4x4, c, subSize)
        elif tile_group.partition == PARTITION.PARTITION_VERT:
            self.__decode_block(av1, r, c, subSize)
            if hasCols:
                self.__decode_block(av1, r, c + halfBlock4x4, subSize)
        elif tile_group.partition == PARTITION.PARTITION_SPLIT:
            self.__decode_partition(av1, r, c, subSize)
            self.__decode_partition(av1, r, c + halfBlock4x4, subSize)
            self.__decode_partition(av1, r + halfBlock4x4, c, subSize)
            self.__decode_partition(
                av1, r + halfBlock4x4, c + halfBlock4x4, subSize)
        elif tile_group.partition == PARTITION.PARTITION_HORZ_A:
            self.__decode_block(av1, r, c, splitSize)
            self.__decode_block(av1, r, c + halfBlock4x4, splitSize)
            self.__decode_block(av1, r + halfBlock4x4, c, subSize)
        elif tile_group.partition == PARTITION.PARTITION_HORZ_B:
            self.__decode_block(av1, r, c, subSize)
            self.__decode_block(av1, r + halfBlock4x4, c, splitSize)
            self.__decode_block(av1, r + halfBlock4x4,
                                c + halfBlock4x4, splitSize)
        elif tile_group.partition == PARTITION.PARTITION_VERT_A:
            self.__decode_block(av1, r, c, splitSize)
            self.__decode_block(av1, r + halfBlock4x4, c, splitSize)
            self.__decode_block(av1, r, c + halfBlock4x4, subSize)
        elif tile_group.partition == PARTITION.PARTITION_VERT_B:
            self.__decode_block(av1, r, c, subSize)
            self.__decode_block(av1, r, c + halfBlock4x4, splitSize)
            self.__decode_block(av1, r + halfBlock4x4,
                                c + halfBlock4x4, splitSize)
        elif tile_group.partition == PARTITION.PARTITION_HORZ_4:
            self.__decode_block(av1, r + quarterBlock4x4 * 0, c, subSize)
            self.__decode_block(av1, r + quarterBlock4x4 * 1, c, subSize)
            self.__decode_block(av1, r + quarterBlock4x4 * 2, c, subSize)
            if r + quarterBlock4x4 * 3 < MiRows:
                self.__decode_block(av1, r + quarterBlock4x4 * 3, c, subSize)
        elif tile_group.partition == PARTITION.PARTITION_VERT_4:
            self.__decode_block(av1, r, c + quarterBlock4x4 * 0, subSize)
            self.__decode_block(av1, r, c + quarterBlock4x4 * 1, subSize)
            self.__decode_block(av1, r, c + quarterBlock4x4 * 2, subSize)
            if c + quarterBlock4x4 * 3 < MiCols:
                self.__decode_block(av1, r, c + quarterBlock4x4 * 3, subSize)

    def __decode_block(self, av1: AV1Decoder, r: int, c: int, subSize: SUB_SIZE):
        """
        规范文档 5.11.5 Decode block syntax

        Args:
            r: Mi行位置
            c: Mi列位置
            subSize: 子块尺寸
        """
        seq_header = av1.seq_header
        frame_header = av1.frame_header
        tile_group = self.tile_group
        MiRows = frame_header.MiRows
        MiCols = frame_header.MiCols
        subsampling_x = seq_header.color_config.subsampling_x
        subsampling_y = seq_header.color_config.subsampling_y
        NumPlanes = seq_header.color_config.NumPlanes

        def reset_block_context(bw4: int, bh4: int):
            """
            重置块上下文

            Args:
                bw4: 块宽度（4x4块单位）
                bh4: 块高度（4x4块单位）
            """
            MiCol = tile_group.MiCol
            MiRow = tile_group.MiRow
            HasChroma = tile_group.HasChroma
            for plane in range(1 + 2 * HasChroma):
                subX = subsampling_x if plane > 0 else 0
                subY = subsampling_y if plane > 0 else 0
                for i in range(MiCol >> subX, (MiCol + bw4) >> subX):
                    tile_group.AboveLevelContext[plane][i] = 0
                    tile_group.AboveDcContext[plane][i] = 0
                for i in range(MiRow >> subY, (MiRow + bh4) >> subY):
                    tile_group.LeftLevelContext[plane][i] = 0
                    tile_group.LeftDcContext[plane][i] = 0

        if tile_group.InterTxSizes is None or len(tile_group.InterTxSizes) < (r + 32) or len(tile_group.InterTxSizes[0]) < (c + 32):
            tile_group.InterTxSizes = Array(
                tile_group.InterTxSizes, (r + 32, c + 32))
            tile_group.LoopfilterTxSizes = Array(
                tile_group.LoopfilterTxSizes, (PLANE_MAX, r + 32, c + 32))
            tile_group.TxTypes = Array(tile_group.TxTypes, (r + 32, c + 32))
            tile_group.YModes = Array(tile_group.YModes, (r + 32, c + 32))
            tile_group.UVModes = Array(tile_group.UVModes, (r + 32, c + 32))
            tile_group.RefFrames = Array(
                tile_group.RefFrames, (r + 32, c + 32, 2))
            tile_group.CompGroupIdxs = Array(
                tile_group.CompGroupIdxs, (r + 32, c + 32))
            tile_group.CompoundIdxs = Array(
                tile_group.CompoundIdxs, (r + 32, c + 32))
            tile_group.InterpFilters = Array(
                tile_group.InterpFilters, (r + 32, c + 32, 2))
            tile_group.Mvs = Array(tile_group.Mvs, (r + 32, c + 32, 2))

            tile_group.IsInters = Array(
                tile_group.IsInters, (r + 32, c + 32))
            tile_group.SkipModes = Array(
                tile_group.SkipModes, (r + 32, c + 32))
            tile_group.Skips = Array(tile_group.Skips, (r + 32, c + 32))
            tile_group.MiSizes = Array(
                tile_group.MiSizes, (r + 32, c + 32))
            tile_group.SegmentIds = Array(
                tile_group.SegmentIds, (r + 32, c + 32))
            tile_group.PaletteSizes = Array(
                tile_group.PaletteSizes, (2, r + 32, c + 32))
            tile_group.DeltaLFs = Array(
                tile_group.DeltaLFs, (r + 32, c + 32, FRAME_LF_COUNT))

        tile_group.MiRow = r
        tile_group.MiCol = c
        tile_group.MiSize = subSize

        bw4 = Num_4x4_Blocks_Wide[subSize]
        bh4 = Num_4x4_Blocks_High[subSize]
        if bh4 == 1 and subsampling_y and (tile_group.MiRow & 1) == 0:
            tile_group.HasChroma = 0
        elif bw4 == 1 and subsampling_x and (tile_group.MiCol & 1) == 0:
            tile_group.HasChroma = 0
        else:
            tile_group.HasChroma = NumPlanes > 1

        tile_group.AvailU = is_inside(av1, r - 1, c)
        tile_group.AvailL = is_inside(av1, r, c - 1)
        tile_group.AvailUChroma = tile_group.AvailU
        tile_group.AvailLChroma = tile_group.AvailL
        if tile_group.HasChroma:
            if subsampling_y and bh4 == 1:
                tile_group.AvailUChroma = is_inside(av1, r - 2, c)
            if subsampling_x and bw4 == 1:
                tile_group.AvailLChroma = is_inside(av1, r, c - 2)
        else:
            tile_group.AvailUChroma = 0
            tile_group.AvailLChroma = 0

        self.__mode_info(av1)
        self.__palette_tokens(av1)
        self.__read_block_tx_size(av1)
        if tile_group.skip:
            reset_block_context(bw4, bh4)

        isCompound = tile_group.RefFrame[1] > REF_FRAME.INTRA_FRAME

        for y in range(bh4):
            for x in range(bw4):
                tile_group.YModes[r + y][c + x] = tile_group.YMode
                if tile_group.RefFrame[0] == REF_FRAME.INTRA_FRAME and tile_group.HasChroma:
                    tile_group.UVModes[r + y][c + x] = tile_group.UVMode
                for refList in range(2):
                    tile_group.RefFrames[r + y][c +
                                                x][refList] = tile_group.RefFrame[refList]
                if tile_group.is_inter:
                    if not tile_group.use_intrabc:
                        tile_group.CompGroupIdxs[r + y][c +
                                                        x] = tile_group.comp_group_idx
                        tile_group.CompoundIdxs[r + y][c +
                                                       x] = tile_group.compound_idx
                    for dir in range(2):
                        tile_group.InterpFilters[r + y][c +
                                                        x][dir] = tile_group.interp_filter[dir]
                    for refList in range(1 + isCompound):
                        tile_group.Mvs[r + y][c +
                                              x][refList] = deepcopy(tile_group.Mv[refList])

        self.__compute_prediction(av1)
        self.__residual(av1)

        tile_group.PaletteColors = Array(tile_group.PaletteColors, (
            2, MiRows + 32, MiCols + 32, tile_group.PaletteSizeY, tile_group.PaletteSizeUV))
        for y in range(bh4):
            for x in range(bw4):
                tile_group.IsInters[r + y][c + x] = tile_group.is_inter
                tile_group.SkipModes[r + y][c + x] = tile_group.skip_mode
                tile_group.Skips[r + y][c + x] = tile_group.skip
                TxSizes = tile_group.TxSize
                tile_group.MiSizes[r + y][c + x] = tile_group.MiSize
                tile_group.SegmentIds[r + y][c + x] = tile_group.segment_id
                tile_group.PaletteSizes[0][r +
                                           y][c + x] = tile_group.PaletteSizeY
                tile_group.PaletteSizes[1][r + y][c +
                                                  x] = tile_group.PaletteSizeUV
                for i in range(tile_group.PaletteSizeY):
                    tile_group.PaletteColors[0][r + y][c +
                                                       x][i] = tile_group.palette_colors_y[i]
                for i in range(tile_group.PaletteSizeUV):
                    tile_group.PaletteColors[1][r + y][c +
                                                       x][i] = tile_group.palette_colors_u[i]
                for i in range(FRAME_LF_COUNT):
                    tile_group.DeltaLFs[r + y][c +
                                               x][i] = tile_group.DeltaLF[i]

    def __mode_info(self, av1: AV1Decoder):
        """
        模式信息解析函数
        规范文档 5.11.6 Mode info syntax
        """
        frame_header = av1.frame_header
        tile_group = self.tile_group

        if frame_header.FrameIsIntra:
            self.__intra_frame_mode_info(av1)
        else:
            self.__inter_frame_mode_info(av1)

        # It is a requirement of bitstream conformance that the postprocessed value of segment_id (i.e. the value returned by neg_deinterleave) is in the range 0 to LastActiveSegId (inclusive of endpoints).
        assert tile_group.segment_id >= 0
        assert tile_group.segment_id <= frame_header.LastActiveSegId

    def __intra_frame_mode_info(self, av1: AV1Decoder):
        """
        解析帧内模式信息
        规范文档 5.11.7 Intra frame mode info syntax
        """
        frame_header = av1.frame_header
        tile_group = self.tile_group
        HasChroma = tile_group.HasChroma

        tile_group.skip = 0
        if frame_header.SegIdPreSkip:
            self.__intra_segment_id(av1)

        tile_group.skip_mode = 0
        self.__read_skip(av1)

        if not frame_header.SegIdPreSkip:
            self.__intra_segment_id(av1)

        self.__read_cdef(av1)
        self.__read_delta_qindex(av1)
        self.__read_delta_lf(av1)

        tile_group.ReadDeltas = 0
        tile_group.RefFrame[0] = REF_FRAME.INTRA_FRAME
        tile_group.RefFrame[1] = REF_FRAME.NONE

        if frame_header.allow_intrabc:
            tile_group.use_intrabc = read_S(av1, 'use_intrabc')
        else:
            tile_group.use_intrabc = 0
        if tile_group.use_intrabc:
            tile_group.is_inter = 1
            tile_group.YMode = Y_MODE.DC_PRED
            tile_group.UVMode = Y_MODE.DC_PRED
            tile_group.motion_mode = MOTION_MODE.SIMPLE
            tile_group.compound_type = COMPOUND_TYPE.COMPOUND_AVERAGE
            tile_group.PaletteSizeY = 0
            tile_group.PaletteSizeUV = 0
            tile_group.interp_filter[0] = INTERPOLATION_FILTER.BILINEAR
            tile_group.interp_filter[1] = INTERPOLATION_FILTER.BILINEAR
            tile_group.RefStackMv = find_mv_stack(av1, 0)
            self.__assign_mv(av1, 0)
        else:
            tile_group.is_inter = 0
            intra_frame_y_mode = Y_MODE(read_S(av1, 'intra_frame_y_mode'))
            tile_group.YMode = intra_frame_y_mode

            self.__intra_angle_info_y(av1)
            if HasChroma:
                uv_mode = Y_MODE(read_S(av1, 'uv_mode'))
                tile_group.UVMode = uv_mode
                if tile_group.UVMode == Y_MODE.UV_CFL_PRED:
                    self.__read_cfl_alphas(av1)
                self.__intra_angle_info_uv(av1)

            tile_group.PaletteSizeY = 0
            tile_group.PaletteSizeUV = 0
            if (tile_group.MiSize >= SUB_SIZE.BLOCK_8X8 and
                Block_Width[tile_group.MiSize] <= 64 and
                Block_Height[tile_group.MiSize] <= 64 and
                    frame_header.allow_screen_content_tools):
                self.__palette_mode_info(av1)

            self.__filter_intra_mode_info(av1)

    def __intra_segment_id(self, av1: AV1Decoder):
        """
        解析帧内Segment ID
        规范文档 5.11.8 Intra segment ID syntax
        """
        frame_header = av1.frame_header
        tile_group = self.tile_group

        if frame_header.segmentation_enabled:
            self.__read_segment_id(av1)
        else:
            tile_group.segment_id = 0

        tile_group.Lossless = frame_header.LosslessArray[tile_group.segment_id]

    def __read_segment_id(self, av1: AV1Decoder):
        """
        读取Segment ID
        规范文档 5.11.9 Read segment ID syntax
        """
        frame_header = av1.frame_header
        tile_group = self.tile_group
        MiRow = tile_group.MiRow
        MiCol = tile_group.MiCol
        AvailU = tile_group.AvailU
        AvailL = tile_group.AvailL

        def neg_deinterleave(diff: int, ref: int, max_val: int) -> int:
            """
            规范文档 5.11.9

            Args:
                diff: 差值
                ref: 参考值
                max_val: 最大值

            Returns:
                去交织后的值
            """
            if not ref:
                return diff

            if ref >= (max_val - 1):
                return max_val - diff - 1

            if 2 * ref < max_val:
                if diff <= 2 * ref:
                    if diff & 1:
                        return ref + ((diff + 1) >> 1)
                    else:
                        return ref - (diff >> 1)
                return diff
            else:
                if diff <= 2 * (max_val - ref - 1):
                    if diff & 1:
                        return ref + ((diff + 1) >> 1)
                    else:
                        return ref - (diff >> 1)
                return max_val - (diff + 1)

        if AvailU and AvailL:
            prevUL = tile_group.SegmentIds[MiRow - 1][MiCol - 1]
        else:
            prevUL = -1

        if AvailU:
            prevU = tile_group.SegmentIds[MiRow - 1][MiCol]
        else:
            prevU = -1

        if AvailL:
            prevL = tile_group.SegmentIds[MiRow][MiCol - 1]
        else:
            prevL = -1

        if prevU == -1:
            pred = 0 if prevL == -1 else prevL
        elif prevL == -1:
            pred = prevU
        else:
            pred = prevU if prevUL == prevU else prevL

        if tile_group.skip:
            tile_group.segment_id = pred
        else:
            tile_group.segment_id = read_S(
                av1, "segment_id", prevU=prevU, prevL=prevL, prevUL=prevUL)
            tile_group.segment_id = neg_deinterleave(
                tile_group.segment_id, pred, frame_header.LastActiveSegId + 1)

    def __read_skip_mode(self, av1: AV1Decoder):
        """
        读取skip_mode
        规范文档 5.11.10 Skip mode syntax
        """
        frame_header = av1.frame_header
        tile_group = self.tile_group

        if (self.__seg_feature_active(av1, SEG_LVL_SKIP) or
            self.__seg_feature_active(av1, SEG_LVL_REF_FRAME) or
            self.__seg_feature_active(av1, SEG_LVL_GLOBALMV) or
            not frame_header.skip_mode_present or
            Block_Width[tile_group.MiSize] < 8 or
                Block_Height[tile_group.MiSize] < 8):
            tile_group.skip_mode = 0
        else:
            tile_group.skip_mode = read_S(av1, 'skip_mode')

    def __read_skip(self, av1: AV1Decoder):
        """
        读取skip标志
        规范文档 5.11.11 Skip syntax
        """
        frame_header = av1.frame_header
        tile_group = self.tile_group

        if frame_header.SegIdPreSkip and self.__seg_feature_active(av1, SEG_LVL_SKIP):
            tile_group.skip = 1
        else:
            tile_group.skip = read_S(av1, 'skip')

    def __read_delta_qindex(self, av1: AV1Decoder):
        """
        读取量化索引增量
        规范文档 5.11.12 Quantizer index delta syntax
        """
        seq_header = av1.seq_header
        frame_header = av1.frame_header
        tile_group = self.tile_group
        use_128x128_superblock = seq_header.use_128x128_superblock

        sbSize = SUB_SIZE.BLOCK_128X128 if use_128x128_superblock else SUB_SIZE.BLOCK_64X64
        if tile_group.MiSize == sbSize and tile_group.skip:
            return

        if tile_group.ReadDeltas:
            delta_q_abs = read_S(av1, 'delta_q_abs')
            if delta_q_abs == DELTA_Q_SMALL:
                delta_q_rem_bits = read_L(av1, 3)
                delta_q_rem_bits += 1
                delta_q_abs_bits = read_L(av1, delta_q_rem_bits)
                delta_q_abs = delta_q_abs_bits + (1 << delta_q_rem_bits) + 1

            if delta_q_abs:
                delta_q_sign_bit = read_L(av1, 1)
                reducedDeltaQIndex = -delta_q_abs if delta_q_sign_bit else delta_q_abs
                tile_group.CurrentQIndex = Clip3(
                    1, 255, tile_group.CurrentQIndex + (reducedDeltaQIndex << frame_header.delta_q_res))

    def __read_delta_lf(self, av1: AV1Decoder):
        """
        读取环路滤波器增量
        规范文档 5.11.13 Loop filter delta syntax
        """
        seq_header = av1.seq_header
        frame_header = av1.frame_header
        tile_group = self.tile_group
        use_128x128_superblock = seq_header.use_128x128_superblock
        NumPlanes = seq_header.color_config.NumPlanes

        sbSize = SUB_SIZE.BLOCK_128X128 if use_128x128_superblock else SUB_SIZE.BLOCK_64X64
        if tile_group.MiSize == sbSize and tile_group.skip:
            return

        if tile_group.ReadDeltas and frame_header.delta_lf_present:
            frameLfCount = 1
            if frame_header.delta_lf_multi:
                frameLfCount = FRAME_LF_COUNT if NumPlanes > 1 else (
                    FRAME_LF_COUNT - 2)

            for i in range(frameLfCount):
                delta_lf_abs = read_S(av1, 'delta_lf_abs', i=i)
                if delta_lf_abs == DELTA_LF_SMALL:
                    delta_lf_rem_bits = read_L(av1, 3)
                    n = delta_lf_rem_bits + 1
                    delta_lf_abs_bits = read_L(av1, n)
                    deltaLfAbs = delta_lf_abs_bits + (1 << n) + 1
                else:
                    deltaLfAbs = delta_lf_abs

                if deltaLfAbs:
                    delta_lf_sign_bit = read_L(av1, 1)
                    reducedDeltaLfLevel = -deltaLfAbs if delta_lf_sign_bit else deltaLfAbs
                    tile_group.DeltaLF[i] = Clip3(-MAX_LOOP_FILTER, MAX_LOOP_FILTER, tile_group.DeltaLF[i] + (
                        reducedDeltaLfLevel << frame_header.delta_lf_res))

    def __seg_feature_active(self, av1: AV1Decoder, feature: int) -> int:
        """
        规范文档 5.11.14 Segmentation feature active syntax

        Args:
            feature: 特征类型（SEG_LVL_*）
        Returns:
            如果特征激活返回1，否则返回0
        """
        tile_group = self.tile_group
        from utils.tile_utils import seg_feature_active_idx
        return seg_feature_active_idx(av1, tile_group.segment_id, feature)

    def __read_tx_size(self, av1: AV1Decoder, allowSelect):
        """
        读取变换尺寸
        规范文档 5.11.15 TX size syntax
        """
        frame_header = av1.frame_header
        tile_group = self.tile_group
        MiSize = tile_group.MiSize

        if tile_group.Lossless:
            tile_group.TxSize = TX_SIZE.TX_4X4
            return

        maxRectTxSize: TX_SIZE = Max_Tx_Size_Rect[MiSize]
        maxTxDepth = Max_Tx_Depth[MiSize]
        tile_group.TxSize = maxRectTxSize
        if MiSize > SUB_SIZE.BLOCK_4X4 and allowSelect and frame_header.TxMode == TX_MODE.TX_MODE_SELECT:
            # tx_depth can only encode values in the range 0 to 2
            # assert maxTxDepth <= 2

            tx_depth = read_S(
                av1, "tx_depth", maxRectTxSize=maxRectTxSize, maxTxDepth=maxTxDepth)
            for i in range(tx_depth):
                tile_group.TxSize = Split_Tx_Size[tile_group.TxSize]

    def __read_block_tx_size(self, av1: AV1Decoder):
        """
        读取块的变换尺寸
        规范文档 5.11.16 Block TX size syntax
        """
        frame_header = av1.frame_header
        tile_group = self.tile_group
        MiRow = tile_group.MiRow
        MiCol = tile_group.MiCol
        MiSize = tile_group.MiSize

        bw4 = Num_4x4_Blocks_Wide[MiSize]
        bh4 = Num_4x4_Blocks_High[MiSize]
        if (frame_header.TxMode == TX_MODE.TX_MODE_SELECT and
            MiSize > SUB_SIZE.BLOCK_4X4 and
            tile_group.is_inter and
            not tile_group.skip and
                not tile_group.Lossless):
            maxTxSz = Max_Tx_Size_Rect[MiSize]
            txW4 = Tx_Width[maxTxSz] // MI_SIZE
            txH4 = Tx_Height[maxTxSz] // MI_SIZE

            for row in range(MiRow, MiRow + bh4, txH4):
                for col in range(MiCol, MiCol + bw4, txW4):
                    self.__read_var_tx_size(av1, row, col, maxTxSz, 0)
        else:
            self.__read_tx_size(
                av1, not tile_group.skip or not tile_group.is_inter)

            for row in range(MiRow, MiRow + bh4):
                for col in range(MiCol, MiCol + bw4):
                    tile_group.InterTxSizes[row][col] = tile_group.TxSize

    def __read_var_tx_size(self, av1: AV1Decoder,
                           row: int, col: int, txSz: TX_SIZE, depth: int):
        """
        读取可变变换尺寸树
        规范文档 5.11.17 Var TX size syntax
        """
        frame_header = av1.frame_header
        tile_group = self.tile_group
        MiRows = frame_header.MiRows
        MiCols = frame_header.MiCols

        if row >= MiRows or col >= MiCols:
            return

        if txSz == TX_SIZE.TX_4X4 or depth == MAX_VARTX_DEPTH:
            txfm_split = 0
        else:
            txfm_split = read_S(av1, "txfm_split", row=row, col=col, txSz=txSz)

        w4 = Tx_Width[txSz] // MI_SIZE
        h4 = Tx_Height[txSz] // MI_SIZE

        if txfm_split:
            subTxSz = Split_Tx_Size[txSz]
            stepW = Tx_Width[subTxSz] // MI_SIZE
            stepH = Tx_Height[subTxSz] // MI_SIZE
            for i in range(0, h4, stepH):
                for j in range(0, w4, stepW):
                    self.__read_var_tx_size(
                        av1, row + i, col + j, subTxSz, depth + 1)
        else:
            for i in range(h4):
                for j in range(w4):
                    tile_group.InterTxSizes[row + i][col + j] = txSz
            tile_group.TxSize = txSz

    def __inter_frame_mode_info(self, av1: AV1Decoder):
        """
        解析帧间模式信息
        规范文档 5.11.18 Inter frame mode info syntax
        """
        frame_header = av1.frame_header
        tile_group = self.tile_group
        MiRow = tile_group.MiRow
        MiCol = tile_group.MiCol
        AvailU = tile_group.AvailU
        AvailL = tile_group.AvailL

        tile_group.use_intrabc = 0
        tile_group.LeftRefFrame[0] = tile_group.RefFrames[MiRow][MiCol -
                                                                 1][0] if AvailL else REF_FRAME.INTRA_FRAME
        tile_group.LeftRefFrame[1] = tile_group.RefFrames[MiRow][MiCol -
                                                                 1][1] if AvailL else REF_FRAME.NONE
        tile_group.AboveRefFrame[0] = tile_group.RefFrames[MiRow -
                                                           1][MiCol][0] if AvailU else REF_FRAME.INTRA_FRAME
        tile_group.AboveRefFrame[1] = tile_group.RefFrames[MiRow -
                                                           1][MiCol][1] if AvailU else REF_FRAME.NONE
        tile_group.LeftIntra = tile_group.LeftRefFrame[0] <= REF_FRAME.INTRA_FRAME
        tile_group.AboveIntra = tile_group.AboveRefFrame[0] <= REF_FRAME.INTRA_FRAME
        tile_group.LeftSingle = tile_group.LeftRefFrame[1] <= REF_FRAME.INTRA_FRAME
        tile_group.AboveSingle = tile_group.AboveRefFrame[1] <= REF_FRAME.INTRA_FRAME
        tile_group.skip = 0

        self.__inter_segment_id(av1, 1)
        self.__read_skip_mode(av1)

        if tile_group.skip_mode:
            tile_group.skip = 1
        else:
            self.__read_skip(av1)

        if not frame_header.SegIdPreSkip:
            self.__inter_segment_id(av1, 0)

        tile_group.Lossless = frame_header.LosslessArray[tile_group.segment_id]

        self.__read_cdef(av1)
        self.__read_delta_qindex(av1)
        self.__read_delta_lf(av1)
        tile_group.ReadDeltas = 0
        self.__read_is_inter(av1)

        if tile_group.is_inter:
            self.__inter_block_mode_info(av1)
        else:
            self.__intra_block_mode_info(av1)

    def __inter_segment_id(self, av1: AV1Decoder, preSkip: int):
        """
        解析帧间Segment ID
        规范文档 5.11.19 Inter segment ID syntax
        """
        frame_header = av1.frame_header
        tile_group = self.tile_group
        MiRow = tile_group.MiRow
        MiCol = tile_group.MiCol
        MiSize = tile_group.MiSize

        if frame_header.segmentation_enabled:
            predictedSegmentId = self.__get_segment_id(av1)
            if frame_header.segmentation_update_map:
                if preSkip and not frame_header.SegIdPreSkip:
                    tile_group.segment_id = 0
                    return

                if not preSkip:
                    if tile_group.skip:
                        seg_id_predicted = 0
                        for i in range(Num_4x4_Blocks_Wide[MiSize]):
                            tile_group.AboveSegPredContext[MiCol +
                                                           i] = seg_id_predicted

                        for i in range(Num_4x4_Blocks_High[MiSize]):
                            tile_group.LeftSegPredContext[MiRow +
                                                          i] = seg_id_predicted

                        self.__read_segment_id(av1)
                        return

                if frame_header.segmentation_temporal_update == 1:
                    seg_id_predicted = read_S(av1, "seg_id_predicted")
                    if seg_id_predicted:
                        tile_group.segment_id = predictedSegmentId
                    else:
                        self.__read_segment_id(av1)

                    for i in range(Num_4x4_Blocks_Wide[MiSize]):
                        tile_group.AboveSegPredContext[MiCol +
                                                       i] = seg_id_predicted

                    for i in range(Num_4x4_Blocks_High[MiSize]):
                        tile_group.LeftSegPredContext[MiRow +
                                                      i] = seg_id_predicted
                else:
                    self.__read_segment_id(av1)
            else:
                tile_group.segment_id = predictedSegmentId
        else:
            tile_group.segment_id = 0

    def __read_is_inter(self, av1: AV1Decoder):
        """
        读取is_inter标志
        规范文档 5.11.20 Is inter syntax
        """
        frame_header = av1.frame_header
        tile_group = self.tile_group

        if tile_group.skip_mode:
            tile_group.is_inter = 1
        elif self.__seg_feature_active(av1, SEG_LVL_REF_FRAME):
            tile_group.is_inter = frame_header.FeatureData[
                tile_group.segment_id][SEG_LVL_REF_FRAME] != REF_FRAME.INTRA_FRAME
        elif self.__seg_feature_active(av1, SEG_LVL_GLOBALMV):
            tile_group.is_inter = 1
        else:
            tile_group.is_inter = read_S(av1, 'is_inter')

    def __get_segment_id(self, av1: AV1Decoder) -> int:
        """
        获取Segment ID
        规范文档 5.11.21 Get segment ID function
        """
        frame_header = av1.frame_header
        tile_group = self.tile_group
        MiRows = frame_header.MiRows
        MiCols = frame_header.MiCols
        MiRow = tile_group.MiRow
        MiCol = tile_group.MiCol
        MiSize = tile_group.MiSize

        bw4 = Num_4x4_Blocks_Wide[MiSize]
        bh4 = Num_4x4_Blocks_High[MiSize]
        xMis = min(MiCols - MiCol, bw4)
        yMis = min(MiRows - MiRow, bh4)
        seg = 7
        for y in range(yMis):
            for x in range(xMis):
                seg = min(
                    seg, frame_header.PrevSegmentIds[MiRow + y][MiCol + x])
        return seg

    def __intra_block_mode_info(self, av1: AV1Decoder):
        """
        解析帧内块模式信息（在帧间帧中的帧内块）
        规范文档 5.11.22 Intra block mode info syntax
        """
        frame_header = av1.frame_header
        tile_group = self.tile_group
        MiSize = tile_group.MiSize
        HasChroma = tile_group.HasChroma

        tile_group.RefFrame[0] = REF_FRAME.INTRA_FRAME
        tile_group.RefFrame[1] = REF_FRAME.NONE
        y_mode = Y_MODE(read_S(av1, 'y_mode'))
        tile_group.YMode = y_mode

        self.__intra_angle_info_y(av1)

        if HasChroma:
            uv_mode = Y_MODE(read_S(av1, 'uv_mode'))
            tile_group.UVMode = uv_mode
            if tile_group.UVMode == Y_MODE.UV_CFL_PRED:
                self.__read_cfl_alphas(av1)
            self.__intra_angle_info_uv(av1)

        tile_group.PaletteSizeY = 0
        tile_group.PaletteSizeUV = 0
        if (MiSize >= SUB_SIZE.BLOCK_8X8 and
            Block_Width[MiSize] <= 64 and
            Block_Height[MiSize] <= 64 and
                frame_header.allow_screen_content_tools):
            self.__palette_mode_info(av1)

        self.__filter_intra_mode_info(av1)

    def __inter_block_mode_info(self, av1: AV1Decoder):
        """
        解析帧间块模式信息
        规范文档 5.11.23 Inter block mode info syntax
        """
        seq_header = av1.seq_header
        frame_header = av1.frame_header
        tile_group = self.tile_group

        def has_nearmv() -> int:
            """
            检查是否有NEARMV模式
            规范文档 5.11.23 has_nearmv()

            Args:
                YMode: Y模式值

            Returns:
                是否有NEARMV
            """
            YMode = tile_group.YMode
            return YMode in [Y_MODE.NEARMV, Y_MODE.NEAR_NEARMV, Y_MODE.NEAR_NEWMV, Y_MODE.NEW_NEARMV]

        def needs_interp_filter() -> int:
            """
            检查是否需要插值滤波器
            规范文档 5.11.23 needs_interp_filter()

            Returns:
                是否需要插值滤波器
            """
            MiSize = tile_group.MiSize

            large = (min(Block_Width[MiSize], Block_Height[MiSize]) >= 8)

            if tile_group.skip_mode or tile_group.motion_mode == MOTION_MODE.LOCALWARP:
                return 0
            elif large and tile_group.YMode == Y_MODE.GLOBALMV:
                return frame_header.GmType[tile_group.RefFrame[0]] == GM_TYPE.TRANSLATION
            elif large and tile_group.YMode == Y_MODE.GLOBAL_GLOBALMV:
                return frame_header.GmType[tile_group.RefFrame[0]] == GM_TYPE.TRANSLATION or frame_header.GmType[tile_group.RefFrame[1]] == GM_TYPE.TRANSLATION
            else:
                return 1

        tile_group.PaletteSizeY = 0
        tile_group.PaletteSizeUV = 0

        self.__read_ref_frames(av1)

        isCompound = (tile_group.RefFrame[1] > REF_FRAME.INTRA_FRAME)

        tile_group.RefStackMv = find_mv_stack(av1, isCompound)

        if tile_group.skip_mode:
            tile_group.YMode = Y_MODE.NEAREST_NEARESTMV
        elif (self.__seg_feature_active(av1, SEG_LVL_SKIP) or
              self.__seg_feature_active(av1, SEG_LVL_GLOBALMV)):
            tile_group.YMode = Y_MODE.GLOBALMV
        elif isCompound:
            compound_mode = read_S(av1, 'compound_mode')
            tile_group.YMode = Y_MODE(Y_MODE.NEAREST_NEARESTMV + compound_mode)
        else:
            new_mv = read_S(av1, 'new_mv')
            if new_mv == 0:
                tile_group.YMode = Y_MODE.NEWMV
            else:
                zero_mv = read_S(av1, "zero_mv")
                if zero_mv == 0:
                    tile_group.YMode = Y_MODE.GLOBALMV
                else:
                    ref_mv = read_S(av1, 'ref_mv')
                    tile_group.YMode = Y_MODE.NEARESTMV if ref_mv == 0 else Y_MODE.NEARMV

        tile_group.RefMvIdx = 0
        if tile_group.YMode in [Y_MODE.NEWMV, Y_MODE.NEW_NEWMV]:
            for idx in range(2):
                if tile_group.NumMvFound > idx + 1:
                    drl_mode = read_S(av1, "drl_mode", idx=idx)
                    if drl_mode == 0:
                        tile_group.RefMvIdx = idx
                        break
                    tile_group.RefMvIdx = idx + 1
        elif has_nearmv():
            tile_group.RefMvIdx = 1
            for idx in range(1, 3):
                if tile_group.NumMvFound > idx + 1:
                    drl_mode = read_S(av1, "drl_mode", idx=idx)
                    if drl_mode == 0:
                        tile_group.RefMvIdx = idx
                        break
                    tile_group.RefMvIdx = idx + 1

        self.__assign_mv(av1, isCompound)
        self.__read_interintra_mode(av1, isCompound)
        self.__read_motion_mode(av1, isCompound)
        self.__read_compound_type(av1, isCompound)

        if frame_header.interpolation_filter == INTERPOLATION_FILTER.SWITCHABLE:
            for dir_val in range(2 if seq_header.enable_dual_filter else 1):
                if needs_interp_filter():
                    tile_group.interp_filter[dir_val] = INTERPOLATION_FILTER(read_S(
                        av1, 'interp_filter', dir_val=dir_val))
                else:
                    tile_group.interp_filter[dir_val] = INTERPOLATION_FILTER.EIGHTTAP
            if not seq_header.enable_dual_filter:
                tile_group.interp_filter[1] = tile_group.interp_filter[0]
        else:
            tile_group.interp_filter[0] = frame_header.interpolation_filter
            tile_group.interp_filter[1] = frame_header.interpolation_filter

    def __filter_intra_mode_info(self, av1: AV1Decoder):
        """
        读取滤波器帧内模式信息
        规范文档 5.11.24 Filter intra mode info syntax
        """
        seq_header = av1.seq_header
        tile_group = self.tile_group
        MiSize = tile_group.MiSize

        tile_group.use_filter_intra = 0
        if (seq_header.enable_filter_intra and
            tile_group.YMode == Y_MODE.DC_PRED and
            tile_group.PaletteSizeY == 0 and
                max(Block_Width[MiSize], Block_Height[MiSize]) <= 32):
            tile_group.use_filter_intra = read_S(av1, 'use_filter_intra')
            if tile_group.use_filter_intra:
                tile_group.filter_intra_mode = FILTER_INTRA_MODE(read_S(av1, 'filter_intra_mode'))

    def __read_ref_frames(self, av1: AV1Decoder):
        """
        读取参考帧
        规范文档 5.11.25 Ref frames syntax
        """
        frame_header = av1.frame_header
        tile_group = self.tile_group
        MiSize = tile_group.MiSize

        if tile_group.skip_mode:
            tile_group.RefFrame[0] = frame_header.SkipModeFrame[0]
            tile_group.RefFrame[1] = frame_header.SkipModeFrame[1]
        elif self.__seg_feature_active(av1, SEG_LVL_REF_FRAME):
            tile_group.RefFrame[0] = REF_FRAME(
                frame_header.FeatureData[tile_group.segment_id][SEG_LVL_REF_FRAME])
            tile_group.RefFrame[1] = REF_FRAME.NONE
        elif (self.__seg_feature_active(av1, SEG_LVL_SKIP) or
              self.__seg_feature_active(av1, SEG_LVL_GLOBALMV)):
            tile_group.RefFrame[0] = REF_FRAME.LAST_FRAME
            tile_group.RefFrame[1] = REF_FRAME.NONE
        else:
            bw4 = Num_4x4_Blocks_Wide[MiSize]
            bh4 = Num_4x4_Blocks_High[MiSize]
            if frame_header.reference_select and (min(bw4, bh4) >= 2):
                comp_mode = COMP_MODE(read_S(av1, 'comp_mode'))
            else:
                comp_mode = COMP_MODE.SINGLE_REFERENCE
            if comp_mode == COMP_MODE.COMPOUND_REFERENCE:
                comp_ref_type = read_S(av1, 'comp_ref_type')
                if comp_ref_type == COMP_REF_TYPE.UNIDIR_COMP_REFERENCE:
                    uni_comp_ref = read_S(av1, 'uni_comp_ref')
                    if uni_comp_ref:
                        tile_group.RefFrame[0] = REF_FRAME.BWDREF_FRAME
                        tile_group.RefFrame[1] = REF_FRAME.ALTREF_FRAME
                    else:
                        uni_comp_ref_p1 = read_S(av1, 'uni_comp_ref_p1')
                        if uni_comp_ref_p1:
                            uni_comp_ref_p2 = read_S(av1, 'uni_comp_ref_p2')
                            if uni_comp_ref_p2:
                                tile_group.RefFrame[0] = REF_FRAME.LAST_FRAME
                                tile_group.RefFrame[1] = REF_FRAME.GOLDEN_FRAME
                            else:
                                tile_group.RefFrame[0] = REF_FRAME.LAST_FRAME
                                tile_group.RefFrame[1] = REF_FRAME.LAST3_FRAME
                        else:
                            tile_group.RefFrame[0] = REF_FRAME.LAST_FRAME
                            tile_group.RefFrame[1] = REF_FRAME.LAST2_FRAME
                else:
                    comp_ref = read_S(av1, 'comp_ref')
                    if comp_ref == 0:
                        comp_ref_p1 = read_S(av1, 'comp_ref_p1')
                        tile_group.RefFrame[0] = REF_FRAME.LAST2_FRAME if comp_ref_p1 else REF_FRAME.LAST_FRAME
                    else:
                        comp_ref_p2 = read_S(av1, 'comp_ref_p2')
                        tile_group.RefFrame[0] = REF_FRAME.GOLDEN_FRAME if comp_ref_p2 else REF_FRAME.LAST3_FRAME

                    comp_bwdref = read_S(av1, 'comp_bwdref')
                    if comp_bwdref == 0:
                        comp_bwdref_p1 = read_S(av1, 'comp_bwdref_p1')
                        tile_group.RefFrame[1] = REF_FRAME.ALTREF2_FRAME if comp_bwdref_p1 else REF_FRAME.BWDREF_FRAME
                    else:
                        tile_group.RefFrame[1] = REF_FRAME.ALTREF_FRAME
            else:
                single_ref_p1 = read_S(av1, 'single_ref_p1')
                if single_ref_p1:
                    single_ref_p2 = read_S(av1, 'single_ref_p2')
                    if single_ref_p2 == 0:
                        single_ref_p6 = read_S(av1, 'single_ref_p6')
                        tile_group.RefFrame[0] = REF_FRAME.ALTREF2_FRAME if single_ref_p6 else REF_FRAME.BWDREF_FRAME
                    else:
                        tile_group.RefFrame[0] = REF_FRAME.ALTREF_FRAME
                else:
                    single_ref_p3 = read_S(av1, 'single_ref_p3')
                    if single_ref_p3:
                        single_ref_p5 = read_S(av1, 'single_ref_p5')
                        tile_group.RefFrame[0] = REF_FRAME.GOLDEN_FRAME if single_ref_p5 else REF_FRAME.LAST3_FRAME
                    else:
                        single_ref_p4 = read_S(av1, 'single_ref_p4')
                        tile_group.RefFrame[0] = REF_FRAME.LAST2_FRAME if single_ref_p4 else REF_FRAME.LAST_FRAME

                tile_group.RefFrame[1] = REF_FRAME.NONE

        # It is a requirement of bitstream conformance that the following conditions are met whenever the parsing process returns from the read_ref_frames syntax:
        # - RefFrame[ 0 ] = LAST_FRAME
        # - RefFrame[ 1 ] = NONE
        if False:
            assert tile_group.RefFrame[0] == REF_FRAME.LAST_FRAME
            assert tile_group.RefFrame[1] == REF_FRAME.NONE

    def __assign_mv(self, av1: AV1Decoder, isCompound: int):
        """
        分配运动向量
        规范文档 5.11.26 Assign MV syntax

        """
        seq_header = av1.seq_header
        tile_group = self.tile_group
        use_128x128_superblock = seq_header.use_128x128_superblock
        MiRow = tile_group.MiRow

        for i in range(1 + isCompound):
            if tile_group.use_intrabc:
                compMode = Y_MODE.NEWMV
            else:
                compMode = self.__get_mode(av1, i)

            if tile_group.use_intrabc:
                tile_group.PredMv[0] = deepcopy(tile_group.RefStackMv[0][0])

                if tile_group.PredMv[0][0] == 0 and tile_group.PredMv[0][1] == 0:
                    tile_group.PredMv[0] = deepcopy(
                        tile_group.RefStackMv[1][0])

                if tile_group.PredMv[0][0] == 0 and tile_group.PredMv[0][1] == 0:
                    sbSize = SUB_SIZE.BLOCK_128X128 if use_128x128_superblock else SUB_SIZE.BLOCK_64X64
                    sbSize4 = Num_4x4_Blocks_High[sbSize]
                    if MiRow - sbSize4 < tile_group.MiRowStart:
                        tile_group.PredMv[0][0] = 0
                        tile_group.PredMv[0][1] = (
                            sbSize4 * MI_SIZE + INTRABC_DELAY_PIXELS) * -8
                    else:
                        tile_group.PredMv[0][0] = -(sbSize4 * MI_SIZE * 8)
                        tile_group.PredMv[0][1] = 0

            elif compMode == Y_MODE.GLOBALMV:
                tile_group.PredMv[i] = deepcopy(tile_group.GlobalMvs[i])
            else:
                pos = 0 if compMode == Y_MODE.NEARESTMV else tile_group.RefMvIdx
                if compMode == Y_MODE.NEWMV and tile_group.NumMvFound <= 1:
                    pos = 0
                tile_group.PredMv[i] = deepcopy(tile_group.RefStackMv[pos][i])

            if compMode == Y_MODE.NEWMV:
                self.__read_mv(av1, i)
            else:
                tile_group.Mv[i] = deepcopy(tile_group.PredMv[i])

        # It is a requirement of bitstream conformance that whenever assign_mv returns, the function is_mv_valid(isCompound) would return 1
        assert self._is_mv_valid(av1, isCompound) == 1

    def __read_motion_mode(self, av1: AV1Decoder, isCompound: bool):
        """
        读取运动模式
        规范文档 5.11.27 Read motion mode syntax

        """
        frame_header = av1.frame_header
        tile_group = self.tile_group
        MiSize = tile_group.MiSize

        if tile_group.skip_mode:
            tile_group.motion_mode = MOTION_MODE.SIMPLE
            return

        if not frame_header.is_motion_mode_switchable:
            tile_group.motion_mode = MOTION_MODE.SIMPLE
            return

        if min(Block_Width[MiSize], Block_Height[MiSize]) < 8:
            tile_group.motion_mode = MOTION_MODE.SIMPLE
            return

        if not frame_header.force_integer_mv and tile_group.YMode in [Y_MODE.GLOBALMV, Y_MODE.GLOBAL_GLOBALMV]:
            if frame_header.GmType[tile_group.RefFrame[0]] > GM_TYPE.TRANSLATION:
                tile_group.motion_mode = MOTION_MODE.SIMPLE
                return

        from mode.find_mv_stack import has_overlappable_candidates
        if isCompound or tile_group.RefFrame[1] == REF_FRAME.INTRA_FRAME or not has_overlappable_candidates(av1):
            tile_group.motion_mode = MOTION_MODE.SIMPLE
            return

        from mode.find_mv_stack import FindMvStack
        find_warp_samples_impl = FindMvStack()
        find_warp_samples_impl.find_warp_samples_process(av1)

        if frame_header.force_integer_mv or tile_group.NumSamples == 0 or not frame_header.allow_warped_motion or is_scaled(av1, tile_group.RefFrame[0]):
            use_obmc = read_S(av1, 'use_obmc')
            tile_group.motion_mode = MOTION_MODE.OBMC if use_obmc else MOTION_MODE.SIMPLE
        else:
            tile_group.motion_mode = read_S(av1, 'motion_mode')

    def __read_interintra_mode(self, av1: AV1Decoder, isCompound: int):
        """
        读取inter-intra模式
        规范文档 5.11.28 Read inter intra syntax

        """
        seq_header = av1.seq_header
        tile_group = self.tile_group
        MiSize = tile_group.MiSize

        if (not tile_group.skip_mode and seq_header.enable_interintra_compound and not isCompound and
                MiSize >= SUB_SIZE.BLOCK_8X8 and MiSize <= SUB_SIZE.BLOCK_32X32):
            tile_group.interintra = read_S(av1, 'interintra')
            if tile_group.interintra:
                tile_group.interintra_mode = read_S(av1, 'interintra_mode')
                tile_group.RefFrame[1] = REF_FRAME.INTRA_FRAME
                tile_group.AngleDeltaY = 0
                tile_group.AngleDeltaUV = 0
                tile_group.use_filter_intra = 0
                tile_group.wedge_interintra = read_S(av1, 'wedge_interintra')
                if tile_group.wedge_interintra:
                    tile_group.wedge_index = read_S(av1, 'wedge_index')
                    tile_group.wedge_sign = 0
        else:
            tile_group.interintra = 0

    def __read_compound_type(self, av1: AV1Decoder, isCompound: int):
        """
        读取compound类型
        规范文档 5.11.29 Read compound type syntax
        """
        seq_header = av1.seq_header
        tile_group = self.tile_group
        MiSize = tile_group.MiSize

        tile_group.comp_group_idx = 0
        tile_group.compound_idx = 1
        if tile_group.skip_mode:
            tile_group.compound_type = COMPOUND_TYPE.COMPOUND_AVERAGE
            return

        if isCompound:
            n = Wedge_Bits[MiSize]
            if seq_header.enable_masked_compound:
                tile_group.comp_group_idx = read_S(av1, 'comp_group_idx')
            if tile_group.comp_group_idx == 0:
                if seq_header.enable_jnt_comp:
                    tile_group.compound_idx = read_S(av1, 'compound_idx')
                    tile_group.compound_type = COMPOUND_TYPE.COMPOUND_AVERAGE if tile_group.compound_idx else COMPOUND_TYPE.COMPOUND_DISTANCE
                else:
                    tile_group.compound_type = COMPOUND_TYPE.COMPOUND_AVERAGE
            else:
                if n == 0:
                    tile_group.compound_type = COMPOUND_TYPE.COMPOUND_DIFFWTD
                else:
                    tile_group.compound_type = read_S(av1, 'compound_type')
            if tile_group.compound_type == COMPOUND_TYPE.COMPOUND_WEDGE:
                tile_group.wedge_index = read_S(av1, 'wedge_index')
                tile_group.wedge_sign = read_L(av1, 1)
            elif tile_group.compound_type == COMPOUND_TYPE.COMPOUND_DIFFWTD:
                tile_group.mask_type = read_L(av1, 1)
        else:
            if tile_group.interintra:
                tile_group.compound_type = COMPOUND_TYPE.COMPOUND_WEDGE if tile_group.wedge_interintra else COMPOUND_TYPE.COMPOUND_INTRA
            else:
                tile_group.compound_type = COMPOUND_TYPE.COMPOUND_AVERAGE

    def __get_mode(self, av1: AV1Decoder, refList: int) -> Y_MODE:
        """
        获取模式
        规范文档 5.11.30 Get mode function

        Args:
            refList: 参考列表索引（0或1）

        Returns:
            compMode值
        """
        tile_group = self.tile_group
        YMode = tile_group.YMode

        if refList == 0:
            if YMode < Y_MODE.NEAREST_NEARESTMV:
                compMode = YMode
            elif YMode in [Y_MODE.NEW_NEWMV, Y_MODE.NEW_NEARESTMV, Y_MODE.NEW_NEARMV]:
                compMode = Y_MODE.NEWMV
            elif YMode in [Y_MODE.NEAREST_NEARESTMV, Y_MODE.NEAREST_NEWMV]:
                compMode = Y_MODE.NEARESTMV
            elif YMode in [Y_MODE.NEAR_NEARMV, Y_MODE.NEAR_NEWMV]:
                compMode = Y_MODE.NEARMV
            else:
                compMode = Y_MODE.GLOBALMV
        else:
            if YMode in [Y_MODE.NEW_NEWMV, Y_MODE.NEAREST_NEWMV, Y_MODE.NEAR_NEWMV]:
                compMode = Y_MODE.NEWMV
            elif YMode in [Y_MODE.NEAREST_NEARESTMV, Y_MODE.NEW_NEARESTMV]:
                compMode = Y_MODE.NEARESTMV
            elif YMode in [Y_MODE.NEAR_NEARMV, Y_MODE.NEW_NEARMV]:
                compMode = Y_MODE.NEARMV
            else:
                compMode = Y_MODE.GLOBALMV

        return compMode

    def __read_mv(self, av1: AV1Decoder, ref: int):
        """
        读取运动向量
        规范文档 5.11.31 MV syntax

        Args:
            ref: 参考索引（0或1）
        """
        tile_group = self.tile_group

        diffMv = [0, 0]
        if tile_group.use_intrabc:
            tile_group.MvCtx = MV_INTRABC_CONTEXT
        else:
            tile_group.MvCtx = 0

        mv_joint = MV_JOINT(read_S(av1, 'mv_joint'))
        if mv_joint in [MV_JOINT.MV_JOINT_HZVNZ, MV_JOINT.MV_JOINT_HNZVNZ]:
            diffMv[0] = self.__read_mv_component(av1, 0)
        if mv_joint in [MV_JOINT.MV_JOINT_HNZVZ, MV_JOINT.MV_JOINT_HNZVNZ]:
            diffMv[1] = self.__read_mv_component(av1, 1)

        tile_group.Mv[ref][0] = tile_group.PredMv[ref][0] + diffMv[0]
        tile_group.Mv[ref][1] = tile_group.PredMv[ref][1] + diffMv[1]

    def __read_mv_component(self, av1: AV1Decoder, comp: int) -> int:
        """
        读取运动向量分量
        规范文档 5.11.32 MV component syntax

        Args:
            comp: 分量索引（0=行，1=列）

        Returns:
            MV分量差值
        """
        frame_header = av1.frame_header

        mv_sign = read_S(av1, 'mv_sign', comp=comp)
        mv_class = read_S(av1, 'mv_class', comp=comp)
        if mv_class == MV_CLASS.MV_CLASS_0:
            mv_class0_bit = read_S(av1, 'mv_class0_bit', comp=comp)
            force_integer_mv = frame_header.force_integer_mv
            if force_integer_mv:
                mv_class0_fr = 3
            else:
                mv_class0_fr = read_S(
                    av1, 'mv_class0_fr', comp=comp, mv_class0_bit=mv_class0_bit)

            if frame_header.allow_high_precision_mv:
                mv_class0_hp = read_S(av1, 'mv_class0_hp', comp=comp)
            else:
                mv_class0_hp = 1

            mag = ((mv_class0_bit << 3) | (
                mv_class0_fr << 1) | mv_class0_hp) + 1
        else:
            d = 0
            for i in range(mv_class):
                mv_bit = read_S(av1, 'mv_bit', comp=comp, i=i)
                d |= mv_bit << i

            mag = CLASS0_SIZE << (mv_class + 2)

            if frame_header.force_integer_mv:
                mv_fr = 3
            else:
                mv_fr = read_S(av1, 'mv_fr', comp=comp)

            if frame_header.allow_high_precision_mv:
                mv_hp = read_S(av1, 'mv_hp', comp=comp)
            else:
                mv_hp = 1

            mag += ((d << 3) | (mv_fr << 1) | mv_hp) + 1

        return -mag if mv_sign else mag

    def __compute_prediction(self, av1: AV1Decoder):
        """
        计算预测
        规范文档 5.11.33 Compute prediction syntax
        """
        seq_header = av1.seq_header
        tile_group = self.tile_group
        use_128x128_superblock = seq_header.use_128x128_superblock
        subsampling_x = seq_header.color_config.subsampling_x
        subsampling_y = seq_header.color_config.subsampling_y
        MiRow = tile_group.MiRow
        MiCol = tile_group.MiCol
        MiSize = tile_group.MiSize
        HasChroma = tile_group.HasChroma
        AvailU = tile_group.AvailU
        AvailL = tile_group.AvailL

        sbMask = 31 if use_128x128_superblock else 15
        subBlockMiRow = MiRow & sbMask
        subBlockMiCol = MiCol & sbMask

        for plane in range(1 + HasChroma * 2):
            planeSz = get_plane_residual_size(av1, MiSize, plane)
            num4x4W = Num_4x4_Blocks_Wide[planeSz]
            num4x4H = Num_4x4_Blocks_High[planeSz]
            log2W = MI_SIZE_LOG2 + Mi_Width_Log2[planeSz]
            log2H = MI_SIZE_LOG2 + Mi_Height_Log2[planeSz]
            subX = subsampling_x if plane > 0 else 0
            subY = subsampling_y if plane > 0 else 0
            baseX = (MiCol >> subX) * MI_SIZE
            baseY = (MiRow >> subY) * MI_SIZE
            candRow = (MiRow >> subY) << subY
            candCol = (MiCol >> subX) << subX

            tile_group.IsInterIntra = (
                tile_group.is_inter and tile_group.RefFrame[1] == REF_FRAME.INTRA_FRAME)
            if tile_group.IsInterIntra:
                if tile_group.interintra_mode == INTERINTRA_MODE.II_DC_PRED:
                    mode = Y_MODE.DC_PRED
                elif tile_group.interintra_mode == INTERINTRA_MODE.II_V_PRED:
                    mode = Y_MODE.V_PRED
                elif tile_group.interintra_mode == INTERINTRA_MODE.II_H_PRED:
                    mode = Y_MODE.H_PRED
                else:
                    mode = Y_MODE.SMOOTH_PRED

                from frame.decoding_process import predict_intra
                predict_intra(av1, plane, baseX, baseY,
                              AvailL if plane == 0 else av1.tile_group.AvailLChroma,
                              AvailU if plane == 0 else av1.tile_group.AvailUChroma,
                              tile_group.BlockDecoded[plane][(
                                  subBlockMiRow >> subY) - 1][(subBlockMiCol >> subX) + num4x4W],
                              tile_group.BlockDecoded[plane][(
                                  subBlockMiRow >> subY) + num4x4H][(subBlockMiCol >> subX) - 1],
                              mode, log2W, log2H)

            if tile_group.is_inter:
                predW = Block_Width[MiSize] >> subX
                predH = Block_Height[MiSize] >> subY
                someUseIntra = 0
                for r in range(num4x4H << subY):
                    for c in range(num4x4W << subX):
                        if tile_group.RefFrames[candRow + r][candCol + c][0] == REF_FRAME.INTRA_FRAME:
                            someUseIntra = 1

                if someUseIntra:
                    predW = num4x4W * 4
                    predH = num4x4H * 4
                    candRow = MiRow
                    candCol = MiCol

                r = 0
                for y in range(0, num4x4H * 4, predH):
                    c = 0
                    for x in range(0, num4x4W * 4, predW):
                        from frame.decoding_process import predict_inter
                        predict_inter(av1, plane, baseX + x, baseY + y,
                                      predW, predH, candRow + r, candCol + c)
                        c += 1
                    r += 1

    def __residual(self, av1: AV1Decoder):
        """
        残差解码主函数
        规范文档 5.11.34 Residual syntax
        """
        seq_header = av1.seq_header
        frame_header = av1.frame_header
        tile_group = self.tile_group
        use_128x128_superblock = seq_header.use_128x128_superblock
        subsampling_x = seq_header.color_config.subsampling_x
        subsampling_y = seq_header.color_config.subsampling_y
        MiRows = frame_header.MiRows
        MiCols = frame_header.MiCols
        MiRow = tile_group.MiRow
        MiCol = tile_group.MiCol
        MiSize = tile_group.MiSize
        HasChroma = tile_group.HasChroma

        sbMask = 31 if use_128x128_superblock else 15

        widthChunks = max(1, Block_Width[MiSize] >> 6)
        heightChunks = max(1, Block_Height[MiSize] >> 6)

        miSizeChunk = SUB_SIZE.BLOCK_64X64 if (
            widthChunks > 1 or heightChunks > 1) else MiSize

        for chunkY in range(heightChunks):
            for chunkX in range(widthChunks):
                miRowChunk = MiRow + (chunkY << 4)
                miColChunk = MiCol + (chunkX << 4)
                subBlockMiRow = miRowChunk & sbMask
                subBlockMiCol = miColChunk & sbMask

                for plane in range(1 + HasChroma * 2):
                    txSz = TX_SIZE.TX_4X4 if tile_group.Lossless else self.__get_tx_size(
                        av1, plane, tile_group.TxSize)
                    stepX = Tx_Width[txSz] >> 2
                    stepY = Tx_Height[txSz] >> 2
                    planeSz = get_plane_residual_size(av1, miSizeChunk, plane)
                    num4x4W = Num_4x4_Blocks_Wide[planeSz]
                    num4x4H = Num_4x4_Blocks_High[planeSz]
                    subX = subsampling_x if plane > 0 else 0
                    subY = subsampling_y if plane > 0 else 0
                    baseX = (miColChunk >> subX) * MI_SIZE
                    baseY = (miRowChunk >> subY) * MI_SIZE
                    if tile_group.is_inter and not tile_group.Lossless and not plane:
                        self.__transform_tree(
                            av1, baseX, baseY, num4x4W * 4, num4x4H * 4)
                    else:
                        baseXBlock = (MiCol >> subX) * MI_SIZE
                        baseYBlock = (MiRow >> subY) * MI_SIZE
                        for y in range(0, num4x4H, stepY):
                            for x in range(0, num4x4W, stepX):
                                self.__transform_block(av1, plane, baseXBlock, baseYBlock, txSz,
                                                       x + ((chunkX << 4)
                                                            >> subX),
                                                       y + ((chunkY << 4) >> subY))

    def __transform_block(self, av1: AV1Decoder, plane: int, baseX: int, baseY: int, txSz: TX_SIZE, x: int, y: int):
        """
        变换块解码
        规范文档 5.11.35 Transform block syntax
        """
        seq_header = av1.seq_header
        frame_header = av1.frame_header
        tile_group = self.tile_group
        use_128x128_superblock = seq_header.use_128x128_superblock
        subsampling_x = seq_header.color_config.subsampling_x
        subsampling_y = seq_header.color_config.subsampling_y
        MiRows = frame_header.MiRows
        MiCols = frame_header.MiCols
        AvailU = tile_group.AvailU
        AvailL = tile_group.AvailL

        startX = baseX + 4 * x
        startY = baseY + 4 * y
        subX = subsampling_x if plane > 0 else 0
        subY = subsampling_y if plane > 0 else 0
        row = (startY << subY) >> MI_SIZE_LOG2
        col = (startX << subX) >> MI_SIZE_LOG2
        sbMask = 31 if use_128x128_superblock else 15
        subBlockMiRow = row & sbMask
        subBlockMiCol = col & sbMask
        stepX = Tx_Width[txSz] >> MI_SIZE_LOG2
        stepY = Tx_Height[txSz] >> MI_SIZE_LOG2
        maxX = (MiCols * MI_SIZE) >> subX
        maxY = (MiRows * MI_SIZE) >> subY
        if startX >= maxX or startY >= maxY:
            return

        if not tile_group.is_inter:
            if ((plane == 0 and tile_group.PaletteSizeY) or (plane != 0 and tile_group.PaletteSizeUV)):
                from frame.decoding_process import predict_palette
                predict_palette(av1, plane, startX, startY, x, y, txSz)
            else:
                isCfl = (plane > 0 and tile_group.UVMode == Y_MODE.UV_CFL_PRED)
                if plane == 0:
                    mode: Y_MODE = tile_group.YMode
                else:
                    mode = Y_MODE.DC_PRED if isCfl else tile_group.UVMode

                log2W = Tx_Width_Log2[txSz]
                log2H = Tx_Height_Log2[txSz]
                from frame.decoding_process import predict_intra
                predict_intra(av1, plane, startX, startY,
                              (AvailL if plane ==
                               0 else av1.tile_group.AvailLChroma) or x > 0,
                              (AvailU if plane ==
                               0 else av1.tile_group.AvailUChroma) or y > 0,
                              tile_group.BlockDecoded[plane][(
                                  subBlockMiRow >> subY) - 1][(subBlockMiCol >> subX) + stepX],
                              tile_group.BlockDecoded[plane][(
                                  subBlockMiRow >> subY) + stepY][(subBlockMiCol >> subX) - 1],
                              mode, log2W, log2H)
                if isCfl:
                    from frame.decoding_process import predict_chroma_from_luma
                    predict_chroma_from_luma(av1, plane, startX, startY, txSz)

            if plane == 0:
                tile_group.MaxLumaW = startX + stepX * 4
                tile_group.MaxLumaH = startY + stepY * 4

        if av1.on_pred_frame is not None:
            w = Tx_Width[txSz]
            h = Tx_Height[txSz]
            pred = Array(None, (h, w), 0)
            for i in range(h):
                for j in range(w):
                    pred[i][j] = av1.CurrFrame[plane][startY + i][startX + j]
            av1.on_pred_frame(plane, startX, startY, pred)

        if not tile_group.skip:
            eob = self.__coeffs(av1, plane, startX, startY, txSz)
            if eob > 0:
                from frame.decoding_process import reconstruct
                reconstruct(av1, plane, startX, startY, txSz)

        if av1.on_residual_frame is not None:
            log2W = Tx_Width_Log2[txSz]
            log2H = Tx_Height_Log2[txSz]
            w = 1 << log2W
            h = 1 << log2H
            residual = Array(None, (h, w), 0)
            for i in range(h):
                for j in range(w):
                    residual[i][j] = av1.CurrFrame[plane][startY + i][startX + j]
            av1.on_residual_frame(plane, startX, startY, residual)

        for i in range(stepY):
            for j in range(stepX):
                tile_group.LoopfilterTxSizes[plane][(
                    row >> subY) + i][(col >> subX) + j] = txSz
                tile_group.BlockDecoded[plane][(
                    subBlockMiRow >> subY) + i][(subBlockMiCol >> subX) + j] = 1

    def __transform_tree(self, av1: AV1Decoder, startX: int, startY: int, w: int, h: int):
        """
        变换树解码
        规范文档 5.11.36 Transform tree syntax
        """
        frame_header = av1.frame_header
        tile_group = self.tile_group
        MiRows = frame_header.MiRows
        MiCols = frame_header.MiCols

        maxX = MiCols * MI_SIZE
        maxY = MiRows * MI_SIZE
        if startX >= maxX or startY >= maxY:
            return

        row = startY >> MI_SIZE_LOG2
        col = startX >> MI_SIZE_LOG2
        lumaTxSz = tile_group.InterTxSizes[row][col]
        lumaW = Tx_Width[lumaTxSz]
        lumaH = Tx_Height[lumaTxSz]
        if w <= lumaW and h <= lumaH:
            from utils.tile_utils import find_tx_size
            txSz = find_tx_size(w, h)
            self.__transform_block(av1, 0, startX, startY, txSz, 0, 0)
        else:
            if w > h:
                self.__transform_tree(av1, startX, startY, w // 2, h)
                self.__transform_tree(av1, startX + w // 2, startY, w // 2, h)
            elif w < h:
                self.__transform_tree(av1, startX, startY, w, h // 2)
                self.__transform_tree(av1, startX, startY + h // 2, w, h // 2)
            else:
                self.__transform_tree(av1, startX, startY, w // 2, h // 2)
                self.__transform_tree(
                    av1, startX + w // 2, startY, w // 2, h // 2)
                self.__transform_tree(
                    av1, startX, startY + h // 2, w // 2, h // 2)
                self.__transform_tree(
                    av1, startX + w // 2, startY + h // 2, w // 2, h // 2)

    def __get_tx_size(self, av1: AV1Decoder, plane: int, txSz: TX_SIZE) -> TX_SIZE:
        """
        获取变换尺寸
        规范文档 5.11.37 Get TX size function

        Returns:
            变换尺寸
        """
        tile_group = self.tile_group
        MiSize = tile_group.MiSize

        if plane == 0:
            return txSz

        planeSz = get_plane_residual_size(av1, MiSize, plane)
        uvTx = Max_Tx_Size_Rect[planeSz]

        if Tx_Width[uvTx] == 64 or Tx_Height[uvTx] == 64:
            if Tx_Width[uvTx] == 16:
                return TX_SIZE.TX_16X32
            if Tx_Height[uvTx] == 16:
                return TX_SIZE.TX_32X16
            return TX_SIZE.TX_32X32
        return uvTx

    def __coeffs(self, av1: AV1Decoder,
                 plane: int, startX: int, startY: int, txSz: TX_SIZE) -> int:
        """
        系数解码
        规范文档 5.11.39 Coefficients syntax

        Args:
            plane: 平面索引
            startX: 起始X坐标
            startY: 起始Y坐标
            txSz: 变换尺寸

        Returns:
            EOB值（End of Block）
        """
        tile_group = self.tile_group

        x4 = startX >> 2
        y4 = startY >> 2
        w4 = Tx_Width[txSz] >> 2
        h4 = Tx_Height[txSz] >> 2

        txSzCtx = (Tx_Size_Sqr[txSz] + Tx_Size_Sqr_Up[txSz] + 1) >> 1
        ptype = plane > 0
        segEob = 512 if (txSz == TX_SIZE.TX_16X64 or txSz == TX_SIZE.TX_64X16) else min(
            1024, Tx_Width[txSz] * Tx_Height[txSz])

        tile_group.Quant = [0] * segEob
        tile_group.Dequant = Array(None, (64, 64), 0)

        eob = 0
        culLevel = 0
        dcCategory = 0

        all_zero = read_S(av1, 'all_zero', plane=plane, txSz=txSz,
                          txSzCtx=txSzCtx, x4=x4, y4=y4, w4=w4, h4=h4)
        if all_zero:
            c = 0
            if plane == 0:
                for i in range(w4):
                    for j in range(h4):
                        tile_group.TxTypes[y4 + j][x4 + i] = DCT_DCT
        else:
            if plane == 0:
                self.__transform_type(av1, x4, y4, txSz)

            tile_group.PlaneTxType = compute_tx_type(av1, plane, txSz, x4, y4)
            scan = self.__get_scan(av1, txSz)

            eobMultisize = min(
                Tx_Width_Log2[txSz], 5) + min(Tx_Height_Log2[txSz], 5) - 4
            if eobMultisize == 0:
                eob_pt_16 = read_S(av1, 'eob_pt_16', plane=plane,
                                   txSz=txSz, ptype=ptype, x4=x4, y4=y4)
                eobPt = eob_pt_16 + 1
            elif eobMultisize == 1:
                eob_pt_32 = read_S(av1, 'eob_pt_32', plane=plane,
                                   txSz=txSz, ptype=ptype, x4=x4, y4=y4)
                eobPt = eob_pt_32 + 1
            elif eobMultisize == 2:
                eob_pt_64 = read_S(av1, 'eob_pt_64', plane=plane,
                                   txSz=txSz, ptype=ptype, x4=x4, y4=y4)
                eobPt = eob_pt_64 + 1
            elif eobMultisize == 3:
                eob_pt_128 = read_S(
                    av1, 'eob_pt_128', plane=plane, txSz=txSz, ptype=ptype, x4=x4, y4=y4)
                eobPt = eob_pt_128 + 1
            elif eobMultisize == 4:
                eob_pt_256 = read_S(
                    av1, 'eob_pt_256', plane=plane, txSz=txSz, ptype=ptype, x4=x4, y4=y4)
                eobPt = eob_pt_256 + 1
            elif eobMultisize == 5:
                eob_pt_512 = read_S(av1, 'eob_pt_512', ptype=ptype)
                eobPt = eob_pt_512 + 1
            else:
                eob_pt_1024 = read_S(av1, 'eob_pt_1024', ptype=ptype)
                eobPt = eob_pt_1024 + 1

            eob = eobPt if eobPt < 2 else ((1 << (eobPt - 2)) + 1)
            eobShift = max(-1, eobPt - 3)
            if eobShift >= 0:
                eob_extra = read_S(
                    av1, 'eob_extra', txSzCtx=txSzCtx, ptype=ptype, eobPt=eobPt)
                if eob_extra:
                    eob += (1 << eobShift)

                for i in range(1, max(0, eobPt - 2)):
                    eobShift = max(0, eobPt - 2) - 1 - i
                    eob_extra_bit = read_L(av1, 1)
                    if eob_extra_bit:
                        eob += (1 << eobShift)

            for c in range(eob - 1, -1, -1):
                pos = scan[c]
                if c == (eob - 1):
                    coeff_base_eob = read_S(av1, 'coeff_base_eob', plane=plane, txSz=txSz,
                                            txSzCtx=txSzCtx, ptype=ptype, x4=x4, y4=y4, scan=scan, c=c)
                    level = coeff_base_eob + 1
                else:
                    coeff_base = read_S(av1, 'coeff_base', plane=plane, txSz=txSz,
                                        txSzCtx=txSzCtx, ptype=ptype, x4=x4, y4=y4, scan=scan, c=c)
                    level = coeff_base

                if level > NUM_BASE_LEVELS:
                    for idx in range(COEFF_BASE_RANGE // (BR_CDF_SIZE - 1)):
                        coeff_br = read_S(av1, 'coeff_br', plane=plane, txSz=txSz,
                                          txSzCtx=txSzCtx, ptype=ptype, x4=x4, y4=y4, pos=pos)
                        level += coeff_br
                        if coeff_br < (BR_CDF_SIZE - 1):
                            break

                tile_group.Quant[pos] = level

            for c in range(eob):
                pos = scan[c]
                if tile_group.Quant[pos] != 0:
                    if c == 0:
                        dc_sign = read_S(
                            av1, 'dc_sign', plane=plane, ptype=ptype, w4=w4, h4=h4, x4=x4, y4=y4)
                        sign = dc_sign
                    else:
                        sign_bit = read_L(av1, 1)
                        sign = sign_bit
                else:
                    sign = 0

                if tile_group.Quant[pos] > (NUM_BASE_LEVELS + COEFF_BASE_RANGE):
                    length = 0
                    while True:
                        length += 1
                        golomb_length_bit = read_L(av1, 1)
                        if golomb_length_bit:
                            break

                    # If length is equal to 20, it is a requirement of bitstream conformance that golomb_length_bit is equal to 1.
                    if length == 20:
                        assert golomb_length_bit == 1

                    x = 1
                    for i in range(length - 2, -1, -1):
                        golomb_data_bit = read_L(av1, 1)
                        x = (x << 1) | golomb_data_bit

                    tile_group.Quant[pos] = (x +
                                             COEFF_BASE_RANGE + NUM_BASE_LEVELS)

                if pos == 0 and tile_group.Quant[pos] > 0:
                    dcCategory = 1 if sign else 2

                tile_group.Quant[pos] = tile_group.Quant[pos] & 0xFFFFF
                culLevel += tile_group.Quant[pos]

                if sign:
                    tile_group.Quant[pos] = -tile_group.Quant[pos]

            culLevel = min(63, culLevel)

        for i in range(w4):
            tile_group.AboveLevelContext[plane][x4 + i] = culLevel
            tile_group.AboveDcContext[plane][x4 + i] = dcCategory

        for i in range(h4):
            tile_group.LeftLevelContext[plane][y4 + i] = culLevel
            tile_group.LeftDcContext[plane][y4 + i] = dcCategory

        return eob

    def __get_scan(self, av1: AV1Decoder, txSz: TX_SIZE) -> list:
        """
        获取扫描顺序
        规范文档 5.11.41 Get scan function

        Args:
            txSz: 变换尺寸

        Returns:
            扫描数组
        """
        tile_group = av1.tile_group
        PlaneTxType = tile_group.PlaneTxType

        def get_mrow_scan(txSz: int) -> list:
            """
            获取行扫描顺序
            规范文档 5.11.41

            Args:
                txSz: 变换尺寸

            Returns:
                扫描数组
            """
            if txSz == TX_SIZE.TX_4X4:
                return Mrow_Scan_4x4
            elif txSz == TX_SIZE.TX_4X8:
                return Mrow_Scan_4x8
            elif txSz == TX_SIZE.TX_8X4:
                return Mrow_Scan_8x4
            elif txSz == TX_SIZE.TX_8X8:
                return Mrow_Scan_8x8
            elif txSz == TX_SIZE.TX_8X16:
                return Mrow_Scan_8x16
            elif txSz == TX_SIZE.TX_16X8:
                return Mrow_Scan_16x8
            elif txSz == TX_SIZE.TX_16X16:
                return Mrow_Scan_16x16
            elif txSz == TX_SIZE.TX_4X16:
                return Mrow_Scan_4x16
            return Mrow_Scan_16x4

        def get_mcol_scan(txSz: int) -> list:
            """
            获取列扫描顺序
            规范文档 5.11.41

            Args:
                txSz: 变换尺寸

            Returns:
                扫描数组
            """
            if txSz == TX_SIZE.TX_4X4:
                return Mcol_Scan_4x4
            elif txSz == TX_SIZE.TX_4X8:
                return Mcol_Scan_4x8
            elif txSz == TX_SIZE.TX_8X4:
                return Mcol_Scan_8x4
            elif txSz == TX_SIZE.TX_8X8:
                return Mcol_Scan_8x8
            elif txSz == TX_SIZE.TX_8X16:
                return Mcol_Scan_8x16
            elif txSz == TX_SIZE.TX_16X8:
                return Mcol_Scan_16x8
            elif txSz == TX_SIZE.TX_16X16:
                return Mcol_Scan_16x16
            elif txSz == TX_SIZE.TX_4X16:
                return Mcol_Scan_4x16
            return Mcol_Scan_16x4

        def get_default_scan(txSz: int) -> list:
            """
            获取默认扫描顺序（Zig-Zag扫描）
            规范文档 5.11.41

            Args:
                txSz: 变换尺寸

            Returns:
                扫描数组
            """
            if txSz == TX_SIZE.TX_4X4:
                return Default_Scan_4x4
            elif txSz == TX_SIZE.TX_4X8:
                return Default_Scan_4x8
            elif txSz == TX_SIZE.TX_8X4:
                return Default_Scan_8x4
            elif txSz == TX_SIZE.TX_8X8:
                return Default_Scan_8x8
            elif txSz == TX_SIZE.TX_8X16:
                return Default_Scan_8x16
            elif txSz == TX_SIZE.TX_16X8:
                return Default_Scan_16x8
            elif txSz == TX_SIZE.TX_16X16:
                return Default_Scan_16x16
            elif txSz == TX_SIZE.TX_16X32:
                return Default_Scan_16x32
            elif txSz == TX_SIZE.TX_32X16:
                return Default_Scan_32x16
            elif txSz == TX_SIZE.TX_4X16:
                return Default_Scan_4x16
            elif txSz == TX_SIZE.TX_16X4:
                return Default_Scan_16x4
            elif txSz == TX_SIZE.TX_8X32:
                return Default_Scan_8x32
            elif txSz == TX_SIZE.TX_32X8:
                return Default_Scan_32x8
            return Default_Scan_32x32

        if txSz == TX_SIZE.TX_16X64:
            return Default_Scan_16x32

        if txSz == TX_SIZE.TX_64X16:
            return Default_Scan_32x16

        if Tx_Size_Sqr_Up[txSz] == TX_SIZE.TX_64X64:
            return Default_Scan_32x32

        if PlaneTxType == IDTX:
            return get_default_scan(txSz)

        preferRow = PlaneTxType in [V_DCT, V_ADST, V_FLIPADST]
        preferCol = PlaneTxType in [H_DCT, H_ADST, H_FLIPADST]

        if preferRow:
            return get_mrow_scan(txSz)
        elif preferCol:
            return get_mcol_scan(txSz)
        return get_default_scan(txSz)

    def __intra_angle_info_y(self, av1: AV1Decoder):
        """
        读取Y角度信息
        规范文档 5.11.42 Intra angle info luma syntax
        """
        tile_group = self.tile_group
        MiSize = tile_group.MiSize
        YMode = tile_group.YMode

        tile_group.AngleDeltaY = 0
        if MiSize >= SUB_SIZE.BLOCK_8X8:
            if is_directional_mode(YMode):
                angle_delta_y = read_S(av1, 'angle_delta_y')
                tile_group.AngleDeltaY = angle_delta_y - MAX_ANGLE_DELTA

    def __intra_angle_info_uv(self, av1: AV1Decoder):
        """
        读取UV角度信息
        规范文档 5.11.43 Intra angle info chroma syntax
        """
        tile_group = self.tile_group
        MiSize = tile_group.MiSize

        tile_group.AngleDeltaUV = 0
        if MiSize >= SUB_SIZE.BLOCK_8X8:
            if is_directional_mode(tile_group.UVMode):
                angle_delta_uv = read_S(av1, 'angle_delta_uv')
                tile_group.AngleDeltaUV = angle_delta_uv - MAX_ANGLE_DELTA

    def __read_cfl_alphas(self, av1: AV1Decoder):
        """
        读取CFL alpha值
        规范文档 5.11.45 Read CFL alphas syntax
        """
        tile_group = self.tile_group

        cfl_alpha_signs = read_S(av1, 'cfl_alpha_signs')
        signU = (cfl_alpha_signs + 1) // 3
        signV = (cfl_alpha_signs + 1) % 3
        if signU != CFL_SIGN.CFL_SIGN_ZERO:
            cfl_alpha_u = read_S(av1, 'cfl_alpha_u', signU=signU, signV=signV)
            tile_group.CflAlphaU = 1 + cfl_alpha_u
            if signU == CFL_SIGN.CFL_SIGN_NEG:
                tile_group.CflAlphaU = -tile_group.CflAlphaU
        else:
            tile_group.CflAlphaU = 0
        if signV != CFL_SIGN.CFL_SIGN_ZERO:
            cfl_alpha_v = read_S(av1, 'cfl_alpha_v', signU=signU, signV=signV)
            tile_group.CflAlphaV = 1 + cfl_alpha_v
            if signV == CFL_SIGN.CFL_SIGN_NEG:
                tile_group.CflAlphaV = -tile_group.CflAlphaV
        else:
            tile_group.CflAlphaV = 0

    def __palette_mode_info(self, av1: AV1Decoder):
        """
        Palette模式信息解析
        规范文档 5.11.46 Palette mode info syntax
        """
        seq_header = av1.seq_header
        tile_group = self.tile_group
        BitDepth = seq_header.color_config.BitDepth
        MiSize = tile_group.MiSize
        HasChroma = tile_group.HasChroma
        YMode = tile_group.YMode

        PaletteCache: List[int] = [NONE] * PALETTE_COLORS

        def sort(arr: List[int], i1: int, i2: int):
            """
            sort函数
            规范文档中定义的sort函数，对数组的子数组进行排序（升序）

            Args:
                arr: 要排序的数组
                i1: 起始索引
                i2: 结束索引（包含）
            """
            for index, value in enumerate(sorted(arr[i1:i2+1])):
                arr[index + i1] = value

        def get_palette_cache(plane: int) -> int:
            """
            get_palette_cache函数
            规范文档 5.11.46 get_palette_cache()

            Args:
                plane: 平面索引（0=Y, 1=UV）

            Returns:
                (PaletteCache, cacheN) - 调色板缓存数组和缓存数量
            """
            MiRow = tile_group.MiRow
            MiCol = tile_group.MiCol
            AvailL = tile_group.AvailL

            aboveN = 0
            if (MiRow * MI_SIZE) % 64:
                aboveN = tile_group.PaletteSizes[plane][MiRow - 1][MiCol]

            leftN = 0
            if AvailL:
                leftN = tile_group.PaletteSizes[plane][MiRow][MiCol - 1]

            aboveIdx = 0
            leftIdx = 0
            n = 0
            while aboveIdx < aboveN and leftIdx < leftN:
                aboveC = tile_group.PaletteColors[plane][MiRow -
                                                         1][MiCol][aboveIdx]
                leftC = tile_group.PaletteColors[plane][MiRow][MiCol - 1][leftIdx]
                if leftC < aboveC:
                    if n == 0 or leftC != PaletteCache[n - 1]:
                        PaletteCache[n] = leftC
                        n += 1
                    leftIdx += 1
                else:
                    if n == 0 or aboveC != PaletteCache[n - 1]:
                        PaletteCache[n] = aboveC
                        n += 1
                    aboveIdx += 1
                    if leftC == aboveC:
                        leftIdx += 1

            while aboveIdx < aboveN:
                val = tile_group.PaletteColors[plane][MiRow -
                                                      1][MiCol][aboveIdx]
                aboveIdx += 1
                if n == 0 or val != PaletteCache[n - 1]:
                    PaletteCache[n] = val
                    n += 1

            while leftIdx < leftN:
                val = tile_group.PaletteColors[plane][MiRow][MiCol - 1][leftIdx]
                leftIdx += 1
                if n == 0 or val != PaletteCache[n - 1]:
                    PaletteCache[n] = val
                    n += 1

            return n

        bsizeCtx = Mi_Width_Log2[MiSize] + Mi_Height_Log2[MiSize] - 2
        if YMode == Y_MODE.DC_PRED:
            has_palette_y = read_S(av1, 'has_palette_y', bsizeCtx=bsizeCtx)
            if has_palette_y:
                palette_size_y_minus_2 = read_S(
                    av1, 'palette_size_y_minus_2', bsizeCtx=bsizeCtx)
                tile_group.PaletteSizeY = palette_size_y_minus_2 + 2
                cacheN = get_palette_cache(0)
                idx = 0
                for i in range(cacheN):
                    if idx >= tile_group.PaletteSizeY:
                        break
                    use_palette_color_cache_y = read_L(av1, 1)
                    if use_palette_color_cache_y:
                        tile_group.palette_colors_y[idx] = PaletteCache[i]
                        idx += 1
                if idx < tile_group.PaletteSizeY:
                    tile_group.palette_colors_y[idx] = read_L(av1, BitDepth)
                    idx += 1
                paletteBits = 0
                if idx < tile_group.PaletteSizeY:
                    minBits = BitDepth - 3
                    palette_num_extra_bits_y = read_L(av1, 2)
                    paletteBits = minBits + palette_num_extra_bits_y

                while idx < tile_group.PaletteSizeY:
                    palette_delta_y = read_L(av1, paletteBits)
                    palette_delta_y += 1
                    tile_group.palette_colors_y[idx] = Clip1(
                        tile_group.palette_colors_y[idx - 1] + palette_delta_y, BitDepth)

                    range_val = ((1 << BitDepth) -
                                 tile_group.palette_colors_y[idx] - 1)
                    paletteBits = min(paletteBits, CeilLog2(range_val))
                    idx += 1

                sort(tile_group.palette_colors_y,
                     0, tile_group.PaletteSizeY - 1)

        if HasChroma and tile_group.UVMode == Y_MODE.DC_PRED:
            has_palette_uv = read_S(av1, 'has_palette_uv')
            if has_palette_uv:
                palette_size_uv_minus_2 = read_S(
                    av1, 'palette_size_uv_minus_2', bsizeCtx=bsizeCtx)
                tile_group.PaletteSizeUV = palette_size_uv_minus_2 + 2
                cacheN = get_palette_cache(1)
                idx = 0
                for i in range(cacheN):
                    if idx >= tile_group.PaletteSizeUV:
                        break
                    use_palette_color_cache_u = read_L(av1, 1)
                    if use_palette_color_cache_u:
                        tile_group.palette_colors_u[idx] = PaletteCache[i]
                        idx += 1

                if idx < tile_group.PaletteSizeUV:
                    tile_group.palette_colors_u[idx] = read_L(av1, BitDepth)
                    idx += 1

                paletteBits = 0
                if idx < tile_group.PaletteSizeUV:
                    minBits = BitDepth - 3
                    palette_num_extra_bits_u = read_L(av1, 2)
                    paletteBits = minBits + palette_num_extra_bits_u

                while idx < tile_group.PaletteSizeUV:
                    palette_delta_u = read_L(av1, paletteBits)
                    tile_group.palette_colors_u[idx] = Clip1(
                        tile_group.palette_colors_u[idx - 1] + palette_delta_u, BitDepth)
                    range_val = ((1 << BitDepth) -
                                 tile_group.palette_colors_u[idx])
                    paletteBits = min(paletteBits, CeilLog2(range_val))
                    idx += 1

                sort(tile_group.palette_colors_u, 0,
                     tile_group.PaletteSizeUV - 1)

                delta_encode_palette_colors_v = read_L(av1, 1)
                if delta_encode_palette_colors_v:
                    minBits = BitDepth - 4
                    maxVal = 1 << BitDepth
                    palette_num_extra_bits_v = read_L(av1, 2)
                    paletteBits = minBits + palette_num_extra_bits_v

                    tile_group.palette_colors_v[0] = read_L(av1, BitDepth)
                    for idx in range(1, tile_group.PaletteSizeUV):
                        palette_delta_v = read_L(av1, paletteBits)
                        if palette_delta_v:
                            palette_delta_sign_bit_v = read_L(av1, 1)
                            if palette_delta_sign_bit_v:
                                palette_delta_v = -palette_delta_v
                        val = tile_group.palette_colors_v[idx -
                                                          1] + palette_delta_v
                        if val < 0:
                            val += maxVal
                        if val >= maxVal:
                            val -= maxVal

                        tile_group.palette_colors_v[idx] = Clip1(val, BitDepth)
                else:
                    for idx in range(tile_group.PaletteSizeUV):
                        tile_group.palette_colors_v[idx] = read_L(
                            av1, BitDepth)

    def __transform_type(self, av1: AV1Decoder,
                         x4: int, y4: int, txSz: TX_SIZE):
        """
        解析变换类型
        规范文档 5.11.47 Transform type syntax

        Args:
            x4: X坐标（4x4块单位）
            y4: Y坐标（4x4块单位）
            txSz: 变换尺寸

        Returns:
            变换类型（TxType）
        """
        from utils.frame_utils import get_qindex
        frame_header = av1.frame_header
        tile_group = self.tile_group

        TxType = DCT_DCT
        set_val = get_tx_set(av1, txSz)
        if set_val > 0 and (get_qindex(av1, 1, tile_group.segment_id) if frame_header.segmentation_enabled else frame_header.base_q_idx) > 0:
            if tile_group.is_inter:
                inter_tx_type = read_S(
                    av1, 'inter_tx_type', set_val=set_val, txSz=txSz)

                if set_val == TX_SET.TX_SET_INTER_1:
                    TxType = Tx_Type_Inter_Inv_Set1[inter_tx_type]
                elif set_val == TX_SET.TX_SET_INTER_2:
                    TxType = Tx_Type_Inter_Inv_Set2[inter_tx_type]
                else:
                    TxType = Tx_Type_Inter_Inv_Set3[inter_tx_type]
            else:
                intra_tx_type = read_S(
                    av1, 'intra_tx_type', set_val=set_val, txSz=txSz)
                if set_val == TX_SET.TX_SET_INTRA_1:
                    TxType = Tx_Type_Intra_Inv_Set1[intra_tx_type]
                else:
                    TxType = Tx_Type_Intra_Inv_Set2[intra_tx_type]
        else:
            TxType = DCT_DCT
        for i in range(Tx_Width[txSz] >> 2):
            for j in range(Tx_Height[txSz] >> 2):
                tile_group.TxTypes[y4 + j][x4 + i] = TxType

    def __palette_tokens(self, av1: AV1Decoder):
        """
        Palette tokens解析
        规范文档 5.11.49 Palette tokens syntax
        """
        reader = av1.reader
        seq_header = av1.seq_header
        frame_header = av1.frame_header
        tile_group = self.tile_group
        subsampling_x = seq_header.color_config.subsampling_x
        subsampling_y = seq_header.color_config.subsampling_y
        MiRows = frame_header.MiRows
        MiCols = frame_header.MiCols
        MiRow = tile_group.MiRow
        MiCol = tile_group.MiCol
        MiSize = tile_group.MiSize

        blockHeight = Block_Height[MiSize]
        blockWidth = Block_Width[MiSize]
        onscreenHeight = min(blockHeight, (MiRows - MiRow) * MI_SIZE)
        onscreenWidth = min(blockWidth, (MiCols - MiCol) * MI_SIZE)

        if tile_group.PaletteSizeY:
            color_index_map_y = read_NS(av1, tile_group.PaletteSizeY)
            tile_group.ColorMapY[0][0] = color_index_map_y
            for i in range(1, onscreenHeight + onscreenWidth - 1):
                for j in range(min(i, onscreenWidth - 1), max(0, i - onscreenHeight + 1) - 1, -1):
                    self.__get_palette_color_context(
                        av1, tile_group.ColorMapY, i - j, j, tile_group.PaletteSizeY)
                    palette_color_idx_y = read_S(av1, 'palette_color_idx_y')
                    tile_group.ColorMapY[i -
                                         j][j] = tile_group.ColorOrder[palette_color_idx_y]

            for i in range(onscreenHeight):
                for j in range(onscreenWidth, blockWidth):
                    tile_group.ColorMapY[i][j] = tile_group.ColorMapY[i][onscreenWidth - 1]

            for i in range(onscreenHeight, blockHeight):
                for j in range(blockWidth):
                    tile_group.ColorMapY[i][j] = tile_group.ColorMapY[onscreenHeight - 1][j]

        if tile_group.PaletteSizeUV:
            color_index_map_uv = read_NS(av1, tile_group.PaletteSizeUV)
            tile_group.ColorMapUV[0][0] = color_index_map_uv
            blockHeight = blockHeight >> subsampling_y
            blockWidth = blockWidth >> subsampling_x
            onscreenHeight = onscreenHeight >> subsampling_y
            onscreenWidth = onscreenWidth >> subsampling_x
            if blockWidth < 4:
                blockWidth += 2
                onscreenWidth += 2
            if blockHeight < 4:
                blockHeight += 2
                onscreenHeight += 2

            for i in range(1, onscreenHeight + onscreenWidth - 1):
                for j in range(min(i, onscreenWidth - 1), max(0, i - onscreenHeight + 1) - 1, -1):
                    self.__get_palette_color_context(
                        av1, tile_group.ColorMapUV, i - j, j, tile_group.PaletteSizeUV)
                    palette_color_idx_uv = read_S(av1, 'palette_color_idx_uv')
                    tile_group.ColorMapUV[i -
                                          j][j] = tile_group.ColorOrder[palette_color_idx_uv]

            for i in range(onscreenHeight):
                for j in range(onscreenWidth, blockWidth):
                    tile_group.ColorMapUV[i][j] = tile_group.ColorMapUV[i][onscreenWidth - 1]

            for i in range(onscreenHeight, blockHeight):
                for j in range(blockWidth):
                    tile_group.ColorMapUV[i][j] = tile_group.ColorMapUV[onscreenHeight - 1][j]

    def __get_palette_color_context(self, av1: AV1Decoder, colorMap: List[List[int]], r: int, c: int, n: int):
        """
        规范文档 5.11.50 Palette color context function

        Args:
            colorMap: 颜色映射数组
            r: 行位置
            c: 列位置
            n: 调色板大小

        Returns:
            ColorContextHash - 颜色上下文哈希值
        """
        tile_group = self.tile_group

        scores = [0] * PALETTE_COLORS
        tile_group.ColorOrder = [i for i in range(PALETTE_COLORS)]

        if c > 0:
            neighbor = colorMap[r][c - 1]
            scores[neighbor] += 2

        if r > 0 and c > 0:
            neighbor = colorMap[r - 1][c - 1]
            scores[neighbor] += 1

        if r > 0:
            neighbor = colorMap[r - 1][c]
            scores[neighbor] += 2

        for i in range(PALETTE_NUM_NEIGHBORS):
            maxScore = scores[i]
            maxIdx = i

            for j in range(i + 1, n):
                if scores[j] > maxScore:
                    maxScore = scores[j]
                    maxIdx = j

            if maxIdx != i:
                maxScore = scores[maxIdx]
                maxColorOrder = tile_group.ColorOrder[maxIdx]

                for k in range(maxIdx, i, -1):
                    scores[k] = scores[k - 1]
                    tile_group.ColorOrder[k] = tile_group.ColorOrder[k - 1]

                scores[i] = maxScore
                tile_group.ColorOrder[i] = maxColorOrder

        tile_group.ColorContextHash = 0
        for i in range(PALETTE_NUM_NEIGHBORS):
            tile_group.ColorContextHash += (scores[i] *
                                            Palette_Color_Hash_Multipliers[i])

    def __clear_cdef(self, av1: AV1Decoder, r: int, c: int):
        """
        清除CDEF上下文
        规范文档 5.11.55 Clear CDEF function

        Args:
            r: 行索引（MI单位）
            c: 列索引（MI单位）
        """
        seq_header = av1.seq_header
        frame_header = av1.frame_header
        tile_group = self.tile_group
        use_128x128_superblock = seq_header.use_128x128_superblock

        tile_group.cdef_idx[r][c] = -1
        if use_128x128_superblock:
            cdefSize4 = Num_4x4_Blocks_Wide[SUB_SIZE.BLOCK_64X64]
            tile_group.cdef_idx[r][c + cdefSize4] = -1
            tile_group.cdef_idx[r + cdefSize4][c] = -1
            tile_group.cdef_idx[r + cdefSize4][c + cdefSize4] = -1

    def __read_cdef(self, av1: AV1Decoder):
        """
        读取CDEF参数
        规范文档 5.11.56 Read CDEF syntax
        """
        seq_header = av1.seq_header
        frame_header = av1.frame_header
        tile_group = self.tile_group
        MiRow = tile_group.MiRow
        MiCol = tile_group.MiCol
        MiSize = tile_group.MiSize

        if tile_group.skip or frame_header.CodedLossless or not seq_header.enable_cdef or frame_header.allow_intrabc:
            return

        cdefSize4 = Num_4x4_Blocks_Wide[SUB_SIZE.BLOCK_64X64]
        cdefMask4 = ~(cdefSize4 - 1)
        r = MiRow & cdefMask4
        c = MiCol & cdefMask4

        if tile_group.cdef_idx[r][c] == -1:
            tile_group.cdef_idx[r][c] = read_L(av1,  frame_header.cdef_bits)
            w4 = Num_4x4_Blocks_Wide[MiSize]
            h4 = Num_4x4_Blocks_High[MiSize]
            for i in range(r, r + h4, cdefSize4):
                for j in range(c, c + w4, cdefSize4):
                    tile_group.cdef_idx[i][j] = tile_group.cdef_idx[r][c]

    def __read_lr(self, av1: AV1Decoder, r: int, c: int, sbSize: int):
        """
        读取Loop Restoration参数
        规范文档 5.11.57 Read loop restoration syntax

        Args:
            r: 行索引（MI单位）
            c: 列索引（MI单位）
            sbSize: Superblock尺寸
        """
        seq_header = av1.seq_header
        frame_header = av1.frame_header
        subsampling_x = seq_header.color_config.subsampling_x
        subsampling_y = seq_header.color_config.subsampling_y
        NumPlanes = seq_header.color_config.NumPlanes

        if frame_header.allow_intrabc:
            return

        w = Num_4x4_Blocks_Wide[sbSize]
        h = Num_4x4_Blocks_High[sbSize]

        for plane in range(NumPlanes):
            if frame_header.FrameRestorationType[plane] != FRAME_RESTORATION_TYPE.RESTORE_NONE:
                subX = 0 if plane == 0 else subsampling_x
                subY = 0 if plane == 0 else subsampling_y
                unitSize = frame_header.LoopRestorationSize[plane]
                unitRows = count_units_in_frame(
                    unitSize, Round2(frame_header.FrameHeight, subY))
                unitCols = count_units_in_frame(
                    unitSize, Round2(frame_header.UpscaledWidth, subX))
                unitRowStart = (r * (MI_SIZE >> subY) +
                                unitSize - 1) // unitSize
                unitRowEnd = min(
                    unitRows, ((r + h) * (MI_SIZE >> subY) + unitSize - 1) // unitSize)
                if frame_header.use_superres:
                    numerator = (MI_SIZE >> subX) * frame_header.SuperresDenom
                    denominator = unitSize * SUPERRES_NUM
                else:
                    numerator = MI_SIZE >> subX
                    denominator = unitSize

                unitColStart = (c * numerator + denominator - 1) // denominator
                unitColEnd = min(
                    unitCols, ((c + w) * numerator + denominator - 1) // denominator)
                for unitRow in range(unitRowStart, unitRowEnd):
                    for unitCol in range(unitColStart, unitColEnd):
                        self.__read_lr_unit(av1, plane, unitRow, unitCol)

    def __read_lr_unit(self, av1: AV1Decoder, plane: int, unitRow: int, unitCol: int):
        """
        读取Loop Restoration单元参数
        规范文档 5.11.58 Read loop restoration unit syntax

        Args:
            plane: 平面索引
            unitRow: 单元行索引
            unitCol: 单元列索引
        """
        frame_header = av1.frame_header
        tile_group = self.tile_group

        def decode_signed_subexp_with_ref_bool(low: int, high: int, k: int, r: int) -> int:
            """
            解码有符号子表达式并检查参考值
            规范文档 5.11.58

            Args:
                low: 最小值
                high: 最大值（不包含）
                k: 参数k
                r: 参考值

            Returns:
                解码的值
            """
            x = decode_unsigned_subexp_with_ref_bool(high - low, k, r - low)
            return x + low

        def decode_unsigned_subexp_with_ref_bool(mx: int, k: int, r: int) -> int:
            """
            解码无符号子表达式并检查参考值
            规范文档 5.11.58

            Args:
                mx: 最大值
                k: 参数k
                r: 参考值

            Returns:
                解码的值
            """
            v = decode_subexp_bool(mx, k)
            if (r << 1) <= mx:
                return inverse_recenter(r, v)
            else:
                return mx - 1 - inverse_recenter(mx - 1 - r, v)

        def decode_subexp_bool(numSyms: int, k: int) -> int:
            """
            解码子表达式并检查参考值
            规范文档 5.11.58

            Args:
                numSyms: 符号数量
                k: 参数k

            Returns:
                解码的值
            """
            i = 0
            mk = 0
            while True:
                b2 = (k + i - 1) if i else k
                a = 1 << b2
                if numSyms <= mk + 3 * a:
                    subexp_unif_bools = read_NS(av1, numSyms - mk)
                    return subexp_unif_bools + mk
                else:
                    subexp_more_bools = read_L(av1, 1)
                    if subexp_more_bools:
                        i += 1
                        mk += a
                    else:
                        subexp_bools = read_L(av1, b2)
                        return subexp_bools + mk

        if frame_header.FrameRestorationType[plane] == FRAME_RESTORATION_TYPE.RESTORE_WIENER:
            use_wiener = read_S(av1, 'use_wiener')
            restoration_type = FRAME_RESTORATION_TYPE.RESTORE_WIENER if use_wiener else FRAME_RESTORATION_TYPE.RESTORE_NONE
        elif frame_header.FrameRestorationType[plane] == FRAME_RESTORATION_TYPE.RESTORE_SGRPROJ:
            use_sgrproj = read_S(av1, 'use_sgrproj')
            restoration_type = FRAME_RESTORATION_TYPE.RESTORE_SGRPROJ if use_sgrproj else FRAME_RESTORATION_TYPE.RESTORE_NONE
        else:
            restoration_type = FRAME_RESTORATION_TYPE(
                read_S(av1, 'restoration_type'))

        tile_group.LrType[plane][unitRow][unitCol] = restoration_type
        if restoration_type == FRAME_RESTORATION_TYPE.RESTORE_WIENER:
            for pass_idx in range(2):
                if plane:
                    firstCoeff = 1
                    tile_group.LrWiener[plane][unitRow][unitCol][pass_idx][0] = 0
                else:
                    firstCoeff = 0

                for j in range(firstCoeff, 3):
                    min_val = Wiener_Taps_Min[j]
                    max_val = Wiener_Taps_Max[j]
                    k = Wiener_Taps_K[j]
                    v = decode_signed_subexp_with_ref_bool(
                        min_val, max_val + 1, k, tile_group.RefLrWiener[plane][pass_idx][j])
                    tile_group.LrWiener[plane][unitRow][unitCol][pass_idx][j] = v
                    tile_group.RefLrWiener[plane][pass_idx][j] = v

        elif restoration_type == FRAME_RESTORATION_TYPE.RESTORE_SGRPROJ:
            lr_sgr_set = read_L(av1, SGRPROJ_PARAMS_BITS)
            tile_group.LrSgrSet[plane][unitRow][unitCol] = lr_sgr_set

            for i in range(2):
                radius = Sgr_Params[lr_sgr_set][i * 2]
                min_val = Sgrproj_Xqd_Min[i]
                max_val = Sgrproj_Xqd_Max[i]
                if radius:
                    v = decode_signed_subexp_with_ref_bool(
                        min_val, max_val + 1, SGRPROJ_PRJ_SUBEXP_K, tile_group.RefSgrXqd[plane][i])
                else:
                    v = 0
                    if i == 1:
                        v = Clip3(
                            min_val, max_val, (1 << SGRPROJ_PRJ_BITS) - tile_group.RefSgrXqd[plane][0])

                tile_group.LrSgrXqd[plane][unitRow][unitCol][i] = v
                tile_group.RefSgrXqd[plane][i] = v

    def _is_mv_valid(self, av1: AV1Decoder, isCompound: int) -> int:
        """
        检查运动向量是否有效
        规范文档 6.10.25 is_mv_valid()

        Args:
            isCompound: 是否为复合预测（0=单预测，1=复合预测）

        Returns:
            1表示运动向量有效，0表示无效
        """
        seq_header = av1.seq_header
        tile_group = self.tile_group
        use_128x128_superblock = seq_header.use_128x128_superblock
        subsampling_x = seq_header.color_config.subsampling_x
        subsampling_y = seq_header.color_config.subsampling_y
        MiRow = tile_group.MiRow
        MiCol = tile_group.MiCol
        MiSize = tile_group.MiSize
        HasChroma = tile_group.HasChroma

        for i in range(1 + isCompound):
            for comp in range(2):
                if abs(tile_group.Mv[i][comp]) >= (1 << 14):
                    return 0

        if not tile_group.use_intrabc:
            return 1

        bw = Block_Width[MiSize]
        bh = Block_Height[MiSize]
        if (tile_group.Mv[0][0] & 7) or (tile_group.Mv[0][1] & 7):
            return 0

        deltaRow = tile_group.Mv[0][0] >> 3
        deltaCol = tile_group.Mv[0][1] >> 3
        srcTopEdge = MiRow * MI_SIZE + deltaRow
        srcLeftEdge = MiCol * MI_SIZE + deltaCol
        srcBottomEdge = srcTopEdge + bh
        srcRightEdge = srcLeftEdge + bw
        if HasChroma:
            if bw < 8 and subsampling_x:
                srcLeftEdge -= 4
            if bh < 8 and subsampling_y:
                srcTopEdge -= 4

        if ((srcTopEdge < tile_group.MiRowStart * MI_SIZE) or
            (srcLeftEdge < tile_group.MiColStart * MI_SIZE) or
            (srcBottomEdge > tile_group.MiRowEnd * MI_SIZE) or
                (srcRightEdge > tile_group.MiColEnd * MI_SIZE)):
            return 0

        sbSize = SUB_SIZE.BLOCK_128X128 if use_128x128_superblock else SUB_SIZE.BLOCK_64X64
        sbH = Block_Height[sbSize]
        activeSbRow = (MiRow * MI_SIZE) // sbH
        activeSb64Col = (MiCol * MI_SIZE) >> 6
        srcSbRow = (srcBottomEdge - 1) // sbH
        srcSb64Col = (srcRightEdge - 1) >> 6
        totalSb64PerRow = (
            (tile_group.MiColEnd - tile_group.MiColStart - 1) >> 4) + 1
        activeSb64 = activeSbRow * totalSb64PerRow + activeSb64Col
        srcSb64 = srcSbRow * totalSb64PerRow + srcSb64Col
        if srcSb64 >= activeSb64 - INTRABC_DELAY_SB64:
            return 0

        gradient = 1 + INTRABC_DELAY_SB64 + use_128x128_superblock
        wfOffset = gradient * (activeSbRow - srcSbRow)
        if ((srcSbRow > activeSbRow) or
                (srcSb64Col >= activeSb64Col - INTRABC_DELAY_SB64 + wfOffset)):
            return 0

        return 1


def clear_left_context(av1: AV1Decoder):
    """
    清除左方上下文
    规范文档 6.10.2 clear_left_context()
    """
    frame_header = av1.frame_header
    tile_group = av1.tile_group
    MiRows = frame_header.MiRows

    tile_group.LeftLevelContext = Array(None, (PLANE_MAX, MiRows + 32), 0)
    tile_group.LeftDcContext = Array(None, (PLANE_MAX, MiRows + 32), 0)
    tile_group.LeftSegPredContext = Array(None, MiRows + 32, 0)


def clear_above_context(av1: AV1Decoder):
    """
    清除上方上下文
    规范文档 6.10.2 clear_above_context()
    """
    frame_header = av1.frame_header
    tile_group = av1.tile_group
    MiCols = frame_header.MiCols

    tile_group.AboveLevelContext = Array(None, (PLANE_MAX, MiCols + 32), 0)
    tile_group.AboveDcContext = Array(None, (PLANE_MAX, MiCols + 32), 0)
    tile_group.AboveSegPredContext = Array(None, MiCols + 32, 0)


def tile_group_obu(av1: AV1Decoder, sz: int):
    """
    规范文档 5.11 Tile group OBU syntax
    """
    tile_group_parser = TileGroupParser(av1)
    if av1.tile_group is not None:
        tile_group_parser.tile_group = av1.tile_group
    else:
        av1.tile_group = tile_group_parser.tile_group

    if av1.decoder is None:
        from entropy.symbol_decoder import SymbolDecoder
        av1.decoder = SymbolDecoder()
    tile_group_parser.tile_group_obu(av1, sz)


def is_transpose(type: str, w: int, h: int) -> int:
    """
    检查是否为转置
    规范文档 9.2 is_transpose()
    """
    from entropy import default_cdfs
    T_wxh = getattr(default_cdfs, type + "_Scan_" + str(w) + "x" + str(h))
    T_hxw = getattr(default_cdfs, type + "_Scan_" + str(h) + "x" + str(w))

    for pos in range(len(T_wxh)):
        x1 = T_wxh[pos] % w
        y1 = T_wxh[pos] // w
        x2 = T_hxw[pos] % h
        y2 = T_hxw[pos] // h
        if x1 != y2 or y1 != x2:
            return 0
    return 1


"""
规范文档 5.11.2 Decode tile syntax
"""
Wiener_Taps_Mid = [3, -7, 15]
Sgrproj_Xqd_Mid = [-32, 31]

"""
规范文档 5.11.15 TX size syntax
"""
Max_Tx_Depth = [0, 1, 1, 1, 2, 2, 2, 3, 3,
                3, 4, 4, 4, 4, 4, 4, 2, 2, 3, 3, 4, 4]


"""
规范文档 5.11.47 Transform type syntax
"""
Tx_Type_Intra_Inv_Set1 = [IDTX, DCT_DCT, V_DCT,
                          H_DCT, ADST_ADST, ADST_DCT, DCT_ADST]
Tx_Type_Intra_Inv_Set2 = [IDTX, DCT_DCT, ADST_ADST, ADST_DCT, DCT_ADST]
Tx_Type_Inter_Inv_Set1 = [
    IDTX,
    V_DCT,
    H_DCT,
    V_ADST,
    H_ADST,
    V_FLIPADST,
    H_FLIPADST,
    DCT_DCT,
    ADST_DCT,
    DCT_ADST,
    FLIPADST_DCT,
    DCT_FLIPADST,
    ADST_ADST,
    FLIPADST_FLIPADST,
    ADST_FLIPADST,
    FLIPADST_ADST,
]
Tx_Type_Inter_Inv_Set2 = [IDTX, V_DCT, H_DCT, DCT_DCT, ADST_DCT, DCT_ADST, FLIPADST_DCT,
                          DCT_FLIPADST, ADST_ADST, FLIPADST_FLIPADST, ADST_FLIPADST, FLIPADST_ADST]
Tx_Type_Inter_Inv_Set3 = [IDTX, DCT_DCT]


"""
规范文档 5.11.58 Read loop restoration unit syntax
"""
Wiener_Taps_Min = [-5, -23, -17]
Wiener_Taps_Max = [10, 8, 46]
Wiener_Taps_K = [1, 2, 3]
Sgrproj_Xqd_Min = [-96, -32]
Sgrproj_Xqd_Max = [31, 95]


"""
规范文档 9.2
"""
Default_Scan_4x4 = [0, 1, 4, 8, 5, 2, 3, 6, 9, 12, 13, 10, 7, 11, 14, 15]

Mcol_Scan_4x4 = [0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15]

Mrow_Scan_4x4 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]

Default_Scan_4x8 = [0, 1, 4, 2, 5, 8, 3, 6, 9, 12, 7, 10, 13, 16, 11,
                    14, 17, 20, 15, 18, 21, 24, 19, 22, 25, 28, 23, 26, 29, 27, 30, 31]

Mcol_Scan_4x8 = [0, 4, 8, 12, 16, 20, 24, 28, 1, 5, 9, 13, 17, 21,
                 25, 29, 2, 6, 10, 14, 18, 22, 26, 30, 3, 7, 11, 15, 19, 23, 27, 31]

Mrow_Scan_4x8 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14,
                 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31]

Default_Scan_8x4 = [0, 8, 1, 16, 9, 2, 24, 17, 10, 3, 25, 18, 11, 4,
                    26, 19, 12, 5, 27, 20, 13, 6, 28, 21, 14, 7, 29, 22, 15, 30, 23, 31]

Mcol_Scan_8x4 = [0, 8, 16, 24, 1, 9, 17, 25, 2, 10, 18, 26, 3, 11,
                 19, 27, 4, 12, 20, 28, 5, 13, 21, 29, 6, 14, 22, 30, 7, 15, 23, 31]

Mrow_Scan_8x4 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14,
                 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31]

Default_Scan_8x8 = [
    0, 1, 8, 16, 9, 2, 3, 10, 17, 24, 32, 25, 18, 11, 4, 5, 12, 19, 26, 33, 40, 48, 41, 34, 27, 20, 13, 6, 7, 14, 21, 28, 35, 42, 49, 56, 57, 50, 43, 36, 29, 22, 15, 23, 30, 37, 44,
    51, 58, 59, 52, 45, 38, 31, 39, 46, 53, 60, 61, 54, 47, 55, 62, 63,
]

Mcol_Scan_8x8 = [
    0, 8, 16, 24, 32, 40, 48, 56, 1, 9, 17, 25, 33, 41, 49, 57, 2, 10, 18, 26, 34, 42, 50, 58, 3, 11, 19, 27, 35, 43, 51, 59, 4, 12, 20, 28, 36, 44, 52, 60, 5, 13, 21, 29, 37, 45,
    53, 61, 6, 14, 22, 30, 38, 46, 54, 62, 7, 15, 23, 31, 39, 47, 55, 63,
]

Mrow_Scan_8x8 = [
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46,
    47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63,
]

Default_Scan_8x16 = [
    0, 1, 8, 2, 9, 16, 3, 10, 17, 24, 4, 11, 18, 25, 32, 5, 12, 19, 26, 33, 40, 6, 13, 20, 27, 34, 41, 48, 7, 14, 21, 28, 35, 42, 49, 56, 15, 22, 29, 36, 43, 50, 57, 64, 23, 30, 37,
    44, 51, 58, 65, 72, 31, 38, 45, 52, 59, 66, 73, 80, 39, 46, 53, 60, 67, 74, 81, 88, 47, 54, 61, 68, 75, 82, 89, 96, 55, 62, 69, 76, 83, 90, 97, 104, 63, 70, 77, 84, 91, 98, 105,
    112, 71, 78, 85, 92, 99, 106, 113, 120, 79, 86, 93, 100, 107, 114, 121, 87, 94, 101, 108, 115, 122, 95, 102, 109, 116, 123, 103, 110, 117, 124, 111, 118, 125, 119, 126, 127,
]

Mcol_Scan_8x16 = [
    0, 8, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120, 1, 9, 17, 25, 33, 41, 49, 57, 65, 73, 81, 89, 97, 105, 113, 121, 2, 10, 18, 26, 34, 42, 50, 58, 66, 74, 82, 90,
    98, 106, 114, 122, 3, 11, 19, 27, 35, 43, 51, 59, 67, 75, 83, 91, 99, 107, 115, 123, 4, 12, 20, 28, 36, 44, 52, 60, 68, 76, 84, 92, 100, 108, 116, 124, 5, 13, 21, 29, 37, 45, 53,
    61, 69, 77, 85, 93, 101, 109, 117, 125, 6, 14, 22, 30, 38, 46, 54, 62, 70, 78, 86, 94, 102, 110, 118, 126, 7, 15, 23, 31, 39, 47, 55, 63, 71, 79, 87, 95, 103, 111, 119, 127,
]

Mrow_Scan_8x16 = [
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46,
    47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90,
    91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127,
]

Default_Scan_16x8 = [
    0, 16, 1, 32, 17, 2, 48, 33, 18, 3, 64, 49, 34, 19, 4, 80, 65, 50, 35, 20, 5, 96, 81, 66, 51, 36, 21, 6, 112, 97, 82, 67, 52, 37, 22, 7, 113, 98, 83, 68, 53, 38, 23, 8, 114, 99,
    84, 69, 54, 39, 24, 9, 115, 100, 85, 70, 55, 40, 25, 10, 116, 101, 86, 71, 56, 41, 26, 11, 117, 102, 87, 72, 57, 42, 27, 12, 118, 103, 88, 73, 58, 43, 28, 13, 119, 104, 89, 74,
    59, 44, 29, 14, 120, 105, 90, 75, 60, 45, 30, 15, 121, 106, 91, 76, 61, 46, 31, 122, 107, 92, 77, 62, 47, 123, 108, 93, 78, 63, 124, 109, 94, 79, 125, 110, 95, 126, 111, 127,
]

Mcol_Scan_16x8 = [
    0, 16, 32, 48, 64, 80, 96, 112, 1, 17, 33, 49, 65, 81, 97, 113, 2, 18, 34, 50, 66, 82, 98, 114, 3, 19, 35, 51, 67, 83, 99, 115, 4, 20, 36, 52, 68, 84, 100, 116, 5, 21, 37, 53,
    69, 85, 101, 117, 6, 22, 38, 54, 70, 86, 102, 118, 7, 23, 39, 55, 71, 87, 103, 119, 8, 24, 40, 56, 72, 88, 104, 120, 9, 25, 41, 57, 73, 89, 105, 121, 10, 26, 42, 58, 74, 90, 106,
    122, 11, 27, 43, 59, 75, 91, 107, 123, 12, 28, 44, 60, 76, 92, 108, 124, 13, 29, 45, 61, 77, 93, 109, 125, 14, 30, 46, 62, 78, 94, 110, 126, 15, 31, 47, 63, 79, 95, 111, 127,
]

Mrow_Scan_16x8 = [
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46,
    47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90,
    91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127,
]

Default_Scan_16x16 = [
    0, 1, 16, 32, 17, 2, 3, 18, 33, 48, 64, 49, 34, 19, 4, 5, 20, 35, 50, 65, 80, 96, 81, 66, 51, 36, 21, 6, 7, 22, 37, 52, 67, 82, 97, 112, 128, 113, 98, 83, 68, 53, 38, 23, 8, 9,
    24, 39, 54, 69, 84, 99, 114, 129, 144, 160, 145, 130, 115, 100, 85, 70, 55, 40, 25, 10, 11, 26, 41, 56, 71, 86, 101, 116, 131, 146, 161, 176, 192, 177, 162, 147, 132, 117, 102,
    87, 72, 57, 42, 27, 12, 13, 28, 43, 58, 73, 88, 103, 118, 133, 148, 163, 178, 193, 208, 224, 209, 194, 179, 164, 149, 134, 119, 104, 89, 74, 59, 44, 29, 14, 15, 30, 45, 60, 75,
    90, 105, 120, 135, 150, 165, 180, 195, 210, 225, 240, 241, 226, 211, 196, 181, 166, 151, 136, 121, 106, 91, 76, 61, 46, 31, 47, 62, 77, 92, 107, 122, 137, 152, 167, 182, 197,
    212, 227, 242, 243, 228, 213, 198, 183, 168, 153, 138, 123, 108, 93, 78, 63, 79, 94, 109, 124, 139, 154, 169, 184, 199, 214, 229, 244, 245, 230, 215, 200, 185, 170, 155, 140,
    125, 110, 95, 111, 126, 141, 156, 171, 186, 201, 216, 231, 246, 247, 232, 217, 202, 187, 172, 157, 142, 127, 143, 158, 173, 188, 203, 218, 233, 248, 249, 234, 219, 204, 189, 174,
    159, 175, 190, 205, 220, 235, 250, 251, 236, 221, 206, 191, 207, 222, 237, 252, 253, 238, 223, 239, 254, 255,
]

Mcol_Scan_16x16 = [
    0, 16, 32, 48, 64, 80, 96, 112, 128, 144, 160, 176, 192, 208, 224, 240, 1, 17, 33, 49, 65, 81, 97, 113, 129, 145, 161, 177, 193, 209, 225, 241, 2, 18, 34, 50, 66, 82, 98, 114,
    130, 146, 162, 178, 194, 210, 226, 242, 3, 19, 35, 51, 67, 83, 99, 115, 131, 147, 163, 179, 195, 211, 227, 243, 4, 20, 36, 52, 68, 84, 100, 116, 132, 148, 164, 180, 196, 212,
    228, 244, 5, 21, 37, 53, 69, 85, 101, 117, 133, 149, 165, 181, 197, 213, 229, 245, 6, 22, 38, 54, 70, 86, 102, 118, 134, 150, 166, 182, 198, 214, 230, 246, 7, 23, 39, 55, 71, 87,
    103, 119, 135, 151, 167, 183, 199, 215, 231, 247, 8, 24, 40, 56, 72, 88, 104, 120, 136, 152, 168, 184, 200, 216, 232, 248, 9, 25, 41, 57, 73, 89, 105, 121, 137, 153, 169, 185,
    201, 217, 233, 249, 10, 26, 42, 58, 74, 90, 106, 122, 138, 154, 170, 186, 202, 218, 234, 250, 11, 27, 43, 59, 75, 91, 107, 123, 139, 155, 171, 187, 203, 219, 235, 251, 12, 28,
    44, 60, 76, 92, 108, 124, 140, 156, 172, 188, 204, 220, 236, 252, 13, 29, 45, 61, 77, 93, 109, 125, 141, 157, 173, 189, 205, 221, 237, 253, 14, 30, 46, 62, 78, 94, 110, 126, 142,
    158, 174, 190, 206, 222, 238, 254, 15, 31, 47, 63, 79, 95, 111, 127, 143, 159, 175, 191, 207, 223, 239, 255,
]

Mrow_Scan_16x16 = [
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46,
    47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90,
    91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127,
    128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162,
    163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197,
    198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232,
    233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255,
]

Default_Scan_16x32 = [
    0, 1, 16, 2, 17, 32, 3, 18, 33, 48, 4, 19, 34, 49, 64, 5, 20, 35, 50, 65, 80, 6, 21, 36, 51, 66, 81, 96, 7, 22, 37, 52, 67, 82, 97, 112, 8, 23, 38, 53, 68, 83, 98, 113, 128, 9,
    24, 39, 54, 69, 84, 99, 114, 129, 144, 10, 25, 40, 55, 70, 85, 100, 115, 130, 145, 160, 11, 26, 41, 56, 71, 86, 101, 116, 131, 146, 161, 176, 12, 27, 42, 57, 72, 87, 102, 117,
    132, 147, 162, 177, 192, 13, 28, 43, 58, 73, 88, 103, 118, 133, 148, 163, 178, 193, 208, 14, 29, 44, 59, 74, 89, 104, 119, 134, 149, 164, 179, 194, 209, 224, 15, 30, 45, 60, 75,
    90, 105, 120, 135, 150, 165, 180, 195, 210, 225, 240, 31, 46, 61, 76, 91, 106, 121, 136, 151, 166, 181, 196, 211, 226, 241, 256, 47, 62, 77, 92, 107, 122, 137, 152, 167, 182,
    197, 212, 227, 242, 257, 272, 63, 78, 93, 108, 123, 138, 153, 168, 183, 198, 213, 228, 243, 258, 273, 288, 79, 94, 109, 124, 139, 154, 169, 184, 199, 214, 229, 244, 259, 274,
    289, 304, 95, 110, 125, 140, 155, 170, 185, 200, 215, 230, 245, 260, 275, 290, 305, 320, 111, 126, 141, 156, 171, 186, 201, 216, 231, 246, 261, 276, 291, 306, 321, 336, 127, 142,
    157, 172, 187, 202, 217, 232, 247, 262, 277, 292, 307, 322, 337, 352, 143, 158, 173, 188, 203, 218, 233, 248, 263, 278, 293, 308, 323, 338, 353, 368, 159, 174, 189, 204, 219,
    234, 249, 264, 279, 294, 309, 324, 339, 354, 369, 384, 175, 190, 205, 220, 235, 250, 265, 280, 295, 310, 325, 340, 355, 370, 385, 400, 191, 206, 221, 236, 251, 266, 281, 296,
    311, 326, 341, 356, 371, 386, 401, 416, 207, 222, 237, 252, 267, 282, 297, 312, 327, 342, 357, 372, 387, 402, 417, 432, 223, 238, 253, 268, 283, 298, 313, 328, 343, 358, 373,
    388, 403, 418, 433, 448, 239, 254, 269, 284, 299, 314, 329, 344, 359, 374, 389, 404, 419, 434, 449, 464, 255, 270, 285, 300, 315, 330, 345, 360, 375, 390, 405, 420, 435, 450,
    465, 480, 271, 286, 301, 316, 331, 346, 361, 376, 391, 406, 421, 436, 451, 466, 481, 496, 287, 302, 317, 332, 347, 362, 377, 392, 407, 422, 437, 452, 467, 482, 497, 303, 318,
    333, 348, 363, 378, 393, 408, 423, 438, 453, 468, 483, 498, 319, 334, 349, 364, 379, 394, 409, 424, 439, 454, 469, 484, 499, 335, 350, 365, 380, 395, 410, 425, 440, 455, 470,
    485, 500, 351, 366, 381, 396, 411, 426, 441, 456, 471, 486, 501, 367, 382, 397, 412, 427, 442, 457, 472, 487, 502, 383, 398, 413, 428, 443, 458, 473, 488, 503, 399, 414, 429,
    444, 459, 474, 489, 504, 415, 430, 445, 460, 475, 490, 505, 431, 446, 461, 476, 491, 506, 447, 462, 477, 492, 507, 463, 478, 493, 508, 479, 494, 509, 495, 510, 511,
]

Default_Scan_32x16 = [
    0, 32, 1, 64, 33, 2, 96, 65, 34, 3, 128, 97, 66, 35, 4, 160, 129, 98, 67, 36, 5, 192, 161, 130, 99, 68, 37, 6, 224, 193, 162, 131, 100, 69, 38, 7, 256, 225, 194, 163, 132, 101,
    70, 39, 8, 288, 257, 226, 195, 164, 133, 102, 71, 40, 9, 320, 289, 258, 227, 196, 165, 134, 103, 72, 41, 10, 352, 321, 290, 259, 228, 197, 166, 135, 104, 73, 42, 11, 384, 353,
    322, 291, 260, 229, 198, 167, 136, 105, 74, 43, 12, 416, 385, 354, 323, 292, 261, 230, 199, 168, 137, 106, 75, 44, 13, 448, 417, 386, 355, 324, 293, 262, 231, 200, 169, 138, 107,
    76, 45, 14, 480, 449, 418, 387, 356, 325, 294, 263, 232, 201, 170, 139, 108, 77, 46, 15, 481, 450, 419, 388, 357, 326, 295, 264, 233, 202, 171, 140, 109, 78, 47, 16, 482, 451,
    420, 389, 358, 327, 296, 265, 234, 203, 172, 141, 110, 79, 48, 17, 483, 452, 421, 390, 359, 328, 297, 266, 235, 204, 173, 142, 111, 80, 49, 18, 484, 453, 422, 391, 360, 329, 298,
    267, 236, 205, 174, 143, 112, 81, 50, 19, 485, 454, 423, 392, 361, 330, 299, 268, 237, 206, 175, 144, 113, 82, 51, 20, 486, 455, 424, 393, 362, 331, 300, 269, 238, 207, 176, 145,
    114, 83, 52, 21, 487, 456, 425, 394, 363, 332, 301, 270, 239, 208, 177, 146, 115, 84, 53, 22, 488, 457, 426, 395, 364, 333, 302, 271, 240, 209, 178, 147, 116, 85, 54, 23, 489,
    458, 427, 396, 365, 334, 303, 272, 241, 210, 179, 148, 117, 86, 55, 24, 490, 459, 428, 397, 366, 335, 304, 273, 242, 211, 180, 149, 118, 87, 56, 25, 491, 460, 429, 398, 367, 336,
    305, 274, 243, 212, 181, 150, 119, 88, 57, 26, 492, 461, 430, 399, 368, 337, 306, 275, 244, 213, 182, 151, 120, 89, 58, 27, 493, 462, 431, 400, 369, 338, 307, 276, 245, 214, 183,
    152, 121, 90, 59, 28, 494, 463, 432, 401, 370, 339, 308, 277, 246, 215, 184, 153, 122, 91, 60, 29, 495, 464, 433, 402, 371, 340, 309, 278, 247, 216, 185, 154, 123, 92, 61, 30,
    496, 465, 434, 403, 372, 341, 310, 279, 248, 217, 186, 155, 124, 93, 62, 31, 497, 466, 435, 404, 373, 342, 311, 280, 249, 218, 187, 156, 125, 94, 63, 498, 467, 436, 405, 374,
    343, 312, 281, 250, 219, 188, 157, 126, 95, 499, 468, 437, 406, 375, 344, 313, 282, 251, 220, 189, 158, 127, 500, 469, 438, 407, 376, 345, 314, 283, 252, 221, 190, 159, 501, 470,
    439, 408, 377, 346, 315, 284, 253, 222, 191, 502, 471, 440, 409, 378, 347, 316, 285, 254, 223, 503, 472, 441, 410, 379, 348, 317, 286, 255, 504, 473, 442, 411, 380, 349, 318,
    287, 505, 474, 443, 412, 381, 350, 319, 506, 475, 444, 413, 382, 351, 507, 476, 445, 414, 383, 508, 477, 446, 415, 509, 478, 447, 510, 479, 511,
]

Default_Scan_32x32 = [
    0, 1, 32, 64, 33, 2, 3, 34, 65, 96, 128, 97, 66, 35, 4, 5, 36, 67, 98, 129, 160, 192, 161, 130, 99, 68, 37, 6, 7, 38, 69, 100, 131, 162, 193, 224, 256, 225, 194, 163, 132, 101,
    70, 39, 8, 9, 40, 71, 102, 133, 164, 195, 226, 257, 288, 320, 289, 258, 227, 196, 165, 134, 103, 72, 41, 10, 11, 42, 73, 104, 135, 166, 197, 228, 259, 290, 321, 352, 384, 353,
    322, 291, 260, 229, 198, 167, 136, 105, 74, 43, 12, 13, 44, 75, 106, 137, 168, 199, 230, 261, 292, 323, 354, 385, 416, 448, 417, 386, 355, 324, 293, 262, 231, 200, 169, 138, 107,
    76, 45, 14, 15, 46, 77, 108, 139, 170, 201, 232, 263, 294, 325, 356, 387, 418, 449, 480, 512, 481, 450, 419, 388, 357, 326, 295, 264, 233, 202, 171, 140, 109, 78, 47, 16, 17, 48,
    79, 110, 141, 172, 203, 234, 265, 296, 327, 358, 389, 420, 451, 482, 513, 544, 576, 545, 514, 483, 452, 421, 390, 359, 328, 297, 266, 235, 204, 173, 142, 111, 80, 49, 18, 19, 50,
    81, 112, 143, 174, 205, 236, 267, 298, 329, 360, 391, 422, 453, 484, 515, 546, 577, 608, 640, 609, 578, 547, 516, 485, 454, 423, 392, 361, 330, 299, 268, 237, 206, 175, 144, 113,
    82, 51, 20, 21, 52, 83, 114, 145, 176, 207, 238, 269, 300, 331, 362, 393, 424, 455, 486, 517, 548, 579, 610, 641, 672, 704, 673, 642, 611, 580, 549, 518, 487, 456, 425, 394, 363,
    332, 301, 270, 239, 208, 177, 146, 115, 84, 53, 22, 23, 54, 85, 116, 147, 178, 209, 240, 271, 302, 333, 364, 395, 426, 457, 488, 519, 550, 581, 612, 643, 674, 705, 736, 768, 737,
    706, 675, 644, 613, 582, 551, 520, 489, 458, 427, 396, 365, 334, 303, 272, 241, 210, 179, 148, 117, 86, 55, 24, 25, 56, 87, 118, 149, 180, 211, 242, 273, 304, 335, 366, 397, 428,
    459, 490, 521, 552, 583, 614, 645, 676, 707, 738, 769, 800, 832, 801, 770, 739, 708, 677, 646, 615, 584, 553, 522, 491, 460, 429, 398, 367, 336, 305, 274, 243, 212, 181, 150,
    119, 88, 57, 26, 27, 58, 89, 120, 151, 182, 213, 244, 275, 306, 337, 368, 399, 430, 461, 492, 523, 554, 585, 616, 647, 678, 709, 740, 771, 802, 833, 864, 896, 865, 834, 803, 772,
    741, 710, 679, 648, 617, 586, 555, 524, 493, 462, 431, 400, 369, 338, 307, 276, 245, 214, 183, 152, 121, 90, 59, 28, 29, 60, 91, 122, 153, 184, 215, 246, 277, 308, 339, 370, 401,
    432, 463, 494, 525, 556, 587, 618, 649, 680, 711, 742, 773, 804, 835, 866, 897, 928, 960, 929, 898, 867, 836, 805, 774, 743, 712, 681, 650, 619, 588, 557, 526, 495, 464, 433,
    402, 371, 340, 309, 278, 247, 216, 185, 154, 123, 92, 61, 30, 31, 62, 93, 124, 155, 186, 217, 248, 279, 310, 341, 372, 403, 434, 465, 496, 527, 558, 589, 620, 651, 682, 713, 744,
    775, 806, 837, 868, 899, 930, 961, 992, 993, 962, 931, 900, 869, 838, 807, 776, 745, 714, 683, 652, 621, 590, 559, 528, 497, 466, 435, 404, 373, 342, 311, 280, 249, 218, 187,
    156, 125, 94, 63, 95, 126, 157, 188, 219, 250, 281, 312, 343, 374, 405, 436, 467, 498, 529, 560, 591, 622, 653, 684, 715, 746, 777, 808, 839, 870, 901, 932, 963, 994, 995, 964,
    933, 902, 871, 840, 809, 778, 747, 716, 685, 654, 623, 592, 561, 530, 499, 468, 437, 406, 375, 344, 313, 282, 251, 220, 189, 158, 127, 159, 190, 221, 252, 283, 314, 345, 376,
    407, 438, 469, 500, 531, 562, 593, 624, 655, 686, 717, 748, 779, 810, 841, 872, 903, 934, 965, 996, 997, 966, 935, 904, 873, 842, 811, 780, 749, 718, 687, 656, 625, 594, 563,
    532, 501, 470, 439, 408, 377, 346, 315, 284, 253, 222, 191, 223, 254, 285, 316, 347, 378, 409, 440, 471, 502, 533, 564, 595, 626, 657, 688, 719, 750, 781, 812, 843, 874, 905,
    936, 967, 998, 999, 968, 937, 906, 875, 844, 813, 782, 751, 720, 689, 658, 627, 596, 565, 534, 503, 472, 441, 410, 379, 348, 317, 286, 255, 287, 318, 349, 380, 411, 442, 473,
    504, 535, 566, 597, 628, 659, 690, 721, 752, 783, 814, 845, 876, 907, 938, 969, 1000, 1001, 970, 939, 908, 877, 846, 815, 784, 753, 722, 691, 660, 629, 598, 567, 536, 505, 474,
    443, 412, 381, 350, 319, 351, 382, 413, 444, 475, 506, 537, 568, 599, 630, 661, 692, 723, 754, 785, 816, 847, 878, 909, 940, 971, 1002, 1003, 972, 941, 910, 879, 848, 817, 786,
    755, 724, 693, 662, 631, 600, 569, 538, 507, 476, 445, 414, 383, 415, 446, 477, 508, 539, 570, 601, 632, 663, 694, 725, 756, 787, 818, 849, 880, 911, 942, 973, 1004, 1005, 974,
    943, 912, 881, 850, 819, 788, 757, 726, 695, 664, 633, 602, 571, 540, 509, 478, 447, 479, 510, 541, 572, 603, 634, 665, 696, 727, 758, 789, 820, 851, 882, 913, 944, 975, 1006,
    1007, 976, 945, 914, 883, 852, 821, 790, 759, 728, 697, 666, 635, 604, 573, 542, 511, 543, 574, 605, 636, 667, 698, 729, 760, 791, 822, 853, 884, 915, 946, 977, 1008, 1009, 978,
    947, 916, 885, 854, 823, 792, 761, 730, 699, 668, 637, 606, 575, 607, 638, 669, 700, 731, 762, 793, 824, 855, 886, 917, 948, 979, 1010, 1011, 980, 949, 918, 887, 856, 825, 794,
    763, 732, 701, 670, 639, 671, 702, 733, 764, 795, 826, 857, 888, 919, 950, 981, 1012, 1013, 982, 951, 920, 889, 858, 827, 796, 765, 734, 703, 735, 766, 797, 828, 859, 890, 921,
    952, 983, 1014, 1015, 984, 953, 922, 891, 860, 829, 798, 767, 799, 830, 861, 892, 923, 954, 985, 1016, 1017, 986, 955, 924, 893, 862, 831, 863, 894, 925, 956, 987, 1018, 1019,
    988, 957, 926, 895, 927, 958, 989, 1020, 1021, 990, 959, 991, 1022, 1023,
]

Default_Scan_4x16 = [
    0, 1, 4, 2, 5, 8, 3, 6, 9, 12, 7, 10, 13, 16, 11, 14, 17, 20, 15, 18, 21, 24, 19, 22, 25, 28, 23, 26, 29, 32, 27, 30, 33, 36, 31, 34, 37, 40, 35, 38, 41, 44, 39, 42, 45, 48, 43,
    46, 49, 52, 47, 50, 53, 56, 51, 54, 57, 60, 55, 58, 61, 59, 62, 63,
]

Mcol_Scan_4x16 = [
    0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 1, 5, 9, 13, 17, 21, 25, 29, 33, 37, 41, 45, 49, 53, 57, 61, 2, 6, 10, 14, 18, 22, 26, 30, 34, 38, 42, 46, 50, 54,
    58, 62, 3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47, 51, 55, 59, 63,
]

Mrow_Scan_4x16 = [
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46,
    47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63,
]

Default_Scan_16x4 = [
    0, 16, 1, 32, 17, 2, 48, 33, 18, 3, 49, 34, 19, 4, 50, 35, 20, 5, 51, 36, 21, 6, 52, 37, 22, 7, 53, 38, 23, 8, 54, 39, 24, 9, 55, 40, 25, 10, 56, 41, 26, 11, 57, 42, 27, 12, 58,
    43, 28, 13, 59, 44, 29, 14, 60, 45, 30, 15, 61, 46, 31, 62, 47, 63,
]

Mcol_Scan_16x4 = [
    0, 16, 32, 48, 1, 17, 33, 49, 2, 18, 34, 50, 3, 19, 35, 51, 4, 20, 36, 52, 5, 21, 37, 53, 6, 22, 38, 54, 7, 23, 39, 55, 8, 24, 40, 56, 9, 25, 41, 57, 10, 26, 42, 58, 11, 27, 43,
    59, 12, 28, 44, 60, 13, 29, 45, 61, 14, 30, 46, 62, 15, 31, 47, 63,
]

Mrow_Scan_16x4 = [
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46,
    47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63,
]

Default_Scan_8x32 = [
    0, 1, 8, 2, 9, 16, 3, 10, 17, 24, 4, 11, 18, 25, 32, 5, 12, 19, 26, 33, 40, 6, 13, 20, 27, 34, 41, 48, 7, 14, 21, 28, 35, 42, 49, 56, 15, 22, 29, 36, 43, 50, 57, 64, 23, 30, 37,
    44, 51, 58, 65, 72, 31, 38, 45, 52, 59, 66, 73, 80, 39, 46, 53, 60, 67, 74, 81, 88, 47, 54, 61, 68, 75, 82, 89, 96, 55, 62, 69, 76, 83, 90, 97, 104, 63, 70, 77, 84, 91, 98, 105,
    112, 71, 78, 85, 92, 99, 106, 113, 120, 79, 86, 93, 100, 107, 114, 121, 128, 87, 94, 101, 108, 115, 122, 129, 136, 95, 102, 109, 116, 123, 130, 137, 144, 103, 110, 117, 124, 131,
    138, 145, 152, 111, 118, 125, 132, 139, 146, 153, 160, 119, 126, 133, 140, 147, 154, 161, 168, 127, 134, 141, 148, 155, 162, 169, 176, 135, 142, 149, 156, 163, 170, 177, 184,
    143, 150, 157, 164, 171, 178, 185, 192, 151, 158, 165, 172, 179, 186, 193, 200, 159, 166, 173, 180, 187, 194, 201, 208, 167, 174, 181, 188, 195, 202, 209, 216, 175, 182, 189,
    196, 203, 210, 217, 224, 183, 190, 197, 204, 211, 218, 225, 232, 191, 198, 205, 212, 219, 226, 233, 240, 199, 206, 213, 220, 227, 234, 241, 248, 207, 214, 221, 228, 235, 242,
    249, 215, 222, 229, 236, 243, 250, 223, 230, 237, 244, 251, 231, 238, 245, 252, 239, 246, 253, 247, 254, 255,
]

Default_Scan_32x8 = [
    0, 32, 1, 64, 33, 2, 96, 65, 34, 3, 128, 97, 66, 35, 4, 160, 129, 98, 67, 36, 5, 192, 161, 130, 99, 68, 37, 6, 224, 193, 162, 131, 100, 69, 38, 7, 225, 194, 163, 132, 101, 70,
    39, 8, 226, 195, 164, 133, 102, 71, 40, 9, 227, 196, 165, 134, 103, 72, 41, 10, 228, 197, 166, 135, 104, 73, 42, 11, 229, 198, 167, 136, 105, 74, 43, 12, 230, 199, 168, 137, 106,
    75, 44, 13, 231, 200, 169, 138, 107, 76, 45, 14, 232, 201, 170, 139, 108, 77, 46, 15, 233, 202, 171, 140, 109, 78, 47, 16, 234, 203, 172, 141, 110, 79, 48, 17, 235, 204, 173,
    142, 111, 80, 49, 18, 236, 205, 174, 143, 112, 81, 50, 19, 237, 206, 175, 144, 113, 82, 51, 20, 238, 207, 176, 145, 114, 83, 52, 21, 239, 208, 177, 146, 115, 84, 53, 22, 240,
    209, 178, 147, 116, 85, 54, 23, 241, 210, 179, 148, 117, 86, 55, 24, 242, 211, 180, 149, 118, 87, 56, 25, 243, 212, 181, 150, 119, 88, 57, 26, 244, 213, 182, 151, 120, 89, 58,
    27, 245, 214, 183, 152, 121, 90, 59, 28, 246, 215, 184, 153, 122, 91, 60, 29, 247, 216, 185, 154, 123, 92, 61, 30, 248, 217, 186, 155, 124, 93, 62, 31, 249, 218, 187, 156, 125,
    94, 63, 250, 219, 188, 157, 126, 95, 251, 220, 189, 158, 127, 252, 221, 190, 159, 253, 222, 191, 254, 223, 255,
]
