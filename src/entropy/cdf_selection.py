"""
CDF选择过程
按照规范文档8.3节实现CDF选择功能
"""

from typing import List, Any, Dict
from constants import PARTITION, REF_FRAME, SUB_SIZE, TX_SIZE
from constants import COMP_NEWMV_CTXS, H_ADST, H_DCT, H_FLIPADST, SIG_COEF_CONTEXTS, SIG_COEF_CONTEXTS_EOB, SIG_REF_DIFF_OFFSET_NUM, TX_CLASS_2D, TX_CLASS_HORIZ, TX_CLASS_VERT, TX_SIZES, V_ADST, V_DCT, V_FLIPADST, Coeff_Base_Ctx_Offset, Coeff_Base_Pos_Ctx_Offset, Compound_Mode_Ctx_Map, Intra_Mode_Context, Y_MODE, Sig_Ref_Diff_Offset, Size_Group, Mi_Width_Log2, Mi_Height_Log2, Tx_Size_Sqr_Up
from constants import Block_Width, Block_Height, Tx_Width, Tx_Height, Adjusted_Tx_Size, Tx_Width_Log2, Mag_Ref_Offset_With_Tx_Class, COEFF_BASE_RANGE, NUM_BASE_LEVELS, Palette_Color_Context, Filter_Intra_Mode_To_Intra_Dir, Tx_Size_Sqr, TX_SET
from obu.decoder import AV1Decoder
from frame.frame_header import inverseCdf


def use_intrabc(av1: AV1Decoder) -> List[int]:
    """
    use_intrabc CDF选择
    规范文档 8.3.2: The cdf for use_intrabc is given by TileIntrabcCdf
    """
    decoder = av1.decoder
    return decoder.tile_cdfs['TileIntrabcCdf']


def intra_frame_y_mode(av1: AV1Decoder) -> List[int]:
    """
    intra_frame_y_mode CDF选择
    规范文档 8.3.2: TileIntraFrameYModeCdf[abovemode][leftmode]
    """
    tile_group = av1.tile_group
    deroder = av1.decoder
    MiRow = tile_group.MiRow
    MiCol = tile_group.MiCol
    AvailU = tile_group.AvailU
    AvailL = tile_group.AvailL

    above_mode_raw = tile_group.YModes[MiRow -
                                       1][MiCol] if AvailU else Y_MODE.DC_PRED
    left_mode_raw = tile_group.YModes[MiRow][MiCol -
                                             1] if AvailL else Y_MODE.DC_PRED
    abovemode = Intra_Mode_Context[above_mode_raw]
    leftmode = Intra_Mode_Context[left_mode_raw]
    return deroder.tile_cdfs['TileIntraFrameYModeCdf'][abovemode][leftmode]


def y_mode(av1: AV1Decoder) -> List[int]:
    """
    y_mode CDF选择
    规范文档 8.3.2: TileYModeCdf[ctx] where ctx = Size_Group[MiSize]
    """
    tile_group = av1.tile_group
    decoder = av1.decoder
    MiSize = tile_group.MiSize

    ctx = Size_Group[MiSize]
    return decoder.tile_cdfs['TileYModeCdf'][ctx]


def uv_mode(av1: AV1Decoder) -> List[int]:
    """
    uv_mode CDF选择
    规范文档 8.3.2: 根据Lossless和块尺寸选择TileUVModeCflAllowedCdf或TileUVModeCflNotAllowedCdf
    """
    from utils.tile_utils import get_plane_residual_size
    tile_group = av1.tile_group
    decoder = av1.decoder
    MiSize = tile_group.MiSize
    YMode = tile_group.YMode

    if tile_group.Lossless == 1 and get_plane_residual_size(av1, MiSize, 1) == SUB_SIZE.BLOCK_4X4:
        return decoder.tile_cdfs['TileUVModeCflAllowedCdf'][YMode]
    elif tile_group.Lossless == 0 and max(Block_Width[MiSize], Block_Height[MiSize]) <= 32:
        return decoder.tile_cdfs['TileUVModeCflAllowedCdf'][YMode]
    else:
        return decoder.tile_cdfs['TileUVModeCflNotAllowedCdf'][YMode]


def angle_delta_y(av1: AV1Decoder) -> List[int]:
    """
    angle_delta_y CDF选择
    规范文档 8.3.2: TileAngleDeltaCdf[YMode - V_PRED]
    """
    tile_group = av1.tile_group
    decoder = av1.decoder
    YMode = tile_group.YMode

    return decoder.tile_cdfs['TileAngleDeltaCdf'][YMode - Y_MODE.V_PRED]


def angle_delta_uv(av1: AV1Decoder) -> List[int]:
    """
    angle_delta_uv CDF选择
    规范文档 8.3.2: TileAngleDeltaCdf[UVMode - V_PRED]
    """
    tile_group = av1.tile_group
    decoder = av1.decoder
    UVMode = tile_group.UVMode

    return decoder.tile_cdfs['TileAngleDeltaCdf'][UVMode - Y_MODE.V_PRED]


def partition(av1: AV1Decoder, r: int, c: int, bSize: SUB_SIZE) -> List[int]:
    """
    partition CDF选择
    规范文档 8.3.2: 根据bsl选择TilePartitionW*Cdf[ctx]
    """
    tile_group = av1.tile_group
    decoder = av1.decoder
    AvailU = tile_group.AvailU
    AvailL = tile_group.AvailL

    bsl = Mi_Width_Log2[bSize]
    above = AvailU and Mi_Width_Log2[tile_group.MiSizes[r - 1][c]] < bsl
    left = AvailL and Mi_Height_Log2[tile_group.MiSizes[r][c - 1]] < bsl
    ctx = left * 2 + above

    if bsl == 1:
        return decoder.tile_cdfs['TilePartitionW8Cdf'][ctx]
    elif bsl == 2:
        return decoder.tile_cdfs['TilePartitionW16Cdf'][ctx]
    elif bsl == 3:
        return decoder.tile_cdfs['TilePartitionW32Cdf'][ctx]
    elif bsl == 4:
        return decoder.tile_cdfs['TilePartitionW64Cdf'][ctx]
    else:
        return decoder.tile_cdfs['TilePartitionW128Cdf'][ctx]


def split_or_horz(av1: AV1Decoder, r: int, c: int, bSize: SUB_SIZE) -> List[int]:
    """
    split_or_horz CDF选择
    规范文档 8.3.2: TileSplitOrHorzCdf[ctx]
    """
    bsl = Mi_Width_Log2[bSize]
    # note that bsl is never equal to 1 when decoding split_or_horz
    assert bsl != 1

    partitionCdf = partition(av1, r, c, bSize)
    psum = (
        partitionCdf[PARTITION.PARTITION_VERT] - partitionCdf[PARTITION.PARTITION_VERT - 1] +
        partitionCdf[PARTITION.PARTITION_SPLIT] - partitionCdf[PARTITION.PARTITION_SPLIT - 1] +
        partitionCdf[PARTITION.PARTITION_HORZ_A] - partitionCdf[PARTITION.PARTITION_HORZ_A - 1] +
        partitionCdf[PARTITION.PARTITION_VERT_A] - partitionCdf[PARTITION.PARTITION_VERT_A - 1] +
        partitionCdf[PARTITION.PARTITION_VERT_B] -
        partitionCdf[PARTITION.PARTITION_VERT_B - 1]
    )
    if bSize != SUB_SIZE.BLOCK_128X128:
        psum += (partitionCdf[PARTITION.PARTITION_VERT_4] -
                 partitionCdf[PARTITION.PARTITION_VERT_4 - 1])

    # Note: Implementations may prefer to store the inverse cdf to move the subtraction out of this loop.
    cdf = [(1 << 15) - psum, 1 << 15, 0]
    inverseCdf(cdf)
    if cdf[-2] != 1 << 15:
        cdf[0] = -cdf[0]
    return cdf


def split_or_vert(av1: AV1Decoder, r: int, c: int, bSize: SUB_SIZE) -> List[int]:
    """
    split_or_vert CDF选择
    规范文档 8.3.2: TileSplitOrVertCdf[ctx]
    """
    bsl = Mi_Width_Log2[bSize]
    # note that bsl is never equal to 1 when decoding split_or_horz
    assert bsl != 1

    partitionCdf = partition(av1, r, c, bSize)
    psum = (
        partitionCdf[PARTITION.PARTITION_HORZ] - partitionCdf[PARTITION.PARTITION_HORZ - 1] +
        partitionCdf[PARTITION.PARTITION_SPLIT] - partitionCdf[PARTITION.PARTITION_SPLIT - 1] +
        partitionCdf[PARTITION.PARTITION_HORZ_A] - partitionCdf[PARTITION.PARTITION_HORZ_A - 1] +
        partitionCdf[PARTITION.PARTITION_HORZ_B] - partitionCdf[PARTITION.PARTITION_HORZ_B - 1] +
        partitionCdf[PARTITION.PARTITION_VERT_A] -
        partitionCdf[PARTITION.PARTITION_VERT_A - 1]
    )
    if bSize != SUB_SIZE.BLOCK_128X128:
        psum += (partitionCdf[PARTITION.PARTITION_HORZ_4] -
                 partitionCdf[PARTITION.PARTITION_HORZ_4 - 1])

    # Note: Implementations may prefer to store the inverse cdf to move the subtraction out of this loop.
    cdf = [(1 << 15)-psum, 1 << 15, 0]
    inverseCdf(cdf)
    if cdf[-2] != 1 << 15:
        cdf[0] = -cdf[0]
    return cdf


def _get_above_tx_width(av1: AV1Decoder, row: int, col: int) -> int:
    """
    get_above_tx_width CDF选择
    规范文档 8.3.2: TileAboveTxWidthCdf[ctx]
    """
    tile_group = av1.tile_group
    MiRow = tile_group.MiRow
    AvailU = tile_group.AvailU

    if row == MiRow:
        if not AvailU:
            return 64
        elif tile_group.Skips[row - 1][col] and tile_group.IsInters[row - 1][col]:
            return Block_Width[tile_group.MiSizes[row - 1][col]]
    return Tx_Width[tile_group.InterTxSizes[row - 1][col]]


def _get_left_tx_height(av1: AV1Decoder, row: int, col: int) -> int:
    """
    get_left_tx_height CDF选择
    规范文档 8.3.2: TileLeftTxHeightCdf[ctx]
    """
    tile_group = av1.tile_group
    MiCol = tile_group.MiCol
    AvailL = tile_group.AvailL

    if col == MiCol:
        if not AvailL:
            return 64
        elif tile_group.Skips[row][col - 1] and tile_group.IsInters[row][col - 1]:
            return Block_Height[tile_group.MiSizes[row][col - 1]]

    return Tx_Height[tile_group.InterTxSizes[row][col - 1]]


def tx_depth(av1: AV1Decoder, maxRectTxSize: int, maxTxDepth: int) -> List[int]:
    """
    tx_depth CDF选择
    规范文档 8.3.2: 根据maxTxDepth选择TileTx*Cdf[ctx]
    """
    tile_group = av1.tile_group
    decoder = av1.decoder
    MiRow = tile_group.MiRow
    MiCol = tile_group.MiCol
    AvailU = tile_group.AvailU
    AvailL = tile_group.AvailL

    maxTxWidth = Tx_Width[maxRectTxSize]
    maxTxHeight = Tx_Height[maxRectTxSize]

    if AvailU and tile_group.IsInters[MiRow - 1][MiCol]:
        aboveW = Block_Width[tile_group.MiSizes[MiRow - 1][MiCol]]
    elif AvailU:
        aboveW = _get_above_tx_width(av1, MiRow, MiCol)
    else:
        aboveW = 0

    if AvailL and tile_group.IsInters[MiRow][MiCol - 1]:
        leftH = Block_Height[tile_group.MiSizes[MiRow][MiCol - 1]]
    elif AvailL:
        leftH = _get_left_tx_height(av1, MiRow, MiCol)
    else:
        leftH = 0

    ctx = (aboveW >= maxTxWidth) + (leftH >= maxTxHeight)

    if maxTxDepth == 4:
        return decoder.tile_cdfs['TileTx64x64Cdf'][ctx]
    elif maxTxDepth == 3:
        return decoder.tile_cdfs['TileTx32x32Cdf'][ctx]
    elif maxTxDepth == 2:
        return decoder.tile_cdfs['TileTx16x16Cdf'][ctx]
    else:
        return decoder.tile_cdfs['TileTx8x8Cdf'][ctx]


def txfm_split(av1: AV1Decoder, row: int, col: int, txSz: int) -> List[int]:
    """
    txfm_split CDF选择
    规范文档 8.3.2: TileTxfmSplitCdf[ctx]
    """
    from utils.tile_utils import find_tx_size
    tile_group = av1.tile_group
    decoder = av1.decoder

    above = _get_above_tx_width(av1, row, col) < Tx_Width[txSz]
    left = _get_left_tx_height(av1, row, col) < Tx_Height[txSz]
    size = min(
        64, max(Block_Width[tile_group.MiSize], Block_Height[tile_group.MiSize]))

    maxTxSz = find_tx_size(size, size)
    txSzSqrUp = Tx_Size_Sqr_Up[txSz]
    ctx = ((txSzSqrUp != maxTxSz) * 3 +
           (TX_SIZES - 1 - maxTxSz) * 6 + above + left)

    return decoder.tile_cdfs['TileTxfmSplitCdf'][ctx]


def segment_id(av1: AV1Decoder, prevUL: int, prevU: int, prevL: int) -> List[int]:
    """
    segment_id CDF选择
    规范文档 8.3.2: TileSegmentIdCdf[ctx]
    """
    decoder = av1.decoder

    if prevUL < 0:
        ctx = 0
    elif prevUL == prevU and prevUL == prevL:
        ctx = 2
    elif prevUL == prevU or prevUL == prevL or prevU == prevL:
        ctx = 1
    else:
        ctx = 0

    return decoder.tile_cdfs['TileSegmentIdCdf'][ctx]


def seg_id_predicted(av1: AV1Decoder) -> List[int]:
    """
    seg_id_predicted CDF选择
    规范文档 8.3.2: TileSegmentIdPredictedCdf[ctx]
    """
    tile_group = av1.tile_group
    decoder = av1.decoder
    MiRow = tile_group.MiRow
    MiCol = tile_group.MiCol

    ctx = (tile_group.LeftSegPredContext[MiRow] +
           tile_group.AboveSegPredContext[MiCol])

    return decoder.tile_cdfs['TileSegmentIdPredictedCdf'][ctx]


def new_mv(av1: AV1Decoder) -> List[int]:
    """
    new_mv CDF选择
    规范文档 8.3.2: TileNewMvCdf[NewMvContext]
    """
    tile_group = av1.tile_group
    decoder = av1.decoder

    return decoder.tile_cdfs['TileNewMvCdf'][tile_group.NewMvContext]


def zero_mv(av1: AV1Decoder) -> List[int]:
    """
    zero_mv CDF选择
    规范文档 8.3.2: TileZeroMvCdf[ZeroMvContext]
    """
    tile_group = av1.tile_group
    decoder = av1.decoder

    return decoder.tile_cdfs['TileZeroMvCdf'][tile_group.ZeroMvContext]


def ref_mv(av1: AV1Decoder) -> List[int]:
    """
    ref_mv CDF选择
    规范文档 8.3.2: TileRefMvCdf[RefMvContext]
    """
    tile_group = av1.tile_group
    decoder = av1.decoder

    return decoder.tile_cdfs['TileRefMvCdf'][tile_group.RefMvContext]


def drl_mode(av1: AV1Decoder, idx: int) -> List[int]:
    """
    drl_mode CDF选择
    规范文档 8.3.2: TileDrlModeCdf[DrlCtxStack[idx]]
    """
    tile_group = av1.tile_group
    decoder = av1.decoder

    return decoder.tile_cdfs['TileDrlModeCdf'][tile_group.DrlCtxStack[idx]]


def is_inter(av1: AV1Decoder) -> List[int]:
    """
    is_inter CDF选择
    规范文档 8.3.2: TileIsInterCdf[ctx]
    """
    tile_group = av1.tile_group
    decoder = av1.decoder
    AvailU = tile_group.AvailU
    AvailL = tile_group.AvailL

    if AvailU and AvailL:
        if tile_group.LeftIntra and tile_group.AboveIntra:
            ctx = 3
        else:
            ctx = tile_group.LeftIntra or tile_group.AboveIntra
    elif AvailU or AvailL:
        ctx = 2 * (tile_group.AboveIntra if AvailU else tile_group.LeftIntra)
    else:
        ctx = 0

    return decoder.tile_cdfs['TileIsInterCdf'][ctx]


def use_filter_intra(av1: AV1Decoder) -> List[int]:
    """
    use_filter_intra CDF选择
    规范文档 8.3.2: TileFilterIntraCdf[MiSize]
    """
    tile_group = av1.tile_group
    decoder = av1.decoder
    MiSize = tile_group.MiSize

    return decoder.tile_cdfs['TileFilterIntraCdf'][MiSize]


def filter_intra_mode(av1: AV1Decoder) -> List[int]:
    """
    filter_intra_mode CDF选择
    规范文档 8.3.2: TileFilterIntraModeCdf
    """
    decoder = av1.decoder
    return decoder.tile_cdfs['TileFilterIntraModeCdf']


def _check_backward(refFrame: REF_FRAME) -> int:
    return refFrame >= REF_FRAME.BWDREF_FRAME and refFrame <= REF_FRAME.ALTREF_FRAME


def comp_mode(av1: AV1Decoder) -> List[int]:
    """
    comp_mode CDF选择
    规范文档 8.3.2: TileCompModeCdf[ctx]
    """
    tile_group = av1.tile_group
    decoder = av1.decoder
    AvailU = tile_group.AvailU
    AvailL = tile_group.AvailL

    if AvailU and AvailL:
        if tile_group.AboveSingle and tile_group.LeftSingle:
            ctx = (_check_backward(tile_group.AboveRefFrame[0]) ^
                   _check_backward(tile_group.LeftRefFrame[0]))
        elif tile_group.AboveSingle:
            ctx = (2 +
                   (_check_backward(
                    tile_group.AboveRefFrame[0]) or tile_group.AboveIntra))
        elif tile_group.LeftSingle:
            ctx = (2 +
                   (_check_backward(
                    tile_group.LeftRefFrame[0]) or tile_group.LeftIntra))
        else:
            ctx = 4
    elif AvailU:
        if tile_group.AboveSingle:
            ctx = _check_backward(tile_group.AboveRefFrame[0])
        else:
            ctx = 3
    elif AvailL:
        if tile_group.LeftSingle:
            ctx = _check_backward(tile_group.LeftRefFrame[0])
        else:
            ctx = 3
    else:
        ctx = 1

    return decoder.tile_cdfs['TileCompModeCdf'][ctx]


def skip_mode(av1: AV1Decoder) -> List[int]:
    """
    skip_mode CDF选择
    规范文档 8.3.2: TileSkipModeCdf[ctx]
    """
    tile_group = av1.tile_group
    decoder = av1.decoder
    MiRow = tile_group.MiRow
    MiCol = tile_group.MiCol
    AvailU = tile_group.AvailU
    AvailL = tile_group.AvailL

    ctx = 0
    if AvailU:
        ctx += tile_group.SkipModes[MiRow - 1][MiCol]
    if AvailL:
        ctx += tile_group.SkipModes[MiRow][MiCol - 1]

    return decoder.tile_cdfs['TileSkipModeCdf'][ctx]


def skip(av1: AV1Decoder) -> List[int]:
    """
    skip CDF选择
    规范文档 8.3.2: TileSkipCdf[ctx]
    """
    tile_group = av1.tile_group
    decoder = av1.decoder
    MiRow = tile_group.MiRow
    MiCol = tile_group.MiCol
    AvailU = tile_group.AvailU
    AvailL = tile_group.AvailL

    ctx = 0
    if AvailU:
        ctx += tile_group.Skips[MiRow - 1][MiCol]
    if AvailL:
        ctx += tile_group.Skips[MiRow][MiCol - 1]

    return decoder.tile_cdfs['TileSkipCdf'][ctx]


def _comp_ref_ctx(av1: AV1Decoder) -> int:
    """
    comp_ref_ctx CDF选择
    规范文档 8.3.2: TileCompRefCdf[ctx]
    """
    last12Count = (_count_refs(av1, REF_FRAME.LAST_FRAME) +
                   _count_refs(av1, REF_FRAME.LAST2_FRAME))
    last3GoldCount = _count_refs(
        av1, REF_FRAME.LAST3_FRAME) + _count_refs(av1, REF_FRAME.GOLDEN_FRAME)
    return _ref_count_ctx(last12Count, last3GoldCount)


def comp_ref(av1: AV1Decoder) -> List[int]:
    """
    comp_ref CDF选择
    规范文档 8.3.2: TileCompRefCdf[ctx]
    """
    decoder = av1.decoder
    ctx = _comp_ref_ctx(av1)
    return decoder.tile_cdfs['TileCompRefCdf'][ctx][0]


def _comp_ref_p1_ctx(av1: AV1Decoder) -> int:
    """
    comp_ref_p1_ctx CDF选择
    规范文档 8.3.2: TileCompRefCdf[ctx]
    """
    lastCount = _count_refs(av1, REF_FRAME.LAST_FRAME)
    last2Count = _count_refs(av1, REF_FRAME.LAST2_FRAME)
    return _ref_count_ctx(lastCount, last2Count)


def comp_ref_p1(av1: AV1Decoder) -> List[int]:
    """
    comp_ref_p1 CDF选择
    规范文档 8.3.2: TileCompRefCdf[ctx]
    """
    decoder = av1.decoder
    ctx = _comp_ref_p1_ctx(av1)
    return decoder.tile_cdfs['TileCompRefCdf'][ctx][1]


def comp_ref_p2(av1: AV1Decoder) -> List[int]:
    """
    comp_ref_p2 CDF选择
    规范文档 8.3.2: TileCompRefCdf[ctx]
    """
    decoder = av1.decoder
    ctx = _comp_ref_p2_ctx(av1)
    return decoder.tile_cdfs['TileCompRefCdf'][ctx][2]


def _comp_bwdref_ctx(av1: AV1Decoder) -> int:
    """
    comp_bwdref_ctx CDF选择
    规范文档 8.3.2: TileCompBwdrefCdf[ctx]
    """
    brfarf2Count = _count_refs(
        av1, REF_FRAME.BWDREF_FRAME) + _count_refs(av1, REF_FRAME.ALTREF2_FRAME)
    arfCount = _count_refs(av1, REF_FRAME.ALTREF_FRAME)
    return _ref_count_ctx(brfarf2Count, arfCount)


def comp_bwdref(av1: AV1Decoder) -> List[int]:
    """
    comp_bwdref CDF选择
    规范文档 8.3.2: TileCompBwdrefCdf[ctx]
    """
    decoder = av1.decoder
    ctx = _comp_bwdref_ctx(av1)
    return decoder.tile_cdfs['TileCompBwdRefCdf'][ctx][0]


def _comp_bwdref_p1_ctx(av1: AV1Decoder) -> int:
    """
    comp_bwdref_p1_ctx CDF选择
    规范文档 8.3.2: TileCompBwdrefCdf[ctx]
    """
    brfCount = _count_refs(av1, REF_FRAME.BWDREF_FRAME)
    arf2Count = _count_refs(av1, REF_FRAME.ALTREF2_FRAME)
    return _ref_count_ctx(brfCount, arf2Count)


def comp_bwdref_p1(av1: AV1Decoder) -> List[int]:
    """
    comp_bwdref_p1 CDF选择
    规范文档 8.3.2: TileCompBwdrefCdf[ctx]
    """
    decoder = av1.decoder
    ctx = _comp_bwdref_p1_ctx(av1)
    return decoder.tile_cdfs['TileCompBwdRefCdf'][ctx][1]


def single_ref_p1(av1: AV1Decoder) -> List[int]:
    """
    single_ref_p1 CDF选择
    规范文档 8.3.2: TileSingleRefCdf[ctx]
    """
    decoder = av1.decoder
    ctx = _single_ref_p1_ctx(av1)
    return decoder.tile_cdfs['TileSingleRefCdf'][ctx][0]


def single_ref_p2(av1: AV1Decoder) -> List[int]:
    """
    single_ref_p2 CDF选择
    规范文档 8.3.2: TileSingleRefCdf[ctx]
    """
    decoder = av1.decoder
    ctx = _comp_bwdref_ctx(av1)
    return decoder.tile_cdfs['TileSingleRefCdf'][ctx][1]


def single_ref_p3(av1: AV1Decoder) -> List[int]:
    """
    single_ref_p3 CDF选择
    规范文档 8.3.2: TileSingleRefCdf[ctx]
    """
    decoder = av1.decoder
    ctx = _comp_ref_ctx(av1)
    return decoder.tile_cdfs['TileSingleRefCdf'][ctx][2]


def single_ref_p4(av1: AV1Decoder) -> List[int]:
    """
    single_ref_p4 CDF选择
    规范文档 8.3.2: TileSingleRefCdf[ctx]
    """
    decoder = av1.decoder
    ctx = _comp_ref_p1_ctx(av1)
    return decoder.tile_cdfs['TileSingleRefCdf'][ctx][3]


def single_ref_p5(av1: AV1Decoder) -> List[int]:
    """
    single_ref_p5 CDF选择
    规范文档 8.3.2: TileSingleRefCdf[ctx]
    """
    decoder = av1.decoder
    ctx = _comp_ref_p2_ctx(av1)
    return decoder.tile_cdfs['TileSingleRefCdf'][ctx][4]


def single_ref_p6(av1: AV1Decoder) -> List[int]:
    """
    single_ref_p6 CDF选择
    规范文档 8.3.2: TileSingleRefCdf[ctx]
    """
    decoder = av1.decoder
    ctx = _comp_bwdref_p1_ctx(av1)
    return decoder.tile_cdfs['TileSingleRefCdf'][ctx][5]


def compound_mode(av1: AV1Decoder) -> List[int]:
    """
    compound_mode CDF选择
    规范文档 8.3.2: TileCompoundModeCdf[ctx]
    """
    tile_group = av1.tile_group
    decoder = av1.decoder

    ctx = Compound_Mode_Ctx_Map[tile_group.RefMvContext >> 1][min(
        tile_group.NewMvContext, COMP_NEWMV_CTXS - 1)]
    return decoder.tile_cdfs['TileCompoundModeCdf'][ctx]


def interp_filter(av1: AV1Decoder, dir_val: int) -> List[int]:
    """
    interp_filter CDF选择
    规范文档 8.3.2: TileInterpFilterCdf[ctx]
    """
    tile_group = av1.tile_group
    decoder = av1.decoder
    MiRow = tile_group.MiRow
    MiCol = tile_group.MiCol
    AvailU = tile_group.AvailU
    AvailL = tile_group.AvailL

    ctx = ((dir_val & 1) * 2 +
           (tile_group.RefFrame[1] > REF_FRAME.INTRA_FRAME)) * 4

    leftType = 3
    aboveType = 3
    if AvailL:
        if (tile_group.RefFrames[MiRow][MiCol - 1][0] == tile_group.RefFrame[0] or
                tile_group.RefFrames[MiRow][MiCol - 1][1] == tile_group.RefFrame[0]):
            leftType = tile_group.InterpFilters[MiRow][MiCol - 1][dir_val]
    if AvailU:
        if (tile_group.RefFrames[MiRow - 1][MiCol][0] == tile_group.RefFrame[0] or
                tile_group.RefFrames[MiRow - 1][MiCol][1] == tile_group.RefFrame[0]):
            aboveType = tile_group.InterpFilters[MiRow - 1][MiCol][dir_val]

    if leftType == aboveType:
        ctx += leftType
    elif leftType == 3:
        ctx += aboveType
    elif aboveType == 3:
        ctx += leftType
    else:
        ctx += 3

    return decoder.tile_cdfs['TileInterpFilterCdf'][ctx]


def motion_mode(av1: AV1Decoder) -> List[int]:
    """
    motion_mode CDF选择
    规范文档 8.3.2: TileMotionModeCdf[MiSize]
    """
    tile_group = av1.tile_group
    decoder = av1.decoder
    MiSize = tile_group.MiSize

    return decoder.tile_cdfs['TileMotionModeCdf'][MiSize]


def mv_joint(av1: AV1Decoder) -> List[int]:
    """
    mv_joint CDF选择
    规范文档 8.3.2: TileMvJointCdf[MvCtx]
    """
    tile_group = av1.tile_group
    decoder = av1.decoder

    return decoder.tile_cdfs['TileMvJointCdf'][tile_group.MvCtx]


def mv_sign(av1: AV1Decoder, comp: int) -> List[int]:
    """
    mv_sign CDF选择
    规范文档 8.3.2: TileMvSignCdf[MvCtx]
    """
    tile_group = av1.tile_group
    decoder = av1.decoder

    return decoder.tile_cdfs['TileMvSignCdf'][tile_group.MvCtx][comp]


def mv_class(av1: AV1Decoder, comp: int) -> List[int]:
    """
    mv_class CDF选择
    规范文档 8.3.2: TileMvClassCdf[MvCtx]
    """
    tile_group = av1.tile_group
    decoder = av1.decoder

    return decoder.tile_cdfs['TileMvClassCdf'][tile_group.MvCtx][comp]


def mv_class0_bit(av1: AV1Decoder, comp: int) -> List[int]:
    """
    mv_class0_bit CDF选择
    规范文档 8.3.2: TileMvClass0BitCdf[MvCtx]
    """
    tile_group = av1.tile_group
    decoder = av1.decoder

    return decoder.tile_cdfs['TileMvClass0BitCdf'][tile_group.MvCtx][comp]


def mv_class0_fr(av1: AV1Decoder, comp: int, mv_class0_bit: int) -> List[int]:
    """
    mv_class0_fr CDF选择
    规范文档 8.3.2: TileMvClass0FrCdf[MvCtx]
    """
    tile_group = av1.tile_group
    decoder = av1.decoder

    return decoder.tile_cdfs['TileMvClass0FrCdf'][tile_group.MvCtx][comp][mv_class0_bit]


def mv_class0_hp(av1: AV1Decoder, comp: int) -> List[int]:
    """
    mv_class0_hp CDF选择
    规范文档 8.3.2: TileMvClass0HpCdf[MvCtx]
    """
    tile_group = av1.tile_group
    decoder = av1.decoder

    return decoder.tile_cdfs['TileMvClass0HpCdf'][tile_group.MvCtx][comp]


def mv_fr(av1: AV1Decoder, comp: int) -> List[int]:
    """
    mv_fr CDF选择
    规范文档 8.3.2: TileMvFrCdf[MvCtx]
    """
    tile_group = av1.tile_group
    decoder = av1.decoder

    return decoder.tile_cdfs['TileMvFrCdf'][tile_group.MvCtx][comp]


def mv_hp(av1: AV1Decoder, comp: int) -> List[int]:
    """
    mv_hp CDF选择
    规范文档 8.3.2: TileMvHpCdf[MvCtx]
    """
    tile_group = av1.tile_group
    decoder = av1.decoder

    return decoder.tile_cdfs['TileMvHpCdf'][tile_group.MvCtx][comp]


def mv_bit(av1: AV1Decoder, comp: int, i: int) -> List[int]:
    """
    mv_bit CDF选择
    规范文档 8.3.2: TileMvBitCdf[MvCtx]
    """
    tile_group = av1.tile_group
    decoder = av1.decoder

    return decoder.tile_cdfs['TileMvBitCdf'][tile_group.MvCtx][comp][i]


def all_zero(av1: AV1Decoder, txSzCtx: int, plane: int, txSz: int, x4: int, y4: int, w4: int, h4: int) -> List[int]:
    """
    all_zero CDF选择
    规范文档 8.3.2: TileTxbSkipCdf[MvCtx]
    """
    from utils.tile_utils import get_plane_residual_size
    seq_header = av1.seq_header
    frame_header = av1.frame_header
    tile_group = av1.tile_group
    decoder = av1.decoder
    subsampling_x = seq_header.color_config.subsampling_x
    subsampling_y = seq_header.color_config.subsampling_y
    MiRows = frame_header.MiRows
    MiCols = frame_header.MiCols
    MiSize = tile_group.MiSize

    maxX4 = MiCols
    maxY4 = MiRows
    if plane > 0:
        maxX4 = maxX4 >> subsampling_x
        maxY4 = maxY4 >> subsampling_y

    w = Tx_Width[txSz]
    h = Tx_Height[txSz]

    bsize = get_plane_residual_size(av1, MiSize, plane)
    bw = Block_Width[bsize]
    bh = Block_Height[bsize]

    if plane == 0:
        top = 0
        left = 0
        for k in range(0, w4):
            if x4 + k < maxX4:
                top = max(top, tile_group.AboveLevelContext[plane][x4 + k])
        for k in range(0, h4):
            if y4 + k < maxY4:
                left = max(left, tile_group.LeftLevelContext[plane][y4 + k])
        top = min(top, 255)
        left = min(left, 255)
        if bw == w and bh == h:
            ctx = 0
        elif top == 0 and left == 0:
            ctx = 1
        elif top == 0 or left == 0:
            ctx = 2 + (max(top, left) > 3)
        elif max(top, left) <= 3:
            ctx = 4
        elif min(top, left) <= 3:
            ctx = 5
        else:
            ctx = 6
    else:
        above = 0
        left = 0
        for i in range(0, w4):
            if x4 + i < maxX4:
                above |= tile_group.AboveLevelContext[plane][x4 + i]
                above |= tile_group.AboveDcContext[plane][x4 + i]
        for i in range(0, h4):
            if y4 + i < maxY4:
                left |= tile_group.LeftLevelContext[plane][y4 + i]
                left |= tile_group.LeftDcContext[plane][y4 + i]
        ctx = (above != 0) + (left != 0)
        ctx += 7
        if bw * bh > w * h:
            ctx += 3

    return decoder.tile_cdfs['TileTxbSkipCdf'][txSzCtx][ctx]


def _get_tx_class(txType: int) -> int:
    """
    get_tx_class
    规范文档 8.3.2: get_tx_class
    """
    if txType in [V_DCT, V_ADST, V_FLIPADST]:
        return TX_CLASS_VERT
    elif txType in [H_DCT, H_ADST, H_FLIPADST]:
        return TX_CLASS_HORIZ
    else:
        return TX_CLASS_2D


def _eob_pt_16_ctx(av1: AV1Decoder, plane: int, txSz: TX_SIZE, x4: int, y4: int) -> int:
    """
    get_eob_pt_ctx
    规范文档 8.3.2: get_eob_pt_ctx
    """
    from utils.tile_utils import compute_tx_type

    txType = compute_tx_type(av1, plane, txSz, x4, y4)
    return 0 if _get_tx_class(txType) == TX_CLASS_2D else 1


def eob_pt_16(av1: AV1Decoder, ptype: int, plane: int, txSz: TX_SIZE, x4: int, y4: int) -> List[int]:
    """
    eob_pt_16 CDF选择
    规范文档 8.3.2: TileEobPt16Cdf[ptype][ctx]
    """
    decoder = av1.decoder

    ctx = _eob_pt_16_ctx(av1, plane, txSz, x4, y4)
    return decoder.tile_cdfs['TileEobPt16Cdf'][ptype][ctx]


def eob_pt_32(av1: AV1Decoder, ptype: int, plane: int, txSz: TX_SIZE, x4: int, y4: int) -> List[int]:
    """
    eob_pt_32 CDF选择
    规范文档 8.3.2: TileEobPt32Cdf[ptype][ctx]
    """
    decoder = av1.decoder

    ctx = _eob_pt_16_ctx(av1, plane, txSz, x4, y4)
    return decoder.tile_cdfs['TileEobPt32Cdf'][ptype][ctx]


def eob_pt_64(av1: AV1Decoder, ptype: int, plane: int, txSz: TX_SIZE, x4: int, y4: int) -> List[int]:
    """
    eob_pt_64 CDF选择
    规范文档 8.3.2: TileEobPt64Cdf[ptype][ctx]
    """
    decoder = av1.decoder

    ctx = _eob_pt_16_ctx(av1, plane, txSz, x4, y4)
    return decoder.tile_cdfs['TileEobPt64Cdf'][ptype][ctx]


def eob_pt_128(av1: AV1Decoder, ptype: int, plane: int, txSz: TX_SIZE, x4: int, y4: int) -> List[int]:
    """
    eob_pt_128 CDF选择
    规范文档 8.3.2: TileEobPt128Cdf[ptype][ctx]
    """
    decoder = av1.decoder

    ctx = _eob_pt_16_ctx(av1, plane, txSz, x4, y4)
    return decoder.tile_cdfs['TileEobPt128Cdf'][ptype][ctx]


def eob_pt_256(av1: AV1Decoder, ptype: int, plane: int, txSz: TX_SIZE, x4: int, y4: int) -> List[int]:
    """
    eob_pt_256 CDF选择
    """
    decoder = av1.decoder

    ctx = _eob_pt_16_ctx(av1, plane, txSz, x4, y4)
    return decoder.tile_cdfs['TileEobPt256Cdf'][ptype][ctx]


def eob_pt_512(av1: AV1Decoder, ptype: int) -> List[int]:
    """
    eob_pt_512 CDF选择
    规范文档 8.3.2: TileEobPt512Cdf[ptype][ctx]
    """
    decoder = av1.decoder

    return decoder.tile_cdfs['TileEobPt512Cdf'][ptype]


def eob_pt_1024(av1: AV1Decoder, ptype: int) -> List[int]:
    """
    eob_pt_1024 CDF选择
    规范文档 8.3.2: TileEobPt1024Cdf[ptype][ctx]
    """
    decoder = av1.decoder

    return decoder.tile_cdfs['TileEobPt1024Cdf'][ptype]


def eob_extra(av1: AV1Decoder, txSzCtx: int, ptype: int, eobPt: int) -> List[int]:
    """
    eob_extra CDF选择
    规范文档 8.3.2: TileEobExtraCdf[txSzCtx][ptype][eobPt]
    """
    decoder = av1.decoder
    return decoder.tile_cdfs['TileEobExtraCdf'][txSzCtx][ptype][eobPt - 3]


def _get_coeff_base_ctx(av1: AV1Decoder, txSz: TX_SIZE, plane: int, blockX: int, blockY: int, pos: int, c: int, isEob: int) -> int:
    """
    get_coeff_base_ctx
    规范文档 8.3.2: get_coeff_base_ctx
    """
    from utils.tile_utils import compute_tx_type
    tile_group = av1.tile_group

    adjTxSz = Adjusted_Tx_Size[txSz]
    bwl = Tx_Width_Log2[adjTxSz]
    width = 1 << bwl
    height = Tx_Height[adjTxSz]
    txType = compute_tx_type(av1, plane, txSz, blockX, blockY)

    if isEob:
        if c == 0:
            return SIG_COEF_CONTEXTS - 4
        if c <= (height << bwl) // 8:
            return SIG_COEF_CONTEXTS - 3
        if c <= (height << bwl) // 4:
            return SIG_COEF_CONTEXTS - 2
        return SIG_COEF_CONTEXTS - 1
    txClass = _get_tx_class(txType)
    row = pos >> bwl
    col = pos - (row << bwl)
    mag = 0

    for idx in range(SIG_REF_DIFF_OFFSET_NUM):
        refRow = row + Sig_Ref_Diff_Offset[txClass][idx][0]
        refCol = col + Sig_Ref_Diff_Offset[txClass][idx][1]
        if refRow >= 0 and refCol >= 0 and refRow < height and refCol < width:
            mag += min(abs(tile_group.Quant[(refRow << bwl) + refCol]), 3)

    ctx = min((mag + 1) >> 1, 4)
    if txClass == TX_CLASS_2D:
        if row == 0 and col == 0:
            return 0
        return ctx + Coeff_Base_Ctx_Offset[txSz][min(row, 4)][min(col, 4)]
    idx = row if txClass == TX_CLASS_VERT else col
    return ctx + Coeff_Base_Pos_Ctx_Offset[min(idx, 2)]


def coeff_base(av1: AV1Decoder, txSz: TX_SIZE, plane: int, x4: int, y4: int, scan: List[int], c: int, txSzCtx: int, ptype: int) -> List[int]:
    """
    coeff_base CDF选择
    规范文档 8.3.2: TileCoeffBaseCdf[txSzCtx][ptype][eobPt]
    """
    decoder = av1.decoder
    ctx = _get_coeff_base_ctx(av1, txSz, plane, x4, y4, scan[c], c, 0)
    return decoder.tile_cdfs['TileCoeffBaseCdf'][txSzCtx][ptype][ctx]


def coeff_base_eob(av1: AV1Decoder, txSz: TX_SIZE, plane: int, x4: int, y4: int, scan: List[int], c: int, txSzCtx: int, ptype: int) -> List[int]:
    """
    coeff_base_eob CDF选择
    规范文档 8.3.2: TileCoeffBaseEobCdf[txSzCtx][ptype][eobPt]
    """
    decoder = av1.decoder
    ctx = _get_coeff_base_ctx(
        av1, txSz, plane, x4, y4, scan[c], c, 1) - SIG_COEF_CONTEXTS + SIG_COEF_CONTEXTS_EOB
    return decoder.tile_cdfs['TileCoeffBaseEobCdf'][txSzCtx][ptype][ctx]


def dc_sign(av1: AV1Decoder, plane: int, w4: int, h4: int, x4: int, y4: int, ptype: int) -> List[int]:
    """
    dc_sign CDF选择
    规范文档 8.3.2: TileDcSignCdf[ptype][ctx]
    """
    seq_header = av1.seq_header
    frame_header = av1.frame_header
    tile_group = av1.tile_group
    decoder = av1.decoder
    subsampling_x = seq_header.color_config.subsampling_x
    subsampling_y = seq_header.color_config.subsampling_y
    MiRows = frame_header.MiRows
    MiCols = frame_header.MiCols

    maxX4 = MiCols
    maxY4 = MiRows
    if plane > 0:
        maxX4 = maxX4 >> subsampling_x
        maxY4 = maxY4 >> subsampling_y

    dcSign = 0
    for k in range(w4):
        if x4 + k < maxX4:
            sign = tile_group.AboveDcContext[plane][x4 + k]
            if sign == 1:
                dcSign -= 1
            elif sign == 2:
                dcSign += 1

    for k in range(h4):
        if y4 + k < maxY4:
            sign = tile_group.LeftDcContext[plane][y4 + k]
            if sign == 1:
                dcSign -= 1
            elif sign == 2:
                dcSign += 1

    if dcSign < 0:
        ctx = 1
    elif dcSign > 0:
        ctx = 2
    else:
        ctx = 0
    return decoder.tile_cdfs['TileDcSignCdf'][ptype][ctx]


def coeff_br(av1: AV1Decoder, pos: int, txSz: TX_SIZE, plane: int, x4: int, y4: int, ptype: int, txSzCtx: int) -> List[int]:
    """
    coeff_br CDF选择
    规范文档 8.3.2: TileCoeffBrCdf[Min(txSzCtx, TX_32X32)][ptype][ctx]
    """
    from utils.tile_utils import compute_tx_type
    tile_group = av1.tile_group
    decoder = av1.decoder

    adjTxSz = Adjusted_Tx_Size[txSz]
    bwl = Tx_Width_Log2[adjTxSz]
    txw = Tx_Width[adjTxSz]
    txh = Tx_Height[adjTxSz]
    row = pos >> bwl
    col = pos - (row << bwl)

    mag = 0

    txType = compute_tx_type(av1, plane, txSz, x4, y4)
    txClass = _get_tx_class(txType)

    for idx in range(3):
        refRow = row + Mag_Ref_Offset_With_Tx_Class[txClass][idx][0]
        refCol = col + Mag_Ref_Offset_With_Tx_Class[txClass][idx][1]
        if refRow >= 0 and refCol >= 0 and refRow < txh and refCol < (1 << bwl):
            quant = tile_group.Quant[refRow * txw + refCol]
            mag += min(quant, COEFF_BASE_RANGE + NUM_BASE_LEVELS + 1)

    mag = min((mag + 1) >> 1, 6)
    if pos == 0:
        ctx = mag
    elif txClass == 0:
        if row < 2 and col < 2:
            ctx = mag + 7
        else:
            ctx = mag + 14
    else:
        if txClass == 1:
            if col == 0:
                ctx = mag + 7
            else:
                ctx = mag + 14
        else:
            if row == 0:
                ctx = mag + 7
            else:
                ctx = mag + 14

    return decoder.tile_cdfs['TileCoeffBrCdf'][min(txSzCtx, TX_SIZE.TX_32X32)][ptype][ctx]


def has_palette_y(av1: AV1Decoder, bsizeCtx: int) -> List[int]:
    """
    has_palette_y CDF选择
    规范文档 8.3.2: TileHasPaletteYCdf
    """
    tile_group = av1.tile_group
    decoder = av1.decoder
    MiRow = tile_group.MiRow
    MiCol = tile_group.MiCol
    AvailU = tile_group.AvailU
    AvailL = tile_group.AvailL

    ctx = 0
    if AvailU and tile_group.PaletteSizes[0][MiRow - 1][MiCol] > 0:
        ctx += 1
    if AvailL and tile_group.PaletteSizes[0][MiRow][MiCol - 1] > 0:
        ctx += 1
    return decoder.tile_cdfs['TilePaletteYModeCdf'][bsizeCtx][ctx]


def has_palette_uv(av1: AV1Decoder) -> List[int]:
    """
    has_palette_uv CDF选择
    规范文档 8.3.2: TileHasPaletteUVCdf
    """
    tile_group = av1.tile_group
    decoder = av1.decoder

    ctx = 1 if tile_group.PaletteSizeY > 0 else 0
    return decoder.tile_cdfs['TilePaletteUVModeCdf'][ctx]


def palette_size_y_minus_2(av1: AV1Decoder, bsizeCtx: int) -> List[int]:
    """
    palette_size_y_minus_2 CDF选择
    规范文档 8.3.2: TilePaletteYSizeCdf[bsizeCtx]
    """
    decoder = av1.decoder

    return decoder.tile_cdfs['TilePaletteYSizeCdf'][bsizeCtx]


def palette_size_uv_minus_2(av1: AV1Decoder, bsizeCtx: int) -> List[int]:
    """
    palette_size_uv_minus_2 CDF选择
    规范文档 8.3.2: TilePaletteUVSizeCdf[bsizeCtx]
    """
    decoder = av1.decoder

    return decoder.tile_cdfs['TilePaletteUVSizeCdf'][bsizeCtx]


def palette_color_idx_y(av1: AV1Decoder) -> List[int]:
    """
    palette_color_idx_y CDF选择
    规范文档 8.3.2: TilePaletteSize{PaletteSizeY}YColorCdf[ctx]
    """
    tile_group = av1.tile_group
    decoder = av1.decoder
    PaletteSizeY = tile_group.PaletteSizeY

    ctx = Palette_Color_Context[tile_group.ColorContextHash]
    if PaletteSizeY == 2:
        return decoder.tile_cdfs['TilePaletteSize2YColorCdf'][ctx]
    elif PaletteSizeY == 3:
        return decoder.tile_cdfs['TilePaletteSize3YColorCdf'][ctx]
    elif PaletteSizeY == 4:
        return decoder.tile_cdfs['TilePaletteSize4YColorCdf'][ctx]
    elif PaletteSizeY == 5:
        return decoder.tile_cdfs['TilePaletteSize5YColorCdf'][ctx]
    elif PaletteSizeY == 6:
        return decoder.tile_cdfs['TilePaletteSize6YColorCdf'][ctx]
    elif PaletteSizeY == 7:
        return decoder.tile_cdfs['TilePaletteSize7YColorCdf'][ctx]
    elif PaletteSizeY == 8:
        return decoder.tile_cdfs['TilePaletteSize8YColorCdf'][ctx]
    assert False


def palette_color_idx_uv(av1: AV1Decoder) -> List[int]:
    """
    palette_color_idx_uv CDF选择
    规范文档 8.3.2: TilePaletteSize{PaletteSizeUV}UVColorCdf[ctx]
    """
    tile_group = av1.tile_group
    decoder = av1.decoder
    PaletteSizeUV = tile_group.PaletteSizeUV

    ctx = Palette_Color_Context[tile_group.ColorContextHash]
    if PaletteSizeUV == 2:
        return decoder.tile_cdfs['TilePaletteSize2UVColorCdf'][ctx]
    elif PaletteSizeUV == 3:
        return decoder.tile_cdfs['TilePaletteSize3UVColorCdf'][ctx]
    elif PaletteSizeUV == 4:
        return decoder.tile_cdfs['TilePaletteSize4UVColorCdf'][ctx]
    elif PaletteSizeUV == 5:
        return decoder.tile_cdfs['TilePaletteSize5UVColorCdf'][ctx]
    elif PaletteSizeUV == 6:
        return decoder.tile_cdfs['TilePaletteSize6UVColorCdf'][ctx]
    elif PaletteSizeUV == 7:
        return decoder.tile_cdfs['TilePaletteSize7UVColorCdf'][ctx]
    elif PaletteSizeUV == 8:
        return decoder.tile_cdfs['TilePaletteSize8UVColorCdf'][ctx]
    assert False


def delta_q_abs(av1: AV1Decoder) -> List[int]:
    """
    delta_q_abs CDF选择
    规范文档 8.3.2: TileDeltaQCdf
    """
    decoder = av1.decoder
    return decoder.tile_cdfs['TileDeltaQCdf']


def delta_lf_abs(av1: AV1Decoder, i: int) -> List[int]:
    """
    delta_lf_abs CDF选择
    规范文档 8.3.2: TileDeltaLFCdf 或 TileDeltaLFMultiCdf[i]
    """
    frame_header = av1.frame_header
    decoder = av1.decoder

    if frame_header.delta_lf_multi == 0:
        return decoder.tile_cdfs['TileDeltaLFCdf']
    else:
        return decoder.tile_cdfs['TileDeltaLFMultiCdf'][i]


def _count_refs(av1: AV1Decoder, frameType: REF_FRAME) -> int:
    """
    count_refs辅助函数
    规范文档 8.3.2: count_refs
    """
    tile_group = av1.tile_group
    AvailU = tile_group.AvailU
    AvailL = tile_group.AvailL

    c = 0
    if AvailU:
        if tile_group.AboveRefFrame[0] == frameType:
            c += 1
        if tile_group.AboveRefFrame[1] == frameType:
            c += 1
    if AvailL:
        if tile_group.LeftRefFrame[0] == frameType:
            c += 1
        if tile_group.LeftRefFrame[1] == frameType:
            c += 1
    return c


def _ref_count_ctx(counts0: int, counts1: int) -> int:
    """
    ref_count_ctx辅助函数
    规范文档 8.3.2: ref_count_ctx
    """
    if counts0 < counts1:
        return 0
    elif counts0 == counts1:
        return 1
    else:
        return 2


def _single_ref_p1_ctx(av1: AV1Decoder) -> int:
    """
    single_ref_p1_ctx辅助函数
    规范文档 8.3.2: single_ref_p1_ctx
    """
    fwdCount = _count_refs(av1, REF_FRAME.LAST_FRAME)
    fwdCount += _count_refs(av1, REF_FRAME.LAST2_FRAME)
    fwdCount += _count_refs(av1, REF_FRAME.LAST3_FRAME)
    fwdCount += _count_refs(av1, REF_FRAME.GOLDEN_FRAME)
    bwdCount = _count_refs(av1, REF_FRAME.BWDREF_FRAME)
    bwdCount += _count_refs(av1, REF_FRAME.ALTREF2_FRAME)
    bwdCount += _count_refs(av1, REF_FRAME.ALTREF_FRAME)
    return _ref_count_ctx(fwdCount, bwdCount)


def _comp_ref_p2_ctx(av1: AV1Decoder) -> int:
    """
    comp_ref_p2_ctx辅助函数
    规范文档 8.3.2: comp_ref_p2_ctx
    """
    last3Count = _count_refs(av1, REF_FRAME.LAST3_FRAME)
    goldCount = _count_refs(av1, REF_FRAME.GOLDEN_FRAME)
    return _ref_count_ctx(last3Count, goldCount)


def _is_samedir_ref_pair(ref0: int, ref1: int) -> bool:
    """
    is_samedir_ref_pair辅助函数
    规范文档 8.3.2: is_samedir_ref_pair
    """
    return (ref0 >= REF_FRAME.BWDREF_FRAME) == (ref1 >= REF_FRAME.BWDREF_FRAME)


def intra_tx_type(av1: AV1Decoder, set_val: int, txSz: TX_SIZE) -> List[int]:
    """
    intra_tx_type CDF选择
    规范文档 8.3.2: TileIntraTxTypeSet1Cdf或TileIntraTxTypeSet2Cdf
    """
    tile_group = av1.tile_group
    decoder = av1.decoder
    YMode = tile_group.YMode

    if tile_group.use_filter_intra == 1:
        intraDir = Filter_Intra_Mode_To_Intra_Dir[tile_group.filter_intra_mode]
    else:
        intraDir = YMode

    if set_val == TX_SET.TX_SET_INTRA_1:
        return decoder.tile_cdfs['TileIntraTxTypeSet1Cdf'][Tx_Size_Sqr[txSz]][intraDir]
    elif set_val == TX_SET.TX_SET_INTRA_2:
        return decoder.tile_cdfs['TileIntraTxTypeSet2Cdf'][Tx_Size_Sqr[txSz]][intraDir]
    assert False


def inter_tx_type(av1: AV1Decoder, set_val: int, txSz: int) -> List[int]:
    """
    inter_tx_type CDF选择
    规范文档 8.3.2: TileInterTxTypeSet1Cdf、TileInterTxTypeSet2Cdf或TileInterTxTypeSet3Cdf
    """
    decoder = av1.decoder

    if set_val == TX_SET.TX_SET_INTER_1:
        return decoder.tile_cdfs['TileInterTxTypeSet1Cdf'][Tx_Size_Sqr[txSz]]
    elif set_val == TX_SET.TX_SET_INTER_2:
        return decoder.tile_cdfs['TileInterTxTypeSet2Cdf']
    elif set_val == TX_SET.TX_SET_INTER_3:
        return decoder.tile_cdfs['TileInterTxTypeSet3Cdf'][Tx_Size_Sqr[txSz]]
    assert False


def comp_ref_type(av1: AV1Decoder) -> List[int]:
    """
    comp_ref_type CDF选择
    规范文档 8.3.2: TileCompRefTypeCdf[ctx]
    """
    tile_group = av1.tile_group
    decoder = av1.decoder
    AvailU = tile_group.AvailU
    AvailL = tile_group.AvailL

    above0 = tile_group.AboveRefFrame[0]
    above1 = tile_group.AboveRefFrame[1]
    left0 = tile_group.LeftRefFrame[0]
    left1 = tile_group.LeftRefFrame[1]
    aboveCompInter = AvailU and not tile_group.AboveIntra and not tile_group.AboveSingle
    leftCompInter = AvailL and not tile_group.LeftIntra and not tile_group.LeftSingle
    aboveUniComp = aboveCompInter and _is_samedir_ref_pair(above0, above1)
    leftUniComp = leftCompInter and _is_samedir_ref_pair(left0, left1)

    if AvailU and not tile_group.AboveIntra and AvailL and not tile_group.LeftIntra:
        samedir = _is_samedir_ref_pair(above0, left0)

        if not aboveCompInter and not leftCompInter:
            ctx = 1 + 2 * samedir
        elif not aboveCompInter:
            if not leftUniComp:
                ctx = 1
            else:
                ctx = 3 + samedir
        elif not leftCompInter:
            if not aboveUniComp:
                ctx = 1
            else:
                ctx = 3 + samedir
        else:
            if not aboveUniComp and not leftUniComp:
                ctx = 0
            elif not aboveUniComp or not leftUniComp:
                ctx = 2
            else:
                ctx = 3 + ((above0 == REF_FRAME.BWDREF_FRAME)
                           == (left0 == REF_FRAME.BWDREF_FRAME))
    elif AvailU and AvailL:
        if aboveCompInter:
            ctx = 1 + 2 * aboveUniComp
        elif leftCompInter:
            ctx = 1 + 2 * leftUniComp
        else:
            ctx = 2
    elif aboveCompInter:
        ctx = 4 * aboveUniComp
    elif leftCompInter:
        ctx = 4 * leftUniComp
    else:
        ctx = 2

    return decoder.tile_cdfs['TileCompRefTypeCdf'][ctx]


def uni_comp_ref(av1: AV1Decoder) -> List[int]:
    """
    uni_comp_ref CDF选择
    规范文档 8.3.2: TileUniCompRefCdf[ctx][0]
    """
    decoder = av1.decoder
    ctx = _single_ref_p1_ctx(av1)
    return decoder.tile_cdfs['TileUniCompRefCdf'][ctx][0]


def uni_comp_ref_p1(av1: AV1Decoder) -> List[int]:
    """
    uni_comp_ref_p1 CDF选择
    规范文档 8.3.2: TileUniCompRefCdf[ctx][1]
    """
    decoder = av1.decoder

    last2Count = _count_refs(av1, REF_FRAME.LAST2_FRAME)
    last3GoldCount = _count_refs(
        av1, REF_FRAME.LAST3_FRAME) + _count_refs(av1, REF_FRAME.GOLDEN_FRAME)
    ctx = _ref_count_ctx(last2Count, last3GoldCount)
    return decoder.tile_cdfs['TileUniCompRefCdf'][ctx][1]


def uni_comp_ref_p2(av1: AV1Decoder) -> List[int]:
    """
    uni_comp_ref_p2 CDF选择
    规范文档 8.3.2: TileUniCompRefCdf[ctx][2]
    """
    decoder = av1.decoder

    ctx = _comp_ref_p2_ctx(av1)
    return decoder.tile_cdfs['TileUniCompRefCdf'][ctx][2]


def comp_group_idx(av1: AV1Decoder) -> List[int]:
    """
    comp_group_idx CDF选择
    规范文档 8.3.2: TileCompGroupIdxCdf[ctx]
    """
    tile_group = av1.tile_group
    decoder = av1.decoder
    MiRow = tile_group.MiRow
    MiCol = tile_group.MiCol
    AvailU = tile_group.AvailU
    AvailL = tile_group.AvailL

    ctx = 0
    if AvailU:
        if not tile_group.AboveSingle:
            ctx += tile_group.CompGroupIdxs[MiRow - 1][MiCol]
        elif tile_group.AboveRefFrame[0] == REF_FRAME.ALTREF_FRAME:
            ctx += 3
    if AvailL:
        if not tile_group.LeftSingle:
            ctx += tile_group.CompGroupIdxs[MiRow][MiCol - 1]
        elif tile_group.LeftRefFrame[0] == REF_FRAME.ALTREF_FRAME:
            ctx += 3
    ctx = min(5, ctx)
    return decoder.tile_cdfs['TileCompGroupIdxCdf'][ctx]


def compound_idx(av1: AV1Decoder) -> List[int]:
    """
    compound_idx CDF选择
    规范文档 8.3.2: TileCompoundIdxCdf[ctx]
    """
    from utils.frame_utils import get_relative_dist
    frame_header = av1.frame_header
    tile_group = av1.tile_group
    decoder = av1.decoder
    OrderHints = frame_header.OrderHints
    MiRow = tile_group.MiRow
    MiCol = tile_group.MiCol
    AvailU = tile_group.AvailU
    AvailL = tile_group.AvailL

    fwd = abs(get_relative_dist(
        av1, OrderHints[tile_group.RefFrame[0]], frame_header.OrderHint))
    bck = abs(get_relative_dist(
        av1, OrderHints[tile_group.RefFrame[1]], frame_header.OrderHint))
    ctx = 3 if (fwd == bck) else 0
    if AvailU:
        if not tile_group.AboveSingle:
            ctx += tile_group.CompoundIdxs[MiRow - 1][MiCol]
        elif tile_group.AboveRefFrame[0] == REF_FRAME.ALTREF_FRAME:
            ctx += 1
    if AvailL:
        if not tile_group.LeftSingle:
            ctx += tile_group.CompoundIdxs[MiRow][MiCol - 1]
        elif tile_group.LeftRefFrame[0] == REF_FRAME.ALTREF_FRAME:
            ctx += 1

    return decoder.tile_cdfs['TileCompoundIdxCdf'][ctx]


def compound_type(av1: AV1Decoder) -> List[int]:
    """
    compound_type CDF选择
    规范文档 8.3.2: TileCompoundTypeCdf[MiSize]
    """
    tile_group = av1.tile_group
    decoder = av1.decoder
    MiSize = tile_group.MiSize

    return decoder.tile_cdfs['TileCompoundTypeCdf'][MiSize]


def interintra(av1: AV1Decoder) -> List[int]:
    """
    interintra CDF选择
    规范文档 8.3.2: TileInterIntraCdf[ctx]
    """
    tile_group = av1.tile_group
    decoder = av1.decoder
    MiSize = tile_group.MiSize

    ctx = Size_Group[MiSize] - 1
    return decoder.tile_cdfs['TileInterIntraCdf'][ctx]


def interintra_mode(av1: AV1Decoder) -> List[int]:
    """
    interintra_mode CDF选择
    规范文档 8.3.2: TileInterIntraModeCdf[ctx]
    """
    tile_group = av1.tile_group
    decoder = av1.decoder
    MiSize = tile_group.MiSize

    ctx = Size_Group[MiSize] - 1
    return decoder.tile_cdfs['TileInterIntraModeCdf'][ctx]


def wedge_index(av1: AV1Decoder) -> List[int]:
    """
    wedge_index CDF选择
    规范文档 8.3.2: TileWedgeIndexCdf[MiSize]
    """
    tile_group = av1.tile_group
    decoder = av1.decoder
    MiSize = tile_group.MiSize

    return decoder.tile_cdfs['TileWedgeIndexCdf'][MiSize]


def wedge_interintra(av1: AV1Decoder) -> List[int]:
    """
    wedge_interintra CDF选择
    规范文档 8.3.2: TileWedgeInterIntraCdf[MiSize]
    """
    tile_group = av1.tile_group
    decoder = av1.decoder
    MiSize = tile_group.MiSize

    return decoder.tile_cdfs['TileWedgeInterIntraCdf'][MiSize]


def use_obmc(av1: AV1Decoder) -> List[int]:
    """
    use_obmc CDF选择
    规范文档 8.3.2: TileUseObmcCdf[MiSize]
    """
    tile_group = av1.tile_group
    decoder = av1.decoder
    MiSize = tile_group.MiSize

    return decoder.tile_cdfs['TileUseObmcCdf'][MiSize]


def cfl_alpha_signs(av1: AV1Decoder) -> List[int]:
    """
    cfl_alpha_signs CDF选择
    规范文档 8.3.2: TileCflSignCdf
    """
    decoder = av1.decoder
    return decoder.tile_cdfs['TileCflSignCdf']


def cfl_alpha_u(av1: AV1Decoder, signU: int, signV: int) -> List[int]:
    """
    cfl_alpha_u CDF选择
    规范文档 8.3.2: TileCflAlphaCdf[ctx]
    """
    decoder = av1.decoder

    ctx = (signU - 1) * 3 + signV
    return decoder.tile_cdfs['TileCflAlphaCdf'][ctx]


def cfl_alpha_v(av1: AV1Decoder, signU: int, signV: int) -> List[int]:
    """
    cfl_alpha_v CDF选择
    规范文档 8.3.2: TileCflAlphaCdf[ctx]
    """
    decoder = av1.decoder

    ctx = (signV - 1) * 3 + signU
    return decoder.tile_cdfs['TileCflAlphaCdf'][ctx]


def use_wiener(av1: AV1Decoder) -> List[int]:
    """
    use_wiener CDF选择
    规范文档 8.3.2: TileUseWienerCdf
    """
    decoder = av1.decoder
    return decoder.tile_cdfs['TileUseWienerCdf']


def use_sgrproj(av1: AV1Decoder) -> List[int]:
    """
    use_sgrproj CDF选择
    规范文档 8.3.2: TileUseSgrprojCdf
    """
    decoder = av1.decoder
    return decoder.tile_cdfs['TileUseSgrprojCdf']


def restoration_type(av1: AV1Decoder) -> List[int]:
    """
    restoration_type CDF选择
    规范文档 8.3.2: TileRestorationTypeCdf
    """
    decoder = av1.decoder
    return decoder.tile_cdfs['TileRestorationTypeCdf']


syntax_element_functions = {
    'use_intrabc': use_intrabc,
    'intra_frame_y_mode': intra_frame_y_mode,
    'y_mode': y_mode,
    'uv_mode': uv_mode,
    'angle_delta_y': angle_delta_y,
    'angle_delta_uv': angle_delta_uv,
    'partition': partition,
    'split_or_horz': split_or_horz,
    'split_or_vert': split_or_vert,
    'tx_depth': tx_depth,
    'txfm_split': txfm_split,
    'segment_id': segment_id,
    'seg_id_predicted': seg_id_predicted,
    'new_mv': new_mv,
    'zero_mv': zero_mv,
    'ref_mv': ref_mv,
    'drl_mode': drl_mode,
    'filter_intra_mode': filter_intra_mode,
    'skip_mode': skip_mode,
    'skip': skip,
    'is_inter': is_inter,
    'use_filter_intra': use_filter_intra,
    'comp_mode': comp_mode,
    'comp_ref': comp_ref,
    'comp_ref_p1': comp_ref_p1,
    'comp_ref_p2': comp_ref_p2,
    'comp_bwdref': comp_bwdref,
    'comp_bwdref_p1': comp_bwdref_p1,
    'single_ref_p1': single_ref_p1,
    'single_ref_p2': single_ref_p2,
    'single_ref_p3': single_ref_p3,
    'single_ref_p4': single_ref_p4,
    'single_ref_p5': single_ref_p5,
    'single_ref_p6': single_ref_p6,
    'mv_joint': mv_joint,
    'mv_sign': mv_sign,
    'mv_class': mv_class,
    'mv_class0_bit': mv_class0_bit,
    'mv_class0_fr': mv_class0_fr,
    'mv_class0_hp': mv_class0_hp,
    'mv_fr': mv_fr,
    'mv_hp': mv_hp,
    'mv_bit': mv_bit,
    'motion_mode': motion_mode,
    'compound_mode': compound_mode,
    'interp_filter': interp_filter,
    'use_wiener': use_wiener,
    'use_sgrproj': use_sgrproj,
    'restoration_type': restoration_type,
    'cfl_alpha_signs': cfl_alpha_signs,
    'cfl_alpha_u': cfl_alpha_u,
    'cfl_alpha_v': cfl_alpha_v,
    'delta_q_abs': delta_q_abs,
    'delta_lf_abs': delta_lf_abs,
    'all_zero': all_zero,
    'eob_pt_16': eob_pt_16,
    'eob_pt_32': eob_pt_32,
    'eob_pt_64': eob_pt_64,
    'eob_pt_128': eob_pt_128,
    'eob_pt_256': eob_pt_256,
    'eob_pt_512': eob_pt_512,
    'eob_pt_1024': eob_pt_1024,
    'eob_extra': eob_extra,
    'coeff_base': coeff_base,
    'coeff_base_eob': coeff_base_eob,
    'coeff_br': coeff_br,
    'dc_sign': dc_sign,
    'has_palette_y': has_palette_y,
    'has_palette_uv': has_palette_uv,
    'palette_size_y_minus_2': palette_size_y_minus_2,
    'palette_size_uv_minus_2': palette_size_uv_minus_2,
    'palette_color_idx_y': palette_color_idx_y,
    'palette_color_idx_uv': palette_color_idx_uv,
    'intra_tx_type': intra_tx_type,
    'inter_tx_type': inter_tx_type,
    'comp_ref_type': comp_ref_type,
    'uni_comp_ref': uni_comp_ref,
    'uni_comp_ref_p1': uni_comp_ref_p1,
    'uni_comp_ref_p2': uni_comp_ref_p2,
    'comp_group_idx': comp_group_idx,
    'compound_idx': compound_idx,
    'compound_type': compound_type,
    'interintra': interintra,
    'interintra_mode': interintra_mode,
    'wedge_index': wedge_index,
    'wedge_interintra': wedge_interintra,
    'use_obmc': use_obmc,
}


def cdf_selection_process(av1: AV1Decoder,
                          syntax_element_name: str,
                          *args, **kwargs) -> List[int]:
    """
    CDF选择过程
    规范文档 8.3.2 Cdf selection process

    输入是语法元素的名称，输出是对CDF数组的引用。
    当描述使用变量时，这些变量取自语法表在解码语法元素时定义的值。

    Args:
        syntax_element_name: 语法元素名称（如 "use_intrabc", "y_mode" 等）
        data: 额外的数据字典（用于需要参数的函数，如partition）

    Returns:
        选择的CDF数组引用
    """
    func = syntax_element_functions[syntax_element_name]
    assert func
    return func(av1, *args, **kwargs)
