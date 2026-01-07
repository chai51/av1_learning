from constants import ADST_ADST, ADST_DCT, DCT_ADST, DCT_DCT, MI_SIZE, REF_FRAME, REF_SCALE_SHIFT, SUB_SIZE, TX_SET, TX_SIZE, TX_SIZES_ALL, Y_MODE, Num_4x4_Blocks_High, Num_4x4_Blocks_Wide, Tx_Height, Tx_Size_Sqr, Tx_Size_Sqr_Up, Tx_Width
from obu.decoder import AV1Decoder
from utils.math_utils import Clip3


def seg_feature_active_idx(av1: AV1Decoder, idx: int, feature: int) -> int:
    """
    规范文档 5.11.14 Segmentation feature active function

    Args:
        idx: Segment索引
        feature: 特征类型（SEG_LVL_*）
    Returns:
        如果特征激活返回1，否则返回0
    """
    frame_header = av1.frame_header
    return frame_header.segmentation_enabled and frame_header.FeatureEnabled[idx][feature]


def is_scaled(av1: AV1Decoder, refFrame: int) -> bool:
    """
    检查参考帧是否缩放
    规范文档 5.11.27
    """
    frame_header = av1.frame_header
    ref_frame_store = av1.ref_frame_store

    refIdx = frame_header.ref_frame_idx[refFrame - REF_FRAME.LAST_FRAME]
    xScale = ((ref_frame_store.RefUpscaledWidth[refIdx] << REF_SCALE_SHIFT) + (
        frame_header.FrameWidth // 2)) // frame_header.FrameWidth
    yScale = ((ref_frame_store.RefFrameHeight[refIdx] << REF_SCALE_SHIFT) + (
        frame_header.FrameHeight // 2)) // frame_header.FrameHeight
    noScale = 1 << REF_SCALE_SHIFT
    return xScale != noScale or yScale != noScale


def find_tx_size(w: int, h: int) -> TX_SIZE:
    """
    查找匹配的变换尺寸
    规范文档 5.11.36 find_tx_size( w, h 
    """
    txSz = 0
    while txSz < TX_SIZES_ALL:
        if Tx_Width[txSz] == w and Tx_Height[txSz] == h:
            break
        txSz += 1
    return TX_SIZE(txSz)


def get_plane_residual_size(av1: AV1Decoder, subsize: int, plane: int) -> SUB_SIZE:
    """
    获取平面残差尺寸
    规范文档 5.11.38 Get plane residual size function
    """
    seq_header = av1.seq_header

    subx = seq_header.color_config.subsampling_x if plane > 0 else 0
    suby = seq_header.color_config.subsampling_y if plane > 0 else 0
    return Subsampled_Size[subsize][subx][suby]


def compute_tx_type(av1: AV1Decoder, plane: int, txSz: TX_SIZE, blockX: int, blockY: int) -> int:
    """
    计算变换类型
    规范文档 5.11.40 Compute transform type function
    Args:
        plane: 平面索引
        txSz: 变换尺寸
        blockX: 块X坐标（4x4单位）
        blockY: 块Y坐标（4x4单位）  
    Returns:
        变换类型
    """
    seq_header = av1.seq_header
    tile_group = av1.tile_group
    subsampling_x = seq_header.color_config.subsampling_x
    subsampling_y = seq_header.color_config.subsampling_y
    MiRow = tile_group.MiRow
    MiCol = tile_group.MiCol

    def is_tx_type_in_set(txSet: int, txType: int) -> int:
        """
        检查变换类型是否在集合中
        规范文档 5.11.40

        Args:
            txSet: 变换集合
            txType: 变换类型

        Returns:
            是否在集合中
        """
        return Tx_Type_In_Set_Inter[txSet][txType] if tile_group.is_inter else Tx_Type_In_Set_Intra[txSet][txType]

    txSzSqrUp = Tx_Size_Sqr_Up[txSz]
    if tile_group.Lossless or txSzSqrUp > TX_SIZE.TX_32X32:
        return DCT_DCT

    txSet = get_tx_set(av1, txSz)

    if plane == 0:
        return tile_group.TxTypes[blockY][blockX]

    if tile_group.is_inter:
        x4 = max(MiCol, blockX << subsampling_x)
        y4 = max(MiRow, blockY << subsampling_y)
        txType = tile_group.TxTypes[y4][x4]
        if not is_tx_type_in_set(txSet, txType):
            return DCT_DCT
        return txType

    txType = Mode_To_Txfm[tile_group.UVMode]

    if not is_tx_type_in_set(txSet, txType):
        return DCT_DCT
    return txType


def is_directional_mode(mode: Y_MODE) -> int:
    """
    检查是否为方向模式
    规范文档 5.11.44 Is directional mode function
    """
    if (mode >= Y_MODE.V_PRED) and (mode <= Y_MODE.D67_PRED):
        return 1
    return 0


def get_tx_set(av1: AV1Decoder, txSz: int) -> int:
    """
    获取变换集合
    规范文档 5.11.48 Get transform set function
    Args:
        txSz: 变换尺寸
    Returns:
        变换集合类型
    """
    frame_header = av1.frame_header
    tile_group = av1.tile_group

    txSzSqr = Tx_Size_Sqr[txSz]
    txSzSqrUp = Tx_Size_Sqr_Up[txSz]
    if txSzSqrUp > TX_SIZE.TX_32X32:
        return TX_SET.TX_SET_DCTONLY

    if tile_group.is_inter:
        if frame_header.reduced_tx_set or txSzSqrUp == TX_SIZE.TX_32X32:
            return TX_SET.TX_SET_INTER_3
        elif txSzSqr == TX_SIZE.TX_16X16:
            return TX_SET.TX_SET_INTER_2
        return TX_SET.TX_SET_INTER_1
    else:
        if txSzSqrUp == TX_SIZE.TX_32X32:
            return TX_SET.TX_SET_DCTONLY
        elif frame_header.reduced_tx_set:
            return TX_SET.TX_SET_INTRA_2
        elif txSzSqr == TX_SIZE.TX_16X16:
            return TX_SET.TX_SET_INTRA_2
        return TX_SET.TX_SET_INTRA_1


def is_inside(av1: AV1Decoder, candidateR: int, candidateC: int) -> int:
    """
    检查候选位置是否在当前Tile内
    规范文档 5.11.51 Is inside function

    Args:
        candidateR: 候选行位置
        candidateC: 候选列位置

    Returns:
        是否在Tile内
    """
    tile_group = av1.tile_group

    return (candidateC >= tile_group.MiColStart and
            candidateC < tile_group.MiColEnd and
            candidateR >= tile_group.MiRowStart and
            candidateR < tile_group.MiRowEnd)


def is_inside_filter_region(av1: AV1Decoder, candidateR: int, candidateC: int) -> int:
    """
    检查候选位置是否在滤波区域内
    规范文档 5.11.52 Is inside filter region function

    Args:
        candidateR: 候选行位置
        candidateC: 候选列位置

    Returns:
        是否在滤波区域内
    """
    frame_header = av1.frame_header

    colStart = 0
    colEnd = frame_header.MiCols
    rowStart = 0
    rowEnd = frame_header.MiRows

    return (candidateC >= colStart and
            candidateC < colEnd and
            candidateR >= rowStart and
            candidateR < rowEnd)


def clamp_mv_row(av1: AV1Decoder, mvec: int, border: int) -> int:
    """
    裁剪MV行分量
    规范文档 5.11.53 Clamp MV row function
    Args:
        mvec: MV行分量
        border: 边界值
    Returns:
        裁剪后的MV行分量
    """
    frame_header = av1.frame_header
    tile_group = av1.tile_group
    MiSize = tile_group.MiSize
    MiRows = frame_header.MiRows
    MiRow = tile_group.MiRow

    bh4 = Num_4x4_Blocks_High[MiSize]
    mbToTopEdge = -((MiRow * MI_SIZE) * 8)
    mbToBottomEdge = ((MiRows - bh4 - MiRow) * MI_SIZE) * 8
    return Clip3(mbToTopEdge - border, mbToBottomEdge + border, mvec)


def clamp_mv_col(av1: AV1Decoder, mvec: int, border: int) -> int:
    """
    裁剪MV列分量
    规范文档 5.11.54 Clamp MV col function
    Args:
        mvec: MV列分量
        border: 边界值
    Returns:
        裁剪后的MV列分量
    """
    frame_header = av1.frame_header
    tile_group = av1.tile_group
    MiSize = tile_group.MiSize
    MiCol = tile_group.MiCol
    MiCols = frame_header.MiCols

    bw4 = Num_4x4_Blocks_Wide[MiSize]
    mbToLeftEdge = -((MiCol * MI_SIZE) * 8)
    mbToRightEdge = ((MiCols - bw4 - MiCol) * MI_SIZE) * 8
    return Clip3(mbToLeftEdge - border, mbToRightEdge + border, mvec)


def count_units_in_frame(unitSize: int, frameSize: int) -> int:
    """
    计算帧中的单元数量
    规范文档 5.11.57
    Args:
        unitSize: 单元尺寸
        frameSize: 帧尺寸
    Returns:
        帧中的单元数量
    """
    return max((frameSize + (unitSize >> 1)) // unitSize, 1)


"""
规范文档 5.11.38
"""
Subsampled_Size = [
    [
        [SUB_SIZE.BLOCK_4X4, SUB_SIZE.BLOCK_4X4],
        [SUB_SIZE.BLOCK_4X4, SUB_SIZE.BLOCK_4X4],
    ],
    [
        [SUB_SIZE.BLOCK_4X8, SUB_SIZE.BLOCK_4X4],
        [SUB_SIZE.BLOCK_INVALID, SUB_SIZE.BLOCK_4X4],
    ],
    [
        [SUB_SIZE.BLOCK_8X4, SUB_SIZE.BLOCK_INVALID],
        [SUB_SIZE.BLOCK_4X4, SUB_SIZE.BLOCK_4X4],
    ],
    [
        [SUB_SIZE.BLOCK_8X8, SUB_SIZE.BLOCK_8X4],
        [SUB_SIZE.BLOCK_4X8, SUB_SIZE.BLOCK_4X4],
    ],
    [
        [SUB_SIZE.BLOCK_8X16, SUB_SIZE.BLOCK_8X8],
        [SUB_SIZE.BLOCK_INVALID, SUB_SIZE.BLOCK_4X8],
    ],
    [
        [SUB_SIZE.BLOCK_16X8, SUB_SIZE.BLOCK_INVALID],
        [SUB_SIZE.BLOCK_8X8, SUB_SIZE.BLOCK_8X4],
    ],
    [
        [SUB_SIZE.BLOCK_16X16, SUB_SIZE.BLOCK_16X8],
        [SUB_SIZE.BLOCK_8X16, SUB_SIZE.BLOCK_8X8],
    ],
    [
        [SUB_SIZE.BLOCK_16X32, SUB_SIZE.BLOCK_16X16],
        [SUB_SIZE.BLOCK_INVALID, SUB_SIZE.BLOCK_8X16],
    ],
    [
        [SUB_SIZE.BLOCK_32X16, SUB_SIZE.BLOCK_INVALID],
        [SUB_SIZE.BLOCK_16X16, SUB_SIZE.BLOCK_16X8],
    ],
    [
        [SUB_SIZE.BLOCK_32X32, SUB_SIZE.BLOCK_32X16],
        [SUB_SIZE.BLOCK_16X32, SUB_SIZE.BLOCK_16X16],
    ],
    [
        [SUB_SIZE.BLOCK_32X64, SUB_SIZE.BLOCK_32X32],
        [SUB_SIZE.BLOCK_INVALID, SUB_SIZE.BLOCK_16X32],
    ],
    [
        [SUB_SIZE.BLOCK_64X32, SUB_SIZE.BLOCK_INVALID],
        [SUB_SIZE.BLOCK_32X32, SUB_SIZE.BLOCK_32X16],
    ],
    [
        [SUB_SIZE.BLOCK_64X64, SUB_SIZE.BLOCK_64X32],
        [SUB_SIZE.BLOCK_32X64, SUB_SIZE.BLOCK_32X32],
    ],
    [
        [SUB_SIZE.BLOCK_64X128, SUB_SIZE.BLOCK_64X64],
        [SUB_SIZE.BLOCK_INVALID, SUB_SIZE.BLOCK_32X64],
    ],
    [
        [SUB_SIZE.BLOCK_128X64, SUB_SIZE.BLOCK_INVALID],
        [SUB_SIZE.BLOCK_64X64, SUB_SIZE.BLOCK_64X32],
    ],
    [
        [SUB_SIZE.BLOCK_128X128, SUB_SIZE.BLOCK_128X64],
        [SUB_SIZE.BLOCK_64X128, SUB_SIZE.BLOCK_64X64],
    ],
    [
        [SUB_SIZE.BLOCK_4X16, SUB_SIZE.BLOCK_4X8],
        [SUB_SIZE.BLOCK_INVALID, SUB_SIZE.BLOCK_4X8],
    ],
    [
        [SUB_SIZE.BLOCK_16X4, SUB_SIZE.BLOCK_INVALID],
        [SUB_SIZE.BLOCK_8X4, SUB_SIZE.BLOCK_8X4],
    ],
    [
        [SUB_SIZE.BLOCK_8X32, SUB_SIZE.BLOCK_8X16],
        [SUB_SIZE.BLOCK_INVALID, SUB_SIZE.BLOCK_4X16],
    ],
    [
        [SUB_SIZE.BLOCK_32X8, SUB_SIZE.BLOCK_INVALID],
        [SUB_SIZE.BLOCK_16X8, SUB_SIZE.BLOCK_16X4],
    ],
    [
        [SUB_SIZE.BLOCK_16X64, SUB_SIZE.BLOCK_16X32],
        [SUB_SIZE.BLOCK_INVALID, SUB_SIZE.BLOCK_8X32],
    ],
    [
        [SUB_SIZE.BLOCK_64X16, SUB_SIZE.BLOCK_INVALID],
        [SUB_SIZE.BLOCK_32X16, SUB_SIZE.BLOCK_32X8],
    ],
]


"""
规范文档 5.11.40
"""
Tx_Type_In_Set_Intra = [
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0],
    [1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
]
Tx_Type_In_Set_Inter = [
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
]


"""
规范文档 9.3
"""
Mode_To_Txfm = [
    DCT_DCT,  # DC_PRED
    ADST_DCT,  # V_PRED
    DCT_ADST,  # H_PRED
    DCT_DCT,  # D45_PRED
    ADST_ADST,  # D135_PRED
    ADST_DCT,  # D113_PRED
    DCT_ADST,  # D157_PRED
    DCT_ADST,  # D203_PRED
    ADST_DCT,  # D67_PRED
    ADST_ADST,  # SMOOTH_PRED
    ADST_DCT,  # SMOOTH_V_PRED
    DCT_ADST,  # SMOOTH_H_PRED
    ADST_ADST,  # PAETH_PRED
    DCT_DCT,  # UV_CFL_PRED
]
