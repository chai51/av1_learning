"""
变换尺寸解析
实现read_block_tx_size、read_var_tx_size、read_tx_size等函数
"""

from entropy.symbol_decoder import SymbolDecoder, read_symbol
from constants import (
    TX_MODE_SELECT, TX_4X4, BLOCK_4X4, MI_SIZE,
    INTRA_FRAME
)
from residual.residual import Tx_Width, Tx_Height, Num_4x4_Blocks_Wide, Num_4x4_Blocks_High


# 查找表（规范文档定义）
# Max_Tx_Size_Rect[BLOCK_SIZES_ALL] - 最大变换尺寸
Max_Tx_Size_Rect = [
    TX_4X4, TX_4X4, TX_8X8, TX_8X8, TX_16X16, TX_16X16, TX_32X32, TX_32X32,  # 方形块
    TX_4X4, TX_4X4, TX_8X8, TX_8X8, TX_16X16, TX_16X16,  # 矩形块
    TX_4X4, TX_4X4, TX_8X8, TX_8X8, TX_16X16, TX_16X16,  # 更细分的矩形块
    TX_4X4, TX_4X4, TX_8X8, TX_8X8,  # 超细矩形块
]

# Split_Tx_Size[TX_SIZES_ALL] - 分裂后的变换尺寸
Split_Tx_Size = [
    TX_4X4,  # TX_4X4不能再分裂
    TX_4X4,  # TX_8X8 -> TX_4X4
    TX_8X8,  # TX_16X16 -> TX_8X8
    TX_16X16,  # TX_32X32 -> TX_16X16
    TX_32X32,  # TX_64X64 -> TX_32X32
    TX_4X4,  # TX_4X8 -> TX_4X4
    TX_4X4,  # TX_8X4 -> TX_4X4
    TX_8X8,  # TX_8X16 -> TX_8X8
    TX_8X8,  # TX_16X8 -> TX_8X8
    TX_16X16,  # TX_16X32 -> TX_16X16
    TX_16X16,  # TX_32X16 -> TX_16X16
    TX_32X32,  # TX_32X64 -> TX_32X32
    TX_32X32,  # TX_64X32 -> TX_32X32
    TX_4X4,  # TX_4X16 -> TX_4X4
    TX_4X4,  # TX_16X4 -> TX_4X4
    TX_8X8,  # TX_8X32 -> TX_8X8
    TX_8X8,  # TX_32X8 -> TX_8X8
    TX_16X16,  # TX_16X64 -> TX_16X16
    TX_16X16,  # TX_64X16 -> TX_16X16
]

# MAX_VARTX_DEPTH常量
MAX_VARTX_DEPTH = 2

# Max_Tx_Depth查找表（规范文档定义）
Max_Tx_Depth = [
    0, 1, 1, 1,  # BLOCK_4X4, BLOCK_4X8, BLOCK_8X4, BLOCK_8X8
    2, 2, 2, 3,  # BLOCK_8X16, BLOCK_16X8, BLOCK_16X16, BLOCK_16X32
    3, 3, 4, 4,  # BLOCK_32X16, BLOCK_32X32, BLOCK_32X64, BLOCK_64X32
    4, 4, 4, 4,  # BLOCK_64X64, BLOCK_64X128, BLOCK_128X64, BLOCK_128X128
    2, 2, 3, 3,  # BLOCK_4X16, BLOCK_16X4, BLOCK_8X32, BLOCK_32X8
    4, 4  # BLOCK_16X64, BLOCK_64X16
]


def read_tx_size(decoder: SymbolDecoder, mode_info_obj,
                 TxMode: int, Lossless: bool,
                 allowSelect: bool = True) -> int:
    """
    读取变换尺寸
    规范文档 6.11.33 read_tx_size()
    
    Args:
        decoder: SymbolDecoder实例
        mode_info_obj: ModeInfo对象
        TxMode: 变换模式
        Lossless: 是否为无损模式
        allowSelect: 是否允许选择（从参数传递）
        
    Returns:
        变换尺寸（TxSize）
    """
    MiSize = mode_info_obj.MiSize
    
    if Lossless:
        TxSize = TX_4X4
        return TxSize
    
    maxRectTxSize = Max_Tx_Size_Rect[MiSize] if MiSize < len(Max_Tx_Size_Rect) else TX_4X4
    maxTxDepth = Max_Tx_Depth[MiSize] if MiSize < len(Max_Tx_Depth) else 0
    
    TxSize = maxRectTxSize
    
    if MiSize > BLOCK_4X4 and allowSelect and TxMode == TX_MODE_SELECT:
        # tx_depth (S())
        # 简化处理：使用均匀CDF，实际应该根据上下文选择CDF
        cdf_size = min(maxTxDepth + 1, MAX_VARTX_DEPTH + 1)
        cdf = [1 << 14] * cdf_size + [1 << 15, 0]  # 简化CDF
        tx_depth = read_symbol(decoder, cdf)
        
        # for (i = 0; i < tx_depth; i++)
        #     TxSize = Split_Tx_Size[TxSize]
        for i in range(tx_depth):
            if TxSize < len(Split_Tx_Size):
                TxSize = Split_Tx_Size[TxSize]
            else:
                break
    
    return TxSize


def read_var_tx_size(decoder: SymbolDecoder,
                     row: int, col: int, txSz: int, depth: int,
                     MiRows: int, MiCols: int) -> int:
    """
    读取可变变换尺寸树
    规范文档 6.11.35 read_var_tx_size()
    
    Args:
        decoder: SymbolDecoder实例
        row: Mi行位置
        col: Mi列位置
        txSz: 当前变换尺寸
        depth: 当前深度
        MiRows: Mi总行数
        MiCols: Mi总列数
        
    Returns:
        变换尺寸
    """
    if row >= MiRows or col >= MiCols:
        return TX_4X4
    
    if txSz == TX_4X4 or depth == MAX_VARTX_DEPTH:
        txfm_split = 0
    else:
        # txfm_split (S())
        cdf = [1 << 14, 1 << 15, 0]  # 简化CDF
        txfm_split = read_symbol(decoder, cdf)
    
    w4 = Tx_Width[txSz] // MI_SIZE if txSz < len(Tx_Width) else 1
    h4 = Tx_Height[txSz] // MI_SIZE if txSz < len(Tx_Height) else 1
    
    if txfm_split:
        subTxSz = Split_Tx_Size[txSz] if txSz < len(Split_Tx_Size) else TX_4X4
        stepW = Tx_Width[subTxSz] // MI_SIZE if subTxSz < len(Tx_Width) else 1
        stepH = Tx_Height[subTxSz] // MI_SIZE if subTxSz < len(Tx_Height) else 1
        
        # 递归读取子块
        # for (i = 0; i < h4; i += stepH)
        #     for (j = 0; j < w4; j += stepW)
        #         read_var_tx_size(row + i, col + j, subTxSz, depth + 1)
        # 简化处理：只返回一个值
        result = read_var_tx_size(decoder, row, col, subTxSz, depth + 1, MiRows, MiCols)
        return result
    else:
        # 不分裂，返回当前尺寸
        return txSz


def read_block_tx_size(decoder: SymbolDecoder,
                       mode_info_obj,
                       MiRow: int, MiCol: int,
                       seq_header, frame_header,
                       TxMode: int = TX_MODE_SELECT,
                       Lossless: bool = False,
                       MiRows: int = 0, MiCols: int = 0) -> int:
    """
    读取块的变换尺寸
    规范文档 6.11.34 read_block_tx_size()
    
    Args:
        decoder: SymbolDecoder实例
        mode_info_obj: ModeInfo对象
        MiRow: Mi行位置
        MiCol: Mi列位置
        seq_header: 序列头
        frame_header: 帧头
        TxMode: 变换模式
        Lossless: 是否为无损模式
        MiRows: Mi总行数
        MiCols: Mi总列数
        
    Returns:
        变换尺寸（TxSize）
    """
    MiSize = mode_info_obj.MiSize
    is_inter = mode_info_obj.is_inter
    skip = mode_info_obj.skip
    
    bw4 = Num_4x4_Blocks_Wide[MiSize] if MiSize < len(Num_4x4_Blocks_Wide) else 1
    bh4 = Num_4x4_Blocks_High[MiSize] if MiSize < len(Num_4x4_Blocks_High) else 1
    
    if (TxMode == TX_MODE_SELECT and
        MiSize > BLOCK_4X4 and
        is_inter and
        not skip and
        not Lossless):
        # 可变变换尺寸
        maxTxSz = Max_Tx_Size_Rect[MiSize] if MiSize < len(Max_Tx_Size_Rect) else TX_4X4
        txW4 = Tx_Width[maxTxSz] // MI_SIZE if maxTxSz < len(Tx_Width) else 1
        txH4 = Tx_Height[maxTxSz] // MI_SIZE if maxTxSz < len(Tx_Height) else 1
        
        # for (row = MiRow; row < MiRow + bh4; row += txH4)
        #     for (col = MiCol; col < MiCol + bw4; col += txW4)
        #         read_var_tx_size(row, col, maxTxSz, 0)
        # 简化处理：只读取第一个块
        TxSize = read_var_tx_size(decoder, MiRow, MiCol, maxTxSz, 0, MiRows, MiCols)
    else:
        # 固定变换尺寸
        allowSelect = not skip or not is_inter
        TxSize = read_tx_size(decoder, mode_info_obj, TxMode, Lossless, allowSelect)
        
        # 更新InterTxSizes数组（简化处理）
        # for (row = MiRow; row < MiRow + bh4; row++)
        #     for (col = MiCol; col < MiCol + bw4; col++)
        #         InterTxSizes[row][col] = TxSize
    
    return TxSize

