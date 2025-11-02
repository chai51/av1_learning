"""
变换相关工具函数
实现get_scan、transform_type、compute_tx_type等函数
"""

from entropy.symbol_decoder import SymbolDecoder, read_symbol
from constants import (
    TX_4X4, TX_8X8, TX_16X16, TX_32X32, TX_64X64,
    DCT_DCT, IDTX, V_DCT, H_DCT, V_ADST, H_ADST, V_FLIPADST, H_FLIPADST,
    DC_PRED
)


# TX_SET常量（规范文档定义）
TX_SET_DCTONLY = 0
TX_SET_INTRA_1 = 1
TX_SET_INTRA_2 = 2
TX_SET_INTER_1 = 3
TX_SET_INTER_2 = 4
TX_SET_INTER_3 = 5

# Tx_Size_Sqr查找表（规范文档定义）
# 返回最接近的方形变换尺寸
Tx_Size_Sqr = [
    TX_4X4, TX_8X8, TX_16X16, TX_32X32, TX_64X64,  # TX_4X4-TX_64X64: 方形，返回自身
    TX_4X4, TX_4X4,  # TX_4X8, TX_8X4: 4x4
    TX_8X8, TX_8X8,  # TX_8X16, TX_16X8: 8x8
    TX_16X16, TX_16X16,  # TX_16X32, TX_32X16: 16x16
    TX_32X32, TX_32X32,  # TX_32X64, TX_64X32: 32x32
    TX_4X4, TX_4X4,  # TX_4X16, TX_16X4: 4x4
    TX_8X8, TX_8X8,  # TX_8X32, TX_32X8: 8x8
    TX_16X16, TX_16X16  # TX_16X64, TX_64X16: 16x16
]

# Tx_Size_Sqr_Up查找表（规范文档定义）
# 返回向上取整的方形变换尺寸
Tx_Size_Sqr_Up = [
    TX_4X4, TX_8X8, TX_16X16, TX_32X32, TX_64X64,  # TX_4X4-TX_64X64: 方形，返回自身
    TX_8X8, TX_8X8,  # TX_4X8, TX_8X4: 8x8
    TX_16X16, TX_16X16,  # TX_8X16, TX_16X8: 16x16
    TX_32X32, TX_32X32,  # TX_16X32, TX_32X16: 32x32
    TX_64X64, TX_64X64,  # TX_32X64, TX_64X32: 64x64
    TX_16X16, TX_16X16,  # TX_4X16, TX_16X4: 16x16
    TX_32X32, TX_32X32,  # TX_8X32, TX_32X8: 32x32
    TX_32X32, TX_32X32  # TX_16X64, TX_64X16: 32x32
]

# 变换类型查找表（规范文档定义）
Tx_Type_Intra_Inv_Set1 = [IDTX, DCT_DCT, V_DCT, H_DCT, ADST_ADST, ADST_DCT, DCT_ADST]
Tx_Type_Intra_Inv_Set2 = [IDTX, DCT_DCT, ADST_ADST, ADST_DCT, DCT_ADST]
Tx_Type_Inter_Inv_Set1 = [
    IDTX, V_DCT, H_DCT, V_ADST, H_ADST, V_FLIPADST, H_FLIPADST,
    DCT_DCT, ADST_DCT, DCT_ADST, 0, 0, ADST_ADST, 0, 0, 0
]  # FLIPADST_DCT等需要添加常量
Tx_Type_Inter_Inv_Set2 = [
    IDTX, V_DCT, H_DCT, DCT_DCT, ADST_DCT, DCT_ADST, 0, 0,
    ADST_ADST, 0, 0, 0
]
Tx_Type_Inter_Inv_Set3 = [IDTX, DCT_DCT]


def get_tx_set(txSz: int, is_inter: bool, reduced_tx_set: bool = False) -> int:
    """
    获取变换集合
    规范文档 6.11.39 get_tx_set()
    
    Args:
        txSz: 变换尺寸
        is_inter: 是否为帧间
        reduced_tx_set: 是否为reduced变换集合
        
    Returns:
        变换集合类型
    """
    txSzSqr = Tx_Size_Sqr[txSz] if txSz < len(Tx_Size_Sqr) else TX_4X4
    txSzSqrUp = Tx_Size_Sqr_Up[txSz] if txSz < len(Tx_Size_Sqr_Up) else TX_4X4
    
    if txSzSqrUp > TX_32X32:
        return TX_SET_DCTONLY
    
    if is_inter:
        if reduced_tx_set or txSzSqrUp == TX_32X32:
            return TX_SET_INTER_3
        elif txSzSqr == TX_16X16:
            return TX_SET_INTER_2
        else:
            return TX_SET_INTER_1
    else:
        if txSzSqrUp == TX_32X32:
            return TX_SET_DCTONLY
        elif reduced_tx_set:
            return TX_SET_INTRA_2
        elif txSzSqr == TX_16X16:
            return TX_SET_INTRA_2
        else:
            return TX_SET_INTRA_1


def transform_type(decoder: SymbolDecoder,
                  x4: int, y4: int, txSz: int,
                  is_inter: bool,
                  segment_id: int = 0,
                  segmentation_enabled: bool = False,
                  base_q_idx: int = 0) -> int:
    """
    解析变换类型
    规范文档 6.11.40 transform_type()
    
    Args:
        decoder: SymbolDecoder实例
        x4: X坐标（4x4块单位）
        y4: Y坐标（4x4块单位）
        txSz: 变换尺寸
        is_inter: 是否为帧间
        segment_id: Segment ID
        segmentation_enabled: 是否启用segmentation
        base_q_idx: 基础量化索引
        
    Returns:
        变换类型（TxType）
    """
    set_val = get_tx_set(txSz, is_inter)
    
    # get_qindex计算（简化处理）
    qindex = base_q_idx if not segmentation_enabled else base_q_idx  # 简化处理
    
    if set_val > 0 and qindex > 0:
        if is_inter:
            # inter_tx_type (S())
            if set_val == TX_SET_INTER_1:
                cdf = [1 << 14] * 16 + [1 << 15, 0]  # 简化CDF
                inter_tx_type = read_symbol(decoder, cdf)
                TxType = Tx_Type_Inter_Inv_Set1[inter_tx_type] if inter_tx_type < len(Tx_Type_Inter_Inv_Set1) else DCT_DCT
            elif set_val == TX_SET_INTER_2:
                cdf = [1 << 14] * 12 + [1 << 15, 0]  # 简化CDF
                inter_tx_type = read_symbol(decoder, cdf)
                TxType = Tx_Type_Inter_Inv_Set2[inter_tx_type] if inter_tx_type < len(Tx_Type_Inter_Inv_Set2) else DCT_DCT
            else:  # TX_SET_INTER_3
                cdf = [1 << 14] * 2 + [1 << 15, 0]  # 简化CDF
                inter_tx_type = read_symbol(decoder, cdf)
                TxType = Tx_Type_Inter_Inv_Set3[inter_tx_type] if inter_tx_type < len(Tx_Type_Inter_Inv_Set3) else DCT_DCT
        else:
            # intra_tx_type (S())
            if set_val == TX_SET_INTRA_1:
                cdf = [1 << 14] * 7 + [1 << 15, 0]  # 简化CDF
                intra_tx_type = read_symbol(decoder, cdf)
                TxType = Tx_Type_Intra_Inv_Set1[intra_tx_type] if intra_tx_type < len(Tx_Type_Intra_Inv_Set1) else DCT_DCT
            else:  # TX_SET_INTRA_2
                cdf = [1 << 14] * 5 + [1 << 15, 0]  # 简化CDF
                intra_tx_type = read_symbol(decoder, cdf)
                TxType = Tx_Type_Intra_Inv_Set2[intra_tx_type] if intra_tx_type < len(Tx_Type_Intra_Inv_Set2) else DCT_DCT
    else:
        TxType = DCT_DCT
    
    # TxTypes数组更新（简化处理，实际应该更新到全局数组）
    # for (i = 0; i < (Tx_Width[txSz] >> 2); i++)
    #     for (j = 0; j < (Tx_Height[txSz] >> 2); j++)
    #         TxTypes[y4 + j][x4 + i] = TxType
    
    return TxType


def compute_tx_type(plane: int, txSz: int, blockX: int, blockY: int,
                   Lossless: bool, is_inter: bool,
                   UVMode: int = DC_PRED,
                   MiRow: int = 0, MiCol: int = 0,
                   subsampling_x: int = 1, subsampling_y: int = 1) -> int:
    """
    计算变换类型
    规范文档 6.11.37 compute_tx_type()
    
    Args:
        plane: 平面索引
        txSz: 变换尺寸
        blockX: 块X坐标（4x4单位）
        blockY: 块Y坐标（4x4单位）
        Lossless: 是否为无损模式
        is_inter: 是否为帧间
        UVMode: UV模式（用于色度）
        MiRow: Mi行位置
        MiCol: Mi列位置
        subsampling_x: 水平下采样
        subsampling_y: 垂直下采样
        
    Returns:
        变换类型
    """
    txSzSqrUp = Tx_Size_Sqr_Up[txSz] if txSz < len(Tx_Size_Sqr_Up) else TX_4X4
    
    if Lossless or txSzSqrUp > TX_32X32:
        return DCT_DCT
    
    txSet = get_tx_set(txSz, is_inter)
    
    if plane == 0:
        # 从TxTypes数组获取（简化处理）
        # return TxTypes[blockY][blockX]
        return DCT_DCT  # 简化处理
    
    # 色度平面处理
    if is_inter:
        x4 = max(MiCol, blockX << subsampling_x)
        y4 = max(MiRow, blockY << subsampling_y)
        # txType = TxTypes[y4][x4]
        txType = DCT_DCT  # 简化处理
        
        # is_tx_type_in_set检查（简化处理）
        if not is_tx_type_in_set(txSet, txType, is_inter):
            return DCT_DCT
        return txType
    
    # Mode_To_Txfm查找表（简化处理）
    Mode_To_Txfm = {
        DC_PRED: DCT_DCT,
        V_PRED: V_DCT,
        H_PRED: H_DCT,
        # 其他模式...
    }
    
    txType = Mode_To_Txfm.get(UVMode, DCT_DCT)
    
    if not is_tx_type_in_set(txSet, txType, is_inter):
        return DCT_DCT
    
    return txType


def is_tx_type_in_set(txSet: int, txType: int, is_inter: bool) -> bool:
    """
    检查变换类型是否在集合中
    规范文档中定义的is_tx_type_in_set()
    
    Args:
        txSet: 变换集合
        txType: 变换类型
        is_inter: 是否为帧间
        
    Returns:
        是否在集合中
    """
    # Tx_Type_In_Set_Inter和Tx_Type_In_Set_Intra查找表（简化处理）
    # 规范文档中定义了这些查找表，这里简化实现
    if txSet == TX_SET_DCTONLY:
        return txType == DCT_DCT
    
    # 简化处理：假设所有标准变换类型都在集合中
    return True


def get_default_scan(txSz: int) -> list:
    """
    获取默认扫描顺序（Zig-Zag扫描）
    规范文档 6.11.42 get_default_scan()
    
    Args:
        txSz: 变换尺寸
        
    Returns:
        扫描数组
    """
    # 获取块尺寸
    from residual.residual import Tx_Width, Tx_Height
    w = Tx_Width[txSz] if txSz < len(Tx_Width) else 4
    h = Tx_Height[txSz] if txSz < len(Tx_Height) else 4
    
    # 生成简化的Zig-Zag扫描（简化实现）
    # 实际应该使用规范文档中定义的查找表
    scan = []
    size = w * h
    
    # 简化的Zig-Zag扫描生成
    for i in range(size):
        scan.append(i)
    
    return scan


def get_mrow_scan(txSz: int) -> list:
    """
    获取行扫描顺序
    规范文档 6.11.43 get_mrow_scan()
    
    Args:
        txSz: 变换尺寸
        
    Returns:
        扫描数组
    """
    # 获取块尺寸
    from residual.residual import Tx_Width, Tx_Height
    w = Tx_Width[txSz] if txSz < len(Tx_Width) else 4
    h = Tx_Height[txSz] if txSz < len(Tx_Height) else 4
    
    # 行扫描：按行顺序扫描
    scan = []
    for row in range(h):
        for col in range(w):
            scan.append(row * w + col)
    
    return scan


def get_mcol_scan(txSz: int) -> list:
    """
    获取列扫描顺序
    规范文档 6.11.44 get_mcol_scan()
    
    Args:
        txSz: 变换尺寸
        
    Returns:
        扫描数组
    """
    # 获取块尺寸
    from residual.residual import Tx_Width, Tx_Height
    w = Tx_Width[txSz] if txSz < len(Tx_Width) else 4
    h = Tx_Height[txSz] if txSz < len(Tx_Height) else 4
    
    # 列扫描：按列顺序扫描
    scan = []
    for col in range(w):
        for row in range(h):
            scan.append(row * w + col)
    
    return scan


def get_scan(txSz: int, PlaneTxType: int) -> list:
    """
    获取扫描顺序
    规范文档 6.11.41 get_scan()
    
    Args:
        txSz: 变换尺寸
        PlaneTxType: 平面变换类型
        
    Returns:
        扫描数组
    """
    # 特殊尺寸处理
    if txSz == TX_16X64:
        return get_default_scan(TX_16X32)
    
    if txSz == TX_64X16:
        return get_default_scan(TX_32X16)
    
    txSzSqrUp = Tx_Size_Sqr_Up[txSz] if txSz < len(Tx_Size_Sqr_Up) else TX_4X4
    if txSzSqrUp == TX_64X64:
        return get_default_scan(TX_32X32)
    
    # 根据PlaneTxType选择扫描顺序
    if PlaneTxType == IDTX:
        return get_default_scan(txSz)
    
    # preferRow检查
    preferRow = (PlaneTxType == V_DCT or
                 PlaneTxType == V_ADST or
                 PlaneTxType == V_FLIPADST)
    
    # preferCol检查
    preferCol = (PlaneTxType == H_DCT or
                 PlaneTxType == H_ADST or
                 PlaneTxType == H_FLIPADST)
    
    if preferRow:
        return get_mrow_scan(txSz)
    elif preferCol:
        return get_mcol_scan(txSz)
    else:
        return get_default_scan(txSz)

