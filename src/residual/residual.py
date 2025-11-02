"""
残差解码实现
规范文档 6.11.34 residual()
"""

from entropy.symbol_decoder import SymbolDecoder, read_symbol
from bitstream.bit_reader import BitReader
from sequence.sequence_header import SequenceHeader
from frame.frame_header import FrameHeader
from mode.mode_info import ModeInfo
from constants import (
    BLOCK_64X64, TX_4X4, TX_16X64, TX_64X16, TX_SIZES_ALL, TX_32X32,
    DCT_DCT, IDTX, V_DCT, H_DCT, V_ADST, H_ADST, V_FLIPADST, H_FLIPADST,
    DC_PRED,
    MI_SIZE, MI_SIZE_LOG2,
    NUM_BASE_LEVELS, COEFF_BASE_RANGE, BR_CDF_SIZE,
    SIG_COEF_CONTEXTS_EOB, SIG_COEF_CONTEXTS_2D, SIG_COEF_CONTEXTS
)
from residual.transform_utils import (
    transform_type, compute_tx_type, get_scan
)
from residual.coeff_context import (
    get_coeff_context_eob, get_coeff_context, get_coeff_br_context
)


# 变换尺寸查找表（规范文档中定义的查找表）
# Tx_Width[TX_SIZES_ALL] - 变换宽度
Tx_Width = [
    4, 8, 16, 32, 64,  # TX_4X4, TX_8X8, TX_16X16, TX_32X32, TX_64X64
    4, 8, 8, 16, 16, 32, 32, 64, 64,  # 非方形
    4, 16, 8, 32, 16, 64  # 更多非方形
]

# Tx_Height[TX_SIZES_ALL] - 变换高度
Tx_Height = [
    4, 8, 16, 32, 64,  # TX_4X4, TX_8X8, TX_16X16, TX_32X32, TX_64X64
    8, 4, 16, 8, 32, 16, 64, 32, 128, 64,  # 非方形
    16, 4, 32, 8, 64, 16  # 更多非方形
]

# Tx_Width_Log2[TX_SIZES_ALL]
Tx_Width_Log2 = [2, 3, 4, 5, 6, 2, 3, 3, 4, 4, 5, 5, 6, 6, 2, 4, 3, 5, 4, 6]

# Tx_Height_Log2[TX_SIZES_ALL]
Tx_Height_Log2 = [2, 3, 4, 5, 6, 3, 2, 4, 3, 5, 4, 6, 5, 7, 4, 2, 5, 3, 6, 4]


def residual(decoder: SymbolDecoder,
            seq_header: SequenceHeader,
            frame_header: FrameHeader,
            mode_info: ModeInfo,
            MiRow: int, MiCol: int, MiSize: int,
            Lossless: bool = False,
            TxSize: int = TX_4X4,
            HasChroma: bool = True,
            subsampling_x: int = 1, subsampling_y: int = 1):
    """
    残差解码主函数
    规范文档 6.11.34 residual()
    
    Args:
        decoder: SymbolDecoder实例
        seq_header: 序列头
        frame_header: 帧头
        mode_info: ModeInfo实例
        MiRow: Mi行位置
        MiCol: Mi列位置
        MiSize: 块尺寸
        Lossless: 是否为无损模式
        TxSize: 变换尺寸
        HasChroma: 是否有色度
        subsampling_x: 色度水平下采样
        subsampling_y: 色度垂直下采样
    """
    # sbMask = use_128x128_superblock ? 31 : 15
    use_128x128_superblock = False  # 从seq_header获取，简化处理
    sbMask = 31 if use_128x128_superblock else 15
    
    # Block_Width和Block_Height从constants导入
    from constants import Block_Width, Block_Height
    
    # widthChunks = Max(1, Block_Width[MiSize] >> 6)
    widthChunks = max(1, Block_Width[MiSize] >> 6)
    # heightChunks = Max(1, Block_Height[MiSize] >> 6)
    heightChunks = max(1, Block_Height[MiSize] >> 6)
    
    # miSizeChunk = (widthChunks > 1 || heightChunks > 1) ? BLOCK_64X64 : MiSize
    miSizeChunk = BLOCK_64X64 if (widthChunks > 1 or heightChunks > 1) else MiSize
    
    # 遍历chunk
    for chunkY in range(heightChunks):
        for chunkX in range(widthChunks):
            miRowChunk = MiRow + (chunkY << 4)
            miColChunk = MiCol + (chunkX << 4)
            subBlockMiRow = miRowChunk & sbMask
            subBlockMiCol = miColChunk & sbMask
            
            # 遍历plane
            for plane in range(1 + (2 if HasChroma else 0)):
                # txSz = Lossless ? TX_4X4 : get_tx_size(plane, TxSize)
                if Lossless:
                    txSz = TX_4X4
                else:
                    txSz = get_tx_size(plane, TxSize, MiSize, seq_header, subsampling_x, subsampling_y)
                
                # stepX和stepY
                stepX = Tx_Width[txSz] >> 2
                stepY = Tx_Height[txSz] >> 2
                
                # planeSz = get_plane_residual_size(miSizeChunk, plane)
                planeSz = get_plane_residual_size(miSizeChunk, plane, subsampling_x, subsampling_y)
                
                # Num_4x4_Blocks_Wide和Num_4x4_Blocks_High从constants导入
                from constants import Num_4x4_Blocks_Wide, Num_4x4_Blocks_High
                num4x4W = Num_4x4_Blocks_Wide[planeSz]
                num4x4H = Num_4x4_Blocks_High[planeSz]
                
                # subX和subY
                subX = subsampling_x if plane > 0 else 0
                subY = subsampling_y if plane > 0 else 0
                
                # baseX和baseY
                baseX = (miColChunk >> subX) * MI_SIZE
                baseY = (miRowChunk >> subY) * MI_SIZE
                
                # 根据is_inter决定调用transform_tree还是直接调用transform_block
                is_inter = mode_info.is_inter
                
                if is_inter and not Lossless and plane == 0:
                    # transform_tree(baseX, baseY, num4x4W * 4, num4x4H * 4)
                    transform_tree(decoder, seq_header, frame_header, mode_info,
                                  baseX, baseY, num4x4W * 4, num4x4H * 4,
                                  MiRow=MiRow, MiCol=MiCol, subX=subX, subY=subY,
                                  subsampling_x=subsampling_x, subsampling_y=subsampling_y)
                else:
                    # baseXBlock和baseYBlock
                    baseXBlock = (MiCol >> subX) * MI_SIZE
                    baseYBlock = (MiRow >> subY) * MI_SIZE
                    
                    # 遍历transform blocks
                    for y in range(0, num4x4H, stepY):
                        for x in range(0, num4x4W, stepX):
                            transform_block(decoder, seq_header, frame_header, mode_info,
                                           plane, baseXBlock, baseYBlock, txSz,
                                           x + ((chunkX << 4) >> subX),
                                           y + ((chunkY << 4) >> subY),
                                           MiRow=MiRow, MiCol=MiCol,
                                           subX=subX, subY=subY,
                                           subsampling_x=subsampling_x, subsampling_y=subsampling_y,
                                           Lossless=Lossless)


def get_tx_size(plane: int, txSz: int, MiSize: int,
               seq_header: SequenceHeader,
               subsampling_x: int, subsampling_y: int) -> int:
    """
    获取变换尺寸
    规范文档 6.11.37 get_tx_size()
    
    Args:
        plane: 平面索引（0=Y, 1=U, 2=V）
        txSz: 初始变换尺寸
        MiSize: 块尺寸
        seq_header: 序列头
        subsampling_x: 水平下采样
        subsampling_y: 垂直下采样
        
    Returns:
        变换尺寸
    """
    if plane == 0:
        return txSz
    
    # uvTx = Max_Tx_Size_Rect[get_plane_residual_size(MiSize, plane)]
    planeSz = get_plane_residual_size(MiSize, plane, subsampling_x, subsampling_y)
    
    # Max_Tx_Size_Rect从tx_size模块导入
    from residual.tx_size import Max_Tx_Size_Rect
    uvTx = Max_Tx_Size_Rect[planeSz] if planeSz < len(Max_Tx_Size_Rect) else TX_4X4
    
    # 如果Tx_Width[uvTx] == 64 || Tx_Height[uvTx] == 64
    if Tx_Width[uvTx] == 64 or Tx_Height[uvTx] == 64:
        if Tx_Width[uvTx] == 16:
            return TX_16X32
        if Tx_Height[uvTx] == 16:
            return TX_32X16
        return TX_32X32
    
    return uvTx


def get_plane_residual_size(subsize: int, plane: int,
                           subsampling_x: int, subsampling_y: int) -> int:
    """
    获取平面残差尺寸
    规范文档 6.11.38 get_plane_residual_size()
    
    Args:
        subsize: 块尺寸
        plane: 平面索引
        subsampling_x: 水平下采样
        subsampling_y: 垂直下采样
        
    Returns:
        残差块尺寸
    """
    subx = subsampling_x if plane > 0 else 0
    suby = subsampling_y if plane > 0 else 0
    
    # Subsampled_Size查找表（规范文档定义）
    from constants import Subsampled_Size, BLOCK_INVALID
    
    if subsize < len(Subsampled_Size):
        result = Subsampled_Size[subsize][subx][suby]
        # 如果结果是BLOCK_INVALID，返回原始尺寸
        if result == BLOCK_INVALID:
            return subsize
        return result
    
    # 如果索引超出范围，返回原始尺寸
    return subsize


def transform_tree(decoder: SymbolDecoder,
                 seq_header: SequenceHeader,
                 frame_header: FrameHeader,
                 mode_info: ModeInfo,
                 startX: int, startY: int, w: int, h: int,
                 MiRow: int, MiCol: int,
                 subX: int = 0, subY: int = 0,
                 subsampling_x: int = 1, subsampling_y: int = 1):
    """
    变换树解码
    规范文档 6.11.35 transform_tree()
    
    Args:
        decoder: SymbolDecoder实例
        seq_header: 序列头
        frame_header: 帧头
        mode_info: ModeInfo实例
        startX: 起始X坐标
        startY: 起始Y坐标
        w: 宽度
        h: 高度
        MiRow: Mi行位置
        MiCol: Mi列位置
        subX: X下采样
        subY: Y下采样
        subsampling_x: 水平下采样
        subsampling_y: 垂直下采样
    """
    # maxX和maxY（需要从frame_header获取，简化处理）
    MiCols = frame_header.MiCols if hasattr(frame_header, 'MiCols') else 64
    MiRows = frame_header.MiRows if hasattr(frame_header, 'MiRows') else 64
    maxX = MiCols * MI_SIZE
    maxY = MiRows * MI_SIZE
    
    if startX >= maxX or startY >= maxY:
        return
    
    # row和col
    row = startY >> MI_SIZE_LOG2
    col = startX >> MI_SIZE_LOG2
    
    # lumaTxSz = InterTxSizes[row][col]
    # InterTxSizes需要从上下文获取（简化处理）
    lumaTxSz = TX_4X4  # 简化处理
    
    lumaW = Tx_Width[lumaTxSz]
    lumaH = Tx_Height[lumaTxSz]
    
    if w <= lumaW and h <= lumaH:
        # txSz = find_tx_size(w, h)
        txSz = find_tx_size(w, h)
        # transform_block(0, startX, startY, txSz, 0, 0)
        transform_block(decoder, seq_header, frame_header, mode_info,
                      0, startX, startY, txSz, 0, 0,
                      MiRow=MiRow, MiCol=MiCol,
                      subX=subX, subY=subY,
                      subsampling_x=subsampling_x, subsampling_y=subsampling_y)
    else:
        # 递归分割
        if w > h:
            # 垂直分割
            transform_tree(decoder, seq_header, frame_header, mode_info,
                         startX, startY, w // 2, h,
                         MiRow=MiRow, MiCol=MiCol,
                         subX=subX, subY=subY,
                         subsampling_x=subsampling_x, subsampling_y=subsampling_y)
            transform_tree(decoder, seq_header, frame_header, mode_info,
                         startX + w // 2, startY, w // 2, h,
                         MiRow=MiRow, MiCol=MiCol,
                         subX=subX, subY=subY,
                         subsampling_x=subsampling_x, subsampling_y=subsampling_y)
        elif w < h:
            # 水平分割
            transform_tree(decoder, seq_header, frame_header, mode_info,
                         startX, startY, w, h // 2,
                         MiRow=MiRow, MiCol=MiCol,
                         subX=subX, subY=subY,
                         subsampling_x=subsampling_x, subsampling_y=subsampling_y)
            transform_tree(decoder, seq_header, frame_header, mode_info,
                         startX, startY + h // 2, w, h // 2,
                         MiRow=MiRow, MiCol=MiCol,
                         subX=subX, subY=subY,
                         subsampling_x=subsampling_x, subsampling_y=subsampling_y)
        else:
            # 分成4个象限
            transform_tree(decoder, seq_header, frame_header, mode_info,
                         startX, startY, w // 2, h // 2,
                         MiRow=MiRow, MiCol=MiCol,
                         subX=subX, subY=subY,
                         subsampling_x=subsampling_x, subsampling_y=subsampling_y)
            transform_tree(decoder, seq_header, frame_header, mode_info,
                         startX + w // 2, startY, w // 2, h // 2,
                         MiRow=MiRow, MiCol=MiCol,
                         subX=subX, subY=subY,
                         subsampling_x=subsampling_x, subsampling_y=subsampling_y)
            transform_tree(decoder, seq_header, frame_header, mode_info,
                         startX, startY + h // 2, w // 2, h // 2,
                         MiRow=MiRow, MiCol=MiCol,
                         subX=subX, subY=subY,
                         subsampling_x=subsampling_x, subsampling_y=subsampling_y)
            transform_tree(decoder, seq_header, frame_header, mode_info,
                         startX + w // 2, startY + h // 2, w // 2, h // 2,
                         MiRow=MiRow, MiCol=MiCol,
                         subX=subX, subY=subY,
                         subsampling_x=subsampling_x, subsampling_y=subsampling_y)


def find_tx_size(w: int, h: int) -> int:
    """
    查找匹配的变换尺寸
    规范文档中定义的find_tx_size()
    
    Args:
        w: 宽度
        h: 高度
        
    Returns:
        变换尺寸
    """
    for txSz in range(TX_SIZES_ALL):
        if Tx_Width[txSz] == w and Tx_Height[txSz] == h:
            return txSz
    # 如果没有找到，返回TX_4X4作为默认值
    return TX_4X4


def transform_block(decoder: SymbolDecoder,
                   seq_header: SequenceHeader,
                   frame_header: FrameHeader,
                   mode_info: ModeInfo,
                   plane: int, baseX: int, baseY: int, txSz: int, x: int, y: int,
                   MiRow: int, MiCol: int,
                   subX: int = 0, subY: int = 0,
                   subsampling_x: int = 1, subsampling_y: int = 1,
                   Lossless: bool = False):
    """
    变换块解码
    规范文档 6.11.33 transform_block()
    
    Args:
        decoder: SymbolDecoder实例
        seq_header: 序列头
        frame_header: 帧头
        mode_info: ModeInfo实例
        plane: 平面索引
        baseX: 基准X坐标
        baseY: 基准Y坐标
        txSz: 变换尺寸
        x: X偏移
        y: Y偏移
        MiRow: Mi行位置
        MiCol: Mi列位置
        subX: X下采样
        subY: Y下采样
        subsampling_x: 水平下采样
        subsampling_y: 垂直下采样
        Lossless: 是否为无损模式
    """
    startX = baseX + 4 * x
    startY = baseY + 4 * y
    
    subX = subsampling_x if plane > 0 else 0
    subY = subsampling_y if plane > 0 else 0
    
    row = (startY << subY) >> MI_SIZE_LOG2
    col = (startX << subX) >> MI_SIZE_LOG2
    
    # sbMask
    use_128x128_superblock = False  # 从seq_header获取，简化处理
    sbMask = 31 if use_128x128_superblock else 15
    
    subBlockMiRow = row & sbMask
    subBlockMiCol = col & sbMask
    
    stepX = Tx_Width[txSz] >> MI_SIZE_LOG2
    stepY = Tx_Height[txSz] >> MI_SIZE_LOG2
    
    # maxX和maxY
    MiCols = frame_header.MiCols if hasattr(frame_header, 'MiCols') else 64
    MiRows = frame_header.MiRows if hasattr(frame_header, 'MiRows') else 64
    maxX = (MiCols * MI_SIZE) >> subX
    maxY = (MiRows * MI_SIZE) >> subY
    
    if startX >= maxX or startY >= maxY:
        return
    
    # if (!is_inter) - 帧内预测
    pred = None
    if not mode_info.is_inter:
        # 帧内预测
        from reconstruction.predict import predict_intra
        
        # 获取参考缓冲区（简化处理）
        ref_buffer = [[], []]  # 上方和左方参考像素，简化处理
        
        pred = predict_intra(mode_info, plane, startX, startY,
                           Tx_Width[txSz], Tx_Height[txSz],
                           ref_buffer, seq_header, frame_header)
    
    # PlaneTxType计算（在coeffs中计算，这里需要预先计算）
    PlaneTxType = DCT_DCT  # 默认值，将在coeffs中更新
    
    # if (!skip) - 系数解码和重建
    skip = mode_info.skip
    if not skip:
        # coeffs(plane, startX, startY, txSz)
        # 需要传递reader以支持L(1)读取
        reader = decoder.reader if hasattr(decoder, 'reader') else None
        eob = coeffs(decoder, seq_header, frame_header, mode_info,
                    plane, startX, startY, txSz,
                    subX=subX, subY=subY, reader=reader,
                    Lossless=Lossless,
                    MiRow=MiRow, MiCol=MiCol,
                    subsampling_x=subsampling_x, subsampling_y=subsampling_y)
        
        if eob > 0:
            # reconstruct(plane, startX, startY, txSz)
            from reconstruction.reconstruct import reconstruct_block
            
            # 获取量化系数（从coeffs函数返回，简化处理）
            # 实际应该从coeffs函数获取Quant数组
            coeffs_array = [[0 for _ in range(Tx_Width[txSz])]
                          for _ in range(Tx_Height[txSz])]  # 简化处理
            
            # 获取量化索引（简化处理）
            qindex = 0  # 从frame_header获取
            
            # 重建
            if pred is not None:
                recon = reconstruct_block(mode_info, plane, startX, startY,
                                        Tx_Width[txSz], Tx_Height[txSz],
                                        pred, coeffs_array, txSz,
                                        qindex, tx_type=PlaneTxType,
                                        bit_depth=8, Lossless=Lossless)


def coeffs(decoder: SymbolDecoder,
          seq_header: SequenceHeader,
          frame_header: FrameHeader,
          mode_info: ModeInfo,
          plane: int, startX: int, startY: int, txSz: int,
          subX: int = 0, subY: int = 0,
          reader: BitReader = None,
          Lossless: bool = False,
          MiRow: int = 0, MiCol: int = 0,
          subsampling_x: int = 1, subsampling_y: int = 1) -> int:
    """
    系数解码
    规范文档 6.11.36 coeffs()
    
    Args:
        decoder: SymbolDecoder实例
        seq_header: 序列头
        frame_header: 帧头
        mode_info: ModeInfo实例
        plane: 平面索引
        startX: 起始X坐标
        startY: 起始Y坐标
        txSz: 变换尺寸
        subX: X下采样
        subY: Y下采样
        
    Returns:
        EOB值（End of Block）
    """
    x4 = startX >> 2
    y4 = startY >> 2
    w4 = Tx_Width[txSz] >> 2
    h4 = Tx_Height[txSz] >> 2
    
    # txSzCtx计算（规范文档定义）
    # txSzCtx = (Tx_Size_Sqr[txSz] + Tx_Size_Sqr_Up[txSz] + 1) >> 1
    from residual.transform_utils import Tx_Size_Sqr, Tx_Size_Sqr_Up
    if txSz < len(Tx_Size_Sqr) and txSz < len(Tx_Size_Sqr_Up):
        sqr = Tx_Size_Sqr[txSz]
        sqrUp = Tx_Size_Sqr_Up[txSz]
        txSzCtx = (sqr + sqrUp + 1) >> 1
    else:
        txSzCtx = 0  # 简化处理
    
    ptype = plane > 0
    
    # segEob计算
    if txSz == TX_16X64 or txSz == TX_64X16:
        segEob = 512
    else:
        segEob = min(1024, Tx_Width[txSz] * Tx_Height[txSz])
    
    # Quant数组初始化（将在后续实现中使用）
    Quant = [0] * segEob
    
    # Dequant数组初始化（将在后续实现中使用）
    Dequant = [[0] * 64 for _ in range(64)]
    
    eob = 0
    culLevel = 0
    dcCategory = 0
    
    # all_zero (S())
    cdf = [1 << 14, 1 << 15, 0]  # 简化CDF
    all_zero = read_symbol(decoder, cdf)
    
    if all_zero:
        c = 0
        if plane == 0:
            # 设置TxTypes（简化处理）
            pass
        return 0
    
    # 如果all_zero == 0，继续解码EOB和系数
    # transform_type解析（仅对plane==0）
    TxType = DCT_DCT  # 默认值
    if plane == 0:
        # transform_type(x4, y4, txSz)
        is_inter = mode_info.is_inter
        segment_id = mode_info.segment_id
        TxType = transform_type(decoder, x4, y4, txSz,
                               is_inter=is_inter,
                               segment_id=segment_id,
                               segmentation_enabled=False,  # 从frame_header获取，简化处理
                               base_q_idx=0)  # 从frame_header获取，简化处理
    
    # PlaneTxType计算
    PlaneTxType = compute_tx_type(plane, txSz, x4, y4,
                                 Lossless=Lossless,
                                 is_inter=mode_info.is_inter,
                                 UVMode=mode_info.UVMode if plane > 0 else DC_PRED,
                                 MiRow=MiRow, MiCol=MiCol,
                                 subsampling_x=subsampling_x, subsampling_y=subsampling_y)
    
    # scan获取
    scan = get_scan(txSz, PlaneTxType)
    
    # EOB解码
    eobMultisize = min(Tx_Width_Log2[txSz], 5) + min(Tx_Height_Log2[txSz], 5) - 4
    
    if eobMultisize == 0:
        # eob_pt_16 (S())
        cdf = [1 << 14] * 16 + [1 << 15, 0]  # 简化CDF
        eob_pt_16 = read_symbol(decoder, cdf)
        eobPt = eob_pt_16 + 1
    elif eobMultisize == 1:
        # eob_pt_32 (S())
        cdf = [1 << 14] * 32 + [1 << 15, 0]  # 简化CDF
        eob_pt_32 = read_symbol(decoder, cdf)
        eobPt = eob_pt_32 + 1
    elif eobMultisize == 2:
        # eob_pt_64 (S())
        cdf = [1 << 14] * 64 + [1 << 15, 0]  # 简化CDF
        eob_pt_64 = read_symbol(decoder, cdf)
        eobPt = eob_pt_64 + 1
    elif eobMultisize == 3:
        # eob_pt_128 (S())
        cdf = [1 << 14] * 128 + [1 << 15, 0]  # 简化CDF
        eob_pt_128 = read_symbol(decoder, cdf)
        eobPt = eob_pt_128 + 1
    elif eobMultisize == 4:
        # eob_pt_256 (S())
        cdf = [1 << 14] * 256 + [1 << 15, 0]  # 简化CDF
        eob_pt_256 = read_symbol(decoder, cdf)
        eobPt = eob_pt_256 + 1
    elif eobMultisize == 5:
        # eob_pt_512 (S())
        cdf = [1 << 14] * 512 + [1 << 15, 0]  # 简化CDF
        eob_pt_512 = read_symbol(decoder, cdf)
        eobPt = eob_pt_512 + 1
    else:
        # eob_pt_1024 (S())
        cdf = [1 << 14] * 1024 + [1 << 15, 0]  # 简化CDF
        eob_pt_1024 = read_symbol(decoder, cdf)
        eobPt = eob_pt_1024 + 1
    
    # eob计算
    eob = eobPt if eobPt < 2 else ((1 << (eobPt - 2)) + 1)
    eobShift = max(-1, eobPt - 3)
    
    if eobShift >= 0:
        # eob_extra (S())
        cdf = [1 << 14, 1 << 15, 0]  # 简化CDF
        eob_extra = read_symbol(decoder, cdf)
        if eob_extra:
            eob += (1 << eobShift)
        
        # eob_extra_bit循环
        if reader is not None:
            for i in range(1, max(0, eobPt - 2)):
                eobShift = max(0, eobPt - 2) - 1 - i
                # eob_extra_bit (L(1))
                eob_extra_bit = reader.read_bit()
                if eob_extra_bit:
                    eob += (1 << eobShift)
    
    # 如果eob == 0，直接返回
    if eob == 0:
        return 0
    
    # scan已经在上面获取了
    # 确保scan数组足够长
    if len(scan) < eob:
        # 如果scan不够长，扩展它
        scan = scan + list(range(len(scan), eob))
    
    # 第一阶段：从EOB-1开始，倒序解码level值
    # for (c = eob - 1; c >= 0; c--)
    for c in range(eob - 1, -1, -1):
        pos = scan[c] if c < len(scan) else c  # 简化处理
        
        if c == (eob - 1):
            # coeff_base_eob (S())
            # 根据上下文选择CDF
            ctx = get_coeff_context_eob(pos, txSz, ptype, txSzCtx)
            # 实际应该使用对应的CDF数组，简化处理使用均匀CDF
            cdf = [1 << 14] * NUM_BASE_LEVELS + [1 << 15, 0]  # 简化CDF
            coeff_base_eob = read_symbol(decoder, cdf)
            level = coeff_base_eob + 1
        else:
            # coeff_base (S())
            # 根据上下文选择CDF
            ctx = get_coeff_context(pos, scan, Quant, txSz, PlaneTxType, txSzCtx,
                                   ptype, x4, y4, w4, h4, c=c)
            # 实际应该使用对应的CDF数组，简化处理使用均匀CDF
            cdf = [1 << 14] * NUM_BASE_LEVELS + [1 << 15, 0]  # 简化CDF
            coeff_base = read_symbol(decoder, cdf)
            level = coeff_base
        
        # if (level > NUM_BASE_LEVELS)
        if level > NUM_BASE_LEVELS:
            # coeff_br循环
            # for (idx = 0; idx < COEFF_BASE_RANGE / (BR_CDF_SIZE - 1); idx++)
            for idx in range(COEFF_BASE_RANGE // (BR_CDF_SIZE - 1)):
                # coeff_br (S())
                # 根据上下文选择CDF
                br_ctx = get_coeff_br_context(pos, level, txSz, ptype)
                # 实际应该使用对应的CDF数组，简化处理使用均匀CDF
                cdf = [1 << 14] * BR_CDF_SIZE + [1 << 15, 0]  # 简化CDF
                coeff_br = read_symbol(decoder, cdf)
                level += coeff_br
                if coeff_br < (BR_CDF_SIZE - 1):
                    break
        
        Quant[pos] = level
    
    # 第二阶段：从0开始，正序解码符号和Exp-Golomb编码
    # for (c = 0; c < eob; c++)
    for c in range(eob):
        pos = scan[c] if c < len(scan) else c  # 简化处理
        
        if Quant[pos] != 0:
            if c == 0:
                # dc_sign (S())
                cdf = [1 << 14, 1 << 15, 0]  # 简化CDF
                dc_sign = read_symbol(decoder, cdf)
                sign = dc_sign
            else:
                # sign_bit (L(1))
                if reader is not None:
                    sign_bit = reader.read_bit()
                    sign = sign_bit
                else:
                    # 如果没有reader，使用decoder.read_bool()作为后备
                    sign_bit = decoder.read_bool()
                    sign = sign_bit
        else:
            sign = 0
        
        # if (Quant[pos] > (NUM_BASE_LEVELS + COEFF_BASE_RANGE))
        if Quant[pos] > (NUM_BASE_LEVELS + COEFF_BASE_RANGE):
            # Exp-Golomb编码
            length = 0
            # do {
            if reader is not None:
                while True:
                    length += 1
                    # golomb_length_bit (L(1))
                    golomb_length_bit = reader.read_bit()
                    if golomb_length_bit:
                        break
                
                x = 1
                # for (i = length - 2; i >= 0; i--)
                for i in range(length - 2, -1, -1):
                    # golomb_data_bit (L(1))
                    golomb_data_bit = reader.read_bit()
                    x = (x << 1) | golomb_data_bit
            else:
                # 如果没有reader，简化处理
                length = 1  # 简化处理
                x = 0  # 简化处理
            
            Quant[pos] = x + COEFF_BASE_RANGE + NUM_BASE_LEVELS
        
        # if (pos == 0 && Quant[pos] > 0)
        if pos == 0 and Quant[pos] > 0:
            dcCategory = 1 if sign else 2
        
        # Quant[pos] = Quant[pos] & 0xFFFFF
        Quant[pos] = Quant[pos] & 0xFFFFF
        
        # culLevel += Quant[pos]
        culLevel += Quant[pos]
        
        # if (sign)
        if sign:
            Quant[pos] = -Quant[pos]
        
        # culLevel = Min(63, culLevel)
        culLevel = min(63, culLevel)
    
    # 更新上下文（AboveLevelContext和LeftLevelContext）
    # 简化处理，实际应该更新到全局数组
    # for (i = 0; i < w4; i++)
    #     AboveLevelContext[plane][x4 + i] = culLevel
    #     AboveDcContext[plane][x4 + i] = dcCategory
    # for (i = 0; i < h4; i++)
    #     LeftLevelContext[plane][y4 + i] = culLevel
    #     LeftDcContext[plane][y4 + i] = dcCategory
    
    return eob

