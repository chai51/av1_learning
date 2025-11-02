"""
系数解码上下文计算
实现get_coeff_context等函数，用于选择coeff_base和coeff_base_eob的CDF
"""

from constants import (
    SIG_COEF_CONTEXTS_EOB, SIG_COEF_CONTEXTS_2D, SIG_COEF_CONTEXTS,
    NUM_BASE_LEVELS
)


def get_coeff_context_eob(pos: int, txSz: int, ptype: bool, txSzCtx: int) -> int:
    """
    获取coeff_base_eob的上下文
    规范文档中定义的上下文计算
    
    Args:
        pos: 系数位置（在扫描顺序中）
        txSz: 变换尺寸
        ptype: 是否为色度平面
        txSzCtx: 变换尺寸上下文
        
    Returns:
        上下文索引（0到SIG_COEF_CONTEXTS_EOB-1）
    """
    # 简化实现：根据位置和变换尺寸计算上下文
    # 实际应该考虑更多因素（上方和左方的非零系数等）
    
    if pos == 0:
        # DC系数
        return 0
    elif pos == 1:
        return 1
    elif pos < 4:
        return 2
    else:
        return 3


def get_coeff_context(pos: int, scan: list, Quant: list,
                     txSz: int, PlaneTxType: int, txSzCtx: int,
                     ptype: bool, x4: int, y4: int,
                     w4: int, h4: int) -> int:
    """
    获取coeff_base的上下文
    规范文档中定义的上下文计算
    
    Args:
        pos: 系数位置（在扫描顺序中）
        scan: 扫描数组
        Quant: 量化系数数组（已解码的部分）
        txSz: 变换尺寸
        PlaneTxType: 平面变换类型
        txSzCtx: 变换尺寸上下文
        ptype: 是否为色度平面
        x4: X坐标（4x4单位）
        y4: Y坐标（4x4单位）
        w4: 宽度（4x4单位）
        h4: 高度（4x4单位）
        
    Returns:
        上下文索引（0到SIG_COEF_CONTEXTS-1）
    """
    # 计算系数在块中的位置（row, col）
    # pos是扫描顺序中的位置，需要找到在块中的实际位置
    # 简化处理：假设scan[pos]直接对应位置
    scan_pos = scan[pos] if pos < len(scan) else pos
    
    # 计算行和列（从扫描位置计算）
    # 需要知道块的宽度和高度
    from residual.residual import Tx_Width, Tx_Height
    block_w = Tx_Width[txSz] if txSz < len(Tx_Width) else 4
    block_h = Tx_Height[txSz] if txSz < len(Tx_Height) else 4
    
    # 计算行和列
    row = scan_pos // block_w if block_w > 0 else 0
    col = scan_pos % block_w if block_w > 0 else scan_pos
    
    # 检查上方和左方的非零系数
    # 简化处理：检查已解码的系数
    aboveNzs = 0
    leftNzs = 0
    
    # 检查上方行（简化实现）
    if row > 0:
        for c in range(col):
            check_pos = scan.index(row * w4 + c) if (row * w4 + c) < len(scan) else -1
            if check_pos >= 0 and check_pos < len(Quant) and Quant[check_pos] != 0:
                aboveNzs += 1
                if aboveNzs >= 2:
                    break
    
    # 检查左方列（简化实现）
    if col > 0:
        for r in range(row):
            check_pos = scan.index(r * w4 + col) if (r * w4 + col) < len(scan) else -1
            if check_pos >= 0 and check_pos < len(Quant) and Quant[check_pos] != 0:
                leftNzs += 1
                if leftNzs >= 2:
                    break
    
    # 上下文计算（简化实现）
    # 规范文档中有详细的上下文计算公式
    # 这里使用简化的上下文计算
    
    # 判断是否为2D变换（水平和垂直方向都有变换）
    is2D = (PlaneTxType != V_DCT and PlaneTxType != H_DCT and
            PlaneTxType != V_ADST and PlaneTxType != H_ADST and
            PlaneTxType != V_FLIPADST and PlaneTxType != H_FLIPADST)
    
    if is2D:
        # 2D变换的上下文计算
        # 简化处理：基于aboveNzs和leftNzs
        ctx = (aboveNzs + leftNzs) * 2
        if pos == 0:
            ctx = 0  # DC系数特殊处理
    else:
        # 1D变换（水平或垂直）的上下文计算
        # 使用SIG_COEF_CONTEXTS_2D偏移
        if PlaneTxType == V_DCT or PlaneTxType == V_ADST or PlaneTxType == V_FLIPADST:
            # 垂直变换，主要看上方
            ctx = SIG_COEF_CONTEXTS_2D + min(aboveNzs, 2)
        else:
            # 水平变换，主要看左方
            ctx = SIG_COEF_CONTEXTS_2D + min(leftNzs, 2)
    
    # 限制上下文范围
    ctx = min(ctx, SIG_COEF_CONTEXTS - 1)
    
    return ctx


def get_coeff_br_context(pos: int, level: int, txSz: int, ptype: bool) -> int:
    """
    获取coeff_br的上下文
    规范文档中定义的上下文计算
    
    Args:
        pos: 系数位置
        level: 当前level值
        txSz: 变换尺寸
        ptype: 是否为色度平面
        
    Returns:
        上下文索引
    """
    # 简化实现：根据level和位置计算上下文
    # 实际应该考虑更多因素
    
    if level <= NUM_BASE_LEVELS:
        return 0
    elif level <= NUM_BASE_LEVELS + 4:
        return 1
    elif level <= NUM_BASE_LEVELS + 8:
        return 2
    else:
        return 3

