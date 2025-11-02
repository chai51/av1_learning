"""
运动向量工具函数
实现MV裁剪等辅助函数
"""

from constants import MI_SIZE, Num_4x4_Blocks_High, Num_4x4_Blocks_Wide


def clamp_mv_row(mvec: int, border: int,
                MiRow: int, MiSize: int, MiRows: int) -> int:
    """
    裁剪MV行分量
    规范文档 6.10.18 clamp_mv_row()
    
    Args:
        mvec: MV行分量
        border: 边界值
        MiRow: Mi行位置
        MiSize: Mi尺寸
        MiRows: Mi总行数
        
    Returns:
        裁剪后的MV行分量
    """
    bh4 = Num_4x4_Blocks_High[MiSize] if MiSize < len(Num_4x4_Blocks_High) else 1
    
    mbToTopEdge = -((MiRow * MI_SIZE) * 8)
    mbToBottomEdge = ((MiRows - bh4 - MiRow) * MI_SIZE) * 8
    
    return clip3(mbToTopEdge - border, mbToBottomEdge + border, mvec)


def clamp_mv_col(mvec: int, border: int,
                MiCol: int, MiSize: int, MiCols: int) -> int:
    """
    裁剪MV列分量
    规范文档 6.10.19 clamp_mv_col()
    
    Args:
        mvec: MV列分量
        border: 边界值
        MiCol: Mi列位置
        MiSize: Mi尺寸
        MiCols: Mi总列数
        
    Returns:
        裁剪后的MV列分量
    """
    bw4 = Num_4x4_Blocks_Wide[MiSize] if MiSize < len(Num_4x4_Blocks_Wide) else 1
    
    mbToLeftEdge = -((MiCol * MI_SIZE) * 8)
    mbToRightEdge = ((MiCols - bw4 - MiCol) * MI_SIZE) * 8
    
    return clip3(mbToLeftEdge - border, mbToRightEdge + border, mvec)


def clip3(min_val: int, max_val: int, val: int) -> int:
    """
    Clip3函数
    将值限制在[min_val, max_val]范围内
    
    Args:
        min_val: 最小值
        max_val: 最大值
        val: 要裁剪的值
        
    Returns:
        裁剪后的值
    """
    if val < min_val:
        return min_val
    elif val > max_val:
        return max_val
    else:
        return val

