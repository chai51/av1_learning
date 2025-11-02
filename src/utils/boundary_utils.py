"""
边界检查工具函数
实现is_inside、is_inside_filter_region等函数
"""


def is_inside(candidateR: int, candidateC: int,
             MiRowStart: int, MiRowEnd: int,
             MiColStart: int, MiColEnd: int) -> bool:
    """
    检查候选位置是否在当前Tile内
    规范文档 6.10.20 is_inside()
    
    Args:
        candidateR: 候选行位置
        candidateC: 候选列位置
        MiRowStart: Tile起始行
        MiRowEnd: Tile结束行
        MiColStart: Tile起始列
        MiColEnd: Tile结束列
        
    Returns:
        是否在Tile内
    """
    return (candidateC >= MiColStart and
            candidateC < MiColEnd and
            candidateR >= MiRowStart and
            candidateR < MiRowEnd)


def is_inside_filter_region(candidateR: int, candidateC: int,
                           MiRows: int, MiCols: int,
                           colStart: int = 0, colEnd: int = None,
                           rowStart: int = 0, rowEnd: int = None) -> bool:
    """
    检查候选位置是否在滤波区域内
    规范文档 6.10.21 is_inside_filter_region()
    
    Args:
        candidateR: 候选行位置
        candidateC: 候选列位置
        MiRows: Mi总行数
        MiCols: Mi总列数
        colStart: 列起始位置（默认0）
        colEnd: 列结束位置（默认MiCols）
        rowStart: 行起始位置（默认0）
        rowEnd: 行结束位置（默认MiRows）
        
    Returns:
        是否在滤波区域内
    """
    if colEnd is None:
        colEnd = MiCols
    if rowEnd is None:
        rowEnd = MiRows
    
    return (candidateC >= colStart and
            candidateC < colEnd and
            candidateR >= rowStart and
            candidateR < rowEnd)

