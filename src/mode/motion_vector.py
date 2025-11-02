"""
运动向量解析模块
实现find_mv_stack、assign_mv、mv_component等函数
"""

from typing import Optional, List, Tuple
from entropy.symbol_decoder import SymbolDecoder, read_symbol
from bitstream.bit_reader import BitReader
from bitstream.descriptors import read_su
from constants import (
    INTRA_FRAME, NONE, LAST_FRAME, LAST2_FRAME, LAST3_FRAME,
    GOLDEN_FRAME, BWDREF_FRAME, ALTREF_FRAME, ALTREF2_FRAME,
    NEWMV, NEW_NEWMV, NEARMV, NEAR_NEWMV, NEW_NEARMV,
    SIMPLE, OBMC, WARPED
)
from utils.mv_utils import clamp_mv_row, clamp_mv_col


class MotionVector:
    """
    运动向量结构
    规范文档中定义的MV
    """
    def __init__(self):
        self.row = 0  # MV行分量（1/8像素精度）
        self.col = 0  # MV列分量（1/8像素精度）
        self.ref_frame = INTRA_FRAME  # 参考帧


def find_mv_stack(decoder: SymbolDecoder,
                 ref_idx: int,
                 mode_info,
                 frame_header,
                 MiRow: int, MiCol: int, MiSize: int,
                 AvailU: bool = False, AvailL: bool = False) -> List[MotionVector]:
    """
    查找运动向量栈
    规范文档 6.10.15 find_mv_stack()
    
    Args:
        decoder: SymbolDecoder实例
        ref_idx: 参考帧索引（0或1）
        mode_info: ModeInfo对象
        frame_header: 帧头
        MiRow: Mi行位置
        MiCol: Mi列位置
        MiSize: Mi尺寸
        AvailU: 上方是否可用
        AvailL: 左方是否可用
        
    Returns:
        运动向量候选列表
    """
    # 简化实现：构建运动向量候选栈
    # 实际应该从上下文数组获取周围块的运动向量
    
    mv_stack = []
    
    # 空间候选（从上方和左方块获取）
    # 简化处理：创建默认候选
    if AvailU:
        # 从上方块获取MV（简化处理）
        mv = MotionVector()
        mv.row = 0  # 简化处理
        mv.col = 0  # 简化处理
        mv.ref_frame = mode_info.RefFrame[ref_idx]
        mv_stack.append(mv)
    
    if AvailL:
        # 从左方块获取MV（简化处理）
        mv = MotionVector()
        mv.row = 0  # 简化处理
        mv.col = 0  # 简化处理
        mv.ref_frame = mode_info.RefFrame[ref_idx]
        mv_stack.append(mv)
    
    # 时间候选（从参考帧获取）
    # 简化处理：跳过时间候选
    
    # 全局MV候选
    if frame_header.allow_warped_motion:  # 简化检查
        mv = MotionVector()
        mv.row = 0  # 全局MV（简化处理）
        mv.col = 0
        mv.ref_frame = mode_info.RefFrame[ref_idx]
        mv_stack.append(mv)
    
    # 零MV候选
    mv_zero = MotionVector()
    mv_zero.row = 0
    mv_zero.col = 0
    mv_zero.ref_frame = mode_info.RefFrame[ref_idx]
    mv_stack.append(mv_zero)
    
    return mv_stack


def assign_mv(decoder: SymbolDecoder,
              ref_idx: int,
              mode_info,
              frame_header,
              MiRow: int, MiCol: int, MiSize: int,
              AvailU: bool = False, AvailL: bool = False,
              mv_stack: Optional[List[MotionVector]] = None) -> MotionVector:
    """
    分配运动向量
    规范文档 6.10.16 assign_mv()
    
    Args:
        decoder: SymbolDecoder实例
        ref_idx: 参考帧索引（0或1）
        mode_info: ModeInfo对象
        frame_header: 帧头
        MiRow: Mi行位置
        MiCol: Mi列位置
        MiSize: Mi尺寸
        AvailU: 上方是否可用
        AvailL: 左方是否可用
        mv_stack: 运动向量栈（如果为None，会调用find_mv_stack）
        
    Returns:
        分配的运动向量
    """
    # 如果mv_stack未提供，先查找
    if mv_stack is None:
        mv_stack = find_mv_stack(decoder, ref_idx, mode_info, frame_header,
                               MiRow, MiCol, MiSize, AvailU, AvailL)
    
    YMode = mode_info.YMode
    
    # 根据YMode选择MV
    if ref_idx == 0:
        if YMode == NEARESTMV or YMode == NEW_NEARESTMV:
            # 使用栈中第一个MV
            if len(mv_stack) > 0:
                return mv_stack[0]
        elif YMode == NEARMV or YMode == NEAR_NEWMV:
            # 使用栈中第二个MV（如果有）
            if len(mv_stack) > 1:
                return mv_stack[1]
            elif len(mv_stack) > 0:
                return mv_stack[0]
        elif YMode == NEWMV or YMode == NEW_NEWMV:
            # 需要解析MV分量
            mv = MotionVector()
            mv.ref_frame = mode_info.RefFrame[ref_idx]
            
            # mv_component(ref_idx, 0) - 行分量
            mv.row = mv_component(decoder, ref_idx, 0, mode_info, frame_header,
                                MiRow, MiCol, MiSize, AvailU, AvailL, mv_stack)
            
            # mv_component(ref_idx, 1) - 列分量
            mv.col = mv_component(decoder, ref_idx, 1, mode_info, frame_header,
                                MiRow, MiCol, MiSize, AvailU, AvailL, mv_stack)
            
            return mv
        elif YMode == GLOBALMV or YMode == GLOBAL_GLOBALMV:
            # 全局MV
            mv = MotionVector()
            mv.row = 0  # 全局MV（简化处理）
            mv.col = 0
            mv.ref_frame = mode_info.RefFrame[ref_idx]
            return mv
    else:  # ref_idx == 1
        if YMode == NEW_NEARESTMV or YMode == NEW_NEARMV or YMode == NEAR_NEWMV:
            # 使用ref_idx=0的MV
            mv0 = assign_mv(decoder, 0, mode_info, frame_header,
                          MiRow, MiCol, MiSize, AvailU, AvailL)
            mv = MotionVector()
            mv.row = mv0.row
            mv.col = mv0.col
            mv.ref_frame = mode_info.RefFrame[ref_idx]
            return mv
        elif YMode == NEW_NEWMV:
            # 需要解析MV分量
            mv = MotionVector()
            mv.ref_frame = mode_info.RefFrame[ref_idx]
            
            # mv_component(ref_idx, 0) - 行分量
            mv.row = mv_component(decoder, ref_idx, 0, mode_info, frame_header,
                                MiRow, MiCol, MiSize, AvailU, AvailL, mv_stack)
            
            # mv_component(ref_idx, 1) - 列分量
            mv.col = mv_component(decoder, ref_idx, 1, mode_info, frame_header,
                                MiRow, MiCol, MiSize, AvailU, AvailL, mv_stack)
            
            return mv
    
    # 默认：返回零MV
    mv = MotionVector()
    mv.row = 0
    mv.col = 0
    mv.ref_frame = mode_info.RefFrame[ref_idx]
    return mv


def mv_component(decoder: SymbolDecoder,
                ref_idx: int, comp: int,
                mode_info,
                frame_header,
                MiRow: int, MiCol: int, MiSize: int,
                AvailU: bool = False, AvailL: bool = False,
                mv_stack: Optional[List[MotionVector]] = None) -> int:
    """
    运动向量分量解析
    规范文档 6.10.17 mv_component()
    
    Args:
        decoder: SymbolDecoder实例
        ref_idx: 参考帧索引（0或1）
        comp: 分量索引（0=行，1=列）
        mode_info: ModeInfo对象
        frame_header: 帧头
        MiRow: Mi行位置
        MiCol: Mi列位置
        MiSize: Mi尺寸
        AvailU: 上方是否可用
        AvailL: 左方是否可用
        mv_stack: 运动向量栈
        
    Returns:
        MV分量值（1/8像素精度）
    """
    # 如果mv_stack未提供，先查找
    if mv_stack is None:
        mv_stack = find_mv_stack(decoder, ref_idx, mode_info, frame_header,
                               MiRow, MiCol, MiSize, AvailU, AvailL)
    
    # 计算预测MV分量
    pred_mv = 0
    if len(mv_stack) > 0:
        # 使用栈中第一个MV作为预测
        pred_mv = mv_stack[0].row if comp == 0 else mv_stack[0].col
    
    # 解析mv_sign和mv_class
    # mv_sign (S())
    cdf = [1 << 14, 1 << 15, 0]  # 简化CDF
    mv_sign = read_symbol(decoder, cdf)
    
    # mv_class (S())
    # 简化处理：使用均匀CDF，实际应该根据上下文选择
    cdf = [1 << 14] * 11 + [1 << 15, 0]  # MV_CLASSES = 11
    mv_class = read_symbol(decoder, cdf)
    
    # 根据mv_class确定mv_fr和mv_hp
    if mv_class == 0:
        mv_fr = 0
        mv_hp = 0
    else:
        # mv_fr (S())
        cdf = [1 << 14] * 4 + [1 << 15, 0]  # 简化CDF
        mv_fr = read_symbol(decoder, cdf)
        
        # mv_hp (S())
        cdf = [1 << 14] * 2 + [1 << 15, 0]  # 简化CDF
        mv_hp = read_symbol(decoder, cdf)
    
    # 计算MV分量值
    # 简化处理：根据mv_class、mv_fr、mv_hp计算
    # 实际应该使用规范文档中的查找表和计算方式
    mv_mag = 0
    if mv_class > 0:
        # 根据mv_class计算基础magnitude
        base = [0, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]  # 简化查找表
        if mv_class < len(base):
            mv_mag = base[mv_class]
        
        # 添加mv_fr和mv_hp
        mv_mag += mv_fr * 2
        mv_mag += mv_hp
    
    # 应用符号
    if mv_sign:
        mv_mag = -mv_mag
    
    # 添加预测值
    mv_value = pred_mv + mv_mag
    
    # 裁剪MV（如果需要）
    # 简化处理：不裁剪，实际应该根据边界裁剪
    # mv_value = clamp_mv_row(mv_value, ...) 或 clamp_mv_col(mv_value, ...)
    
    return mv_value

