"""
参考帧存储数据结构
用于保存和加载参考帧数据，对应规范文档7.20和7.21节
"""

from typing import Dict, List, Optional, Any
from constants import (
    FRAME_TYPE, NONE, NUM_REF_FRAMES, REF_FRAME
)


class RefFrameStore:
    """
    参考帧存储数据结构
    用于保存reference_frame_update_process中保存的所有参考帧数据
    对应规范文档7.20 Reference frame update process

    这个数据结构保存了NUM_REF_FRAMES个参考帧的所有信息
    """

    def __init__(self):
        # 参考帧有效性标志 [NUM_REF_FRAMES]
        self.RefValid: List[int] = NONE

        # 参考帧ID [NUM_REF_FRAMES]
        self.RefFrameId: List[int] = [NONE] * NUM_REF_FRAMES

        # 参考帧尺寸信息 [NUM_REF_FRAMES]
        self.RefUpscaledWidth: List[int] = [NONE] * NUM_REF_FRAMES
        self.RefFrameWidth: List[int] = [NONE] * NUM_REF_FRAMES
        self.RefFrameHeight: List[int] = [NONE] * NUM_REF_FRAMES
        self.RefRenderWidth: List[int] = [NONE] * NUM_REF_FRAMES
        self.RefRenderHeight: List[int] = [NONE] * NUM_REF_FRAMES
        self.RefMiCols: List[int] = [NONE] * NUM_REF_FRAMES
        self.RefMiRows: List[int] = [NONE] * NUM_REF_FRAMES

        # 参考帧类型 [NUM_REF_FRAMES]
        self.RefFrameType: List[FRAME_TYPE] = [NONE] * NUM_REF_FRAMES

        # 参考帧颜色配置 [NUM_REF_FRAMES]
        self.RefSubsamplingX: List[int] = [NONE] * NUM_REF_FRAMES
        self.RefSubsamplingY: List[int] = [NONE] * NUM_REF_FRAMES
        self.RefBitDepth: List[int] = [NONE] * NUM_REF_FRAMES

        # 参考帧OrderHint [NUM_REF_FRAMES]
        self.RefOrderHint: List[int] = NONE

        # 保存的OrderHints [NUM_REF_FRAMES][TOTAL_REFS_PER_FRAME]
        self.SavedOrderHints: List[List[int]] = [NONE] * NUM_REF_FRAMES

        # 帧存储数据 [NUM_REF_FRAMES][PLANE_MAX][height][width]
        # 注意：实际尺寸根据每帧的UpscaledWidth和FrameHeight动态分配
        # 这里使用None表示未初始化，实际使用时需要根据帧尺寸动态分配
        self.FrameStore: List[List[List[List[int]]]] = [
            NONE] * NUM_REF_FRAMES

        # 保存的运动场参考帧 [NUM_REF_FRAMES][MiRows][MiCols]
        # 注意：实际尺寸根据每帧的MiRows和MiCols动态分配
        self.SavedRefFrames: List[List[List[REF_FRAME]]] = [
            NONE] * NUM_REF_FRAMES

        # 保存的运动向量 [NUM_REF_FRAMES][MiRows][MiCols][2]
        # 注意：实际尺寸根据每帧的MiRows和MiCols动态分配
        self.SavedMvs: List[List[List[List[int]]]] = [
            NONE] * NUM_REF_FRAMES

        # 保存的全局运动参数 [NUM_REF_FRAMES][TOTAL_REFS_PER_FRAME][6]
        self.SavedGmParams: List[List[List[int]]] = [NONE] * NUM_REF_FRAMES

        # 保存的分段ID [NUM_REF_FRAMES][MAX_TILE_ROWS][MAX_TILE_COLS]
        self.SavedSegmentIds: List[List[List[int]]] = [NONE] * NUM_REF_FRAMES

        # CDF数据（通过SymbolDecoder保存和加载）
        self.SavedCdfs: List[Dict[str, Any]] = [NONE] * NUM_REF_FRAMES

        # Film Grain参数（通过save_grain_params和load_grain_params处理）
        from frame.frame_header import FilmGrainParams
        self.SavedFilmGrainParams: List[FilmGrainParams] = [
            NONE] * NUM_REF_FRAMES

        # Loop Filter参数（通过save_loop_filter_params和load_loop_filter_params处理）
        self.SavedLoopFilterRefDeltas: List[List[int]] = [
            NONE] * NUM_REF_FRAMES
        self.SavedLoopFilterModeDeltas: List[List[int]] = [
            NONE] * NUM_REF_FRAMES

        # Segmentation参数（通过save_segmentation_params和load_segmentation_params处理）
        self.SavedFeatureEnabled: List[List[List[int]]] = [NONE] * NUM_REF_FRAMES
        self.SavedFeatureData: List[List[List[int]]] = [NONE] * NUM_REF_FRAMES

        # 参考帧符号偏置 [NUM_REF_FRAMES]
        self.RefFrameSignBias: List[int] = [NONE] * NUM_REF_FRAMES

        # 参考帧显示性标志 [NUM_REF_FRAMES]
        self.RefShowableFrame: List[int] = [NONE] * NUM_REF_FRAMES

        # CDF数据（通过SymbolDecoder保存和加载）
        self.cdfs: Dict[str, Any] = {}
