"""
解码过程模块
实现规范文档7节"Decoding process"中描述的所有解码过程函数
"""

from typing import List, Optional
from copy import deepcopy
# 参考帧相关常量
from constants import (
    FILTER_BITS, FRAME_RESTORATION_TYPE, FRAME_TYPE, REF_FRAME,
    NUM_REF_FRAMES, REFS_PER_FRAME, Y_MODE, Ref_Frame_List,
)
# 块尺寸相关常量
# MI尺寸相关常量
from constants import MI_SIZE, MI_SIZE_LOG2
# Loop Restoration相关常量
from constants import FRAME_RESTORATION_TYPE
# Superres相关常量
from constants import (
    SUPERRES_SCALE_BITS, SUPERRES_SCALE_MASK,
    SUPERRES_EXTRA_BITS, SUPERRES_FILTER_TAPS, SUPERRES_FILTER_OFFSET
)
# CDEF相关常量
# 参考MV相关常量
from constants import REFMVS_LIMIT
# 查找表
from constants import Upscale_Filter
from obu.decoder import AV1Decoder
from entropy.symbol_decoder import save_cdfs
from utils.math_utils import Array
from utils.math_utils import Clip1, Clip3, Round2


def large_scale_tile_decoding_process(av1: AV1Decoder):
    """
    大规模Tile解码过程
    规范文档 7.3 Large scale tile decoding process
    """
    pass


def decode_frame_wrapup(av1: AV1Decoder):
    """
    解码帧包装过程
    规范文档 7.4 Decode frame wrapup process
    """
    frame_header = av1.frame_header
    tile_group = av1.tile_group
    MiRows = frame_header.MiRows
    MiCols = frame_header.MiCols

    if frame_header.show_existing_frame == 0:
        # 1.
        if frame_header.loop_filter_level[0] != 0 or frame_header.loop_filter_level[1] != 0:
            loop_filter_process(av1)

        # 2.
        CdefFrame = cdef_process(av1)

        # 3.
        av1.UpscaledCdefFrame = upscaling_process(av1, deepcopy(CdefFrame))

        # 4.
        av1.UpscaledCurrFrame = upscaling_process(av1, deepcopy(av1.CurrFrame))

        # 5.
        av1.LrFrame = loop_restoration_process(
            av1, av1.UpscaledCurrFrame, av1.UpscaledCdefFrame)

        # 6.
        motion_field_motion_vector_storage_process(av1)

        # 7.
        if frame_header.segmentation_enabled == 1 and frame_header.segmentation_update_map == 0:
            for row in range(MiRows):
                for col in range(MiCols):
                    tile_group.SegmentIds[row][col] = frame_header.PrevSegmentIds[row][col]
    else:
        if frame_header.frame_type == FRAME_TYPE.KEY_FRAME:
            reference_frame_loading_process(av1)

    # 1.
    reference_frame_update_process(av1)

    # 2.
    if frame_header.show_frame == 1 or frame_header.show_existing_frame == 1:
        output_process(av1)


def frame_end_update_cdf(av1: AV1Decoder):
    """
    帧结束更新CDF过程
    规范文档 7.7 Frame end update CDF process
    """
    decoder = av1.decoder
    ref_frame_store = av1.ref_frame_store

    for name, cdf in decoder.saved_cdfs.items():
        name2 = name[5:]
        ref_frame_store.cdfs[name2] = cdf


def set_frame_refs(av1: AV1Decoder):
    """
    设置帧参考
    规范文档 7.8 Set frame refs process
    """
    from utils.frame_utils import get_relative_dist
    seq_header = av1.seq_header
    frame_header = av1.frame_header
    ref_frame_store = av1.ref_frame_store
    ref_frame_idx = frame_header.ref_frame_idx

    for i in range(REFS_PER_FRAME):
        ref_frame_idx[i] = -1
    ref_frame_idx[REF_FRAME.LAST_FRAME -
                  REF_FRAME.LAST_FRAME] = frame_header.last_frame_idx
    ref_frame_idx[REF_FRAME.GOLDEN_FRAME -
                  REF_FRAME.LAST_FRAME] = frame_header.gold_frame_idx

    usedFrame = [0] * NUM_REF_FRAMES
    usedFrame[frame_header.last_frame_idx] = 1
    usedFrame[frame_header.gold_frame_idx] = 1

    curFrameHint = 1 << (seq_header.OrderHintBits - 1)

    shiftedOrderHints = [0] * NUM_REF_FRAMES
    for i in range(NUM_REF_FRAMES):
        shiftedOrderHints[i] = curFrameHint + get_relative_dist(
            av1, ref_frame_store.RefOrderHint[i], frame_header.OrderHint)

    lastOrderHint = shiftedOrderHints[frame_header.last_frame_idx]
    # It is a requirement of bitstream conformance that lastOrderHint is strictly less than curFrameHint.
    assert lastOrderHint < curFrameHint

    goldOrderHint = shiftedOrderHints[frame_header.gold_frame_idx]
    # It is a requirement of bitstream conformance that goldOrderHint is strictly less than curFrameHint.
    assert goldOrderHint < curFrameHint

    def find_latest_backward():
        ref = -1
        latestOrderHint = 0
        for i in range(NUM_REF_FRAMES):
            hint = shiftedOrderHints[i]
            if not usedFrame[i] and hint >= curFrameHint and (ref < 0 or hint >= latestOrderHint):
                ref = i
                latestOrderHint = hint
        return ref

    def find_earliest_backward():
        ref = -1
        earliestOrderHint = 0
        for i in range(NUM_REF_FRAMES):
            hint = shiftedOrderHints[i]
            if not usedFrame[i] and hint >= curFrameHint and (ref < 0 or hint < earliestOrderHint):
                ref = i
                earliestOrderHint = hint
        return ref

    def find_latest_forward():
        ref = -1
        latestOrderHint = 0
        for i in range(NUM_REF_FRAMES):
            hint = shiftedOrderHints[i]
            if not usedFrame[i] and hint < curFrameHint and (ref < 0 or hint >= latestOrderHint):
                ref = i
                latestOrderHint = hint
        return ref

    ref = find_latest_backward()
    if ref >= 0:
        ref_frame_idx[REF_FRAME.ALTREF_FRAME - REF_FRAME.LAST_FRAME] = ref
        usedFrame[ref] = 1

    ref = find_earliest_backward()
    if ref >= 0:
        ref_frame_idx[REF_FRAME.BWDREF_FRAME - REF_FRAME.LAST_FRAME] = ref
        usedFrame[ref] = 1

    ref = find_earliest_backward()
    if ref >= 0:
        ref_frame_idx[REF_FRAME.ALTREF2_FRAME - REF_FRAME.LAST_FRAME] = ref
        usedFrame[ref] = 1

    for i in range(REFS_PER_FRAME - 2):
        refFrame = Ref_Frame_List[i]
        if ref_frame_idx[refFrame - REF_FRAME.LAST_FRAME] < 0:
            ref = find_latest_forward()
            if ref >= 0:
                ref_frame_idx[refFrame - REF_FRAME.LAST_FRAME] = ref
                usedFrame[ref] = 1

    ref = -1
    earliestOrderHint = 0
    for i in range(NUM_REF_FRAMES):
        hint = shiftedOrderHints[i]
        if ref < 0 or hint < earliestOrderHint:
            ref = i
            earliestOrderHint = hint

    for i in range(REFS_PER_FRAME):
        if ref_frame_idx[i] < 0:
            ref_frame_idx[i] = ref


def motion_field_estimation(av1: AV1Decoder) -> List[List[List[List[int]]]]:
    """
    运动场估计过程
    规范文档 7.9 Motion field estimation process
    """
    from mode.motion_field_estimation import MotionFieldEstimation
    motion_field_estimation_impl = MotionFieldEstimation(av1)
    return motion_field_estimation_impl.motion_field_estimation(av1)


def find_mv_stack(av1: AV1Decoder, isCompound: int):
    """
    查找MV栈过程
    规范文档 7.10 Find MV stack process

    Args:
        isCompound: 是否复合预测
    """
    from mode.find_mv_stack import FindMvStack
    find_mv_stack_impl = FindMvStack()
    return find_mv_stack_impl.find_mv_stack(av1, isCompound)


def predict_intra(av1: AV1Decoder, plane: int, x: int, y: int,
                  haveLeft: Optional[int], haveAbove: Optional[int],
                  haveAboveRight: int, haveBelowLeft: int,
                  mode: Y_MODE, log2W: int, log2H: int):
    """
    帧内预测过程
    规范文档 7.11.2 Intra prediction process
    """
    prediction = av1.tile_group.prediction
    return prediction.predict_intra(av1, plane, x, y, haveLeft, haveAbove, haveAboveRight, haveBelowLeft, mode, log2W, log2H)


def predict_inter(av1: AV1Decoder, plane: int, x: int, y: int,
                  w: int, h: int, candRow: int, candCol: int):
    """
    帧间预测过程
    规范文档 7.11.3 Inter prediction process
    """
    prediction = av1.tile_group.prediction
    return prediction.predict_inter(av1, plane, x, y, w, h, candRow, candCol)


def predict_palette(av1: AV1Decoder, plane: int, startX: int, startY: int, x: int, y: int, txSz: int):
    """
    调色板预测过程
    规范文档 7.11.4 Palette prediction process
    """
    prediction = av1.tile_group.prediction
    return prediction.predict_palette(av1, plane, startX, startY, x, y, txSz)


def predict_chroma_from_luma(av1: AV1Decoder, plane: int, startX: int, startY: int, txSz: int):
    """
    Chroma from Luma (CfL) 预测过程
    规范文档 7.11.5 Chroma from luma prediction process

    Args:
        plane: 色度分量索引 (通常为1或2)
        startX: 块的左上角X坐标
        startY: 块的左上角Y坐标
        txSz: 变换块大小索引
    """
    prediction = av1.tile_group.prediction
    return prediction.predict_chroma_from_luma(av1, plane, startX, startY, txSz)


def reconstruct(av1: AV1Decoder, plane: int, startX: int, startY: int, txSz: int):
    """
    重建过程
    规范文档 7.12.3 Reconstruct process
    """
    from reconstruction.reconstruct import reconstruct_process
    reconstruct_process(av1, plane, startX, startY, txSz)


def loop_filter_process(av1: AV1Decoder):
    """
    环路滤波过程
    规范文档 7.14 Loop filter process
    """
    from frame.loop_filter import LoopFilter
    loopFilter = LoopFilter()
    loopFilter.loop_filter_process(av1)


def cdef_process(av1: AV1Decoder) -> List[List[List[int]]]:
    """
    CDEF过程
    规范文档 7.15 CDEF process

    Returns:
        CdefFrame - CDEF处理后的帧
    """
    from frame.cdef import CdefProcess
    cdef_process_impl = CdefProcess(av1)
    return cdef_process_impl.cdef_process(av1)


def upscaling_process(av1: AV1Decoder, frame: List[List[List[int]]]) -> List[List[List[int]]]:
    """
    上采样过程
    规范文档 7.16 Upscaling process

    Args:
        Frame: 输入帧（如果为None，使用CurrFrame）

    Returns:
        上采样后的帧
    """
    seq_header = av1.seq_header
    frame_header = av1.frame_header
    subsampling_x = seq_header.color_config.subsampling_x
    subsampling_y = seq_header.color_config.subsampling_y
    BitDepth = seq_header.color_config.BitDepth
    NumPlanes = seq_header.color_config.NumPlanes
    MiCols = frame_header.MiCols

    if frame_header.use_superres == 0:
        return frame

    outputFrame: List[List[List[int]]] = Array(
        None, (NumPlanes, frame_header.FrameHeight, frame_header.UpscaledWidth), 0)
    for plane in range(NumPlanes):
        if plane > 0:
            subX = subsampling_x
            subY = subsampling_y
        else:
            subX = 0
            subY = 0

        downscaledPlaneW = Round2(frame_header.FrameWidth, subX)
        upscaledPlaneW = Round2(frame_header.UpscaledWidth, subX)

        # It is a requirement of bitstream conformance that upscaledPlaneW is strictly greater than downscaledPlaneW.
        assert upscaledPlaneW > downscaledPlaneW

        planeH = Round2(frame_header.FrameHeight, subY)

        stepX = ((downscaledPlaneW << SUPERRES_SCALE_BITS) +
                 (upscaledPlaneW // 2)) // upscaledPlaneW
        err = (upscaledPlaneW * stepX -
               (downscaledPlaneW << SUPERRES_SCALE_BITS))
        initialSubpelX = (((-((upscaledPlaneW - downscaledPlaneW) << (SUPERRES_SCALE_BITS - 1)) + (upscaledPlaneW // 2)) // upscaledPlaneW) +
                          (1 << (SUPERRES_EXTRA_BITS - 1)) - (err // 2))
        initialSubpelX &= SUPERRES_SCALE_MASK

        miW = MiCols >> subX
        minX = 0
        maxX = miW * MI_SIZE - 1

        for y in range(planeH):
            for x in range(upscaledPlaneW):
                srcX = -(1 << SUPERRES_SCALE_BITS) + initialSubpelX + x * stepX
                srcXPx = srcX >> SUPERRES_SCALE_BITS
                srcXSubpel = (
                    srcX & SUPERRES_SCALE_MASK) >> SUPERRES_EXTRA_BITS

                sum_val = 0
                for k in range(SUPERRES_FILTER_TAPS):
                    sampleX = Clip3(minX, maxX, srcXPx +
                                    (k - SUPERRES_FILTER_OFFSET))
                    px = frame[plane][y][sampleX]
                    sum_val += px * Upscale_Filter[srcXSubpel][k]

                outputFrame[plane][y][x] = Clip1(
                    Round2(sum_val, FILTER_BITS), BitDepth)

    return outputFrame


def loop_restoration_process(av1: AV1Decoder, UpscaledCurrFrame: List[List[List[int]]], UpscaledCdefFrame: List[List[List[int]]]) -> List[List[List[int]]]:
    """
    环路恢复过程
    规范文档 7.17 Loop restoration process

    Args:
        UpscaledCurrFrame: 上采样后的当前帧
        UpscaledCdefFrame: 上采样后的CDEF帧

    Returns:
        LrFrame - 环路恢复后的帧
    """
    seq_header = av1.seq_header
    frame_header = av1.frame_header
    NumPlanes = seq_header.color_config.NumPlanes

    LrFrame = deepcopy(UpscaledCdefFrame)
    if frame_header.UsesLr == 0:
        return LrFrame

    from frame.loop_restoration import LoopRestoration
    loop_restoration_impl = LoopRestoration(LrFrame)
    for y in range(0, frame_header.FrameHeight, MI_SIZE):
        for x in range(0, frame_header.UpscaledWidth, MI_SIZE):
            for plane in range(NumPlanes):
                if frame_header.FrameRestorationType[plane] != FRAME_RESTORATION_TYPE.RESTORE_NONE:
                    row = y >> MI_SIZE_LOG2
                    col = x >> MI_SIZE_LOG2
                    loop_restoration_impl.loop_restore_block(
                        av1, plane, row, col)

    return loop_restoration_impl.LrFrame


def output_process(av1: AV1Decoder):
    """
    输出过程
    规范文档 7.18 Output process
    """
    seq_header = av1.seq_header
    frame_header = av1.frame_header
    film_grain_params = frame_header.film_grain_params

    if seq_header.OperatingPointIdc != 0:
        pass  # TODO: 实现多操作点支持

    from frame.output_process import FilmGrainSynthesisProcess, intermediate_output_preparation
    film_grain_synthesis = FilmGrainSynthesisProcess(av1)
    w, h, subX, subY = intermediate_output_preparation(
        av1, film_grain_synthesis)

    if seq_header.film_grain_params_present == 1 and film_grain_params.apply_grain == 1:
        film_grain_synthesis.film_grain_synthesis(av1, w, h, subX, subY)


def motion_field_motion_vector_storage_process(av1: AV1Decoder):
    """
    运动场运动向量存储过程
    规范文档 7.19 Motion field motion vector storage process
    """
    frame_header = av1.frame_header
    tile_group = av1.tile_group
    ref_frame_store = av1.ref_frame_store
    OrderHint = frame_header.OrderHint
    ref_frame_idx = frame_header.ref_frame_idx
    MiRows = frame_header.MiRows
    MiCols = frame_header.MiCols

    tile_group.MfRefFrames = Array(None, (MiRows, MiCols), REF_FRAME.NONE)
    tile_group.MfMvs = Array(None, (MiRows, MiCols, 2), 0)
    for row in range(MiRows):
        for col in range(MiCols):
            for list in range(2):
                r = tile_group.RefFrames[row][col][list]
                if r > REF_FRAME.INTRA_FRAME:
                    refIdx = ref_frame_idx[r - REF_FRAME.LAST_FRAME]
                    from utils.frame_utils import get_relative_dist
                    dist = get_relative_dist(
                        av1, ref_frame_store.RefOrderHint[refIdx], OrderHint)
                    if dist < 0:
                        mvRow = tile_group.Mvs[row][col][list][0]
                        mvCol = tile_group.Mvs[row][col][list][1]
                        if abs(mvRow) <= REFMVS_LIMIT and abs(mvCol) <= REFMVS_LIMIT:
                            tile_group.MfRefFrames[row][col] = r
                            tile_group.MfMvs[row][col][0] = mvRow
                            tile_group.MfMvs[row][col][1] = mvCol


def reference_frame_update_process(av1: AV1Decoder):
    """
    参考帧更新过程
    规范文档 7.20 Reference frame update process
    """
    from frame.frame_header import save_loop_filter_params, save_segmentation_params
    seq_header = av1.seq_header
    frame_header = av1.frame_header
    tile_group = av1.tile_group
    subsampling_x = seq_header.color_config.subsampling_x
    subsampling_y = seq_header.color_config.subsampling_y
    BitDepth = seq_header.color_config.BitDepth
    MiRows = frame_header.MiRows
    MiCols = frame_header.MiCols
    gm_params = frame_header.gm_params
    ref_frame_store = av1.ref_frame_store

    for i in range(NUM_REF_FRAMES):
        if ((frame_header.refresh_frame_flags >> i) & 1) == 1:
            ref_frame_store.RefValid[i] = 1
            ref_frame_store.RefFrameId[i] = frame_header.current_frame_id
            ref_frame_store.RefUpscaledWidth[i] = frame_header.UpscaledWidth
            ref_frame_store.RefFrameWidth[i] = frame_header.FrameWidth
            ref_frame_store.RefFrameHeight[i] = frame_header.FrameHeight
            ref_frame_store.RefRenderWidth[i] = frame_header.RenderWidth
            ref_frame_store.RefRenderHeight[i] = frame_header.RenderHeight
            ref_frame_store.RefMiCols[i] = frame_header.MiCols
            ref_frame_store.RefMiRows[i] = frame_header.MiRows
            ref_frame_store.RefFrameType[i] = frame_header.frame_type

            ref_frame_store.RefSubsamplingX[i] = subsampling_x
            ref_frame_store.RefSubsamplingY[i] = subsampling_y
            ref_frame_store.RefBitDepth[i] = BitDepth
            ref_frame_store.RefOrderHint[i] = frame_header.OrderHint

            ref_frame_store.SavedOrderHints[i] = deepcopy(
                frame_header.OrderHints)
            ref_frame_store.FrameStore[i] = deepcopy(av1.LrFrame)
            ref_frame_store.SavedRefFrames[i] = deepcopy(
                tile_group.MfRefFrames)
            ref_frame_store.SavedMvs[i] = deepcopy(tile_group.MfMvs)
            ref_frame_store.SavedGmParams[i] = deepcopy(gm_params)
            ref_frame_store.SavedSegmentIds[i] = deepcopy(
                tile_group.SegmentIds)

            save_cdfs(av1, i)
            if seq_header.film_grain_params_present == 1:
                from .frame_header import save_grain_params
                save_grain_params(av1, i)
            save_loop_filter_params(av1, i)
            save_segmentation_params(av1, i)

            ref_frame_store.RefShowableFrame[i] = frame_header.showable_frame


def reference_frame_loading_process(av1: AV1Decoder):
    """
    参考帧加载过程
    规范文档 7.21 Reference frame loading process

    这是参考帧更新过程的逆过程。它将保存的参考帧值加载回当前帧变量。
    要加载的保存参考帧索引由语法元素frame_to_show_map_idx给出。
    """
    seq_header = av1.seq_header
    frame_header = av1.frame_header
    tile_group = av1.tile_group
    frame_to_show_map_idx = frame_header.frame_to_show_map_idx
    ref_frame_store = av1.ref_frame_store
    decoder = av1.decoder

    # 1.
    frame_header.current_frame_id = ref_frame_store.RefFrameId[frame_to_show_map_idx]

    # 2.
    frame_header.UpscaledWidth = ref_frame_store.RefUpscaledWidth[frame_to_show_map_idx]

    # 3.
    frame_header.FrameWidth = ref_frame_store.RefFrameWidth[frame_to_show_map_idx]

    # 4.
    frame_header.FrameHeight = ref_frame_store.RefFrameHeight[frame_to_show_map_idx]

    # 5.
    frame_header.RenderWidth = ref_frame_store.RefRenderWidth[frame_to_show_map_idx]

    # 6.
    frame_header.RenderHeight = ref_frame_store.RefRenderHeight[frame_to_show_map_idx]

    # 7.
    frame_header.MiCols = ref_frame_store.RefMiCols[frame_to_show_map_idx]

    # 8.
    frame_header.MiRows = ref_frame_store.RefMiRows[frame_to_show_map_idx]

    # 9.
    seq_header.color_config.subsampling_x = ref_frame_store.RefSubsamplingX[
        frame_to_show_map_idx]

    # 10.
    seq_header.color_config.subsampling_y = ref_frame_store.RefSubsamplingY[
        frame_to_show_map_idx]

    # 11.
    seq_header.color_config.BitDepth = ref_frame_store.RefBitDepth[frame_to_show_map_idx]

    # 12.
    frame_header.OrderHint = ref_frame_store.RefOrderHint[frame_to_show_map_idx]

    # 13.
    frame_header.OrderHints = deepcopy(
        ref_frame_store.SavedOrderHints[frame_to_show_map_idx])

    # 14.
    av1.LrFrame = deepcopy(ref_frame_store.FrameStore[frame_to_show_map_idx])

    from tile.tile_group import TileGroup
    tile_group = TileGroup(av1)
    av1.tile_group = tile_group
    # 16.
    tile_group.MfRefFrames = deepcopy(
        ref_frame_store.SavedRefFrames[frame_to_show_map_idx])

    # 17.
    tile_group.MfMvs = deepcopy(
        ref_frame_store.SavedMvs[frame_to_show_map_idx])

    # 18.
    frame_header.gm_params = deepcopy(
        ref_frame_store.SavedGmParams[frame_to_show_map_idx])

    # 19.
    tile_group.SegmentIds = deepcopy(
        ref_frame_store.SavedSegmentIds[frame_to_show_map_idx])

    # 20.
    from frame.frame_header import load_cdfs
    load_cdfs(av1, frame_to_show_map_idx)

    # 21.
    if seq_header.film_grain_params_present == 1:
        from .frame_header import load_grain_params
        load_grain_params(av1, frame_to_show_map_idx)

    # 22.
    from .frame_header import load_loop_filter_params
    load_loop_filter_params(av1, frame_to_show_map_idx)

    # 23.
    from .frame_header import load_segmentation_params
    load_segmentation_params(av1, frame_to_show_map_idx)
