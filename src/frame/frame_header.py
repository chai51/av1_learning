"""
帧头OBU解析器
按照规范文档6.8节实现frame_header_obu()
"""

from constants import GM_TYPE, MAX_LOOP_FILTER, NONE
from typing import Optional, List, Any
from copy import deepcopy
from bitstream.descriptors import read_f, read_su, read_ns
from utils.frame_utils import inverse_recenter
from obu.decoder import AV1Decoder
from constants import OBU_HEADER_TYPE, FRAME_RESTORATION_TYPE, INTERPOLATION_FILTER, TX_MODE
from constants import FRAME_LF_COUNT, REF_FRAME
from constants import GM_ABS_ALPHA_BITS, GM_ABS_TRANS_BITS, GM_ABS_TRANS_ONLY_BITS, GM_ALPHA_PREC_BITS, GM_TRANS_ONLY_PREC_BITS, GM_TRANS_PREC_BITS, PLANE_MAX, SEG_LVL_REF_FRAME
from constants import SEG_LVL_MAX, MAX_SEGMENTS, MV_CONTEXTS
from constants import TOTAL_REFS_PER_FRAME, REFS_PER_FRAME
from constants import FRAME_TYPE, NUM_REF_FRAMES
from constants import SELECT_SCREEN_CONTENT_TOOLS, SELECT_INTEGER_MV, PRIMARY_REF_NONE
from constants import SUPERRES_DENOM_BITS, SUPERRES_DENOM_MIN, SUPERRES_NUM
from constants import REF_FRAME
from constants import FRAME_RESTORATION_TYPE, RESTORATION_TILESIZE_MAX
from constants import MAX_TILE_WIDTH, MAX_TILE_AREA, MAX_TILE_COLS, MAX_TILE_ROWS
from constants import WARPEDMODEL_PREC_BITS
from entropy import default_cdfs
from frame.decoding_process import decode_frame_wrapup
from utils.math_utils import Array, Clip3
from entropy.symbol_decoder import inverseCdf


class FilmGrainParams:
    """
    Film Grain参数结构
    规范文档 5.9.25 film_grain_params()
    """

    def __init__(self):
        self.apply_grain: Optional[int] = None
        self.grain_seed: int = NONE
        self.num_y_points: int = NONE
        self.point_y_value: List[int] = [NONE] * 15
        self.point_y_scaling: List[int] = [NONE] * 15
        self.chroma_scaling_from_luma: Optional[int] = None
        self.num_cb_points: int = NONE
        self.point_cb_value: List[int] = [NONE] * 11
        self.point_cb_scaling: List[int] = [NONE] * 11
        self.num_cr_points: int = NONE
        self.point_cr_value: List[int] = [NONE] * 11
        self.point_cr_scaling: List[int] = [NONE] * 11
        self.grain_scaling_minus_8: int = NONE
        self.ar_coeff_lag: int = NONE
        self.ar_coeffs_y_plus_128: List[int] = [NONE] * 25
        self.ar_coeffs_cb_plus_128: List[int] = [NONE] * 25
        self.ar_coeffs_cr_plus_128: List[int] = [NONE] * 25
        self.ar_coeff_shift_minus_6: int = NONE
        self.grain_scale_shift: int = NONE
        self.cb_mult: int = NONE
        self.cb_offset: int = NONE
        self.cb_luma_mult: int = NONE
        self.cr_mult: int = NONE
        self.cr_luma_mult: int = NONE
        self.cr_offset: int = NONE
        self.overlap_flag: Optional[int] = None
        self.clip_to_restricted_range: Optional[int] = None


class FrameHeader:
    """
    帧头结构
    规范文档 6.8.1 uncompressed_header()
    """

    def __init__(self):
        # TileNum - 当前Tile索引
        self.TileNum: Optional[int] = None

        # 基础信息
        self.show_existing_frame: Optional[int] = None
        self.frame_to_show_map_idx: int = NONE
        self.frame_type: FRAME_TYPE = NONE
        self.show_frame: Optional[int] = None
        self.showable_frame = 0
        self.error_resilient_mode: Optional[int] = None

        # 功能标志
        self.disable_cdf_update: Optional[int] = None
        self.current_frame_id: int = NONE
        self.frame_size_override_flag: Optional[int] = None
        self.OrderHint = 0
        self.primary_ref_frame = 0
        self.allow_screen_content_tools: Optional[int] = None

        # 内联BC相关
        self.allow_intrabc: Optional[int] = None

        # 整数MV相关
        self.force_integer_mv: Optional[int] = None

        # 参考帧刷新
        self.refresh_frame_flags: int = NONE

        # 参考帧索引
        self.last_frame_idx: int = NONE
        self.gold_frame_idx: int = NONE
        self.ref_frame_idx: List[int] = [NONE] * TOTAL_REFS_PER_FRAME

        # 高精度MV相关
        self.allow_high_precision_mv = 0

        # 运动模式可切换
        self.is_motion_mode_switchable: Optional[int] = None

        # 使用参考帧MV
        self.use_ref_frame_mvs: Optional[int] = None

        # 禁止CDF更新
        self.disable_frame_end_update_cdf: Optional[int] = None

        # 当前帧的OrderHints
        self.OrderHints: List[int] = [NONE] * TOTAL_REFS_PER_FRAME

        # 无损相关
        self.CodedLossless: Optional[int] = None
        self.AllLossless: Optional[int] = None

        # 允许扭曲运动
        self.allow_warped_motion: Optional[int] = None

        self.reduced_tx_set: Optional[int] = None

        # 超分辨率相关
        self.use_superres: Optional[int] = None
        self.SuperresDenom: int = NONE

        # MiCols和MiRows
        self.MiCols: int = NONE
        self.MiRows: int = NONE

        # 插值滤波
        self.interpolation_filter: INTERPOLATION_FILTER = NONE

        self.loop_filter_level: List[int] = [NONE] * FRAME_LF_COUNT
        self.loop_filter_sharpness: int = NONE
        self.loop_filter_delta_enabled: Optional[int] = None
        self.loop_filter_ref_deltas: List[int] = [NONE] * TOTAL_REFS_PER_FRAME
        self.loop_filter_mode_deltas: List[int] = [NONE] * 2

        # 帧ID是否存在
        self.frame_id_numbers_present_flag: Optional[int] = None

        # 帧是否为帧内
        self.FrameIsIntra: Optional[int] = None

        # 帧尺寸
        self.FrameWidth: int = NONE
        self.FrameHeight: int = NONE
        self.UpscaledWidth: int = NONE
        self.RenderWidth: int = NONE
        self.RenderHeight: int = NONE

        # Tile信息
        self.TileCols = 0
        self.TileRows = 0
        self.TileColsLog2 = 0
        self.TileRowsLog2 = 0
        self.MiColStarts: List[int] = [NONE] * MAX_TILE_COLS
        self.MiRowStarts: List[int] = [NONE] * MAX_TILE_ROWS
        self.TileSizeBytes: int = NONE

        # 量化参数
        self.base_q_idx = NONE
        self.DeltaQYDc: int = NONE
        self.DeltaQUDc: int = NONE
        self.DeltaQUAc: int = NONE
        self.DeltaQVDc: int = NONE
        self.DeltaQVAc: int = NONE
        self.using_qmatrix: Optional[int] = None
        self.qm_y: int = NONE
        self.qm_u: int = NONE
        self.qm_v: int = NONE

        # 分段参数
        self.SegIdPreSkip: Optional[int] = None
        self.LastActiveSegId = 0
        self.segmentation_enabled = 0
        self.segmentation_update_map: Optional[int] = None
        self.segmentation_temporal_update: Optional[int] = None

        # Tile信息
        self.context_update_tile_id: Optional[int] = None

        # delta_q_params
        self.delta_q_present: Optional[int] = None
        self.delta_q_res: int = NONE

        # delta_lf_params
        self.delta_lf_present: Optional[int] = None
        self.delta_lf_res: int = NONE
        self.delta_lf_multi: Optional[int] = None

        # Film grain参数
        self.film_grain_params: FilmGrainParams = FilmGrainParams()

        # skip模式帧
        self.SkipModeFrame: List[REF_FRAME] = [NONE, NONE]

        # skip模式是否存在
        self.skip_mode_present: Optional[int] = None

        # 参考模式
        self.reference_select: Optional[int] = None

        # CDEF参数
        self.cdef_bits: int = NONE
        self.cdef_y_pri_strength: List[int] = [NONE] * 8
        self.cdef_uv_pri_strength: List[int] = [NONE] * 8
        self.cdef_y_sec_strength: List[int] = [NONE] * 8
        self.cdef_uv_sec_strength: List[int] = [NONE] * 8

        self.FrameRestorationType: List[FRAME_RESTORATION_TYPE] = [
            NONE] * PLANE_MAX

        self.UsesLr: Optional[int] = None

        self.LoopRestorationSize: List[int] = [NONE] * PLANE_MAX

        # 特征相关
        self.FeatureEnabled: List[List[int]] = NONE
        self.FeatureData: List[List[int]] = NONE

        # 前一帧的GmParams
        self.PrevGmParams: List[List[int]] = Array(
            None, (TOTAL_REFS_PER_FRAME, 6))

        # 无损相关
        self.LosslessArray: List[int] = [NONE] * MAX_SEGMENTS

        self.SegQMLevel: List[List[int]] = Array(
            None, (PLANE_MAX, MAX_SEGMENTS))

        # 全局运动参数
        self.GmType: List[GM_TYPE] = [NONE] * TOTAL_REFS_PER_FRAME
        self.gm_params: List[List[int]] = Array(
            None, (TOTAL_REFS_PER_FRAME, 6))

        self.TxMode: Optional[TX_MODE] = None

        self.PrevSegmentIds: List[List[int]] = NONE

        self.MotionFieldMvs: List[List[List[List[int]]]] = NONE

        self.CdefDamping = 3


class FrameHeaderParser:
    """
    帧头解析器
    """

    def __init__(self):
        self.frame_header = FrameHeader()

    def frame_header_obu(self, av1: AV1Decoder) -> FrameHeader:
        """
        规范文档 5.9.1 General frame header OBU syntax
        """
        header = av1.obu.header
        frame_header = self.frame_header

        # It is a requirement of bitstream conformance that a sequence header OBU has been received before a frame header OBU.
        assert av1.seq_header is not None

        # If obu_type is equal to OBU_FRAME_HEADER or obu_type is equal to OBU_FRAME, it is a requirement of bitstream conformance that SeenFrameHeader is equal to 0.
        if (header.obu_type in [OBU_HEADER_TYPE.OBU_FRAME_HEADER, OBU_HEADER_TYPE.OBU_FRAME]):
            assert av1.SeenFrameHeader == 0
        # If obu_type is equal to OBU_REDUNDANT_FRAME_HEADER, it is a requirement of bitstream conformance that SeenFrameHeader is equal to 1.
        # Note: These requirements ensure that the first frame header for a frame has obu_type equal to OBU_FRAME_HEADER, while later copies of this frame header (if present) have obu_type equal to OBU_REDUNDANT_FRAME_HEADER.
        elif header.obu_type == OBU_HEADER_TYPE.OBU_REDUNDANT_FRAME_HEADER:
            assert av1.SeenFrameHeader == 1

        if av1.SeenFrameHeader == 1:
            self.__frame_header_copy(av1)
        else:
            av1.SeenFrameHeader = 1
            self.__uncompressed_header(av1)

            if frame_header.show_existing_frame:
                decode_frame_wrapup(av1)
                av1.SeenFrameHeader = 0
            else:
                frame_header.TileNum = 0
                av1.SeenFrameHeader = 1

        return frame_header

    def __uncompressed_header(self, av1: AV1Decoder):
        """
        解析未压缩帧头
        规范文档 5.9.2 Uncompressed header syntax
        """
        from utils.frame_utils import get_qindex
        reader = av1.reader
        header = av1.obu.header
        seq_header = av1.seq_header
        frame_header = self.frame_header
        ref_frame_store = av1.ref_frame_store

        idLen = 0
        if seq_header.frame_id_numbers_present_flag:
            idLen = (seq_header.additional_frame_id_length_minus_1 +
                     seq_header.delta_frame_id_length_minus_2 + 3)

        allFrames = (1 << NUM_REF_FRAMES) - 1
        if seq_header.reduced_still_picture_header:
            frame_header.show_existing_frame = 0
            frame_header.frame_type = FRAME_TYPE.KEY_FRAME
            frame_header.FrameIsIntra = 1
            frame_header.show_frame = 1
            frame_header.showable_frame = 0
        else:
            frame_header.show_existing_frame = read_f(reader, 1)
            # If obu_type is equal to OBU_FRAME, it is a requirement of bitstream conformance that show_existing_frame is equal to 0.
            if header.obu_type == OBU_HEADER_TYPE.OBU_FRAME:
                assert frame_header.show_existing_frame == 0

            if frame_header.show_existing_frame == 1:
                frame_header.frame_to_show_map_idx = read_f(reader, 3)
                if (seq_header.decoder_model_info_present_flag
                        and not seq_header.equal_picture_interval):
                    self.__temporal_point_info(av1)

                frame_header.refresh_frame_flags = 0
                if seq_header.frame_id_numbers_present_flag:
                    # It is a requirement of bitstream conformance that the number of bits needed to read display_frame_id does not exceed 16.  This is equivalent to the constraint that idLen <= 16.
                    assert idLen <= 16

                    display_frame_id = read_f(reader, idLen)

                    # It is a requirement of bitstream conformance that whenever display_frame_id is read, the value matches RefFrameId[ frame_to_show_map_idx ] (the value of current_frame_id at the time that the frame indexed by frame_to_show_map_idx was stored), and that RefValid[ frame_to_show_map_idx ] is equal to 1.
                    assert display_frame_id == ref_frame_store.RefFrameId[
                        frame_header.frame_to_show_map_idx]
                    assert ref_frame_store.RefValid[frame_header.frame_to_show_map_idx] == 1

                frame_header.frame_type = ref_frame_store.RefFrameType[
                    frame_header.frame_to_show_map_idx]
                if frame_header.frame_type == FRAME_TYPE.KEY_FRAME:
                    frame_header.refresh_frame_flags = allFrames
                if seq_header.film_grain_params_present:
                    load_grain_params(av1, frame_header.frame_to_show_map_idx)
                return

            frame_header.frame_type = FRAME_TYPE(read_f(reader, 2))
            frame_header.FrameIsIntra = (frame_header.frame_type in [
                                         FRAME_TYPE.INTRA_ONLY_FRAME, FRAME_TYPE.KEY_FRAME])
            frame_header.show_frame = read_f(reader, 1)
            if (frame_header.show_frame and
                seq_header.decoder_model_info_present_flag and
                    not seq_header.equal_picture_interval):
                self.__temporal_point_info(av1)
            if frame_header.show_frame:
                frame_header.showable_frame = (
                    frame_header.frame_type != FRAME_TYPE.KEY_FRAME)
            else:
                frame_header.showable_frame = read_f(reader, 1)

            # It is a requirement of bitstream conformance that when show_existing_frame is used to show a previous frame, that the value of showable_frame for the previous frame was equal to 1.
            if frame_header.show_existing_frame == 1:
                assert ref_frame_store.RefShowableFrame[frame_header.frame_to_show_map_idx] == 1
            # It is a requirement of bitstream conformance that when show_existing_frame is used to show a previous frame with RefFrameType[ frame_to_show_map_idx ] equal to KEY_FRAME, that the frame is output via the show_existing_frame mechanism at most once.

            if (frame_header.frame_type in [FRAME_TYPE.SWITCH_FRAME, FRAME_TYPE.KEY_FRAME] and frame_header.show_frame):
                frame_header.error_resilient_mode = 1
            else:
                frame_header.error_resilient_mode = read_f(reader, 1)

        if frame_header.frame_type == FRAME_TYPE.KEY_FRAME and frame_header.show_frame:
            ref_frame_store.RefValid = [0] * NUM_REF_FRAMES
            ref_frame_store.RefOrderHint = [0] * NUM_REF_FRAMES
            for i in range(REFS_PER_FRAME):
                frame_header.OrderHints[REF_FRAME.LAST_FRAME + i] = 0

        frame_header.disable_cdf_update = read_f(reader, 1)

        if seq_header.seq_force_screen_content_tools == SELECT_SCREEN_CONTENT_TOOLS:
            frame_header.allow_screen_content_tools = read_f(reader, 1)
        else:
            frame_header.allow_screen_content_tools = seq_header.seq_force_screen_content_tools
        if frame_header.allow_screen_content_tools:
            if seq_header.seq_force_integer_mv == SELECT_INTEGER_MV:
                frame_header.force_integer_mv = read_f(reader, 1)
            else:
                frame_header.force_integer_mv = seq_header.seq_force_integer_mv
        else:
            frame_header.force_integer_mv = 0

        if frame_header.FrameIsIntra:
            frame_header.force_integer_mv = 1

        if seq_header.frame_id_numbers_present_flag:
            PrevFrameID = frame_header.current_frame_id
            frame_header.current_frame_id = read_f(reader, idLen)

            # If frame_type is not equal to KEY_FRAME or show_frame is equal to 0, it is a requirement of bitstream conformance that all of the following conditions are true:
            # - current_frame_id is not equal to PrevFrameID
            # - DiffFrameID is less than 1 << ( idLen - 1 )
            if frame_header.frame_type != FRAME_TYPE.KEY_FRAME or frame_header.show_frame == 0:
                if frame_header.current_frame_id > PrevFrameID:
                    DiffFrameID = frame_header.current_frame_id - PrevFrameID
                else:
                    DiffFrameID = ((1 << idLen) +
                                   frame_header.current_frame_id - PrevFrameID)
                assert frame_header.current_frame_id != PrevFrameID
                assert DiffFrameID < (1 << (idLen - 1))

            self.__mark_ref_frames(av1, idLen)
        else:
            frame_header.current_frame_id = 0

        if frame_header.frame_type == FRAME_TYPE.SWITCH_FRAME:
            frame_header.frame_size_override_flag = 1
        elif seq_header.reduced_still_picture_header:
            frame_header.frame_size_override_flag = 0
        else:
            frame_header.frame_size_override_flag = read_f(reader, 1)

        order_hint = read_f(reader, seq_header.OrderHintBits)
        frame_header.OrderHint = order_hint

        if frame_header.FrameIsIntra or frame_header.error_resilient_mode:
            frame_header.primary_ref_frame = PRIMARY_REF_NONE
        else:
            frame_header.primary_ref_frame = read_f(reader, 3)

        if seq_header.decoder_model_info_present_flag:
            buffer_removal_time_present_flag = read_f(reader, 1)
            if buffer_removal_time_present_flag:
                for opNum in range(seq_header.operating_points_cnt_minus_1 + 1):
                    if seq_header.decoder_model_present_for_this_op[opNum]:
                        opPtIdc = seq_header.operating_point_idc[opNum]
                        inTemporalLayer = (opPtIdc >> header.temporal_id) & 1
                        inSpatialLayer = (opPtIdc >> (
                            header.spatial_id + 8)) & 1
                        if opPtIdc == 0 or (inTemporalLayer and inSpatialLayer):
                            n = seq_header.buffer_removal_time_length_minus_1 + 1
                            buffer_removal_time = read_f(reader, n)

        frame_header.allow_high_precision_mv = 0
        frame_header.use_ref_frame_mvs = 0
        frame_header.allow_intrabc = 0

        if (frame_header.frame_type in [FRAME_TYPE.SWITCH_FRAME, FRAME_TYPE.KEY_FRAME] and frame_header.show_frame):
            frame_header.refresh_frame_flags = allFrames
        else:
            frame_header.refresh_frame_flags = read_f(reader, 8)

        # If frame_type is equal to INTRA_ONLY_FRAME, it is a requirement of bitstream conformance that refresh_frame_flags is not equal to 0xff.
        if frame_header.frame_type == FRAME_TYPE.INTRA_ONLY_FRAME:
            assert frame_header.refresh_frame_flags != 0xff

        if not frame_header.FrameIsIntra or frame_header.refresh_frame_flags != allFrames:
            if frame_header.error_resilient_mode and seq_header.enable_order_hint:
                for i in range(NUM_REF_FRAMES):
                    ref_order_hint = read_f(reader, seq_header.OrderHintBits)
                    if ref_order_hint != ref_frame_store.RefOrderHint[i]:
                        ref_frame_store.RefValid[i] = 0

        if frame_header.FrameIsIntra:
            self.__frame_size(av1)
            self.__render_size(av1)
            if (frame_header.allow_screen_content_tools and
                    frame_header.UpscaledWidth == frame_header.FrameWidth):
                frame_header.allow_intrabc = read_f(reader, 1)
        else:
            if not seq_header.enable_order_hint:
                frame_refs_short_signaling = 0
            else:
                frame_refs_short_signaling = read_f(reader, 1)
                if frame_refs_short_signaling:
                    frame_header.last_frame_idx = read_f(reader, 3)
                    frame_header.gold_frame_idx = read_f(reader, 3)
                    from frame.decoding_process import set_frame_refs
                    set_frame_refs(av1)

            for i in range(REFS_PER_FRAME):
                if not frame_refs_short_signaling:
                    ref_frame_idx = read_f(reader, 3)
                    frame_header.ref_frame_idx[i] = ref_frame_idx

                    # It is a requirement of bitstream conformance that RefValid[ ref_frame_idx[ i ] ] is equal to 1, and that the selected reference frames match the current frame in bit depth, profile, chroma subsampling, and color space.
                    assert ref_frame_store.RefValid[ref_frame_idx] == 1
                    assert ref_frame_store.RefBitDepth[ref_frame_idx] == seq_header.color_config.BitDepth
                    assert ref_frame_store.RefSubsamplingX[ref_frame_idx] == seq_header.color_config.subsampling_x
                    assert ref_frame_store.RefSubsamplingY[ref_frame_idx] == seq_header.color_config.subsampling_y

                if seq_header.frame_id_numbers_present_flag:
                    n = seq_header.delta_frame_id_length_minus_2 + 2
                    delta_frame_id_minus_1 = read_f(reader, n)
                    DeltaFrameId = delta_frame_id_minus_1 + 1
                    expectedFrameId = ((frame_header.current_frame_id + (1 << idLen) -
                                        DeltaFrameId) % (1 << idLen))
                    # It is a requirement of bitstream conformance that whenever expectedFrameId[ i ] is calculated, the value matches RefFrameId[ ref_frame_idx[ i ] ] (this contains the value of current_frame_id at the time that the frame indexed by ref_frame_idx[ i ] was stored).
                    assert expectedFrameId == ref_frame_store.RefFrameId[frame_header.ref_frame_idx[i]]

            if frame_header.frame_size_override_flag and not frame_header.error_resilient_mode:
                self.__frame_size_with_refs(av1)
            else:
                self.__frame_size(av1)
                self.__render_size(av1)

            if frame_header.force_integer_mv:
                frame_header.allow_high_precision_mv = 0
            else:
                frame_header.allow_high_precision_mv = read_f(reader, 1)

            self.__read_interpolation_filter(av1)
            frame_header.is_motion_mode_switchable = read_f(reader, 1)

            if frame_header.error_resilient_mode or not seq_header.enable_ref_frame_mvs:
                frame_header.use_ref_frame_mvs = 0
            else:
                frame_header.use_ref_frame_mvs = read_f(reader, 1)

            for i in range(REFS_PER_FRAME):
                refFrame = REF_FRAME.LAST_FRAME + i
                hint = ref_frame_store.RefOrderHint[frame_header.ref_frame_idx[i]]
                frame_header.OrderHints[refFrame] = hint
                if not seq_header.enable_order_hint:
                    ref_frame_store.RefFrameSignBias[refFrame] = 0
                else:
                    from utils.frame_utils import get_relative_dist
                    ref_frame_store.RefFrameSignBias[refFrame] = get_relative_dist(
                        av1, hint, frame_header.OrderHint) > 0

        if seq_header.reduced_still_picture_header or frame_header.disable_cdf_update:
            frame_header.disable_frame_end_update_cdf = 1
        else:
            frame_header.disable_frame_end_update_cdf = read_f(reader, 1)

        if frame_header.primary_ref_frame == PRIMARY_REF_NONE:
            self.__init_non_coeff_cdfs(av1)
            self.__setup_past_independence(av1)
        else:
            load_cdfs(
                av1, frame_header.ref_frame_idx[frame_header.primary_ref_frame])
            self.__load_previous(av1)

        if frame_header.use_ref_frame_mvs == 1:
            from frame.decoding_process import motion_field_estimation
            frame_header.MotionFieldMvs = motion_field_estimation(av1)

        self.__tile_info(av1)
        self.__quantization_params(av1)
        self.__segmentation_params(av1)
        self.__delta_q_params(av1)
        self.__delta_lf_params(av1)

        if frame_header.primary_ref_frame == PRIMARY_REF_NONE:
            self.__init_coeff_cdfs(av1)
        else:
            self.__load_previous_segment_ids(av1)

        frame_header.CodedLossless = 1

        for segmentId in range(MAX_SEGMENTS):
            qindex = get_qindex(av1, 1, segmentId)
            frame_header.LosslessArray[segmentId] = (qindex == 0 and
                                                     frame_header.DeltaQYDc == 0 and
                                                     frame_header.DeltaQUAc == 0 and
                                                     frame_header.DeltaQUDc == 0 and
                                                     frame_header.DeltaQVAc == 0 and
                                                     frame_header.DeltaQVDc == 0)

            if not frame_header.LosslessArray[segmentId]:
                frame_header.CodedLossless = 0
            if frame_header.using_qmatrix:
                if frame_header.LosslessArray[segmentId]:
                    frame_header.SegQMLevel[0][segmentId] = 15
                    frame_header.SegQMLevel[1][segmentId] = 15
                    frame_header.SegQMLevel[2][segmentId] = 15
                else:
                    frame_header.SegQMLevel[0][segmentId] = frame_header.qm_y
                    frame_header.SegQMLevel[1][segmentId] = frame_header.qm_u
                    frame_header.SegQMLevel[2][segmentId] = frame_header.qm_v

        # It is a requirement of bitstream conformance that delta_q_present is equal to 0 when CodedLossless is equal to 1.
        if frame_header.CodedLossless == 1:
            assert frame_header.delta_q_present == 0

        frame_header.AllLossless = frame_header.CodedLossless and (
            frame_header.FrameWidth == frame_header.UpscaledWidth)

        self.__loop_filter_params(av1)
        self.__cdef_params(av1)
        self.__lr_params(av1)
        self.__read_tx_mode(av1)
        self.__frame_reference_mode(av1)
        self.__skip_mode_params(av1)

        if (frame_header.FrameIsIntra or
            frame_header.error_resilient_mode or
                not seq_header.enable_warped_motion):
            frame_header.allow_warped_motion = 0
        else:
            frame_header.allow_warped_motion = read_f(reader, 1)

        frame_header.reduced_tx_set = read_f(reader, 1)

        self.__global_motion_params(av1)
        self.__film_grain_params(av1)

    def __mark_ref_frames(self, av1: AV1Decoder, idLen: int):
        """
        标记参考帧
        规范文档 5.9.4 Reference frame marking function

        Args:
            idLen: 帧ID长度
        """
        seq_header = av1.seq_header
        frame_header = self.frame_header
        ref_frame_store = av1.ref_frame_store

        diffLen = seq_header.delta_frame_id_length_minus_2 + 2
        for i in range(NUM_REF_FRAMES):
            if frame_header.current_frame_id > (1 << diffLen):
                if (ref_frame_store.RefFrameId[i] > frame_header.current_frame_id or
                        ref_frame_store.RefFrameId[i] < (frame_header.current_frame_id - (1 << diffLen))):
                    ref_frame_store.RefValid[i] = 0
            else:
                if (ref_frame_store.RefFrameId[i] > frame_header.current_frame_id and
                        ref_frame_store.RefFrameId[i] < ((1 << idLen) + frame_header.current_frame_id - (1 << diffLen))):
                    ref_frame_store.RefValid[i] = 0

    def __frame_size(self, av1: AV1Decoder):
        """
        解析帧尺寸
        规范文档 5.9.5 Frame size syntax
        """
        reader = av1.reader
        seq_header = av1.seq_header
        frame_header = self.frame_header
        ref_frame_store = av1.ref_frame_store

        if frame_header.frame_size_override_flag:
            n = seq_header.frame_width_bits_minus_1 + 1
            frame_width_minus_1 = read_f(reader, n)

            n = seq_header.frame_height_bits_minus_1 + 1
            frame_height_minus_1 = read_f(reader, n)

            # It is a requirement of bitstream conformance that frame_width_minus_1 is less than or equal to max_frame_width_minus_1.
            assert frame_width_minus_1 <= seq_header.max_frame_width_minus_1
            # It is a requirement of bitstream conformance that frame_height_minus_1 is less than or equal to max_frame_height_minus_1.
            assert frame_height_minus_1 <= seq_header.max_frame_height_minus_1

            frame_header.FrameWidth = frame_width_minus_1 + 1
            frame_header.FrameHeight = frame_height_minus_1 + 1
        else:
            frame_header.FrameWidth = seq_header.max_frame_width_minus_1 + 1
            frame_header.FrameHeight = seq_header.max_frame_height_minus_1 + 1

        self.__superres_params(av1)
        self.__compute_image_size(av1)

        # If FrameIsIntra is equal to 0 (indicating that this frame may use inter prediction), the requirements described in the frame size with refs semantics of section 6.8.6 must also be satisfied.
        if frame_header.FrameIsIntra == 0:
            for i in range(REFS_PER_FRAME):
                ref_frame_idx = frame_header.ref_frame_idx[i]
                assert (2 *
                        frame_header.FrameWidth >= ref_frame_store.RefUpscaledWidth[ref_frame_idx])
                assert (2 *
                        frame_header.FrameHeight >= ref_frame_store.RefFrameHeight[ref_frame_idx])
                assert (frame_header.FrameWidth <= 16 *
                        ref_frame_store.RefUpscaledWidth[ref_frame_idx])
                assert (frame_header.FrameHeight <= 16 *
                        ref_frame_store.RefFrameHeight[ref_frame_idx])

    def __render_size(self, av1: AV1Decoder):
        """
        解析渲染尺寸
        规范文档 5.9.6 Render size syntax
        """
        reader = av1.reader
        frame_header = self.frame_header
        render_and_frame_size_different = read_f(reader, 1)

        if render_and_frame_size_different == 1:
            render_width_minus_1 = read_f(reader, 16)
            frame_header.RenderWidth = render_width_minus_1 + 1

            render_height_minus_1 = read_f(reader, 16)
            frame_header.RenderHeight = render_height_minus_1 + 1
        else:
            frame_header.RenderWidth = frame_header.UpscaledWidth
            frame_header.RenderHeight = frame_header.FrameHeight

    def __frame_size_with_refs(self, av1: AV1Decoder):
        """
        解析带参考的帧尺寸
        规范文档 5.9.7 Frame size with refs syntax
        """
        reader = av1.reader
        frame_header = self.frame_header
        ref_frame_store = av1.ref_frame_store

        found_ref = 0
        for i in range(REFS_PER_FRAME):
            found_ref = read_f(reader, 1)
            if found_ref == 1:
                ref_frame_idx = frame_header.ref_frame_idx[i]
                frame_header.UpscaledWidth = ref_frame_store.RefUpscaledWidth[ref_frame_idx]
                frame_header.FrameWidth = frame_header.UpscaledWidth
                frame_header.FrameHeight = ref_frame_store.RefFrameHeight[ref_frame_idx]
                frame_header.RenderWidth = ref_frame_store.RefRenderWidth[ref_frame_idx]
                frame_header.RenderHeight = ref_frame_store.RefRenderHeight[ref_frame_idx]
                break

        if found_ref == 0:
            self.__frame_size(av1)
            self.__render_size(av1)
        else:
            self.__superres_params(av1)
            self.__compute_image_size(av1)

        # Once the FrameWidth and FrameHeight have been computed for an inter frame, it is a requirement of bitstream conformance that for all values of i in the range 0..(REFS_PER_FRAME - 1), all the following conditions are true:
        # - 2 * FrameWidth >= RefUpscaledWidth[ ref_frame_idx[ i ] ]
        # - 2 * FrameHeight >= RefFrameHeight[ ref_frame_idx[ i ] ]
        # - FrameWidth <= 16 * RefUpscaledWidth[ ref_frame_idx[ i ] ]
        # - FrameHeight <= 16 * RefFrameHeight[ ref_frame_idx[ i ] ]
        for i in range(REFS_PER_FRAME):
            ref_frame_idx = frame_header.ref_frame_idx[i]
            assert (2 *
                    frame_header.FrameWidth >= ref_frame_store.RefUpscaledWidth[ref_frame_idx])
            assert (2 *
                    frame_header.FrameHeight >= ref_frame_store.RefFrameHeight[ref_frame_idx])
            assert (frame_header.FrameWidth <= 16 *
                    ref_frame_store.RefUpscaledWidth[ref_frame_idx])
            assert (frame_header.FrameHeight <= 16 *
                    ref_frame_store.RefFrameHeight[ref_frame_idx])

    def __superres_params(self, av1: AV1Decoder):
        """
        解析超级分辨率参数
        规范文档 5.9.8 Superres params syntax
        """
        reader = av1.reader
        seq_header = av1.seq_header
        frame_header = self.frame_header

        if seq_header.enable_superres:
            frame_header.use_superres = read_f(reader, 1)
        else:
            frame_header.use_superres = 0

        if frame_header.use_superres:
            coded_denom = read_f(reader, SUPERRES_DENOM_BITS)
            frame_header.SuperresDenom = coded_denom + SUPERRES_DENOM_MIN
        else:
            frame_header.SuperresDenom = SUPERRES_NUM

        frame_header.UpscaledWidth = frame_header.FrameWidth
        frame_header.FrameWidth = (frame_header.UpscaledWidth * SUPERRES_NUM +
                                   (frame_header.SuperresDenom // 2)) // frame_header.SuperresDenom

    def __compute_image_size(self, av1: AV1Decoder):
        """
        计算图像尺寸
        规范文档 5.9.9 Compute image size function
        """
        frame_header = self.frame_header

        MiCols = 2 * ((frame_header.FrameWidth + 7) >> 3)
        MiRows = 2 * ((frame_header.FrameHeight + 7) >> 3)

        frame_header.MiCols = MiCols
        frame_header.MiRows = MiRows
        frame_header.PrevSegmentIds = Array(None, (MiRows, MiCols))

    def __read_interpolation_filter(self, av1: AV1Decoder):
        """
        读取插值滤波
        规范文档 5.9.10 Interpolation filter syntax
        """
        reader = av1.reader
        frame_header = self.frame_header
        is_filter_switchable = read_f(reader, 1)

        if is_filter_switchable == 1:
            frame_header.interpolation_filter = INTERPOLATION_FILTER.SWITCHABLE
        else:
            frame_header.interpolation_filter = INTERPOLATION_FILTER(
                read_f(reader, 2))

    def __loop_filter_params(self, av1: AV1Decoder):
        """
        解析环路滤波参数
        规范文档 5.9.11 Loop filter params syntax
        """
        reader = av1.reader
        seq_header = av1.seq_header
        frame_header = self.frame_header

        if frame_header.CodedLossless or frame_header.allow_intrabc:
            frame_header.loop_filter_level[0] = 0
            frame_header.loop_filter_level[1] = 0
            frame_header.loop_filter_ref_deltas[REF_FRAME.INTRA_FRAME] = 1
            frame_header.loop_filter_ref_deltas[REF_FRAME.LAST_FRAME] = 0
            frame_header.loop_filter_ref_deltas[REF_FRAME.LAST2_FRAME] = 0
            frame_header.loop_filter_ref_deltas[REF_FRAME.LAST3_FRAME] = 0
            frame_header.loop_filter_ref_deltas[REF_FRAME.BWDREF_FRAME] = 0
            frame_header.loop_filter_ref_deltas[REF_FRAME.GOLDEN_FRAME] = -1
            frame_header.loop_filter_ref_deltas[REF_FRAME.ALTREF_FRAME] = -1
            frame_header.loop_filter_ref_deltas[REF_FRAME.ALTREF2_FRAME] = -1
            frame_header.loop_filter_mode_deltas[0] = 0
            frame_header.loop_filter_mode_deltas[1] = 0
            return

        frame_header.loop_filter_level[0] = read_f(reader, 6)
        frame_header.loop_filter_level[1] = read_f(reader, 6)
        if seq_header.color_config.NumPlanes > 1:
            if frame_header.loop_filter_level[0] or frame_header.loop_filter_level[1]:
                frame_header.loop_filter_level[2] = read_f(reader, 6)
                frame_header.loop_filter_level[3] = read_f(reader, 6)
        frame_header.loop_filter_sharpness = read_f(reader, 3)
        frame_header.loop_filter_delta_enabled = read_f(reader, 1)
        if frame_header.loop_filter_delta_enabled == 1:
            loop_filter_delta_update = read_f(reader, 1)
            if loop_filter_delta_update == 1:
                for i in range(TOTAL_REFS_PER_FRAME):
                    update_ref_delta = read_f(reader, 1)
                    if update_ref_delta == 1:
                        frame_header.loop_filter_ref_deltas[i] = read_su(
                            reader, 1 + 6)

                for i in range(2):
                    update_mode_delta = read_f(reader, 1)
                    if update_mode_delta == 1:
                        frame_header.loop_filter_mode_deltas[i] = read_su(
                            reader, 1 + 6)

    def __quantization_params(self, av1: AV1Decoder):
        """
        解析量化参数
        规范文档 5.9.12 Quantization params syntax
        """
        reader = av1.reader
        seq_header = av1.seq_header
        frame_header = self.frame_header

        frame_header.base_q_idx = read_f(reader, 8)
        frame_header.DeltaQYDc = self.__read_delta_q(av1)
        if seq_header.color_config.NumPlanes > 1:
            if seq_header.color_config.separate_uv_delta_q:
                diff_uv_delta = read_f(reader, 1)
            else:
                diff_uv_delta = 0

            frame_header.DeltaQUDc = self.__read_delta_q(av1)
            frame_header.DeltaQUAc = self.__read_delta_q(av1)
            if diff_uv_delta:
                frame_header.DeltaQVDc = self.__read_delta_q(av1)
                frame_header.DeltaQVAc = self.__read_delta_q(av1)
            else:
                frame_header.DeltaQVDc = frame_header.DeltaQUDc
                frame_header.DeltaQVAc = frame_header.DeltaQUAc
        else:
            frame_header.DeltaQUDc = 0
            frame_header.DeltaQUAc = 0
            frame_header.DeltaQVDc = 0
            frame_header.DeltaQVAc = 0

        frame_header.using_qmatrix = read_f(reader, 1)
        if frame_header.using_qmatrix:
            frame_header.qm_y = read_f(reader, 4)
            frame_header.qm_u = read_f(reader, 4)
            if not seq_header.color_config.separate_uv_delta_q:
                frame_header.qm_v = frame_header.qm_u
            else:
                frame_header.qm_v = read_f(reader, 4)

    def __read_delta_q(self, av1: AV1Decoder) -> int:
        """
        读取Delta Q
        规范文档 5.9.13 Delta quantizer syntax

        Returns:
            delta_q值
        """
        reader = av1.reader
        delta_coded = read_f(reader, 1)

        if delta_coded:
            delta_q = read_su(reader, 1 + 6)
        else:
            delta_q = 0

        return delta_q

    def __segmentation_params(self, av1: AV1Decoder):
        """
        解析分段参数
        规范文档 5.9.14 Segmentation params syntax
        """
        reader = av1.reader
        frame_header = self.frame_header

        frame_header.segmentation_enabled = read_f(reader, 1)
        if frame_header.segmentation_enabled == 1:
            if frame_header.primary_ref_frame == PRIMARY_REF_NONE:
                frame_header.segmentation_update_map = 1
                frame_header.segmentation_temporal_update = 0
                segmentation_update_data = 1
            else:
                frame_header.segmentation_update_map = read_f(reader, 1)
                if frame_header.segmentation_update_map == 1:
                    frame_header.segmentation_temporal_update = read_f(
                        reader, 1)
                segmentation_update_data = read_f(reader, 1)

            if segmentation_update_data == 1:
                for i in range(MAX_SEGMENTS):
                    for j in range(SEG_LVL_MAX):
                        feature_enabled = read_f(reader, 1)
                        frame_header.FeatureEnabled[i][j] = feature_enabled

                        clippedValue = 0
                        if feature_enabled == 1:
                            bitsToRead = Segmentation_Feature_Bits[j]
                            limit = Segmentation_Feature_Max[j]

                            if Segmentation_Feature_Signed[j] == 1:
                                feature_value = read_su(reader, 1 + bitsToRead)
                                clippedValue = Clip3(-limit,
                                                     limit, feature_value)
                            else:
                                feature_value = read_f(reader, bitsToRead)
                                clippedValue = Clip3(0, limit, feature_value)

                        frame_header.FeatureData[i][j] = clippedValue
        else:
            for i in range(MAX_SEGMENTS):
                for j in range(SEG_LVL_MAX):
                    frame_header.FeatureEnabled[i][j] = 0
                    frame_header.FeatureData[i][j] = 0

        frame_header.SegIdPreSkip = 0
        frame_header.LastActiveSegId = 0
        for i in range(MAX_SEGMENTS):
            for j in range(SEG_LVL_MAX):
                if frame_header.FeatureEnabled[i][j]:
                    frame_header.LastActiveSegId = i
                    if j >= SEG_LVL_REF_FRAME:
                        frame_header.SegIdPreSkip = 1

    def __tile_info(self, av1: AV1Decoder):
        """
        解析Tile信息
        规范文档 5.9.15 Tile info syntax
        """
        reader = av1.reader
        seq_header = av1.seq_header
        frame_header = self.frame_header
        use_128x128_superblock = seq_header.use_128x128_superblock
        MiCols = frame_header.MiCols
        MiRows = frame_header.MiRows

        sbCols = ((MiCols + 31) >>
                  5) if use_128x128_superblock else ((MiCols + 15) >> 4)
        sbRows = ((MiRows + 31) >>
                  5) if use_128x128_superblock else ((MiRows + 15) >> 4)
        sbShift = 5 if use_128x128_superblock else 4
        sbSize = sbShift + 2

        maxTileWidthSb = MAX_TILE_WIDTH >> sbSize
        maxTileAreaSb = MAX_TILE_AREA >> (2 * sbSize)

        minLog2TileCols = self.__tile_log2(maxTileWidthSb, sbCols)
        maxLog2TileCols = self.__tile_log2(1, min(sbCols, MAX_TILE_COLS))
        maxLog2TileRows = self.__tile_log2(1, min(sbRows, MAX_TILE_ROWS))
        minLog2Tiles = max(minLog2TileCols, self.__tile_log2(
            maxTileAreaSb, sbRows * sbCols))

        uniform_tile_spacing_flag = read_f(reader, 1)
        if uniform_tile_spacing_flag:
            # 均匀tile间距

            frame_header.TileColsLog2 = minLog2TileCols
            while frame_header.TileColsLog2 < maxLog2TileCols:
                increment_tile_cols_log2 = read_f(reader, 1)
                if increment_tile_cols_log2 == 1:
                    frame_header.TileColsLog2 += 1
                else:
                    break

            tileWidthSb = (
                sbCols + (1 << frame_header.TileColsLog2) - 1) >> frame_header.TileColsLog2
            # It is a requirement of bitstream conformance that tileWidthSb is less than or equal to maxTileWidthSb.
            assert tileWidthSb <= maxTileWidthSb

            i = 0
            for startSb in range(0, sbCols, tileWidthSb):
                frame_header.MiColStarts[i] = startSb << sbShift
                i += 1
            frame_header.MiColStarts[i] = MiCols
            frame_header.TileCols = i

            minLog2TileRows = max(minLog2Tiles - frame_header.TileColsLog2, 0)
            frame_header.TileRowsLog2 = minLog2TileRows
            while frame_header.TileRowsLog2 < maxLog2TileRows:
                increment_tile_rows_log2 = read_f(reader, 1)
                if increment_tile_rows_log2 == 1:
                    frame_header.TileRowsLog2 += 1
                else:
                    break

            tileHeightSb = (
                sbRows + (1 << frame_header.TileRowsLog2) - 1) >> frame_header.TileRowsLog2
            # It is a requirement of bitstream conformance that tileWidthSb * tileHeightSb is less than or equal to maxTileAreaSb.
            assert tileHeightSb <= maxTileAreaSb

            i = 0
            startSb = 0
            for startSb in range(0, sbRows, tileHeightSb):
                frame_header.MiRowStarts[i] = startSb << sbShift
                i += 1
            frame_header.MiRowStarts[i] = MiRows
            frame_header.TileRows = i
        else:
            # 非均匀tile间距
            widestTileSb = 0
            startSb = 0
            i = 0
            while startSb < sbCols:
                frame_header.MiColStarts[i] = startSb << sbShift
                maxWidth = min(sbCols - startSb, maxTileWidthSb)
                width_in_sbs_minus_1 = read_ns(reader, maxWidth)
                sizeSb = width_in_sbs_minus_1 + 1
                widestTileSb = max(sizeSb, widestTileSb)
                startSb += sizeSb
                i += 1
            # If uniform_tile_spacing_flag is equal to 0, it is a requirement of bitstream conformance that startSb is equal to sbCols when the loop writing MiColStarts exits.
            assert startSb == sbCols

            frame_header.MiColStarts[i] = MiCols
            frame_header.TileCols = i
            frame_header.TileColsLog2 = self.__tile_log2(
                1, frame_header.TileCols)

            if minLog2Tiles > 0:
                maxTileAreaSb = (sbRows * sbCols) >> (minLog2Tiles + 1)
            else:
                maxTileAreaSb = sbRows * sbCols
            maxTileHeightSb = max(maxTileAreaSb // widestTileSb, 1)

            startSb = 0
            i = 0
            while startSb < sbRows:
                frame_header.MiRowStarts[i] = startSb << sbShift
                maxHeight = min(sbRows - startSb, maxTileHeightSb)
                height_in_sbs_minus_1 = read_ns(reader, maxHeight)
                sizeSb = height_in_sbs_minus_1 + 1
                startSb += sizeSb
                i += 1
            # If uniform_tile_spacing_flag is equal to 0, it is a requirement of bitstream conformance that startSb is equal to sbRows when the loop writing MiRowStarts exits.
            assert startSb == sbRows

            frame_header.MiRowStarts[i] = MiRows
            frame_header.TileRows = i
            frame_header.TileRowsLog2 = self.__tile_log2(
                1, frame_header.TileRows)

        # It is a requirement of bitstream conformance that TileCols is less than or equal to MAX_TILE_COLS.
        assert frame_header.TileCols <= MAX_TILE_COLS
        # It is a requirement of bitstream conformance that TileRows is less than or equal to MAX_TILE_ROWS.
        assert frame_header.TileRows <= MAX_TILE_ROWS

        if frame_header.TileColsLog2 > 0 or frame_header.TileRowsLog2 > 0:
            frame_header.context_update_tile_id = read_f(
                reader, frame_header.TileColsLog2 + frame_header.TileRowsLog2)
            # It is a requirement of bitstream conformance that context_update_tile_id is less than TileCols * TileRows.
            assert frame_header.context_update_tile_id < frame_header.TileCols * frame_header.TileRows

            tile_size_bytes_minus_1 = read_f(reader, 2)
            frame_header.TileSizeBytes = tile_size_bytes_minus_1 + 1
        else:
            frame_header.context_update_tile_id = 0

    def __tile_log2(self, blkSize: int, target: int) -> int:
        """
        tile_log2辅助函数
        规范文档 5.9.16 Tile size calculation function

        Args:
            blkSize: 块大小
            target: 目标值

        Returns:
            k值
        """
        k = 0
        while (blkSize << k) < target:
            k += 1
        return k

    def __delta_q_params(self, av1: AV1Decoder):
        """
        解析Delta Q参数
        规范文档 5.9.17 Quantizer index delta parameters syntax
        """
        reader = av1.reader
        frame_header = self.frame_header

        frame_header.delta_q_res = 0
        frame_header.delta_q_present = 0
        if frame_header.base_q_idx > 0:
            frame_header.delta_q_present = read_f(reader, 1)

        if frame_header.delta_q_present:
            frame_header.delta_q_res = read_f(reader, 2)

    def __delta_lf_params(self, av1: AV1Decoder):
        """
        解析Delta LF参数
        规范文档 5.9.18 Loop filter delta parameters syntax
        """
        reader = av1.reader
        frame_header = self.frame_header

        frame_header.delta_lf_present = 0
        frame_header.delta_lf_res = 0
        frame_header.delta_lf_multi = 0
        if frame_header.delta_q_present:
            if not frame_header.allow_intrabc:
                frame_header.delta_lf_present = read_f(reader, 1)
            if frame_header.delta_lf_present:
                frame_header.delta_lf_res = read_f(reader, 2)
                frame_header.delta_lf_multi = read_f(reader, 1)

    def __cdef_params(self, av1: AV1Decoder):
        """
        解析CDEF参数
        规范文档 5.9.19 CDEF parameters syntax
        """
        reader = av1.reader
        seq_header = av1.seq_header
        frame_header = self.frame_header
        NumPlanes = seq_header.color_config.NumPlanes

        if frame_header.CodedLossless or frame_header.allow_intrabc or not seq_header.enable_cdef:
            frame_header.cdef_bits = 0
            frame_header.cdef_y_pri_strength[0] = 0
            frame_header.cdef_y_sec_strength[0] = 0
            frame_header.cdef_uv_pri_strength[0] = 0
            frame_header.cdef_uv_sec_strength[0] = 0
            frame_header.CdefDamping = 3
            return

        cdef_damping_minus_3 = read_f(reader, 2)
        frame_header.CdefDamping = cdef_damping_minus_3 + 3
        frame_header.cdef_bits = read_f(reader, 2)
        for i in range(1 << frame_header.cdef_bits):
            frame_header.cdef_y_pri_strength[i] = read_f(reader, 4)
            frame_header.cdef_y_sec_strength[i] = read_f(reader, 2)
            if frame_header.cdef_y_sec_strength[i] == 3:
                frame_header.cdef_y_sec_strength[i] += 1
            if NumPlanes > 1:
                frame_header.cdef_uv_pri_strength[i] = read_f(reader, 4)
                frame_header.cdef_uv_sec_strength[i] = read_f(reader, 2)
                if frame_header.cdef_uv_sec_strength[i] == 3:
                    frame_header.cdef_uv_sec_strength[i] += 1

    def __lr_params(self, av1: AV1Decoder):
        """
        解析LR参数
        规范文档 5.9.20 Loop restoration parameters syntax
        """
        reader = av1.reader
        seq_header = av1.seq_header
        frame_header = self.frame_header
        use_128x128_superblock = seq_header.use_128x128_superblock
        subsampling_x = seq_header.color_config.subsampling_x
        subsampling_y = seq_header.color_config.subsampling_y
        NumPlanes = seq_header.color_config.NumPlanes

        if (frame_header.AllLossless or
            frame_header.allow_intrabc or
                not seq_header.enable_restoration):
            frame_header.FrameRestorationType[0] = FRAME_RESTORATION_TYPE.RESTORE_NONE
            frame_header.FrameRestorationType[1] = FRAME_RESTORATION_TYPE.RESTORE_NONE
            frame_header.FrameRestorationType[2] = FRAME_RESTORATION_TYPE.RESTORE_NONE
            frame_header.UsesLr = 0
            return

        frame_header.UsesLr = 0
        usesChromaLr = 0

        for i in range(NumPlanes):
            lr_type = read_f(reader, 2)
            frame_header.FrameRestorationType[i] = Remap_Lr_Type[lr_type]
            if frame_header.FrameRestorationType[i] != FRAME_RESTORATION_TYPE.RESTORE_NONE:
                frame_header.UsesLr = 1
                if i > 0:
                    usesChromaLr = 1

        if frame_header.UsesLr:
            if use_128x128_superblock:
                lr_unit_shift = read_f(reader, 1)
                lr_unit_shift += 1
            else:
                lr_unit_shift = read_f(reader, 1)
                if lr_unit_shift:
                    lr_unit_extra_shift = read_f(reader, 1)
                    lr_unit_shift += lr_unit_extra_shift

            frame_header.LoopRestorationSize[0] = RESTORATION_TILESIZE_MAX >> (
                2 - lr_unit_shift)

            if subsampling_x and subsampling_y and usesChromaLr:
                lr_uv_shift = read_f(reader, 1)
            else:
                lr_uv_shift = 0

            frame_header.LoopRestorationSize[1] = frame_header.LoopRestorationSize[0] >> lr_uv_shift
            frame_header.LoopRestorationSize[2] = frame_header.LoopRestorationSize[0] >> lr_uv_shift

    def __read_tx_mode(self, av1: AV1Decoder):
        """
        读取变换模式
        规范文档 5.9.21 TX mode syntax
        """
        reader = av1.reader
        frame_header = self.frame_header

        if frame_header.CodedLossless == 1:
            frame_header.TxMode = TX_MODE.ONLY_4X4
        else:
            tx_mode_select = read_f(reader, 1)
            if tx_mode_select:
                frame_header.TxMode = TX_MODE.TX_MODE_SELECT
            else:
                frame_header.TxMode = TX_MODE.TX_MODE_LARGEST

    def __skip_mode_params(self, av1: AV1Decoder):
        """
        解析skip模式参数
        规范文档 5.9.22 Skip mode parameters syntax
        """
        reader = av1.reader
        seq_header = av1.seq_header
        frame_header = self.frame_header
        ref_frame_store = av1.ref_frame_store

        if (frame_header.FrameIsIntra or
            not frame_header.reference_select or
                not seq_header.enable_order_hint):
            skipModeAllowed = 0
        else:
            forwardIdx = -1
            backwardIdx = -1
            forwardHint = 0
            backwardHint = 0

            from utils.frame_utils import get_relative_dist
            for i in range(REFS_PER_FRAME):
                ref_frame_idx_i = frame_header.ref_frame_idx[i]
                refHint = ref_frame_store.RefOrderHint[ref_frame_idx_i]

                if get_relative_dist(av1, refHint, frame_header.OrderHint) < 0:
                    if (forwardIdx < 0 or
                            get_relative_dist(av1, refHint, forwardHint) > 0):
                        forwardIdx = i
                        forwardHint = refHint
                elif get_relative_dist(av1, refHint, frame_header.OrderHint) > 0:
                    if (backwardIdx < 0 or
                            get_relative_dist(av1, refHint, backwardHint) < 0):
                        backwardIdx = i
                        backwardHint = refHint

            if forwardIdx < 0:
                skipModeAllowed = 0
            elif backwardIdx >= 0:
                skipModeAllowed = 1
                frame_header.SkipModeFrame[0] = REF_FRAME((REF_FRAME.LAST_FRAME +
                                                           min(forwardIdx, backwardIdx)))
                frame_header.SkipModeFrame[1] = REF_FRAME((REF_FRAME.LAST_FRAME +
                                                           max(forwardIdx, backwardIdx)))
            else:
                secondForwardIdx = -1
                secondForwardHint = 0
                for i in range(REFS_PER_FRAME):
                    ref_frame_idx_i = frame_header.ref_frame_idx[i]
                    refHint = ref_frame_store.RefOrderHint[ref_frame_idx_i]
                    if get_relative_dist(av1, refHint, forwardHint) < 0:
                        if secondForwardIdx < 0 or get_relative_dist(av1, refHint, secondForwardHint) > 0:
                            secondForwardIdx = i
                            secondForwardHint = refHint

                if secondForwardIdx < 0:
                    skipModeAllowed = 0
                else:
                    skipModeAllowed = 1
                    frame_header.SkipModeFrame[0] = REF_FRAME((REF_FRAME.LAST_FRAME +
                                                               min(forwardIdx, secondForwardIdx)))
                    frame_header.SkipModeFrame[1] = REF_FRAME((REF_FRAME.LAST_FRAME +
                                                               max(forwardIdx, secondForwardIdx)))

        if skipModeAllowed:
            frame_header.skip_mode_present = read_f(reader, 1)
        else:
            frame_header.skip_mode_present = 0

    def __frame_reference_mode(self, av1: AV1Decoder):
        """
        解析帧参考模式
        规范文档 5.9.23 Frame reference mode syntax
        """
        reader = av1.reader
        frame_header = self.frame_header
        if frame_header.FrameIsIntra:
            frame_header.reference_select = 0
        else:
            frame_header.reference_select = read_f(reader, 1)

    def __global_motion_params(self, av1: AV1Decoder):
        """
        解析全局运动参数
        规范文档 5.9.24 Global motion parameters syntax
        """
        reader = av1.reader
        frame_header = self.frame_header
        gm_params = frame_header.gm_params

        for ref in range(REF_FRAME.LAST_FRAME, REF_FRAME.ALTREF_FRAME + 1):
            frame_header.GmType[ref] = GM_TYPE.IDENTITY
            for i in range(6):
                gm_params[ref][i] = (1 << WARPEDMODEL_PREC_BITS) if (
                    i % 3 == 2) else 0

        if frame_header.FrameIsIntra:
            return

        for ref in range(REF_FRAME.LAST_FRAME, REF_FRAME.ALTREF_FRAME + 1):
            is_global = read_f(reader, 1)
            if is_global:
                is_rot_zoom = read_f(reader, 1)
                if is_rot_zoom:
                    type = GM_TYPE.ROTZOOM
                else:
                    is_translation = read_f(reader, 1)
                    type = GM_TYPE.TRANSLATION if is_translation else GM_TYPE.AFFINE
            else:
                type = GM_TYPE.IDENTITY

            frame_header.GmType[ref] = type

            if type >= GM_TYPE.ROTZOOM:
                self.__read_global_param(av1, type, ref, 2)
                self.__read_global_param(av1, type, ref, 3)
                if type == GM_TYPE.AFFINE:
                    self.__read_global_param(av1, type, ref, 4)
                    self.__read_global_param(av1, type, ref, 5)
                else:
                    gm_params[ref][4] = -gm_params[ref][3]
                    gm_params[ref][5] = gm_params[ref][2]

            if type >= GM_TYPE.TRANSLATION:
                self.__read_global_param(av1, type, ref, 0)
                self.__read_global_param(av1, type, ref, 1)

    def __read_global_param(self, av1: AV1Decoder, type: GM_TYPE, ref: int, idx: int):
        """
        读取全局参数
        规范文档 5.9.25 Global parameter syntax

        Args:
            type: 变换类型（IDENTITY, TRANSLATION, ROTZOOM, AFFINE）
            ref: 参考帧索引
            idx: 参数索引（0-5）
        """
        frame_header = self.frame_header
        gm_params = frame_header.gm_params

        absBits = GM_ABS_ALPHA_BITS
        precBits = GM_ALPHA_PREC_BITS
        if idx < 2:
            if type == GM_TYPE.TRANSLATION:
                absBits = (GM_ABS_TRANS_ONLY_BITS -
                           (1 - frame_header.allow_high_precision_mv))
                precBits = (GM_TRANS_ONLY_PREC_BITS -
                            (1 - frame_header.allow_high_precision_mv))
            else:
                absBits = GM_ABS_TRANS_BITS
                precBits = GM_TRANS_PREC_BITS

        precDiff = WARPEDMODEL_PREC_BITS - precBits
        round_val = (1 << WARPEDMODEL_PREC_BITS) if (idx % 3) == 2 else 0
        sub = (1 << precBits) if (idx % 3) == 2 else 0
        mx = 1 << absBits
        r = (frame_header.PrevGmParams[ref][idx] >> precDiff) - sub

        gm_params[ref][idx] = (self.__decode_signed_subexp_with_ref(
            av1, -mx, mx + 1, r) << precDiff) + round_val

    def __decode_signed_subexp_with_ref(self, av1: AV1Decoder, low: int, high: int, r: int) -> int:
        """
        解码有符号subexp with ref
        规范文档 5.9.26 Decode signed subexp with ref syntax

        Args:
            low: 最小值
            high: 最大值（不包含）
            r: 参考值
        """
        x = self.__decode_unsigned_subexp_with_ref(av1, high - low, r - low)
        return x + low

    def __decode_unsigned_subexp_with_ref(self, av1: AV1Decoder, mx: int, r: int) -> int:
        """
        解码无符号subexp with ref
        规范文档 5.9.27 Decode unsigned subexp with ref syntax

        Args:
            mx: 最大值
            r: 参考值
        """
        v = self.__decode_subexp(av1, mx)
        if (r << 1) <= mx:
            return inverse_recenter(r, v)
        else:
            return mx - 1 - inverse_recenter(mx - 1 - r, v)

    def __decode_subexp(self, av1: AV1Decoder, numSyms: int) -> int:
        """
        解码subexp
        规范文档 5.9.28 Decode subexp syntax

        Args:
            numSyms: 符号数量
        """
        reader = av1.reader

        i = 0
        mk = 0
        k = 3
        while True:
            b2 = (k + i - 1) if i else k
            a = 1 << b2
            if numSyms <= mk + 3 * a:
                subexp_final_bits = read_ns(reader, numSyms - mk)
                return subexp_final_bits + mk
            else:
                subexp_more_bits = read_f(reader, 1)
                if subexp_more_bits:
                    i += 1
                    mk += a
                else:
                    subexp_bits = read_f(reader, b2)
                    return subexp_bits + mk

    def __film_grain_params(self, av1: AV1Decoder):
        """
        解析Film grain参数
        规范文档 5.9.30 Film grain parameters syntax
        """
        reader = av1.reader
        seq_header = av1.seq_header
        frame_header = self.frame_header
        mono_chrome = seq_header.color_config.mono_chrome
        subsampling_x = seq_header.color_config.subsampling_x
        subsampling_y = seq_header.color_config.subsampling_y
        film_grain_params = frame_header.film_grain_params

        if (not seq_header.film_grain_params_present or
                (not frame_header.show_frame and not frame_header.showable_frame)):
            self._reset_grain_params(av1)
            return

        film_grain_params.apply_grain = read_f(reader, 1)

        if not film_grain_params.apply_grain:
            self._reset_grain_params(av1)
            return

        film_grain_params.grain_seed = read_f(reader, 16)

        if frame_header.frame_type == FRAME_TYPE.INTER_FRAME:
            update_grain = read_f(reader, 1)
        else:
            update_grain = 1

        if not update_grain:
            film_grain_params_ref_idx = read_f(reader, 3)
            # It is a requirement of bitstream conformance that film_grain_params_ref_idx is equal to ref_frame_idx[ j ] for some value of j in the range 0 to REFS_PER_FRAME - 1.
            assert film_grain_params_ref_idx in frame_header.ref_frame_idx

            tempGrainSeed = film_grain_params.grain_seed
            load_grain_params(av1, film_grain_params_ref_idx)
            film_grain_params.grain_seed = tempGrainSeed
            return

        film_grain_params.num_y_points = read_f(reader, 4)
        # It is a requirement of bitstream conformance that num_y_points is less than or equal to 14.
        assert film_grain_params.num_y_points <= 14

        for i in range(film_grain_params.num_y_points):
            point_y_value = read_f(reader, 8)
            # If i is greater than 0, it is a requirement of bitstream conformance that point_y_value[ i ] is greater than point_y_value[ i - 1 ] (this ensures the x coordinates are specified in increasing order).
            if i:
                assert point_y_value > film_grain_params.point_y_value[i-1]

            film_grain_params.point_y_value[i] = point_y_value
            film_grain_params.point_y_scaling[i] = read_f(reader, 8)

        if mono_chrome:
            film_grain_params.chroma_scaling_from_luma = 0
        else:
            film_grain_params.chroma_scaling_from_luma = read_f(reader, 1)
        if (mono_chrome or
            film_grain_params.chroma_scaling_from_luma or
                (subsampling_x == 1 and subsampling_y == 1 and film_grain_params.num_y_points == 0)):
            film_grain_params.num_cb_points = 0
            film_grain_params.num_cr_points = 0
        else:
            film_grain_params.num_cb_points = read_f(reader, 4)
            # It is a requirement of bitstream conformance that num_cb_points is less than or equal to 10.
            assert film_grain_params.num_cb_points <= 10

            for i in range(film_grain_params.num_cb_points):
                point_cb_value = read_f(reader, 8)
                # If i is greater than 0, it is a requirement of bitstream conformance that point_cb_value[ i ] is greater than point_cb_value[ i - 1 ].
                if i > 0:
                    assert point_cb_value > film_grain_params.point_cb_value[i-1]

                film_grain_params.point_cb_value[i] = point_cb_value
                film_grain_params.point_cb_scaling[i] = read_f(reader, 8)

            film_grain_params.num_cr_points = read_f(reader, 4)
            # It is a requirement of bitstream conformance that num_cr_points is less than or equal to 10.
            assert film_grain_params.num_cr_points <= 10

            # If subsampling_x is equal to 1 and subsampling_y is equal to 1 and num_cb_points is equal to 0, it is a requirement of bitstream conformance that num_cr_points is equal to 0.
            # If subsampling_x is equal to 1 and subsampling_y is equal to 1 and num_cb_points is not equal to 0, it is a requirement of bitstream conformance that num_cr_points is not equal to 0.
            if subsampling_x == 1 and subsampling_y == 1:
                if film_grain_params.num_cb_points == 0:
                    assert film_grain_params.num_cr_points == 0
                if film_grain_params.num_cb_points != 0:
                    assert film_grain_params.num_cr_points != 0

            for i in range(film_grain_params.num_cr_points):
                point_cr_value = read_f(reader, 8)
                # If i is greater than 0, it is a requirement of bitstream conformance that point_cr_value[ i ] is greater than point_cr_value[ i - 1 ].
                if i > 0:
                    assert point_cr_value > film_grain_params.point_cr_value[i-1]

                film_grain_params.point_cr_value[i] = point_cr_value
                film_grain_params.point_cr_scaling[i] = read_f(reader, 8)

        film_grain_params.grain_scaling_minus_8 = read_f(reader, 2)
        film_grain_params.ar_coeff_lag = read_f(reader, 2)
        numPosLuma = (2 * film_grain_params.ar_coeff_lag *
                      (film_grain_params.ar_coeff_lag + 1))

        if film_grain_params.num_y_points:
            numPosChroma = numPosLuma + 1
            for i in range(numPosLuma):
                film_grain_params.ar_coeffs_y_plus_128[i] = read_f(reader, 8)
        else:
            numPosChroma = numPosLuma

        if film_grain_params.chroma_scaling_from_luma or film_grain_params.num_cb_points:
            for i in range(numPosChroma):
                film_grain_params.ar_coeffs_cb_plus_128[i] = read_f(reader, 8)

        if film_grain_params.chroma_scaling_from_luma or film_grain_params.num_cr_points:
            for i in range(numPosChroma):
                film_grain_params.ar_coeffs_cr_plus_128[i] = read_f(reader, 8)

        film_grain_params.ar_coeff_shift_minus_6 = read_f(reader, 2)
        film_grain_params.grain_scale_shift = read_f(reader, 2)

        if film_grain_params.num_cb_points:
            film_grain_params.cb_mult = read_f(reader, 8)
            film_grain_params.cb_luma_mult = read_f(reader, 8)
            film_grain_params.cb_offset = read_f(reader, 9)

        if film_grain_params.num_cr_points:
            film_grain_params.cr_mult = read_f(reader, 8)
            film_grain_params.cr_luma_mult = read_f(reader, 8)
            film_grain_params.cr_offset = read_f(reader, 9)

        film_grain_params.overlap_flag = read_f(reader, 1)
        film_grain_params.clip_to_restricted_range = read_f(reader, 1)

    def __temporal_point_info(self, av1: AV1Decoder):
        """
        解析时间点信息
        规范文档 5.9.31 Temporal point info syntax
        """
        reader = av1.reader
        n = av1.seq_header.frame_presentation_time_length_minus_1 + 1
        frame_presentation_time = read_f(reader, n)

    def __frame_header_copy(self, av1: AV1Decoder):
        """
        规范文档 6.8.1
        """
        self.frame_header = deepcopy(av1.frame_header)

    def __setup_past_independence(self, av1: AV1Decoder):
        """
        设置过去独立性
        规范文档 6.8.2

        将所有保存的状态重置为默认值，使得当前帧不依赖于之前的帧。
        """
        frame_header = self.frame_header
        MiRows = frame_header.MiRows
        MiCols = frame_header.MiCols

        frame_header.FeatureData = Array(None, (MAX_SEGMENTS, SEG_LVL_MAX), 0)
        frame_header.FeatureEnabled = Array(
            None, (MAX_SEGMENTS, SEG_LVL_MAX), 0)
        frame_header.PrevSegmentIds = Array(None, (MiRows, MiCols), 0)

        for ref in range(REF_FRAME.LAST_FRAME, REF_FRAME.ALTREF_FRAME + 1):
            for i in range(6):
                frame_header.PrevGmParams[ref][i] = 1 << WARPEDMODEL_PREC_BITS if (
                    i % 3 == 2) else 0

        frame_header.loop_filter_delta_enabled = 1
        frame_header.loop_filter_ref_deltas[REF_FRAME.INTRA_FRAME] = 1
        frame_header.loop_filter_ref_deltas[REF_FRAME.LAST_FRAME] = 0
        frame_header.loop_filter_ref_deltas[REF_FRAME.LAST2_FRAME] = 0
        frame_header.loop_filter_ref_deltas[REF_FRAME.LAST3_FRAME] = 0
        frame_header.loop_filter_ref_deltas[REF_FRAME.BWDREF_FRAME] = 0
        frame_header.loop_filter_ref_deltas[REF_FRAME.GOLDEN_FRAME] = -1
        frame_header.loop_filter_ref_deltas[REF_FRAME.ALTREF_FRAME] = -1
        frame_header.loop_filter_ref_deltas[REF_FRAME.ALTREF2_FRAME] = -1
        frame_header.loop_filter_mode_deltas = [0] * 2

    def __init_non_coeff_cdfs(self, av1: AV1Decoder):
        """
        初始化非系数CDF数组
        规范文档 6.8.2 init_non_coeff_cdfs()

        初始化所有非系数相关的CDF数组，从Default_*复制到cdfs数组。
        """
        ref_frame_store = av1.ref_frame_store

        cdf_mappings = [
            ('Default_Y_Mode_Cdf', 'YModeCdf'),
            ('Default_Uv_Mode_Cfl_Not_Allowed_Cdf', 'UVModeCflNotAllowedCdf'),
            ('Default_Uv_Mode_Cfl_Allowed_Cdf', 'UVModeCflAllowedCdf'),
            ('Default_Angle_Delta_Cdf', 'AngleDeltaCdf'),
            ('Default_Intrabc_Cdf', 'IntrabcCdf'),
            ('Default_Partition_W8_Cdf', 'PartitionW8Cdf'),
            ('Default_Partition_W16_Cdf', 'PartitionW16Cdf'),
            ('Default_Partition_W32_Cdf', 'PartitionW32Cdf'),
            ('Default_Partition_W64_Cdf', 'PartitionW64Cdf'),
            ('Default_Partition_W128_Cdf', 'PartitionW128Cdf'),
            ('Default_Segment_Id_Cdf', 'SegmentIdCdf'),
            ('Default_Segment_Id_Predicted_Cdf', 'SegmentIdPredictedCdf'),
            ('Default_Tx_8x8_Cdf', 'Tx8x8Cdf'),
            ('Default_Tx_16x16_Cdf', 'Tx16x16Cdf'),
            ('Default_Tx_32x32_Cdf', 'Tx32x32Cdf'),
            ('Default_Tx_64x64_Cdf', 'Tx64x64Cdf'),
            ('Default_Txfm_Split_Cdf', 'TxfmSplitCdf'),
            ('Default_Filter_Intra_Mode_Cdf', 'FilterIntraModeCdf'),
            ('Default_Filter_Intra_Cdf', 'FilterIntraCdf'),
            ('Default_Interp_Filter_Cdf', 'InterpFilterCdf'),
            ('Default_Motion_Mode_Cdf', 'MotionModeCdf'),
            ('Default_New_Mv_Cdf', 'NewMvCdf'),
            ('Default_Zero_Mv_Cdf', 'ZeroMvCdf'),
            ('Default_Ref_Mv_Cdf', 'RefMvCdf'),
            ('Default_Compound_Mode_Cdf', 'CompoundModeCdf'),
            ('Default_Drl_Mode_Cdf', 'DrlModeCdf'),
            ('Default_Is_Inter_Cdf', 'IsInterCdf'),
            ('Default_Comp_Mode_Cdf', 'CompModeCdf'),
            ('Default_Skip_Mode_Cdf', 'SkipModeCdf'),
            ('Default_Skip_Cdf', 'SkipCdf'),
            ('Default_Comp_Ref_Cdf', 'CompRefCdf'),
            ('Default_Comp_Bwd_Ref_Cdf', 'CompBwdRefCdf'),
            ('Default_Single_Ref_Cdf', 'SingleRefCdf'),
            ('Default_Palette_Y_Mode_Cdf', 'PaletteYModeCdf'),
            ('Default_Palette_Uv_Mode_Cdf', 'PaletteUVModeCdf'),
            ('Default_Palette_Y_Size_Cdf', 'PaletteYSizeCdf'),
            ('Default_Palette_Uv_Size_Cdf', 'PaletteUVSizeCdf'),
            ('Default_Palette_Size_2_Y_Color_Cdf', 'PaletteSize2YColorCdf'),
            ('Default_Palette_Size_2_Uv_Color_Cdf', 'PaletteSize2UVColorCdf'),
            ('Default_Palette_Size_3_Y_Color_Cdf', 'PaletteSize3YColorCdf'),
            ('Default_Palette_Size_3_Uv_Color_Cdf', 'PaletteSize3UVColorCdf'),
            ('Default_Palette_Size_4_Y_Color_Cdf', 'PaletteSize4YColorCdf'),
            ('Default_Palette_Size_4_Uv_Color_Cdf', 'PaletteSize4UVColorCdf'),
            ('Default_Palette_Size_5_Y_Color_Cdf', 'PaletteSize5YColorCdf'),
            ('Default_Palette_Size_5_Uv_Color_Cdf', 'PaletteSize5UVColorCdf'),
            ('Default_Palette_Size_6_Y_Color_Cdf', 'PaletteSize6YColorCdf'),
            ('Default_Palette_Size_6_Uv_Color_Cdf', 'PaletteSize6UVColorCdf'),
            ('Default_Palette_Size_7_Y_Color_Cdf', 'PaletteSize7YColorCdf'),
            ('Default_Palette_Size_7_Uv_Color_Cdf', 'PaletteSize7UVColorCdf'),
            ('Default_Palette_Size_8_Y_Color_Cdf', 'PaletteSize8YColorCdf'),
            ('Default_Palette_Size_8_Uv_Color_Cdf', 'PaletteSize8UVColorCdf'),
            ('Default_Delta_Q_Cdf', 'DeltaQCdf'),
            ('Default_Delta_Lf_Cdf', 'DeltaLFCdf'),
            ('Default_Intra_Tx_Type_Set1_Cdf', 'IntraTxTypeSet1Cdf'),
            ('Default_Intra_Tx_Type_Set2_Cdf', 'IntraTxTypeSet2Cdf'),
            ('Default_Inter_Tx_Type_Set1_Cdf', 'InterTxTypeSet1Cdf'),
            ('Default_Inter_Tx_Type_Set2_Cdf', 'InterTxTypeSet2Cdf'),
            ('Default_Inter_Tx_Type_Set3_Cdf', 'InterTxTypeSet3Cdf'),
            ('Default_Use_Obmc_Cdf', 'UseObmcCdf'),
            ('Default_Inter_Intra_Cdf', 'InterIntraCdf'),
            ('Default_Comp_Ref_Type_Cdf', 'CompRefTypeCdf'),
            ('Default_Cfl_Sign_Cdf', 'CflSignCdf'),
            ('Default_Uni_Comp_Ref_Cdf', 'UniCompRefCdf'),
            ('Default_Wedge_Inter_Intra_Cdf', 'WedgeInterIntraCdf'),
            ('Default_Comp_Group_Idx_Cdf', 'CompGroupIdxCdf'),
            ('Default_Compound_Idx_Cdf', 'CompoundIdxCdf'),
            ('Default_Compound_Type_Cdf', 'CompoundTypeCdf'),
            ('Default_Inter_Intra_Mode_Cdf', 'InterIntraModeCdf'),
            ('Default_Wedge_Index_Cdf', 'WedgeIndexCdf'),
            ('Default_Cfl_Alpha_Cdf', 'CflAlphaCdf'),
            ('Default_Use_Wiener_Cdf', 'UseWienerCdf'),
            ('Default_Use_Sgrproj_Cdf', 'UseSgrprojCdf'),
            ('Default_Restoration_Type_Cdf', 'RestorationTypeCdf'),
        ]
        for default_name, name in cdf_mappings:
            default_cdf = getattr(default_cdfs, default_name)
            cdf = deepcopy(default_cdf)
            ref_frame_store.cdfs[name] = inverseCdf(cdf)

        cdf_mappings = [
            ('Default_Mv_Joint_Cdf', 'MvJointCdf'),
            ('Default_Mv_Class_Cdf', 'MvClassCdf'),
            ('Default_Mv_Fr_Cdf', 'MvFrCdf'),
            ('Default_Mv_Class0_Fr_Cdf', 'MvClass0FrCdf'),
        ]
        for default_name, name in cdf_mappings:
            default_cdf = getattr(default_cdfs, default_name)
            cdf = [deepcopy(default_cdf) for _ in range(MV_CONTEXTS)]
            ref_frame_store.cdfs[name] = inverseCdf(cdf)

        cdf_mappings = [
            ('Default_Mv_Class0_Bit_Cdf', 'MvClass0BitCdf'),
            ('Default_Mv_Class0_Hp_Cdf', 'MvClass0HpCdf'),
            ('Default_Mv_Sign_Cdf', 'MvSignCdf'),
            ('Default_Mv_Bit_Cdf', 'MvBitCdf'),
            ('Default_Mv_Hp_Cdf', 'MvHpCdf'),
        ]
        for default_name, name in cdf_mappings:
            default_cdf = getattr(default_cdfs, default_name)
            cdf = [[deepcopy(default_cdf) for _ in range(2)]
                   for _ in range(MV_CONTEXTS)]
            ref_frame_store.cdfs[name] = inverseCdf(cdf)

        cdf_mappings = [
            ('Default_Delta_Lf_Cdf', 'DeltaLFMultiCdf'),
        ]
        for default_name, name in cdf_mappings:
            default_cdf = getattr(default_cdfs, default_name)
            cdf = [deepcopy(default_cdf) for _ in range(FRAME_LF_COUNT)]
            ref_frame_store.cdfs[name] = inverseCdf(cdf)

    def __init_coeff_cdfs(self, av1: AV1Decoder):
        """
        初始化系数CDF数组
        规范文档 6.8.2 init_coeff_cdfs()
        """
        frame_header = self.frame_header
        ref_frame_store = av1.ref_frame_store
        base_q_idx = frame_header.base_q_idx

        if base_q_idx <= 20:
            idx = 0
        elif base_q_idx <= 60:
            idx = 1
        elif base_q_idx <= 120:
            idx = 2
        else:
            idx = 3

        coeff_cdf_mappings = [
            ('Default_Txb_Skip_Cdf', 'TxbSkipCdf'),
            ('Default_Eob_Pt_16_Cdf', 'EobPt16Cdf'),
            ('Default_Eob_Pt_32_Cdf', 'EobPt32Cdf'),
            ('Default_Eob_Pt_64_Cdf', 'EobPt64Cdf'),
            ('Default_Eob_Pt_128_Cdf', 'EobPt128Cdf'),
            ('Default_Eob_Pt_256_Cdf', 'EobPt256Cdf'),
            ('Default_Eob_Pt_512_Cdf', 'EobPt512Cdf'),
            ('Default_Eob_Pt_1024_Cdf', 'EobPt1024Cdf'),
            ('Default_Eob_Extra_Cdf', 'EobExtraCdf'),
            ('Default_Dc_Sign_Cdf', 'DcSignCdf'),
            ('Default_Coeff_Base_Eob_Cdf', 'CoeffBaseEobCdf'),
            ('Default_Coeff_Base_Cdf', 'CoeffBaseCdf'),
            ('Default_Coeff_Br_Cdf', 'CoeffBrCdf'),
        ]

        for default_name, name in coeff_cdf_mappings:
            default_cdf_array = getattr(default_cdfs, default_name)
            cdf = deepcopy(default_cdf_array[idx])
            ref_frame_store.cdfs[name] = inverseCdf(cdf)

    def __load_previous(self, av1: AV1Decoder):
        """
        加载之前的信息
        规范文档 6.8.2 load_previous()
        """
        frame_header = av1.frame_header
        ref_frame_store = av1.ref_frame_store

        prevFrame = frame_header.ref_frame_idx[frame_header.primary_ref_frame]
        frame_header.PrevGmParams = deepcopy(
            ref_frame_store.SavedGmParams[prevFrame])
        load_loop_filter_params(av1, prevFrame)
        load_segmentation_params(av1, prevFrame)

    def __load_previous_segment_ids(self, av1: AV1Decoder):
        """
        加载之前的segment ID
        规范文档 6.8.2 load_previous_segment_ids()
        """
        frame_header = av1.frame_header
        ref_frame_store = av1.ref_frame_store
        MiCols = frame_header.MiCols
        MiRows = frame_header.MiRows

        # 1.
        prevFrame = frame_header.ref_frame_idx[frame_header.primary_ref_frame]

        # 2.
        if frame_header.segmentation_enabled == 1:
            ref_frame_store.RefMiCols[prevFrame] = MiCols
            ref_frame_store.RefMiRows[prevFrame] = MiRows
            for row in range(MiRows):
                for col in range(MiCols):
                    frame_header.PrevSegmentIds[row][col] = ref_frame_store.SavedSegmentIds[prevFrame][row][col]
        else:
            frame_header.PrevSegmentIds = Array(None, (MiRows, MiCols), 0)

    def _reset_grain_params(self, av1: AV1Decoder):
        """
        重置Film grain参数
        规范文档 6.8.20 reset_grain_params()
        """
        av1.frame_header.film_grain_params = FilmGrainParams()


def load_cdfs(av1: AV1Decoder, ctx: int):
    """
    加载CDF数组
    规范文档 6.8.2 load_cdfs()

    从参考帧上下文ctx加载CDF数组到当前的Tile*数组。
    加载后，将每个CDF数组的最后一个条目（symbol count）设置为0。

    Args:
        ctx: 上下文索引
    """
    ref_frame_store = av1.ref_frame_store
    saved_cdfs = ref_frame_store.SavedCdfs[ctx]

    def zero_last_entry(cdfs: Any):
        if type(cdfs[-1]) == int:
            cdfs[-1] = 0
            return
        for cdf in cdfs:
            zero_last_entry(cdf)

    for cdf_name, saved_cdf in saved_cdfs.items():
        loaded_cdf = deepcopy(saved_cdf)
        zero_last_entry(loaded_cdf)
        ref_frame_store.cdfs[cdf_name] = loaded_cdf


def load_grain_params(av1: AV1Decoder, idx: int):
    """
    加载颗粒参数
    规范文档 6.8.20 load_grain_params()

    load_grain_params(idx) 是一个函数调用，表示 film_grain_params 中读取的所有语法元素
    应该被设置为等于索引 idx 指向的内存区域中存储的值。

    Args:
        idx: 颗粒参数索引（参考帧索引）
    """
    frame_header = av1.frame_header
    ref_frame_store = av1.ref_frame_store

    frame_header.film_grain_params = deepcopy(
        ref_frame_store.SavedFilmGrainParams[idx])


def save_grain_params(av1: AV1Decoder, i: int):
    """
    保存颗粒参数
    规范文档 7.20 save_grain_params()

    Args:
        i: 参考帧索引（保存位置）
    """
    frame_header = av1.frame_header
    ref_frame_store = av1.ref_frame_store
    ref_frame_store.SavedFilmGrainParams[i] = deepcopy(
        frame_header.film_grain_params)


def save_loop_filter_params(av1: AV1Decoder, i: int):
    """
    保存Loop Filter参数
    规范文档 7.20 save_loop_filter_params()

    Args:
        i: 参考帧索引（保存位置）
    """
    frame_header = av1.frame_header
    ref_frame_store = av1.ref_frame_store

    ref_frame_store.SavedLoopFilterRefDeltas[i] = deepcopy(
        frame_header.loop_filter_ref_deltas)
    ref_frame_store.SavedLoopFilterModeDeltas[i] = deepcopy(
        frame_header.loop_filter_mode_deltas)


def save_segmentation_params(av1: AV1Decoder, i: int):
    """
    保存Segmentation参数
    规范文档 7.20 save_segmentation_params()

    Args:
        i: 参考帧索引（保存位置）
    """
    frame_header = av1.frame_header
    ref_frame_store = av1.ref_frame_store

    ref_frame_store.SavedFeatureEnabled[i] = deepcopy(
        frame_header.FeatureEnabled)
    ref_frame_store.SavedFeatureData[i] = deepcopy(frame_header.FeatureData)


def load_loop_filter_params(av1: AV1Decoder, idx: int):
    """
    加载Loop Filter参数
    规范文档 7.21 Reference frame loading process
    load_loop_filter_params( i ) 函数调用表示应该从索引为i的内存区域加载loop filter参数

    应该加载的值：
    - loop_filter_ref_deltas[ j ] for j = 0 .. TOTAL_REFS_PER_FRAME-1
    - loop_filter_mode_deltas[ j ] for j = 0 .. 1

    Args:
        idx: 参考帧索引
    """
    frame_header = av1.frame_header
    ref_frame_store = av1.ref_frame_store

    frame_header.loop_filter_ref_deltas = deepcopy(
        ref_frame_store.SavedLoopFilterRefDeltas[idx])
    frame_header.loop_filter_mode_deltas = deepcopy(
        ref_frame_store.SavedLoopFilterModeDeltas[idx])


def load_segmentation_params(av1: AV1Decoder, idx: int):
    """
    加载Segmentation参数
    规范文档 7.21 Reference frame loading process
    load_segmentation_params( i ) 函数调用表示应该从索引为i的内存区域加载segmentation参数

    应该加载的值：
    - FeatureEnabled[ j ][ k ] for j = 0 .. MAX_SEGMENTS-1, for k = 0 .. SEG_LVL_MAX-1
    - FeatureData[ j ][ k ] for j = 0 .. MAX_SEGMENTS-1, for k = 0 .. SEG_LVL_MAX-1

    Args:
        idx: 参考帧索引
    """
    frame_header = av1.frame_header
    ref_frame_store = av1.ref_frame_store

    frame_header.FeatureEnabled = deepcopy(
        ref_frame_store.SavedFeatureEnabled[idx])
    frame_header.FeatureData = deepcopy(ref_frame_store.SavedFeatureData[idx])


def frame_header_obu(av1: AV1Decoder):
    """
    帧头OBU解析函数
    规范文档 5.9 Frame header OBU syntax
    """
    parser = FrameHeaderParser()
    av1.obu.frame_header = parser.frame_header
    av1.frame_header = parser.frame_header
    parser.frame_header_obu(av1)


def frame_obu(av1: AV1Decoder, sz: int):
    """
    规范文档 5.10 Frame OBU syntax
    """
    reader = av1.reader

    startBitPos = reader.get_position()
    frame_header_obu(av1)
    if av1.obu.frame_header is not None:
        av1.frame_header = av1.obu.frame_header
    reader.byte_alignment()
    endBitPos = reader.get_position()

    headerBytes = (endBitPos - startBitPos) // 8
    sz -= headerBytes
    from tile.tile_group import tile_group_obu
    tile_group_obu(av1, sz)


"""
规范文档 5.9.14 Segmentation params syntax
"""
Segmentation_Feature_Bits = [8, 6, 6, 6, 6, 3, 0, 0]
Segmentation_Feature_Signed = [1, 1, 1, 1, 1, 0, 0, 0]
Segmentation_Feature_Max = [
    255, MAX_LOOP_FILTER, MAX_LOOP_FILTER, MAX_LOOP_FILTER, MAX_LOOP_FILTER, 7, 0, 0]

"""
规范文档 5.9.20 Loop restoration params syntax
"""
Remap_Lr_Type = [
    FRAME_RESTORATION_TYPE.RESTORE_NONE,
    FRAME_RESTORATION_TYPE.RESTORE_SWITCHABLE,
    FRAME_RESTORATION_TYPE.RESTORE_WIENER,
    FRAME_RESTORATION_TYPE.RESTORE_SGRPROJ,
]
