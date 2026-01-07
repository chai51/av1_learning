"""
序列头OBU解析器
按照规范文档6.3节实现sequence_header_obu()
"""

from typing import Any, List, Optional
from bitstream.bit_reader import BitReader
from bitstream.descriptors import read_f, read_uvlc
from constants import BUFFER_POOL_MAX_SIZE, CHROMA_SAMPLE_POSITION, MATRIX_COEFFICIENTS, MAX_OPERATING_POINTS, NONE, SELECT_SCREEN_CONTENT_TOOLS, SELECT_INTEGER_MV, UINT32_MAX
from constants import COLOR_PRIMARIES, TRANSFER_CHARACTERISTICS
from obu.decoder import AV1Decoder


class ColorConfig:
    """
    颜色配置
    规范文档 6.3.2 color_config()
    """

    def __init__(self):
        self.mono_chrome: int = 0
        self.matrix_coefficients = MATRIX_COEFFICIENTS.MC_IDENTITY
        self.subsampling_x: int = 0
        self.subsampling_y: int = 0
        self.separate_uv_delta_q: int = 0

        self.BitDepth: int = 0
        self.NumPlanes: int = 0


class OperatingParametersInfo:
    """
    操作参数信息
    规范文档 6.3.5 operating_parameters_info()
    """

    def __init__(self):
        pass


class SequenceHeader:
    """
    序列头结构
    规范文档 6.3.1 sequence_header_obu()
    """

    def __init__(self):
        self.seq_profile = 0
        self.still_picture = 0
        self.reduced_still_picture_header = 0

        # Timing and decoder model
        self.timing_info_present_flag = 0
        self.decoder_model_info_present_flag = 0
        self.initial_display_delay_present_flag = 0

        # Operating points
        self.operating_points_cnt_minus_1 = 0
        self.operating_point_idc: List[int] = [NONE] * MAX_OPERATING_POINTS
        self.decoder_model_present_for_this_op: List[int] = [
            NONE] * MAX_OPERATING_POINTS

        self.OperatingPointIdc = 0
        # Frame size
        self.frame_width_bits_minus_1 = 0
        self.frame_height_bits_minus_1 = 0
        self.max_frame_width_minus_1 = 0
        self.max_frame_height_minus_1 = 0

        # Frame ID
        self.frame_id_numbers_present_flag = 0
        self.additional_frame_id_length_minus_1: int = NONE
        self.delta_frame_id_length_minus_2: int = NONE

        # Feature flags
        self.use_128x128_superblock = 0
        self.enable_filter_intra: Optional[int] = None
        self.enable_intra_edge_filter: Optional[int] = None
        self.enable_interintra_compound: Optional[int] = None
        self.enable_masked_compound: Optional[int] = None
        self.enable_warped_motion: Optional[int] = None
        self.enable_order_hint: Optional[int] = None
        self.enable_dual_filter: Optional[int] = None
        self.enable_jnt_comp: Optional[int] = None
        self.enable_ref_frame_mvs: Optional[int] = None
        self.seq_force_screen_content_tools: Optional[int] = None
        self.seq_force_integer_mv: Optional[int] = None
        self.OrderHintBits = 0
        self.enable_superres: Optional[int] = None
        self.enable_cdef: Optional[int] = None
        self.enable_restoration: Optional[int] = None

        # Film grain
        self.film_grain_params_present: Optional[int] = None

        # Color config
        self.color_config: ColorConfig = ColorConfig()

        # Timing info
        self.equal_picture_interval: Optional[int] = None

        # Decoder model info
        self.buffer_delay_length_minus_1: int = NONE
        self.buffer_removal_time_length_minus_1: int = NONE
        self.frame_presentation_time_length_minus_1: int = NONE


class SequenceHeaderParser:
    """
    序列头解析器
    实现规范文档中描述的sequence_header_obu()函数
    """

    def __init__(self):
        self.seq_header: SequenceHeader = SequenceHeader()

    def sequence_header_obu(self, av1: AV1Decoder) -> SequenceHeader:
        """
        规范文档 5.5.1 General sequence header OBU syntax
        """
        reader = av1.reader
        seq_header = self.seq_header

        # It is a requirement of bitstream conformance that seq_profile is not greater than 2 (values 3 to 7 are reserved).
        seq_header.seq_profile = read_f(reader, 3)
        assert seq_header.seq_profile <= 2

        seq_header.still_picture = read_f(reader, 1)

        # If reduced_still_picture_header is equal to 1, it is a requirement of bitstream conformance that still_picture is equal to 1.
        seq_header.reduced_still_picture_header = read_f(reader, 1)
        if seq_header.reduced_still_picture_header == 1:
            assert seq_header.still_picture == 1

        if seq_header.reduced_still_picture_header:
            seq_header.timing_info_present_flag = 0
            seq_header.decoder_model_info_present_flag = 0
            seq_header.initial_display_delay_present_flag = 0
            seq_header.operating_points_cnt_minus_1 = 0
            seq_header.operating_point_idc[0] = 0
            seq_level_idx = read_f(reader, 5)
            seq_tier = 0
            seq_header.decoder_model_present_for_this_op[0] = 0
            initial_display_delay_present_for_this_op = 0
        else:
            seq_header.timing_info_present_flag = read_f(reader, 1)
            if seq_header.timing_info_present_flag:
                self.__timing_info(reader)
                seq_header.decoder_model_info_present_flag = read_f(reader, 1)
                if seq_header.decoder_model_info_present_flag:
                    self.__decoder_model_info(reader)
            else:
                seq_header.decoder_model_info_present_flag = 0

            seq_header.initial_display_delay_present_flag = read_f(reader, 1)
            seq_header.operating_points_cnt_minus_1 = read_f(reader, 5)
            for i in range(seq_header.operating_points_cnt_minus_1 + 1):
                seq_header.operating_point_idc[i] = read_f(reader, 12)
                seq_level_idx = read_f(reader, 5)
                if seq_level_idx > 7:
                    seq_tier = read_f(reader, 1)
                else:
                    seq_tier = 0
                if seq_header.decoder_model_info_present_flag:
                    seq_header.decoder_model_present_for_this_op[i] = read_f(
                        reader, 1)
                    if seq_header.decoder_model_present_for_this_op[i]:
                        self.__operating_parameters_info(reader)
                else:
                    seq_header.decoder_model_present_for_this_op[i] = 0

                initial_display_delay_minus_1: Optional[int] = None
                if seq_header.initial_display_delay_present_flag:
                    initial_display_delay_present_for_this_op = read_f(
                        reader, 1)
                    if initial_display_delay_present_for_this_op:
                        initial_display_delay_minus_1 = read_f(reader, 4)

                # If not signaled then initial_display_delay_minus_1[ i ] = BUFFER_POOL_MAX_SIZE - 1.
                if initial_display_delay_minus_1 is None:
                    initial_display_delay_minus_1 = BUFFER_POOL_MAX_SIZE - 1

                # If operating_point_idc[ op ] is not equal to 0 for any value of op from 0 to operating_points_cnt_minus_1, it is a requirement of bitstream conformance that obu_extension_flag is equal to 1 for all layer-specific OBUs in the coded video sequence.
                if seq_header.operating_point_idc[i] != 0:
                    temporal_id = av1.obu.header.temporal_id
                    spatial_id = av1.obu.header.spatial_id
                    s = 8 + spatial_id if spatial_id else 0
                    # if temporal_id + s == i:
                    #     assert av1.obu.header.obu_extension_flag == 1

            # It is a requirement of bitstream conformance that operating_point_idc[ i ] is not equal to operating_point_idc[ j ] for j = 0.. (i - 1).
            operating_points_cnt = seq_header.operating_points_cnt_minus_1 + 1
            assert len(set[int](
                seq_header.operating_point_idc[:operating_points_cnt])) == operating_points_cnt

        operatingPoint = self.__choose_operating_point()
        seq_header.OperatingPointIdc = seq_header.operating_point_idc[operatingPoint]
        seq_header.frame_width_bits_minus_1 = read_f(reader, 4)
        seq_header.frame_height_bits_minus_1 = read_f(reader, 4)
        n = seq_header.frame_width_bits_minus_1 + 1
        seq_header.max_frame_width_minus_1 = read_f(reader, n)
        n = seq_header.frame_height_bits_minus_1 + 1
        seq_header.max_frame_height_minus_1 = read_f(reader, n)
        if seq_header.reduced_still_picture_header:
            seq_header.frame_id_numbers_present_flag = 0
        else:
            seq_header.frame_id_numbers_present_flag = read_f(reader, 1)

        if seq_header.frame_id_numbers_present_flag:
            seq_header.delta_frame_id_length_minus_2 = read_f(reader, 4)
            seq_header.additional_frame_id_length_minus_1 = read_f(reader, 3)

        seq_header.use_128x128_superblock = read_f(reader, 1)
        seq_header.enable_filter_intra = read_f(reader, 1)
        seq_header.enable_intra_edge_filter = read_f(reader, 1)

        if seq_header.reduced_still_picture_header:
            seq_header.enable_interintra_compound = 0
            seq_header.enable_masked_compound = 0
            seq_header.enable_warped_motion = 0
            seq_header.enable_dual_filter = 0
            seq_header.enable_order_hint = 0
            seq_header.enable_jnt_comp = 0
            seq_header.enable_ref_frame_mvs = 0
            seq_header.seq_force_screen_content_tools = SELECT_SCREEN_CONTENT_TOOLS
            seq_header.seq_force_integer_mv = SELECT_INTEGER_MV
            seq_header.OrderHintBits = 0
        else:
            seq_header.enable_interintra_compound = read_f(reader, 1)
            seq_header.enable_masked_compound = read_f(reader, 1)
            seq_header.enable_warped_motion = read_f(reader, 1)
            seq_header.enable_dual_filter = read_f(reader, 1)
            seq_header.enable_order_hint = read_f(reader, 1)
            if seq_header.enable_order_hint:
                seq_header.enable_jnt_comp = read_f(reader, 1)
                seq_header.enable_ref_frame_mvs = read_f(reader, 1)
            else:
                seq_header.enable_jnt_comp = 0
                seq_header.enable_ref_frame_mvs = 0

            seq_choose_screen_content_tools = read_f(reader, 1)
            if seq_choose_screen_content_tools:
                seq_header.seq_force_screen_content_tools = SELECT_SCREEN_CONTENT_TOOLS
            else:
                seq_header.seq_force_screen_content_tools = read_f(reader, 1)

            if seq_header.seq_force_screen_content_tools > 0:
                seq_choose_integer_mv = read_f(reader, 1)
                if seq_choose_integer_mv:
                    seq_header.seq_force_integer_mv = SELECT_INTEGER_MV
                else:
                    seq_header.seq_force_integer_mv = read_f(reader, 1)
            else:
                seq_header.seq_force_integer_mv = SELECT_INTEGER_MV

            if seq_header.enable_order_hint:
                order_hint_bits_minus_1 = read_f(reader, 3)
                seq_header.OrderHintBits = order_hint_bits_minus_1 + 1
            else:
                seq_header.OrderHintBits = 0

        seq_header.enable_superres = read_f(reader, 1)
        seq_header.enable_cdef = read_f(reader, 1)
        seq_header.enable_restoration = read_f(reader, 1)
        self.__color_config(reader)
        seq_header.film_grain_params_present = read_f(reader, 1)
        return seq_header

    def __color_config(self, reader: BitReader):
        """
        解析颜色配置
        规范文档 5.5.2 Color config syntax
        """
        seq_header = self.seq_header
        color_config = self.seq_header.color_config

        high_bitdepth = read_f(reader, 1)
        if seq_header.seq_profile == 2 and high_bitdepth:
            twelve_bit = read_f(reader, 1)
            color_config.BitDepth = 12 if twelve_bit else 10
        elif seq_header.seq_profile <= 2:
            color_config.BitDepth = 10 if high_bitdepth else 8

        if seq_header.seq_profile == 1:
            color_config.mono_chrome = 0
        else:
            color_config.mono_chrome = read_f(reader, 1)

        color_config.NumPlanes = 1 if color_config.mono_chrome else 3
        color_description_present_flag = read_f(reader, 1)
        if color_description_present_flag:
            color_primaries = COLOR_PRIMARIES(read_f(reader, 8))
            transfer_characteristics = TRANSFER_CHARACTERISTICS(
                read_f(reader, 8))
            color_config.matrix_coefficients = MATRIX_COEFFICIENTS(
                read_f(reader, 8))
        else:
            color_primaries = COLOR_PRIMARIES.CP_UNSPECIFIED
            transfer_characteristics = TRANSFER_CHARACTERISTICS.TC_UNSPECIFIED
            color_config.matrix_coefficients = MATRIX_COEFFICIENTS.MC_UNSPECIFIED

        if color_config.mono_chrome:
            color_range = read_f(reader, 1)
            color_config.subsampling_x = 1
            color_config.subsampling_y = 1
            chroma_sample_position = CHROMA_SAMPLE_POSITION.CSP_UNKNOWN
            color_config.separate_uv_delta_q = 0
            return

        # 特殊颜色空间处理
        elif (color_primaries == COLOR_PRIMARIES.CP_BT_709 and
              transfer_characteristics == TRANSFER_CHARACTERISTICS.TC_SRGB and
              color_config.matrix_coefficients == MATRIX_COEFFICIENTS.MC_IDENTITY):
            color_range = 1
            color_config.subsampling_x = 0
            color_config.subsampling_y = 0
        else:
            color_range = read_f(reader, 1)
            if seq_header.seq_profile == 0:
                color_config.subsampling_x = 1
                color_config.subsampling_y = 1
            elif seq_header.seq_profile == 1:
                color_config.subsampling_x = 0
                color_config.subsampling_y = 0
            else:
                if color_config.BitDepth == 12:
                    color_config.subsampling_x = read_f(reader, 1)
                    if color_config.subsampling_x:
                        color_config.subsampling_y = read_f(reader, 1)
                    else:
                        color_config.subsampling_y = 0
                else:
                    color_config.subsampling_x = 1
                    color_config.subsampling_y = 0

            if color_config.subsampling_x and color_config.subsampling_y:
                chroma_sample_position = CHROMA_SAMPLE_POSITION(
                    read_f(reader, 2))

        # If matrix_coefficients is equal to MC_IDENTITY, it is a requirement of bitstream conformance that subsampling_x is equal to 0 and subsampling_y is equal to 0.
        if color_config.matrix_coefficients == MATRIX_COEFFICIENTS.MC_IDENTITY:
            assert color_config.subsampling_x == 0
            assert color_config.subsampling_y == 0

        color_config.separate_uv_delta_q = read_f(reader, 1)

    def __timing_info(self, reader: BitReader):
        """
        解析时序信息
        规范文档 5.5.3 Timing info syntax
        """
        seq_header = self.seq_header

        # It is a requirement of bitstream conformance that num_units_in_display_tick is greater than 0.
        num_units_in_display_tick = read_f(reader, 32)
        assert num_units_in_display_tick > 0

        # It is a requirement of bitstream conformance that time_scale is greater than 0.
        time_scale = read_f(reader, 32)
        assert time_scale > 0

        seq_header.equal_picture_interval = read_f(reader, 1)
        if seq_header.equal_picture_interval:
            # It is a requirement of bitstream conformance that the value of num_ticks_per_picture_minus_1 shall be in the range of 0 to (1 << 32) − 2, inclusive.
            num_ticks_per_picture_minus_1 = read_uvlc(reader)
            assert num_ticks_per_picture_minus_1 >= 0
            assert num_ticks_per_picture_minus_1 < UINT32_MAX

    def __decoder_model_info(self, reader: BitReader):
        """
        解析解码器模型信息
        规范文档 5.5.4 Decoder model info syntax
        """
        seq_header = self.seq_header

        seq_header.buffer_delay_length_minus_1 = read_f(reader, 5)

        num_units_in_decoding_tick = read_f(reader, 32)
        assert num_units_in_decoding_tick > 0

        seq_header.buffer_removal_time_length_minus_1 = read_f(reader, 5)
        seq_header.frame_presentation_time_length_minus_1 = read_f(reader, 5)

    def __operating_parameters_info(self, reader: BitReader) -> OperatingParametersInfo:
        """
        解析操作参数信息
        规范文档 5.5.5 Operating parameters info syntax
        """
        seq_header = self.seq_header

        n = seq_header.buffer_delay_length_minus_1 + 1
        decoder_buffer_delay = read_f(reader, n)
        encoder_buffer_delay = read_f(reader, n)
        low_delay_mode_flag = read_f(reader, 1)

        return OperatingParametersInfo()

    def __choose_operating_point(self) -> int:
        """
        选择操作点
        规范文档 6.4.1 choose_operating_point()
        """
        return 0


def sequence_header_obu(av1: AV1Decoder):
    """
    序列头OBU解析函数
    规范文档 5.5 sequence_header_obu()
    这是规范文档中定义的主函数
    """
    parser = SequenceHeaderParser()
    seq_header = parser.sequence_header_obu(av1)
    av1.obu.seq_header = seq_header
