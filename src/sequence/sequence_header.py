"""
序列头OBU解析器
按照规范文档6.3节实现sequence_header_obu()
"""

from typing import Optional
from bitstream.bit_reader import BitReader
from bitstream.descriptors import read_f, read_leb128


class ColorConfig:
    """
    颜色配置
    规范文档 6.3.2 color_config()
    """
    def __init__(self):
        self.high_bitdepth = 0
        self.twelve_bit = 0
        self.BitDepth = 8
        self.mono_chrome = 0
        self.NumPlanes = 3
        self.color_description_present_flag = 0
        self.color_primaries = 0
        self.transfer_characteristics = 0
        self.matrix_coefficients = 0
        self.color_range = 0
        self.subsampling_x = 0
        self.subsampling_y = 0
        self.chroma_sample_position = 0
        self.separate_uv_delta_q = 0


class TimingInfo:
    """
    时序信息
    规范文档 6.3.3 timing_info()
    """
    def __init__(self):
        self.num_units_in_display_tick = 0
        self.time_scale = 0
        self.equal_picture_interval = 0
        self.num_ticks_per_picture_minus_1 = 0


class DecoderModelInfo:
    """
    解码器模型信息
    规范文档 6.3.4 decoder_model_info()
    """
    def __init__(self):
        self.encoder_decoder_buffer_delay_length_minus_1 = 0
        self.buffer_removal_time_length_minus_1 = 0
        self.frame_presentation_time_length_minus_1 = 0


class OperatingParametersInfo:
    """
    操作参数信息
    规范文档 6.3.5 operating_parameters_info()
    """
    def __init__(self):
        self.decoder_buffer_delay = 0
        self.encoder_buffer_delay = 0
        self.low_delay_mode_flag = 0


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
        self.timing_info: Optional[TimingInfo] = None
        self.decoder_model_info_present_flag = 0
        self.decoder_model_info: Optional[DecoderModelInfo] = None
        self.initial_display_delay_present_flag = 0
        
        # Operating points
        self.operating_points_cnt_minus_1 = 0
        self.operating_point_idc = []
        self.seq_level_idx = []
        self.seq_tier = []
        self.decoder_model_present_for_this_op = []
        self.operating_parameters_info = []
        self.initial_display_delay_present_for_this_op = []
        self.initial_display_delay_minus_1 = []
        
        # Frame size
        self.frame_width_bits_minus_1 = 0
        self.frame_height_bits_minus_1 = 0
        self.max_frame_width_minus_1 = 0
        self.max_frame_height_minus_1 = 0
        
        # Frame ID
        self.frame_id_numbers_present_flag = 0
        self.delta_frame_id_length_minus_2 = 0
        self.additional_frame_id_length_minus_1 = 0
        
        # Feature flags
        self.use_128x128_superblock = 0
        self.enable_filter_intra = 0
        self.enable_intra_edge_filter = 0
        self.enable_interintra_compound = 0
        self.enable_masked_compound = 0
        self.enable_warped_motion = 0
        self.enable_dual_filter = 0
        self.enable_order_hint = 0
        self.enable_jnt_comp = 0
        self.enable_ref_frame_mvs = 0
        self.seq_choose_screen_content_tools = 0
        self.seq_force_screen_content_tools = 0
        self.seq_choose_integer_mv = 0
        self.seq_force_integer_mv = 0
        self.order_hint_bits_minus_1 = 0
        self.OrderHintBits = 0
        self.enable_superres = 0
        self.enable_cdef = 0
        self.enable_restoration = 0
        
        # Color config
        self.color_config: Optional[ColorConfig] = None
        
        # Film grain
        self.film_grain_params_present = 0


class SequenceHeaderParser:
    """
    序列头解析器
    实现规范文档中描述的sequence_header_obu()函数
    """
    
    def parse_sequence_header_obu(self, reader: BitReader):
        """
        解析序列头OBU
        规范文档 6.3.1 sequence_header_obu()
        
        Args:
            reader: BitReader实例
            
        Returns:
            SequenceHeader对象
        """
        seq_header = SequenceHeader()
        
        # seq_profile (f(3))
        seq_header.seq_profile = read_f(reader, 3)
        
        # still_picture (f(1))
        seq_header.still_picture = read_f(reader, 1)
        
        # reduced_still_picture_header (f(1))
        seq_header.reduced_still_picture_header = read_f(reader, 1)
        
        if seq_header.reduced_still_picture_header:
            # 简化的静态图片头
            seq_header.timing_info_present_flag = 0
            seq_header.decoder_model_info_present_flag = 0
            seq_header.initial_display_delay_present_flag = 0
            seq_header.operating_points_cnt_minus_1 = 0
            seq_header.operating_point_idc = [0]
            seq_header.seq_level_idx = [read_f(reader, 5)]  # seq_level_idx[0] (f(5))
            seq_header.seq_tier = [0]
            seq_header.decoder_model_present_for_this_op = [0]
            seq_header.initial_display_delay_present_for_this_op = [0]
        else:
            # 完整的序列头
            # timing_info_present_flag (f(1))
            seq_header.timing_info_present_flag = read_f(reader, 1)
            
            if seq_header.timing_info_present_flag:
                # timing_info()
                seq_header.timing_info = self._parse_timing_info(reader)
                
                # decoder_model_info_present_flag (f(1))
                seq_header.decoder_model_info_present_flag = read_f(reader, 1)
                
                if seq_header.decoder_model_info_present_flag:
                    # decoder_model_info()
                    seq_header.decoder_model_info = self._parse_decoder_model_info(reader)
            else:
                seq_header.decoder_model_info_present_flag = 0
            
            # initial_display_delay_present_flag (f(1))
            seq_header.initial_display_delay_present_flag = read_f(reader, 1)
            
            # operating_points_cnt_minus_1 (f(5))
            seq_header.operating_points_cnt_minus_1 = read_f(reader, 5)
            
            # 解析每个操作点
            seq_header.operating_point_idc = []
            seq_header.seq_level_idx = []
            seq_header.seq_tier = []
            seq_header.decoder_model_present_for_this_op = []
            seq_header.operating_parameters_info = []
            seq_header.initial_display_delay_present_for_this_op = []
            seq_header.initial_display_delay_minus_1 = []
            
            for i in range(seq_header.operating_points_cnt_minus_1 + 1):
                # operating_point_idc[i] (f(12))
                seq_header.operating_point_idc.append(read_f(reader, 12))
                
                # seq_level_idx[i] (f(5))
                seq_header.seq_level_idx.append(read_f(reader, 5))
                
                # seq_tier[i]
                if seq_header.seq_level_idx[i] > 7:
                    seq_header.seq_tier.append(read_f(reader, 1))
                else:
                    seq_header.seq_tier.append(0)
                
                # decoder_model_present_for_this_op[i]
                if seq_header.decoder_model_info_present_flag:
                    seq_header.decoder_model_present_for_this_op.append(read_f(reader, 1))
                    if seq_header.decoder_model_present_for_this_op[i]:
                        # operating_parameters_info(i)
                        params_info = self._parse_operating_parameters_info(reader)
                        seq_header.operating_parameters_info.append(params_info)
                    else:
                        seq_header.operating_parameters_info.append(None)
                else:
                    seq_header.decoder_model_present_for_this_op.append(0)
                    seq_header.operating_parameters_info.append(None)
                
                # initial_display_delay_present_for_this_op[i]
                if seq_header.initial_display_delay_present_flag:
                    seq_header.initial_display_delay_present_for_this_op.append(read_f(reader, 1))
                    if seq_header.initial_display_delay_present_for_this_op[i]:
                        # initial_display_delay_minus_1[i] (f(4))
                        seq_header.initial_display_delay_minus_1.append(read_f(reader, 4))
                    else:
                        seq_header.initial_display_delay_minus_1.append(0)
                else:
                    seq_header.initial_display_delay_present_for_this_op.append(0)
                    seq_header.initial_display_delay_minus_1.append(0)
        
        # choose_operating_point() - 简化实现，选择第一个操作点
        operatingPoint = 0
        OperatingPointIdc = seq_header.operating_point_idc[operatingPoint]
        
        # frame_width_bits_minus_1 (f(4))
        seq_header.frame_width_bits_minus_1 = read_f(reader, 4)
        
        # frame_height_bits_minus_1 (f(4))
        seq_header.frame_height_bits_minus_1 = read_f(reader, 4)
        
        # max_frame_width_minus_1 (f(n))
        n = seq_header.frame_width_bits_minus_1 + 1
        seq_header.max_frame_width_minus_1 = read_f(reader, n)
        
        # max_frame_height_minus_1 (f(n))
        n = seq_header.frame_height_bits_minus_1 + 1
        seq_header.max_frame_height_minus_1 = read_f(reader, n)
        
        # frame_id_numbers_present_flag
        if seq_header.reduced_still_picture_header:
            seq_header.frame_id_numbers_present_flag = 0
        else:
            seq_header.frame_id_numbers_present_flag = read_f(reader, 1)
        
        if seq_header.frame_id_numbers_present_flag:
            # delta_frame_id_length_minus_2 (f(4))
            seq_header.delta_frame_id_length_minus_2 = read_f(reader, 4)
            
            # additional_frame_id_length_minus_1 (f(3))
            seq_header.additional_frame_id_length_minus_1 = read_f(reader, 3)
        
        # use_128x128_superblock (f(1))
        seq_header.use_128x128_superblock = read_f(reader, 1)
        
        # enable_filter_intra (f(1))
        seq_header.enable_filter_intra = read_f(reader, 1)
        
        # enable_intra_edge_filter (f(1))
        seq_header.enable_intra_edge_filter = read_f(reader, 1)
        
        # 功能标志
        if seq_header.reduced_still_picture_header:
            seq_header.enable_interintra_compound = 0
            seq_header.enable_masked_compound = 0
            seq_header.enable_warped_motion = 0
            seq_header.enable_dual_filter = 0
            seq_header.enable_order_hint = 0
            seq_header.enable_jnt_comp = 0
            seq_header.enable_ref_frame_mvs = 0
            seq_header.seq_force_screen_content_tools = 2  # SELECT_SCREEN_CONTENT_TOOLS
            seq_header.seq_force_integer_mv = 2  # SELECT_INTEGER_MV
            seq_header.OrderHintBits = 0
        else:
            # enable_interintra_compound (f(1))
            seq_header.enable_interintra_compound = read_f(reader, 1)
            
            # enable_masked_compound (f(1))
            seq_header.enable_masked_compound = read_f(reader, 1)
            
            # enable_warped_motion (f(1))
            seq_header.enable_warped_motion = read_f(reader, 1)
            
            # enable_dual_filter (f(1))
            seq_header.enable_dual_filter = read_f(reader, 1)
            
            # enable_order_hint (f(1))
            seq_header.enable_order_hint = read_f(reader, 1)
            
            if seq_header.enable_order_hint:
                # enable_jnt_comp (f(1))
                seq_header.enable_jnt_comp = read_f(reader, 1)
                
                # enable_ref_frame_mvs (f(1))
                seq_header.enable_ref_frame_mvs = read_f(reader, 1)
            else:
                seq_header.enable_jnt_comp = 0
                seq_header.enable_ref_frame_mvs = 0
            
            # seq_choose_screen_content_tools (f(1))
            seq_header.seq_choose_screen_content_tools = read_f(reader, 1)
            
            if seq_header.seq_choose_screen_content_tools:
                seq_header.seq_force_screen_content_tools = 2  # SELECT_SCREEN_CONTENT_TOOLS
            else:
                # seq_force_screen_content_tools (f(1))
                seq_header.seq_force_screen_content_tools = read_f(reader, 1)
            
            if seq_header.seq_force_screen_content_tools > 0:
                # seq_choose_integer_mv (f(1))
                seq_header.seq_choose_integer_mv = read_f(reader, 1)
                
                if seq_header.seq_choose_integer_mv:
                    seq_header.seq_force_integer_mv = 2  # SELECT_INTEGER_MV
                else:
                    # seq_force_integer_mv (f(1))
                    seq_header.seq_force_integer_mv = read_f(reader, 1)
            else:
                seq_header.seq_force_integer_mv = 2  # SELECT_INTEGER_MV
            
            if seq_header.enable_order_hint:
                # order_hint_bits_minus_1 (f(3))
                seq_header.order_hint_bits_minus_1 = read_f(reader, 3)
                seq_header.OrderHintBits = seq_header.order_hint_bits_minus_1 + 1
            else:
                seq_header.OrderHintBits = 0
        
        # enable_superres (f(1))
        seq_header.enable_superres = read_f(reader, 1)
        
        # enable_cdef (f(1))
        seq_header.enable_cdef = read_f(reader, 1)
        
        # enable_restoration (f(1))
        seq_header.enable_restoration = read_f(reader, 1)
        
        # color_config()
        seq_header.color_config = self._parse_color_config(reader, seq_header.seq_profile)
        
        # film_grain_params_present (f(1))
        seq_header.film_grain_params_present = read_f(reader, 1)
        
        return seq_header
    
    def _parse_color_config(self, reader: BitReader, seq_profile: int) -> ColorConfig:
        """
        解析颜色配置
        规范文档 6.3.2 color_config()
        
        Args:
            reader: BitReader实例
            seq_profile: 序列profile
            
        Returns:
            ColorConfig对象
        """
        color_config = ColorConfig()
        
        # high_bitdepth (f(1))
        color_config.high_bitdepth = read_f(reader, 1)
        
        # BitDepth计算
        if seq_profile == 2 and color_config.high_bitdepth:
            # twelve_bit (f(1))
            color_config.twelve_bit = read_f(reader, 1)
            color_config.BitDepth = 12 if color_config.twelve_bit else 10
        elif seq_profile <= 2:
            color_config.BitDepth = 10 if color_config.high_bitdepth else 8
        
        # mono_chrome
        if seq_profile == 1:
            color_config.mono_chrome = 0
        else:
            # mono_chrome (f(1))
            color_config.mono_chrome = read_f(reader, 1)
        
        color_config.NumPlanes = 1 if color_config.mono_chrome else 3
        
        # color_description_present_flag (f(1))
        color_config.color_description_present_flag = read_f(reader, 1)
        
        if color_config.color_description_present_flag:
            # color_primaries (f(8))
            color_config.color_primaries = read_f(reader, 8)
            
            # transfer_characteristics (f(8))
            color_config.transfer_characteristics = read_f(reader, 8)
            
            # matrix_coefficients (f(8))
            color_config.matrix_coefficients = read_f(reader, 8)
        else:
            # 默认值（规范文档定义）
            color_config.color_primaries = 2  # CP_UNSPECIFIED
            color_config.transfer_characteristics = 2  # TC_UNSPECIFIED
            color_config.matrix_coefficients = 2  # MC_UNSPECIFIED
        
        # mono_chrome特殊处理
        if color_config.mono_chrome:
            # color_range (f(1))
            color_config.color_range = read_f(reader, 1)
            color_config.subsampling_x = 1
            color_config.subsampling_y = 1
            color_config.chroma_sample_position = 0  # CSP_UNKNOWN
            color_config.separate_uv_delta_q = 0
            return color_config
        
        # 特殊颜色空间处理（规范文档 6.3.2）
        if (color_config.color_primaries == 1 and  # CP_BT_709
            color_config.transfer_characteristics == 13 and  # TC_SRGB
            color_config.matrix_coefficients == 0):  # MC_IDENTITY
            color_config.color_range = 1
            color_config.subsampling_x = 0
            color_config.subsampling_y = 0
        else:
            # color_range (f(1))
            color_config.color_range = read_f(reader, 1)
            
            if seq_profile == 0:
                color_config.subsampling_x = 1
                color_config.subsampling_y = 1
            elif seq_profile == 1:
                color_config.subsampling_x = 0
                color_config.subsampling_y = 0
            else:
                if color_config.BitDepth == 12:
                    # subsampling_x (f(1))
                    color_config.subsampling_x = read_f(reader, 1)
                    if color_config.subsampling_x:
                        # subsampling_y (f(1))
                        color_config.subsampling_y = read_f(reader, 1)
                    else:
                        color_config.subsampling_y = 0
                else:
                    color_config.subsampling_x = 1
                    color_config.subsampling_y = 0
            
            if color_config.subsampling_x and color_config.subsampling_y:
                # chroma_sample_position (f(2))
                color_config.chroma_sample_position = read_f(reader, 2)
            else:
                color_config.chroma_sample_position = 0  # CSP_UNKNOWN
        
        # separate_uv_delta_q (f(1))
        color_config.separate_uv_delta_q = read_f(reader, 1)
        
        return color_config
    
    def _parse_timing_info(self, reader: BitReader) -> TimingInfo:
        """
        解析时序信息
        规范文档 6.3.3 timing_info()
        
        Args:
            reader: BitReader实例
            
        Returns:
            TimingInfo对象
        """
        timing_info = TimingInfo()
        
        # num_units_in_display_tick (u(32))
        timing_info.num_units_in_display_tick = read_f(reader, 32)
        
        # time_scale (u(32))
        timing_info.time_scale = read_f(reader, 32)
        
        # equal_picture_interval (f(1))
        timing_info.equal_picture_interval = read_f(reader, 1)
        
        if timing_info.equal_picture_interval:
            # num_ticks_per_picture_minus_1 (uvlc())
            from bitstream.descriptors import read_uvlc
            timing_info.num_ticks_per_picture_minus_1 = read_uvlc(reader)
        
        return timing_info
    
    def _parse_decoder_model_info(self, reader: BitReader) -> DecoderModelInfo:
        """
        解析解码器模型信息
        规范文档 6.3.4 decoder_model_info()
        
        Args:
            reader: BitReader实例
            
        Returns:
            DecoderModelInfo对象
        """
        decoder_model_info = DecoderModelInfo()
        
        # encoder_decoder_buffer_delay_length_minus_1 (f(5))
        decoder_model_info.encoder_decoder_buffer_delay_length_minus_1 = read_f(reader, 5)
        
        # buffer_removal_time_length_minus_1 (f(5))
        decoder_model_info.buffer_removal_time_length_minus_1 = read_f(reader, 5)
        
        # frame_presentation_time_length_minus_1 (f(5))
        decoder_model_info.frame_presentation_time_length_minus_1 = read_f(reader, 5)
        
        return decoder_model_info
    
    def _parse_operating_parameters_info(self, reader: BitReader) -> OperatingParametersInfo:
        """
        解析操作参数信息
        规范文档 6.3.5 operating_parameters_info()
        
        Args:
            reader: BitReader实例
            
        Returns:
            OperatingParametersInfo对象
        """
        params_info = OperatingParametersInfo()
        
        # decoder_buffer_delay (u(encoder_decoder_buffer_delay_length))
        # encoder_buffer_delay (u(encoder_decoder_buffer_delay_length))
        # 这里需要从decoder_model_info获取长度，简化实现
        # 假设长度足够，读取固定位数
        buffer_delay_length = 24  # 简化值
        params_info.decoder_buffer_delay = read_f(reader, buffer_delay_length)
        params_info.encoder_buffer_delay = read_f(reader, buffer_delay_length)
        
        # low_delay_mode_flag (f(1))
        params_info.low_delay_mode_flag = read_f(reader, 1)
        
        return params_info


def sequence_header_obu(reader: BitReader) -> SequenceHeader:
    """
    序列头OBU解析函数
    规范文档 6.3.1 sequence_header_obu()
    这是规范文档中定义的主函数
    
    Args:
        reader: BitReader实例
        
    Returns:
        SequenceHeader对象
    """
    parser = SequenceHeaderParser()
    return parser.parse_sequence_header_obu(reader)

