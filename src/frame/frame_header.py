"""
帧头OBU解析器
按照规范文档6.8节实现frame_header_obu()
"""

from typing import Optional, List
from bitstream.bit_reader import BitReader
from bitstream.descriptors import read_f, read_uvlc
from constants import *
from sequence.sequence_header import SequenceHeader


class FrameHeader:
    """
    帧头结构
    规范文档 6.8.1 uncompressed_header()
    """
    def __init__(self):
        # 基础信息
        self.show_existing_frame = 0
        self.frame_to_show_map_idx = 0
        self.frame_type = 0
        self.FrameIsIntra = 0
        self.show_frame = 0
        self.showable_frame = 0
        self.error_resilient_mode = 0
        
        # 功能标志
        self.disable_cdf_update = 0
        self.allow_screen_content_tools = 0
        self.force_integer_mv = 0
        self.frame_id_numbers_present_flag = 0
        self.current_frame_id = 0
        self.frame_size_override_flag = 0
        self.order_hint = 0
        self.OrderHint = 0
        self.primary_ref_frame = 0
        
        # 缓冲区移除时间
        self.buffer_removal_time_present_flag = 0
        self.buffer_removal_time = []
        
        # 运动向量相关
        self.allow_high_precision_mv = 0
        self.use_ref_frame_mvs = 0
        self.allow_intrabc = 0
        
        # 参考帧刷新
        self.refresh_frame_flags = 0
        self.ref_order_hint = []  # [NUM_REF_FRAMES]
        
        # 参考帧索引
        self.frame_refs_short_signaling = 0
        self.last_frame_idx = 0
        self.gold_frame_idx = 0
        self.ref_frame_idx = []  # [REFS_PER_FRAME]
        self.delta_frame_id_minus_1 = []  # [REFS_PER_FRAME]
        self.expectedFrameId = []  # [REFS_PER_FRAME]
        
        # 帧尺寸
        self.frame_width = 0
        self.frame_height = 0
        self.UpscaledWidth = 0
        self.FrameWidth = 0
        self.FrameHeight = 0
        self.render_width = 0
        self.render_height = 0
        self.RenderWidth = 0
        self.RenderHeight = 0
        
        # 插值滤波
        self.interpolation_filter = 0
        self.is_motion_mode_switchable = 0
        
        # CDF更新
        self.disable_frame_end_update_cdf = 0
        
        # Tile信息（将在后续实现）
        self.tile_cols = 0
        self.tile_rows = 0
        
        # 量化参数（将在后续实现）
        self.base_q_idx = 0
        
        # 分段参数（将在后续实现）
        
        # 环路滤波参数（将在后续实现）


class FrameHeaderParser:
    """
    帧头解析器
    实现规范文档中描述的frame_header_obu()和uncompressed_header()函数
    """
    
    def __init__(self):
        # 解码器状态（规范文档中的状态变量）
        self.SeenFrameHeader = 0
        
    def frame_header_obu(self, reader: BitReader, seq_header: SequenceHeader):
        """
        解析帧头OBU
        规范文档 6.8.1 frame_header_obu()
        
        Args:
            reader: BitReader实例
            seq_header: 序列头（用于获取序列参数）
            
        Returns:
            FrameHeader对象
        """
        if self.SeenFrameHeader == 1:
            # frame_header_copy() - 复制上一帧的帧头
            # 简化实现：返回None表示使用上一帧
            return None
        else:
            self.SeenFrameHeader = 1
            frame_header = self.uncompressed_header(reader, seq_header)
            
            if frame_header.show_existing_frame:
                # decode_frame_wrapup() - 将在后续实现
                self.SeenFrameHeader = 0
            else:
                # TileNum = 0
                self.SeenFrameHeader = 1
            
            return frame_header
    
    def uncompressed_header(self, reader: BitReader, seq_header: SequenceHeader) -> FrameHeader:
        """
        解析未压缩帧头
        规范文档 6.8.1 uncompressed_header()
        
        Args:
            reader: BitReader实例
            seq_header: 序列头
            
        Returns:
            FrameHeader对象
        """
        frame_header = FrameHeader()
        
        # 计算idLen（如果frame_id_numbers_present_flag为真）
        idLen = 0
        if seq_header.frame_id_numbers_present_flag:
            idLen = (seq_header.additional_frame_id_length_minus_1 + 
                    seq_header.delta_frame_id_length_minus_2 + 3)
        
        allFrames = (1 << NUM_REF_FRAMES) - 1
        
        # 简化静态图片头处理
        if seq_header.reduced_still_picture_header:
            frame_header.show_existing_frame = 0
            frame_header.frame_type = KEY_FRAME
            frame_header.FrameIsIntra = 1
            frame_header.show_frame = 1
            frame_header.showable_frame = 0
        else:
            # show_existing_frame (f(1))
            frame_header.show_existing_frame = read_f(reader, 1)
            
            if frame_header.show_existing_frame == 1:
                # frame_to_show_map_idx (f(3))
                frame_header.frame_to_show_map_idx = read_f(reader, 3)
                
                # temporal_point_info() - 如果decoder_model_info_present_flag且!equal_picture_interval
                # 简化处理，暂时跳过
                
                frame_header.refresh_frame_flags = 0
                
                if seq_header.frame_id_numbers_present_flag:
                    # display_frame_id (f(idLen))
                    read_f(reader, idLen)  # 简化处理，暂不存储
                
                # frame_type = RefFrameType[frame_to_show_map_idx]
                # 这里需要从参考帧缓冲区获取，简化处理
                frame_header.frame_type = KEY_FRAME  # 默认值
                
                # 如果是KEY_FRAME，refresh_frame_flags = allFrames
                if frame_header.frame_type == KEY_FRAME:
                    frame_header.refresh_frame_flags = allFrames
                
                # load_grain_params() - 将在后续实现
                return frame_header
            
            # frame_type (f(2))
            frame_header.frame_type = read_f(reader, 2)
            
            # FrameIsIntra
            frame_header.FrameIsIntra = (frame_header.frame_type == INTRA_ONLY_FRAME or 
                                        frame_header.frame_type == KEY_FRAME)
            
            # show_frame (f(1))
            frame_header.show_frame = read_f(reader, 1)
            
            # temporal_point_info() - 如果show_frame && decoder_model_info_present_flag && !equal_picture_interval
            # 简化处理，暂时跳过
            
            # showable_frame
            if frame_header.show_frame:
                frame_header.showable_frame = (frame_header.frame_type != KEY_FRAME)
            else:
                # showable_frame (f(1))
                frame_header.showable_frame = read_f(reader, 1)
            
            # error_resilient_mode
            if (frame_header.frame_type == SWITCH_FRAME or
                (frame_header.frame_type == KEY_FRAME and frame_header.show_frame)):
                frame_header.error_resilient_mode = 1
            else:
                # error_resilient_mode (f(1))
                frame_header.error_resilient_mode = read_f(reader, 1)
        
        # KEY_FRAME && show_frame时的处理
        if frame_header.frame_type == KEY_FRAME and frame_header.show_frame:
            # RefValid和RefOrderHint重置
            # OrderHints重置
            # 简化处理，状态管理将在后续实现
            pass
        
        # disable_cdf_update (f(1))
        frame_header.disable_cdf_update = read_f(reader, 1)
        
        # allow_screen_content_tools
        if seq_header.seq_force_screen_content_tools == 2:  # SELECT_SCREEN_CONTENT_TOOLS
            # allow_screen_content_tools (f(1))
            frame_header.allow_screen_content_tools = read_f(reader, 1)
        else:
            frame_header.allow_screen_content_tools = seq_header.seq_force_screen_content_tools
        
        # force_integer_mv
        if frame_header.allow_screen_content_tools:
            if seq_header.seq_force_integer_mv == 2:  # SELECT_INTEGER_MV
                # force_integer_mv (f(1))
                frame_header.force_integer_mv = read_f(reader, 1)
            else:
                frame_header.force_integer_mv = seq_header.seq_force_integer_mv
        else:
            frame_header.force_integer_mv = 0
        
        # FrameIsIntra时强制force_integer_mv = 1
        if frame_header.FrameIsIntra:
            frame_header.force_integer_mv = 1
        
        # frame_id处理
        if seq_header.frame_id_numbers_present_flag:
            # PrevFrameID = current_frame_id
            # current_frame_id (f(idLen))
            frame_header.current_frame_id = read_f(reader, idLen)
            # mark_ref_frames(idLen) - 将在后续实现
        else:
            frame_header.current_frame_id = 0
        
        # frame_size_override_flag
        if frame_header.frame_type == SWITCH_FRAME:
            frame_header.frame_size_override_flag = 1
        elif seq_header.reduced_still_picture_header:
            frame_header.frame_size_override_flag = 0
        else:
            # frame_size_override_flag (f(1))
            frame_header.frame_size_override_flag = read_f(reader, 1)
        
        # order_hint (f(OrderHintBits))
        if seq_header.OrderHintBits > 0:
            frame_header.order_hint = read_f(reader, seq_header.OrderHintBits)
            frame_header.OrderHint = frame_header.order_hint
        
        # primary_ref_frame
        if frame_header.FrameIsIntra or frame_header.error_resilient_mode:
            frame_header.primary_ref_frame = 7  # PRIMARY_REF_NONE
        else:
            # primary_ref_frame (f(3))
            frame_header.primary_ref_frame = read_f(reader, 3)
        
        # buffer_removal_time处理（decoder_model_info_present_flag相关）
        # 简化处理，暂时跳过
        
        # 初始化一些标志
        frame_header.allow_high_precision_mv = 0
        frame_header.use_ref_frame_mvs = 0
        frame_header.allow_intrabc = 0
        
        # refresh_frame_flags
        if (frame_header.frame_type == SWITCH_FRAME or
            (frame_header.frame_type == KEY_FRAME and frame_header.show_frame)):
            frame_header.refresh_frame_flags = allFrames
        else:
            # refresh_frame_flags (f(8))
            frame_header.refresh_frame_flags = read_f(reader, 8)
        
        # ref_order_hint处理（error_resilient_mode相关）
        # 简化处理
        
        # 帧尺寸解析
        if frame_header.FrameIsIntra:
            # frame_size()
            self._parse_frame_size(reader, seq_header, frame_header)
            # render_size()
            self._parse_render_size(reader, seq_header, frame_header)
            
            # allow_intrabc
            if (frame_header.allow_screen_content_tools and 
                frame_header.UpscaledWidth == frame_header.FrameWidth):
                # allow_intrabc (f(1))
                frame_header.allow_intrabc = read_f(reader, 1)
        else:
            # 帧间帧的参考帧处理
            if not seq_header.enable_order_hint:
                frame_header.frame_refs_short_signaling = 0
            else:
                # frame_refs_short_signaling (f(1))
                frame_header.frame_refs_short_signaling = read_f(reader, 1)
                
                if frame_header.frame_refs_short_signaling:
                    # last_frame_idx (f(3))
                    frame_header.last_frame_idx = read_f(reader, 3)
                    # gold_frame_idx (f(3))
                    frame_header.gold_frame_idx = read_f(reader, 3)
                    # set_frame_refs() - 将在后续实现
            
            # ref_frame_idx处理
            frame_header.ref_frame_idx = []
            frame_header.delta_frame_id_minus_1 = []
            REFS_PER_FRAME = 3  # 规范文档定义
            
            for i in range(REFS_PER_FRAME):
                if not frame_header.frame_refs_short_signaling:
                    # ref_frame_idx[i] (f(3))
                    frame_header.ref_frame_idx.append(read_f(reader, 3))
                else:
                    # 从set_frame_refs()的结果获取，简化处理
                    frame_header.ref_frame_idx.append(0)
                
                # delta_frame_id_minus_1处理
                if seq_header.frame_id_numbers_present_flag:
                    n = seq_header.delta_frame_id_length_minus_2 + 2
                    # delta_frame_id_minus_1 (f(n))
                    delta = read_f(reader, n)
                    frame_header.delta_frame_id_minus_1.append(delta)
                    # expectedFrameId计算
                    DeltaFrameId = delta + 1
                    expectedId = ((frame_header.current_frame_id + (1 << idLen) - 
                                 DeltaFrameId) % (1 << idLen))
                    frame_header.expectedFrameId.append(expectedId)
            
            # 帧尺寸
            if frame_header.frame_size_override_flag and not frame_header.error_resilient_mode:
                # frame_size_with_refs()
                self._parse_frame_size_with_refs(reader, seq_header, frame_header)
            else:
                # frame_size()
                self._parse_frame_size(reader, seq_header, frame_header)
                # render_size()
                self._parse_render_size(reader, seq_header, frame_header)
            
            # allow_high_precision_mv
            if frame_header.force_integer_mv:
                frame_header.allow_high_precision_mv = 0
            else:
                # allow_high_precision_mv (f(1))
                frame_header.allow_high_precision_mv = read_f(reader, 1)
            
            # read_interpolation_filter()
            # 简化实现
            frame_header.interpolation_filter = 0
            
            # is_motion_mode_switchable (f(1))
            frame_header.is_motion_mode_switchable = read_f(reader, 1)
            
            # use_ref_frame_mvs
            if frame_header.error_resilient_mode or not seq_header.enable_ref_frame_mvs:
                frame_header.use_ref_frame_mvs = 0
            else:
                # use_ref_frame_mvs (f(1))
                frame_header.use_ref_frame_mvs = read_f(reader, 1)
            
            # OrderHints和RefFrameSignBias处理
            # 简化处理，将在后续实现
        
        # disable_frame_end_update_cdf
        if seq_header.reduced_still_picture_header or frame_header.disable_cdf_update:
            frame_header.disable_frame_end_update_cdf = 1
        else:
            # disable_frame_end_update_cdf (f(1))
            frame_header.disable_frame_end_update_cdf = read_f(reader, 1)
        
        # CDF和previous状态处理
        # init_non_coeff_cdfs(), setup_past_independence(), load_cdfs(), load_previous()
        # 这些将在后续熵解码模块实现
        
        # motion_field_estimation() - 将在后续实现
        if frame_header.use_ref_frame_mvs == 1:
            # motion_field_estimation()
            pass
        
        # tile_info()
        self._parse_tile_info(reader, seq_header, frame_header)
        
        # quantization_params()
        self._parse_quantization_params(reader, frame_header)
        
        # segmentation_params()
        # 将在后续实现
        
        # 其他参数将在后续实现
        
        return frame_header
    
    def _parse_frame_size(self, reader: BitReader, seq_header: SequenceHeader, 
                         frame_header: FrameHeader):
        """
        解析帧尺寸
        规范文档 6.8.3 frame_size()
        """
        # frame_width_minus_1 (f(n))
        n = seq_header.frame_width_bits_minus_1 + 1
        frame_width_minus_1 = read_f(reader, n)
        frame_header.frame_width = frame_width_minus_1 + 1
        
        # frame_height_minus_1 (f(n))
        n = seq_header.frame_height_bits_minus_1 + 1
        frame_height_minus_1 = read_f(reader, n)
        frame_header.frame_height = frame_height_minus_1 + 1
        
        # compute_image_size() - 简化实现
        frame_header.UpscaledWidth = frame_header.frame_width
        frame_header.FrameWidth = frame_header.frame_width
        frame_header.FrameHeight = frame_header.frame_height
    
    def _parse_render_size(self, reader: BitReader, seq_header: SequenceHeader,
                          frame_header: FrameHeader):
        """
        解析渲染尺寸
        规范文档 6.8.4 render_size()
        """
        # render_and_frame_size_different (f(1))
        render_and_frame_size_different = read_f(reader, 1)
        
        if render_and_frame_size_different:
            # render_width_minus_1 (f(16))
            render_width_minus_1 = read_f(reader, 16)
            frame_header.render_width = render_width_minus_1 + 1
            
            # render_height_minus_1 (f(16))
            render_height_minus_1 = read_f(reader, 16)
            frame_header.render_height = render_height_minus_1 + 1
        else:
            frame_header.render_width = frame_header.FrameWidth
            frame_header.render_height = frame_header.FrameHeight
        
        frame_header.RenderWidth = frame_header.render_width
        frame_header.RenderHeight = frame_header.render_height
    
    def _parse_frame_size_with_refs(self, reader: BitReader, seq_header: SequenceHeader,
                                   frame_header: FrameHeader):
        """
        解析带参考的帧尺寸
        规范文档 6.8.5 frame_size_with_refs()
        """
        # found_ref - 检查每个参考帧是否有匹配的尺寸
        # 简化实现：总是使用自己的尺寸
        found_ref = 0
        
        # 尝试从参考帧获取尺寸（简化处理）
        # 实际应该检查RefFrameWidth和RefFrameHeight
        
        if not found_ref:
            # 使用自己的尺寸
            self._parse_frame_size(reader, seq_header, frame_header)
            # render_size()
            self._parse_render_size(reader, seq_header, frame_header)
        else:
            # 从参考帧复制尺寸
            # 简化处理
            pass
    
    def _parse_tile_info(self, reader: BitReader, seq_header: SequenceHeader,
                        frame_header: FrameHeader):
        """
        解析Tile信息
        规范文档 6.8.10 tile_info()
        """
        # uniform_tile_spacing_flag (f(1))
        uniform_tile_spacing_flag = read_f(reader, 1)
        
        if uniform_tile_spacing_flag:
            # tile_cols_log2, tile_rows_log2通过计算得到
            # 简化实现
            frame_header.tile_cols = 1
            frame_header.tile_rows = 1
        else:
            # tile_cols_minus_1 (f(ceil(log2(MiCols))))
            # tile_rows_minus_1 (f(ceil(log2(MiRows))))
            # 简化实现
            tile_cols_minus_1 = read_f(reader, 2)  # 简化
            tile_rows_minus_1 = read_f(reader, 2)  # 简化
            frame_header.tile_cols = tile_cols_minus_1 + 1
            frame_header.tile_rows = tile_rows_minus_1 + 1
    
    def _parse_quantization_params(self, reader: BitReader, frame_header: FrameHeader):
        """
        解析量化参数
        规范文档 6.8.11 quantization_params()
        """
        # base_q_idx (f(8))
        frame_header.base_q_idx = read_f(reader, 8)
        
        # DeltaQ处理
        # 简化实现，只读取base_q_idx
        # delta_q_y_dc = 0
        # delta_q_u_dc = 0
        # delta_q_u_ac = 0
        # delta_q_v_dc = 0
        # delta_q_v_ac = 0


def frame_header_obu(reader: BitReader, seq_header: SequenceHeader) -> Optional[FrameHeader]:
    """
    帧头OBU解析函数
    规范文档 6.8.1 frame_header_obu()
    这是规范文档中定义的主函数
    
    Args:
        reader: BitReader实例
        seq_header: 序列头
        
    Returns:
        FrameHeader对象，如果使用帧头副本则返回None
    """
    parser = FrameHeaderParser()
    return parser.frame_header_obu(reader, seq_header)

