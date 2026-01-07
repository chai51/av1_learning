"""
Metadata OBU解析器
按照规范文档5.8节实现metadata_obu()及相关函数
"""

from bitstream.descriptors import read_f, read_leb128
from constants import METADATA_TYPE, SCALABILITY_MODE_IDC
from obu.decoder import AV1Decoder


class Metadata:
    """
    Metadata OBU解析器
    """

    def __init__(self):

        self.metadata_type: METADATA_TYPE = METADATA_TYPE.METADATA_TYPE_HDR_CLL
        self.data = None


class MetadataOBUParser:
    """
    Metadata OBU解析器
    实现规范文档中描述的metadata_obu()函数
    """

    def __init__(self):
        self.metadata = Metadata()

    def metadata_obu(self, av1: AV1Decoder) -> Metadata:
        """
        规范文档 5.8.1 General metadata OBU syntax
        """
        reader = av1.reader
        metadata = self.metadata

        metadata.metadata_type = METADATA_TYPE(read_leb128(reader))
        if metadata.metadata_type == METADATA_TYPE.METADATA_TYPE_ITUT_T35:
            self.__metadata_itut_t35(av1)
        elif metadata.metadata_type == METADATA_TYPE.METADATA_TYPE_HDR_CLL:
            self.__metadata_hdr_cll(av1)
        elif metadata.metadata_type == METADATA_TYPE.METADATA_TYPE_HDR_MDCV:
            self.__metadata_hdr_mdcv(av1)
        elif metadata.metadata_type == METADATA_TYPE.METADATA_TYPE_SCALABILITY:
            self.__metadata_scalability(av1)
        elif metadata.metadata_type == METADATA_TYPE.METADATA_TYPE_TIMECODE:
            self.__metadata_timecode(av1)

        return metadata

    def __metadata_itut_t35(self, av1: AV1Decoder):
        """
        解析ITU-T T.35 Metadata
        规范文档 5.8.2 Metadata ITUT T35 syntax
        """
        reader = av1.reader

        itu_t_t35_country_code = read_f(reader, 8)
        if itu_t_t35_country_code == 0xFF:
            itu_t_t35_country_code_extension_byte = read_f(reader, 8)
        # itu_t_t35_payload_bytes

    def __metadata_hdr_cll(self, av1: AV1Decoder):
        """
        解析HDR CLL (Content Light Level) Metadata
        规范文档 5.8.3 Metadata high dynamic range content light level syntax
        """
        reader = av1.reader

        max_cll = read_f(reader, 16)
        max_fall = read_f(reader, 16)

    def __metadata_hdr_mdcv(self, av1: AV1Decoder):
        """
        解析HDR MDCV (Mastering Display Color Volume) Metadata
        规范文档 5.8.4 Metadata high dynamic range mastering display color volume syntax
        """
        reader = av1.reader

        for i in range(3):
            primary_chromaticity_x = read_f(reader, 16)
            primary_chromaticity_y = read_f(reader, 16)
        white_point_chromaticity_x = read_f(reader, 16)
        white_point_chromaticity_y = read_f(reader, 16)
        luminance_max = read_f(reader, 32)
        luminance_min = read_f(reader, 32)

    def __metadata_scalability(self, av1: AV1Decoder):
        """
        解析Scalability Metadata
        规范文档 5.8.5 Metadata scalability syntax
        """
        reader = av1.reader

        scalability_mode_idc = read_f(reader, 8)
        if scalability_mode_idc == SCALABILITY_MODE_IDC.SCALABILITY_SS:
            self.__scalability_structure(av1)

    def __scalability_structure(self, av1: AV1Decoder):
        """
        解析Scalability Structure
        规范文档 5.8.6 Scalability structure syntax
        """
        reader = av1.reader

        spatial_layers_cnt_minus_1 = read_f(reader, 2)
        spatial_layer_dimensions_present_flag = read_f(reader, 1)
        spatial_layer_description_present_flag = read_f(reader, 1)
        temporal_group_description_present_flag = read_f(reader, 1)
        scalability_structure_reserved_3bits = read_f(reader, 3)
        if spatial_layer_dimensions_present_flag:
            for i in range(spatial_layers_cnt_minus_1 + 1):
                spatial_layer_max_width = read_f(reader, 16)
                spatial_layer_max_height = read_f(reader, 16)
        if spatial_layer_description_present_flag:
            for i in range(spatial_layers_cnt_minus_1 + 1):
                spatial_layer_ref_id = read_f(reader, 8)
        if temporal_group_description_present_flag:
            temporal_group_size = read_f(reader, 8)
            for i in range(temporal_group_size):
                temporal_group_temporal_id = read_f(reader, 3)
                temporal_group_temporal_switching_up_point_flag = read_f(
                    reader, 1)
                temporal_group_spatial_switching_up_point_flag = read_f(
                    reader, 1)
                temporal_group_ref_cnt = read_f(reader, 3)
                for j in range(temporal_group_ref_cnt):
                    temporal_group_ref_pic_diff = read_f(reader, 8)

    def __metadata_timecode(self, av1: AV1Decoder):
        """
        解析Timecode Metadata
        规范文档 5.8.7 Metadata timecode syntax
        """
        reader = av1.reader

        counting_type = read_f(reader, 5)
        full_timestamp_flag = read_f(reader, 1)
        discontinuity_flag = read_f(reader, 1)
        cnt_dropped_flag = read_f(reader, 1)
        n_frames = read_f(reader, 9)
        if full_timestamp_flag:
            seconds_value = read_f(reader, 6)
            minutes_value = read_f(reader, 6)
            hours_value = read_f(reader, 5)
        else:
            seconds_flag = read_f(reader, 1)
            if seconds_flag:
                seconds_value = read_f(reader, 6)
                minutes_flag = read_f(reader, 1)
                if minutes_flag:
                    minutes_value = read_f(reader, 6)
                    hours_flag = read_f(reader, 1)
                    if hours_flag:
                        hours_value = read_f(reader, 5)
        time_offset_length = read_f(reader, 5)
        if time_offset_length > 0:
            time_offset_value = read_f(reader, time_offset_length)


def metadata_obu(av1: AV1Decoder):
    """
    解析Metadata OBU
    规范文档 5.8 metadata_obu()
    这是模块级函数，作为对外接口
    """
    parser = MetadataOBUParser()
    av1.obu.metadata_data = parser.metadata_obu(av1)
