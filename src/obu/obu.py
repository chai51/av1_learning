"""
OBU (Open Bitstream Unit) 解析器
按照规范文档第5.3节实现OBU解析
"""

from typing import Optional
from bitstream.bit_reader import BitReader
from bitstream.descriptors import read_f
from constants import NONE, OBU_HEADER_TYPE


class OBUHeader:
    """
    OBU头结构
    规范文档 5.3.2
    """

    def __init__(self):
        self.obu_type: OBU_HEADER_TYPE = OBU_HEADER_TYPE.OBU_PADDING
        self.obu_extension_flag: int = 0
        self.obu_has_size_field: int = 0
        self.obu_size: int = 0
        self.temporal_id: int = 0
        self.spatial_id: int = 0
        self.header_size: int = 0


class OBU:
    """
    Bitstream Unit (OBU) 数据结构
    保存完整的OBU信息，包括头和payload数据
    规范文档 5.3.1 open_bitstream_unit()

    这个类作为根引用，用于透传所有额外数据。
    其他模块可以直接导入OBU，而不会形成循环依赖。
    """

    def __init__(self):
        self.header = OBUHeader()
        # OBU payload数据（根据类型不同）
        # OBU_SEQUENCE_HEADER
        from .seq_header import SequenceHeader
        self.seq_header: SequenceHeader = NONE

        # OBU_FRAME_HEADER / OBU_REDUNDANT_FRAME_HEADER
        from frame.frame_header import FrameHeader
        self.frame_header: Optional[FrameHeader] = None

        # OBU_TILE_GROUP
        from tile.tile_group import TileGroup
        self.tile_group: Optional[TileGroup] = None  # Tile组解析数据（待实现时补充具体类型）

        # OBU_METADATA
        from obu.metadata_obu import Metadata
        self.metadata_data: Optional[Metadata] = None

        # OBU_TILE_LIST
        from obu.tile_list_obu import TileList
        self.tile_list: Optional[TileList] = None

        from entropy.symbol_decoder import SymbolDecoder
        self.decoder: Optional[SymbolDecoder] = None  # SymbolDecoder实例

        from reconstruction.prediction import Prediction
        self.prediction: Optional[Prediction] = None


class OBUHeaderParser:
    """
    OBU头解析器
    实现规范文档中描述的OBU头解析流程
    """

    def __init__(self):
        self.header = OBUHeader()

    def obu_header(self, reader: BitReader) -> OBUHeader:
        """
        规范文档 5.3.2 OBU header syntax
        """
        header = self.header

        obu_forbidden_bit = read_f(reader, 1)
        assert obu_forbidden_bit == 0

        header.obu_type = OBU_HEADER_TYPE(read_f(reader, 4))
        header.obu_extension_flag = read_f(reader, 1)
        header.obu_has_size_field = read_f(reader, 1)
        obu_reserved_1bit = read_f(reader, 1)
        if header.obu_extension_flag == 1:
            self.__obu_extension_header(reader)

        return header

    def __obu_extension_header(self, reader: BitReader):
        """
        规范文档 5.3.3 OBU extension header syntax
        """
        header = self.header

        header.temporal_id = read_f(reader, 3)
        header.spatial_id = read_f(reader, 2)
        extension_header_reserved_3bits = read_f(reader, 3)
