"""
OBU (Open Bitstream Unit) 解析器
按照规范文档第5.3节实现OBU解析
"""

from typing import Optional
from bitstream.bit_reader import BitReader
from bitstream.descriptors import read_f, read_leb128
from constants import *
from sequence.sequence_header import sequence_header_obu, SequenceHeader
from frame.frame_header import frame_header_obu, FrameHeader
from tile.tile_group import tile_group_obu


class OBUHeader:
    """
    OBU头结构
    规范文档 5.3.2
    """
    def __init__(self):
        self.obu_forbidden_bit = 0
        self.obu_type = 0
        self.obu_extension_flag = 0
        self.obu_has_size_field = 0
        self.obu_size = 0
        
        # Extension header (规范文档 5.3.3)
        self.temporal_id = 0
        self.spatial_id = 0
        
        # OBU payload数据（根据类型不同）
        self.sequence_header = None  # 如果是序列头OBU
        self.frame_header = None  # 如果是帧头OBU
        self.tile_group_parsed = False  # Tile组是否已解析


class OBUParser:
    """
    OBU解析器
    实现规范文档中描述的OBU解析流程
    """
    
    def parse_obu_header(self, reader: BitReader) -> OBUHeader:
        """
        解析OBU头
        规范文档 5.3.2
        
        Args:
            reader: BitReader实例
            
        Returns:
            OBUHeader对象
        """
        header = OBUHeader()
        
        # obu_forbidden_bit (f(1))
        header.obu_forbidden_bit = read_f(reader, 1)
        
        # obu_type (f(4))
        header.obu_type = read_f(reader, 4)
        
        # obu_extension_flag (f(1))
        header.obu_extension_flag = read_f(reader, 1)
        
        # obu_has_size_field (f(1))
        header.obu_has_size_field = read_f(reader, 1)
        
        # 如果有扩展头，解析扩展头
        if header.obu_extension_flag == 1:
            # temporal_id (f(3))
            header.temporal_id = read_f(reader, 3)
            # spatial_id (f(2))
            header.spatial_id = read_f(reader, 2)
            # extension_header_reserved_3bits (f(3))
            read_f(reader, 3)  # 保留位，丢弃
        
        # 如果有大小字段，读取大小
        if header.obu_has_size_field == 1:
            # obu_size (leb128())
            header.obu_size = read_leb128(reader)
        
        return header
    
    def parse_obu(self, data: bytes, sz: int, 
                 seq_header: Optional[SequenceHeader] = None,
                 frame_header: Optional[FrameHeader] = None) -> Optional[OBUHeader]:
        """
        解析OBU
        规范文档 5.3.1 general_obu_syntax()
        
        Args:
            data: OBU字节数据
            sz: OBU大小（字节）
            
        Returns:
            OBUHeader对象，如果解析失败返回None
        """
        reader = BitReader(data)
        
        start_position = reader.get_position()
        
        # 解析OBU头
        header = self.parse_obu_header(reader)
        
        # 检查forbidden bit（规范文档要求）
        if header.obu_forbidden_bit != 0:
            return None
        
        # 如果没有大小字段，计算大小
        if header.obu_has_size_field == 0:
            # obu_size = sz - 1 - obu_extension_flag
            header.obu_size = sz - 1 - header.obu_extension_flag
        
        # 根据OBU类型解析payload
        if header.obu_type == OBU_SEQUENCE_HEADER:
            # sequence_header_obu()
            header.sequence_header = sequence_header_obu(reader)
        elif header.obu_type == OBU_FRAME_HEADER or header.obu_type == OBU_REDUNDANT_FRAME_HEADER:
            # frame_header_obu()
            if seq_header:
                header.frame_header = frame_header_obu(reader, seq_header)
        elif header.obu_type == OBU_TILE_GROUP:
            # tile_group_obu(obu_size)
            if seq_header and frame_header:
                tile_group_obu(reader, header.obu_size, seq_header, frame_header)
                header.tile_group_parsed = True
        # 其他OBU类型的解析将在后续实现
        
        # 获取当前位置
        current_position = reader.get_position()
        
        # 计算payload位数
        payload_bits = current_position - start_position
        
        # 对于某些OBU类型，需要处理trailing_bits
        # （规范文档 5.3.5）
        if (header.obu_size > 0 and 
            header.obu_type != OBU_TILE_GROUP and
            header.obu_type != OBU_TILE_LIST and
            header.obu_type != OBU_FRAME):
            # 计算需要读取的trailing bits
            trailing_bits = header.obu_size * 8 - payload_bits
            if trailing_bits > 0:
                # trailing_bits() 函数（规范文档 5.3.5）
                self._parse_trailing_bits(reader, trailing_bits)
        
        return header
    
    def _parse_trailing_bits(self, reader: BitReader, num_bits: int):
        """
        解析trailing bits
        规范文档 5.3.5 trailing_bits_syntax()
        
        Args:
            reader: BitReader实例
            num_bits: 要读取的trailing bits数量
        """
        # trailing_bits是一个或多个1位，后跟0位
        # 读取所有位直到遇到0
        for _ in range(num_bits):
            bit = reader.read_bit()
            if bit == 0:
                break
    
    def byte_alignment(self, reader: BitReader):
        """
        字节对齐
        规范文档 5.3.6 byte_alignment_syntax()
        
        Args:
            reader: BitReader实例
        """
        # alignment_bit_equal_to_one (f(1))
        alignment_bit = read_f(reader, 1)
        # 规范要求此位必须为1
        # 但按照用户要求，不做额外的异常判断
        
        # 读取剩余的位直到字节对齐
        while reader.bit_offset != 0:
            # zero_bit (f(1)) - 必须为0
            zero_bit = read_f(reader, 1)
            # 不做异常判断


def obu_syntax(reader: BitReader, sz: int):
    """
    OBU语法解析函数
    规范文档 5.3.1 general_obu_syntax()
    这是规范文档中定义的主函数
    
    Args:
        reader: BitReader实例
        sz: OBU大小（字节）
        
    Returns:
        OBUHeader对象
    """
    parser = OBUParser()
    data = reader.data
    return parser.parse_obu(data, sz)

