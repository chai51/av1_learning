"""
AV1解码器主类
用于长期存储参考帧相关信息，并提供外部调用接口
"""

from bitstream.bit_reader import BitReader
from bitstream.descriptors import read_f, read_leb128

from obu.obu import OBU
from typing import List, Optional
from frame.ref_frame_store import RefFrameStore
from constants import OBU_HEADER_TYPE, NONE
from utils.math_utils import Array


class AV1Decoder:
    """
    AV1解码器主类
    用于长期存储参考帧相关信息，并提供统一的解码接口
    """

    def __init__(self):
        """
        初始化AV1解码器
        创建长期存储的参考帧存储和序列头
        """
        self.reader: BitReader = NONE

        self.obu: OBU = NONE
        # 序列头 - 在解码器生命周期内长期保存
        # 序列头在多个帧之间共享，直到遇到新的序列头
        from .seq_header import SequenceHeader
        self.seq_header: SequenceHeader = NONE

        # 当前帧头 - 临时存储，每帧更新
        from frame.frame_header import FrameHeader
        self.frame_header: FrameHeader = NONE

        # 解码器状态
        self.SeenFrameHeader = 0

        # 参考帧存储 - 在解码器生命周期内长期保存
        # 对应规范文档7.20 Reference frame update process
        self.ref_frame_store = RefFrameStore()

        # Tile组 - 在解码器生命周期内长期保存
        # 对应规范文档5.11 Tile group
        from tile.tile_group import TileGroup
        self.tile_group: TileGroup = NONE

        # 符号解码器 - 在解码器生命周期内长期保存
        # 对应规范文档8.2 Symbol decoder
        from entropy.symbol_decoder import SymbolDecoder
        self.decoder: SymbolDecoder = NONE

        self.obus: List[OBU] = list[OBU]()

        # 当前帧
        self.CurrFrame: List[List[List[int]]] = NONE

        # 输出帧
        self.OutY = NONE
        self.OutU = NONE
        self.OutV = NONE

        self.UpscaledCdefFrame: List[List[List[int]]] = NONE
        self.UpscaledCurrFrame: List[List[List[int]]] = NONE
        self.LrFrame: List[List[List[int]]] = NONE

        self.on_cdf = None
        self.on_symbol = None
        self.on_wmmat = None
        self.on_pred = None
        self.on_pred_frame = None
        self.on_residual_frame = None
        self.on_film_grain_frame = None

    def decode(self, frame_data: bytes) -> List[OBU]:
        self.__frame_unit(frame_data)
        return self.obus

    def __frame_unit(self, frame_data: bytes) -> None:
        """
        解析帧单元
        规范文档 B.2 Length delimited bitstream syntax
        """
        sz = len(frame_data)
        offset = 0

        while sz > 0:
            self.reader = BitReader(frame_data[offset:])
            obu_length = self.__open_bitstream_unit(sz)
            obu = self.obu

            if obu.header.obu_type == OBU_HEADER_TYPE.OBU_SEQUENCE_HEADER:
                self.seq_header = obu.seq_header
                width = self.seq_header.max_frame_width_minus_1 + 1
                height = self.seq_header.max_frame_height_minus_1 + 1
                subsampling_x = self.seq_header.color_config.subsampling_x
                subsampling_y = self.seq_header.color_config.subsampling_y
                self.OutY = Array(None, (height, width))
                self.OutU = Array(None, ((height + subsampling_y) >>
                                  subsampling_y, (width + subsampling_x) >> subsampling_x))
                self.OutV = Array(None, ((height + subsampling_y) >>
                                  subsampling_y, (width + subsampling_x) >> subsampling_x))

            offset += obu_length
            sz -= obu_length
            self.obus.append(obu)

        if self.on_film_grain_frame:
            self.on_film_grain_frame([self.OutY, self.OutU, self.OutV])

    def __open_bitstream_unit(self, sz: int) -> int:
        """
        解析开放比特流单元
        规范文档 5.3.1 General OBU syntax

        Args:
            sz: OBU大小（字节）

        Returns:
            OBU大小（字节）
        """
        from .obu import OBUHeaderParser
        self.obu = OBU()
        reader = self.reader
        seq_header = self.seq_header

        parser = OBUHeaderParser()
        headerPosition = reader.get_position()

        self.obu.header = parser.obu_header(reader)
        header = self.obu.header

        if header.obu_has_size_field:
            header.obu_size = read_leb128(reader)
        else:
            header.obu_size = sz - 1 - header.obu_extension_flag

        header.header_size = (reader.get_position() - headerPosition) // 8

        # It is a requirement of bitstream conformance that if OperatingPointIdc is equal to 0, then obu_extension_flag is equal to 0 for all OBUs that follow this sequence header until the next sequence header.
        if seq_header and seq_header.OperatingPointIdc == 0:
            assert header.obu_extension_flag == 0

        startPosition = reader.get_position()

        if (header.obu_type != OBU_HEADER_TYPE.OBU_SEQUENCE_HEADER and
            header.obu_type != OBU_HEADER_TYPE.OBU_TEMPORAL_DELIMITER and
            seq_header and seq_header.OperatingPointIdc != 0 and
                header.obu_extension_flag == 1):
            inTemporalLayer = (seq_header.OperatingPointIdc >>
                               header.temporal_id) & 1
            inSpatialLayer = (seq_header.OperatingPointIdc >>
                              (header.spatial_id + 8)) & 1
            if not inTemporalLayer or not inSpatialLayer:
                # drop_obu()
                return header.header_size + header.obu_size

        if header.obu_type == OBU_HEADER_TYPE.OBU_SEQUENCE_HEADER:
            from .seq_header import sequence_header_obu
            sequence_header_obu(self)
        elif header.obu_type == OBU_HEADER_TYPE.OBU_TEMPORAL_DELIMITER:
            self.__temporal_delimiter_obu()
        elif header.obu_type == OBU_HEADER_TYPE.OBU_FRAME_HEADER:
            from frame.frame_header import frame_header_obu
            frame_header_obu(self)
        elif header.obu_type == OBU_HEADER_TYPE.OBU_REDUNDANT_FRAME_HEADER:
            from frame.frame_header import frame_header_obu
            frame_header_obu(self)
        elif header.obu_type == OBU_HEADER_TYPE.OBU_TILE_GROUP:
            from tile.tile_group import tile_group_obu
            tile_group_obu(self, header.obu_size)
        elif header.obu_type == OBU_HEADER_TYPE.OBU_METADATA:
            from obu.metadata_obu import metadata_obu
            metadata_obu(self)
        elif header.obu_type == OBU_HEADER_TYPE.OBU_FRAME:
            from frame.frame_header import frame_obu
            frame_obu(self, header.obu_size)
        elif header.obu_type == OBU_HEADER_TYPE.OBU_TILE_LIST:
            from obu.tile_list_obu import tile_list_obu
            tile_list_obu(self)
        elif header.obu_type == OBU_HEADER_TYPE.OBU_PADDING:
            self.__padding_obu(reader, header.obu_size)
        else:
            # reserved_obu()
            pass

        currentPosition = reader.get_position()
        payloadBits = currentPosition - startPosition

        if (header.obu_size > 0 and
            header.obu_type != OBU_HEADER_TYPE.OBU_TILE_GROUP and
            header.obu_type != OBU_HEADER_TYPE.OBU_TILE_LIST and
                header.obu_type != OBU_HEADER_TYPE.OBU_FRAME):
            self.__trailing_bits(reader, header.obu_size * 8 - payloadBits)

        return header.header_size + header.obu_size

    def __trailing_bits(self, reader: BitReader, nbBits: int):
        """
        规范文档 5.3.4 Trailing bits syntax
        """
        trailing_one_bit = read_f(reader, 1)
        nbBits -= 1
        while nbBits > 0:
            trailing_zero_bit = read_f(reader, 1)
            nbBits -= 1

    def __temporal_delimiter_obu(self):
        """
        规范文档 5.6 Temporal delimiter obu syntax

        用于标记当前帧的帧头是否已被接收的变量
        """
        self.SeenFrameHeader = 0

    def __padding_obu(self, reader: BitReader, obu_padding_length: int):
        """
        规范文档 5.7 Padding OBU syntax
        """
        for i in range(obu_padding_length):
            obu_padding_byte = read_f(reader, 8)
