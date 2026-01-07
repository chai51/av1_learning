"""
Tile List OBU解析器
按照规范文档5.12节实现tile_list_obu()及相关函数
"""

from typing import List, Optional
from bitstream.descriptors import read_f
from obu.decoder import AV1Decoder
from constants import NONE


class TileListEntry:
    """
    Tile List Entry数据结构
    规范文档 5.12.2 tile_list_entry()
    """

    def __init__(self):
        self.anchor_frame_idx = 0
        self.anchor_tile_row = 0
        self.anchor_tile_col = 0
        self.tile_data_size_minus_1 = 0


class TileList:
    """
    Tile List OBU数据结构
    规范文档 5.12.1 tile_list_obu()
    """

    def __init__(self):
        self.output_frame_width_in_tiles_minus_1: Optional[int] = None
        self.output_frame_height_in_tiles_minus_1: Optional[int] = None
        self.tile_count_minus_1: Optional[int] = None

        self.tile_list_entries: List[TileListEntry] = [NONE] * 512


class TileListParser:
    """
    Tile List OBU解析器
    """

    def __init__(self):
        self.tile_list = TileList()

    def tile_list_obu(self, av1: AV1Decoder):
        """
        解析Tile List OBU
        规范文档 5.12.1 General tile list OBU syntax
        """
        reader = av1.reader
        tile_list = self.tile_list

        tile_list.output_frame_width_in_tiles_minus_1 = read_f(reader, 8)
        tile_list.output_frame_height_in_tiles_minus_1 = read_f(reader, 8)

        # It is a requirement of bitstream conformance that tile_count_minus_1 is less than or equal to 511.
        tile_list.tile_count_minus_1 = read_f(reader, 16)
        assert tile_list.tile_count_minus_1 <= 511

        for tile in range(tile_list.tile_count_minus_1 + 1):
            tile_list.tile_list_entries[tile] = self.__tile_list_entry(av1)

        return tile_list

    def __tile_list_entry(self, av1: AV1Decoder) -> TileListEntry:
        """
        解析Tile List Entry
        规范文档 5.12.2 Tile list entry syntax
        """
        reader = av1.reader
        frame_header = av1.frame_header
        tile_list_entry = TileListEntry()

        # It is a requirement of bitstream conformance that anchor_frame_idx is less than or equal to 127.
        tile_list_entry.anchor_frame_idx = read_f(reader, 8)
        assert tile_list_entry.anchor_frame_idx < 127

        # It is a requirement of bitstream conformance that anchor_tile_row is less than TileRows.
        tile_list_entry.anchor_tile_row = read_f(reader, 8)
        assert tile_list_entry.anchor_tile_row < frame_header.TileRows

        # It is a requirement of bitstream conformance that anchor_tile_col is less than TileCols.
        tile_list_entry.anchor_tile_col = read_f(reader, 8)
        assert tile_list_entry.anchor_tile_col < frame_header.TileCols

        tile_list_entry.tile_data_size_minus_1 = read_f(reader, 16)
        N = 8 * (tile_list_entry.tile_data_size_minus_1 + 1)
        # tile_list_entry.tile_data

        from frame.decoding_process import large_scale_tile_decoding_process
        large_scale_tile_decoding_process(av1)

        return tile_list_entry


def tile_list_obu(av1: AV1Decoder):
    """
    解析Tile List OBU
    规范文档 5.12.1 Tile list OBU syntax
    """
    tile_list_parser = TileListParser()
    av1.obu.tile_list = tile_list_parser.tile_list
    tile_list_parser.tile_list_obu(av1)
