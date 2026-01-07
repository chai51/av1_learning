from bitstream.bit_reader import BitReader
from bitstream.descriptors import (
    read_f, read_uvlc, read_leb128, read_su, read_ns,
    read_le
)

def test_read_su():
    data = bytes([0xAA, 0xBB, 0xCC, 0xDD])
    reader = BitReader(data)
    assert read_su(reader, 8) == data[0] - (1 << 8)

