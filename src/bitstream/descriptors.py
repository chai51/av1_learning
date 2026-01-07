"""
描述符实现
按照规范文档第3.3节描述的各种描述符实现
"""

from constants import UINT32_MAX
from bitstream.bit_reader import BitReader


def read_f(reader: BitReader, n: int) -> int:
    """
    规范文档 8.1 Parsing process for f(n)

    Args:
        reader: BitReader实例
        n: 位数

    Returns:
        读取的n位无符号整数
    """
    return reader.read_bits(n)


def read_uvlc(reader: BitReader) -> int:
    """
    uvlc() 描述符：无符号可变长度编码
    规范文档 4.10.3

    解析方式：
    - 从比特流中最高有效位开始
    - 读取x个前导零位(leading zeros)
    - 读取x位的无符号整数值
    - 在x+1位设置为1

    Returns:
        读取的uvlc编码的无符号整数
    """
    leadingZeros = 0
    while reader.read_bit() == 0:
        leadingZeros += 1

    if leadingZeros >= 32:
        return UINT32_MAX

    value = reader.read_bits(leadingZeros)
    return value + (1 << leadingZeros) - 1


def read_le(reader: BitReader, n: int) -> int:
    """
    小端序无符号整数
    规范文档 4.10.4

    Args:
        reader: BitReader实例
        n: 字节数（1, 2, 4, 8）

    Returns:
        小端序解码的无符号整数
    """
    # Note: This syntax element will only be present when the bitstream position is byte aligned.
    assert reader.get_position() % 8 == 0

    t = 0
    for i in range(n):
        byte = reader.read_bits(8)
        t += byte << (i * 8)
    return t


def read_leb128(reader: BitReader) -> int:
    """
    可变数量的小端字节表示的无符号整数
    规范文档 4.10.5

    解析方式：
    - 每个字节的最高位为1表示需要读取更多字节，为0则表示编码结束

    Returns:
        解码后的无符号整数
    """
    # Note: This syntax element will only be present when the bitstream position is byte aligned.
    assert reader.get_position() % 8 == 0

    value = 0
    for i in range(8):
        leb128_byte = reader.read_bits(8)
        value |= (leb128_byte & 0x7F) << (i * 7)

        if not (leb128_byte & 0x80):
            break

        # It is a requirement of bitstream conformance that the most significant bit of leb128_byte is equal to 0 if i is equal to 7. (This ensures that this syntax descriptor never uses more than 8 bytes.)
        if i == 7:
            assert (leb128_byte & 0x80) == 0

    # It is a requirement of bitstream conformance that the value returned from the leb128 parsing process is less than or equal to (1 << 32) - 1.
    assert value <= UINT32_MAX

    return value


def read_su(reader: BitReader, n: int) -> int:
    """
    n位无符号整数转换而来的有符号整数
    规范文档 4.10.6

    Args:
        reader: BitReader实例
        n: 位数

    Returns:
        有符号整数
    """
    value = reader.read_bits(n)
    signMask = 1 << (n - 1)
    # 转换为有符号（补码表示）
    if value & signMask:
        value = value - 2 * signMask
    return value


def read_ns(reader: BitReader, n: int) -> int:
    """
    非对称数值编码
    规范文档 4.10.7

    如果 n > 0，则读取 uvlc() + 1
    如果 n == 0，则读取 uvlc()

    Args:
        reader: BitReader实例
        n: 参数（用于调整值）

    Returns:
        解码后的值
    """
    from utils.math_utils import FloorLog2

    w = FloorLog2(n) + 1
    m = (1 << w) - n
    v = read_f(reader, w - 1)

    if v < m:
        return v

    extra_bit = read_f(reader, 1)
    return (v << 1) - m + extra_bit
