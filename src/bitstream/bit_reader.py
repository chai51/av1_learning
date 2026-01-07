"""
比特流读取器
实现规范文档中描述的各种位读取操作
"""


class BitReader:
    """
    比特流读取器
    """

    def __init__(self, data: bytes):
        """
        初始化比特流读取器

        Args:
            data: 输入的字节数据
        """
        self.data = data
        self.byte_pos = 0  # 当前字节位置
        self.bit_offset = 0  # 当前位偏移（0-7）

    def get_position(self) -> int:
        """
        获取当前读取位置（以位为单位）
        规范文档 4.9

        Returns:
            当前读取的位数
        """
        return self.byte_pos * 8 + self.bit_offset

    def read_bit(self) -> int:
        """
        读取单个位

        Returns:
            读取的位值（0 或 1）
        """
        if self.byte_pos >= len(self.data):
            assert False

        byte_val = self.data[self.byte_pos]
        bit = (byte_val >> (7 - self.bit_offset)) & 1

        self.bit_offset += 1
        if self.bit_offset >= 8:
            self.bit_offset = 0
            self.byte_pos += 1

        return bit

    def read_bits(self, n: int) -> int:
        """
        规范文档 8.1 Parsing process for f(n)

        Args:
            n: 要读取的位数

        Returns:
            读取的n位值
        """
        x = 0
        for i in range(n):
            x = 2 * x + self.read_bit()
        return x

    def read_bytes(self, n: int) -> bytes:
        """
        读取n个字节
        在需要字节对齐时使用

        Args:
            n: 要读取的字节数

        Returns:
            读取的字节数据
        """
        # 确保字节对齐
        if self.bit_offset != 0:
            self.byte_pos += 1
            self.bit_offset = 0

        result = self.data[self.byte_pos:self.byte_pos + n]
        self.byte_pos += n
        return result

    def byte_alignment(self):
        """
        字节对齐
        规范文档 5.3.5 Byte alignment syntax
        """
        if self.bit_offset != 0:
            self.byte_pos += 1
            self.bit_offset = 0

    def set_position(self, position: int):
        """
        设置读取位置
        """
        self.byte_pos = position // 8
        self.bit_offset = position % 8

    def set_offset(self, offset: int):
        """
        设置当前位置偏移
        """
        self.set_position(self.get_position() + offset)
