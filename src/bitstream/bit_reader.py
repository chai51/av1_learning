"""
比特流读取器
实现规范文档中描述的各种位读取操作
"""


class BitReader:
    """
    比特流读取器
    按照规范文档描述的比特流读取操作实现
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
        规范文档中使用 get_position() 函数
        
        Returns:
            当前读取的位数
        """
        return self.byte_pos * 8 + self.bit_offset
    
    def read_bit(self) -> int:
        """
        读取单个位
        规范文档中的 f(1) 描述符会调用此函数
        
        Returns:
            读取的位值（0 或 1）
        """
        if self.byte_pos >= len(self.data):
            return 0
            
        byte_val = self.data[self.byte_pos]
        bit = (byte_val >> (7 - self.bit_offset)) & 1
        
        self.bit_offset += 1
        if self.bit_offset >= 8:
            self.bit_offset = 0
            self.byte_pos += 1
            
        return bit
    
    def read_bits(self, n: int) -> int:
        """
        读取n位
        对应规范文档中的 f(n) 描述符
        
        Args:
            n: 要读取的位数
            
        Returns:
            读取的n位值
        """
        value = 0
        for i in range(n):
            bit = self.read_bit()
            value = (value << 1) | bit
        return value
    
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
    
    def byte_align(self):
        """
        字节对齐
        规范文档中的 byte_alignment() 语法元素会调用此函数
        """
        if self.bit_offset != 0:
            self.byte_pos += 1
            self.bit_offset = 0
    
    def bits_remaining(self) -> int:
        """
        计算剩余位数
        
        Returns:
            剩余的位数
        """
        total_bits = len(self.data) * 8
        current_pos = self.get_position()
        return total_bits - current_pos

