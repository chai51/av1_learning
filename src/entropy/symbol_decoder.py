"""
符号解码器（Symbol Decoder）
按照规范文档8.2节实现熵解码功能
"""

from typing import List
from bitstream.bit_reader import BitReader
from bitstream.descriptors import read_f


# 规范文档中定义的常量
EC_PROB_SHIFT = 6  # 规范文档中定义
EC_MIN_PROB = 4    # 规范文档中定义


class SymbolDecoder:
    """
    符号解码器
    实现规范文档8.2节描述的符号解码功能
    """
    
    def __init__(self, reader: BitReader):
        """
        初始化符号解码器
        
        Args:
            reader: BitReader实例
        """
        self.reader = reader
        self.SymbolValue = 0
        self.SymbolRange = 0
        self.SymbolMaxBits = 0
    
    def init_symbol(self, sz: int):
        """
        初始化符号解码器
        规范文档 8.2.2 initialization_process_for_symbol_decoder()
        
        Args:
            sz: 要读取的字节数
        """
        # numBits = Min(sz * 8, 15)
        numBits = min(sz * 8, 15)
        
        # buf = f(numBits)
        buf = read_f(self.reader, numBits)
        
        # paddedBuf = buf << (15 - numBits)
        paddedBuf = buf << (15 - numBits)
        
        # SymbolValue = ((1 << 15) - 1) ^ paddedBuf
        self.SymbolValue = ((1 << 15) - 1) ^ paddedBuf
        
        # SymbolRange = 1 << 15
        self.SymbolRange = 1 << 15
        
        # SymbolMaxBits = 8 * sz - 15
        self.SymbolMaxBits = 8 * sz - 15
    
    def read_bool(self) -> int:
        """
        读取布尔值
        规范文档 8.2.3 boolean_decoding_process()
        
        Returns:
            解码的布尔值（0或1）
        """
        # 构造CDF数组
        cdf = [1 << 14, 1 << 15, 0]
        
        # 调用read_symbol，但不更新CDF（因为每次调用read_bool都重新构造CDF）
        symbol = self.read_symbol(cdf, update_cdf=False)
        
        return symbol
    
    def read_literal(self, n: int) -> int:
        """
        读取字面量
        规范文档 8.2.5 parsing_process_for_read_literal()
        
        Args:
            n: 要读取的位数
            
        Returns:
            解码的n位无符号整数
        """
        x = 0
        for i in range(n):
            x = 2 * x + self.read_bool()
        return x
    
    def read_symbol(self, cdf: List[int], update_cdf: bool = True) -> int:
        """
        读取符号
        规范文档 8.2.4 symbol_decoding_process()
        
        Args:
            cdf: 累积分布函数数组（长度为N+1，N为符号数量）
            update_cdf: 是否更新CDF（默认True）
            
        Returns:
            解码的符号值
        """
        N = len(cdf) - 1  # 符号数量
        
        # cur, prev, symbol计算
        cur = self.SymbolRange
        symbol = -1
        
        # 查找符号
        while True:
            symbol += 1
            prev = cur
            
            # f = (1 << 15) - cdf[symbol]
            f = (1 << 15) - cdf[symbol]
            
            # cur = ((SymbolRange >> 8) * (f >> EC_PROB_SHIFT)) >> (7 - EC_PROB_SHIFT)
            cur = ((self.SymbolRange >> 8) * (f >> EC_PROB_SHIFT)) >> (7 - EC_PROB_SHIFT)
            
            # cur += EC_MIN_PROB * (N - symbol - 1)
            cur += EC_MIN_PROB * (N - symbol - 1)
            
            if self.SymbolValue < cur:
                break
        
        # SymbolRange = prev - cur
        self.SymbolRange = prev - cur
        
        # SymbolValue = SymbolValue - cur
        self.SymbolValue -= cur
        
        # 重归一化
        self._renormalize()
        
        # CDF更新
        if update_cdf:
            self._update_cdf(cdf, symbol)
        
        return symbol
    
    def _renormalize(self):
        """
        重归一化
        规范文档 8.2.4中描述的重归一化步骤
        """
        # bits = 15 - FloorLog2(SymbolRange)
        bits = 15 - self._floor_log2(self.SymbolRange)
        
        # SymbolRange = SymbolRange << bits
        self.SymbolRange <<= bits
        
        # numBits = Min(bits, Max(0, SymbolMaxBits))
        numBits = min(bits, max(0, self.SymbolMaxBits))
        
        # newData = f(numBits)
        newData = read_f(self.reader, numBits) if numBits > 0 else 0
        
        # paddedData = newData << (bits - numBits)
        paddedData = newData << (bits - numBits)
        
        # SymbolValue = paddedData ^ (((SymbolValue + 1) << bits) - 1)
        self.SymbolValue = paddedData ^ (((self.SymbolValue + 1) << bits) - 1)
        
        # SymbolMaxBits = SymbolMaxBits - bits
        self.SymbolMaxBits -= bits
    
    def _update_cdf(self, cdf: List[int], symbol: int):
        """
        更新CDF
        规范文档 8.2.4中描述的CDF更新过程
        
        Args:
            cdf: CDF数组（会被修改）
            symbol: 解码的符号值
        """
        N = len(cdf) - 1
        
        # rate计算
        rate = (3 + 
                (1 if cdf[N] > 15 else 0) +
                (1 if cdf[N] > 31 else 0) +
                min(self._floor_log2(N), 2))
        
        tmp = 0
        for i in range(N - 1):
            if i == symbol:
                tmp = 1 << 15
            
            if tmp < cdf[i]:
                cdf[i] -= ((cdf[i] - tmp) >> rate)
            else:
                cdf[i] += ((tmp - cdf[i]) >> rate)
        
        # cdf[N] += (cdf[N] < 32)
        if cdf[N] < 32:
            cdf[N] += 1
    
    def exit_symbol(self):
        """
        退出符号解码器
        规范文档 8.2.4 exit_process_for_symbol_decoder()
        """
        # trailingBitPosition = get_position() - Min(15, SymbolMaxBits + 15)
        current_pos = self.reader.get_position()
        trailingBitPosition = current_pos - min(15, self.SymbolMaxBits + 15)
        
        # 跳过剩余的位
        # bitstream position indicator is advanced by Max(0, SymbolMaxBits)
        bits_to_skip = max(0, self.SymbolMaxBits)
        if bits_to_skip > 0:
            self.reader.read_bits(bits_to_skip)
        
        # paddingEndPosition = get_position()
        paddingEndPosition = self.reader.get_position()
        
        # 规范要求trailingBitPosition位置必须是1，之后到paddingEndPosition必须是0
        # 但按照用户要求，不做异常判断
    
    def _floor_log2(self, x: int) -> int:
        """
        计算FloorLog2(x)
        
        Args:
            x: 输入值
            
        Returns:
            FloorLog2(x)的值
        """
        if x <= 0:
            return 0
        
        result = 0
        while x > 1:
            x >>= 1
            result += 1
        return result


# 全局函数，按照规范文档命名
def init_symbol(reader: BitReader, sz: int) -> SymbolDecoder:
    """
    初始化符号解码器
    规范文档中定义的函数
    
    Args:
        reader: BitReader实例
        sz: 字节数
        
    Returns:
        SymbolDecoder实例
    """
    decoder = SymbolDecoder(reader)
    decoder.init_symbol(sz)
    return decoder


def read_bool(decoder: SymbolDecoder) -> int:
    """
    读取布尔值
    规范文档中定义的函数
    
    Args:
        decoder: SymbolDecoder实例
        
    Returns:
        布尔值（0或1）
    """
    return decoder.read_bool()


def read_literal(decoder: SymbolDecoder, n: int) -> int:
    """
    读取字面量
    规范文档中定义的函数
    
    Args:
        decoder: SymbolDecoder实例
        n: 位数
        
    Returns:
        n位无符号整数
    """
    return decoder.read_literal(n)


def read_symbol(decoder: SymbolDecoder, cdf: List[int]) -> int:
    """
    读取符号
    规范文档中定义的函数
    
    Args:
        decoder: SymbolDecoder实例
        cdf: CDF数组
        
    Returns:
        解码的符号值
    """
    return decoder.read_symbol(cdf)


def exit_symbol(decoder: SymbolDecoder):
    """
    退出符号解码器
    规范文档中定义的函数
    
    Args:
        decoder: SymbolDecoder实例
    """
    decoder.exit_symbol()

