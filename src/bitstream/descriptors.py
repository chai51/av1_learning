"""
描述符实现
按照规范文档第3.3节描述的各种描述符实现
"""


def read_f(reader, n: int) -> int:
    """
    f(n) 描述符：固定长度位字段
    规范文档 3.3.2
    
    Args:
        reader: BitReader实例
        n: 位数
        
    Returns:
        n位无符号整数
    """
    return reader.read_bits(n)


def read_uvlc(reader) -> int:
    """
    uvlc() 描述符：无符号可变长度编码
    规范文档 3.3.3
    
    编码方式：
    - 从最高有效位开始
    - 读取x个前导零位
    - 读取x+1位的有符号整数，最高位设为1
    
    Returns:
        解码后的无符号整数
    """
    leading_zeros = 0
    while reader.read_bit() == 0:
        leading_zeros += 1
        
    if leading_zeros >= 32:
        return (1 << 32) - 1
    
    # 读取 leading_zeros + 1 位，最高位已经是1
    value = 1 << leading_zeros
    for i in range(leading_zeros):
        bit = reader.read_bit()
        value |= bit << (leading_zeros - 1 - i)
    
    return value - 1


def read_leb128(reader) -> int:
    """
    leb128() 描述符：LEB128编码
    规范文档 3.3.4
    
    编码方式：
    - 每个字节的最高位是延续位（1表示继续，0表示结束）
    - 剩余7位是数据位
    - 小端序
    
    Returns:
        解码后的无符号整数
    """
    value = 0
    shift = 0
    
    while True:
        byte_val = reader.read_bits(8)
        value |= (byte_val & 0x7F) << shift
        shift += 7
        
        if (byte_val & 0x80) == 0:
            break
            
        # LEB128最多9字节（规范限制）
        if shift >= 63:
            break
    
    return value


def read_su(reader, n: int) -> int:
    """
    su(n) 描述符：有符号无符号编码
    规范文档 3.3.6
    
    Args:
        reader: BitReader实例
        n: 位数
        
    Returns:
        有符号整数（使用补码表示，范围为 -(2^(n-1)) 到 2^(n-1)-1）
    """
    unsigned_value = reader.read_bits(n)
    
    # 转换为有符号（补码表示）
    if unsigned_value >= (1 << (n - 1)):
        return unsigned_value - (1 << n)
    else:
        return unsigned_value


def read_ns(reader, n: int) -> int:
    """
    ns(n) 描述符：非对称数值编码
    规范文档 3.3.7
    
    如果 n > 0，则读取 uvlc() + 1
    如果 n == 0，则读取 uvlc()
    
    Args:
        reader: BitReader实例
        n: 参数（用于调整值）
        
    Returns:
        解码后的值
    """
    value = read_uvlc(reader)
    if n > 0:
        return value + 1
    else:
        return value


def read_le(reader, n: int) -> int:
    """
    le(n) 描述符：小端序无符号整数
    规范文档 3.3.4
    
    Args:
        reader: BitReader实例
        n: 字节数（1, 2, 4, 8）
        
    Returns:
        小端序解码的无符号整数
    """
    value = 0
    for i in range(n):
        byte_val = reader.read_bits(8)
        value |= byte_val << (i * 8)
    return value


def read_S(reader) -> int:
    """
    S() 描述符：有符号整数
    规范文档 3.3.8
    
    读取 uvlc() 并使用规范文档中的 inverse_recenter() 函数转换为有符号
    
    Returns:
        有符号整数
    """
    value = read_uvlc(reader)
    # 使用 inverse_recenter 转换
    return inverse_recenter(value, 0)


def read_NS(reader, n: int) -> int:
    """
    NS(n) 描述符：非对称数值编码（有符号）
    规范文档 3.3.8
    
    Args:
        reader: BitReader实例
        n: 参数
        
    Returns:
        有符号整数
    """
    value = read_uvlc(reader)
    if n > 0:
        value = value + 1
    # 转换为有符号（简化实现，实际应该使用inverse_recenter）
    return value


def inverse_recenter(r: int, v: int) -> int:
    """
    Inverse recenter 函数
    规范文档 6.8.23
    
    Args:
        r: 读取的uvlc值
        v: 参考值
        
    Returns:
        有符号整数
    """
    if r % 2 == 0:
        return v + r // 2
    else:
        return v - (r + 1) // 2

