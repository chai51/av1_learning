
from typing import Any, Optional, Tuple, Union


def integer_div(a: int, b: int) -> int:
    """
    向零截断结果的整数除法
    规范文档 4.2 Arithmetic operators

    For example:
        7 / 4 = 1
        -7 / -4 = 1
        -7 / 4 = -1
        7 / -4 = -1
    """
    if a >= 0 and b > 0:
        return a // b
    elif a < 0 and b < 0:
        return a // b
    elif a < 0:
        return -(-a // b)
    else:
        return -(a // -b)


def Clip1(x: int, BitDepth: int) -> int:
    """
    Clip1函数将值裁剪到BitDepth位深度范围
    规范文档 4.7 Mathematical functions
    """
    high = 1 << BitDepth
    return Clip3(0, high - 1, x)


def Clip3(low: int, high: int, value: int) -> int:
    """
    Clip3函数将值裁剪到[low, high]范围
    规范文档 4.7 Mathematical functions
    """
    if value < low:
        return low
    elif value > high:
        return high
    else:
        return value


def Round2(x: int, n: int) -> int:
    """
    Round2函数使用的是标准的数学幂运算和除法运算
    规范文档 4.7 Mathematical functions

    For example:
        Round2(0b1000, 3) = 0b1
        Round2(0b0100, 3) = 0b1
        Round2(0b1011, 3) = 0b1
        Round2(0b0111, 3) = 0b1
    """
    if n == 0:
        return x
    return (x + (1 << (n - 1))) >> n


def Round2Signed(x: int, n: int) -> int:
    """
    Round2Signed函数进行有符号舍入n位
    规范文档 4.7 Mathematical functions
    """
    if x >= 0:
        return Round2(x, n)
    return -Round2(-x, n)


def FloorLog2(x: int) -> int:
    """
    FloorLog2函数计算给定整数 x 的每个分量的以2为底的对数，并向下取整
    规范文档 4.7 Mathematical functions
    """
    # The input x will always be an integer, and will always be greater than or equal to 1.
    assert x >= 1

    s = 0
    while x != 0:
        x >>= 1
        s += 1
    return s - 1


def CeilLog2(x: int) -> int:
    """
    CeilLog2函数用于提取对0到x-1范围内的值进行编码所需的位数
    规范文档 4.7 Mathematical functions
    """
    # The input x will always be an integer, and will always be greater than or equal to 0.
    assert x >= 0

    if x < 2:
        return 0
    i = 1
    p = 2
    while p < x:
        i += 1
        p <<= 1
    return i


def bits_signed(x: int, r: int) -> int:
    """
    BitsSigned函数将x转换为r位精度有符号整数
    保留 r 位精度有符号整数
    """
    if r <= 0:
        return 0
    mask = (1 << r) - 1
    raw = abs(x) & mask
    return raw if x >= 0 else -raw


def __create_array(s: Tuple[int, ...], fill: Optional[int] = None) -> Any:
    if len(s) == 0:
        return fill
    if len(s) == 1:
        return [fill] * s[0]
    return [__create_array(s[1:], fill) for _ in range(s[0])]


def __array_shape(arr: Any) -> Tuple[int, ...]:
    """递归获取多维数组的形状"""
    if not isinstance(arr, list):
        return ()
    if len(arr) == 0:
        return (0,)
    return (len(arr),) + __array_shape(arr[0])


def Array(array: Any, shape: Union[Tuple[int, ...], int], fill: Optional[int] = None) -> Any:
    """
    创建或扩充多维数组到指定形状

    Args:
        array: 要扩充的数组，如果为None则新建数组
        shape: 目标形状，可以是tuple或单个整数（表示1维数组）
        fill: 填充值，默认为None。如果array为None创建新数组时fill为None则使用0；
              如果扩充数组时fill为None则保持None值

    Returns:
        创建或扩充后的数组，保持原有值不变。如果数组已符合目标shape，直接返回原数组（效率优化）
    """
    # 标准化shape为tuple
    if isinstance(shape, int):
        target_shape = (shape,)
    elif isinstance(shape, tuple):
        if len(shape) > 5:
            raise ValueError("shape最多支持5个维度")
        target_shape = shape
    else:
        raise ValueError("shape类型错误，必须是int或tuple")

    # 如果array为None，创建新数组
    if array is None:
        return __create_array(target_shape, fill)

    current_shape = __array_shape(array)
    if current_shape == target_shape:
        return array

    # 计算目标实际形状
    tmp_shape = []
    max_len = len(current_shape)
    for i in range(max_len):
        if i < len(target_shape):
            tmp_shape.append(max(current_shape[i], target_shape[i]))
        else:
            tmp_shape.append(current_shape[i])
    target_shape = tuple[int, ...](tmp_shape)

    current_len = current_shape[0]
    target_len = target_shape[0]

    # 保留原有的数据
    if current_len != target_len:
        new_array = [fill] * target_len
        idx = 0
        while idx < current_len:
            new_array[idx] = array[idx]
            idx += 1
        for i in range(idx, target_len):
            new_array[i] = __create_array(target_shape[1:], fill)
    else:
        new_array = array

    # 递归处理剩余维度
    if current_shape[1:] != target_shape[1:]:
        for i in range(target_len):
            new_array[i] = Array(new_array[i], target_shape[1:], fill)

    return new_array
