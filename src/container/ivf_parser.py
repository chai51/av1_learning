"""
IVF容器格式解析器
IVF (Indeo Video File) 是一种简单的容器格式，用于存储AV1比特流
"""

from typing import Optional, Tuple, List
from bitstream.bit_reader import BitReader


class IVFHeader:
    """
    IVF文件头结构
    """

    def __init__(self):
        self.signature = b''  # "DKIF"
        self.version = 0  # 版本号（通常为0）
        self.header_size = 0  # 头大小（字节）
        self.fourcc = b''  # 编解码器标识（"AV01"）
        self.width = 0  # 视频宽度
        self.height = 0  # 视频高度
        self.timebase_den = 0  # 时间基准分母
        self.timebase_num = 0  # 时间基准分子
        self.num_frames = 0  # 帧数量


class IVFFrame:
    """
    IVF帧结构
    """

    def __init__(self):
        self.frame_size = 0  # 帧大小（字节）
        self.timestamp = 0  # 时间戳
        self.data = b''  # 帧数据（AV1比特流）


def parse_ivf_header(data: bytes) -> Optional[Tuple[IVFHeader, int]]:
    """
    解析IVF文件头

    Args:
        data: IVF文件数据

    Returns:
        (IVFHeader, header_size) 或 None（如果解析失败）
    """
    if len(data) < 32:
        return None

    header = IVFHeader()

    # 读取文件头（32字节）
    # 0-3: signature "DKIF"
    header.signature = data[0:4]
    if header.signature != b'DKIF':
        return None

    # 4-5: version (little-endian)
    header.version = int.from_bytes(data[4:6], 'little')

    # 6-7: header_size (little-endian)
    header.header_size = int.from_bytes(data[6:8], 'little')

    # 8-11: fourcc "AV01"
    header.fourcc = data[8:12]
    if header.fourcc != b'AV01':
        return None

    # 12-13: width (little-endian)
    header.width = int.from_bytes(data[12:14], 'little')

    # 14-15: height (little-endian)
    header.height = int.from_bytes(data[14:16], 'little')

    # 16-19: timebase_den (little-endian)
    header.timebase_den = int.from_bytes(data[16:20], 'little')

    # 20-23: timebase_num (little-endian)
    header.timebase_num = int.from_bytes(data[20:24], 'little')

    # 24-27: num_frames (little-endian)
    header.num_frames = int.from_bytes(data[24:28], 'little')

    # 28-31: unused/reserved
    # (实际header_size可能更大，但标准是32字节)

    return (header, header.header_size if header.header_size > 0 else 32)


class IVFParser:
    """
    IVF容器解析器
    解析IVF文件并提取AV1比特流
    """

    def __init__(self):
        self.header: Optional[IVFHeader] = None
        self.frames: List[IVFFrame] = []

    def parse_file(self, data: bytes) -> bool:
        """
        解析整个IVF文件

        Args:
            data: IVF文件数据（完整文件）

        Returns:
            是否解析成功
        """
        # 解析文件头
        result = parse_ivf_header(data)
        if result is None:
            return False

        self.header, header_size = result

        # 解析帧数据
        pos = header_size
        frame_count = 0

        while pos < len(data):
            # 检查是否有足够的字节读取帧头（12字节）
            if pos + 12 > len(data):
                break

            frame = IVFFrame()

            # 帧头：12字节
            # 0-3: frame_size (little-endian)
            frame.frame_size = int.from_bytes(data[pos:pos+4], 'little')
            pos += 4

            # 4-11: timestamp (little-endian, 64位)
            frame.timestamp = int.from_bytes(data[pos:pos+8], 'little')
            pos += 8

            # 检查帧数据大小
            if pos + frame.frame_size > len(data):
                # 数据不完整
                break

            # 读取帧数据
            frame.data = data[pos:pos+frame.frame_size]
            pos += frame.frame_size

            self.frames.append(frame)
            frame_count += 1

        return True

    def get_frames(self) -> List[IVFFrame]:
        """
        获取所有解析的帧

        Returns:
            帧列表
        """
        return self.frames

    def get_header(self) -> Optional[IVFHeader]:
        """
        获取文件头

        Returns:
            IVFHeader或None
        """
        return self.header
