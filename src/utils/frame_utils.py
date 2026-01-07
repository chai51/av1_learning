
from typing import Any
from obu.decoder import AV1Decoder
from constants import SEG_LVL_ALT_Q
from utils.math_utils import Clip3


def get_relative_dist(av1: AV1Decoder, a: int, b: int) -> int:
    """
    获取相对距离
    规范文档 5.9.3 Get relative distance function

    Args:
        a: 第一个order hint
        b: 第二个order hint
    Returns:
        相对距离
    """
    seq_header = av1.seq_header

    if not seq_header.enable_order_hint:
        return 0
    diff = a - b
    m = 1 << (seq_header.OrderHintBits - 1)
    diff = (diff & (m - 1)) - (diff & m)
    return diff


def inverse_recenter(r: int, v: int) -> int:
    """
    规范文档 5.9.29 Inverse recenter function

    Args:
        r: 读取的uvlc值
        v: 解码的值
    """
    if v > 2 * r:
        return v
    elif v & 1:
        return r - ((v + 1) >> 1)
    else:
        return r + (v >> 1)


def get_qindex(av1: AV1Decoder, ignoreDeltaQ: int, segmentId: int) -> int:
    """
    获取量化索引
    规范文档 7.12.2 Dequantization functions - get_qindex()

    Args:
        ignoreDeltaQ: 是否忽略DeltaQ
        segmentId: Segment ID

    Returns:
        量化索引
    """
    from utils.tile_utils import seg_feature_active_idx
    frame_header = av1.frame_header
    tile_group = av1.tile_group

    if seg_feature_active_idx(av1, segmentId, SEG_LVL_ALT_Q):
        # 1.
        data = frame_header.FeatureData[segmentId][SEG_LVL_ALT_Q]
        # 2.
        qindex = frame_header.base_q_idx + data
        # 3.
        if ignoreDeltaQ == 0 and frame_header.delta_q_present == 1:
            qindex = tile_group.CurrentQIndex + data
        # 4.
        return Clip3(0, 255, qindex)
    elif ignoreDeltaQ == 0 and frame_header.delta_q_present == 1:
        return tile_group.CurrentQIndex
    else:
        return frame_header.base_q_idx
