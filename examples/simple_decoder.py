"""
AV1解码器验证脚本
解析IVF文件并验证AV1比特流解码流程
"""

from copy import deepcopy
from io import SEEK_SET
import json
import sys
from pathlib import Path
from typing import List

# 添加src到路径
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from entropy.symbol_decoder import inverseCdf
from obu.decoder import AV1Decoder
from utils.math_utils import Round2
from utils.math_utils import Array


from constants import OBU_HEADER_TYPE
from container.ivf_parser import IVFParser

class Av1Dump:
    def __init__(self, path: str):
        self.dumpfile = open(path, 'r')
        self.indexPredFrame = 0
        self.indexResidual = 0
        self.indexWmmat = 0
        self.indexCdf = 0
        self.indexPred = 0
        self.indexSymbol = 0
        self.indexFilmGrainFrame = 0

        self.posResidual = 0
        self.posFilmGrainFrame = 0
        self.posWmmat = 0
        self.posSymbol = 0
        self.posCdf = 0
        self.posPred = 0
        self.posPredFrame = 0

        # 写yuv文件
        self.yuvfile = open(path + ".yuv", 'wb')
    
    def read(self, name: str):
        while True:
            line = self.dumpfile.readline()
            # 如果读到结尾
            if line == '':
                return None, None

            if name in line:
                try:
                    dump = line.split('=')
                    return dump[0], json.loads(dump[1])
                except Exception as e:
                    print(f'Error: {e}')
                    return None, None


    def on_pred_frame(self, plane: int, x: int, y: int, pred_frame: List[List[int]]):
        self.indexPredFrame += 1

        self.dumpfile.seek(self.posPredFrame, SEEK_SET)
        dump = self.read("PRED_FRAME,")
        self.posPredFrame = self.dumpfile.tell()

        if dump is None or dump[0] is None:
            return
        params = dump[0].split(',')
        plane2 = int(params[1])
        x2 = int(params[2])
        y2 = int(params[3])
        if plane != plane2 or x != x2 or y != y2:
            print(f'plane = {plane}, x = {x}, y = {y}, plane2 = {plane2}, x2 = {x2}, y2 = {y2}')
            return

        if dump[1] != pred_frame:
            zeroLine = [0] * len(pred_frame[0])
            for i in range(len(pred_frame)):
                if pred_frame[i] == dump[1][i]:
                    pass
                elif dump[1][i] == zeroLine:
                    # 如果aom缺省模式，则检查是否和上一行相同
                    if pred_frame[i] == pred_frame[i - 1]:
                        pass
                    else:
                        print(f'pred_frame[{i}] = {pred_frame[i]}, dump[{i}] = {dump[1][i]}')
                        return

                else:
                    # 如果aom缺省模式，则检查是否为零块
                    zeroBlock = [0] * 16
                    for pos in range(0, len(pred_frame[i]), 16):
                        if dump[1][i][pos:pos + 15] == zeroBlock:
                            if pred_frame[i][0:15] == pred_frame[i][pos:pos + 15]:
                                pass
                            else:
                                print(f'pred_frame[{i}][{0:15}] = {pred_frame[i][0:15]}, dump[{i}][{pos:pos + 15}] = {dump[1][i][pos:pos + 16]}')
                            return

    def on_residual_frame(self, plane: int, x: int, y: int, residual_frame: List[List[int]]):
        self.indexResidual += 1

        self.dumpfile.seek(self.posResidual, SEEK_SET)
        dump = self.read("RESIDUAL_FRAME,")
        self.posResidual = self.dumpfile.tell()

        if dump is None or dump[0] is None:
            return
        params = dump[0].split(',')
        plane2 = int(params[1])
        x2 = int(params[2])
        y2 = int(params[3])
        if plane != plane2 or x != x2 or y != y2:
            print(f'plane = {plane}, x = {x}, y = {y}, plane2 = {plane2}, x2 = {x2}, y2 = {y2}')
            return

        if dump[1] != residual_frame:
            zeroLine = [0] * len(residual_frame[0])
            for i in range(len(residual_frame)):
                if residual_frame[i] == dump[1][i]:
                    pass
                elif dump[1][i] == zeroLine:
                    if residual_frame[i] == residual_frame[i - 1]:
                        pass
                    else:
                        print(f'residual_frame[{i}] = {residual_frame[i]}, dump[{i}] = {dump[1][i]}')
                        return
                else:
                    zeroBlock = [0] * 16
                    for pos in range(0, len(residual_frame[i]), 16):
                        if dump[1][i][pos:pos + 15] == zeroBlock:
                            if residual_frame[i][0:15] == residual_frame[i][pos:pos + 15]:
                                pass
                            else:
                                print(f'residual_frame[{i}][{0:15}] = {residual_frame[i][0:15]}, dump[{i}][{pos:pos + 15}] = {dump[1][i][pos:pos + 16]}')
                                return

    def on_film_grain_frame(self, film_grain_frame: List[List[List[int]]]):
        for i in range(len(film_grain_frame)):

            self.dumpfile.seek(self.posFilmGrainFrame, SEEK_SET)
            dump = self.read("FILM_GRAIN_FRAME")
            self.posFilmGrainFrame = self.dumpfile.tell()

            if dump is None or dump[0] is None:
                return

            if dump[1] != film_grain_frame[i]:
                print(f'film_grain_frame[{i}] = {film_grain_frame[i]}, dump[{i}] = {dump[1]}')
                
                for i2 in range(len(dump[1])):
                    for j2 in range(len(dump[1][i2])):
                        if dump[1][i2][j2] != film_grain_frame[i][i2][j2]:
                            print(f'film_grain_frame[{i}][{j2}] = {film_grain_frame[i][j2]}, dump[{i2}][{j2}] = {dump[1][i2][j2]}')
                            return

        # 写yuv文件
        # 将film_grain_frame写入文件中，将每一行转换为字节流写入文件
        for row in film_grain_frame[0]:
            self.yuvfile.write(bytes(row))
        for row in film_grain_frame[1]:
            self.yuvfile.write(bytes(row))
        for row in film_grain_frame[2]:
            self.yuvfile.write(bytes(row))

    def on_wmmat(self, wmmat: List[int]):
        self.indexWmmat += 1

        self.dumpfile.seek(self.posWmmat, SEEK_SET)
        dump = self.read("WMMAT,")
        self.posWmmat = self.dumpfile.tell()

        if dump is None or dump[0] is None:
            return
        if dump[1] != wmmat:
            print(f'wmmat = {wmmat}, dump = {dump[1]}')

    def on_symbol(self, symbol: List[int]):
        self.indexSymbol += 1

        self.dumpfile.seek(self.posSymbol, SEEK_SET)
        dump = self.read("SYMBOL")
        self.posSymbol = self.dumpfile.tell()

        if dump is None or dump[0] is None:
            return
        if dump[1] != symbol:
            print(f'symbol = {symbol}, dump = {dump[1]}')

    def on_cdf(self, cdf: List[int]):
        self.indexCdf += 1

        self.dumpfile.seek(self.posCdf, SEEK_SET)
        dumpcdf = self.read("CDF")
        self.posCdf = self.dumpfile.tell()

        if dumpcdf is None or dumpcdf[0] is None:
            return

        cdf2 = deepcopy(cdf)
        if cdf[-2] == 1 << 15:
            for i in range(len(cdf2)):
                if i != len(cdf2) - 1:
                    cdf2[i] = (1 << 15) - cdf2[i]
        for i in range(len(cdf2)):
            if i == len(cdf2) - 1:
                if cdf2[i] != dumpcdf[1][i]:
                    return
            elif cdf2[i] != dumpcdf[1][i]:
                print(f'cdf[{i}] = {cdf2[i]}, dumpcdf[{i}] = {dumpcdf[i]}')
                return

    def on_pred(self, plane: int, x: int, y: int, pred: List[List[int]]):
        self.indexPred += 1

        self.dumpfile.seek(self.posPred, SEEK_SET)
        dumppred = self.read("PRED,")
        self.posPred = self.dumpfile.tell()

        if dumppred is None or dumppred[0] is None:
            return
        params = dumppred[0].split(',')
        plane2 = int(params[1])
        x2 = int(params[2])
        y2 = int(params[3])

        # if plane != plane2 or x != x2 or y != y2:
        #     print(f'plane = {plane}, x = {x}, y = {y}, plane2 = {plane2}, x2 = {x2}, y2 = {y2}')
        #     return

        if dumppred[1] != pred:
            print(f'pred = {pred}, dumppred = {dumppred[1]}')
            for i in range(len(pred)):
                for j in range(len(pred[i])):
                    if pred[i][j] != dumppred[1][i][j]:
                        print(f'pred[{i}][{j}] = {pred[i][j]}, dumppred[{i}][{j}] = {dumppred[1][i][j]}')
                        return
    

def decode_av1_file(ivf_file: str, verbose: bool = True):
    """
    解码AV1文件
    
    Args:
        ivf_file: IVF文件路径
        verbose: 是否输出详细信息
    """
    print(f"正在解析IVF文件: {ivf_file}")
    print("=" * 60)

    # 1. 读取IVF文件
    with open(ivf_file, 'rb') as f:
        ivf_data = f.read()
    
    if verbose:
        print(f"IVF文件大小: {len(ivf_data)} 字节\n")
    
    # 2. 解析IVF容器
    ivf_parser = IVFParser()
    if not ivf_parser.parse_file(ivf_data):
        print("错误: 无法解析IVF文件头")
        return
    
    header = ivf_parser.get_header()
    if header and verbose:
        print("IVF文件头信息:")
        print(f"  编解码器: {header.fourcc.decode()}")
        print(f"  分辨率: {header.width}x{header.height}")
        print(f"  时间基准: {header.timebase_num}/{header.timebase_den}")
        print(f"  帧数量: {header.num_frames}")
        print()
    
    frames = ivf_parser.get_frames()
    if verbose:
        print(f"解析到 {len(frames)} 帧\n")
    
    arr = Array(None, (5, 2, 4), 0)
    arr = Array(arr, (5, 4))

    
    # 3. 解析每帧的AV1比特流
    av1Decoder = AV1Decoder()
    av1Dump = Av1Dump(ivf_file + '.dump')
    seq_header = None

    av1Decoder.on_cdf = av1Dump.on_cdf
    av1Decoder.on_symbol = av1Dump.on_symbol
    av1Decoder.on_wmmat = av1Dump.on_wmmat
    av1Decoder.on_pred = av1Dump.on_pred
    av1Decoder.on_pred_frame = av1Dump.on_pred_frame
    av1Decoder.on_residual_frame = av1Dump.on_residual_frame
    av1Decoder.on_film_grain_frame = av1Dump.on_film_grain_frame
    
    for frame_idx, frame in enumerate(frames):
        if verbose:
            print(f"{'=' * 60}")
            print(f"帧 #{frame_idx + 1} (时间戳: {frame.timestamp})")
            print(f"帧大小: {frame.frame_size} 字节")
            print("-" * 60)
        
        # 解析帧中的OBU
        frame_data = frame.data
        
        frame_header = None
        
        try:
            obu_headers = av1Decoder.decode(frame_data)
          
        except Exception as e:
            if verbose:
                print(f"  错误: 解析frame_unit时出错: {e}")
                import traceback
                traceback.print_exc()
            print(f"  本帧解析失败\n")
    
    print("=" * 60)
    print("解码完成!")
    print(f"总计: {len(frames)} 帧")


if __name__ == '__main__':
    decode_av1_file('res/av1-film_grain.ivf')
    decode_av1_file('res/av1-show_existing_frame.ivf')
    decode_av1_file('res/av1-svc-L2T2.ivf')
    decode_av1_file('res/blackwhite_yuv444p-frame.av1.ivf')
    decode_av1_file('res/test-25fps.av1.ivf')

    

