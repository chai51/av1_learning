"""
输出过程模块
实现规范文档7.18节"Output process"至7.18.3.5"Add noise synthesis process"中描述的所有过程函数
"""

from typing import List, Tuple
from constants import NONE, Gaussian_Sequence, MATRIX_COEFFICIENTS, PLANE_MAX
from obu.decoder import AV1Decoder
from utils.math_utils import Array
from utils.math_utils import Clip1, Clip3, Round2, bits_signed


class FilmGrainSynthesisProcess:
    def __init__(self, av1: AV1Decoder):
        seq_header = av1.seq_header
        height = seq_header.max_frame_height_minus_1 + 1
        width = seq_header.max_frame_width_minus_1 + 1
        subsampling_x = seq_header.color_config.subsampling_x
        subsampling_y = seq_header.color_config.subsampling_y

        self.RandomRegister = 0
        self.GrainCenter = 0
        self.GrainMin = 0
        self.GrainMax = 0
        self.LumaGrain: List[List[int]] = Array(None, (73, 82))
        self.CbGrain: List[List[int]] = Array(None, (73, 82))
        self.CrGrain: List[List[int]] = Array(None, (73, 82))
        self.ScalingLut: List[List[int]] = Array(None, (PLANE_MAX, 256))
        self.ScalingShift = 0

    def film_grain_synthesis(self, av1: AV1Decoder, w: int, h: int, subX: int, subY: int):
        """
        Film Grain合成过程
        规范文档 7.18.3 Film grain synthesis process

        The inputs to this process are:
        - variables w and h specifying the width and height of the frame,
        - variables subX and subY specifying the subsampling parameters of the frame.

        The process modifies the arrays OutY, OutU, OutV to add film grain noise as follows:

        Args:
            w: 帧宽度
            h: 帧高度
            subX: X方向子采样
            subY: Y方向子采样
        """
        seq_header = av1.seq_header
        frame_header = av1.frame_header
        BitDepth = seq_header.color_config.BitDepth
        film_grain_params = frame_header.film_grain_params

        # 1.
        self.RandomRegister = film_grain_params.grain_seed

        # 2.
        self.GrainCenter = 128 << (BitDepth - 8)

        # 3.
        self.GrainMin = -self.GrainCenter

        # 4.
        self.GrainMax = (256 << (BitDepth - 8)) - 1 - self.GrainCenter

        # 5.
        self.generate_grain(av1)

        # 6.
        self.scaling_lookup_initialization(av1)

        # 7.
        self.add_noise_synthesis(av1, w, h, subX, subY)

    def get_random_number(self, bits: int) -> int:
        """
        随机数生成过程
        规范文档 7.18.3.2 Random number process

        The input to this process is a variable bits specifying the number of random bits to return.
        The output of this process is a pseudo-random number based on the state in RandomRegister.

        Args:
            RandomRegister: 随机数寄存器状态
            bits: 要返回的随机位数

        Returns:
            Tuple[result, new_RandomRegister]: 随机数和新的寄存器状态
        """
        r = self.RandomRegister
        bit = ((r >> 0) ^ (r >> 1) ^ (r >> 3) ^ (r >> 12)) & 1
        r = (r >> 1) | (bit << 15)
        result = (r >> (16 - bits)) & ((1 << bits) - 1)

        self.RandomRegister = r
        return result

    def generate_grain(self, av1: AV1Decoder) -> None:
        """
        生成Grain过程
        规范文档 7.18.3.3 Generate grain process

        This process generates noise via an auto-regressive filter.

        Args:
            RandomRegister: 随机数寄存器状态
            GrainMin: 颗粒最小值
            GrainMax: 颗粒最大值
        """
        seq_header = av1.seq_header
        frame_header = av1.frame_header
        mono_chrome = seq_header.color_config.mono_chrome
        subsampling_x = seq_header.color_config.subsampling_x
        subsampling_y = seq_header.color_config.subsampling_y
        BitDepth = seq_header.color_config.BitDepth
        film_grain_params = frame_header.film_grain_params

        shift = 12 - BitDepth + film_grain_params.grain_scale_shift
        for y in range(73):
            for x in range(82):
                if film_grain_params.num_y_points > 0:
                    g = Gaussian_Sequence[self.get_random_number(11)]
                else:
                    g = 0
                self.LumaGrain[y][x] = Round2(g, shift)

        shift = (film_grain_params.ar_coeff_shift_minus_6 + 6)
        for y in range(3, 73):
            for x in range(3, 82 - 3):
                s = 0
                pos = 0
                for deltaRow in range(-film_grain_params.ar_coeff_lag, 1):
                    for deltaCol in range(-film_grain_params.ar_coeff_lag, film_grain_params.ar_coeff_lag + 1):
                        if deltaRow == 0 and deltaCol == 0:
                            break
                        c = film_grain_params.ar_coeffs_y_plus_128[pos] - 128
                        s += self.LumaGrain[y + deltaRow][x + deltaCol] * c
                        pos += 1
                self.LumaGrain[y][x] = Clip3(
                    self.GrainMin, self.GrainMax, self.LumaGrain[y][x] + Round2(s, shift))

        if mono_chrome == 0:
            pass

        chromaW = 44 if subsampling_x else 82
        chromaH = 38 if subsampling_y else 73
        shift = 12 - BitDepth + film_grain_params.grain_scale_shift
        self.RandomRegister = film_grain_params.grain_seed ^ 0xb524
        for y in range(chromaH):
            for x in range(chromaW):
                if film_grain_params.num_cb_points > 0 or film_grain_params.chroma_scaling_from_luma:
                    g = Gaussian_Sequence[self.get_random_number(11)]
                else:
                    g = 0
                self.CbGrain[y][x] = Round2(g, shift)
        self.RandomRegister = film_grain_params.grain_seed ^ 0x49d8
        for y in range(chromaH):
            for x in range(chromaW):
                if film_grain_params.num_cr_points > 0 or film_grain_params.chroma_scaling_from_luma:
                    g = Gaussian_Sequence[self.get_random_number(11)]
                else:
                    g = 0
                self.CrGrain[y][x] = Round2(g, shift)

        shift = film_grain_params.ar_coeff_shift_minus_6 + 6
        for y in range(3, chromaH):
            for x in range(3, chromaW - 3):
                s0 = 0
                s1 = 0
                pos = 0
                for deltaRow in range(-film_grain_params.ar_coeff_lag, 1):
                    for deltaCol in range(-film_grain_params.ar_coeff_lag, film_grain_params.ar_coeff_lag + 1):
                        c0 = film_grain_params.ar_coeffs_cb_plus_128[pos] - 128
                        c1 = film_grain_params.ar_coeffs_cr_plus_128[pos] - 128

                        if deltaRow == 0 and deltaCol == 0:
                            if film_grain_params.num_y_points > 0:
                                luma = 0
                                lumaX = ((x - 3) << subsampling_x) + 3
                                lumaY = ((y - 3) << subsampling_y) + 3
                                for i in range(subsampling_y + 1):
                                    for j in range(subsampling_x + 1):
                                        luma += self.LumaGrain[lumaY +
                                                               i][lumaX + j]
                                luma = Round2(
                                    luma, subsampling_x + subsampling_y)
                                s0 += luma * c0
                                s1 += luma * c1
                            break

                        s0 += self.CbGrain[y + deltaRow][x + deltaCol] * c0
                        s1 += self.CrGrain[y + deltaRow][x + deltaCol] * c1
                        pos += 1
                self.CbGrain[y][x] = Clip3(
                    self.GrainMin, self.GrainMax, self.CbGrain[y][x] + Round2(s0, shift))
                self.CrGrain[y][x] = Clip3(
                    self.GrainMin, self.GrainMax, self.CrGrain[y][x] + Round2(s1, shift))

    def scaling_lookup_initialization(self, av1: AV1Decoder):
        """
        Scaling lookup initialization process
        """
        seq_header = av1.seq_header
        frame_header = av1.frame_header
        NumPlanes = seq_header.color_config.NumPlanes
        film_grain_params = frame_header.film_grain_params

        def get_x(plane: int, i: int) -> int:
            if plane == 0 or film_grain_params.chroma_scaling_from_luma:
                return film_grain_params.point_y_value[i]
            elif plane == 1:
                return film_grain_params.point_cb_value[i]
            else:
                return film_grain_params.point_cr_value[i]

        def get_y(plane: int, i: int) -> int:
            if plane == 0 or film_grain_params.chroma_scaling_from_luma:
                return film_grain_params.point_y_scaling[i]
            elif plane == 1:
                return film_grain_params.point_cb_scaling[i]
            else:
                return film_grain_params.point_cr_scaling[i]

        for plane in range(NumPlanes):
            if plane == 0 or film_grain_params.chroma_scaling_from_luma:
                numPoints = film_grain_params.num_y_points
            elif plane == 1:
                numPoints = film_grain_params.num_cb_points
            else:
                numPoints = film_grain_params.num_cr_points
            if numPoints == 0:
                for x in range(256):
                    self.ScalingLut[plane][x] = 0
            else:
                for x in range(get_x(plane, 0)):
                    self.ScalingLut[plane][x] = get_y(plane, 0)
                for i in range(numPoints - 1):
                    deltaY = get_y(plane, i + 1) - get_y(plane, i)
                    deltaX = get_x(plane, i + 1) - get_x(plane, i)
                    delta = deltaY * ((65536 + (deltaX >> 1)) // deltaX)
                    for x in range(deltaX):
                        v = get_y(plane, i) + ((x * delta + 32768) >> 16)
                        self.ScalingLut[plane][get_x(plane, i) + x] = v
                for x in range(get_x(plane, numPoints - 1), 256):
                    self.ScalingLut[plane][x] = get_y(plane, numPoints - 1)

    def add_noise_synthesis(self, av1: AV1Decoder, w: int, h: int, subX: int, subY: int):
        """
        添加噪声合成过程
        规范文档 7.18.3.5 Add noise synthesis process

        Args:
            w: 输出帧宽度
            h: 输出帧高度
            subX: X方向子采样
            subY: Y方向子采样
        """
        seq_header = av1.seq_header
        frame_header = av1.frame_header
        BitDepth = seq_header.color_config.BitDepth
        NumPlanes = seq_header.color_config.NumPlanes
        film_grain_params = frame_header.film_grain_params

        noiseStripe = Array(None, (64, NumPlanes, 64, (w * 2)))
        lumaNum = 0
        for y in range(0, (h + 1) // 2, 16):
            self.RandomRegister = film_grain_params.grain_seed
            self.RandomRegister ^= ((lumaNum * 37 + 178) & 255) << 8
            self.RandomRegister ^= ((lumaNum * 173 + 105) & 255)
            for x in range(0, (w + 1) // 2, 16):
                rand = self.get_random_number(8)
                offsetX = rand >> 4
                offsetY = rand & 15
                for plane in range(NumPlanes):
                    planeSubX = subX if plane > 0 else 0
                    planeSubY = subY if plane > 0 else 0
                    planeOffsetX = 6 + offsetX if planeSubX else 9 + offsetX * 2
                    planeOffsetY = 6 + offsetY if planeSubY else 9 + offsetY * 2
                    for i in range(34 >> planeSubY):
                        for j in range(34 >> planeSubX):
                            if plane == 0:
                                g = self.LumaGrain[planeOffsetY +
                                                   i][planeOffsetX + j]
                            elif plane == 1:
                                g = self.CbGrain[planeOffsetY +
                                                 i][planeOffsetX + j]
                            else:
                                g = self.CrGrain[planeOffsetY +
                                                 i][planeOffsetX + j]

                            if planeSubX == 0:
                                if j < 2 and film_grain_params.overlap_flag and x > 0:
                                    old = noiseStripe[lumaNum][plane][i][x * 2 + j]
                                    if j == 0:
                                        g = old * 27 + g * 17
                                    else:
                                        g = old * 17 + g * 27
                                    g = Clip3(self.GrainMin,
                                              self.GrainMax, Round2(g, 5))
                                noiseStripe[lumaNum][plane][i][x * 2 + j] = g
                            else:
                                if j == 0 and film_grain_params.overlap_flag and x > 0:
                                    old = noiseStripe[lumaNum][plane][i][x + j]
                                    g = old * 23 + g * 22
                                    g = Clip3(self.GrainMin,
                                              self.GrainMax, Round2(g, 5))
                                noiseStripe[lumaNum][plane][i][x + j] = g
            lumaNum += 1

        noiseImage = Array(None, (NumPlanes, (h * 2 + subY)
                           >> subY, (w * 2 + subX) >> subX), 0)
        for plane in range(NumPlanes):
            planeSubX = subX if plane > 0 else 0
            planeSubY = subY if plane > 0 else 0
            for y in range((h + planeSubY) >> planeSubY):
                lumaNum = y >> (5 - planeSubY)
                i = y - (lumaNum << (5 - planeSubY))
                for x in range((w + planeSubX) >> planeSubX):
                    g = noiseStripe[lumaNum][plane][i][x]
                    if planeSubY == 0:
                        if i < 2 and lumaNum > 0 and film_grain_params.overlap_flag:
                            old = noiseStripe[lumaNum - 1][plane][i + 32][x]
                            if i == 0:
                                g = old * 27 + g * 17
                            else:
                                g = old * 17 + g * 27
                            g = Clip3(self.GrainMin,
                                      self.GrainMax, Round2(g, 5))
                    else:
                        if i < 1 and lumaNum > 0 and film_grain_params.overlap_flag:
                            old = noiseStripe[lumaNum - 1][plane][i + 16][x]
                            g = old * 23 + g * 22
                            g = Clip3(self.GrainMin,
                                      self.GrainMax, Round2(g, 5))
                    noiseImage[plane][y][x] = g

        if film_grain_params.clip_to_restricted_range:
            minValue = 16 << (BitDepth - 8)
            maxLuma = 235 << (BitDepth - 8)
            if seq_header.color_config.matrix_coefficients == MATRIX_COEFFICIENTS.MC_IDENTITY:
                maxChroma = maxLuma
            else:
                maxChroma = 240 << (BitDepth - 8)
        else:
            minValue = 0
            maxLuma = (256 << (BitDepth - 8)) - 1
            maxChroma = maxLuma

        def scale_lut(plane: int, index: int) -> int:
            """
            缩放查找表
            规范文档 7.18.3.5 Add noise synthesis process

            Args:
                plane: 平面索引
                index: 索引值

            Returns:
                缩放值
            """
            BitDepth = seq_header.color_config.BitDepth

            shift = BitDepth - 8
            x = index >> shift
            rem = index - (x << shift)
            if BitDepth == 8 or x == 255:
                return self.ScalingLut[plane][x]
            else:
                start = self.ScalingLut[plane][x]
                end = self.ScalingLut[plane][x + 1]
                return start + Round2((end - start) * rem, shift)

        self.ScalingShift = film_grain_params.grain_scaling_minus_8 + 8
        for y in range(0, (h + subY) >> subY):
            for x in range(0, (w + subX) >> subX):
                lumaX = x << subX
                lumaY = y << subY
                lumaNextX = min(lumaX + 1, w - 1)
                if subX:
                    averageLuma = Round2(
                        av1.OutY[lumaY][lumaX] + av1.OutY[lumaY][lumaNextX], 1)
                else:
                    averageLuma = av1.OutY[lumaY][lumaX]
                if film_grain_params.num_cb_points > 0 or film_grain_params.chroma_scaling_from_luma:
                    orig = av1.OutU[y][x]
                    if film_grain_params.chroma_scaling_from_luma:
                        merged = averageLuma
                    else:
                        combined = (averageLuma *
                                    (film_grain_params.cb_luma_mult - 128) +
                                    orig * (film_grain_params.cb_mult - 128))
                        merged = Clip1(
                            (combined >> 6) + ((film_grain_params.cb_offset - 256) << (BitDepth - 8)), BitDepth)
                    noise = noiseImage[1][y][x]
                    noise = Round2(scale_lut(1, merged) *
                                   noise, self.ScalingShift)
                    av1.OutU[y][x] = Clip3(minValue, maxChroma, orig + noise)

                if film_grain_params.num_cr_points > 0 or film_grain_params.chroma_scaling_from_luma:
                    orig = av1.OutV[y][x]
                    if film_grain_params.chroma_scaling_from_luma:
                        merged = averageLuma
                    else:
                        combined = (averageLuma *
                                    (film_grain_params.cr_luma_mult - 128) +
                                    orig * (film_grain_params.cr_mult - 128))
                        merged = Clip1(
                            (combined >> 6) + ((film_grain_params.cr_offset - 256) << (BitDepth - 8)), BitDepth)
                    noise = noiseImage[2][y][x]
                    noise = Round2(scale_lut(2, merged) *
                                   noise, self.ScalingShift)
                    av1.OutV[y][x] = Clip3(minValue, maxChroma, orig + noise)

        for y in range(0, h):
            for x in range(0, w):
                orig = av1.OutY[y][x]
                noise = noiseImage[0][y][x]
                noise = Round2(scale_lut(0, orig) * noise, self.ScalingShift)
                if film_grain_params.num_y_points > 0:
                    av1.OutY[y][x] = Clip3(minValue, maxLuma, orig + noise)


def intermediate_output_preparation(av1: AV1Decoder, filmGrainSynthesis: FilmGrainSynthesisProcess) -> Tuple[int, int, int, int]:
    """
    中间输出准备过程
    规范文档 7.18.2 Intermediate output preparation process

    Returns:
        Tuple[w, h, subX, subY]: 输出格式描述变量
    """
    seq_header = av1.seq_header
    frame_header = av1.frame_header
    tile_group = av1.tile_group
    ref_frame_store = av1.ref_frame_store
    subsampling_x = seq_header.color_config.subsampling_x
    subsampling_y = seq_header.color_config.subsampling_y
    BitDepth = seq_header.color_config.BitDepth
    frame_to_show_map_idx = frame_header.frame_to_show_map_idx
    OutY = av1.OutY
    OutU = av1.OutU
    OutV = av1.OutV
    if frame_header.show_existing_frame == 1:
        w = ref_frame_store.RefUpscaledWidth[frame_to_show_map_idx]
        h = ref_frame_store.RefFrameHeight[frame_to_show_map_idx]
        subX = ref_frame_store.RefSubsamplingX[frame_to_show_map_idx]
        subY = ref_frame_store.RefSubsamplingY[frame_to_show_map_idx]
        for x in range(w):
            for y in range(h):
                OutY[y][x] = ref_frame_store.FrameStore[frame_to_show_map_idx][0][y][x]
                # The bit depth for each sample is BitDepth.
                OutY[y][x] = bits_signed(OutY[y][x], BitDepth)

        for x in range((w + subX) >> subX):
            for y in range((h + subY) >> subY):
                OutU[y][x] = ref_frame_store.FrameStore[frame_to_show_map_idx][1][y][x]
                # The bit depth for each sample is BitDepth.
                OutU[y][x] = bits_signed(OutU[y][x], BitDepth)

        for x in range((w + subX) >> subX):
            for y in range((h + subY) >> subY):
                OutV[y][x] = ref_frame_store.FrameStore[frame_to_show_map_idx][2][y][x]
                # The bit depth for each sample is BitDepth.
                OutV[y][x] = bits_signed(OutV[y][x], BitDepth)

        seq_header.color_config.BitDepth = ref_frame_store.RefBitDepth[frame_to_show_map_idx]

        return w, h, subX, subY
    else:
        w = frame_header.UpscaledWidth

        h = frame_header.FrameHeight

        subX = subsampling_x

        subY = subsampling_y

        OutY = Array(OutY, (h, w))
        for x in range(w):
            for y in range(h):
                OutY[y][x] = av1.LrFrame[0][y][x]
                OutY[y][x] = bits_signed(OutY[y][x], BitDepth)

        OutU = Array(OutU, ((h + subY) >> subY, (w + subX) >> subX))
        for x in range((w + subX) >> subX):
            for y in range((h + subY) >> subY):
                OutU[y][x] = av1.LrFrame[1][y][x]
                # The bit depth for each sample is BitDepth.
                OutU[y][x] = bits_signed(OutU[y][x], BitDepth)

        OutV = Array(OutV, ((h + subY) >> subY, (w + subX) >> subX))
        for x in range((w + subX) >> subX):
            for y in range((h + subY) >> subY):
                OutV[y][x] = av1.LrFrame[2][y][x]
                # The bit depth for each sample is BitDepth.
                OutV[y][x] = bits_signed(OutV[y][x], BitDepth)

        return w, h, subX, subY
