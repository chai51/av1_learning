# AV1 学习解码器

一个用于学习 AV1 视频编解码技术的 Python 解码器实现。

## 目录

- [项目概述](#项目概述)
- [当前实现状态](#当前实现状态)
- [项目结构](#项目结构)
- [快速开始](#快速开始)
- [代码规范](#代码规范)
- [使用示例](#使用示例)
- [测试](#测试)
- [开发进度](#开发进度)
- [架构设计](#架构设计)
- [学习资源](#学习资源)
- [常见问题](#常见问题)
- [贡献](#贡献)
- [许可证](#许可证)

## 项目概述

本项目是一个用于学习AV1视频编解码技术的解码器实现。目标是按照AV1规范文档逐步实现一个功能完整的解码器，以便深入理解AV1的编解码原理。

**特点**：
- 严格按照AV1规范文档实现，变量名、函数名、常量名与文档保持一致
- 使用Python实现，注重可读性和学习价值，性能不是首要目标
- 使用`uv`进行Python环境管理
- 使用`pytest`进行测试

## 当前实现状态

### ✅ 已完成模块

#### 1. 比特流读取层
- ✅ `BitReader` - 比特流读取器（支持按位读取、字节对齐）
- ✅ 描述符实现（`f(n)`, `uvlc()`, `leb128()`, `ns(n)`, `su(n)`, `le(n)`, `S()`, `NS()`, `inverse_recenter`）

#### 2. OBU解析
- ✅ `OBUParser` - OBU解析器
- ✅ `parse_obu_header()` - OBU头解析
- ✅ `parse_obu()` - 完整OBU解析（支持序列头、帧头、Tile组等）

#### 3. 序列和帧管理
- ✅ `SequenceHeader` - 序列头结构及解析
- ✅ `FrameHeader` - 帧头结构及解析
- ✅ 颜色配置、时序信息、量化参数等关键字段解析

#### 4. 熵解码器
- ✅ `SymbolDecoder` - 符号解码器（ANS解码）
- ✅ 布尔解码和符号解码
- ✅ CDF管理和更新

#### 5. Tile和块解码
- ✅ `TileGroupParser` - Tile组解析器
- ✅ 块划分解析（`decode_partition`）
- ✅ 块解码框架（`decode_block`）
- ✅ 模式信息解析（`ModeInfo`, `intra_frame_mode_info`, `inter_frame_mode_info`）

#### 6. 残差解码
- ✅ 变换树解析（`transform_tree`）
- ✅ 系数解码（`coeffs`, `coeff_base`, `coeff_br`, `dc_sign`, `sign_bit`）
- ✅ 变换大小解析（`read_block_tx_size`, `read_var_tx_size`）
- ✅ 系数上下文计算（`get_coeff_context_eob`, `get_coeff_context`, `get_coeff_br_context`）
- ✅ 扫描顺序（`get_scan`, `get_default_scan`, `get_mrow_scan`, `get_mcol_scan`）
- ✅ 变换类型计算（`transform_type`, `compute_tx_type`, `get_tx_set`）
- ✅ Palette模式基础框架（`palette_mode_info`, `palette_tokens`）

#### 7. 运动向量
- ✅ `MotionVector` - 运动向量结构
- ✅ `find_mv_stack()` - MV候选列表构建
- ✅ `assign_mv()` - MV分配
- ✅ `mv_component()` - MV分量解码

#### 8. 预测和重构
- ✅ 预测框架（`predict_intra`, `predict_inter`, `predict_chroma_from_luma`, `predict_palette`）
- ✅ 重构框架（`dequantize`, `inverse_transform`, `reconstruct`, `reconstruct_block`）

#### 9. 工具模块
- ✅ MV工具（`clamp_mv_row`, `clamp_mv_col`, `clip3`）
- ✅ 边界检查（`is_inside`, `is_inside_filter_region`）
- ✅ 上下文管理（`clear_above_context`, `clear_left_context`, `get_segment_id`）

#### 10. 容器格式
- ✅ `IVFParser` - IVF容器格式解析器

#### 11. 预测算法详细实现
- ✅ 完整的角度插值（方向预测）- `_directional_intra_prediction_process`
- ✅ 完整的平滑预测和Paeth预测算法 - `_smooth_intra_prediction_process`, `_basic_intra_prediction_process`
- ✅ 子像素运动补偿和插值滤波器 - `block_inter_prediction`（支持8抽头和双线性插值）
- ✅ Warped Motion完整逻辑 - `_block_warp`, `_warp_estimation`, `_setup_shear`

#### 12. 反变换详细实现
- ✅ 完整的DCT/ADST反变换算法 - `inverse_dct_process`, `inverse_adst_process`
- ✅ 1D和2D变换的正确实现 - `inverse_2d_transform_process`
- ✅ Walsh-Hadamard变换和Identity变换

#### 13. 反量化详细实现
- ✅ 完整的量化查找表 - `dc_q`, `ac_q`（使用`Dc_Qlookup`, `Ac_Qlookup`）
- ✅ DC和AC分量的不同处理 - `get_dc_quant`, `get_ac_quant`
- ✅ 量化矩阵支持

#### 14. 上下文管理详细实现
- ✅ 参考缓冲区的正确维护
- ✅ 全局上下文数组的完整更新逻辑 - `AboveLevelContext`, `LeftLevelContext`, `AboveDcContext`, `LeftDcContext`
- ✅ 上下文清除函数 - `clear_above_context`, `clear_left_context`
- ✅ `SegmentIds`维护

#### 15. 高级特性
- ✅ 完整的OBMC（Overlapped Block Motion Compensation）- `_overlapped_motion_compensation`
- ✅ Filter Intra模式 - `_recursive_intra_prediction_process`
- ✅ Loop Filter - `loop_filter_process`（支持Normal和Wide滤波器）
- ✅ CDEF（Constrained Directional Enhancement Filter）- `cdef_process`
- ✅ Loop Restoration - `loop_restore_block`（支持Wiener和Self-Guided Restoration）

### 🚧 待完善模块

- **性能优化**：
  - 代码优化和性能提升
  - 内存使用优化

- **测试覆盖**：
  - 增加更多测试用例
  - 端到端解码测试

- **文档完善**：
  - 补充详细的代码注释
  - 添加更多使用示例

## 项目结构

```
av1_learning/
├── docs/                  # AV1规范文档
├── src/                   # 源代码
│   ├── bitstream/         # 比特流读取
│   │   ├── bit_reader.py  # BitReader实现
│   │   └── descriptors.py # 描述符实现
│   ├── obu/               # OBU解析
│   │   └── obu_parser.py  # OBU解析器
│   ├── sequence/          # 序列头解析
│   │   └── sequence_header.py
│   ├── frame/              # 帧头解析
│   │   └── frame_header.py
│   ├── entropy/            # 熵解码
│   │   └── symbol_decoder.py
│   ├── tile/               # Tile组解析
│   │   └── tile_group.py
│   ├── mode/               # 模式信息解析
│   │   ├── mode_info.py
│   │   └── motion_vector.py
│   ├── residual/           # 残差解码
│   │   ├── residual.py
│   │   ├── transform_utils.py
│   │   ├── coeff_context.py
│   │   ├── tx_size.py
│   │   └── palette.py
│   ├── reconstruction/     # 预测和重构
│   │   ├── predict.py
│   │   └── reconstruct.py
│   ├── utils/              # 工具函数
│   │   ├── mv_utils.py
│   │   ├── boundary_utils.py
│   │   └── context_utils.py
│   ├── container/          # 容器格式解析
│   │   └── ivf_parser.py
│   └── constants.py        # 常量定义
├── tests/                 # 测试文件
│   ├── test_bit_reader.py
│   ├── test_descriptors.py
│   ├── test_obu_parser.py
│   ├── test_sequence_header.py
│   ├── test_frame_header.py
│   ├── test_symbol_decoder.py
│   └── ...
├── examples/              # 示例代码
│   ├── simple_decoder.py  # AV1解码验证脚本
│   └── parse_obu.py
├── res/                   # 测试资源文件
│   └── av1-film_grain.ivf
├── pyproject.toml         # 项目配置
├── README.md              # 本文件
├── ARCHITECTURE.md        # 架构设计文档
└── ROADMAP.md             # 开发路线图
```

## 快速开始

### 环境要求

- Python 3.8+
- `uv` - Python包管理工具

### 安装 uv

如果还没有安装 `uv`，可以通过以下方式安装：

```bash
# Linux/macOS
curl -LsSf https://astral.sh/uv/install.sh | sh

# 或使用 pip
pip install uv
```

### 安装依赖

```bash
# 使用uv安装依赖
uv sync
```

### 运行示例

```bash
# 解析并验证AV1 IVF文件
uv run python examples/simple_decoder.py res/av1-film_grain.ivf

# 或使用默认文件
uv run python examples/simple_decoder.py
```

### 运行测试

```bash
# 运行所有测试
uv run pytest

# 运行特定测试文件
uv run pytest tests/test_bit_reader.py

# 显示详细输出
uv run pytest -v
```

## 代码规范

本项目严格按照AV1规范文档实现，遵循以下原则：

1. **变量名、函数名、常量名与文档保持一致**
   - 例如：`obu_type`, `frame_type`, `decode_partition`等
   - 常量定义在`src/constants.py`中

2. **仅实现文档明确描述的异常处理**
   - 不做额外的异常判断，避免增加代码复杂度
   - 专注于学习和理解解码本身

3. **模块化设计**
   - 每个模块对应规范文档的相应章节
   - 便于理解和维护

## 使用示例

### 解析IVF文件

```python
import sys
from pathlib import Path

# 添加src目录到Python路径
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from container.ivf_parser import IVFParser

# 读取IVF文件
with open('res/av1-film_grain.ivf', 'rb') as f:
    ivf_data = f.read()

# 解析IVF容器
ivf_parser = IVFParser()
ivf_parser.parse_file(ivf_data)

# 获取帧数据
frames = ivf_parser.get_frames()
for frame in frames:
    # frame.data 包含AV1比特流
    pass
```

### 解析OBU

```python
import sys
from pathlib import Path

# 添加src目录到Python路径
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from obu.obu_parser import OBUParser
from bitstream.bit_reader import BitReader

# 创建OBU解析器
parser = OBUParser()

# 解析OBU（包含完整的header和payload）
obu_data = b'...'  # OBU字节数据
obu_header = parser.parse_obu(obu_data, len(obu_data))

if obu_header:
    print(f"OBU类型: {obu_header.obu_type}")
    print(f"OBU大小: {obu_header.obu_size}")
    
    # 根据类型访问解析结果
    if obu_header.sequence_header:
        seq_header = obu_header.sequence_header
```

### 解码比特流

```python
import sys
from pathlib import Path

# 添加src目录到Python路径
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from bitstream.bit_reader import BitReader
from bitstream.descriptors import read_f, read_leb128

# 创建BitReader
reader = BitReader(data)

# 使用描述符读取数据
value = read_f(reader, 8)  # 读取8位
size = read_leb128(reader)  # 读取LEB128编码的值
```

## 测试

项目包含完整的单元测试，覆盖以下模块：
- 比特流读取器
- 描述符（f(n), uvlc, leb128等）
- OBU解析器
- 序列头解析
- 帧头解析
- 符号解码器

运行测试：
```bash
uv run pytest
```

## 开发进度

详细开发进度请参考 [ROADMAP.md](ROADMAP.md)

## 架构设计

详细架构设计请参考 [ARCHITECTURE.md](ARCHITECTURE.md)

## 学习资源

- [AV1规范文档](https://aomediacodec.github.io/av1-spec/) - 官方AV1规范
- [libaom](https://aomedia.googlesource.com/aom/) - AV1参考实现（C语言）
- [AV1 Bitstream & Decoding Process Specification](https://aomediacodec.github.io/av1-spec/av1-spec.pdf) - AV1规范PDF版本

## 常见问题

### Q: 为什么使用 Python 而不是 C/C++？

A: 本项目的主要目标是学习和理解 AV1 解码原理，Python 代码更易读、更易理解，便于学习。性能不是首要考虑因素。

### Q: 解码器能完整解码视频吗？

A: 目前实现还在进行中，部分模块（如预测算法、反变换等）还需要完善。详见"待完善模块"部分。

### Q: 如何获取测试用的 AV1 文件？

A: 可以使用 `ffmpeg` 或其他工具将视频编码为 AV1 格式，或从网上下载 AV1 测试文件。

## 贡献

本项目主要用于学习和教育目的。欢迎提交问题和建议！

### 如何贡献

1. Fork 本项目
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启 Pull Request

### 报告问题

如果发现 bug 或有改进建议，请在项目的 Issues 页面中提交。

## 许可证

本项目仅用于学习和教育目的。代码实现遵循 AV1 规范文档的许可证要求。

**注意**：本项目是学习性质的实现，不应用于生产环境。AV1 规范文档和相关参考实现（如 libaom）的许可证要求请参考各自的官方文档。
