<p align="center">
  <img src="../figs/logo_full.png" alt="OpenMedVisionX" width="900">
</p>

<p align="center">
  <a href="../README.md">English</a>
</p>

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyQt5](https://img.shields.io/badge/PyQt5-5.15+-41CD52.svg)](https://pypi.org/project/PyQt5/)
[![pydicom](https://img.shields.io/badge/pydicom-2.4+-f37626.svg)](https://pydicom.github.io/)
[![SciPy](https://img.shields.io/badge/SciPy-1.11+-8CAAE6.svg)](https://scipy.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](../LICENSE)

# OpenMedVisionX：开放交互式医学计算机视觉学习与探索平台

正式英文名称为 **OpenMedVisionX: An Open Interactive Platform for Medical Computer Vision Learning and Exploration**。

OpenMedVisionX 是一个本地部署、GUI 驱动的医学图像、CT 重建与外部视觉模型
推理教学平台。它强调展示图像语义、空间几何、预处理、算法中间过程和模型输出，
而不是只给学生一个不透明的预测按钮。

> [!WARNING]
> 本软件仅用于教学和研究，不是医疗器械，未经临床验证，不得用于诊断、治疗、
> 分诊或任何患者相关决策。

本仓库仅提供源代码，明确不包含医学影像、数据集、模型权重、API Key、可执行
文件或打包构建产物。

## 项目状态

项目目前处于 alpha 阶段，正在从早期 CT 教学查看器重构为分层平台。首要目标是
正确的图像领域模型、安全的本地加载、传统重建、能力驱动的 PyQt 界面以及可测试
的扩展协议。模型运行时、研究模型适配器和云端 LLM Provider 始终是可选能力，
不会成为基础查看器的强制依赖。

1.0 之前的公共接口在下文中明确列出，以便示例和插件逐步稳定。
兼容性变化记录在 [CHANGELOG.md](../CHANGELOG.md)。

## 设计原则

- 本地优先，输入和导出目录均由用户明确选择。
- 明确区分二维栅格图、二维页/帧序列和具有物理空间含义的医学体数据。
- 空间体数据内部统一为 RAS+；普通二维图不伪造 spacing、affine、Z 轴、毫米或
  HU 语义。
- 正确处理 DICOM rescale、方向、spacing 和三视图几何。
- IO 保留原始 dtype 和动态范围，显示映射与模型预处理分别记录。
- 加载、重建、推理和网络请求均使用可取消后台任务。
- 外部模型和 LLM 服务仅作为用户配置的可选扩展。
- 不自动扫描数据集、下载模型、上传图像或覆盖原图。

## 格式策略

| 格式 | 数据对象 | 安装方式与语义 |
| --- | --- | --- |
| DICOM 文件夹或经过校验的 DICOM ZIP | `ImageVolume` | 基础安装；ZIP 条目和解压数据受安全上限约束，CT 强度统一应用 slope/intercept。 |
| NIfTI `.nii` / `.nii.gz` | `ImageVolume` | 安装 `nifti` 可选组；affine 转换为内部 RAS+。 |
| PNG | `RasterImage2D` | 基础安装；覆盖常见灰度、RGB、RGBA 和调色板图，并保留源 dtype。 |
| JPEG `.jpg` / `.jpeg` | `RasterImage2D` | 基础安装；统一校正 EXIF Orientation，并显示有损压缩提示。 |
| TIFF `.tif` / `.tiff` | `RasterImage2D` 或 `ImageSequence2D` | 基础后端保留高位深页面，并提供可取消、按需 page、缩略图与平面 tile 读取；解码器与 WSI 后端仍可替换。 |
| MHA/MHD、NRRD、BMP、WebP、WSI | 插件定义 | 不属于基础必选依赖，加载器插件必须声明并校验能力。 |

多页 TIFF 默认是页序列。除非空间元数据通过校验，否则不会因为页数多就获得冠状
面、矢状面、体渲染或三维测量能力。

对于高分辨率平面图像，`RasterTileSource` 只返回请求的页面结果、缩略图或规范
坐标区域，并放入同时受条目数和解码字节数约束的线程安全 LRU 缓存。打开源文件
时同时校验扩展名和签名，每一页均受尺寸、像素数、帧数和解码字节上限约束；读取
调用接受与后台任务一致的协作式取消检查。EXIF 方向、调色板/颜色转换与 alpha
语义会记录在返回对象的变换和运行时追踪中，源路径和像素载荷不会进入运行时元数据。
默认 `ImageService` 会把预计解码载荷超过 64 MiB 的平面栅格图交给该后端，显示最长
边不超过 2048 像素、带可逆源坐标变换的缩略图，而不会在会话中保留完整页面。

该基础 API 表示普通平面页面，不表示病理金字塔。Pillow 编解码器生成区域时可能
仍需解码底层 strip，因此这不是内存映射或 WSI 流式读取承诺。真正的多分辨率 WSI
读取器必须实现独立的 `WsiPyramidTileSource` 插件边界，并声明自己的层级几何、
调度器与资源上限；普通 TIFF 永远不会被自动提升为 WSI。

## 架构

```text
src/dicom_viewer/
├── domain/       ImageData、几何、显示映射、坐标变换和结果模型
├── io/           格式探测以及安全的 DICOM、NIfTI、栅格加载器
├── algorithms/   Radon、DFR、BP、FBP、SART 和图像质量指标
├── inference/    清单、预处理、运行时、适配器和后处理
├── llm/          Provider 协议和教学助手适配器
├── runtime/      任务、取消、配置、凭据和错误处理
├── services/     供 GUI 调用的应用服务
└── ui/           PyQt 页面、状态呈现和可视化
```

UI 只依赖服务和公共协议，不能直接导入具体解码器、模型运行时或厂商 API 适配器。
工具是否可用由当前数据对象的能力决定，而不是由文件扩展名硬编码。

### 稳定扩展点

- `ImageLoader.can_load/probe/load`
- `ImageData`、`RasterImage2D`、`ImageSequence2D`、`ImageVolume`
- `TransformRecord.forward/inverse`
- `ReconstructionAlgorithm.reconstruct`
- `ModelPlugin.describe/validate/capabilities/load/predict/visualize`
- 类型化 `InferenceResult`
- `LLMProvider.chat/stream/capabilities`
- `BackgroundTask.cancel/progress/result`

插件应依赖这些协议，不能直接访问某个加载器、Qt 控件、模型实现或 Provider
内部适配器。

## 安装

需要 Python 3.10 或更高版本；支持 Python 3.10–3.12。

### Conda

```bash
conda env create -f environment.yml
conda activate openmedvisionx
openmedvisionx
```

项目展示名称为 `OpenMedVisionX`；Conda 环境、发行包和命令统一使用
`openmedvisionx`。

### pip

```bash
python -m venv .venv
# Windows PowerShell
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e .
openmedvisionx
```

默认安装即 `base` 基础查看器。只安装需要的扩展：

```bash
python -m pip install -e ".[nifti]"
python -m pip install -e ".[onnx]"
python -m pip install -e ".[pytorch-plugin]"
python -m pip install -e ".[llm]"
python -m pip install -e ".[dev,nifti,onnx]"
```

基础安装不会安装 PyTorch、ONNX Runtime、云端 SDK、CUDA、模型代码或模型权重。

## 图像语义与测量

`RasterImage2D` 只接受 `H x W` 灰度数组或 `H x W x C` 彩色数组，并记录
位深、颜色空间、通道顺序和 alpha 处理。其坐标原点位于左上角，默认只允许像素
距离和像素面积。只有用户输入可信的 pixel spacing 后才启用物理测量，并在界面
标记为“用户提供”。DPI 永远不会被当作医学 spacing。

`ImageSequence2D` 表示缺乏物理体几何的页或帧序列。`ImageVolume` 表示具有
可信空间语义的数据，并携带 affine、spacing、origin、direction、模态和强度语义。

显示映射不会修改源数组。EXIF 校正、resize、crop 和 letterbox 会生成可逆的
`TransformRecord`，使 mask、检测框、关键点、热图和概率图能够返回源像素坐标。
离散标签图回映时使用最近邻插值。

## 本地 mask、标注与实验文件

配对必须由用户显式选择，并且只作用于当前可见平面。mask 必须是与当前平面
`H x W` 完全一致的无损单页灰度 PNG/TIFF；非零值形成叠加区域，程序不会隐式
resize。标注 JSON 上限为 4 MiB，拒绝未知字段和外部路径，采用左上角原点的像素坐标：

```json
{
  "schema_version": 1,
  "coordinate_system": "pixel_xy_top_left",
  "image_size": [512, 512],
  "boxes": [{"x1": 20, "y1": 30, "x2": 120, "y2": 160, "label": "example"}],
  "points": [{"x": 64, "y": 80, "label": "landmark"}]
}
```

`image_size` 必须匹配当前平面；框需满足图像范围内 `x1 < x2`、`y1 < y2`，点也
必须位于图像内。渲染 PNG 与实验 JSON 采用排他式新建，既不覆盖已有文件，也不覆盖
源图。实验记录只包含不含像素的图像摘要、参数和数值指标；原始像素与源元数据会被拒绝。

## CT 重建教学

传统算法区域覆盖：

- Radon 变换和正弦图生成；
- 直接傅里叶重建及其插值选择；
- 无滤波反投影和滤波反投影；
- SART 迭代与收敛过程；
- 平行束扫描 180 度与 360 度投影的冗余关系；
- 在统一强度范围计算 MSE、PSNR、SSIM、差值图、误差热图和 ROI 指标；
- 频域滤波、反投影和迭代状态的中间过程。

测试与课程在运行时生成 phantom，不会为教程提交 DICOM 或其他医学样例图像。

## 外部模型插件

平台不提供模型实现或权重。用户可以指向本地 ONNX/TorchScript 模型，或选择包含
`manifest.yaml` 和 Python adapter 的研究插件。

每个清单必须描述：

- 模型名称、版本、家族、任务、子任务和许可证；
- 运行时、Python/Conda 环境、adapter 入口和外部权重引用；
- 模态以及 2D/2.5D/3D/4D 维度；
- 输入尺寸、layout、颜色空间、通道、alpha、dtype、数值范围、spacing、
  归一化和 resize/crop/letterbox 策略；
- 输出张量语义、坐标系、标签、后处理和不确定性；
- prompts、中间特征、attention、embedding、多尺度输出、向量场、采样轨迹和
  文本等可选能力。

`1.0` 版 schema 的顶层必填字段为 `schema_version`、`name`、
`version`、`family`、`source`、`description`、`tasks`、
`subtasks`、`license`、`runtime`、`entrypoint`、`weights`、
`inputs`、`outputs` 和 `capabilities`。可参考纯源码的
[示例清单](../src/dicom_viewer/inference/examples/manifest.yaml)，其中模型路径
仅为用户本地引用，不随项目分发。

任务名称使用受控枚举，例如 `classification`、`segmentation`、
`detection`、`registration`、`reconstruction`、`restoration`、
`generation`、`retrieval`、`vqa`、`report_generation`、
`wsi_mil`、`tracking`、`anomaly_detection`。重名家族必须同时记录
来源和任务，例如自监督 DINO encoder、DINO detector 与 Grounding DINO 不是
同一种能力。

> [!CAUTION]
> Python 插件能够执行任意代码。插件运行在独立进程和用户选择的环境中，但进程
> 隔离不等于安全沙箱。加载前必须检查并信任其代码。

应用不会自动下载模型，也不会把权重复制进仓库。插件必须展示代码、模型和权重
许可证。本项目的 MIT License 不覆盖第三方插件或权重。

全切片病理需要未来的流式金字塔加载器与 tile 调度器。在此之前，WSI 插件只能
接收用户提供的预切片或特征。GUI 负责导入、推理与可视化，不提供训练、验证或
断点续训。

## LLM 教学助手

Provider 协议面向 OpenAI、Anthropic Claude、Moonshot/Kimi、智谱 GLM、
DeepSeek 和用户配置的 OpenAI-compatible endpoint，其他厂商可通过插件扩展。

Provider 设置包括服务地址、用户填写的模型 ID、能力、超时和凭据引用，代码不
硬编码“最新模型”。OpenAI 适配器使用 Responses API。

保存的 Provider 配置只能包含凭据引用，不能包含 Key 本身：

```toml
provider_id = "openai"
endpoint = "https://provider.example.invalid/v1"
model_id = "user-selected-model-id"
credential_ref = "env:OPENAI_API_KEY"
supports_vision = false
timeout = 30
```

凭据引用格式为 `env:VARIABLE_NAME` 或
`keyring:service/account-name`。服务地址和模型 ID 必须由用户按照 Provider
当前文档填写；只有确认该模型支持视觉输入后，才能声明视觉能力。

API Key 必须存入操作系统凭据库或环境变量。配置文件只记录凭据引用。源码、
日志、异常、导出对话、截图和缺陷报告中均不得出现 Key。

云端图像传输默认关闭，需要用户按 Provider 授权，并可随时撤销。开启时 GUI 会
持续显示警告。只能发送用户明确选择且预览过的渲染切片；不得发送原始
DICOM/NIfTI、完整序列或 DICOM 元数据。发送前会移除元数据，但烧录在像素中的
文字仍可能包含身份信息。

助手回答显示 Provider、模型 ID、时间和教学用途免责声明，不能被呈现为医学
建议或诊断。

## 仓库与数据安全

提交前运行：

```bash
python scripts/check_repository.py
```

仓库策略检查会拒绝：

- 医学体数据和 DICOM，包括能够通过签名发现的错误扩展名文件；
- 压缩包、可执行文件、编译产物和构建目录；
- 模型权重与序列化模型；
- 疑似 API Key、访问令牌、私钥和密码；
- 超过仓库策略上限的大文件。

如果敏感数据或 Key 曾经进入提交历史，只删除工作树文件并不够。必须撤销泄漏
凭据，在公开仓库前从 Git 历史删除对应对象，并遵循
[SECURITY.md](../SECURITY.md)。

应用只读取用户明确选择的文件，不递归扫描设备上的医学数据。派生图像只写入
用户选择的本地目录，并且默认不覆盖原图。

## 详细文档

- [架构与依赖规则](ARCHITECTURE.md)
- [模型插件开发](PLUGIN_DEVELOPMENT.md)
- [LLM Provider 与云端图像安全](LLM_SECURITY.md)
- [教学课程：从像素到多模态 AI](TEACHING_CURRICULUM.md)

## 开发与测试

```bash
python -m pip install -e ".[dev,nifti,onnx]"
python scripts/check_repository.py
python -m pytest
ruff check src tests scripts
```

本地开发中 PyTorch 是可选依赖。未安装 PyTorch 时，极小 TorchScript 集成测试会
明确显示为 `skipped`；如需实际运行，请安装插件扩展后单独执行：

```bash
python -m pip install -e ".[pytorch-plugin]"
python -m pytest -q \
  tests/test_inference_execution.py::test_tiny_torchscript_runtime_when_pytorch_is_available
```

测试在临时目录运行时生成极小 DICOM、NIfTI、栅格图、ONNX/TorchScript 模型和
mock Provider 数据。请勿向仓库添加测试影像或权重。

架构与 PR 要求见 [CONTRIBUTING.md](../CONTRIBUTING.md)，私密安全报告见
[SECURITY.md](../SECURITY.md)，社区行为要求见
[CODE_OF_CONDUCT.md](../CODE_OF_CONDUCT.md)。

## 路线图

1. 仓库清理、许可证、双语文档与提交保护。
2. 统一图像领域模型以及安全的 DICOM、NIfTI、PNG、JPEG、TIFF 加载器。
3. 能力驱动的二维/三维 PyQt 界面、后台任务和传统 CT 教学可视化。
4. ONNX、TorchScript 与独立进程 Python 模型插件。
5. 覆盖分类、分割、检测、配准、重建、增强、生成、多模态、WSI、时序/4D 和
   异常检测的结果类型与可视化协议。
6. 多 Provider LLM 助手、授权控制和安全凭据。
7. 自动化测试、插件开发指南与首个纯源码版本。

## 许可证

平台源代码采用 [MIT License](../LICENSE)。用户导入的数据、插件、模型和权重保留
各自条款，其合法使用责任由用户承担。
