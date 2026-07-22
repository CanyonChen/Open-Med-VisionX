# 使用自定义本地模型

[English](CUSTOM_MODELS.md) · [文档首页](INDEX.zh-CN.md) · [用户指南](USER_GUIDE.zh-CN.md#模型)

本指南面向已经拥有本地模型、希望通过“**模型 → 外部清单**”运行模型的用户。
OpenMedVisionX 不负责训练模型、搜索模型目录、下载权重、猜测缺失的预处理，也不能让
任意 checkpoint 自动变为可运行模型。

> [!CAUTION]
> 只加载您信任且有权使用的模型。Python adapter 会执行第三方代码。独立进程只能提供
> 故障隔离，并不是安全沙箱。

## 1. 先确认桌面能否运行该模型

当前外部模型页面只能提供**一个活动二维影像**。模型必须能在完成清单声明的预处理后
接受这个输入。

| 要求 | 当前桌面支持情况 |
| --- | --- |
| 单个二维栅格图/影像平面 | 当模态、间距、dtype、通道与预处理契约匹配时支持。 |
| ONNX 模型 | 安装 `.[onnx]` 后支持。默认 ONNX Runtime 安装不保证 GPU 执行。 |
| TorchScript 模型 | 安装兼容的 PyTorch 运行时后支持。 |
| Python adapter | 只有在明确同意信任代码后才支持；依赖与环境仍由用户负责。 |
| 多张影像/多模态 | 通用桌面流程不支持。 |
| 完整 3D/4D 体数据 | 通用桌面流程不支持。 |
| 提示词、文本、k-space/mask、sinogram、时间序列或 WSI 输入 | 通用桌面流程不支持。 |
| `.pth`、`.pt` 或 `.ckpt` 训练 checkpoint | 除非它是有效且经过检查的 TorchScript，否则不直接支持；普通 pickle checkpoint 需要原始架构与可信代码。 |

协议能描述的任务多于当前桌面能够收集的输入。因此，一份有效清单仍可能被可见的能力
门禁拒绝。拒绝可以避免得到看似合理、实则科学上无效的结果。

## 2. 准备运行时与本地文件

先激活项目环境：

```bash
conda activate openmedvisionx
```

只安装模型需要的运行时：

```bash
python -m pip install -e ".[onnx]"     # ONNX + ONNX Runtime
```

TorchScript 需要与本机匹配的 PyTorch。请按照[随包模型指南](MODEL_BUNDLES.zh-CN.md#安装-pytorch)
中的 CPU/GPU 说明选择，不要假设一个 CUDA wheel 适合所有电脑。

建议把模型保存在 OpenMedVisionX 仓库之外，例如：

```text
my-local-model/
├── manifest.yaml
├── model.onnx                 # 或本地 TorchScript 产物
├── README.md                  # 来源、预期用途、限制与安装说明
└── LICENSE-or-NOTICE.txt
```

Python adapter 的适配器文件与依赖声明应放在同一个可信包目录。OpenMedVisionX 不会自动
安装这些依赖。

远程 URL、自动下载和缺失的本地文件会被拒绝。不要在清单中写入 API key、患者路径或
患者标识符。

## 3. 使用清单描述模型

清单是采用 `"1.0"` schema 的 UTF-8 YAML 文件。请从维护的
[示例清单](../src/workbench/inference/examples/manifest.yaml)开始，但要将每个示例值
替换为模型的真实事实。示例引用的 `models/example_classifier.onnx` 有意未提供，因此
示例本身不能运行。

清单至少需要回答以下问题：

| 部分 | 必须说明什么 |
| --- | --- |
| 身份 | 稳定的名称、版本、family、描述、来源与作者。 |
| 任务 | 主要任务和精确子任务，例如分类与二分类。 |
| 许可 | 分别说明代码、模型定义与权重条款。 |
| 运行时 | `onnx`、`torchscript` 或 `python-adapter`，以及设备策略。 |
| 本地文件 | 入口与权重路径；能够获得时写入预期 SHA-256 和字节数。 |
| 输入 | 名称、影像语义、模态、2D 维度、张量形状/布局、颜色/通道顺序、dtype/范围、间距要求及有序预处理。 |
| 输出 | 张量名、语义、shape/dtype、标签或坐标、激活/后处理、阈值与不确定性含义。 |
| 能力 | 只声明模型真正实现的可选功能。 |

下面的精简示例展示单影像 ONNX 分类器的结构。它不是通用模板，也不能直接运行：

```yaml
schema_version: "1.0"
name: my-local-classifier
version: "1.0.0"
family: resnet
source:
  name: my-reviewed-source
  organization: my-organization
  repository: null
  publication: null
  model_id: classifier-v1
description: Binary classifier for one documented 2-D image domain.
tasks: [classification]
subtasks: [binary-classification]
license:
  code: MIT
  model: Apache-2.0
  weights: user-supplied
  notice: Review all terms before loading.
  urls: {}
runtime:
  kind: onnx
  device: auto
  provider: null
  options: {}
entrypoint:
  path: model.onnx
weights:
  - name: model
    path: model.onnx
    format: onnx
    sha256: REPLACE_WITH_SHA256
    size_bytes: REPLACE_WITH_BYTES
    required: true
    license: user-supplied
inputs:
  - name: image
    semantic: image
    modalities: [generic-image]
    dimensionality: 2d
    shape: [1, 3, 224, 224]
    spacing: {required: false, values: null, unit: pixel, tolerance: null}
    preprocessing:
      layout: nchw
      color_space: rgb
      channel_order: [R, G, B]
      alpha_handling: drop
      dtype: float32
      value_range: [0, 255]
      normalization:
        scale: 0.00392156862745098
        offset: 0.0
        mean: [0.485, 0.456, 0.406]
        std: [0.229, 0.224, 0.225]
      spatial:
        - operation: letterbox
          size: [224, 224]
          interpolation: bilinear
          anchor: center
          pad_value: [0, 0, 0]
          allow_upscale: true
      orientation: apply-exif
    description: Canonically oriented RGB image.
outputs:
  - name: scores
    semantic: class-scores
    dtype: float32
    shape: [1, 2]
    coordinate_system: not-applicable
    labels: {"0": negative, "1": positive}
    postprocessing:
      activation: softmax
      threshold: null
      nms_iou_threshold: null
      interpolation: null
      discrete_labels: false
      parameters: {}
    uncertainty: {kind: none, description: Not calibrated, calibrated: false, units: null}
    description: Scores in manifest label order.
capabilities:
  prompts: false
  intermediate_features: false
  attention: false
  embeddings: false
  multiscale_outputs: false
  vector_fields: false
  sampling_trajectory: false
  multimodal_text: false
  uncertainty: false
authors: [Local user]
references: []
extensions: {}
```

除非这些设置对您的模型确实成立，否则不要照抄示例的归一化、标签顺序、数值范围、
resize 或通道顺序。错误但 shape 兼容的预处理可能产生非常逼真的错误结果。

## 4. 检查信任、文件与许可

加载前确认以下所有事项：

- 模型来自清单中声明的来源；
- 本地文件大小与 SHA-256 和来源提供的值一致；
- 代码、模型、权重与数据许可允许预期用途；
- 任务、人群、模态、维度、输入范围、通道顺序、间距与归一化符合模型训练/评价记录；
- 每个输出标签与坐标系都有定义；
- 您清楚分数是概率、logit、校准值还是其他量；
- 没有路径指向网络共享或不可信可执行文件。

对于 Python adapter，在接受信任对话框前阅读其源码与依赖文件。独立进程无法阻止恶意
代码访问其有权读取的凭据或本地文件。

## 5. 加载并运行模型

1. 在“**影像**”中打开一份获准使用的影像。
2. 单击要使用的精确平面或图像，确认“**活动视图**”提示显示正确平面与切片。
3. 打开“**模型 → 外部清单**”。
4. 选择“**1. 选择清单……**”，打开本地 YAML 文件。
5. 阅读显示的身份、来源、任务、输入/输出契约、运行时、许可和引用文件。解决所有校验
   错误后再继续。
6. 选择“**2. 加载模型**”。如果运行时是 Python adapter，只有完成上述信任检查后才
   接受。
7. 阅读能力门禁。它必须说明活动输入满足已声明的影像数量、语义、维度、模态、间距与
   预处理要求。
8. 如果超大源影像当前使用有界预览，只有确认这些预览像素对实验在科学上有效时才批准
   预览 override。
9. 选择“**3. 运行推理**”。
10. 等待完成，或使用“**停止**”请求取消。

**成功标准：**状态提示运行完成，可视化对应一个已声明输出，并且模型名称/版本/运行时
与预处理来源始终和结果一起可见。

## 6. 安全解释结果

请结合契约检查可视化，绝不能只把图片当作完整结果。

- 分类：确认标签顺序、激活、阈值以及分数是否经过校准。
- 分割：确认源平面映射、类别 ID、插值以及 mask 是离散还是概率值。
- 检测/关键点：确认坐标已正确逆映射 resize、crop、letterbox 与 EXIF orientation。
- 重建/复原影像：确认单位、源几何、数值范围以及是否评价测量一致性。
- 热图/attention：把它视为模型派生信号，不能自动视为解剖解释。

记录输入来源与许可、清单/模型哈希、运行时/设备、预处理、输出映射、警告与限制。成功
运行只证明技术兼容，不能证明泛化能力或临床有效性。

## 7. 常见阻止及其含义

| 信息或现象 | 应如何处理 |
| --- | --- |
| 清单有效，但“**运行推理**”被禁用 | 当前 GUI 无法提供声明的输入。使用单影像二维模型或外部任务专用流程；不要为了通过门禁而篡改契约。 |
| 找不到引用的模型文件 | 修正本地路径，或从可信来源恢复文件。OpenMedVisionX 不会下载它。 |
| ONNX 运行时不可用 | 激活 `openmedvisionx`，安装 `.[onnx]` 并重启。GPU 执行需要另行兼容的 provider/运行时。 |
| TorchScript 运行时不可用 | 安装与本机兼容的 PyTorch 构建。 |
| `.pth` 或 `.ckpt` 被拒绝 | 获取可信的 TorchScript/ONNX 导出，或配合原始架构使用经过检查的 Python adapter。重命名文件不等于转换。 |
| Python adapter 要求信任 | 停止并检查代码、来源、哈希、环境与许可。任何一项不清楚都应拒绝。 |
| 模态、间距、通道或范围不匹配 | 使用正确输入。不要为了运行而编造元数据或改变归一化。 |
| 出现大图预览警告 | 活动像素可能是有界预览而不是完整源图。除非该精确输入适合任务，否则取消。 |

更多现象见[故障排查 · 模型](TROUBLESHOOTING.zh-CN.md#模型)。

## 8. 相关指南

- [数据与格式](DATA_FORMATS.zh-CN.md)——正确准备活动影像。
- [用户指南 · 模型](USER_GUIDE.zh-CN.md#模型)——了解完整模型工作区。
- [随包模型指南](MODEL_BUNDLES.zh-CN.md)——使用三条经过审查的固定工作流。
- [学习课程 · 本地推理](TEACHING_CURRICULUM.zh-CN.md#6-可复现的本地模型推理)——把一次运行变成可复现实验。
- [术语表](GLOSSARY.zh-CN.md)——查询清单、来源、校准与几何术语。

返回[文档首页](INDEX.zh-CN.md)。
