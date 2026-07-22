# OpenMedVisionX 快速开始

[English](QUICKSTART.md) · [文档首页](INDEX.zh-CN.md) · [项目概览](README.zh-CN.md)

本指南将带您从全新检出开始，完成第一次有效实验。首个工作流不需要影像、模型运行时、
网络连接或患者数据。

> [!WARNING]
> 只使用您有权处理的数据。OpenMedVisionX 仅用于学习与研究，不得用于临床用途。

## 1. 开始前准备

您需要：

- Git；
- 通过 Miniconda、Anaconda 或 Miniforge 提供的 Conda；
- 图形桌面环境；
- 足够容纳应用窗口的显示区域。

项目仅支持 **Python 3.11**（`>=3.11,<3.12`）。使用提供的 Conda 环境时，无需另行
安装该版本 Python。

窗口最小尺寸为 **900 × 620 逻辑像素**。在操作系统 150% 缩放下，约等于
**1350 × 930 物理像素**。当前版本不承诺完整界面可在 150% 缩放的 1024 × 680
物理像素屏幕中完全容纳。如果控件被截断，请最大化窗口、使用更大显示区域或降低缩放
比例后重启应用。

## 2. 安装与启动

首次使用时执行：

```bash
git clone https://github.com/CanyonChen/Open-Med-VisionX.git
cd Open-Med-VisionX
conda env create -f environment.yml
```

每次打开新终端后激活环境，再启动应用：

```bash
conda activate openmedvisionx
openmedvisionx
```

安装完成后，`openmedvisionx` 可以在任意目录运行。如果找不到该命令，请返回项目目录
修复可编辑安装：

```bash
conda activate openmedvisionx
python -m pip install -e .
openmedvisionx
```

也可以暂时在项目目录使用兼容启动方式：

```bash
python main.py
```

窗口打开后，应看到六个工作区：“**影像**”“**CT 实验**”“**模型**”“**教学**”
“**评价**”和“**AI 助手**”。顶部按钮可在英文和简体中文间切换，并保留当前状态。
使用 `Alt+1` 至 `Alt+6` 可以直接打开对应工作区。

如果窗口没有出现，请按照[安装与启动故障排查](TROUBLESHOOTING.zh-CN.md#安装与启动)
操作。

## 3. 完成第一次体验

### A. 运行不需要数据的 CT 实验

1. 打开“**教学**”。
2. 选择“**开始首个实验**”。
3. OpenMedVisionX 会切换到“**CT 实验**”，保持选中“**合成仿体（推荐）**”，并运行
   确定性的默认实验。
4. 等待状态提示完成。计算期间界面仍可响应。
5. 检查四个主要视图：输入衰减图、sinogram、重建和绝对误差。
6. 阅读报告的指标。影像比较使用同一显示范围，不会通过单独拉伸让某个结果显得更好。
7. 只改变一个参数——例如投影数、重建方法、滤波器或 SART 迭代次数——然后再次运行。

**成功标准：**您能说明结果发生了什么变化、看到重建与误差视图，并指出产生变化的
参数。整个过程没有使用本地或网络数据。

### B. 检查一份可选本地影像

如果您有权使用不含敏感信息的 PNG、JPEG、TIFF 或 DICOM 文件：

1. 打开“**影像**”，选择“**打开影像 / DICOM ZIP**”（`Ctrl+O`）。如需打开 DICOM
   目录，使用“**打开 DICOM 文件夹**”（`Ctrl+Shift+O`）。
2. 检查来源类型、dtype、尺寸、数值范围与警告。
3. 单击一个可见影像平面，把它设为“**活动视图**”。CT 实验、外部模型和 AI 图像预览
   工作流会使用这个精确的活动二维平面。
4. 打开“**直方图**”并调整显示范围，确认记录的源数值范围没有改变。
5. 尝试“**距离**”或“**面积 / ROI**”，然后选择“**清除标记**”。
6. 如果导出渲染平面，请选择新文件名并仔细阅读覆盖确认。导出的 PNG 不是屏幕截图：
   不包含视口缩放/平移、测量、标注或叠加层。

处理 DICOM/NIfTI 体数据、4D 选择、图层或分割导入前，请先阅读[数据与格式](DATA_FORMATS.zh-CN.md)。

### C. 运行内置评价示例

1. 打开“**评价**”。
2. 保留示例真值标签与预测概率。
3. 保持显式决策阈值为 `0.50`，选择“**开始评价**”。
4. 对比依赖阈值的敏感度/特异度/F1 与 AUROC、AUPRC、Brier score 和校准分箱。
5. 可将实验记录导出为 JSON 或 YAML。记录包含哈希、参数与指标，不包含输入向量或
   影像像素。

**成功标准：**状态提示评价完成，并且“**指标**”与“**校准**”标签页都有结果。

## 4. 添加可选能力

安装所需能力前，请激活同一个环境：

```bash
conda activate openmedvisionx
python -m pip install -e ".[nifti]"          # 使用 nibabel 加载 NIfTI
python -m pip install -e ".[onnx]"           # ONNX 与 ONNX Runtime
python -m pip install -e ".[llm]"            # 操作系统凭据引用
```

也可以组合安装，例如：

```bash
python -m pip install -e ".[nifti,onnx,llm]"
```

添加可选依赖后，请重启 OpenMedVisionX。

### NIfTI

安装 `nifti` 后，通过“**打开影像 / DICOM ZIP**”选择 `.nii` 或 `.nii.gz`。
OpenMedVisionX 会将空间体数据统一到 RAS+。4D 文件始终要求显式选择一个时间点/通道，
应用不会猜测。NIfTI 容器本身不能证明模态、HU 或其他强度语义。详见
[数据与格式](DATA_FORMATS.zh-CN.md#nifti-体数据)。

### 经过审查的离线模型

PyTorch 需要单独安装，因为正确构建取决于您的电脑。本版本可移植的 CPU 安装方式为：

```bash
python -m pip install -r requirements/torch-cpu.txt
python -m workbench.models
```

下面的 GPU 命令只适用于已说明的测试机器：Windows 开发机、**NVIDIA RTX 4060 Laptop
GPU**、8 GiB VRAM 和兼容 CUDA 13.0 的驱动。不要假设它适合其他电脑。

```bash
python -m pip install -r requirements/torch-cu130.txt
python -m workbench.models --smoke --device auto
```

其他 GPU/驱动/操作系统组合请从 PyTorch 官方 [Start
Locally](https://pytorch.org/get-started/locally/) 页面选择匹配的构建，再运行上述校验命令。

三条经过审查的工作流具有不同且不可互换的输入：

- [DIVal LoDoPaB FBP-U-Net](../src/workbench/resources/model_bundles/dival-lodopab-fbpunet/MODEL_CARD.md)运行随包公开 LoDoPaB 案例，输入属于固定 Hann-FBP 基准域，不是任意 CT 或 HU 图像。
- [DeepInverse MRI MoDL](../src/workbench/resources/model_bundles/deepinv-mri-modl/MODEL_CARD.md)运行确定性的**合成 MRI** 演示，使用图像派生的单线圈 k-space，而不是扫描仪原始数据。
- [MONAI BraTS 三维分割](../src/workbench/resources/model_bundles/monai-brats-segmentation/MODEL_CARD.md)在引导流程校验用户管理的本地 T1ce/T1/T2/FLAIR/SEG 案例前保持**禁用**。项目不包含或下载 BraTS 案例。

运行前请阅读[随包模型指南](MODEL_BUNDLES.zh-CN.md)。

### 其他本地模型

“**外部清单**”流程不会下载模型或安装依赖。当前桌面能力只接受一个受支持的活动二维
影像输入。即使外部清单有效，复合 k-space/mask 输入、提示词、多模态、完整 3D/4D
体数据、时间序列、sinogram 和全切片流程也会被拒绝。维护的
[示例清单](../src/workbench/inference/examples/manifest.yaml)不可运行，因为示例模型文件
有意未提供。请按照[自定义模型指南](CUSTOM_MODELS.zh-CN.md)操作。

### AI 助手

第一次体验不需要任何 AI 服务。启用前请阅读[AI 助手、隐私与云端图像](LLM_SECURITY.zh-CN.md)。
网络使用和图像附加是两个独立的主动选择；图像还需要检查精确的待发送 PNG，并逐次授权。

## 5. 选择下一份指南

- 要了解全部六个工作区，请继续阅读[用户指南](USER_GUIDE.zh-CN.md)。
- 要准备 DICOM、NIfTI、mask 或标签图层，请阅读[数据与格式](DATA_FORMATS.zh-CN.md)。
- 要按顺序系统学习，请使用[学习课程](TEACHING_CURRICULUM.zh-CN.md)。
- 要查询术语，请使用[术语表](GLOSSARY.zh-CN.md)。
- 如果任何步骤失败，请打开[故障排查](TROUBLESHOOTING.zh-CN.md)。

返回[文档首页](INDEX.zh-CN.md)。
