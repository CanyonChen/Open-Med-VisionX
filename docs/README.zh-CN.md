<h1 align="center">OpenMedVisionX</h1>

<p align="center">
  <img src="../figs/logo_full.png" alt="OpenMedVisionX" width="900">
</p>

<p align="center">
  在一个本地优先的桌面工作台中学习医学影像、重建、模型推理与评价。
</p>

<p align="center">
  <a href="../README.md">English</a>
</p>

<p align="center">
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/Python-3.11-blue.svg" alt="Python 3.11"></a>
  <a href="https://pypi.org/project/PyQt5/"><img src="https://img.shields.io/badge/PyQt5-5.15+-41CD52.svg" alt="PyQt5 5.15+"></a>
  <a href="../LICENSE"><img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License: MIT"></a>
</p>

OpenMedVisionX 面向正在学习或探索医学计算机视觉的用户。它会把影像语义、空间几何、
预处理、中间结果、评价指标、来源和隐私选择明确呈现出来，帮助您理解完整工作流，而
不只是按下一个预测按钮。

> [!WARNING]
> OpenMedVisionX 仅用于学习与研究，不是医疗器械，不得用于诊断、治疗、分诊或任何
> 患者相关决策。

## 为什么使用 OpenMedVisionX？

应用由六个工作区组成。它们没有强制顺序，但建议从“教学”中的安全首个实验开始。

| 工作区 | 可以学习和完成什么 |
| --- | --- |
| **影像** | 打开栅格图、DICOM、Enhanced 多帧 CT/MR 和可选 NIfTI；检查平面、像素值、几何、测量与图层。 |
| **CT 实验** | 生成合成衰减仿体，比较投影数据、FBP、BP、直接傅里叶重建、SART、误差和中间状态。 |
| **模型** | 运行三条经过审查的离线流程，或运行满足契约的用户本地模型；输入与输出含义始终可见。 |
| **教学** | 按引导课程学习像素、医学几何、重建、模型输入、评价与负责任的 AI 使用。 |
| **评价** | 检查按组隔离的数据集划分，在显式阈值下评价二分类概率，查看校准并导出无像素实验记录。 |
| **AI 助手** | 向用户配置的服务请求教学解释。单独的“**结构化产物 · API 预览**”只审查由可信集成提供的类型化产物，普通对话不会填充它。 |

界面可在英文与简体中文之间切换，不会改变已加载数据或当前工作区状态。本地影像查看、
CT 实验、评价和内置模型运行均不需要网络。

## 5 分钟开始使用

请准备 Git、Conda（Miniconda、Anaconda 或 Miniforge）和图形桌面环境。项目环境会
提供受支持的 **Python 3.11**。

```bash
git clone https://github.com/CanyonChen/Open-Med-VisionX.git
cd Open-Med-VisionX
conda env create -f environment.yml
conda activate openmedvisionx
openmedvisionx
```

后续启动时，在项目目录打开终端并只执行：

```bash
conda activate openmedvisionx
openmedvisionx
```

第一次体验无需文件、模型、网络或患者数据：

1. 打开“**教学**”，选择“**开始首个实验**”。
2. 应用会打开“**CT 实验**”，并运行确定性的合成仿体流程。
3. 对照查看输入、sinogram、重建、绝对误差与指标。

窗口最小尺寸为 **900 × 620 逻辑像素**。在操作系统 150% 缩放下，约等于
**1350 × 930 物理像素**。当前版本不承诺完整界面可在 150% 缩放的 1024 × 680
物理像素屏幕中完全容纳；如果控件被截断，请最大化窗口、使用更大显示区域或降低
缩放比例。

接下来请阅读逐步讲解的[快速开始](QUICKSTART.zh-CN.md)。

## 选择阅读路线

| 您的目标 | 推荐路径 |
| --- | --- |
| 安装并完成第一次体验 | [文档首页](INDEX.zh-CN.md) → [快速开始](QUICKSTART.zh-CN.md) |
| 打开自己的影像或体数据 | [数据与格式](DATA_FORMATS.zh-CN.md) → [用户指南](USER_GUIDE.zh-CN.md) |
| 了解每个工作区 | [用户指南](USER_GUIDE.zh-CN.md) |
| 运行经过审查的离线模型 | [随包模型指南](MODEL_BUNDLES.zh-CN.md) |
| 使用其他本地模型 | [自定义模型指南](CUSTOM_MODELS.zh-CN.md) |
| 系统学习相关概念 | [学习课程](TEACHING_CURRICULUM.zh-CN.md) → [术语表](GLOSSARY.zh-CN.md) |
| 安全启用 AI 服务 | [AI 助手、隐私与云端图像](LLM_SECURITY.zh-CN.md) |
| 排查错误 | [故障排查](TROUBLESHOOTING.zh-CN.md) |

## 支持的数据与可选能力

基础环境支持 PNG、JPEG、TIFF、DICOM 文件/文件夹/ZIP、受支持的 DICOM SEG 与
RTSTRUCT，以及无损栅格标签图。只在需要时安装可选能力：

```bash
conda activate openmedvisionx
python -m pip install -e ".[nifti]"          # NIfTI .nii / .nii.gz
python -m pip install -e ".[onnx]"           # 用户提供的 ONNX 模型
python -m pip install -e ".[llm]"            # 操作系统凭据存储
```

PyTorch 需要单独安装，因为其构建必须匹配 CPU、GPU、操作系统与驱动。请按照
[随包模型指南](MODEL_BUNDLES.zh-CN.md)操作，不要猜测 CUDA 安装包。

OpenMedVisionX 不会推断缺失的医学语义。普通栅格图没有可信来源时不具备 HU 或物理
间距；TIFF 页面不会自动成为体数据；4D NIfTI 必须显式选择一个体；模型输入只有满足
已声明任务契约时才会被接受。完整说明见[格式支持矩阵](DATA_FORMATS.zh-CN.md)。

## 隐私、科学边界与许可

- OpenMedVisionX 只读取您明确选择的路径，不会自动扫描相邻文件夹、下载数据集或
  上传影像。
- 只有您明确启用并调用 AI 服务，或选择打开外部链接时才会发生网络活动。发送图像
  还需要额外检查精确预览并逐次授权。
- 源像素保持不变；显示映射、叠加层与明确确认的派生重采样预览会分别记录。
- 项目包含三份经过审查的离线模型包和一个公开 LoDoPaB-CT 教学案例；不包含训练
  数据集、BraTS 案例、API 凭据或患者影像。
- OpenMedVisionX 目前处于 Alpha 阶段。请保留备份并独立验证所有输出。

应用源代码采用 [MIT License](../LICENSE)。导入的数据、模型、适配器和权重保留
各自条款。疑似漏洞或泄露请按照[安全策略](SECURITY.zh-CN.md)私下报告。

## 文档导航

请从双语[文档首页](INDEX.zh-CN.md)开始。主要指南包括：

- [快速开始](QUICKSTART.zh-CN.md) · [Quickstart](QUICKSTART.md)
- [用户指南](USER_GUIDE.zh-CN.md) · [User guide](USER_GUIDE.md)
- [随包模型指南](MODEL_BUNDLES.zh-CN.md) · [Bundled model guide](MODEL_BUNDLES.md)
- [学习课程](TEACHING_CURRICULUM.zh-CN.md) · [Learning curriculum](TEACHING_CURRICULUM.md)
- [AI 助手与隐私](LLM_SECURITY.zh-CN.md) · [AI Assistant and privacy](LLM_SECURITY.md)
- [故障排查](TROUBLESHOOTING.zh-CN.md) · [Troubleshooting](TROUBLESHOOTING.md)

所有用户指南都提供英文与简体中文版本，并能返回文档首页。
