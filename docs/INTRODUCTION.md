# About OpenMedVisionX · 项目简介

[Documentation home / 文档首页](INDEX.md) · [English README](../README.md) · [中文 README](README.zh-CN.md)

## English

OpenMedVisionX is a bilingual, local-first desktop workbench for learning
medical computer vision. It connects six user workspaces—Images, CT Lab,
Models, Learn, Evaluate, and AI Assistant—so that image semantics, geometry,
preprocessing, intermediate states, model contracts, metrics, provenance, and
privacy decisions remain visible throughout an experiment.

A new user can begin without any local data: **Start first experiment** opens a
deterministic synthetic CT phantom, generates its sinogram, reconstructs it,
and presents the error and metrics. The Images workspace then supports raster,
DICOM, supported Enhanced multi-frame CT/MR, and optional NIfTI inputs without
silently inventing HU, spacing, orientation, or volume meaning.

Three reviewed offline learning workflows are included:

- DIVal FBP-U-Net uses one public **LoDoPaB-CT** benchmark case in its fixed
  Hann-FBP domain;
- **DeepInverse** MRI MoDL uses a deterministic, image-derived synthetic
  single-coil k-space demonstration; and
- **MONAI** BraTS segmentation runs only after the user selects and validates
  **four co-registered** T1ce/T1/T2/FLAIR volumes plus a reference SEG in an
  authorized local case.

No training dataset, BraTS case, API credential, or patient image is included.
The optional provider-backed Teaching chat sends data only after explicit
network use. The separate **Structured artifacts · API preview** explains typed
artifact contracts; ordinary chat does not populate it, and local review does
not create an image layer.

OpenMedVisionX is for learning and research only. It is not a medical device
and its output must not be used for diagnosis, treatment, triage, or patient
decisions.

Start with the [Quickstart](QUICKSTART.md), then use the [User
guide](USER_GUIDE.md) or the task-oriented [Learning
curriculum](TEACHING_CURRICULUM.md).

## 简体中文

OpenMedVisionX 是一个双语、本地优先的医学计算机视觉学习桌面工作台。它把“影像”
“CT 实验”“模型”“教学”“评价”和“AI 助手”六个用户工作区连接起来，使影像语义、
空间几何、预处理、中间状态、模型契约、指标、来源与隐私决定在实验全过程中保持可见。

新用户无需本地数据即可开始：“开始首个实验”会打开确定性合成 CT 仿体，生成
sinogram、执行重建并展示误差与指标。随后可以在“影像”工作区打开栅格图、DICOM、
受支持的 Enhanced 多帧 CT/MR 和可选 NIfTI；应用不会静默编造 HU、间距、方向或体数据
语义。

项目提供三条经过审查的离线学习流程：

- DIVal FBP-U-Net 使用一个公开 **LoDoPaB-CT** benchmark 案例，并限定在固定 Hann-FBP
  输入域；
- **DeepInverse** MRI MoDL 使用确定性、图像派生的合成单线圈 k-space 演示；
- **MONAI** BraTS 分割只有在用户选择并校验已获授权本地案例中的**四个完成配准**的
  T1ce/T1/T2/FLAIR 体数据与参考 SEG 后才运行。

项目不包含训练数据集、BraTS 案例、API 凭据或患者影像。可选的服务端“教学对话”只有
在用户明确启用网络后才会发送数据。单独的“**结构化产物 · API 预览**”用于说明类型化
产物契约；普通对话不会填充它，本地审查也不会创建影像图层。

OpenMedVisionX 仅用于学习与研究，不是医疗器械，输出不得用于诊断、治疗、分诊或患者
相关决策。

请从[快速开始](QUICKSTART.zh-CN.md)入门，再使用[用户指南](USER_GUIDE.zh-CN.md)或按任务
组织的[学习课程](TEACHING_CURRICULUM.zh-CN.md)。
