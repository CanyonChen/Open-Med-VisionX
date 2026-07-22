# MONAI BraTS 3D Segmentation / MONAI BraTS 三维分割

## English

Purpose: demonstrate a real four-modal MRI segmentation workflow. The required
channel order is **T1ce, T1, T2, FLAIR**. Volumes must already be co-registered,
RAS+ oriented, and verified as 1 mm spacing. Each channel is z-score normalized
over its nonzero voxels. The native output order is TC, WT, ET; OpenMedVisionX
must visibly reorder it to WT, TC, ET before evaluation.

The model was trained on BraTS 2018 glioma MRI. It is not a general brain-lesion
model, has no clinical validation in this product, and must not be used for
diagnosis. No BraTS case is included. Obtain an authorized case separately and
complete geometry checks before inference.

## 中文

用途：演示真实的四模态 MRI 分割流程。输入通道顺序必须为
**T1ce、T1、T2、FLAIR**；四个体数据应预先完成配准，经检查为 RAS+ 方向和
1 mm 间距。每个通道仅使用非零体素进行 z-score 标准化。模型原生输出顺序为
TC、WT、ET；OpenMedVisionX 在评价前必须明确重排为 WT、TC、ET。

该模型的训练域是 BraTS 2018 胶质瘤 MRI，并非通用脑部病灶模型，且未在本工具
中完成临床验证，不得用于诊断。项目不附带任何 BraTS 病例；请通过获授权的官方
途径取得病例，并在推理前完成几何一致性检查。
