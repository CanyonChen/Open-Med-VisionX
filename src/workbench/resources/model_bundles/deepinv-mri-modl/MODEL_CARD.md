# DeepInverse MRI MoDL Teaching Model / DeepInverse MRI MoDL 教学模型

## English

Purpose: make k-space undersampling and learned data consistency observable.
Inputs are simulated single-coil complex k-space and a binary mask, both shaped
`B×2×H×W` with real/imaginary channels. The two reviewed HQS iterations use the
checkpoint's learned step sizes and one shared two-layer DnCNN. Display magnitude
is derived only after complex reconstruction.

This tiny demo checkpoint is for mechanism exploration. Inputs derived from a
reconstructed MRI must remain labelled **image-derived simulation—not scanner
raw data**. It has no clinical validation and must not be used for diagnosis.

## 中文

用途：直观展示 k-space 欠采样和学习型数据一致性。输入为模拟的单线圈复数
k-space 与二值采样 mask，二者形状均为 `B×2×H×W`，通道分别表示实部和虚部。
两个经审查的 HQS 迭代使用 checkpoint 中学习得到的步长，并共享一个两层 DnCNN；
显示用幅度图仅在复数重建完成后派生。

这个小型 demo checkpoint 只用于理解机制。从重建 MRI 反向生成的输入必须持续标记为
**图像派生模拟，并非扫描仪原始数据**。它未经过临床验证，不得用于诊断。
