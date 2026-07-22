# Third-party notice / 第三方声明

## English

The source checkpoint is `demo_tour_mri.pth` from the DeepInverse demo repository
at revision `35ef1ffedb3c8973721754a5f2bd1ca1a36added`. OpenMedVisionX loaded the
reviewed tensor dictionary with `weights_only=True`, rebuilt the official
DeepInverse 0.4.1 two-iteration MoDL/HQS model, and proved the trusted runtime
adapter numerically equivalent on the same input. Only numeric arrays are
exported to `weights.npz`; the original pickle checkpoint is not distributed.
The pinned demo repository declares the BSD 3-Clause License.

## 中文

源 checkpoint 是 DeepInverse demo 仓库固定版本
`35ef1ffedb3c8973721754a5f2bd1ca1a36added` 中的 `demo_tour_mri.pth`。
OpenMedVisionX 使用 `weights_only=True` 读取经审查的张量字典，根据固定版本的
DeepInverse 0.4.1 重建 MRI tour 的两次 MoDL/HQS 计算图，并在同一输入上证明可信
运行时适配器数值等价。发行包仅在 `weights.npz` 中保存数值数组，原始 pickle
checkpoint 不进入发行包。该固定 demo 仓库声明采用 BSD 3-Clause 许可证。
