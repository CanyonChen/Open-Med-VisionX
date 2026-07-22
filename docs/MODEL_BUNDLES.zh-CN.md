# OpenMedVisionX 内置模型用户指南

[English](MODEL_BUNDLES.md) · [文档索引](INDEX.zh-CN.md) · [数据格式](DATA_FORMATS.zh-CN.md) · [自定义模型](CUSTOM_MODELS.zh-CN.md)

本指南从安装模型运行环境开始，带你完成三条经过审查的桌面流程，并读懂运行结果。
整个过程不需要编程。安装 PyTorch 后，内置模型及其准备好的教学输入都在本地运行，
无需下载模型。

> [!WARNING]
> 内置模型是固定的学习与研究参考，不是医疗器械，也没有在 OpenMedVisionX 中完成
> 临床验证。不得将输出用于诊断、治疗、分诊或任何患者相关决策。

## 安装 PyTorch

先完成[快速开始](QUICKSTART.zh-CN.md)，再激活项目环境：

```bash
conda activate openmedvisionx
```

三份内置模型都需要 PyTorch。要安装与本次发布匹配、可跨设备使用的纯 CPU 版本，
请运行：

```bash
python -m pip install -r requirements/torch-cpu.txt
```

CPU 模式最容易配置，不需要 NVIDIA GPU。如果需要在其他操作系统或 GPU 上运行，
请在 PyTorch 官方 [Start Locally](https://pytorch.org/get-started/locally/) 页面根据
实际操作系统、GPU 和驱动选择命令，不要猜测 CUDA wheel。仓库唯一实测 GPU 配置的
命令及其严格适用范围记录在[版本完整性附录](#实测-gpu-参考)中。

安装完成后启动应用：

```bash
openmedvisionx
```

启动模型流程时，OpenMedVisionX 不会下载内置模型、BraTS 案例或额外运行环境。
如果缺少必要组件，流程会停止并明确说明缺失项。

## 验证安装

在桌面应用中打开“**模型 → 内置模型**”，选择“**验证目录与设备**”。应用会校验
三份经过审查的模型包，并显示本地可用的 PyTorch 设备。

也可以在终端完成同样的检查。先执行只校验完整性、不运行模型的命令：

```bash
python -m workbench.models
```

安装 PyTorch 后，在首选设备上加载每份模型并运行小型确定性输入：

```bash
python -m workbench.models --smoke --device auto
```

如需最严格的随包数值校验，将每份结果与经过审查的预期输出比较：

```bash
python -m workbench.models --golden --device auto
```

`auto` 会在 PyTorch 报告 CUDA 可用时使用 CUDA，否则使用 CPU。在 CPU 上执行 smoke
或 golden 检查可能明显更慢，尤其是三维 MONAI 模型。校验成功只能证明随包文件与
本地运行环境可以协同工作，不能证明医学准确性、临床有效性或泛化能力。

如果校验报告文件缺失、大小不符、哈希不符或出现预期之外的模型包 ID，不要绕过
检查，也不要手工替换单个文件。请从可信项目发行版获取完整的新副本，再重新校验。

## 理解每个模型的输入

每份模型都有专用流程，因为它们的输入具有不同的物理含义。应用绝不会把任意已打开
图像静默转换为这些任务专用输入。

| 模型 | 输入来自哪里 | 你需要提供什么 | 科学边界 |
| --- | --- | --- | --- |
| [DIVal LoDoPaB FBP-U-Net](../src/workbench/resources/model_bundles/dival-lodopab-fbpunet/MODEL_CARD.md) | 随包公开 LoDoPaB-CT 教学案例 | 无 | 只接受固定的 `1×1×362×362` DIVal Hann 滤波 FBP 域；数值是归一化衰减，不是临床 HU |
| [DeepInverse MRI MoDL 教学模型](../src/workbench/resources/model_bundles/deepinv-mri-modl/MODEL_CARD.md) | 在内存中生成的确定性合成 MRI 演示 | 无 | 使用模拟单线圈复数 k-space 和二值 mask；不是扫描仪原始数据 |
| [MONAI BraTS 三维分割](../src/workbench/resources/model_bundles/monai-brats-segmentation/MODEL_CARD.md) | 由你选择并校验的一份本地 BraTS 风格案例 | T1、T1ce、T2、FLAIR 与 Task 1 SEG NIfTI 文件 | BraTS 2018 胶质瘤模型在 BraTS 2021 Task 1 案例上评价；不是通用脑部病灶模型 |

模型包不包含训练数据集、患者影像、BraTS 案例或扫描仪采集数据。DIVal 使用一份
单独审查的公开基准案例，DeepInverse 创建模拟输入，MONAI 只使用你选择的本地案例。

## 流程一——DIVal LoDoPaB CT 重建

### 开始前

此流程需要 PyTorch，但不需要用户数据。它使用随包的
`lodopab-ct-test-03456` 基准示例，确保观测值、解析基线、模型结果和参考值都属于
同一个经过审查的域。

### 运行流程

1. 打开“**模型 → 内置模型**”。
2. 选择“**DIVal LoDoPaB FBP-U-Net**”，阅读面板中的输入契约与模型卡。
3. 选择“**运行 LoDoPaB 案例**”。
4. 等待案例校验、模型推理和参考指标计算完成。状态区会记录实际设备与任何回退。
5. 如需保留可复现且不含像素的摘要，选择“**导出实验 JSON…**”并指定合适位置。

### 阅读此流程的视图

| 视图 | 含义 |
| --- | --- |
| 观测值 | `1000×513` 的模拟低剂量平行束线积分数据 |
| Hann FBP | 固定的 `362×362` 解析重建，也是模型输入 |
| 模型输出 | 同一归一化基准域中的 FBP-U-Net 后处理重建 |
| 真值 | 经过审查的 `362×362` 基准参考，不是临床 HU 图像 |
| 绝对误差 | 模型输出与真值逐像素绝对差，使用固定参考量程显示 |

页面还会报告相对于这一个参考值的 MAE、RMSE 和 PSNR。对于本案例和固定数据量程，
MAE、RMSE 越低，PSNR 越高，数值匹配越接近。这些指标不能说明模型在其他扫描仪、
解剖结构、剂量或人群中的表现。

不要向该模型提供任意 CT 图像、HU 切片或扫描仪 sinogram。学习型组件与文档规定的
DIVal LoDoPaB FBP 算子及基准几何严格绑定。

## 流程二——DeepInverse 合成 MRI 重建

### 开始前

此流程同样需要 PyTorch，但不需要你提供文件。应用会在内存中生成确定性的
`128×128` 数字模体，将其转换为模拟单线圈 k-space，并应用固定欠采样 mask。
该演示用于直观看到欠采样与学习型数据一致性的作用。

### 运行流程

1. 打开“**模型 → 内置模型**”。
2. 选择“**DeepInverse MRI MoDL 教学模型**”，阅读其 k-space/mask 输入契约。
3. 选择“**运行合成 MRI 演示**”。
4. 比较源模体、零填充基线、MoDL 结果、绝对误差与指标。

该演示当前不支持 JSON 导出。复合 k-space 与 mask 输入只由内置模型专用流程创建；
通用外部清单页面不能借用这条输入路径。

### 阅读此流程的视图

| 视图 | 含义 |
| --- | --- |
| 数字模体 | 用于创建模拟输入的确定性源图像 |
| 零填充 IFFT | 将缺失 k-space 样本填为零后直接执行逆变换得到的基线 |
| MoDL 幅值图 | 由模型双通道复数重建明确派生的显示幅值 |
| 绝对误差 | MoDL 幅值图与源模体之间的逐像素绝对差 |

MAE、RMSE 与 PSNR 描述模型结果和生成模体的一致程度，只适用于这项受控机制演示。
模拟 k-space 必须始终标记为图像派生模拟，不能标记成临床采集或扫描仪原始数据。

## 流程三——MONAI BraTS 三维分割

### 开始前

桌面流程需要一个由用户管理的目录，并且能在其中唯一识别以下五份三维 NIfTI 文件：

- T1；
- T1ce；
- T2；
- FLAIR；
- Task 1 SEG，整数标签只能是 `{0,1,2,4}`。

使用 `case_t1.nii.gz`、`case_t1ce.nii.gz`、`case_t2.nii.gz`、
`case_flair.nii.gz`、`case_seg.nii.gz` 这类名称，可以明确区分各模态。四个 MRI 体数据
和 SEG 必须具有相同的 shape、spacing、affine、方向、world coverage、qform 与
sform。模型流程要求已经配准并对齐到 1 mm 的数据，不会静默配准或重采样案例。

项目不附带 BraTS 案例。只能在你获准使用数据时获取并处理它们。配置过程会原地校验
所选文件，不会下载、复制、重命名或上传文件。

### 配置并运行流程

1. 打开“**模型 → 内置模型**”，选择“**MONAI BraTS 三维分割模型**”。
2. 选择“**配置本地 BraTS 2021…**”，再选择案例所在目录。
3. 检查校验报告。必须先修复模态缺失或重复、数据不可读、非有限值、几何不一致、
   qform/sform 不一致或 SEG 标签无效等问题。
4. 保存可选的匿名本地 manifest 时，需要接受对话框中的条款。该 manifest 只包含
   哈希和几何，不含文件路径、名称或人员标识。本次会话中的有效校验报告已足够运行，
   保存 manifest 不是必需步骤。
5. 校验通过后，MONAI 不再处于禁用状态。选择“**运行 BraTS 分割**”。
6. 检查实际设备与参数。经过审查的桌面默认值是 `96×96×96` 滑窗 patch、`0.5`
   overlap 和 `0.5` 概率阈值。
7. 运行后如需保留不含像素的参数和指标记录，选择“**导出实验 JSON…**”。

### 阅读此流程的视图

流程会将已经对齐的输入统一到 RAS+，分别对每个模态的非零体素执行 z-score，运行
完整体滑窗推理，再将结果映射回源几何。页面显示：

- **WT 概率**、**TC 概率**和 **ET 概率**。适配器会把模型原生的 TC/WT/ET 输出
  明确重排为页面显示的 WT/TC/ET 顺序。
- **WT / TC / ET 掩膜**，由可见阈值生成。参考 SEG 中 WT 使用标签 `{1,2,4}`，
  TC 使用 `{1,4}`，ET 使用 `{4}`。
- 每个区域相对所提供 SEG 的 Dice、以毫米计的 HD95，以及以毫升计的绝对体积误差。

该模型面向文档规定的 BraTS 2018 胶质瘤域训练。在 BraTS 2021 Task 1 案例上运行
属于明确的域偏移，不是外部临床验证。它不是适用于其他病灶、疾病、扫描仪或采集协议
的通用模型。

## 阅读结果

应结合显示图像与指标进行判断。任何单一数字都不足以证明结果可信。

| 术语 | 如何理解 |
| --- | --- |
| 基线 | DIVal 的 FBP 或 DeepInverse 的零填充 IFFT，用于显示学习型结果正在与什么比较 |
| 模型输出 | 任务专用估计，不是诊断，也不会自动成为物理测量值 |
| 真值／参考 | 用于比较的已准备基准或所提供 SEG；不能证明模型可以泛化 |
| 绝对误差 | 估计值与参考值在何处、相差多少；明暗含义取决于显示量程 |
| MAE／RMSE | 平均数值误差；在同一案例和量程内越低越接近 |
| PSNR | 以分贝表示的信号误差比；只有参考量程与任务相同时，越高才表示越接近 |
| 概率 | sigmoid 后的模型输出，不是经过校准的临床置信度；BraTS 掩膜默认阈值为 `0.5` |
| Dice | `0` 到 `1` 的区域重叠度；越高表示重叠越多，但小结构可能不稳定 |
| HD95 | 以毫米计的稳健边界距离；越低表示越接近，空区域案例可能显示 `n/a` |
| 绝对体积误差 | 以毫升计的分割体积差；相对该参考值越接近零越好 |

解释指标前，必须检查输入契约、预处理、实际设备、警告、显示量程与参考值。视觉上
可信的重建或掩膜仍可能抑制、偏移或虚构结构。单个案例、smoke 测试或 golden 校验
都不能证明临床性能。

## 选择运行设备

桌面流程会记录实际使用的设备。命令行检查支持以下选择：

| 选择 | 行为 | 适用情况 |
| --- | --- | --- |
| `auto` | PyTorch 报告 CUDA 可用时优先使用 CUDA，否则使用 CPU；CUDA 加载或推理错误会先明确报告，再重试 CPU | 推荐默认值 |
| `cpu` | 始终使用 CPU | 跨设备检查、没有兼容 GPU 的系统，或从 GPU 显存压力中恢复 |
| `cuda` | 必须使用 CUDA；不可用时明确失败，不会在 CPU 上静默通过 | 有意进行 CUDA 专用检查 |

DIVal 与 DeepInverse 是小型教学流程。MONAI 完整体分割在 CPU 上可能慢得多，因此
建议为该流程使用兼容 GPU。如果 GPU 运行耗尽显存，请关闭其他 GPU 任务，必要时改用
CPU。不要为了让运行完成而改变模态顺序、spacing、标准化、patch 契约或其他科学输入。

## 解决常见问题

| 问题 | 处理方法 |
| --- | --- |
| 缺少 `torch` 或模型按钮无法运行 | 激活 `openmedvisionx`，安装 CPU 运行环境或与平台严格匹配的 PyTorch，重启应用并重新校验 |
| 完整性校验失败 | 从可信发行版恢复完整项目；不要编辑、重命名模型包文件，也不要混用不同版本文件 |
| CUDA 不可用 | 使用 `auto` 或 `cpu`，或者重装与操作系统、GPU、驱动严格匹配的 PyTorch |
| MONAI 一直处于禁用状态 | 在当前会话中成功完成“配置本地 BraTS 2021…”；只保存 manifest 不能替代重新校验 |
| BraTS 校验失败 | 确保 T1、T1ce、T2、FLAIR、SEG 均可唯一识别，再按照对话框修复 shape、spacing、affine、方向、qform/sform、有限值或标签问题 |
| BraTS 很慢或显存不足 | 确认实际设备，关闭其他 GPU 任务，必要时在 CPU 上重试；CPU 完整体推理本来就会更慢 |
| 无法导出 | 先完成受支持的 LoDoPaB 或 BraTS 运行；DeepInverse 当前不支持 JSON 导出 |
| 有效外部清单不能使用这些输入 | 这是预期行为：当前通用路径只接受一个活动二维图像，不接受复合 k-space/mask、四个 MRI 体数据、提示或完整 3D/4D 体数据 |
| 结果看起来合理但不一致 | 重新检查模型卡、输入来源、模态／通道顺序、单位、预处理、警告、显示量程与参考值，再重复运行 |

其他启动、依赖、数据和 GPU 症状请查看[故障排查](TROUBLESHOOTING.zh-CN.md)。

## 相关指南

- [文档索引](INDEX.zh-CN.md)：根据要完成的任务选择最短阅读路径。
- [快速开始](QUICKSTART.zh-CN.md)：安装并启动基础应用。
- [用户指南](USER_GUIDE.zh-CN.md)：学习完整桌面工作区。
- [数据格式](DATA_FORMATS.zh-CN.md)：安全准备 DICOM、NIfTI、普通图像、mask 与标注。
- [自定义本地模型](CUSTOM_MODELS.zh-CN.md)：在当前桌面能力边界内创建并运行本地模型清单。
- [故障排查](TROUBLESHOOTING.zh-CN.md)：诊断安装、运行环境与输入校验问题。

“**模型 → 外部清单**”与三条经过审查的内置流程相互独立。它目前只接受一个受支持的
活动二维图像或平面，不能复用 DeepInverse 的合成 MRI k-space/mask 生成器，也不能
收集 MONAI 使用的四个模态。界面明确拒绝超出能力边界的契约是有意设计，目的是防止
伪造不符合科学要求的输入。

## 版本完整性附录

本附录记录用于识别精确审查资产的固定发行事实。大多数用户只需执行前文的校验命令。

### 固定模型载荷

允许列表中只有以下三个 ID：

- `dival-lodopab-fbpunet`；
- `deepinv-mri-modl`；
- `monai-brats-segmentation`。

| 模型包 | 文件 | 精确大小 | SHA-256 |
| --- | --- | ---: | --- |
| DeepInverse | `weights.npz` | 9,929 bytes | `702658708828d13135228e32fef980ba1048e200f5c2fa4ebf54fa12d653f8ab` |
| DeepInverse | `golden.npz` | 11,251 bytes | `2b709a7c16aedd65110aaf929bb2c6cc35db1c94d9fe01b751a29b06634d29af` |
| DIVal | `weights.npz` | 2,318,110 bytes | `8b18fd2a88355ddec043ae7c737ddf3321424e2ba52102869d3dbaf6bf68504c` |
| DIVal | `golden.npz` | 920,732 bytes | `6789c93592ab6cfd3d4924e6d077ce8966d0e7fd7bb0b7f1a7305d66f742a3df` |
| MONAI | `model.ts` | 18,911,784 bytes | `729980a0bd9347bf2397701eb329e12517918dc282a2d09c40458e95b24ceed9` |
| MONAI | `golden.npz` | 330,439 bytes | `d7982ada82f56b28615ed6ad170641ee1f3f0cb6a819285598c0380efa957e45` |

六个模型和 golden 载荷精确合计 **22,502,245 bytes**，低于 25 MiB 发布预算。DIVal
与 DeepInverse 使用经过审查的数值 NPZ，并以 `allow_pickle=False` 加载；MONAI 使用
固定的 TorchScript 资产。校验器还会检查每份机器可读记录、许可证据、文件大小、摘要、
允许列表成员关系与总预算。

### 固定 LoDoPaB 教学案例

单独随包的案例为 `lodopab-ct-test-03456`（case version 2），来自
[LoDoPaB-CT](https://doi.org/10.5281/zenodo.3384092) test index 3456。它的
`sample.npz` 精确大小为 **1,912,858 bytes**，SHA-256 为
`b323cdef2529927336069b3385605d1049117fe69e59583072861fa573493846`。文件包含有限的
`float32` 数组：`1000×513` 观测值、`362×362` 固定 Hann FBP、`362×362` 真值，
以及规范元数据。项目依照 **CC BY 4.0** 再分发该案例，并保留随包的
[说明与署名](../src/workbench/resources/teaching_cases/lodopab-ct/NOTICE.md)。

安全加载器会先检查案例记录、精确大小与哈希、NPZ entry 允许列表，以及每个数组的
哈希、shape 和 dtype，再以 `allow_pickle=False` 打开文件。案例不含 DICOM 元数据、
患者标识符或源路径。单个公开基准示例不能证明泛化能力或临床有效性。

### 实测 GPU 参考

仓库记录的 Windows GPU 测试使用 **NVIDIA RTX 4060 Laptop GPU、8 GiB VRAM** 与
PyTorch `2.13.0+cu130`：

```bash
python -m pip install -r requirements/torch-cu130.txt
```

该命令只适用于这套已测试的驱动与 GPU 配置，不是通用 CUDA 建议。其他操作系统、
NVIDIA GPU 或驱动应使用 PyTorch 官方选择器；纯 CPU 安装请使用本指南第一节的 CPU
命令。
