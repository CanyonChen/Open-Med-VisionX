# OpenMedVisionX 故障排查

[English](TROUBLESHOOTING.md) · [文档首页](INDEX.zh-CN.md) · [快速开始](QUICKSTART.zh-CN.md)

在下面找到对应现象，按顺序检查，并在恢复预期状态后停止。错误通常不会改变当前源数据。

> [!IMPORTANT]
> 绝不能把患者数据、凭据、私有路径或受限模型文件附加到公开 Issue。疑似安全漏洞或
> 数据泄露时，请按照[安全策略](SECURITY.zh-CN.md)私下报告。

## 查找问题类型

| 现象 | 前往 |
| --- | --- |
| 找不到 `conda` 或 `openmedvisionx`；窗口没有出现 | [安装与启动](#安装与启动) |
| 影像、体数据、文件夹或 ZIP 被拒绝 | [打开影像与体数据](#打开影像与体数据) |
| mask、SEG、RTSTRUCT 或 label map 被拒绝/隐藏 | [标注与图层](#标注与图层) |
| CT 实验不能使用活动影像，或指标看起来不对 | [CT 实验](#ct-实验) |
| PyTorch、内置模型、外部清单或 BraTS 失败 | [模型](#模型) |
| 评价输入或导出失败 | [评价与导出](#评价与导出) |
| AI 发送/图像附加被禁用，或凭据失败 | [AI 助手](#ai-助手) |

先阅读完整状态行和对话框，其中通常会指出无效字段、缺失依赖或不受支持的能力。不要
同时反复修改不相关设置；每次只改变一个条件。

## 安装与启动

### 无法识别 `conda`

打开 **Anaconda Prompt**、**Miniforge Prompt** 或其他已初始化 Conda 的终端。如果
Conda 已安装，但普通 shell 找不到它，请运行适合平台的 `conda init`，完全关闭终端后
重新打开。

### 环境不存在

在项目目录执行：

```bash
conda env create -f environment.yml
conda activate openmedvisionx
```

项目仅支持 **Python 3.11**。检查活动环境：

```bash
python --version
```

如果不是 Python 3.11，请停用当前环境并激活 `openmedvisionx`，不要把应用安装进无关
环境。

### 找不到 `openmedvisionx`

检查并修复可编辑安装：

```bash
conda activate openmedvisionx
python -m pip show openmedvisionx
python -m pip install -e .
openmedvisionx
```

也可以暂时在项目根目录运行：

```bash
python main.py
```

### 环境创建后项目发生了变化

更新可编辑安装及实际使用的可选能力：

```bash
conda activate openmedvisionx
python -m pip install -e .
```

只有在依赖修复失败时才重建环境；删除前先确认环境中没有其他工作。

### Linux 或远程 SSH 没有出现窗口

OpenMedVisionX 需要图形桌面、可用显示服务以及 PyQt5 所需 Qt 系统库。没有显示转发的
headless SSH 会话无法显示应用。请在本地图形会话中启动，或配置受支持的远程桌面/显示
环境。

### 文字或控件被截断

窗口最低要求是 900 × 620 逻辑像素。请最大化窗口。在高系统缩放下，使用更大显示器或
降低到常见缩放级别，然后重启让 Qt 重新计算布局。排查期间避免自定义非整数缩放。

## 打开影像与体数据

### NIfTI 提示缺少依赖

在活动环境安装可选加载器并重启：

```bash
conda activate openmedvisionx
python -m pip install -e ".[nifti]"
```

### 4D NIfTI 要求选择一个体

这是预期行为。请显式选择一个时间点/通道。对话框使用从 1 开始的编号；OpenMedVisionX
不会猜测目标三维体，当前版本也不提供完整 4D 播放。

### DICOM 文件夹中包含多个序列

在“**选择 DICOM 序列**”中选中一行，并检查模态、尺寸、帧数、几何状态与警告。
OpenMedVisionX 不会静默选择序列；不受支持的行保持不可用。

### DICOM 文件夹被拒绝

确认所选文件夹表示一个一致的影像序列：

- 单色影像，rows/columns 和 pixel spacing 一致；
- orientation 兼容，slice position 唯一且规则排列；
- 没有混合 Enhanced 多帧对象与独立单帧实例；
- 压缩 transfer syntax 所需 decoder 已安装（如需要）。

选择只包含目标序列的最小文件夹。OpenMedVisionX 不会递归地把相邻所有 DICOM 文件
当成一个 study。

### Enhanced 多帧对象被拒绝

只接受受支持、几何一致的单色 Enhanced CT/MR。彩色、不受支持的 functional group、
帧几何不一致，或多帧/单帧混合选择都会被拒绝。“Enhanced”不代表支持所有多帧对象。

### DICOM ZIP 被拒绝

归档必须未加密，并只含安全相对路径。成员数、展开大小或压缩比过大会被阻止，以防归档
滥用。请自行解压到受控本地目录，检查内容后使用“**打开 DICOM 文件夹**”。不要绕过
对不可信归档的检查。

### CT 没有显示 HU，或 PET 没有显示 SUV

只有 CT 模态与有效 rescale 证据支持时，OpenMedVisionX 才会标记 HU。PET SUV 同样
需要所需单位和校正证据；缺失时数值保持 arbitrary/unknown。不要因为图像看起来像 CT
或 PET 就手动重命名语义。

### TIFF 被当作页面而不是体数据

这是有意设计。页面顺序本身不提供 spacing、orientation 或 patient-space affine。请使用
页面控件；除非可信空间格式提供几何，否则不要解释正交解剖或毫米距离。

### 超大图看起来分辨率降低

查看器可能对超大平面图使用有界预览。测量或把活动影像用于模型前，请检查影像详情。
适用时模型运行还会要求明确的预览 override；如果预览像素不适合任务，请取消。

精确支持边界见[数据与格式](DATA_FORMATS.zh-CN.md)。

## 标注与图层

### 当前平面 mask 被拒绝

使用单页、无损、灰度 PNG 或 TIFF，并确保高度和宽度与当前活动平面完全一致。JPEG、
彩色、多页和 shape 不匹配的 mask 都会被拒绝。非零值形成叠加，应用不会 resize。

### 标注 JSON 被拒绝

确认 schema version 为 1、坐标系为 `pixel_xy_top_left`、`image_size` 精确匹配，并且
只使用受支持的 `boxes` 与 `points` 字段。删除外部路径和未知 key。示例见
[数据与格式](DATA_FORMATS.zh-CN.md#标注mask-与标签图层)。

### 无法导入 DICOM SEG 或 RTSTRUCT

先打开其引用的 DICOM 影像序列，再确认标注引用相同的 Series、Frame of Reference、
SOP 实例与兼容几何。RTSTRUCT 仅支持受支持的 closed planar contours，并保持为矢量
轮廓层。

### PNG、TIFF 或 NIfTI label map 保持隐藏

该图层正在等待几何决定。请从提供的操作中选择：

- 保留原始不匹配图层并隐藏；
- 取消导入；或
- 仅在科学上合理时，明确创建派生显示重采样图层。

OpenMedVisionX 绝不静默重采样。离散标签使用最近邻插值，原始源图层保持不变。JPEG
label map 始终被拒绝。

### 调整不透明度或标签可见性没有改变源数据

这是预期行为。可见性、不透明度、锁定和单标签开关属于呈现状态，不会编辑导入的源
数组。图层面板把 Study → Series → Layer 来源与显示选择分开。

## CT 实验

### 首个实验不需要影像

保持选择“**合成仿体（推荐）**”，然后选择“**运行首个仿体实验**”。当没有活动影像
或活动影像不兼容时，这条本地确定性路径始终可用。

### “活动影像（高级）”不可用

在“**影像**”中打开影像，单击目标平面并确认活动视图提示，然后返回 CT 实验。输入
必须是有限、非负、非常量、方形的二维灰度域，并与投影模拟兼容。应用不会静默转换
彩色、负 HU、不兼容支撑域或超大有界预览，也不会裁剪或重命名其语义。

### 高级流程有 sinogram，但没有重建

活动影像路径有意分为两步：先生成并检查投影数据，再选择“**2. 重建**”。首个仿体
按钮会自动运行两步。

### 出现圆形裁剪或支撑域警告

所选重建假设明确的圆形支撑域。请返回合成仿体，或使用符合文档支撑域的方形非负
衰减图。不要把普通临床 HU 切片当作扫描仪衰减物体。

### 指标与图像观感不一致

确认比较使用同一参考显示范围，且指标采用声明的原始数据范围。各自自动拉伸的图像可能
看起来相似，但误差不同。除了 PSNR/SSIM，还要比较 MAE/RMSE/bias 与误差图，并确认
几何和 ROI 一致。

## 模型

### PyTorch 不可用或没有选择 CUDA

按照[安装 PyTorch](MODEL_BUNDLES.zh-CN.md#安装-pytorch)为本机安装运行时，再检查环境：

```bash
conda activate openmedvisionx
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
python -m workbench.models --smoke --device auto
```

`auto` 在 PyTorch 报告 CUDA 可用时优先使用，否则记录 CPU 回退。普通 ONNX Runtime
安装同样不代表 GPU 可用。

### 内置模型文件缺失或哈希失败

运行不执行模型的完整性检查：

```bash
python -m workbench.models
```

不要编辑随包权重或登记哈希；请从同一版本的可信副本恢复精确文件。smoke 通过只证明
运行时/产物兼容，不能证明医学准确性。

### 外部清单有效，但“运行推理”被禁用

通用桌面路径目前只接受一个受支持的活动二维 `image` 输入，不能提供多模态、完整
3D/4D 体、prompt、k-space/mask、sinogram、时间序列或 WSI。请使用兼容模型或其任务
专用流程，不要篡改清单。详见[自定义本地模型](CUSTOM_MODELS.zh-CN.md)。

### ONNX 或 TorchScript 运行时不可用

ONNX：

```bash
python -m pip install -e ".[onnx]"
```

TorchScript 请安装兼容的 PyTorch。改变依赖后重启应用。

### Python adapter 要求信任确认

这是预期行为，因为它会执行第三方代码。检查源码、来源、哈希、环境、依赖与许可。
subprocess 不是安全沙箱；任何一项不清楚都应拒绝。

### MONAI BraTS 保持禁用

打开“**配置本地 BraTS 2021……**”，选择一个获准使用且包含唯一可识别 T1、T1ce、
T2、FLAIR 和 SEG NIfTI 的目录。它们必须是有限 3D 体，并具有匹配的 shape、spacing、
affine/orientation、world coverage 和可用 qform/sform；SEG 只能含整数标签 `{0,1,2,4}`。
桌面流程要求全部五个文件。完成本次会话校验后才可运行。应用不会下载、复制、重命名或
上传案例。

### BraTS 很慢或 GPU 内存不足

完整体滑窗推理明显重于两个合成演示。关闭其他 GPU 工作、确认所选设备，或改用 CPU
并预期显著更长的时间。不要为了强行运行而改变 patch 几何或 resize 案例；这会改变经过
审查的契约。

### DeepInverse 没有实验 JSON 按钮

这是当前限制，不代表运行失败。确定性合成 MRI 演示会显示 phantom、zero-filled IFFT、
模型幅值、误差与指标，但本版本不导出 JSON。DIVal LoDoPaB 与已验证 BraTS 工作流支持
无像素实验记录。

## 评价与导出

### 数据集清单被拒绝

使用 JSON 或 YAML，只含有界假名化 sample ID/hash，不包含影像路径。每个 sample 属于
一个 group，同一 group 不得跨 train、validation 与 test。继续前修复重复 ID、未知
split、无效 hash 或 group 泄漏。

### 二分类评价被拒绝

真值标签与预测概率必须具有相同数量的有限值。标签只能为 `0` 或 `1`，概率位于
`[0,1]`，并且两类都必须出现。支持逗号或换行分隔。请显式选择决策阈值。

### 指标为空或不符合预期

小样本或退化样本可能使部分区间/指标无定义。确认两类均存在并阅读警告。AUROC 不使用
所选工作阈值；敏感度、特异度、预测值、accuracy 与 F1 会使用。校准依赖概率值，不是
硬预测。

### 导出失败或询问已有文件

选择可写本地目录和新文件名，并遵循该导出的原生覆盖提示。不要覆盖源数据。影像导出是
渲染平面，不是视口截图；实验记录不含输入像素/向量，但仍可能含模型/任务元数据，共享
前请检查。

## AI 助手

### 无法解析凭据

字段只接受 `env:NAME`、`keyring:service/username`，或仅供精确回环服务使用的 `none`，
不接受原始密钥。在仓库外设置环境变量，或安装 `.[llm]` 并创建对应 keyring 条目。
环境改变后请重启应用。

### Endpoint 被拒绝

远程服务使用带 hostname 的 HTTPS。普通 HTTP 只允许精确 `localhost`、`127.0.0.1`
或 `::1`。删除内嵌凭据、query string 与 fragment。OpenAI-compatible 的
`example.invalid` 是占位值，必须替换。

### “发送”被禁用

确认以下条件全部成立：已选择 provider、填写精确 Model ID、endpoint 有效、凭据引用
存在、prompt 非空、“**启用网络**”打开且没有活动请求。OpenMedVisionX 不提供默认
Model ID。

### 图像附加被禁用

加载影像，单击目标活动平面，确认 provider 模型确实支持视觉，打开“**图像输入**”，
再选择“**附带活动渲染平面**”。DeepSeek 被当作纯文本。发送前仍需最终精确 PNG 检查
和一次性授权。

### 预览中没有叠加层或缩放

这是有意设计。待发送 PNG 是经过显示映射的完整活动二维平面，不包含视口缩放/平移、
测量、标注或叠加层。请检查最终确认预览，而不是视口，寻找烧录的私密文字。

### “结构化产物 · API 预览”始终为空

普通桌面会话中这是预期行为。教学对话与文件选择器不会填充它，本版本也没有桌面产物
导入器。只有可信宿主集成能够提供类型化请求/响应。本地确认/拒绝不会发送数据或创建
图层。

### 取消或撤销没有召回请求

取消可以停止仍在等待的工作，但不能召回已发送数据。完成传输后需使用服务商的删除/
事件流程。图像重试需要新的精确计划和授权。详见
[AI 助手、隐私与云端图像](LLM_SECURITY.zh-CN.md#取消撤回或改变决定)。

## 仍未解决？

请求帮助前，只记录不敏感的诊断事实：

- 操作系统与显示缩放；
- `python --version` 和安装方式；
- 精确工作区与操作；
- 完整且已脱敏的错误文本；
- 输入格式与不具身份信息的尺寸（不要附文件）；
- 可选运行时版本与请求/实际设备；
- 已尝试的检查。

只有报告不包含密钥或受限数据时，才在项目 [GitHub Issue
页面](https://github.com/CanyonChen/Open-Med-VisionX/issues/new/choose)搜索或新建公开报告。
安全或泄露问题请按照[安全策略](SECURITY.zh-CN.md)私下报告。

返回[文档首页](INDEX.zh-CN.md)。
