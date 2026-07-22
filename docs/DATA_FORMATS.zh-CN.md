# 数据与格式

[English](DATA_FORMATS.md) · [文档首页](INDEX.zh-CN.md) · [影像工作区](USER_GUIDE.zh-CN.md#影像)

本文帮助您在 OpenMedVisionX 中选择、准备、打开并正确理解数据。如果不确定该用哪个
按钮、一个文件会被当作页面还是体数据，或者标注与屏幕上的影像是什么关系，请从这里
开始。

> [!WARNING]
> 只使用您有权处理的数据。OpenMedVisionX 是学习与研究工具，不是医疗器械，也不是
> DICOM 匿名化工具。请保留每份源数据的原始备份，不得将任何输出用于患者相关决策。

## 选择数据前

第一次使用时，优先选择内置合成 CT phantom，或公开且不含敏感信息的栅格图像。使用
自己的数据前：

1. 确认许可证、知情同意和所在机构的规则允许您在本地使用这些数据。
2. 将患者数据、私有模型文件、凭据和导出结果放在源码仓库之外。
3. 选择满足任务所需的最小范围：一个文件、只含目标序列的目录或最小 ZIP，不要选择
   宽泛归档。
4. 保持原始文件不变。OpenMedVisionX 只读取您选择的路径，并把显示状态或派生结果
   分开管理。
5. 先了解数值语义。记录数据说明中的 modality、单位、spacing、orientation 和任何
   预处理。
6. 同时检查元数据和可见像素中的标识信息。删除文件名或某个元数据字段，并不能去掉
   固化在图像中的文字。

OpenMedVisionX 会校验实际内容，而不是只相信文件扩展名。因此，即使后缀看起来正确，
经过改名、损坏、含糊、过大或不受支持的文件仍可能被拒绝。

## 支持矩阵

| 输入 | 打开方式 | 安装要求 | 解释方式 | 重要边界 |
| --- | --- | --- | --- | --- |
| PNG 或 JPEG 图像 | **打开影像 / DICOM ZIP** | 基础安装 | 一张二维栅格图像 | 不推断医学 spacing、modality 或 HU。JPEG 有损，不能作为 label map。 |
| TIFF 图像 | **打开影像 / DICOM ZIP** | 基础安装 | 一张二维图像或相互独立的页面序列 | 页面顺序与 DPI 不能建立物理体数据；超大平面图可能使用受限预览。 |
| 单个 DICOM 影像文件 | **打开影像 / DICOM ZIP** | 基础安装 | 经过校验的影像或受支持的多帧对象 | 内容、像素数据、modality 和几何必须受支持；仅有 `.dcm` 后缀不能证明文件有效。 |
| DICOM 目录 | **打开 DICOM 文件夹** | 基础安装 | 用户明确选择的一个影像序列 | 如果发现多个序列，应在检查安全摘要后选择目标序列；不要选择宽泛患者归档。 |
| DICOM ZIP | **打开影像 / DICOM ZIP** | 基础安装 | 从受限归档中选择的一个序列 | 成员路径、数量、大小、压缩比和解码后大小必须通过安全限制。 |
| NIfTI `.nii` 或 `.nii.gz` | **打开影像 / DICOM ZIP** | 可选 `nifti` 依赖 | 内部以 RAS+ 表示的空间体数据 | 4D 文件必须显式选择时间点/通道；除非可信工作流声明，否则强度语义未知。 |
| 当前平面 mask（PNG 或 TIFF） | **掩膜…** | 基础安装 | 只作用于一个可见平面的快速二值叠加 | 必须无损、灰度、单页，并与当前平面的宽高完全一致；非零值成为叠加。 |
| 当前平面标注 JSON | **标注…** | 基础安装 | 当前平面左上角像素坐标中的 box 与 point | 必须使用 OpenMedVisionX schema、匹配当前图像尺寸，并且不包含外部引用。 |
| DICOM SEG 或 RTSTRUCT | 打开其 DICOM 引用后选择 **DICOM SEG / RTSTRUCT…** | 基础安装 | 活动 study 中不可变的分割层或轮廓层 | 所引用的 series、Frame of Reference、SOP instance 与几何必须匹配；RTSTRUCT 保持矢量轮廓。 |
| PNG 或无损 TIFF label map | 选择引用 series 后使用 **标签图…** | 基础安装 | 不可变的离散分割层 | 数值必须是有限非负整数；拒绝 JPEG、彩色 mask 与连续值 map。 |
| NIfTI label map | 选择引用 series 后使用 **标签图…** | 可选 `nifti` 依赖 | 带自身 affine 的空间离散分割层 | 4D label map 必须显式选择一个体；几何绝不会被静默强制匹配。 |

本矩阵描述当前桌面工作流，并不代表底层库理论上能够解码的所有格式。不受支持的采集
对象、压缩归档、时序工作流或标注变体会被拒绝，而不是通过猜测进行解释。

## 打开 DICOM 数据

### 选择文件、文件夹或 ZIP

使用**打开影像 / DICOM ZIP**选择一个 DICOM 文件或经过有意整理的 DICOM ZIP；使用
**打开 DICOM 文件夹**选择包含目标序列的本地目录。

- 单个文件可以是一张经典影像，也可以是一个受支持的 Enhanced 多帧对象。
- 一个文件夹可以包含一个或多个 series。OpenMedVisionX 会检查所选位置并让您选择
  series，但不会递归扫描无关的相邻目录。
- ZIP 应只包含目标 series 所需的数据。使用前会检查不安全成员路径、过多成员、过大
  展开体积和异常压缩比。

不要把非 DICOM 文件改名为 `.dcm`，不要根据文件名重建切片顺序，也不要为了打开未知
输入而关闭归档限制。

### 选择目标序列

发现多个 series 时，选择器会显示 modality、尺寸、instance 或 frame 数等安全摘要。
继续前，请把这些信息与数据集文档对照。

只有能够确认目标 series 时才继续。如果多个序列含糊、预期 instance 缺失，或显示的
modality 与尺寸不符合任务，请取消。选错 series 仍可能得到看起来合理的图像，因此
不能只凭视觉外观判断。

### 理解 Enhanced 多帧边界

受支持的 Enhanced 路径是 shared 或 per-frame functional group 中具有一致几何的
单色 CT 或 MR。遇到以下情况时，OpenMedVisionX 会拒绝，而不是猜测：

- 缺少 image orientation 或 position；
- frame spacing 不一致或不均匀；
- frame 数或解码大小超过安全限制；
- 不受支持的彩色像素数据；
- 一个多帧 instance 与多份经典影像 instance 混合。

Enhanced 对象被拒绝时，请保留原文件并请求符合标准的导出。不要根据文件名或屏幕顺序
虚构几何。

### 加载后检查 DICOM 语义

确认应用显示的 series、modality、尺寸、spacing、orientation、frame 数和强度标签。
DICOM 元数据可以描述患者空间几何，但应用只使用经过校验的证据：

- 只有有效 rescale 证据支持时，CT 数值才标记为 Hounsfield unit。
- 只有具备所需单位、SUV 类型和校正证据时，PET 数值才标记为 SUV。
- 缺失或不一致的几何会阻止物理体数据操作，不会被静默修复。

应用内部的体数据坐标约定为 RAS+。DICOM 患者坐标通常使用 LPS；OpenMedVisionX 会
执行经过校验的坐标转换，但不会改变源文件。

## NIfTI 体数据

在已激活的项目环境中安装可选加载器，然后重启应用：

```bash
conda activate openmedvisionx
python -m pip install -e ".[nifti]"
```

使用**打开影像 / DICOM ZIP**选择 `.nii` 或 `.nii.gz`。加载后，在测量或比较结果前
检查 shape、voxel spacing、orientation、affine 和 intensity meaning。

### 三维 NIfTI

三维 NIfTI 被解释为空间体数据，并统一到应用内部的 RAS+ 约定。其 affine 负责连接
voxel index 与物理世界坐标。有效 affine 本身不能证明 modality、采集协议或强度单位。

### 四维 NIfTI

OpenMedVisionX 不会猜测第四轴代表时间、通道、echo 还是其他维度。因此选择对话框会
要求指定一个时间点或通道，并在 provenance 中记录从 0 开始的索引。桌面随后处理
所选三维体；这不等于完整 4D 播放或分析。

如果数据集文档没有解释第四轴，请取消。4D NIfTI label map 遵循相同规则。

### NIfTI 强度与几何注意事项

- 除非可信工作流明确声明，否则 intensity semantics 保持未知；不要把一般 NIfTI
  数值称为 HU。
- 把两个文件当作对齐数据前，应检查 qform/sform 警告和 affine orientation。
- 相同 array shape 不能证明物理几何相同。
- 转换到内部坐标约定，不代表可以隐式重采样不匹配的 label map。

## 打开栅格图像、TIFF 页面与大图

PNG、JPEG 和 TIFF 适合学习像素、display mapping、直方图、测量以及兼容的二维模型
输入，但不会自动获得医学语义。

### PNG 与 JPEG

- PNG 可以保留无损灰度或彩色样本。
- JPEG 是有损格式：即使外观看起来可以接受，压缩也可能改变边缘和细小结构。
- 两种格式都不能建立 modality、物理 spacing 或 HU。
- EXIF orientation 与颜色/alpha 处理属于明确变换；与其他文件比较坐标前，请阅读
  影像详情。

只有从可信来源获得真实物理 pixel spacing 时才使用**设置像素间距…**。OpenMedVisionX
会把该值记录为“用户提供”。DPI 是打印属性，不是医学 pixel spacing。

### TIFF 页面不会自动成为体数据

除非可信空间元数据另有证明，多页 TIFF 会作为页面序列导航。页面顺序、相同页面尺寸
和 DPI 都不能证明 slice spacing、orientation、origin 或患者坐标系。需要物理正交
视图时，应使用经过校验的 DICOM 或 NIfTI 体数据。

### 超大平面图可能使用受限预览

为了限制内存占用，超大平面栅格可能从缩小预览显示，而不是把完整解码页面保留在内存。
测量或把活动影像用于模型前，请在影像详情中检查预览尺寸和记录的变换。只有记录的
模型输入可以证明时，才能认为模型接收了原始分辨率像素。

## 标注、mask 与标签图层

OpenMedVisionX 提供两套相互分开的标注生命周期。请根据任务选择：

| 需求 | 使用功能 | 作用范围与生命周期 | 几何行为 |
| --- | --- | --- | --- |
| 快速把简单 mask、box 或 point 与当前屏幕平面对照 | **掩膜…**或**标注…** | 绑定到一个当前平面或页面；返回该平面时重新显示，选择**清除标记**或开始新 study 后清除；它不是临床 Study layer。 | 只接受精确像素尺寸匹配；不 resize、不重采样。 |
| 检查与影像 series 建立引用关系且可复用的分割或轮廓对象 | **DICOM SEG / RTSTRUCT…**或**标签图…** | 连同 provenance 和独立 presentation state 导入不可变的 Study → Series → Layer 层级。 | 校验身份与物理几何；错配绝不静默叠加。 |

简单配对不会把 mask 或 JSON 转换为 DICOM SEG、RTSTRUCT 或体 label map；导入临床
图层也不会把它展平为临时当前平面叠加。

### 关联简单的当前平面 mask

1. 打开目标图像、页面或切片，再单击对应视图，使其成为活动视图。
2. 选择**掩膜…**，把本地 mask 与当前平面配对。
3. 选择与活动平面宽高完全一致的灰度单页 PNG 或无损 TIFF。
4. 检查配对状态。所有非零像素显示为一个二值叠加，源数值不会改变。

RGB/RGBA mask、JPEG mask、多页 TIFF mask 和尺寸不匹配都会被拒绝。配对不会 resize
mask，也不会推断物理坐标。

### 关联简单的标注 JSON

让目标平面保持活动状态，再选择**标注…**。JSON 文件必须是 UTF-8、自包含、最大
4 MiB，并使用左上角像素坐标。`image_size`
采用 `[width, height]`，必须与活动平面一致。Box 的 `(x1, y1)` 是左上角，
`(x2, y2)` 是右下边界；所有 point 与 box 都必须位于图像内部。

最小示例：

```json
{
  "schema_version": 1,
  "coordinate_system": "pixel_xy_top_left",
  "image_size": [512, 512],
  "boxes": [
    {"x1": 20, "y1": 30, "x2": 120, "y2": 160, "label": "example"}
  ],
  "points": [
    {"x": 64, "y": 80, "label": "landmark"}
  ]
}
```

至少需要一个 box 或 point。未知顶层字段和外部文件引用会被拒绝；加载标注不会继续
访问其他路径。该格式用于受限教学叠加，不是通用标注交换标准。

### 导入 DICOM SEG 或 RTSTRUCT

先打开其引用的 DICOM 影像 series，再选择 **DICOM SEG / RTSTRUCT…**。添加图层前，
导入器会检查所引用的 Series Instance UID、Frame of Reference UID、SOP instance
与几何。

- Binary 与 fractional DICOM SEG 保留原生表示。只有用户明确确认派生显示重采样后，
  fractional layer 才会为该显示派生使用连续插值。
- RTSTRUCT closed planar contour 会从 DICOM LPS 转换到内部 RAS+ 以便显示；始终保持
  矢量轮廓，不会栅格化。
- 引用其他 series 的对象会被拒绝，不会附着到碰巧处于活动状态的图像。

### 导入 label map

打开并选中目标引用 series，再选择**标签图…**。接受的 map 必须包含有限、非负的
离散整数标签：

- 一张 PNG 或可以明确确认为无损的 TIFF；或
- 安装 `nifti` 可选依赖后的 NIfTI。

JPEG、彩色 mask、负值标签、非有限值和连续值 map 会被拒绝。4D NIfTI 必须显式选择
一个体。栅格 map 使用用户选择的引用轴；NIfTI 保留自身 canonical RAS+ affine 用于
几何比较。

### 显式处理几何错配

栅格 SEG 或 label map 与引用网格不匹配时，请在应用提供的选项中选择：

- 取消导入；
- 将原始对象保留为隐藏的未匹配图层；
- 或者在提供该选项时，明确确认一个单独的派生显示层。

原始导入及其几何始终不变。离散标签使用最近邻插值，fractional value 需要连续插值。
可见性、透明度、锁定状态和逐标签可见性只属于 presentation，不会改写原始数组或
派生数组。

## 几何、强度与显示边界

测量、叠加、重建或运行模型前，请使用这张检查表：

| 检查项 | 需要确认 | OpenMedVisionX 不会做的假设 |
| --- | --- | --- |
| 对象类型 | 二维栅格、页面序列或物理体数据 | 多个页面不会自动变成体数据。 |
| Shape | 宽、高、slice/frame、channel | 相同 shape 不能证明几何相同。 |
| 几何 | Spacing、origin、orientation/direction、affine、坐标约定 | DPI 不是医学 spacing；只有 array axis 不能证明 axial/coronal/sagittal。 |
| 强度 | dtype、数值范围、modality、单位、rescale、normalization | 灰度不会自动成为 HU；不推断 NIfTI 强度。 |
| 活动输入 | plane、slice/page index，以及可用时的 RAS+ 坐标 | 下游任务不会静默选择另一个平面。 |
| 变换 | EXIF orientation、canonicalization、crop、resize 或已确认重采样 | 显示变换不会改变源数组。 |

**Display mapping（显示映射）**控制存储数值怎样变成可见亮度或颜色。Window、range、
brightness、contrast、gamma 和 inversion 可以让相同源数值看起来不同；反过来，分别
自动拉伸也可能让数值不同的数组看起来相似。请比较数值和几何，不要只比较外观。

## 在不改变源数据的前提下导出

在“影像”工作区中：

- **导出 PNG…**根据当前 display mapping 从完整当前二维平面创建新 PNG。它不会替换
  源数组，也不包含 viewport zoom、pan、配对叠加、标注或临床图层。
- **保存记录…**创建不含像素的实验 JSON，保存影像摘要、显示参数、数值测量与配对
  状态，不嵌入影像像素、私有路径或患者元数据。

两项操作都写入用户选择的新目标，并拒绝覆盖现有文件。渲染 PNG 适合交流或教学，
但它不是 DICOM 导出、可复用分割对象，也不能证明强度与几何语义得到保留。

## 打开或分享数据前的隐私检查

打开数据前：

- 优先使用合成、公开或经机构批准的资料；
- 将源数据和导出结果保存在源码仓库之外；
- 只选择所需文件、series 目录或最小归档；
- 记住 DICOM 元数据和文件名都可能包含标识；
- 检查可见像素中的姓名、日期、accession number、机构标签或其他固化文字。

本地影像加载、导航、测量、标注导入和导出本身不会联系 AI provider。OpenMedVisionX
不会自动上传数据或下载缺失模型。但应用不是匿名化工具：界面中的安全摘要与元数据
过滤不能证明源文件或截图已经完成去标识。

分享导出文件前，请再次检查实际文件。使用 AI 助手前，请阅读
[AI 助手、隐私与云端图像](LLM_SECURITY.zh-CN.md)：文本同样可能包含敏感信息，而发送
渲染图像需要另行完成精确的一次性请求检查。

如果文件被拒绝，请保持原文件不变并查看
[故障排查](TROUBLESHOOTING.zh-CN.md#打开影像与体数据)，不要削弱安全或几何检查。

## 下一步

- 操作查看器：[用户指南 · 影像](USER_GUIDE.zh-CN.md#影像)
- 理解基础概念：[术语表](GLOSSARY.zh-CN.md)
- 练习像素与几何：[学习课程](TEACHING_CURRICULUM.zh-CN.md#1-像素dtype-与显示)
- 解决加载问题：[故障排查 · 打开影像与体数据](TROUBLESHOOTING.zh-CN.md#打开影像与体数据)

下一步：[学习“影像”工作区](USER_GUIDE.zh-CN.md#影像)。
