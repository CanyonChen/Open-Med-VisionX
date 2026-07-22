# OpenMedVisionX 术语表

[English](GLOSSARY.md) · [文档首页](INDEX.zh-CN.md) · [学习课程](TEACHING_CURRICULUM.zh-CN.md)

本术语表解释 OpenMedVisionX 界面和用户指南中出现的概念。这里提供的是实用解释，而非
百科全书式定义：每一项都聚焦初学用户在信任影像、比较、模型输出、指标或 AI 辅助解释
前需要确认的内容。

> [!NOTE]
> 熟悉的名称不等于熟悉的语义。解释术语时，请始终结合当前数据来源、几何、任务、参数
> 与 provenance（来源记录）。

## 如何使用本术语表

请在按主题组织的表格中查找陌生术语，再前往相应指南完成整个工作流。必要时用**粗体**
表示界面标签；缩写会在第一次出现时给出全称。

| 如果您想问…… | 从这些术语开始…… |
| --- | --- |
| “下一个操作会使用哪张图或哪个切片？” | **活动视图**、**平面**与 **provenance** |
| “这些页面是否构成物理体数据？” | **Voxel**、**spacing**、**orientation** 与 **affine** |
| “数据没变，为什么画面变了？” | **Display mapping** 与 **windowing** |
| “CT 实验显示的是什么？” | **Sinogram**、**FBP** 与 **SART** |
| “这个模型能接收我的文件吗？” | **输入契约**、**manifest** 与**能力门禁** |
| “这个分数能否证明模型很好？” | **评价单位**、**数据泄漏**、**校准**与具体指标 |
| “究竟有哪些内容可能离开我的电脑？” | **信任边界**、**渲染预览**、**传输计划**与**固化文字** |

## 界面与工作流

| 术语 | 通俗解释 | 为什么重要 |
| --- | --- | --- |
| **工作区（workspace）** | 六个顶层区域之一：影像、CT 实验、模型、教学、评价或 AI 助手。 | 每个工作区有不同的输入与证据边界；切换页面不会把一种数据变成另一种。 |
| **活动视图（active view）** | 最近被选中、供下游操作使用的影像视图，包含明确平面及当前 slice/page。 | CT 实验的高级模式、兼容模型和助手会使用它；操作前应单击目标视图并阅读上下文行。 |
| **平面（plane）** | 一个可见二维截面或图像，例如 axial、coronal、sagittal、TIFF 页面或平面栅格图。 | 一个平面不等于完整体数据；当前平面 mask 和通用外部模型只作用于一个平面。 |
| **切片（slice）** | 从空间体数据取得的一个二维截面。 | 只有体数据几何有效时，它的物理位置才有意义。 |
| **页面 / 帧（page / frame）** | 二维序列中的一个项目，例如一页 TIFF。 | 序列顺序本身不能证明三维 spacing 或患者坐标。 |
| **Study → Series → Layer** | 查看器层级：study 包含影像 series，series 包含源图层和派生图层。 | 它让影像、分割、轮廓、显示状态和 provenance 保持关联，而不改写源数据。 |
| **图层（layer）** | 随 series 显示的一个源对象或派生对象，例如 volume、segmentation 或 contour set。 | 可见性与透明度可以独立变化；图层不会自动融合进源影像。 |
| **源数据（source data）** | 从用户选择的文件或 series 解码得到的数据。 | 显示调整、叠加和派生重采样不会替换它；应保留原始文件及其许可。 |
| **派生结果 / 派生图层（derived result / layer）** | 根据源数据计算出的新结果，例如 reconstruction 或明确重采样后的显示层。 | 必须保留 transform、参数与 provenance，避免被误认为原始数据。 |
| **显示状态（presentation state）** | 可见性、透明度、锁定、标签可见性、window 等影响显示效果的选择。 | 它改变视图，不改变底层源数值。 |
| **来源记录（provenance）** | 记录数据从何而来，以及哪些选择、版本、变换或设备产生了结果。 | 没有 provenance，其他用户无法复现结果或发现不匹配。 |
| **能力门禁（capability gate）** | 操作启用前显示的兼容性检查。 | 它阻止不受支持的影像、维度、modality、spacing、prompt 或复合输入被伪装成有效输入。 |
| **实验记录（experiment record）** | 保存任务上下文、参数、hash、指标、警告与运行事实的 JSON/YAML。 | OpenMedVisionX 记录有意不含像素与路径；它支持复现，但不是源数据。 |
| **本地优先（local-first）** | 查看与核心学习流程使用用户选择的本地数据，不自动上传或下载。 | 网络使用是另一次明确操作；“本地优先”不代表永远无法发起云端请求。 |
| **不可变（immutable）** | 创建后不再修改；新的解释或显示状态会形成新记录或新状态。 | 这样可以追踪源证据与派生证据，并避免静默覆盖。 |

## 影像数据与几何

| 术语 | 通俗解释 | 为什么重要 |
| --- | --- | --- |
| **像素（pixel）** | 二维栅格中位于 `(x, y)` 的一个样本。 | 除非掌握可信 spacing，否则像素没有物理尺寸。 |
| **体素（voxel）** | 三维体数据中位于 `(x, y, z)` index 的一个样本。 | 毫米测量需要 voxel 到世界坐标的几何，不能只看 array index。 |
| **栅格图像（raster image）** | 由矩形像素网格构成的图像，例如 PNG、JPEG 或一页 TIFF。 | 通常使用像素坐标解释，而不是患者空间。 |
| **页面序列（page sequence）** | 多个相互独立的二维页面或 frame。 | 尺寸相同且按顺序排列，也不能建立物理体数据。 |
| **体数据（volume）** | 具有经过校验空间几何的三维数组。 | 可以支持物理测量、世界坐标和正交视图。 |
| **Shape** | Array 每个维度上的样本数量。 | Shape 相同不能证明两幅影像覆盖相同物理空间。 |
| **dtype** | 数值存储类型，例如 `uint8`、`int16` 或 `float32`。 | 它影响范围与精度；屏幕上的 8-bit 显示不一定是源 dtype。 |
| **位深（bit depth）** | 表示一个样本或 channel 使用的 bit 数。 | 更高位深可能保留 8-bit 渲染中消失的差异。 |
| **通道（channel）** | 一个样本的一个分量，例如红/绿/蓝，或一个 MRI modality/时间点。 | Channel 顺序与语义必须符合任务；复制或猜测通道不是有效预处理。 |
| **模态（modality）** | CT、MR、PET 等采集类型。 | 它约束强度语义、几何、预处理和模型兼容性；不能仅凭文件名安全确定。 |
| **间距（spacing / pixel spacing / voxel spacing）** | 相邻样本之间的物理距离，通常以毫米计。 | 物理长度、面积、体积和表面距离指标都需要它；DPI 不是医学 spacing。 |
| **原点（origin）** | 分配给某个参考 voxel 的世界坐标位置。 | 即使 spacing 和 shape 相同，origin 不同也可能描述不同位置。 |
| **方向（orientation / direction）** | Array 各轴在物理空间中的指向。 | Array axis 0 不会自动成为 axial；只有方向有效时才能正确标记平面。 |
| **Affine** | 通常是一个 `4×4` 矩阵，把 voxel index 映射到世界坐标，并包含 spacing、orientation 与 position。 | 两份 NIfTI 只有在 affine 和网格一致时才物理对齐，不能只比较 shape。 |
| **世界坐标（world coordinate）** | 按某种坐标约定表达的物理位置，通常以毫米而不是 array index 表示。 | 它能把不同视图和兼容图层中的同一解剖位置联系起来。 |
| **RAS+** | 正方向分别指向 Right、Anterior、Superior 的坐标约定；OpenMedVisionX 内部体数据使用它。 | 它提供统一内部坐标；转换到 RAS+ 不改变源文件，也不能证明两个 volume 已对齐。 |
| **LPS** | 正方向分别指向 Left、Posterior、Superior 的坐标约定；DICOM 患者坐标常用它。 | DICOM contour 与几何必须在 LPS 和内部 RAS+ 之间一致转换。 |
| **规范化方向（canonicalization）** | 在记录映射的同时，用选定坐标约定（例如 RAS+）重新表达影像。 | 它统一导航方式，但不会编造缺失几何，也不允许任意叠加。 |
| **几何匹配（geometry match）** | Shape/grid、spacing、orientation、origin、affine 及相关引用身份一致。 | 只有经过校验的关系才适合叠加 segmentation。 |
| **重采样（resampling）** | 在另一个网格上重新计算数值。 | 它会创建派生数据，也可能移动边界或改变数值；必须显式执行，不能静默发生。 |
| **插值（interpolation）** | 重采样时估计数值所采用的规则。 | 离散标签适合 nearest-neighbour；连续插值用于 fractional value，不用于 class ID。 |
| **ROI（region of interest）** | 用户选中、用于检查或测量的影像区域。 | ROI 只限制分析范围，不会自动成为解剖标签或 ground truth。 |
| **Mask** | 通常是一个二值数组，用来标记包含或不包含的 pixel/voxel。 | 当前平面 mask 是简单叠加，不会自动成为可复用临床分割。 |
| **Label map** | 用整数值表示背景和一个或多个类别的数组。 | Label 需要 schema、匹配几何，以及明确重采样时的最近邻处理。 |
| **轮廓（contour）** | 描述边界的矢量路径，不是填充后的栅格网格。 | RTSTRUCT contour 保持矢量数据，不应被静默栅格化。 |
| **DICOM** | 医学影像及相关对象的一组标准，包含元数据和患者空间引用。 | 同一文件夹中的文件不一定属于同一个 series；身份、几何和像素需要校验。 |
| **Enhanced 多帧 DICOM** | 在一个 DICOM 对象中包含多个 frame，并使用 shared/per-frame geometry。 | OpenMedVisionX 支持几何一致且受安全限制的单色 CT/MR，不支持所有 Enhanced 对象。 |
| **NIfTI** | 常用于神经影像数组并附带 affine 的文件格式。 | 4D NIfTI 必须显式选择一个体；格式本身不能保证强度语义。 |
| **DICOM SEG** | 引用源 DICOM 影像的 DICOM 分割对象。 | 必须保留并校验 binary/fractional value 和引用身份。 |
| **RTSTRUCT** | 含有命名患者空间 contour 的 DICOM 放疗结构集。 | 必须引用目标 series，并在 OpenMedVisionX 中保持矢量几何。 |

各格式的准备与导入方式见[数据与格式](DATA_FORMATS.zh-CN.md)。

## 强度与显示

| 术语 | 通俗解释 | 为什么重要 |
| --- | --- | --- |
| **强度（intensity）** | 存储在 pixel 或 voxel 中的数值。 | 其含义取决于 modality 与预处理；“更亮”不是单位。 |
| **强度语义（intensity semantics）** | 对数值的声明解释，例如 HU、归一化衰减值或未知。 | 证据不足时，OpenMedVisionX 会保持未知，而不是猜测。 |
| **Hounsfield unit（HU）** | 以水和空气为参照的校准 CT 标度。 | 一般灰度图或归一化 benchmark array 不是 HU；需要有效 DICOM rescale 证据。 |
| **SUV（standardized uptake value）** | 依赖单位与校正元数据的 PET 摄取度量。 | 缺少必要证据时，PET pixel 不会被标记为 SUV。 |
| **显示映射（display mapping）** | 把源数值转换为可见亮度或颜色的过程。 | 它改变看到的效果，不改变源数组。导出与 AI preview 使用映射后的平面，因此需要明确检查。 |
| **Windowing / window level 与 width** | 选择一个中心与数值区间，并映射到可见灰度范围。 | Window 外的结构可能显示为全黑或全白，但并未从源数据中消失。 |
| **显示范围（display range）** | 映射到可见输出范围的 lower/upper value。 | 比较 reference 与 result 时应使用相同范围；各自 auto-range 会误导。 |
| **Brightness / contrast / gamma** | 对可见数值进行平移、拉伸或非线性映射的显示控制。 | 它们影响外观，不应被误认为对源数据执行了预处理。 |
| **直方图（histogram）** | 统计各数值出现频率。 | 可帮助发现 clipping、窄范围和 outlier，但不提供空间位置。 |
| **归一化（normalization）** | 有文档说明的数值变换，例如缩放到某区间或标准化一个 channel。 | 模型要求精确的训练时约定；方便观察的视觉归一化可能在科学上错误。 |
| **z-score normalization** | 减去 mean 再除以 standard deviation，常在声明的 mask 内计算。 | 计算 mean/std 所用的人群或区域属于模型契约。 |
| **无损 / 有损（lossless / lossy）** | 无损编码保留解码后的样本值；JPEG 等有损编码可能改变数值。 | 有损文件不适合离散标签，并可能改变小结构或测量。 |
| **元数据（metadata）** | 像素之外的信息，例如尺寸、几何、采集字段或标识。 | 元数据对解释可能必不可少，也可能包含敏感信息。 |
| **固化文字（burned-in text）** | 直接写入图像像素的文字，例如姓名、日期、accession number 或机构标签。 | 删除 metadata 无法去掉它；导出或传到云端前必须检查实际渲染图。 |

## CT 投影与重建

| 术语 | 通俗解释 | 为什么重要 |
| --- | --- | --- |
| **衰减图（attenuation map）** | 描述物质对 X 射线衰减强度的空间图。 | CT 实验的合成 phantom 使用明确的非负教学域；任意图像不会自动成为衰减图。 |
| **Phantom（仿体）** | 具有已知结构的合成或物理测试对象。 | 内置 phantom 为比较重建设置提供隐私安全、可复现的 reference。 |
| **投影（projection）** | 在教学模型中，从一个角度取得的物体线积分测量。 | 对重建图像进行数学投影，不等于恢复原始扫描仪采集。 |
| **Radon transform** | 在多个角度形成理想化线积分投影的数学操作。 | 它是 CT 实验 sinogram 的 forward model，其假设与真实扫描仪不同。 |
| **Sinogram（正弦图）** | 按 detector position 和 projection angle 排列的投影值。 | 它属于 measurement domain，不是 CT 图像；临床切片不能被重新命名为 sinogram。 |
| **角度采样（angular sampling）** | 投影角度的数量与范围。 | 角度过少或分布不合适会产生 streak 和信息缺失。 |
| **探测器 bin（detector bin）** | 一次投影中一个被采样的 detector position。 | Detector 与 angle sampling 会共同影响可恢复细节。 |
| **反投影（backprojection）** | 沿采集方向把每份 projection 展开回 image space。 | 未滤波反投影会模糊，可以用来理解为什么需要滤波或迭代校正。 |
| **FBP（filtered backprojection，滤波反投影）** | 先对每份 projection 滤波，再反投影形成解析重建。 | Filter 和 geometry 选择会显著影响锐度、噪声与 artifact。 |
| **直接傅里叶重建（direct Fourier reconstruction）** | 利用 projection 与 image 的 Fourier-domain 关系完成重建。 | 它提供另一条解析路径，并有自己的 interpolation/sampling 假设。 |
| **SART（simultaneous algebraic reconstruction technique）** | 利用测得 projection 与预测 projection 的差异反复更新 image 的迭代方法。 | Iteration 数、geometry、noise 与停止行为都会影响质量和耗时。 |
| **迭代（iteration）** | 迭代算法的一次重复更新。 | 更多 iteration 不保证结果更好或更有临床意义；每次只改变一个设置。 |
| **重建滤波器（reconstruction filter）** | FBP 使用的频率权重，例如 ramp 及其平滑变体。 | 更平滑的 filter 可减小噪声但模糊细节；比较时应固定显示范围。 |
| **支持区域（support）** | 假定重建对象存在的区域；教学几何中常为圆形。 | 违反 support 假设可能产生裁剪或警告。 |
| **参考 / ground truth** | 受控比较时使用的目标。 | 合成或 benchmark reference 只支持声明的实验，不能证明临床真值或泛化。 |
| **误差图（error map）** | 结果与 reference 差异的空间视图。 | 它显示误差发生位置，而单一汇总指标可能隐藏这些位置。 |

## 模型与推理

| 术语 | 通俗解释 | 为什么重要 |
| --- | --- | --- |
| **模型（model）** | 把声明输入映射为声明输出的学习型或固定计算。 | 只有模型名称不能定义 modality、预处理、标签或有效用途。 |
| **权重 / checkpoint（weights / checkpoint）** | 存储的学习参数；有些格式还包含其他运行状态。 | 来源、hash、序列化安全与许可都很重要；`.pt` 后缀不能证明它是安全 TorchScript。 |
| **推理（inference）** | 运行训练完成的模型得到输出，不是训练。 | 成功运行只证明 runtime 兼容，不能证明医学准确性。 |
| **预处理（preprocessing）** | 推理前按顺序执行的变换，例如 orientation、resize、channel order 和 normalization。 | 必须符合模型契约并被记录；为了匹配 shape 而擅自改变会使结果失效。 |
| **后处理（postprocessing）** | 把原始模型 tensor 转换为 score、label、mask、box 或 image 的声明步骤。 | Threshold、activation、interpolation 和坐标恢复都会改变输出语义。 |
| **输入/输出契约（input/output contract）** | 对必需输入和返回输出语义的完整规定。 | 运行前应检查 modality、dimension、channel、spacing、range、label、coordinate 与限制。 |
| **Manifest（清单）** | 描述用户模型 runtime、文件、许可、输入/输出契约和 capability 的本地机器可读文件。 | Manifest 有效只表示描述有效，不保证当前桌面能收集所有必需输入。 |
| **运行时（runtime）** | 执行模型的软件，例如 ONNX Runtime 或 PyTorch。 | 必须在本地安装，并与模型和设备兼容；OpenMedVisionX 不会自动安装。 |
| **ONNX** | 常由 ONNX Runtime 执行的可移植模型图格式。 | 仍需正确契约与兼容 operator；文件本身不会说明 preprocessing。 |
| **TorchScript** | 设计为无需原始 Python model class 即可运行的序列化 PyTorch program。 | 它不同于任意 pickle checkpoint，但仍需可信来源、完整性验证和兼容 PyTorch runtime。 |
| **Python adapter** | 用户提供、负责准备并运行模型的 Python 代码。 | 它可以执行任意代码；独立进程只提供依赖/故障隔离，不是安全沙箱。 |
| **随包离线参考（bundled offline reference）** | OpenMedVisionX 随包提供的固定、经过审查的模型资产之一。 | 它具有窄范围任务工作流、完整性记录、模型卡和输入边界，不是通用 predictor。 |
| **外部模型（external model）** | 用户提供的本地模型与 manifest。 | 通用桌面当前只接收一个受支持的活动二维输入，不支持 manifest 能描述的所有任务。 |
| **任务（task）** | Classification、segmentation、detection、registration 或 reconstruction 等问题类型。 | 指标和输出与任务相关；classification score 不是 segmentation mask。 |
| **分类（classification）** | 为 image、volume 或其他声明单位产生 class score 或 probability。 | 必须明确评价单位、threshold、calibration 与 class definition。 |
| **分割（segmentation）** | 为 pixel/voxel 分配 class 或 probability。 | 除 Dice 外，还要考虑 geometry、label schema、boundary error 与小结构。 |
| **检测（detection）** | 以 box、point 或相关位置及 class/score 找到对象。 | Matching rule、IoU threshold 与评价单位会决定报告的 performance。 |
| **配准（registration）** | 估计 transform，使 moving data 与 fixed data 对齐。 | 应检查 landmark、overlap、folding、inverse consistency 与 interpolation；看起来对齐并不充分。 |
| **重建模型（reconstruction model）** | 把 acquisition-domain measurement 或声明的中间表示映射成 image。 | 必须保留采集假设；普通图像不能代替 k-space 或 sinogram。 |
| **CUDA / CPU** | Runtime 使用的 GPU 与处理器执行路径。 | `auto` 可优先 CUDA 并记录 CPU fallback。设备影响速度和部分数值容差，不改变任务有效性。 |
| **回退（fallback）** | 从请求路径切换到可用替代路径并记录原因，例如 CUDA 转 CPU。 | 实验记录应保留实际设备与原因；不要假设所有外部模型采用相同策略。 |
| **Smoke check** | 使用小型确定性输入检查 artifact/runtime 兼容性。 | 通过 smoke check 不能证明医学准确性或泛化能力。 |
| **Golden check** | 把确定性输出与随包 expected output 对比。 | 它在容差内检测发布/runtime 漂移，不是临床验证。 |
| **数据域（domain）** | 模型或 benchmark 数据所代表的人群、modality、protocol、preprocessing、label 与条件。 | 即使 input shape 匹配，域外结果也可能不可靠。 |
| **域偏移（domain shift）** | Training data 与当前数据在 scanner、protocol、population、BraTS release 等方面的实质差异。 | Performance 可能改变，解释结果时必须保留这项警告。 |
| **泛化（generalization）** | 方法在构建或调参样本之外的适当数据上保持 performance 的能力。 | 单个案例、随包演示或 smoke test 都不能证明它。 |

## 评价与指标

| 术语 | 通俗解释 | 为什么重要 |
| --- | --- | --- |
| **评价单位（evaluation unit）** | 被计为一个 observation 的实体：pixel、lesion、image、series、slide、study 或 patient。 | 单位改变会改变指标语义，必须明确说明。 |
| **Train / validation / test split** | 分别用于拟合、选择设置和最终评价的独立分区。 | 不应使用保留 test set 来选择 preprocessing 或 threshold。 |
| **组泄漏 / 数据泄漏（group/data leakage）** | 同一患者、study 或相关 group 跨分区出现，或不当影响评价。 | 按 slice 随机拆分可能让 performance 远高于真实泛化水平。 |
| **阈值（threshold）** | 把 probability 或连续 score 转换为 decision 的分界值。 | Sensitivity、specificity、PPV、NPV、accuracy 与 F1 都依赖它；AUROC 不会替您选择阈值。 |
| **混淆矩阵（confusion matrix）** | True positive（TP）、true negative（TN）、false positive（FP）、false negative（FN）的计数。 | 许多 classification metric 是在一个 threshold 下对这些计数的不同汇总。 |
| **敏感度 / recall / true-positive rate** | 实际 positive 被识别为 positive 的比例：`TP / (TP + FN)`。 | 高 sensitivity 可能伴随更多 false positive；应报告 threshold 与评价单位。 |
| **特异度 / true-negative rate** | 实际 negative 被识别为 negative 的比例：`TN / (TN + FP)`。 | 它与 sensitivity 互补，同样依赖工作阈值。 |
| **Precision / PPV** | Positive prediction 中正确项的比例：`TP / (TP + FP)`。 | 它依赖评价数据中的 prevalence，换一个人群可能改变。 |
| **NPV** | Negative prediction 中正确项的比例：`TN / (TN + FN)`。 | 与 PPV 一样，它依赖 prevalence 与所选 threshold。 |
| **准确率（accuracy）** | 所有 decision 中正确项的比例。 | 数据不平衡时可能看起来很高，同时 minority class 表现很差。 |
| **F1 score** | Precision 与 recall 的调和平均。 | 它忽略 true negative 且依赖 threshold，不能作为完整 performance 总结。 |
| **AUROC** | Receiver operating characteristic curve 下面积，汇总跨 threshold 的正负样本排序。 | 它不衡量 calibration，也不选择临床工作点；评价数据必须同时包含两类。 |
| **AUPRC / average precision** | 跨 threshold 的 precision–recall 关系汇总；OpenMedVisionX 报告 average precision 作为 PR 汇总。 | 它强调 positive-class retrieval 并受 prevalence 影响；应保留精确指标名称和定义。 |
| **校准（calibration）** | Predicted probability 与实际发生频率的一致程度。 | 模型可以有很高 AUROC，同时 probability 过度自信或不够自信。 |
| **Brier score** | Predicted probability 与 binary outcome 之间 squared difference 的平均；越低越好。 | 它同时受 probability calibration 与 discrimination 影响，应结合数据集与 prevalence 解释。 |
| **ECE（expected calibration error）** | 分箱后 average confidence 与 observed frequency 差距的汇总；越低越好。 | 它依赖 binning 和 sample size，应保留 bin 定义与 calibration plot。 |
| **置信区间（confidence interval）** | 由声明的统计方法产生、用于表达抽样不确定性的范围。 | 它不是“真值一定在其中”的保证，也无法覆盖所有偏差来源。 |
| **Dice coefficient** | Segmentation overlap：`2 × intersection / (predicted size + reference size)`；越高越好。 | 它可能隐藏 boundary displacement 与 small-lesion failure；必须声明 empty-mask 约定。 |
| **IoU（intersection over union）** | Prediction 与 reference 的 intersection 除以 union；越高越好。 | 与 Dice 有关但数值不同；在 detection 中还可定义 box/object matching。 |
| **HD95** | 双向 surface distance 的第 95 百分位，通常以毫米报告；越低越好。 | 它描述接近最差情况的边界分离，需要有效 spacing/geometry，并与 overlap 互补。 |
| **ASSD** | Prediction 与 reference 之间的 average symmetric surface distance；越低越好。 | 它汇总典型 boundary error，但可能平滑局部大误差。 |
| **Surface Dice** | 位于声明距离 tolerance 内的 surface 比例；越高越好。 | Tolerance 与物理单位属于指标定义，必须报告。 |
| **体积误差（volume error）** | Prediction 与 reference 物理体积的差，按声明可为 absolute 或 relative。 | 总体积接近也可能空间位置错误，应配合 overlap 与 boundary metric。 |
| **MAE / MSE / RMSE** | Mean absolute、mean squared 和 root mean squared numeric error；越低越好。 | 需要共享 numeric domain 与 geometry；MSE 对大误差惩罚更重。 |
| **PSNR** | 根据 MSE 和声明 data range 计算的 peak signal-to-noise ratio；越高越好。 | Peak/data range 不一致或被隐藏时，该值没有可比意义。 |
| **SSIM** | 比较局部 luminance、contrast 与 structure 的 structural similarity index；通常越高越好。 | Window 和 data range 会影响结果；高分不能证明临床 fidelity。 |
| **AP / mAP** | 单一 class 的 average precision，以及按声明跨 class/threshold 的平均。 | Detection 结果取决于 matching、IoU threshold、confidence filtering 与评价单位。 |
| **FROC** | Free-response receiver operating characteristic，描述 lesion sensitivity 与每张 image/scan false positive 的关系。 | 适用于 detection，但必须有精确 lesion-matching rule。 |

## AI、隐私与信任

| 术语 | 通俗解释 | 为什么重要 |
| --- | --- | --- |
| **AI provider（服务商）** | 接收请求并返回响应的外部或本地服务。 | Provider 的政策、保留机制、能力和目标位于本地应用边界之外。 |
| **Endpoint** | Provider 请求使用的精确网络地址。 | 应检查 scheme 与 host；更改它就是更改目标，并会使之前的图像授权失效。 |
| **Model ID** | Provider 用来指定所选 AI model 的标识。 | 不同 model 能力不同；只有 provider 名称不能证明支持 vision。 |
| **凭据引用（credential reference）** | `env:OPENAI_API_KEY` 或 keyring entry 等指针，不是 secret value 本身。 | 真实 key 可以保留在项目文件、截图、日志和 issue 之外。 |
| **环境变量 / keyring** | 请求时提供凭据的两种方式；keyring 使用操作系统 credential store。 | 共享机器通常更适合 keyring；不得打印或提交解析后的真实值。 |
| **Prompt（提示文本）** | 作为 instruction 或 context 发送给 AI model 的文本。 | 即使没有附加图像，文本也可能包含患者或机构信息。 |
| **信任边界（trust boundary）** | 数据离开本地应用、进入另一系统或机构的分界点。 | 本地查看不越界；provider 请求会越界，因此必须由用户主动决定。 |
| **网络选择加入（network opt-in）** | 用户明确决定启用并发送 provider 请求。 | 仅仅打开 AI 助手页面不会让 OpenMedVisionX 联系 provider。 |
| **渲染预览（rendered preview）** | 图像请求前展示的 PNG：由 display mapping 后的完整当前二维平面重新编码。 | 它是需要检查的实际图像 payload；不含源 DICOM/NIfTI bytes、metadata、viewport zoom/pan、overlay 或 annotation。 |
| **无元数据 PNG（metadata-free PNG）** | 重新编码且只保留像素与必要图像 chunk，不含 EXIF/text/XMP/ICC/time metadata 的 PNG。 | 它降低 metadata 泄露，但不能去掉像素中可见的标识。 |
| **传输计划（transfer plan）** | 针对一个精确 provider、endpoint、model、task、prompt digest 与 PNG payload 的检查记录。 | 任一字段变化都需要新计划；批准不是长期“始终允许图像”开关。 |
| **一次性授权（one-shot authorization）** | 在一次图像发送尝试前立即封存并消费的同意。 | 不能复用；撤销也不能召回已经发出的请求。 |
| **Hash / SHA-256 / digest** | 根据 bytes 或规范内容计算的固定长度指纹。 | 匹配 hash 有助于绑定授权或校验 artifact；hash 不是加密，也不证明科学有效性。 |
| **去标识（de-identification）** | 旨在删除或降低身份识别信息的有文档流程。 | “删除 metadata”不等于完成去标识；可见文字与上下文仍可能存在。 |
| **固化文字（burned-in text）** | 作为图像像素而不是 metadata 存储的标识文字。 | 除非完全不使用这份数据，否则它会进入 rendered preview；自动 OCR 不能证明安全。 |
| **敏感信息 / PHI** | 能识别个人，或受适用政策/法律保护的信息。 | 没有适当权限与保护措施时，不得粘贴到 prompt 或通过像素发送。 |
| **残余风险（residual risk）** | 采取措施后仍存在的风险，例如 provider retention、固化标识或上下文重识别。 | 每次图像传输前都要检查；技术控制无法把风险降为零。 |
| **结构化产物（structured artifact）** | 与声明 request 和 geometry 绑定的类型化结果，例如 class score、label、mask 或 reconstruction。 | 它不同于流畅 chat 文本；当前桌面对话不会自动把响应转换为这些产物。 |
| **确认 / 拒绝产物（Confirm / Reject artifact）** | 对已提供类型化 artifact 做出的本地审查决定。 | 它只记录审查：不发送数据，也不创建或覆盖 Study layer。 |
| **证据定位（grounding）** | 把 AI statement 与可见证据或可信来源联系起来。 | 没有 grounding 的流畅响应只是需要核验的解释，不是证据或诊断。 |
| **幻觉（hallucination）** | 看似自信但缺乏依据、被虚构或与输入不一致的输出。 | 应把 provider 文本和看似合理的图像当作 hypothesis，并独立核验。 |

完整用户流程见[AI 助手、隐私与云端图像](LLM_SECURITY.zh-CN.md)。

## 常见概念辨析

| 不要混淆…… | 与…… | 实用判断方法 |
| --- | --- | --- |
| 源数组 | Display mapping | 改变 window/gamma；如果记录的 dtype/range 不变，变化的只是显示。 |
| 页面序列 | 物理体数据 | 检查 spacing、origin、orientation 与 affine/等价几何是否经过校验。 |
| Pixel index | World coordinate | 判断当前值是 array location，还是 RAS+/LPS 中的毫米位置。 |
| Binary mask | Multi-class label map | Mask 标记包含/不包含；label map 分配声明的整数 class。 |
| Label map | Vector contour | 前者占据 raster grid，后者存储某坐标系中的 path。 |
| Probability | Binary decision | Threshold 把前者转换成后者；threshold 变化会改变 confusion matrix。 |
| AUROC | Calibration | AUROC 衡量跨 threshold 排序；calibration 衡量 probability 是否符合发生频率。 |
| 视觉质量 | 数值/科学有效性 | 使用共享显示范围、geometry check、reference、metric 与 failure inspection。 |
| Runtime 校验 | 模型准确性 | Smoke/golden 只说明随包计算按预期运行，不能说明模型具有泛化能力。 |
| 本地打开影像 | 云端图像传输 | 打开操作留在本地；经过检查的 provider 请求会跨越 trust boundary。 |
| 删除 metadata | 去标识 | 除 metadata 字段外，还要检查 pixel、filename、context 与 residual risk。 |
| 模型输出 | 临床结论 | OpenMedVisionX 输出仅用于学习/研究，必须独立验证，永远不是诊断。 |

## 下一步

- 准备并打开数据：[数据与格式](DATA_FORMATS.zh-CN.md)
- 完成第一次体验：[快速开始](QUICKSTART.zh-CN.md)
- 学习各工作区：[用户指南](USER_GUIDE.zh-CN.md)
- 按系统课程学习：[学习课程](TEACHING_CURRICULUM.zh-CN.md)
- 检查 AI 隐私：[AI 助手、隐私与云端图像](LLM_SECURITY.zh-CN.md)
- 解决问题：[故障排查](TROUBLESHOOTING.zh-CN.md)

下一步：[按照学习课程继续](TEACHING_CURRICULUM.zh-CN.md)。
