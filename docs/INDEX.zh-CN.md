# OpenMedVisionX 文档

[English](INDEX.md) · [项目概览](README.zh-CN.md)

这是 OpenMedVisionX 用户文档的起点。请按照自己想完成的任务选择阅读路线，无需从头到尾
依次阅读所有页面。

> [!NOTE]
> 所有工作流仅用于学习与研究。OpenMedVisionX 不是医疗器械，其输出不得用于任何患者
> 相关决策。

## 第一次使用？请按这条路线阅读

1. **安装与启动：**阅读[快速开始](QUICKSTART.zh-CN.md)第 1～2 节。
2. **不使用本地数据也能得到结果：**按照[快速开始](QUICKSTART.zh-CN.md)第 3 节运行首个
   合成 CT 实验。
3. **认识界面：**阅读[用户指南](USER_GUIDE.zh-CN.md)中的界面地图。
4. **选择主题：**使用下面的“按工作区查阅”和“按任务查阅”表格。
5. **建立可靠的学习习惯：**按照[学习课程](TEACHING_CURRICULUM.zh-CN.md)逐步练习。

如果安装或加载在任何阶段失败，请直接前往[故障排查](TROUBLESHOOTING.zh-CN.md)。

## 按工作区查阅

| 工作区 | 从这里开始 | 深入阅读 |
| --- | --- | --- |
| **影像** | [用户指南 · 影像](USER_GUIDE.zh-CN.md#影像) | [数据与格式](DATA_FORMATS.zh-CN.md) |
| **CT 实验** | [用户指南 · CT 实验](USER_GUIDE.zh-CN.md#ct-实验) | [学习课程 · CT 投影](TEACHING_CURRICULUM.zh-CN.md#3-从物体到-sinogram再返回图像) |
| **模型** | [用户指南 · 模型](USER_GUIDE.zh-CN.md#模型) | [随包模型](MODEL_BUNDLES.zh-CN.md) · [自定义本地模型](CUSTOM_MODELS.zh-CN.md) |
| **教学** | [用户指南 · 教学](USER_GUIDE.zh-CN.md#教学) | [学习课程](TEACHING_CURRICULUM.zh-CN.md) · [术语表](GLOSSARY.zh-CN.md) |
| **评价** | [用户指南 · 评价](USER_GUIDE.zh-CN.md#评价) | [学习课程 · 评价](TEACHING_CURRICULUM.zh-CN.md#5-避免数据泄漏的评价) |
| **AI 助手** | [用户指南 · AI 助手](USER_GUIDE.zh-CN.md#ai-助手) | [AI 助手、隐私与云端图像](LLM_SECURITY.zh-CN.md) |

## 按任务查阅

| 我想要…… | 请阅读…… |
| --- | --- |
| 打开 PNG、TIFF、DICOM 序列/ZIP 或 NIfTI 体数据 | [数据与格式](DATA_FORMATS.zh-CN.md)，再读[用户指南 · 影像](USER_GUIDE.zh-CN.md#影像) |
| 导入 mask、DICOM SEG、RTSTRUCT 或 label map | [数据与格式 · 标注与标签](DATA_FORMATS.zh-CN.md#标注mask-与标签图层) |
| 完成内置 CT 实验 | [快速开始 · 第一次体验](QUICKSTART.zh-CN.md#3-完成第一次体验) |
| 安装 PyTorch 并运行经过审查的模型 | [随包模型指南](MODEL_BUNDLES.zh-CN.md) |
| 加载自己的 ONNX、TorchScript 或 Python adapter 模型 | [自定义模型指南](CUSTOM_MODELS.zh-CN.md) |
| 检查数据集划分或二分类预测 | [用户指南 · 评价](USER_GUIDE.zh-CN.md#评价) |
| 配置 AI 服务且不把密钥存入项目 | [AI 助手、隐私与云端图像](LLM_SECURITY.zh-CN.md) |
| 理解陌生术语 | [术语表](GLOSSARY.zh-CN.md) |
| 排查错误信息 | [故障排查](TROUBLESHOOTING.zh-CN.md) |

## 核心指南

| 指南 | 可以获得什么 |
| --- | --- |
| [快速开始](QUICKSTART.zh-CN.md) | 安装、启动、第一次隐私安全体验和可选能力配置。 |
| [用户指南](USER_GUIDE.zh-CN.md) | 六个桌面工作区的完整任务式操作说明。 |
| [数据与格式](DATA_FORMATS.zh-CN.md) | 支持的文件、选择规则、几何/强度语义、标注和安全准备方式。 |
| [随包模型指南](MODEL_BUNDLES.zh-CN.md) | PyTorch 配置、完整性校验以及 DIVal、DeepInverse、MONAI 工作流。 |
| [自定义模型指南](CUSTOM_MODELS.zh-CN.md) | 其他本地模型在桌面中校验、加载和运行前需要满足的条件。 |
| [AI 助手、隐私与云端图像](LLM_SECURITY.zh-CN.md) | 服务配置、凭据引用、文本请求、精确图像授权与事件处理。 |
| [学习课程](TEACHING_CURRICULUM.zh-CN.md) | 从像素与几何到重建、评价和负责任 AI 的系统学习路线。 |
| [故障排查](TROUBLESHOOTING.zh-CN.md) | 按现象解决安装、数据、模型、评价、导出与 AI 请求问题。 |

## 参考与边界

- [单页双语简介](INTRODUCTION.md)适合在完整指南之前快速分享项目定位。
- [术语表](GLOSSARY.zh-CN.md)解释界面与指南中使用的术语。
- [支持的数据与格式](DATA_FORMATS.zh-CN.md#支持矩阵)是桌面能打开哪些内容、如何解释
  各类输入的权威说明。
- [随包模型卡](MODEL_BUNDLES.zh-CN.md#理解每个模型的输入)是模型专用输入和限制的权威说明。
- [AI 隐私指南](LLM_SECURITY.zh-CN.md#什么内容可能离开您的电脑)解释文本或渲染图在什么
  情况下可能离开本地信任边界。
- [安全策略](SECURITY.zh-CN.md)说明如何私下报告安全漏洞或疑似数据泄露。

项目不包含训练数据集、BraTS 案例、API 凭据或患者影像。OpenMedVisionX 不会通过
静默编造间距、模态、通道、几何或采集语义，把不受支持的输入伪装成看似可用的输入。

## 语言与版本范围

用户阅读路线中的每一页都提供英文与简体中文版本；使用页面顶部的语言链接即可在同一
主题间切换。按钮名称按照相应界面语言准确书写。

这些指南描述当前仓库中的 Alpha 版本。如果某条流程有意不可用——例如通过通用外部
模型页面提交复合 k-space/mask 输入——文档会明确说明，而不会把它写成可用功能。

下一步：[安装并完成第一次体验](QUICKSTART.zh-CN.md)。
