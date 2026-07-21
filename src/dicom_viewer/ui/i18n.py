"""Small, dependency-free translations for the desktop interface.

The English strings remain the canonical identifiers used by the application.
Translations are deliberately bounded to registered strings and format templates:
text supplied by plugins, image metadata, or exception messages is returned unchanged.
"""

from __future__ import annotations

import re
from string import Formatter
from typing import Literal

Language = Literal["en", "zh_CN"]


_TRANSLATIONS: tuple[tuple[str, str], ...] = (
    # Application identity and top-level navigation.
    (
        "An Open Interactive Platform for Medical Computer Vision Learning and Exploration",
        "面向医学计算机视觉学习与探索的开放式交互平台",
    ),
    ("Medical vision workspace", "医学视觉工作台"),
    (
        "OpenMedVisionX: An Open Interactive Platform for Medical Computer Vision Learning "
        "and Exploration",
        "OpenMedVisionX：面向医学计算机视觉学习与探索的开放式交互平台",
    ),
    ("OpenMedVisionX contributors", "OpenMedVisionX 贡献者"),
    ("Image Explorer", "影像浏览"),
    ("CT Reconstruction Lab", "CT 重建实验室"),
    ("External Models", "模型推理"),
    ("Learning Experiments", "教学实验"),
    (" Learning Experiments", " 教学实验"),
    ("AI Teaching Assistant", "AI 教学助手"),
    ("Images", "影像"),
    ("CT Lab", "CT 实验"),
    ("Models", "模型"),
    ("Learn", "教学"),
    ("AI Assistant", "AI 助手"),
    ("Local · Research", "本地 · 研究"),
    ("Local first · Research only", "本地优先 · 仅供研究"),
    (
        "Local-first. No data is uploaded automatically. Not for clinical use.",
        "本地优先。不会自动上传任何数据。不可用于临床。",
    ),
    ("中文", "中文"),
    ("English", "English"),
    ("Switch language", "切换语言"),
    ("OpenMedVisionX workspaces", "OpenMedVisionX 工作区"),

    # Image explorer: toolbar, views, controls, and initial state.
    ("Open image / DICOM ZIP", "打开影像 / DICOM ZIP"),
    ("Open DICOM folder", "打开 DICOM 文件夹"),
    ("Open local medical image", "打开本地医学影像"),
    (
        "Open an image, DICOM ZIP, or NIfTI file (Ctrl+O)",
        "打开影像、DICOM ZIP 或 NIfTI 文件 (Ctrl+O)",
    ),
    ("Open a DICOM folder (Ctrl+Shift+O)", "打开 DICOM 文件夹 (Ctrl+Shift+O)"),
    ("Cancel loading", "取消加载"),
    ("Histogram", "直方图"),
    ("Fit views", "缩放至窗口"),
    ("Fit all image views (Ctrl+0)", "缩放全部影像视图至窗口 (Ctrl+0)"),
    ("Pan / zoom", "平移 / 缩放"),
    ("Distance", "距离"),
    ("Area / ROI", "面积 / ROI"),
    ("Annotation", "标注"),
    ("Clear marks", "清除标记"),
    ("Set raster spacing…", "设置像素间距…"),
    ("Tool:", "工具："),
    ("Image / Axial", "图像 / 轴位"),
    ("Image", "图像"),
    ("Coronal (RAS+)", "冠状位 (RAS+)"),
    ("Sagittal (RAS+)", "矢状位 (RAS+)"),
    ("Overlay / comparison", "叠加 / 对比"),
    ("Navigation", "导航"),
    ("Play pages", "播放序列"),
    ("Pause", "暂停"),
    ("Slice / page", "切片 / 页面"),
    ("Coronal Y", "冠状位 Y"),
    ("Sagittal X", "矢状位 X"),
    (
        "Display mapping (decoded data is unchanged)",
        "显示映射（解码数据保持不变）",
    ),
    ("Display", "显示"),
    ("View only — source values stay unchanged.", "仅调整显示，不改变源数据。"),
    (
        "Display mapping changes only the view, not decoded data.",
        "显示映射只改变视图，不改变解码数据。",
    ),
    ("Intensity range", "强度范围"),
    ("HU window", "HU 窗"),
    ("Lower", "下限"),
    ("Upper", "上限"),
    ("RGB brightness", "RGB 亮度"),
    ("RGB contrast", "RGB 对比度"),
    ("RGB gamma", "RGB 伽马"),
    ("Auto range", "自动范围"),
    ("Explicit local pairing and output", "本地关联与导出"),
    ("Local files & export", "本地文件与导出"),
    (
        "Files are used only when you select them. OpenMedVisionX never scans neighbouring "
        "files or uploads paired data. Outputs are created only at your selected path and "
        "never overwrite an existing file.",
        "仅使用您明确选择的文件。OpenMedVisionX 不会扫描相邻文件，也不会上传配对数据。"
        "输出仅创建在您选择的路径中，并且绝不会覆盖现有文件。",
    ),
    (
        "Only selected files are used. Nothing nearby is scanned or uploaded, and exports "
        "never overwrite an existing file.",
        "只使用您选择的文件，不会扫描相邻文件或上传数据；导出时也不会覆盖已有文件。",
    ),
    ("Pair mask…", "关联掩膜…"),
    ("Pair annotation JSON…", "关联标注 JSON…"),
    ("Mask…", "掩膜…"),
    ("Annotations…", "标注…"),
    ("Export rendered PNG…", "导出渲染后的 PNG…"),
    ("Save pixel-free experiment JSON…", "保存无图像实验记录…"),
    ("Cancel local operation", "取消本地操作"),
    ("No local mask or annotation is paired.", "尚未配对本地掩膜或标注。"),
    ("Image information (non-sensitive)", "图像信息（非敏感）"),
    ("Image details", "影像详情"),
    (
        "Open a local image, DICOM folder/ZIP, or NIfTI volume.",
        "打开本地图像、DICOM 文件夹/ZIP 或 NIfTI 体数据。",
    ),

    # Image explorer: dialogs, notices, view titles, and complete status sentences.
    ("Open local image", "打开本地影像"),
    (
        "Supported images (*.dcm *.dicom *.zip *.nii *.nii.gz *.png *.jpg *.jpeg *.tif "
        "*.tiff);;All files (*)",
        "支持的图像 (*.dcm *.dicom *.zip *.nii *.nii.gz *.png *.jpg *.jpeg *.tif "
        "*.tiff);;所有文件 (*)",
    ),
    ("All files (*)", "所有文件 (*)"),
    ("Loading in background…", "正在后台加载…"),
    ("Cancelling…", "正在取消…"),
    ("Cancelling load safely…", "正在安全取消加载…"),
    ("Cancelling safely…", "正在安全取消…"),
    ("Loading cancelled.", "已取消加载。"),
    ("Probing image format", "正在识别图像格式"),
    (
        "Resampling physical volume to RAS+ display grid",
        "正在将物理体数据重采样到 RAS+ 显示网格",
    ),
    ("Preparing maximum-intensity projection", "正在准备最大强度投影"),
    ("Committing validated image state", "正在完成影像加载"),
    ("Image ready", "图像已就绪"),
    (
        "Large flat raster detected; preparing bounded thumbnails",
        "检测到大型平面光栅图像；正在准备受限缩略图",
    ),
    ("Loaded {kind}: {shape}", "已加载 {kind}：{shape}"),
    ("Load failed: {error}", "加载失败：{error}"),
    ("OpenMedVisionX – load failed", "OpenMedVisionX – 加载失败"),
    ("2-D raster image", "二维光栅图像"),
    ("Page / frame {current}/{total}", "页面 / 帧 {current}/{total}"),
    ("Axial RAS+ – z {current}/{total}", "轴位 RAS+ – z {current}/{total}"),
    ("Coronal RAS+ – y {current}/{total}", "冠状位 RAS+ – y {current}/{total}"),
    ("Sagittal RAS+ – x {current}/{total}", "矢状位 RAS+ – x {current}/{total}"),
    ("Axial maximum-intensity projection", "轴位最大强度投影"),
    (
        "A local overlay is paired to another slice/page; return to it to view the overlay.",
        "本地叠加层已配对到另一切片/页面；请返回该处查看叠加层。",
    ),
    ("mask", "掩膜"),
    ("annotation", "标注项"),
    (" and ", " 和 "),
    (
        "Explicit local pairing on the current plane: {items}.",
        "当前平面上的明确本地配对：{items}。",
    ),
    ("Decoded-value histogram", "解码值直方图"),
    (
        "Decoded-value histogram (display mapping does not alter these values).",
        "解码值直方图（显示映射不会改变这些数值）。",
    ),
    ("User-provided pixel spacing", "用户提供的像素间距"),
    ("X spacing (mm)", "X 间距 (mm)"),
    ("Y spacing (mm)", "Y 间距 (mm)"),
    (
        "This value is user-provided. OpenMedVisionX never infers medical spacing from DPI.",
        "此数值由用户提供。OpenMedVisionX 绝不会根据 DPI 推断医学图像间距。",
    ),
    ("Apply", "应用"),
    ("Cancel", "取消"),
    ("Close", "关闭"),
    ("Local operation failed", "本地操作失败"),
    ("Local operation failed safely: {error}", "本地操作已安全失败：{error}"),
    ("Validating selected mask", "正在验证所选掩膜"),
    ("Converting non-zero labels to overlay", "正在将非零标签转换为叠加层"),
    ("Mask paired locally", "掩膜已在本地配对"),
    ("Reading selected annotation", "正在读取所选标注"),
    ("Annotations paired locally", "标注已在本地配对"),
    ("Writing new file", "正在写入新文件"),
    ("Saved without overwriting", "已保存且未覆盖现有文件"),
    ("Applying display mapping", "正在应用显示映射"),
    ("Encoding rendered PNG", "正在编码渲染后的 PNG"),
    ("Saving rendered PNG", "正在保存渲染后的 PNG"),
    ("Select one local mask for the current plane", "为当前平面选择一个本地掩膜"),
    ("Lossless masks (*.png *.tif *.tiff)", "无损掩膜 (*.png *.tif *.tiff)"),
    (
        "Local mask paired explicitly; no files were scanned or uploaded.",
        "已明确配对本地掩膜；未扫描或上传任何文件。",
    ),
    (
        "Select one local annotation JSON for the current plane",
        "为当前平面选择一个本地标注 JSON",
    ),
    ("OpenMedVisionX annotation (*.json)", "OpenMedVisionX 标注 (*.json)"),
    (
        "Local annotation paired explicitly; external references are not followed.",
        "已明确配对本地标注；不会跟随外部引用。",
    ),
    ("Export rendered plane to a new local file", "将渲染平面导出为新的本地文件"),
    ("PNG image (*.png)", "PNG 图像 (*.png)"),
    ("Rendered PNG saved as new file: {name}", "渲染后的 PNG 已保存为新文件：{name}"),
    ("Save a pixel-free experiment record", "保存不含像素的实验记录"),
    ("Experiment JSON (*.json)", "实验 JSON (*.json)"),
    (
        "Pixel-free experiment parameters and numeric metrics saved: {name}",
        "不含像素的实验参数和数值指标已保存：{name}",
    ),
    ("Image exploration", "图像探索"),
    ("Type", "类型"),
    ("Source", "来源"),
    ("Shape", "形状"),
    ("Dtype", "数据类型"),
    ("Semantics", "语义"),
    ("Capabilities", "功能"),
    ("Color", "颜色"),
    ("Bit depth", "位深"),
    ("Alpha", "Alpha"),
    ("Spacing", "间距"),
    ("Spacing source", "间距来源"),
    ("Pages/frames", "页面/帧"),
    ("Spatial volume semantics", "空间体数据语义"),
    ("Spatial volume semantics: disabled", "空间体数据语义：已禁用"),
    ("disabled", "已禁用"),
    ("Modality", "模态"),
    ("Spacing XYZ (mm)", "XYZ 间距 (mm)"),
    ("Origin RAS+ (mm)", "RAS+ 原点 (mm)"),
    ("Direction", "方向"),
    ("Runtime metadata (PHI-filtered):", "运行时元数据（已过滤 PHI）："),
    ("Runtime metadata (PHI-filtered)", "运行时元数据（已过滤 PHI）"),
    ("none (pixel units only)", "无（仅像素单位）"),
    (
        "Large flat raster: displaying a bounded thumbnail with reversible source coordinates; "
        "full-resolution pixels were not retained in the session.",
        "大型平面光栅图像：当前显示具有可逆源坐标的受限缩略图；会话中未保留全分辨率像素。",
    ),
    (
        "Lossy JPEG compression: artifacts are not original signal or model output.",
        "有损 JPEG 压缩：伪影既不是原始信号，也不是模型输出。",
    ),
    (
        "Page sequence only: 3-D tools and physical volume measurements are disabled.",
        "仅为页面序列：三维工具和物理体积测量已禁用。",
    ),
    (
        "Physical spacing is user-provided, not inferred from the file.",
        "物理间距由用户提供，并非从文件推断。",
    ),

    # CT reconstruction lab.
    (
        "Parallel-beam teaching note: projections over 180° contain the complete information. "
        "A 360° scan repeats it as p(s, θ + 180°) = p(−s, θ); OpenMedVisionX folds and "
        "averages the redundant half.",
        "平行束教学说明：180° 范围内的投影已包含完整信息。360° 扫描会按照 "
        "p(s, θ + 180°) = p(−s, θ) 重复这些信息；OpenMedVisionX 会折叠并平均冗余的半周数据。",
    ),
    (
        "For parallel-beam CT, 180° contains the complete projection set; 360° data is folded "
        "and averaged.",
        "对于平行束 CT，180° 已包含完整投影；360° 数据会折叠并取平均。",
    ),
    ("Reconstruction setup", "重建设置"),
    ("Open an image in Images to enable reconstruction.", "请先在“影像”中打开图像以启用重建。"),
    ("No additional parameters.", "无需其他参数。"),
    ("Circular support", "圆形支撑域"),
    ("Preparing Radon transform", "正在准备 Radon 变换"),
    ("Radon transform complete", "Radon 变换完成"),
    (
        "Computing detector-axis Fourier transform",
        "正在计算探测器轴向 Fourier 变换",
    ),
    (
        "Applying inverse 2-D Fourier transform",
        "正在应用二维逆 Fourier 变换",
    ),
    ("Direct Fourier reconstruction complete", "直接 Fourier 重建完成"),
    ("nearest", "最近邻"),
    ("linear", "线性"),
    ("cubic", "三次"),
    ("1. Generate sinogram", "1. 生成正弦图"),
    ("2. Reconstruct", "2. 重建"),
    ("Range °", "扫描范围（°）"),
    ("Angles", "角度数"),
    ("Algorithm", "算法"),
    ("DFR interpolation", "DFR 插值"),
    ("FBP filter", "FBP 滤波器"),
    ("SART iterations", "SART 迭代次数"),
    ("Relaxation", "松弛因子"),
    ("Input", "输入"),
    ("Sinogram", "正弦图"),
    ("Reconstruction", "重建结果"),
    ("Absolute error heatmap", "绝对误差热力图"),
    (
        "MSE / PSNR / SSIM will be computed in one joint intensity range.",
        "MSE / PSNR / SSIM 将在同一个联合强度范围内计算。",
    ),
    ("Metrics use one shared intensity range.", "指标采用统一的强度范围。"),
    ("Intermediate process:", "中间过程："),
    ("Export rendered result PNG…", "导出渲染结果 PNG…"),
    ("Export PNG…", "导出 PNG…"),
    ("Save record…", "保存记录…"),
    ("No reconstruction input", "没有重建输入"),
    ("Signed normalized difference", "带符号归一化差值"),
    ("Metrics unavailable", "指标不可用"),
    ("Metrics unavailable: {error}", "指标不可用：{error}"),
    ("Joint range", "统一强度范围"),
    ("Intermediate", "中间结果"),
    ("Radon projection", "Radon 投影"),
    ("Radon projection {current}/{total}", "Radon 投影 {current}/{total}"),
    ("Backprojection", "反投影"),
    ("Backprojection progress", "反投影进度"),
    ("{algorithm} progress {current}/{total}", "{algorithm} 进度 {current}/{total}"),
    ("SART iteration", "SART 迭代"),
    ("SART iteration {current}/{total}", "SART 迭代 {current}/{total}"),
    ("Intermediate: {name}", "中间过程：{name}"),
    ("Local output failed", "本地输出失败"),
    ("Local output failed safely: {error}", "本地输出已安全失败：{error}"),
    ("Export reconstruction to a new local file", "将重建结果导出为新的本地文件"),
    ("Rendered reconstruction saved as new file", "渲染后的重建结果已保存为新文件"),
    (
        "Rendered reconstruction saved as new file: {name}",
        "渲染后的重建结果已保存为新文件：{name}",
    ),
    (
        "Save reconstruction parameters and numeric metrics",
        "保存重建参数和数值指标",
    ),
    ("CT reconstruction", "CT 重建"),
    ("Pixel-free experiment record saved", "不含像素的实验记录已保存"),
    (
        "Pixel-free experiment record saved: {name}",
        "不含像素的实验记录已保存：{name}",
    ),
    ("Reconstruction failed", "重建失败"),
    ("{message} – %p%", "{message} – %p%"),

    # External-model page.
    (
        "OpenMedVisionX never bundles or downloads model weights. Import an ONNX/TorchScript "
        "manifest or a manifest.yaml + Python adapter. Compatibility is defined by the "
        "protocol, not a checkpoint-name whitelist.",
        "OpenMedVisionX 绝不会捆绑或下载模型权重。请导入 ONNX/TorchScript 清单，或导入 "
        "manifest.yaml + Python 适配器。兼容性由协议定义，而不是由检查点名称白名单定义。",
    ),
    (
        "Models and weights stay local. Choose a manifest, load its referenced model, then "
        "run it on the current image.",
        "模型和权重始终保留在本地。选择模型清单、加载其引用的模型，然后对当前影像运行推理。",
    ),
    ("Validate local manifest.yaml…", "选择并验证模型清单…"),
    ("Load referenced local model", "加载本地模型"),
    ("Run on current image", "对当前影像推理"),
    ("Cancel / stop plugin", "停止模型任务"),
    ("1. Choose manifest…", "1. 选择模型清单…"),
    ("2. Load model", "2. 加载模型"),
    ("3. Run inference", "3. 运行推理"),
    ("Stop", "停止"),
    ("No model configured", "尚未配置模型"),
    ("Choose a model manifest to begin.", "选择模型清单开始。"),
    ("SAM download & setup", "SAM 下载与配置"),
    ("A compatible plugin manifest is required.", "仍需配套兼容的插件清单。"),
    ("Model manifest ready", "模型清单已就绪"),
    (
        "All referenced local weight files are available.",
        "清单引用的本地权重文件均可用。",
    ),
    ("Model ready", "模型已就绪"),
    (
        "Loaded from local paths; no files were downloaded or copied.",
        "已从本地路径加载；未下载或复制任何文件。",
    ),
    ("Weight not found: {name}", "未找到权重：{name}"),
    (
        "Download it, then configure its local path as described by the plugin.",
        "下载后，请按插件说明配置本地路径。",
    ),
    ("Open official guide", "查看官方说明"),
    (
        "No GitHub source is declared in this manifest.",
        "此模型清单未声明 GitHub 来源。",
    ),
    (
        "Manifest details and model outputs appear here. Review the runtime, licenses, "
        "preprocessing, and local weight paths before loading.\n\nPython adapters execute "
        "third-party code and require explicit consent.",
        "模型清单详情与输出将显示在此处。加载前请检查运行时、许可证、预处理和本地权重路径。"
        "\n\nPython 适配器会执行第三方代码，必须明确授权。",
    ),
    ("Current model input", "当前模型输入"),
    ("Typed model visualization", "模型输出预览"),
    ("Visualization", "可视化"),
    ("Open model manifest", "打开模型清单"),
    ("YAML (*.yaml *.yml)", "YAML 文件 (*.yaml *.yml)"),
    ("Manifest validation failed", "清单验证失败"),
    ("Name", "名称"),
    ("Version", "版本"),
    ("Family/source", "模型族 / 来源"),
    ("Repository", "代码仓库"),
    ("Tasks", "任务"),
    ("Runtime", "运行时"),
    ("License", "许可证"),
    ("Inputs", "输入项"),
    ("outputs", "输出项"),
    ("Inputs: {inputs}; outputs: {outputs}", "输入项：{inputs}；输出项：{outputs}"),
    ("Weights", "权重文件"),
    ("External weight references", "本地权重文件"),
    (
        "External weight references: {count} (not copied)",
        "本地权重文件：{count}（未复制）",
    ),
    ("not copied", "未复制"),
    ("SECURITY WARNING:", "安全警告："),
    (
        "Python model adapters execute third-party code. Review the adapter and its licenses, "
        "then run it only in the declared isolated subprocess environment.",
        "Python 模型适配器会执行第三方代码。请审查适配器及其许可证，"
        "并且仅在声明的隔离子进程环境中运行。",
    ),
    ("Execute user-supplied Python code?", "执行用户提供的 Python 代码？"),
    (
        "A subprocess contains crashes and dependencies but is not a security sandbox. "
        "Continue only after reviewing the adapter source.",
        "子进程可隔离崩溃和依赖项，但它不是安全沙箱。只有在审查适配器源代码后才可继续。",
    ),
    ("Validating external references", "正在验证外部引用"),
    ("External model ready", "外部模型已就绪"),
    (
        "Model loaded from its existing external path; no files were downloaded or copied.",
        "模型已从其现有外部路径加载；未下载或复制任何文件。",
    ),
    ("Multi-input plugin", "模型需要多个输入"),
    (
        "This teaching page can supply one current image. Use the Python API for "
        "multi-input/prompts.",
        "此教学页面只能提供一幅当前图像。多输入/提示词场景请使用 Python API。",
    ),
    ("No model input", "暂无可用影像"),
    ("Preparing declared model input", "正在准备模型输入"),
    ("Inference complete", "推理完成"),
    ("not reported", "未报告"),
    ("Prediction result:", "预测结果："),
    ("Prediction result", "预测结果"),
    ("Task", "任务类型"),
    ("Duration", "耗时"),
    ("Visualizations", "可视化项"),
    ("none", "无"),
    ("External model failed safely", "模型任务未完成"),
    ("Model weight not found", "未找到模型权重"),
    (
        "One or more required local weight files are missing.",
        "缺少一个或多个必需的本地权重文件。",
    ),
    (
        "Review the model resource card and manifest paths, then retry.",
        "请检查模型资源卡与清单路径，配置后重试。",
    ),
    (
        "The cancelled Python adapter must be explicitly loaded again.",
        "已取消的 Python 适配器必须再次明确加载。",
    ),

    # AI teaching assistant.
    (
        "Educational assistant only — not a medical diagnosis, treatment recommendation, or "
        "clinical decision tool. API keys are resolved from environment variables or the "
        "system keyring and are never stored here.",
        "仅供教学辅助——不提供医学诊断、治疗建议，也不是临床决策工具。"
        "API 密钥从环境变量或系统密钥环解析，绝不会存储在此处。",
    ),
    (
        "For learning only — not medical advice or a clinical decision tool. Credentials stay "
        "in your environment or system keyring.",
        "仅供学习，不提供医疗建议，也不可用于临床决策。凭据始终保留在环境变量或系统密钥环中。",
    ),
    ("Provider configuration", "提供商配置"),
    ("Assistant provider setup", "助手提供商设置"),
    ("AI provider", "AI 提供商"),
    ("Provider model ID", "提供商模型 ID"),
    ("Provider endpoint", "提供商端点"),
    ("Credential environment or keyring reference", "凭据环境变量或密钥环引用"),
    ("Cloud image transfer status", "云端图像传输状态"),
    ("Show or hide assistant provider setup", "显示或隐藏助手提供商设置"),
    ("Latest assistant response in Markdown", "采用 Markdown 的最新助手回复"),
    ("Question for the AI teaching assistant", "向 AI 教学助手提出的问题"),
    (
        "Multi-line question. Press Control or Command plus Enter to send.",
        "多行问题。按 Control 或 Command 加 Enter 发送。",
    ),
    ("User-supplied model ID (never hard-coded)", "用户提供的模型 ID（绝不硬编码）"),
    (
        "Enable network requests for this configured provider",
        "为当前配置的提供商启用网络请求",
    ),
    ("Allow network requests", "允许网络请求"),
    ("Enable network", "启用网络"),
    ("Configured model supports vision", "已配置的模型支持视觉输入"),
    ("Model accepts images", "模型支持图像输入"),
    ("Vision input", "图像输入"),
    ("Send the currently previewed rendered slice", "发送当前预览的渲染切片"),
    ("Include current rendered slice", "附带当前渲染切片"),
    ("Attach visible slice", "附带可见切片"),
    (
        "Revoke image authorization for this destination",
        "撤销此目标地址的图像授权",
    ),
    ("Revoke image permission", "撤销图像授权"),
    ("Revoke permission", "撤销授权"),
    ("Cloud image transfer: OFF", "云端图像传输：关"),
    ("Provider", "提供商"),
    ("Model ID", "模型 ID"),
    ("Endpoint", "API 地址"),
    ("Credential reference", "凭据来源"),
    ("Credential", "凭据"),
    (
        "If image transfer is enabled, only the visible rendered PNG slice is sent — never "
        "original DICOM/NIfTI, a full series, or DICOM metadata. Burned-in text may still "
        "contain private information; inspect the preview first.",
        "启用图像传输后，只会发送当前可见的渲染 PNG 切片——绝不会发送原始 DICOM/NIfTI、"
        "完整序列或 DICOM 元数据。图像中固化的文字仍可能包含隐私信息；请先检查预览。",
    ),
    (
        "Image sharing sends only the visible rendered PNG. Review it for burned-in private "
        "text before authorizing a destination.",
        "共享影像时只会发送当前可见的渲染 PNG。授权目标地址前，请检查其中是否含有固化的隐私文字。",
    ),
    ("Ask the assistant", "向助手提问"),
    ("Assistant responses appear here.", "助手回复将显示在这里。"),
    (
        "Markdown supported · Responses are for learning only.",
        "支持 Markdown · 回复仅供学习。",
    ),
    ("Hide setup", "隐藏设置"),
    ("Show setup", "显示设置"),
    ("Ready when you are", "随时可以开始"),
    (
        "Ask about an imaging concept, a reconstruction parameter, or the visible result. "
        "The latest answer will appear here with Markdown formatting.",
        "可询问影像概念、重建参数或当前可见结果。最新回复会以 Markdown 格式显示在这里。",
    ),
    ("Ctrl/⌘+Enter to send", "Ctrl/⌘+Enter 发送"),
    ("{count} characters", "{count} 个字符"),
    ("Question", "问题"),
    ("Generating a new response…", "正在生成新回复…"),
    ("Response ready · Markdown rendered safely.", "回复已就绪 · Markdown 已安全渲染。"),
    (
        "Ask about concepts, parameters, reconstruction results, or learning guidance…",
        "询问概念、参数、重建结果或学习指导…",
    ),
    ("Send to configured provider", "发送到已配置的提供商"),
    ("Cancel current request", "取消当前请求"),
    ("Send", "发送"),
    (
        "Send to the configured provider (Ctrl/⌘+Enter)",
        "发送到已配置的提供商 (Ctrl/⌘+Enter)",
    ),
    ("Cancel request", "取消请求"),
    ("OFF — destination authorization retained", "关——保留目标地址授权"),
    ("OFF", "关"),
    ("ON — destination authorized", "开——目标地址已授权"),
    ("ON — confirmation required before transfer", "开——传输前需要确认"),
    ("Cloud image transfer", "云端图像传输"),
    ("Cloud image transfer: {status}", "云端图像传输：{status}"),
    ("Network disabled", "网络已禁用"),
    (
        "Explicitly enable provider network requests first.",
        "请先明确启用提供商网络请求。",
    ),
    (
        "Declare vision capability before authorizing an image preview.",
        "授权图像预览前，请先声明视觉能力。",
    ),
    ("Authorize cloud image transfer", "授权云端图像传输"),
    (
        "Only the currently rendered PNG preview will be sent to this provider endpoint. "
        "Burned-in text may contain private information. Keep this destination authorized "
        "until explicitly revoked?",
        "只会把当前渲染的 PNG 预览发送到此提供商端点。图像中固化的文字可能包含隐私信息。"
        "是否保持对此目标地址的授权，直到明确撤销？",
    ),
    ("Provider configuration error", "提供商配置错误"),
    (
        "Open an image before attaching the visible slice.",
        "附带当前可见切片前，请先打开影像。",
    ),
    ("Link blocked", "链接已阻止"),
    ("Only confirmed HTTP or HTTPS links can be opened.", "只能打开经确认的 HTTP 或 HTTPS 链接。"),
    ("Open external link", "打开外部链接"),
    (
        "Open this external link in your browser?\n\n{url}",
        "是否在浏览器中打开此外部链接？\n\n{url}",
    ),
    ("Request failed safely: {error}", "请求已安全失败：{error}"),
    ("Request cancelled.", "请求已取消。"),
    ("Cancelling request safely…", "正在安全取消请求…"),

    # Learning-experiment page shell and section headings.
    ("Principle", "原理"),
    ("Formula", "公式"),
    ("Parameter explanation", "参数说明"),
    ("Steps", "步骤"),
    ("Expected observation", "预期现象"),
    ("Common mistakes", "常见错误"),
    ("Reflection question", "思考题"),
    ("OpenMedVisionX Learning Experiments", "OpenMedVisionX 教学实验"),
    (
        "Choose an experiment in the staged learning path. Every page includes the same "
        "seven-part structure, and all suggested inputs can be generated arrays, phantoms, "
        "or mock tensors—no medical dataset is bundled.",
        "请在分阶段学习路径中选择一个实验。每个页面都采用相同的七部分结构，所有建议输入"
        "均可使用生成数组、仿体或模拟张量——不捆绑任何医学数据集。",
    ),
    (
        "Choose an experiment to review its principle, parameters, expected result, and "
        "reflection prompt. No medical dataset is bundled.",
        "选择一个实验，查看其原理、参数、预期结果与思考题。项目不捆绑任何医学数据集。",
    ),
    ("Experiment:", "实验："),
    (
        "OpenMedVisionX is local-first and for learning/research only—not for clinical use. "
        "Cloud image transfer remains off unless separately authorized in the assistant.",
        "OpenMedVisionX 采用本地优先设计，仅用于学习/研究，不可用于临床。"
        "除非在助手中单独授权，云端图像传输始终关闭。",
    ),

    # Experiment 1: 2-D pixels, bit depth, and interpolation.
    ("2-D pixels, bit depth, and interpolation", "二维像素、位深与插值"),
    (
        "A raster is a sampled signal. Bit depth controls representable values; color space "
        "defines channel meaning. Display mapping changes only the view, never decoded data.",
        "光栅图像是采样信号。位深决定可表示的数值；色彩空间定义通道含义。显示映射只改变视图，绝不改变解码数据。",
    ),
    (
        "Quantization step Δ = (Imax − Imin)/(2ᵇ − 1); bilinear interpolation is a weighted "
        "sum of four neighbours.",
        "量化步长 Δ = (Imax − Imin)/(2ᵇ − 1)；双线性插值是四个相邻像素的加权和。",
    ),
    (
        "b is bit depth; display bounds select a visible range. Spacing is pixel-only unless "
        "entered by the user.",
        "b 是位深；显示上下界用于选择可见范围。除非由用户输入，否则间距仅采用像素单位。",
    ),
    (
        "Open generated PNG/TIFF examples, inspect dtype and histogram, adjust the display "
        "range, zoom, and compare interpolation modes.",
        "打开生成的 PNG/TIFF 示例，检查数据类型和直方图，调整显示范围与缩放，并比较不同插值模式。",
    ),
    (
        "Higher bit depth preserves more levels; JPEG may add artifacts; display changes do "
        "not alter stored values.",
        "更高的位深可保留更多灰度级；JPEG 可能引入伪影；显示调整不会改变存储值。",
    ),
    (
        "Treating DPI as medical spacing, calling RGB values HU, or assuming a screenshot "
        "retains 16-bit values.",
        "把 DPI 当作医学间距、把 RGB 数值称为 HU，或认为截图会保留 16 位数值。",
    ),
    (
        "Which changes affect measurement data, and which affect only visualization?",
        "哪些更改会影响测量数据，哪些只影响可视化？",
    ),

    # Experiment 2: DICOM and NIfTI physical geometry.
    ("DICOM and NIfTI physical geometry", "DICOM 与 NIfTI 物理几何"),
    (
        "Medical volumes combine voxels with an affine. OpenMedVisionX displays validated "
        "volumes in RAS+ coordinates.",
        "医学体数据将体素与仿射变换结合。OpenMedVisionX 在 RAS+ 坐标系中显示经过验证的体数据。",
    ),
    (
        "x_world = A · [i, j, k, 1]ᵀ; DICOM intensity = stored value × slope + intercept.",
        "x_world = A · [i, j, k, 1]ᵀ；DICOM 强度 = 存储值 × 斜率 + 截距。",
    ),
    (
        "Spacing is physical sample distance; origin and direction locate axes; slope and "
        "intercept define quantitative intensity.",
        "间距是物理采样距离；原点和方向用于定位坐标轴；斜率和截距定义定量强度。",
    ),
    (
        "Open a generated volume, inspect geometry, navigate three orthogonal planes, and "
        "measure in millimetres.",
        "打开生成的体数据，检查几何信息，浏览三个正交平面，并以毫米为单位进行测量。",
    ),
    (
        "Anisotropic spacing changes physical aspect ratio; CT reports HU while ordinary "
        "rasters do not.",
        "各向异性间距会改变物理纵横比；CT 报告 HU，而普通光栅图像不报告 HU。",
    ),
    (
        "Sorting by filename, ignoring LPS→RAS conversion, or inventing volume geometry for "
        "a TIFF sequence.",
        "按文件名排序、忽略 LPS→RAS 转换，或为 TIFF 序列臆造体数据几何。",
    ),
    (
        "Why can equal-shaped arrays represent different anatomy orientations?",
        "为什么形状相同的数组可以表示不同的解剖方向？",
    ),

    # Experiment 3: Radon transform and filtered backprojection.
    (
        "Radon transform and filtered backprojection",
        "Radon 变换与滤波反投影",
    ),
    (
        "A sinogram contains line integrals. FBP filters detector profiles before "
        "backprojecting them over image space.",
        "正弦图包含线积分。FBP 先对探测器剖面进行滤波，再将其反投影到图像空间。",
    ),
    (
        "p(s,θ)=∫f(x,y)δ(s−x cosθ−y sinθ)dxdy; f≈B{F⁻¹[|ω|F(p)]}.",
        "p(s,θ)=∫f(x,y)δ(s−x cosθ−y sinθ)dxdy；f≈B{F⁻¹[|ω|F(p)]}。",
    ),
    (
        "Angles control sampling; circular support defines the object region; filters trade "
        "resolution against noise.",
        "角度控制采样；圆形支撑域定义对象区域；滤波器在分辨率与噪声之间进行权衡。",
    ),
    (
        "Generate a 180° sinogram, inspect projections and spectrum, reconstruct with BP/FBP, "
        "then compare filters and 360° redundancy.",
        "生成 180° 正弦图，检查投影和频谱，使用 BP/FBP 重建，然后比较滤波器与 360° 冗余。",
    ),
    (
        "BP blurs; ramp filtering sharpens; 360° parallel-beam data repeats 180° information "
        "with detector reversal.",
        "BP 会产生模糊；斜坡滤波会增强清晰度；360° 平行束数据通过探测器反向重复 180° 信息。",
    ),
    (
        "Mismatching angles, normalizing images independently before metrics, or changing "
        "circle/output geometry between stages.",
        "角度不匹配、在计算指标前分别归一化图像，或在不同阶段改变圆形支撑域/输出几何。",
    ),
    (
        "Why does a smoother filter reduce noise while softening edges?",
        "为什么更平滑的滤波器能降低噪声，却也会软化边缘？",
    ),

    # Experiment 4: SART iterative reconstruction.
    ("SART iterative reconstruction", "SART 迭代重建"),
    (
        "SART repeatedly corrects an image using the mismatch between measured and predicted "
        "projections.",
        "SART 利用实测投影与预测投影之间的差异反复校正图像。",
    ),
    (
        "xᵏ⁺¹ = xᵏ + λ C Aᵀ R (b − A xᵏ).",
        "xᵏ⁺¹ = xᵏ + λ C Aᵀ R (b − A xᵏ)。",
    ),
    (
        "Iterations set update count; relaxation λ controls update size; snapshots reveal "
        "convergence and artifacts.",
        "迭代次数设定更新次数；松弛因子 λ 控制更新步长；快照可展示收敛过程与伪影。",
    ),
    (
        "Use one sinogram, vary iteration count and relaxation, then compare snapshots and "
        "joint-range metrics.",
        "使用同一个正弦图，改变迭代次数与松弛因子，然后比较快照和联合范围指标。",
    ),
    (
        "Early iterations recover coarse structure; excessive updates can amplify streaks "
        "and noise.",
        "早期迭代可恢复粗略结构；过度更新可能放大条纹和噪声。",
    ),
    (
        "Judging a stretched display, using unstable λ, or interpreting convergence as "
        "clinical validity.",
        "依据拉伸后的显示作判断、使用不稳定的 λ，或把收敛误解为具有临床有效性。",
    ),
    (
        "How would noise level change your stopping criterion?",
        "噪声水平会如何改变您的停止准则？",
    ),

    # Experiment 5: external model inference.
    ("External model inference", "外部模型推理"),
    (
        "A manifest makes preprocessing, tensor semantics, runtime, licenses, and output "
        "coordinates explicit.",
        "清单会明确规定预处理、张量语义、运行时、许可证和输出坐标。",
    ),
    (
        "x_norm = (resize/crop(x)/scale − mean)/std; y_image = T⁻¹(y_model).",
        "x_norm = (resize/crop(x)/scale − mean)/std；y_image = T⁻¹(y_model)。",
    ),
    (
        "Layout, color order, dtype, resize, and normalization must match training; the "
        "TransformRecord maps results back.",
        "布局、颜色顺序、数据类型、尺寸调整和归一化必须与训练一致；"
        "TransformRecord 用于将结果映射回原空间。",
    ),
    (
        "Validate a manifest, inspect licenses and capabilities, explicitly load its local "
        "model, then run one rendered plane.",
        "验证清单，检查许可证和功能，明确加载其本地模型，然后在一个渲染平面上运行。",
    ),
    (
        "Typed scores, masks, boxes, or maps follow declared output semantics.",
        "类型化的分数、掩膜、边界框或图遵循声明的输出语义。",
    ),
    (
        "Loading a bare checkpoint, swapping RGB/BGR, stretching aspect ratio, or trusting "
        "unknown adapters.",
        "加载裸检查点、混淆 RGB/BGR、拉伸纵横比，或信任未知适配器。",
    ),
    (
        "Which manifest fields reproduce a prediction on another machine?",
        "哪些清单字段能够在另一台计算机上复现预测？",
    ),

    # Experiment 6: multimodal AI teaching assistant.
    ("Multimodal AI teaching assistant", "多模态 AI 教学助手"),
    (
        "A provider can discuss a reviewed rendered slice, but it is not a diagnostic system "
        "and may hallucinate.",
        "提供商可以讨论经过检查的渲染切片，但它不是诊断系统，并且可能产生幻觉。",
    ),
    (
        "response = Provider(model_id, prompt, optional rendered PNG); raw volumes and metadata "
        "are excluded.",
        "response = Provider(model_id, prompt, optional rendered PNG)；不包含原始体数据和元数据。",
    ),
    (
        "Endpoint and model ID are user supplied; network and image transfer are separate "
        "opt-ins; credentials remain references.",
        "端点和模型 ID 由用户提供；网络访问与图像传输需要分别选择启用；凭据始终以引用形式保存。",
    ),
    (
        "Choose a provider and model ID, inspect the preview, opt in to network and optional "
        "image transfer, then ask a concept question.",
        "选择提供商和模型 ID，检查预览，选择启用网络及可选的图像传输，然后提出概念问题。",
    ),
    (
        "The answer records provider/model/time and a disclaimer; revoking consent stops image "
        "transfer.",
        "回答会记录提供商/模型/时间和免责声明；撤销授权将停止图像传输。",
    ),
    (
        "Sending burned-in identifiers, treating text as diagnosis, or placing API keys in "
        "project files.",
        "发送图像中固化的标识符、把文本当作诊断，或将 API 密钥放入项目文件。",
    ),
    (
        "What evidence is needed before trusting an AI explanation of an imaging artifact?",
        "在信任 AI 对成像伪影的解释之前，需要哪些证据？",
    ),
)


def _build_translation_maps() -> tuple[dict[str, str], dict[str, str]]:
    english_to_chinese: dict[str, str] = {}
    chinese_to_english: dict[str, str] = {}
    for english, chinese in _TRANSLATIONS:
        existing_chinese = english_to_chinese.get(english)
        if existing_chinese is not None and existing_chinese != chinese:
            raise RuntimeError(f"Conflicting Chinese translations for {english!r}.")
        existing_english = chinese_to_english.get(chinese)
        if existing_english is not None and existing_english != english:
            raise RuntimeError(
                f"Chinese translation {chinese!r} is shared by {existing_english!r} "
                f"and {english!r}."
            )
        english_to_chinese[english] = chinese
        chinese_to_english[chinese] = english
    return english_to_chinese, chinese_to_english


_ENGLISH_TO_CHINESE, _CHINESE_TO_ENGLISH = _build_translation_maps()


def _is_named_template(template: str) -> bool:
    try:
        fields = [
            field_name
            for _literal, field_name, _format_spec, _conversion in Formatter().parse(template)
            if field_name is not None
        ]
    except ValueError:
        return False
    return bool(fields) and all(field_name.isidentifier() for field_name in fields)


def _compile_template(template: str) -> re.Pattern[str]:
    pieces: list[str] = []
    for literal, field_name, _format_spec, _conversion in Formatter().parse(template):
        pieces.append(re.escape(literal))
        if field_name is not None:
            pieces.append(f"(?P<{field_name}>.*?)")
    return re.compile("^" + "".join(pieces) + "$", re.DOTALL)


_TRANSLATABLE_TEMPLATE_FIELDS: dict[str, frozenset[str]] = {
    "Cloud image transfer: {status}": frozenset({"status"}),
    "Intermediate: {name}": frozenset({"name"}),
    "{message} – %p%": frozenset({"message"}),
}


_ENGLISH_TEMPLATES = tuple(
    (
        _compile_template(english),
        chinese,
        _TRANSLATABLE_TEMPLATE_FIELDS.get(english, frozenset()),
    )
    for english, chinese in _TRANSLATIONS
    if _is_named_template(english)
)
_CHINESE_TEMPLATES = tuple(
    (
        _compile_template(chinese),
        english,
        _TRANSLATABLE_TEMPLATE_FIELDS.get(english, frozenset()),
    )
    for english, chinese in _TRANSLATIONS
    if _is_named_template(chinese)
)


def _translate_template(
    text: str,
    language: Language,
    templates: tuple[tuple[re.Pattern[str], str, frozenset[str]], ...],
) -> str | None:
    for pattern, translated_template, translatable_fields in templates:
        match = pattern.fullmatch(text)
        if match is None:
            continue
        values: dict[str, str] = {}
        for name, value in match.groupdict().items():
            raw_value = value if value is not None else ""
            values[name] = (
                translate(raw_value, language)
                if name in translatable_fields
                else raw_value
            )
        return translated_template.format(**values)
    return None


def translate(text: str, language: Language) -> str:
    """Translate registered UI text, preserving unknown and technical text.

    English is the canonical source language.  Passing ``"en"`` also reverses
    any known Simplified Chinese translation, which makes repeated language
    toggles deterministic without storing the original widget text elsewhere.
    """

    if language == "zh_CN":
        exact = _ENGLISH_TO_CHINESE.get(text)
        if exact is not None:
            return exact
        return _translate_template(text, language, _ENGLISH_TEMPLATES) or text
    if language == "en":
        exact = _CHINESE_TO_ENGLISH.get(text)
        if exact is not None:
            return exact
        return _translate_template(text, language, _CHINESE_TEMPLATES) or text
    raise ValueError(f"Unsupported language: {language!r}")


__all__ = ["Language", "translate"]
