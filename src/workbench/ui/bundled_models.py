"""Qt catalog and reproducible experiment UI for reviewed bundled models."""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from PyQt5.QtCore import Qt, QTimer, pyqtSignal
from PyQt5.QtWidgets import (
    QComboBox,
    QFileDialog,
    QFrame,
    QGridLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QSizePolicy,
    QSplitter,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from .. import __version__
from ..cases import LodopabTeachingCase, load_lodopab_case
from ..domain.display import GrayscaleDisplayMapping
from ..errors import OperationCancelled
from ..evaluation import create_brats_experiment_record, export_experiment_record
from ..models import (
    BUNDLED_MODEL_IDS,
    ModelRunResult,
    list_bundled_models,
    load_bundle_record,
    run_dival_fbp_unet,
)
from ..runtime import BackgroundTask, TaskContext, TaskRunner
from ..services.brats_segmentation import (
    BraTSSegmentationConfig,
    BraTSSegmentationResult,
    run_brats2021_segmentation,
)
from ..services.bundled_models import (
    BundledDemoResult,
    TorchDeviceStatus,
    inspect_torch_device,
    run_deepinv_synthetic_demo,
)
from .brats_installer import BraTS2021InstallerDialog
from .i18n import Language
from .image_view import ImageView
from .widgets import SafeMarkdownBrowser

_DIVAL_ID = "dival-lodopab-fbpunet"
_DEEPINV_ID = "deepinv-mri-modl"
_MONAI_ID = "monai-brats-segmentation"
_CATALOG_ORDER = (
    _DIVAL_ID,
    _DEEPINV_ID,
    _MONAI_ID,
)
assert set(_CATALOG_ORDER) == set(BUNDLED_MODEL_IDS)

_USER_ROLE = int(Qt.UserRole)  # type: ignore[attr-defined]
_ACCESSIBLE_TEXT_ROLE = int(Qt.AccessibleTextRole)  # type: ignore[attr-defined]
_TEXT_SELECTABLE_BY_MOUSE = Qt.TextSelectableByMouse  # type: ignore[attr-defined]
_VERTICAL = Qt.Vertical  # type: ignore[attr-defined]
_HORIZONTAL = Qt.Horizontal  # type: ignore[attr-defined]

_COPY: dict[str, tuple[str, str]] = {
    "catalog_title": ("Bundled model catalog", "随包模型目录"),
    "catalog_accessible": (
        "Three reviewed built-in medical vision models",
        "三个经过审查的内置医学视觉模型",
    ),
    "intro": (
        "Start with the reviewed offline catalog. Inspect each input/output contract, "
        "then run the public LoDoPaB case or deterministic synthetic MRI demo. Models "
        "that need dedicated inputs remain inspectable and explicitly gated.",
        "从经过审查的离线模型目录开始。先检查每个模型的输入与输出契约，再运行公开的 "
        "LoDoPaB 案例或确定性的合成 MRI 演示。需要专用输入的模型仍可查看，并会明确保持门禁状态。",
    ),
    "device_label": ("Execution device", "执行设备"),
    "device_auto": ("Auto · prefer GPU", "自动 · 优先 GPU"),
    "device_cpu": ("CPU", "CPU"),
    "device_cuda": ("CUDA GPU", "CUDA GPU"),
    "device_pending": (
        "Device not inspected this session. Auto visibly falls back to CPU when needed.",
        "本次会话尚未检查设备；自动模式会在需要时明确回退到 CPU。",
    ),
    "verify_button": ("Verify catalog & device", "验证目录与设备"),
    "verify_accessible": (
        "Verify every bundled model artifact and inspect the local Torch device",
        "验证全部内置模型文件并检查本地 Torch 设备",
    ),
    "integrity_pending": (
        "Session verification pending · every model load is still integrity-gated",
        "会话验证待执行 · 每次模型加载仍会经过完整性门禁",
    ),
    "integrity_ok": (
        "3/3 models verified against registered SHA-256",
        "3/3 个模型已通过登记的 SHA-256 验证",
    ),
    "integrity_partial": (
        "Catalog verification did not complete",
        "模型目录验证未完成",
    ),
    "model_card_tab": ("Bilingual model card", "双语模型卡"),
    "experiment_tab": ("Experiment", "实验"),
    "details_accessible": ("Complete bundled model contract", "完整的内置模型契约"),
    "model_card_accessible": ("Bilingual bundled model card", "双语内置模型卡"),
    "run_dival": ("Run LoDoPaB case", "运行 LoDoPaB 案例"),
    "run_deepinv": ("Run synthetic MRI demo", "运行合成 MRI 演示"),
    "setup_brats": ("Set up local BraTS 2021…", "配置本地 BraTS 2021…"),
    "review_brats": ("Review BraTS 2021 setup…", "检查 BraTS 2021 配置…"),
    "run_brats": ("Run BraTS segmentation", "运行 BraTS 分割"),
    "run_unavailable": ("Dedicated input required", "需要专用输入"),
    "stop": ("Stop", "停止"),
    "export": ("Export experiment JSON…", "导出实验 JSON…"),
    "run_accessible": (
        "Run the bundled LoDoPaB teaching case with the selected device",
        "使用所选设备运行内置 LoDoPaB 教学案例",
    ),
    "run_deepinv_accessible": (
        "Run the deterministic image-derived MRI simulation with the selected device",
        "使用所选设备运行确定性的图像派生 MRI 模拟",
    ),
    "setup_brats_accessible": (
        "Validate a user-managed local BraTS 2021 case through the guided workflow",
        "通过引导流程校验由用户管理的本地 BraTS 2021 案例",
    ),
    "run_brats_accessible": (
        "Run full-volume WT, TC, and ET segmentation on the validated local case",
        "在已校验的本地案例上运行完整体 WT、TC 与 ET 分割",
    ),
    "export_accessible": (
        "Export a reproducible experiment record without image pixels or private paths",
        "导出不含图像像素或私有路径的可复现实验记录",
    ),
    "ready": (
        "Ready · public benchmark case · local execution · learning/research only",
        "已就绪 · 公开基准案例 · 本地执行 · 仅供学习与研究",
    ),
    "deepinv_ready": (
        "Ready · deterministic digital phantom · image-derived simulation · "
        "local execution · learning/research only",
        "已就绪 · 确定性数字模体 · 图像派生模拟 · 本地执行 · 仅供学习与研究",
    ),
    "catalog_only": (
        "This model is inspectable here, but its dedicated verified input workflow "
        "is not yet connected.",
        "可在此检查该模型，但其专用且经过验证的输入工作流尚未接通。",
    ),
    "deepinv_gate": (
        "Requires dedicated reviewed k-space and mask inputs; that complete workflow "
        "is not yet connected, so the model cannot run here.",
        "需要专用且经过审查的 k-space 与掩码输入；完整工作流尚未接通，因此当前不可运行。",
    ),
    "deepinv_demo_notice": (
        "The built-in demo deterministically derives k-space and a sampling mask from a "
        "digital phantom. It is image-derived simulation, never scanner raw data.",
        "内置演示从数字模体确定性生成 k-space 与采样掩码。它属于图像派生模拟，"
        "绝不是扫描仪原始数据。",
    ),
    "monai_gate": (
        "Requires four co-registered T1ce, T1, T2, and FLAIR volumes; the workflow "
        "never substitutes one visible slice.",
        "需要四个已配准的 T1ce、T1、T2 和 FLAIR 体数据；绝不会用单个可见切片替代。",
    ),
    "brats_setup_ready": (
        "Next: connect an authorized local BraTS 2021 case; no data will be downloaded.",
        "下一步：连接已获授权的本地 BraTS 2021 案例；不会下载任何数据。",
    ),
    "brats_validated": (
        "Local BraTS 2021 inputs passed geometry and label validation in this session.",
        "本次会话中的本地 BraTS 2021 输入已通过几何与标签校验。",
    ),
    "case_incompatible": (
        "The installed teaching-case API does not expose the reviewed Hann-FBP input. "
        "Update the bundled case assets before running.",
        "当前教学案例 API 未提供经过审查的 Hann-FBP 输入；请先更新内置案例资源。",
    ),
    "record_blocked": (
        "The packaged model record could not be validated.",
        "无法验证随包提供的模型记录。",
    ),
    "busy": (
        "Running in the background. The interface remains responsive.",
        "正在后台运行，界面仍可响应。",
    ),
    "cancelled": ("Background operation cancelled safely.", "后台操作已安全取消。"),
    "failed": ("Bundled model operation failed safely", "内置模型操作已安全失败"),
    "observation_view": ("LoDoPaB observation", "LoDoPaB 观测值"),
    "fbp_view": ("Reviewed Hann FBP", "经过审查的 Hann FBP"),
    "output_view": ("FBP-U-Net output", "FBP-U-Net 输出"),
    "ground_truth_view": ("Ground truth", "真值"),
    "lodopab_error_view": (
        "Absolute error · fixed reference scale",
        "绝对误差 · 固定参考量程",
    ),
    "lodopab_value_unit": ("normalized attenuation", "归一化衰减"),
    "mri_source_view": ("Digital phantom", "数字模体"),
    "mri_prepared_view": ("Zero-filled IFFT", "零填充 IFFT"),
    "mri_output_view": ("MoDL magnitude", "MoDL 幅值图"),
    "mri_error_view": ("Absolute error", "绝对误差"),
    "brats_wt_view": ("WT probability", "WT 概率"),
    "brats_tc_view": ("TC probability", "TC 概率"),
    "brats_et_view": ("ET probability", "ET 概率"),
    "brats_regions_view": ("WT / TC / ET masks", "WT / TC / ET 掩膜"),
    "empty_experiment": (
        "Run the LoDoPaB case to compare observation, FBP, model output, ground truth, "
        "and fixed-scale absolute error.",
        "运行 LoDoPaB 案例，以比较观测值、FBP、模型输出、真值与固定量程的绝对误差。",
    ),
    "empty_deepinv": (
        "Run the synthetic MRI demo to compare the phantom, zero-filled IFFT, MoDL "
        "magnitude, and absolute error.",
        "运行合成 MRI 演示，以比较数字模体、零填充 IFFT、MoDL 幅值图与绝对误差。",
    ),
    "empty_brats": (
        "Validate a local BraTS 2021 case, then run full-volume segmentation.",
        "先校验本地 BraTS 2021 案例，再运行完整体分割。",
    ),
    "export_dival_only": (
        "Run a reviewed LoDoPaB or BraTS experiment before exporting JSON.",
        "请先运行经过审查的 LoDoPaB 或 BraTS 实验，再导出 JSON。",
    ),
    "exported": ("Experiment JSON saved atomically", "实验 JSON 已原子保存"),
}

_TASK_LABELS: dict[str, tuple[str, str]] = {
    _DIVAL_ID: ("CT reconstruction post-processing", "CT 重建后处理"),
    "deepinv-mri-modl": ("Single-coil MRI reconstruction", "单线圈 MRI 重建"),
    "monai-brats-segmentation": ("Four-modal 3-D MRI segmentation", "四模态三维 MRI 分割"),
}

_LIMITATION_ZH = {
    _DIVAL_ID: (
        "该模型仅适用于 LoDoPaB-CT 基准分布与固定几何；不接受临床 HU 图像或任意扫描仪投影，"
        "也不是诊断模型。"
    ),
    "deepinv-mri-modl": (
        "这是用于解释欠采样与数据一致性的精简教学检查点；输入不是扫描仪原始 k-space，"
        "也未作为临床重建系统进行验证。"
    ),
    "monai-brats-segmentation": (
        "该模型基于 BraTS 2018 胶质瘤 MRI；未针对临床使用、其他疾病、扫描仪或 "
        "BraTS 2021 Task 1 进行外部验证。"
    ),
}

_CONTRACT_ZH: dict[str, dict[str, str]] = {
    _DIVAL_ID: {
        "input": "固定 DIVal LoDoPaB Hann 滤波 FBP，张量形状 1×1×362×362",
        "intensity": "LoDoPaB 归一化衰减图像域；不是临床 HU，也不是扫描仪原生投影",
        "output": "固定评估域中的单通道 FBP 后处理重建",
        "preprocessing": (
            "使用规定的 DIVal LoDoPaB FBP 算子；输入与输出不得各自自动窗宽窗位；不隐式缩放"
        ),
    },
    "deepinv-mri-modl": {
        "input": "模拟单线圈复数 k-space 与二值采样掩码，均为 B×2×H×W（实部、虚部）",
        "intensity": "归一化图像坐标中的图像派生模拟；不是扫描仪原始 k-space，不虚构物理间距",
        "output": "双通道复数重建图像；幅值图仅作为明确派生的显示结果",
        "preprocessing": (
            "正交 FFT；两次 HQS 数据一致性迭代与共享的两层 DnCNN；仅在重建后计算显示幅值"
        ),
    },
    "monai-brats-segmentation": {
        "input": "四个已配准 MRI 体数据，通道顺序 T1ce、T1、T2、FLAIR，形状 B×4×D×H×W",
        "intensity": "RAS+、1 mm 对齐体数据；每个通道在非零体素上进行 z-score 标准化",
        "output": "原生 sigmoid logits 顺序为 TC、WT、ET；产品适配器明确重排为 WT、TC、ET",
        "preprocessing": (
            "不隐式重采样；按通道进行非零体素 z-score；sigmoid 后使用可见阈值，并按源仿射映射结果"
        ),
    },
}

_WARNING_ZH = {
    "Input must come from the documented LoDoPaB FBP operator; it is not HU.": (
        "输入必须来自规定的 LoDoPaB FBP 算子；它不是 HU。"
    ),
    "Education and research only; not for diagnosis.": "仅供学习与研究，不得用于诊断。",
    "Pipeline: digital phantom -> single-coil FFT -> deterministic undersampling -> "
    "reviewed DeepInverse task adapter.": (
        "流程：数字模体 → 单线圈 FFT → 确定性欠采样 → 经过审查的 DeepInverse 任务适配器。"
    ),
    "The k-space is simulated from a digital phantom; it is not scanner raw data.": (
        "k-space 由数字模体模拟生成，并非扫描仪原始数据。"
    ),
    "Synthetic phantom simulation; not scanner raw data and not a clinical acquisition.": (
        "合成模体模拟；并非扫描仪原始数据，也不是临床采集。"
    ),
    "Synthetic phantom -> single-coil FFT -> deterministic undersampling mask -> "
    "reviewed DeepInverse task adapter.": (
        "合成模体 → 单线圈 FFT → 确定性欠采样掩码 → 经过审查的 DeepInverse 任务适配器。"
    ),
    "The k-space is simulated from an image phantom; it is not scanner raw data.": (
        "k-space 由图像模体模拟生成，并非扫描仪原始数据。"
    ),
}

_PROGRESS_ZH = {
    "Verifying packaged model integrity": "正在验证随包模型的完整性",
    "Inspecting the local execution device": "正在检查本地执行设备",
    "Loading the reviewed LoDoPaB teaching case": "正在加载经过审查的 LoDoPaB 教学案例",
    "Running the reviewed DIVal FBP-U-Net": "正在运行经过审查的 DIVal FBP-U-Net",
    "Computing reference metrics": "正在计算参考指标",
    "Writing reproducible experiment JSON": "正在写入可复现实验 JSON",
    "Generating a synthetic image phantom": "正在生成合成图像模体",
    "Simulating single-coil k-space and sampling mask": "正在模拟单线圈 k-space 与采样掩码",
    "Running the reviewed DeepInverse task adapter": "正在运行经过审查的 DeepInverse 任务适配器",
    "Synthetic MRI demonstration complete": "合成 MRI 演示已完成",
    "Revalidating the selected local BraTS case": "正在重新校验所选本地 BraTS 案例",
    "Applying auditable nonzero-voxel z-score normalization": (
        "正在执行可审计的非零体素 z-score 标准化"
    ),
    "Loading the reviewed local MONAI model": "正在加载经审查的本地 MONAI 模型",
    "Running full-volume sliding-window inference": "正在运行完整体滑窗推理",
    "Reordering TC/WT/ET to WT/TC/ET and thresholding": (
        "正在将 TC/WT/ET 重排为 WT/TC/ET 并应用阈值"
    ),
    "Evaluating Task 1 WT, TC, and ET regions": "正在评价 Task 1 的 WT、TC 与 ET 区域",
    "BraTS full-volume segmentation complete": "BraTS 完整体分割已完成",
    "Building the pixel-free BraTS experiment record": "正在构建无像素的 BraTS 实验记录",
    "Writing the pixel-free BraTS experiment record": "正在写入无像素的 BraTS 实验记录",
}


def _copy(key: str, language: Language) -> str:
    return _COPY[key][1 if language == "zh_CN" else 0]


def _repolish(widget: QWidget) -> None:
    style = widget.style()
    if style is None:
        return
    style.unpolish(widget)
    style.polish(widget)
    widget.update()


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _basic_metrics(output: np.ndarray, reference: np.ndarray) -> dict[str, float]:
    output64 = np.asarray(output, dtype=np.float64)
    reference64 = np.asarray(reference, dtype=np.float64)
    if output64.shape != reference64.shape:
        raise ValueError("Model output and ground truth must have the same shape")
    difference = output64 - reference64
    mse = float(np.mean(np.square(difference), dtype=np.float64))
    rmse = math.sqrt(mse)
    data_range = float(np.max(reference64) - np.min(reference64))
    psnr = math.inf if mse == 0.0 else 20.0 * math.log10(data_range / rmse)
    return {
        "mae": float(np.mean(np.abs(difference), dtype=np.float64)),
        "rmse": rmse,
        "psnr_db": psnr,
        "reference_data_range": data_range,
    }


def _lodopab_reference_mappings(
    ground_truth: np.ndarray,
) -> tuple[GrayscaleDisplayMapping, GrayscaleDisplayMapping]:
    """Return output-independent image and absolute-error display mappings.

    The complete, integrity-checked ground-truth domain is the comparison authority.
    Model output is deliberately excluded so an outlier cannot visually flatten the
    FBP/reference comparison.  Absolute error remains in the original normalized-
    attenuation units and uses the reference span as its fixed display ceiling.
    """

    reference = np.asarray(ground_truth)
    if reference.ndim != 2 or not np.issubdtype(reference.dtype, np.number):
        raise ValueError("LoDoPaB ground truth must be a two-dimensional numeric image")
    if not np.isfinite(reference).all():
        raise ValueError("LoDoPaB ground truth must contain only finite values")
    lower = float(np.min(reference))
    upper = float(np.max(reference))
    if upper <= lower:
        upper = lower + 1.0
    comparison = GrayscaleDisplayMapping(lower, upper)
    absolute_error = GrayscaleDisplayMapping(0.0, upper - lower)
    return comparison, absolute_error


@dataclass(frozen=True, slots=True)
class _VerificationResult:
    records: tuple[Any, ...]
    device: TorchDeviceStatus | None
    device_error: str | None


@dataclass(frozen=True, slots=True)
class BundledExperimentResult:
    case: Any
    model_result: ModelRunResult
    output: np.ndarray
    metrics: Mapping[str, float]
    requested_device: str
    total_elapsed_seconds: float
    completed_at_utc: str
    manifest_sha256: str
    weight_sha256: str


@dataclass(frozen=True, slots=True)
class _DemoExecutionResult:
    result: BundledDemoResult
    requested_device: str
    total_elapsed_seconds: float
    completed_at_utc: str


class BundledModelsPanel(QWidget):
    """Reviewed catalog with safe LoDoPaB and synthetic MRI workflows."""

    statusChanged = pyqtSignal(str)

    def __init__(
        self,
        parent: QWidget | None = None,
        *,
        record_loader: Callable[[str], Any] = load_bundle_record,
        bundle_verifier: Callable[[str], Any] | None = None,
        device_probe: Callable[[TaskContext], TorchDeviceStatus] = inspect_torch_device,
        case_loader: Callable[[], Any] = load_lodopab_case,
        dival_runner: Callable[..., ModelRunResult] = run_dival_fbp_unet,
        demo_operations: Mapping[str, Callable[..., BundledDemoResult]] | None = None,
        brats_dialog_factory: Callable[..., BraTS2021InstallerDialog] = (BraTS2021InstallerDialog),
        brats_runner: Callable[..., BraTSSegmentationResult] = run_brats2021_segmentation,
    ) -> None:
        super().__init__(parent)
        self._language: Language = "en"
        self._record_loader = record_loader
        self._bundle_verifier = bundle_verifier
        self._device_probe = device_probe
        self._case_loader = case_loader
        self._dival_runner = dival_runner
        self._demo_operations = dict(
            {_DEEPINV_ID: run_deepinv_synthetic_demo}
            if demo_operations is None
            else demo_operations
        )
        self._brats_dialog_factory = brats_dialog_factory
        self._brats_runner = brats_runner
        self._brats_case_directory: Path | None = None
        self._brats_validation_report: Any | None = None
        self._last_brats_result: BraTSSegmentationResult | None = None
        self._case_api_supports_fbp = case_loader is not load_lodopab_case or hasattr(
            LodopabTeachingCase, "fbp"
        )
        self._records: dict[str, Any] = {}
        self._record_errors: dict[str, str] = {}
        self._verified_ids: set[str] = set()
        self._latest_device: TorchDeviceStatus | None = None
        self._device_error: str | None = None
        self._last_experiment: BundledExperimentResult | None = None
        self._last_demo_execution: _DemoExecutionResult | None = None
        self._current_selection_id: str | None = None
        self._displayed_bundle_id: str | None = None
        self._active_task: BackgroundTask[Any] | None = None
        self._active_operation = ""
        self._last_progress: tuple[float, str] | None = None
        self._status_pair = _COPY["ready"]
        self._status_state = "ready"
        self._runner = TaskRunner(max_workers=1, thread_name_prefix="openmedvisionx-bundled")
        self._poll_timer = QTimer(self)
        self._poll_timer.setInterval(60)
        self._poll_timer.timeout.connect(self._poll_task)
        self._build_ui()
        self._load_catalog_records()
        self.set_language("en")

    @property
    def language(self) -> Language:
        return self._language

    @property
    def selected_bundle_id(self) -> str:
        item = self.catalog.currentItem()
        return str(item.data(_USER_ROLE)) if item is not None else _CATALOG_ORDER[0]

    @property
    def active_task(self) -> BackgroundTask[Any] | None:
        return self._active_task

    @property
    def last_experiment(self) -> BundledExperimentResult | None:
        return self._last_experiment

    @property
    def last_demo(self) -> BundledDemoResult | None:
        execution = self._last_demo_execution
        return None if execution is None else execution.result

    @property
    def brats_case_directory(self) -> Path | None:
        """Return the validated user-managed BraTS directory for this session."""

        return self._brats_case_directory

    @property
    def brats_validation_report(self) -> Any | None:
        """Return the last valid BraTS report accepted by the model workspace."""

        return self._brats_validation_report

    @property
    def last_brats_result(self) -> BraTSSegmentationResult | None:
        """Return the latest completed local BraTS segmentation result."""

        return self._last_brats_result

    def _build_ui(self) -> None:
        self.setMinimumSize(0, 0)
        self.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Expanding)
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 8, 0, 0)
        root.setSpacing(10)

        self.intro = QLabel()
        self.intro.setObjectName("infoBanner")
        self.intro.setWordWrap(True)
        root.addWidget(self.intro)

        self.device_surface = QFrame()
        self.device_surface.setObjectName("toolbarSurface")
        self.device_surface.setMinimumWidth(0)
        self.device_surface.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        self.device_layout = QGridLayout(self.device_surface)
        self.device_layout.setContentsMargins(10, 8, 10, 8)
        self.device_layout.setHorizontalSpacing(8)
        self.device_layout.setVerticalSpacing(6)
        self.device_label = QLabel()
        self.device_combo = QComboBox()
        self.device_combo.addItem("", "auto")
        self.device_combo.addItem("", "cpu")
        self.device_combo.addItem("", "cuda")
        self.verify_button = QPushButton()
        self.verify_button.clicked.connect(self._verify_models_and_device)
        self.device_status = QLabel()
        self.device_status.setObjectName("mutedText")
        self.device_status.setWordWrap(True)
        self.device_status.setTextInteractionFlags(_TEXT_SELECTABLE_BY_MOUSE)
        self.integrity_status = QLabel()
        self.integrity_status.setObjectName("actionStatus")
        self.integrity_status.setWordWrap(True)
        self.integrity_status.setTextInteractionFlags(_TEXT_SELECTABLE_BY_MOUSE)
        root.addWidget(self.device_surface)

        self.main_splitter = QSplitter(_HORIZONTAL)
        self.main_splitter.setChildrenCollapsible(False)
        self.main_splitter.setMinimumWidth(0)
        self.main_splitter.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Expanding)
        catalog_panel = QWidget()
        catalog_panel.setMinimumWidth(0)
        catalog_panel.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Expanding)
        catalog_layout = QVBoxLayout(catalog_panel)
        catalog_layout.setContentsMargins(0, 0, 0, 0)
        catalog_layout.setSpacing(6)
        self.catalog_title = QLabel()
        self.catalog_title.setObjectName("pageTitle")
        self.catalog = QListWidget()
        self.catalog.setObjectName("modelCatalog")
        self.catalog.setWordWrap(True)
        self.catalog.setMinimumWidth(0)
        self.catalog.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Expanding)
        self.catalog.currentItemChanged.connect(self._selection_changed)
        catalog_layout.addWidget(self.catalog_title)
        catalog_layout.addWidget(self.catalog, 1)
        self.main_splitter.addWidget(catalog_panel)

        detail_panel = QWidget()
        detail_panel.setMinimumWidth(0)
        detail_panel.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Expanding)
        detail_layout = QVBoxLayout(detail_panel)
        detail_layout.setContentsMargins(0, 0, 0, 0)
        detail_layout.setSpacing(8)
        self.model_name = QLabel()
        self.model_name.setObjectName("pageTitle")
        self.model_name.setWordWrap(True)
        self.contract_details = QPlainTextEdit()
        self.contract_details.setObjectName("bundleContract")
        self.contract_details.setReadOnly(True)
        self.contract_details.setLineWrapMode(QPlainTextEdit.WidgetWidth)
        self.contract_details.setMinimumHeight(118)
        self.contract_details.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        self.limitation = QLabel()
        self.limitation.setObjectName("warningBanner")
        self.limitation.setWordWrap(True)
        self.limitation.setTextInteractionFlags(_TEXT_SELECTABLE_BY_MOUSE)
        detail_layout.addWidget(self.model_name)
        detail_layout.addWidget(self.contract_details)
        detail_layout.addWidget(self.limitation)

        actions = QFrame()
        actions.setObjectName("toolbarSurface")
        actions.setMinimumWidth(0)
        actions.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        self.action_layout = QGridLayout(actions)
        self.action_layout.setContentsMargins(10, 8, 10, 8)
        self.action_layout.setHorizontalSpacing(8)
        self.action_layout.setVerticalSpacing(8)
        self.run_button = QPushButton()
        self.run_button.setObjectName("primary")
        self.run_button.clicked.connect(self._run_experiment)
        self.stop_button = QPushButton()
        self.stop_button.clicked.connect(self._cancel_active)
        self.stop_button.setEnabled(False)
        self.export_button = QPushButton()
        self.export_button.clicked.connect(self._choose_export_path)
        self.export_button.setEnabled(False)
        self.configure_brats_button = QPushButton()
        self.configure_brats_button.clicked.connect(self._open_brats_installer)
        self.configure_brats_button.setVisible(False)
        detail_layout.addWidget(actions)

        self.operation_status = QLabel()
        self.operation_status.setObjectName("actionStatus")
        self.operation_status.setWordWrap(True)
        self.progress = QProgressBar()
        self.progress.setVisible(False)
        detail_layout.addWidget(self.operation_status)
        detail_layout.addWidget(self.progress)

        self.detail_tabs = QTabWidget()
        self.detail_tabs.setObjectName("modelOutputTabs")
        self.detail_tabs.setMinimumWidth(0)
        self.detail_tabs.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Expanding)
        self.model_card = SafeMarkdownBrowser()
        self.model_card.setObjectName("modelCard")
        self.model_card.setMinimumWidth(0)
        self.model_card.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Expanding)
        self.detail_tabs.addTab(self.model_card, "")

        experiment_page = QWidget()
        experiment_layout = QVBoxLayout(experiment_page)
        experiment_layout.setContentsMargins(0, 6, 0, 0)
        experiment_layout.setSpacing(7)
        self.demo_views = QSplitter(_VERTICAL)
        self.demo_views.setChildrenCollapsible(False)
        self.demo_top_row = QSplitter(_HORIZONTAL)
        self.demo_bottom_row = QSplitter(_HORIZONTAL)
        self.observation_view = ImageView("LoDoPaB observation")
        self.fbp_view = ImageView("Reviewed Hann FBP")
        self.output_view = ImageView("FBP-U-Net output")
        self.ground_truth_view = ImageView("Ground truth")
        self.lodopab_error_view = ImageView("Absolute error · fixed reference scale")
        # Compatibility aliases for integrations that previously inspected the demo views.
        self.source_view = self.observation_view
        self.prepared_view = self.fbp_view
        for view in (
            self.observation_view,
            self.fbp_view,
            self.output_view,
            self.ground_truth_view,
            self.lodopab_error_view,
        ):
            view.setMinimumSize(100, 92)
            view.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Expanding)
        self.demo_top_row.addWidget(self.observation_view)
        self.demo_top_row.addWidget(self.fbp_view)
        self.demo_bottom_row.addWidget(self.output_view)
        self.demo_bottom_row.addWidget(self.ground_truth_view)
        self.demo_bottom_row.addWidget(self.lodopab_error_view)
        self.demo_views.addWidget(self.demo_top_row)
        self.demo_views.addWidget(self.demo_bottom_row)
        experiment_layout.addWidget(self.demo_views, 1)
        self.result_details = QLabel()
        self.result_details.setObjectName("mutedText")
        self.result_details.setWordWrap(True)
        self.result_details.setTextInteractionFlags(_TEXT_SELECTABLE_BY_MOUSE)
        experiment_layout.addWidget(self.result_details)
        self.detail_tabs.addTab(experiment_page, "")
        detail_layout.addWidget(self.detail_tabs, 1)
        self.main_splitter.addWidget(detail_panel)
        self.main_splitter.setStretchFactor(0, 1)
        self.main_splitter.setStretchFactor(1, 3)
        self.main_splitter.setSizes([250, 680])
        root.addWidget(self.main_splitter, 1)

        self._layout_device_controls()
        self._layout_action_controls()

    def _load_catalog_records(self) -> None:
        self.catalog.clear()
        for bundle_id in _CATALOG_ORDER:
            try:
                self._records[bundle_id] = self._record_loader(bundle_id)
            except BaseException as exc:  # malformed package remains visible but cannot run
                self._record_errors[bundle_id] = str(exc)
            item = QListWidgetItem()
            item.setData(_USER_ROLE, bundle_id)
            self.catalog.addItem(item)
        if self.catalog.count():
            self.catalog.setCurrentRow(0)

    def set_language(self, language: Language) -> None:
        if language not in {"en", "zh_CN"}:
            raise ValueError(f"Unsupported UI language: {language}")
        self._language = language
        self.intro.setText(_copy("intro", language))
        self.catalog_title.setText(_copy("catalog_title", language))
        self.catalog.setAccessibleName(_copy("catalog_accessible", language))
        self.device_label.setText(_copy("device_label", language))
        for index, key in enumerate(("device_auto", "device_cpu", "device_cuda")):
            self.device_combo.setItemText(index, _copy(key, language))
        self.device_combo.setAccessibleName(_copy("device_label", language))
        self.verify_button.setText(_copy("verify_button", language))
        self.verify_button.setAccessibleName(_copy("verify_accessible", language))
        self.detail_tabs.setTabText(0, _copy("model_card_tab", language))
        self.detail_tabs.setTabText(1, _copy("experiment_tab", language))
        self.contract_details.setAccessibleName(_copy("details_accessible", language))
        self.model_card.setAccessibleName(_copy("model_card_accessible", language))
        self.stop_button.setText(_copy("stop", language))
        self.export_button.setText(_copy("export", language))
        self.export_button.setAccessibleName(_copy("export_accessible", language))
        self.configure_brats_button.setText(_copy("review_brats", language))
        self.configure_brats_button.setAccessibleName(_copy("setup_brats_accessible", language))
        for view in (
            self.observation_view,
            self.fbp_view,
            self.output_view,
            self.ground_truth_view,
            self.lodopab_error_view,
        ):
            view.set_language(language)
        self._refresh_catalog_text()
        self._refresh_device_text()
        self._refresh_integrity_text()
        self._refresh_selection(reset_status=False)
        if self._displayed_bundle_id == _DIVAL_ID and self._last_experiment is not None:
            self._refresh_experiment_text(self._last_experiment)
            self._refresh_lodopab_scale_labels(self._last_experiment)
        elif self._displayed_bundle_id == _DEEPINV_ID and self._last_demo_execution is not None:
            self._refresh_demo_text(self._last_demo_execution)
        elif self._displayed_bundle_id == _MONAI_ID and self._last_brats_result is not None:
            self._refresh_brats_text(self._last_brats_result)
        self._render_status()

    def _record_name(self, bundle_id: str, record: Any | None) -> str:
        fallback = {
            _DIVAL_ID: "DIVal LoDoPaB FBP-U-Net",
            "deepinv-mri-modl": "DeepInverse MRI MoDL",
            "monai-brats-segmentation": "MONAI BraTS 3D Segmentation",
        }[bundle_id]
        attribute = "display_name_zh" if self._language == "zh_CN" else "display_name"
        return str(getattr(record, attribute, fallback))

    def _refresh_catalog_text(self) -> None:
        for index in range(self.catalog.count()):
            item = self.catalog.item(index)
            if item is None:
                continue
            bundle_id = str(item.data(_USER_ROLE))
            record = self._records.get(bundle_id)
            name = self._record_name(bundle_id, record)
            task = _TASK_LABELS[bundle_id][1 if self._language == "zh_CN" else 0]
            if bundle_id in self._record_errors:
                state = "记录错误" if self._language == "zh_CN" else "record error"
            elif bundle_id in self._verified_ids or bool(getattr(record, "verified", False)):
                state = "已验证" if self._language == "zh_CN" else "verified"
            elif self._language == "zh_CN":
                state = "运行时完整性门禁"
            else:
                state = "runtime integrity gate"
            item.setText(f"{name}\n{task} · {state}")
            item.setToolTip(self._record_errors.get(bundle_id, task))
            item.setData(_ACCESSIBLE_TEXT_ROLE, f"{name}; {task}; {state}")

    def _selection_changed(
        self,
        _current: QListWidgetItem | None,
        _previous: QListWidgetItem | None,
    ) -> None:
        self._refresh_selection()

    def _refresh_selection(self, *, reset_status: bool = True) -> None:
        if self.catalog.currentItem() is None:
            return
        bundle_id = self.selected_bundle_id
        selection_changed = bundle_id != self._current_selection_id
        self._current_selection_id = bundle_id
        if selection_changed and self._displayed_bundle_id != bundle_id:
            for view in (
                self.observation_view,
                self.fbp_view,
                self.output_view,
                self.ground_truth_view,
                self.lodopab_error_view,
            ):
                view.clear_image()
            self.result_details.clear()
            self._displayed_bundle_id = None
        record = self._records.get(bundle_id)
        self.model_name.setText(self._record_name(bundle_id, record))
        limitation = (
            _LIMITATION_ZH[bundle_id]
            if self._language == "zh_CN"
            else str(getattr(record, "limitations", "Limitations unavailable."))
        )
        if bundle_id == _MONAI_ID:
            limitation = f"{limitation}\n\n{_copy('monai_gate', self._language)}"
        elif bundle_id == _DEEPINV_ID:
            notice_key = (
                "deepinv_demo_notice" if bundle_id in self._demo_operations else "deepinv_gate"
            )
            limitation = f"{limitation}\n\n{_copy(notice_key, self._language)}"
        self.limitation.setText(limitation)
        self.contract_details.setPlainText(self._contract_text(bundle_id, record))
        self.model_card.set_markdown_text(self._read_model_card(bundle_id, record))
        if bundle_id == _DIVAL_ID:
            run_key = "run_dival"
            accessible_key = "run_accessible"
        elif bundle_id == _DEEPINV_ID and bundle_id in self._demo_operations:
            run_key = "run_deepinv"
            accessible_key = "run_deepinv_accessible"
        elif bundle_id == _MONAI_ID:
            run_key = "run_brats" if self._brats_case_directory is not None else "setup_brats"
            accessible_key = (
                "run_brats_accessible"
                if self._brats_case_directory is not None
                else "setup_brats_accessible"
            )
        else:
            run_key = "run_unavailable"
            accessible_key = "monai_gate" if bundle_id == _MONAI_ID else "catalog_only"
        self.run_button.setText(_copy(run_key, self._language))
        self.run_button.setAccessibleName(_copy(accessible_key, self._language))
        self.configure_brats_button.setVisible(
            bundle_id == _MONAI_ID and self._brats_case_directory is not None
        )
        self._layout_action_controls()
        self._refresh_view_semantics(bundle_id)
        experiment_available = (
            bundle_id == _DIVAL_ID
            or bundle_id in self._demo_operations
            or (bundle_id == _MONAI_ID and self._brats_case_directory is not None)
        )
        self.detail_tabs.setTabEnabled(1, experiment_available)
        if not experiment_available:
            self.detail_tabs.setCurrentIndex(0)
        self._update_run_gate(reset_status=reset_status)

    def _refresh_view_semantics(self, bundle_id: str) -> None:
        is_lodopab = bundle_id == _DIVAL_ID
        self.lodopab_error_view.setVisible(is_lodopab)
        if bundle_id == _DEEPINV_ID:
            keys = (
                "mri_source_view",
                "mri_prepared_view",
                "mri_output_view",
                "mri_error_view",
            )
            empty_key = "empty_deepinv"
        elif bundle_id == _MONAI_ID:
            keys = (
                "brats_wt_view",
                "brats_tc_view",
                "brats_et_view",
                "brats_regions_view",
            )
            empty_key = "empty_brats"
        else:
            keys = (
                "observation_view",
                "fbp_view",
                "output_view",
                "ground_truth_view",
            )
            empty_key = "empty_experiment"
        for key, view in zip(
            keys,
            (
                self.observation_view,
                self.fbp_view,
                self.output_view,
                self.ground_truth_view,
            ),
            strict=True,
        ):
            view.set_title(_copy(key, self._language))
            view.set_empty_message(_copy(empty_key, self._language))
        if is_lodopab:
            self.lodopab_error_view.set_title(_copy("lodopab_error_view", self._language))
            self.lodopab_error_view.set_empty_message(_copy("empty_experiment", self._language))

    def _contract_text(self, bundle_id: str, record: Any | None) -> str:
        if record is None:
            error = self._record_errors.get(bundle_id, "unknown record error")
            return f"Bundle ID / 模型 ID: {bundle_id}\nRecord error / 记录错误: {error}"
        contract = _mapping(getattr(record, "task_contract", {}))
        resources = _mapping(getattr(record, "resources", {}))
        if self._language == "zh_CN":
            translated = _CONTRACT_ZH[bundle_id]
            description = str(getattr(record, "description_zh", "—"))
            input_semantics = translated["input"]
            intensity = translated["intensity"]
            output = translated["output"]
            preprocessing = translated["preprocessing"]
            if bundle_id in self._verified_ids:
                verified = "是"
            else:
                verified = "待本次会话验证（运行时仍强制验证）"
            labels = {
                "description": "说明",
                "task": "任务",
                "input": "输入布局与语义",
                "range": "方向、间距与强度范围",
                "preprocess": "预处理与后处理",
                "output": "输出语义",
                "device": "设备策略",
                "license": "权重许可",
                "source": "来源",
                "limitations": "限制",
                "verified": "已验证",
            }
        else:
            description = str(getattr(record, "description_en", "—"))
            input_semantics = str(contract.get("input_semantics", "—"))
            intensity = str(contract.get("spacing_orientation_intensity", "—"))
            output = str(contract.get("labels_and_output_semantics", "—"))
            preprocessing = str(contract.get("preprocessing_postprocessing", "—"))
            verified = (
                "yes"
                if bundle_id in self._verified_ids
                else "pending this session (runtime enforced)"
            )
            labels = {
                "description": "Description",
                "task": "Task",
                "input": "Input layout & semantics",
                "range": "Orientation, spacing & intensity range",
                "preprocess": "Preprocessing & postprocessing",
                "output": "Output semantics",
                "device": "Device policy",
                "license": "Weights license",
                "source": "Source",
                "limitations": "Limitations",
                "verified": "Verified",
            }
        task = _TASK_LABELS[bundle_id][1 if self._language == "zh_CN" else 0]
        modalities = ", ".join(str(value) for value in getattr(record, "modalities", ()))
        dimensionality = str(getattr(record, "dimensionality", "—"))
        device_backend = resources.get(
            "gpu_backend",
            getattr(record, "runtime", "—"),
        )
        device = f"{device_backend}; {resources.get('expected_runtime_cpu_gpu', '—')}"
        return "\n".join(
            (
                f"{labels['description']}: {description}",
                f"{labels['task']}: {task}",
                f"{labels['input']}: {dimensionality}; {modalities}; {input_semantics}",
                f"{labels['range']}: {intensity}",
                f"{labels['preprocess']}: {preprocessing}",
                f"{labels['output']}: {output}",
                f"{labels['device']}: {device}",
                f"{labels['license']}: {getattr(record, 'license', '—')}",
                f"{labels['source']}: {getattr(record, 'source_url', '—')}",
                f"{labels['limitations']}: {getattr(record, 'limitations', '—')}",
                f"{labels['verified']}: {verified}",
                f"Bundle ID: {bundle_id}",
                f"Weight SHA-256: {getattr(record, 'artifact_sha256', '—')}",
            )
        )

    @staticmethod
    def _read_model_card(bundle_id: str, record: Any | None) -> str:
        manifest = getattr(record, "manifest_path", getattr(record, "record_path", None))
        path = (
            Path(manifest).parent / "MODEL_CARD.md"
            if manifest is not None
            else Path(__file__).resolve().parents[1]
            / "resources"
            / "model_bundles"
            / bundle_id
            / "MODEL_CARD.md"
        )
        try:
            return path.read_text(encoding="utf-8")
        except (OSError, UnicodeError) as exc:
            return f"# Model card unavailable / 模型卡不可用\n\n{exc}"

    def _update_run_gate(self, *, reset_status: bool = True) -> None:
        bundle_id = self.selected_bundle_id
        if self._active_task is not None:
            enabled = False
            reason = _copy("busy", self._language)
            status_key = "busy"
        elif bundle_id in self._record_errors:
            enabled = False
            reason = _copy("record_blocked", self._language)
            status_key = "record_blocked"
        elif bundle_id == _MONAI_ID:
            enabled = True
            if self._brats_case_directory is None:
                reason = _copy("brats_setup_ready", self._language)
                status_key = "brats_setup_ready"
            else:
                reason = _copy("brats_validated", self._language)
                status_key = "brats_validated"
        elif bundle_id == _DEEPINV_ID and bundle_id not in self._demo_operations:
            enabled = False
            reason = _copy("deepinv_gate", self._language)
            status_key = "deepinv_gate"
        elif bundle_id == _DIVAL_ID and not self._case_api_supports_fbp:
            enabled = False
            reason = _copy("case_incompatible", self._language)
            status_key = "case_incompatible"
        elif bundle_id not in {_DIVAL_ID, _DEEPINV_ID}:
            enabled = False
            reason = _copy("catalog_only", self._language)
            status_key = "catalog_only"
        else:
            enabled = True
            status_key = "deepinv_ready" if bundle_id == _DEEPINV_ID else "ready"
            reason = _copy(status_key, self._language)
        self.run_button.setEnabled(enabled)
        self.run_button.setToolTip("" if enabled else reason)
        self.run_button.setAccessibleDescription(reason)
        self.configure_brats_button.setEnabled(
            bundle_id == _MONAI_ID
            and self._brats_case_directory is not None
            and self._active_task is None
        )
        self.configure_brats_button.setAccessibleDescription(
            _copy("brats_validated", self._language)
        )
        self.export_button.setEnabled(
            (
                (bundle_id == _DIVAL_ID and self._last_experiment is not None)
                or (bundle_id == _MONAI_ID and self._last_brats_result is not None)
            )
            and self._active_task is None
        )
        if self.export_button.isEnabled():
            export_hint = ""
        elif bundle_id not in {_DIVAL_ID, _MONAI_ID}:
            export_hint = _copy("export_dival_only", self._language)
        elif bundle_id == _MONAI_ID:
            export_hint = _copy("empty_brats", self._language)
        else:
            export_hint = _copy("empty_experiment", self._language)
        self.export_button.setToolTip(export_hint)
        if self._active_task is None and reset_status:
            state = (
                "ready"
                if status_key in {"ready", "deepinv_ready", "brats_validated"}
                else "next"
                if status_key == "brats_setup_ready"
                else "blocked"
            )
            self._set_status_pair(_COPY[status_key], state)

    def _refresh_device_text(self) -> None:
        if self._device_error:
            self.device_status.setText(self._device_error)
            return
        status = self._latest_device
        if status is None:
            self.device_status.setText(_copy("device_pending", self._language))
            return
        if status.cuda_available:
            memory = "" if status.total_vram_gib is None else f" · {status.total_vram_gib:.1f} GiB"
            available = "CUDA 可用" if self._language == "zh_CN" else "CUDA available"
            self.device_status.setText(
                f"PyTorch {status.torch_version} · {available} · {status.device_name}{memory}"
            )
        else:
            unavailable = (
                "CUDA 不可用 · 自动模式将使用 CPU"
                if self._language == "zh_CN"
                else "CUDA unavailable · Auto will use CPU"
            )
            self.device_status.setText(f"PyTorch {status.torch_version} · {unavailable}")

    def _refresh_integrity_text(self) -> None:
        verified = len(self._verified_ids) == len(_CATALOG_ORDER)
        key = "integrity_ok" if verified else "integrity_pending"
        self.integrity_status.setText(_copy(key, self._language))
        self.integrity_status.setAccessibleName(self.integrity_status.text())
        self.integrity_status.setProperty("state", "ready" if verified else "next")
        _repolish(self.integrity_status)

    def _verify_models_and_device(self) -> None:
        if self._active_task is not None:
            return

        def operation(context: TaskContext) -> _VerificationResult:
            context.report_progress(0.05, message="Verifying packaged model integrity")
            if self._bundle_verifier is None:
                records = tuple(list_bundled_models(verify=True))
            else:
                records = tuple(self._bundle_verifier(bundle_id) for bundle_id in _CATALOG_ORDER)
            context.raise_if_cancelled()
            context.report_progress(0.75, message="Inspecting the local execution device")
            try:
                device = self._device_probe(context)
            except OperationCancelled:
                raise
            except BaseException as exc:
                device = None
                device_error = str(exc)
            else:
                device_error = None
            return _VerificationResult(records, device, device_error)

        self._start_task("verify", self._runner.submit(operation))

    def _run_experiment(self) -> None:
        if self._active_task is not None or not self.run_button.isEnabled():
            return
        requested_device = str(self.device_combo.currentData())
        bundle_id = self.selected_bundle_id
        if bundle_id == _MONAI_ID:
            directory = self._brats_case_directory
            if directory is None:
                self._open_brats_installer()
                return
            config = BraTSSegmentationConfig(device=requested_device)
            self.detail_tabs.setCurrentIndex(1)

            def run_brats(context: TaskContext) -> BraTSSegmentationResult:
                return self._brats_runner(
                    context,
                    directory,
                    config=config,
                )

            self._start_task("brats", self._runner.submit(run_brats))
            return
        if bundle_id == _DEEPINV_ID and bundle_id in self._demo_operations:
            demo_operation = self._demo_operations[bundle_id]
            self.detail_tabs.setCurrentIndex(1)

            def run_demo(context: TaskContext) -> _DemoExecutionResult:
                started = time.perf_counter()
                result = demo_operation(context, device=requested_device)
                context.raise_if_cancelled()
                if result.bundle_id != bundle_id:
                    raise ValueError("The demonstration returned a different bundled model ID")
                arrays = (
                    np.asarray(result.source),
                    np.asarray(result.prepared_input),
                    np.asarray(result.output),
                )
                source_shape = arrays[0].shape
                if len(source_shape) != 2:
                    raise ValueError("Bundled demonstration views must be two-dimensional")
                if any(array.shape != source_shape for array in arrays):
                    raise ValueError(
                        "Bundled demonstration source, prepared input, and output must match"
                    )
                if any(not np.isfinite(array).all() for array in arrays):
                    raise ValueError("Bundled demonstration views must contain finite values")
                completed = datetime.now(UTC).isoformat().replace("+00:00", "Z")
                return _DemoExecutionResult(
                    result=result,
                    requested_device=requested_device,
                    total_elapsed_seconds=time.perf_counter() - started,
                    completed_at_utc=completed,
                )

            self._start_task("demo", self._runner.submit(run_demo))
            return
        if bundle_id != _DIVAL_ID:
            return
        record = self._records[_DIVAL_ID]
        self.detail_tabs.setCurrentIndex(1)

        def operation(context: TaskContext) -> BundledExperimentResult:
            started = time.perf_counter()
            context.report_progress(0.05, message="Loading the reviewed LoDoPaB teaching case")
            case = self._case_loader()
            fbp = getattr(case, "fbp", None)
            if fbp is None:
                raise RuntimeError(_COPY["case_incompatible"][0])
            fbp_array = np.asarray(fbp, dtype=np.float32)
            if fbp_array.shape != (362, 362) or not np.isfinite(fbp_array).all():
                raise ValueError("The reviewed LoDoPaB FBP must be finite with shape 362x362")
            context.raise_if_cancelled()
            context.report_progress(0.35, message="Running the reviewed DIVal FBP-U-Net")
            model_result = self._dival_runner(fbp_array, device=requested_device)
            context.raise_if_cancelled()
            if model_result.bundle_id != _DIVAL_ID:
                raise ValueError("The task adapter returned a different bundled model ID")
            try:
                output = np.asarray(model_result.outputs["reconstruction"], dtype=np.float32)
            except KeyError as exc:
                raise ValueError("DIVal result is missing the reconstruction output") from exc
            ground_truth = np.asarray(case.ground_truth, dtype=np.float32)
            context.report_progress(0.9, message="Computing reference metrics")
            metrics = _basic_metrics(output, ground_truth)
            manifest = Path(getattr(record, "manifest_path", record.record_path))
            completed = datetime.now(UTC).isoformat().replace("+00:00", "Z")
            return BundledExperimentResult(
                case=case,
                model_result=model_result,
                output=np.array(output, copy=True),
                metrics=metrics,
                requested_device=requested_device,
                total_elapsed_seconds=time.perf_counter() - started,
                completed_at_utc=completed,
                manifest_sha256=_sha256_file(manifest),
                weight_sha256=str(record.artifact_sha256),
            )

        self._start_task("experiment", self._runner.submit(operation))

    def _open_brats_installer(self) -> None:
        """Open the local-only BraTS setup flow without retaining its dialog."""

        dialog = self._brats_dialog_factory(self, language=self._language)
        dialog.validationFinished.connect(
            lambda report, current=dialog: self._capture_brats_validation(current, report)
        )
        try:
            dialog.exec_()
        finally:
            shutdown = getattr(dialog, "shutdown", None)
            if callable(shutdown):
                shutdown()
            dialog.deleteLater()

    def _capture_brats_validation(self, dialog: Any, report: Any) -> None:
        """Keep only a valid report and its user-selected local directory."""

        directory = getattr(dialog, "selected_directory", None)
        if not bool(getattr(report, "is_valid", False)) or directory is None:
            return
        self._brats_case_directory = Path(directory)
        self._brats_validation_report = report
        self._refresh_selection(reset_status=False)
        self._set_status_pair(_COPY["brats_validated"], "ready")

    def _choose_export_path(self) -> None:
        if self._active_task is not None:
            return
        bundle_id = self.selected_bundle_id
        if bundle_id == _MONAI_ID and self._last_brats_result is not None:
            suggested_name = "openmedvisionx-brats-experiment.json"
        elif bundle_id == _DIVAL_ID and self._last_experiment is not None:
            suggested_name = "openmedvisionx-lodopab-experiment.json"
        else:
            return
        filename, _ = QFileDialog.getSaveFileName(
            self,
            _copy("export", self._language),
            suggested_name,
            "JSON (*.json)",
        )
        if filename:
            self.export_experiment(Path(filename))

    def export_experiment(self, path: str | Path) -> None:
        """Atomically export the latest pixel-free reproducibility record."""

        if self._active_task is not None:
            raise RuntimeError("Another bundled-model operation is still running")
        destination = Path(path)
        if self.selected_bundle_id == _MONAI_ID:
            result = self._last_brats_result
            if result is None:
                raise RuntimeError("Run the BraTS experiment before exporting")

            def brats_operation(context: TaskContext) -> Path:
                context.report_progress(
                    0.1,
                    message="Building the pixel-free BraTS experiment record",
                )
                record = create_brats_experiment_record(result)
                context.raise_if_cancelled()
                context.report_progress(
                    0.65,
                    message="Writing the pixel-free BraTS experiment record",
                )
                context.enter_commit_phase()
                return export_experiment_record(destination, record)

            self._start_task("export", self._runner.submit(brats_operation))
            return

        if self.selected_bundle_id != _DIVAL_ID or self._last_experiment is None:
            raise RuntimeError("Run the LoDoPaB experiment before exporting")
        payload = self._experiment_payload(self._last_experiment)

        def operation(context: TaskContext) -> Path:
            context.report_progress(0.1, message="Writing reproducible experiment JSON")
            content = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
            temporary_path: Path | None = None
            try:
                with tempfile.NamedTemporaryFile(
                    mode="w",
                    encoding="utf-8",
                    newline="\n",
                    delete=False,
                    dir=destination.parent,
                    prefix=f".{destination.name}.",
                    suffix=".tmp",
                ) as stream:
                    temporary_path = Path(stream.name)
                    stream.write(content)
                    stream.flush()
                    os.fsync(stream.fileno())
                context.raise_if_cancelled()
                context.enter_commit_phase()
                os.replace(temporary_path, destination)
                temporary_path = None
            finally:
                if temporary_path is not None:
                    temporary_path.unlink(missing_ok=True)
            return destination

        self._start_task("export", self._runner.submit(operation))

    @staticmethod
    def _case_fbp_sha256(case: Any) -> str | None:
        arrays = _mapping(getattr(case.record, "arrays", {}))
        fbp = _mapping(arrays.get("fbp"))
        digest = fbp.get("sha256")
        return str(digest) if digest else None

    def _experiment_payload(self, result: BundledExperimentResult) -> dict[str, Any]:
        case = result.case
        model = result.model_result
        fbp = np.asarray(case.fbp)
        return {
            "schema": "openmedvisionx.bundled-experiment.v1",
            "application": {"name": "OpenMedVisionX", "version": __version__},
            "timestamp_utc": result.completed_at_utc,
            "model": {
                "id": model.bundle_id,
                "manifest_sha256": result.manifest_sha256,
                "weight_sha256": result.weight_sha256,
            },
            "case": {
                "id": str(case.record.case_id),
                "artifact_sha256": str(case.record.artifact_sha256),
                "fbp_sha256": self._case_fbp_sha256(case),
            },
            "execution": {
                "requested_device": result.requested_device,
                "actual_device": model.device,
                "fallback_reason": model.fallback_reason,
                "parameters": {
                    "input": "bundled-reviewed-hann-fbp",
                    "input_shape": list(fbp.shape),
                    "input_dtype": str(fbp.dtype),
                },
                "model_elapsed_seconds": model.elapsed_seconds,
                "total_elapsed_seconds": result.total_elapsed_seconds,
                "warnings": list(model.warnings),
            },
            "metrics": {
                **dict(result.metrics),
                "reference": "bundled LoDoPaB ground truth",
                "psnr_data_range": "max(ground_truth) - min(ground_truth)",
            },
            "privacy": {
                "contains_image_pixels": False,
                "contains_original_file_path": False,
                "contains_dicom_metadata": False,
            },
        }

    def _start_task(self, operation_name: str, task: BackgroundTask[Any]) -> None:
        self._active_operation = operation_name
        self._active_task = task
        self._last_progress = None
        self.progress.setValue(0)
        self.progress.setFormat("%p%")
        self.progress.setVisible(True)
        self.stop_button.setEnabled(True)
        self.verify_button.setEnabled(False)
        self.device_combo.setEnabled(False)
        self.catalog.setEnabled(False)
        self._set_status_pair(_COPY["busy"], "busy")
        self._update_run_gate(reset_status=False)
        self._poll_timer.start()

    def _poll_task(self) -> None:
        task = self._active_task
        if task is None:
            self._poll_timer.stop()
            return
        snapshot = task.progress()
        marker = (snapshot.fraction, snapshot.message)
        if marker != self._last_progress:
            self._last_progress = marker
            self.progress.setValue(int(round(snapshot.fraction * 100)))
            message = snapshot.message
            if self._language == "zh_CN":
                message = _PROGRESS_ZH.get(message, message)
            self.progress.setFormat(f"{message} · %p%" if message else "%p%")
        if not task.done:
            return
        self._poll_timer.stop()
        self._active_task = None
        operation = self._active_operation
        try:
            result = task.result(timeout=0)
        except BaseException as exc:
            if isinstance(exc, OperationCancelled):
                self._set_status_pair(_COPY["cancelled"], "next")
            else:
                details = str(exc)
                self._set_status_pair(
                    (
                        f"{_COPY['failed'][0]}: {details}",
                        f"{_COPY['failed'][1]}：{details}",
                    ),
                    "blocked",
                )
        else:
            if operation == "verify":
                self._apply_verification_result(result)
            elif operation == "experiment":
                self._render_experiment_result(result)
            elif operation == "demo":
                self._render_demo_result(result)
            elif operation == "brats":
                self._render_brats_result(result)
            elif operation == "export":
                path = Path(result)
                self._set_status_pair(
                    (
                        f"{_COPY['exported'][0]}: {path}",
                        f"{_COPY['exported'][1]}：{path}",
                    ),
                    "ready",
                )
        finally:
            self._active_operation = ""
            self.progress.setVisible(False)
            self.stop_button.setEnabled(False)
            self.verify_button.setEnabled(True)
            self.device_combo.setEnabled(True)
            self.catalog.setEnabled(True)
            self._update_run_gate(reset_status=False)

    def _apply_verification_result(self, result: _VerificationResult) -> None:
        for record in result.records:
            bundle_id = str(record.bundle_id)
            self._records[bundle_id] = record
            self._verified_ids.add(bundle_id)
        self._latest_device = result.device
        self._device_error = result.device_error
        self._refresh_catalog_text()
        self._refresh_selection(reset_status=False)
        self._refresh_device_text()
        self._refresh_integrity_text()
        self._set_status_pair(_COPY["integrity_ok"], "ready")

    def _render_experiment_result(self, result: BundledExperimentResult) -> None:
        self._last_experiment = result
        self._last_demo_execution = None
        self._displayed_bundle_id = _DIVAL_ID
        case = result.case
        observation = np.asarray(case.observation)
        fbp = np.asarray(case.fbp)
        output = np.asarray(result.output)
        ground_truth = np.asarray(case.ground_truth)
        absolute_error = np.abs(output - ground_truth)
        observation_mapping = GrayscaleDisplayMapping.from_percentiles(observation, 0.5, 99.5)
        comparison_mapping, error_mapping = _lodopab_reference_mappings(ground_truth)
        self.observation_view.set_array(
            observation,
            grayscale_mapping=observation_mapping,
            fit=True,
        )
        for view, array in (
            (self.fbp_view, fbp),
            (self.output_view, output),
            (self.ground_truth_view, ground_truth),
        ):
            view.set_array(array, grayscale_mapping=comparison_mapping, fit=True)
        self.lodopab_error_view.set_array(
            absolute_error,
            grayscale_mapping=error_mapping,
            fit=True,
        )
        self._refresh_lodopab_scale_labels(result)
        self._refresh_experiment_text(result)
        self.export_button.setEnabled(True)
        self._set_status_pair(
            (
                f"Completed locally on {result.model_result.device} in "
                f"{result.model_result.elapsed_seconds:.3f} s model time.",
                f"已在本地使用 {result.model_result.device} 完成，模型耗时 "
                f"{result.model_result.elapsed_seconds:.3f} 秒。",
            ),
            "ready",
        )

    def _refresh_lodopab_scale_labels(self, result: BundledExperimentResult) -> None:
        """Keep visible quantitative scales synchronized with the UI language."""

        comparison, absolute_error = _lodopab_reference_mappings(result.case.ground_truth)
        unit = _copy("lodopab_value_unit", self._language)
        for view in (self.fbp_view, self.output_view, self.ground_truth_view):
            view.set_value_scale(comparison.lower, comparison.upper, unit=unit)
        self.lodopab_error_view.set_value_scale(
            absolute_error.lower,
            absolute_error.upper,
            unit=unit,
        )

    def _render_demo_result(self, execution: _DemoExecutionResult) -> None:
        result = execution.result
        arrays = {
            "source": np.asarray(result.source, dtype=np.float32),
            "prepared input": np.asarray(result.prepared_input, dtype=np.float32),
            "output": np.asarray(result.output, dtype=np.float32),
        }
        source_shape = arrays["source"].shape
        if len(source_shape) != 2:
            raise ValueError("Bundled demonstration views must be two-dimensional")
        for name, array in arrays.items():
            if array.shape != source_shape or not np.isfinite(array).all():
                raise ValueError(
                    f"Bundled demonstration {name} must be finite with shape {source_shape}"
                )
        absolute_error = np.abs(arrays["output"] - arrays["source"])
        shared_values = np.concatenate(tuple(array.ravel() for array in arrays.values()))
        shared_mapping = GrayscaleDisplayMapping.from_percentiles(shared_values, 0.5, 99.5)
        error_mapping = GrayscaleDisplayMapping.from_percentiles(absolute_error, 0.0, 99.5)
        for view, array in (
            (self.observation_view, arrays["source"]),
            (self.fbp_view, arrays["prepared input"]),
            (self.output_view, arrays["output"]),
        ):
            view.set_array(array, grayscale_mapping=shared_mapping, fit=True)
        self.ground_truth_view.set_array(
            absolute_error,
            grayscale_mapping=error_mapping,
            fit=True,
        )
        self._last_demo_execution = execution
        self._last_experiment = None
        self._displayed_bundle_id = result.bundle_id
        self._refresh_demo_text(execution)
        self.export_button.setEnabled(False)
        self._set_status_pair(
            (
                f"Completed synthetic MRI locally on {result.device} in "
                f"{result.elapsed_seconds:.3f} s model time.",
                f"已在本地使用 {result.device} 完成合成 MRI，模型耗时 "
                f"{result.elapsed_seconds:.3f} 秒。",
            ),
            "ready",
        )

    def _render_brats_result(self, result: BraTSSegmentationResult) -> None:
        probabilities = np.asarray(result.probabilities, dtype=np.float32)
        masks = np.asarray(result.masks, dtype=np.bool_)
        if (
            probabilities.ndim != 4
            or probabilities.shape[0] != 3
            or masks.shape != probabilities.shape
            or not np.isfinite(probabilities).all()
        ):
            raise ValueError("BraTS results must contain three finite aligned 3-D regions")
        if tuple(result.output_regions) != ("WT", "TC", "ET"):
            raise ValueError("BraTS result regions must use the WT, TC, ET product order")
        lesion_per_slice = np.count_nonzero(np.any(masks, axis=0), axis=(1, 2))
        slice_index = (
            int(np.argmax(lesion_per_slice))
            if np.any(lesion_per_slice)
            else probabilities.shape[1] // 2
        )
        probability_mapping = GrayscaleDisplayMapping(0.0, 1.0)
        for view, region_index in (
            (self.observation_view, 0),
            (self.fbp_view, 1),
            (self.output_view, 2),
        ):
            view.set_array(
                probabilities[region_index, slice_index],
                grayscale_mapping=probability_mapping,
                fit=True,
            )
        composite = np.zeros((*masks.shape[2:], 3), dtype=np.uint8)
        composite[masks[0, slice_index]] = (72, 166, 255)
        composite[masks[1, slice_index]] = (255, 184, 76)
        composite[masks[2, slice_index]] = (255, 79, 124)
        self.ground_truth_view.set_array(composite, fit=True)
        for view, region in zip(
            (self.observation_view, self.fbp_view, self.output_view),
            result.output_regions,
            strict=True,
        ):
            view.set_title(f"{region} probability · axial z {slice_index + 1}")
        self.ground_truth_view.set_title(f"WT / TC / ET masks · axial z {slice_index + 1}")
        self._last_brats_result = result
        self._last_demo_execution = None
        self._last_experiment = None
        self._displayed_bundle_id = _MONAI_ID
        self._refresh_brats_text(result, slice_index=slice_index)
        self.export_button.setEnabled(True)
        self._set_status_pair(
            (
                f"BraTS segmentation completed locally on {result.device} in "
                f"{result.elapsed_seconds:.2f} s.",
                f"BraTS 分割已在本地使用 {result.device} 完成，耗时 "
                f"{result.elapsed_seconds:.2f} 秒。",
            ),
            "ready",
        )

    def _refresh_brats_text(
        self,
        result: BraTSSegmentationResult,
        *,
        slice_index: int | None = None,
    ) -> None:
        if slice_index is None:
            lesion_per_slice = np.count_nonzero(
                np.any(np.asarray(result.masks), axis=0), axis=(1, 2)
            )
            slice_index = (
                int(np.argmax(lesion_per_slice))
                if np.any(lesion_per_slice)
                else result.masks.shape[1] // 2
            )
        fallback = result.fallback_reason or ("无" if self._language == "zh_CN" else "none")
        metric_lines: list[str] = []
        for region in result.output_regions:
            metric = result.evaluations.get(region)
            if metric is None:
                continue
            hd95 = "n/a" if metric.hd95_mm is None else f"{metric.hd95_mm:.3f} mm"
            metric_lines.append(
                f"{region}: Dice {metric.dice:.4f} · HD95 {hd95} · "
                f"|volume error| {metric.absolute_volume_error_ml:.3f} mL"
            )
        if not metric_lines:
            metric_lines.append(
                "Task 1 SEG not available; reference metrics were not computed."
                if self._language == "en"
                else "未提供 Task 1 SEG；未计算参考指标。"
            )
        warnings = list(result.warnings)
        if self._language == "zh_CN":
            warnings = [
                {
                    "Domain shift: model training domain is BraTS 2018; the selected case and "
                    "evaluation domain are BraTS 2021 Task 1.": (
                        "领域差异：模型训练域为 BraTS 2018，所选案例与评价域为 BraTS 2021 Task 1。"
                    ),
                    "Education and research only; not for diagnosis.": (
                        "仅供学习与研究，不得用于诊断。"
                    ),
                    (
                        "Task 1 SEG was not present; Dice, HD95, and volume errors "
                        "were not computed."
                    ): ("未提供 Task 1 SEG；未计算 Dice、HD95 与体积误差。"),
                }.get(item, item)
                for item in warnings
            ]
        warning_text = "\n".join(f"• {item}" for item in warnings)
        normalization = "; ".join(
            f"{item.modality}: n={item.nonzero_voxels:,}, "
            f"μ={item.nonzero_mean:.4g}, σ={item.nonzero_standard_deviation:.4g}"
            for item in result.normalizations
        )
        if self._language == "zh_CN":
            lines = (
                f"匿名案例：{result.case_alias}",
                f"输入通道：{' → '.join(result.input_channels)}",
                f"原生输出 → 产品输出：{'/'.join(result.native_output_regions)} → "
                f"{'/'.join(result.output_regions)}",
                f"非零体素 z-score：{normalization}",
                f"推理：{result.device} · patch {result.patch_size} · overlap "
                f"{result.overlap:.2f} · threshold {result.threshold:.2f} · "
                f"{result.patch_count} patches",
                f"模型 / 总耗时：{result.model_elapsed_seconds:.2f} / "
                f"{result.elapsed_seconds:.2f} 秒 · 回退：{fallback}",
                f"模型 SHA-256：{result.model_sha256}",
                f"显示切片：轴位 z {slice_index + 1}/{result.masks.shape[1]}",
                "Task 1 评价：\n" + "\n".join(metric_lines),
                "警告：\n" + warning_text,
            )
        else:
            lines = (
                f"Anonymous case: {result.case_alias}",
                f"Input channels: {' → '.join(result.input_channels)}",
                f"Native → product output: {'/'.join(result.native_output_regions)} → "
                f"{'/'.join(result.output_regions)}",
                f"Nonzero-voxel z-score: {normalization}",
                f"Inference: {result.device} · patch {result.patch_size} · overlap "
                f"{result.overlap:.2f} · threshold {result.threshold:.2f} · "
                f"{result.patch_count} patches",
                f"Model / total time: {result.model_elapsed_seconds:.2f} / "
                f"{result.elapsed_seconds:.2f} s · fallback: {fallback}",
                f"Model SHA-256: {result.model_sha256}",
                f"Displayed slice: axial z {slice_index + 1}/{result.masks.shape[1]}",
                "Task 1 evaluation:\n" + "\n".join(metric_lines),
                "Warnings:\n" + warning_text,
            )
        self.result_details.setText("\n".join(lines))

    def _refresh_demo_text(self, execution: _DemoExecutionResult) -> None:
        result = execution.result
        metrics = _basic_metrics(result.output, result.source)
        warnings = (
            tuple(_WARNING_ZH.get(item, item) for item in result.warnings)
            if self._language == "zh_CN"
            else result.warnings
        )
        fallback = result.fallback_reason or ("无" if self._language == "zh_CN" else "none")
        psnr = metrics["psnr_db"]
        psnr_text = "∞" if math.isinf(psnr) else f"{psnr:.3f} dB"
        if self._language == "zh_CN":
            labels = (
                "流程",
                "请求设备 → 实际设备",
                "模型耗时",
                "总耗时",
                "回退",
                "相对数字模体的指标",
                "提示",
            )
            workflow = "数字模体 → 单线圈 FFT → 欠采样 → 零填充 IFFT / MoDL"
            no_warnings = "无"
        else:
            labels = (
                "Pipeline",
                "Requested → actual device",
                "Model time",
                "Total time",
                "Fallback",
                "Metrics against digital phantom",
                "Warnings",
            )
            workflow = "digital phantom → single-coil FFT → undersampling → zero-filled IFFT / MoDL"
            no_warnings = "none"
        warning_text = "\n".join(f"• {item}" for item in warnings) or no_warnings
        self.result_details.setText(
            f"{labels[0]}: {workflow}\n"
            f"{labels[1]}: {execution.requested_device} → {result.device} · "
            f"{labels[2]}: {result.elapsed_seconds:.3f} s · "
            f"{labels[3]}: {execution.total_elapsed_seconds:.3f} s\n"
            f"{labels[4]}: {fallback}\n"
            f"{labels[5]}: MAE {metrics['mae']:.6g} · RMSE {metrics['rmse']:.6g} · "
            f"PSNR {psnr_text}\n{labels[6]}:\n{warning_text}"
        )

    def _refresh_experiment_text(self, result: BundledExperimentResult) -> None:
        model = result.model_result
        metrics = result.metrics
        comparison_mapping, error_mapping = _lodopab_reference_mappings(result.case.ground_truth)
        warnings = (
            tuple(_WARNING_ZH.get(item, item) for item in model.warnings)
            if self._language == "zh_CN"
            else model.warnings
        )
        fallback = model.fallback_reason or ("无" if self._language == "zh_CN" else "none")
        psnr = metrics["psnr_db"]
        psnr_text = "∞" if math.isinf(psnr) else f"{psnr:.3f} dB"
        labels = (
            ("案例", "请求设备 → 实际设备", "模型耗时", "总耗时", "回退", "指标", "提示")
            if self._language == "zh_CN"
            else (
                "Case",
                "Requested → actual device",
                "Model time",
                "Total time",
                "Fallback",
                "Metrics",
                "Warnings",
            )
        )
        warning_text = "\n".join(f"• {item}" for item in warnings)
        if self._language == "zh_CN":
            display_line = (
                "固定显示量程：FBP / 模型 / 真值 "
                f"[{comparison_mapping.lower:.6g}, {comparison_mapping.upper:.6g}] "
                "归一化衰减；绝对误差 "
                f"[{error_mapping.lower:.6g}, {error_mapping.upper:.6g}] 归一化衰减"
            )
        else:
            display_line = (
                "Fixed display scale: FBP / model / ground truth "
                f"[{comparison_mapping.lower:.6g}, {comparison_mapping.upper:.6g}] "
                "normalized attenuation; absolute error "
                f"[{error_mapping.lower:.6g}, {error_mapping.upper:.6g}] "
                "normalized attenuation"
            )
        self.result_details.setText(
            f"{labels[0]}: {result.case.record.case_id}\n"
            f"{labels[1]}: {result.requested_device} → {model.device} · "
            f"{labels[2]}: {model.elapsed_seconds:.3f} s · "
            f"{labels[3]}: {result.total_elapsed_seconds:.3f} s\n"
            f"{labels[4]}: {fallback}\n"
            f"{labels[5]}: MAE {metrics['mae']:.6g} · RMSE {metrics['rmse']:.6g} · "
            f"PSNR {psnr_text}\n{display_line}\n{labels[6]}:\n{warning_text}"
        )

    def _cancel_active(self) -> None:
        if self._active_task is None:
            return
        self._active_task.cancel()
        self.stop_button.setEnabled(False)
        self._set_status_pair(("Cancelling safely…", "正在安全取消…"), "busy")

    def _set_status_pair(self, pair: tuple[str, str], state: str) -> None:
        self._status_pair = pair
        self._status_state = state
        self._render_status()

    def _render_status(self) -> None:
        text = self._status_pair[1 if self._language == "zh_CN" else 0]
        self.operation_status.setText(text)
        self.operation_status.setAccessibleName(text)
        self.operation_status.setProperty("state", self._status_state)
        _repolish(self.operation_status)
        self.statusChanged.emit(text)

    def _layout_device_controls(self) -> None:
        for widget in (
            self.device_label,
            self.device_combo,
            self.verify_button,
            self.device_status,
            self.integrity_status,
        ):
            self.device_layout.removeWidget(widget)
        self.device_layout.addWidget(self.device_label, 0, 0)
        self.device_layout.addWidget(self.device_combo, 0, 1)
        self.device_layout.addWidget(self.verify_button, 1, 0, 1, 2)
        self.device_layout.addWidget(self.device_status, 2, 0, 1, 2)
        self.device_layout.addWidget(self.integrity_status, 3, 0, 1, 2)
        self.device_layout.setColumnStretch(1, 1)

    def _layout_action_controls(self) -> None:
        for button in (
            self.run_button,
            self.configure_brats_button,
            self.stop_button,
            self.export_button,
        ):
            self.action_layout.removeWidget(button)
        self.action_layout.addWidget(self.run_button, 0, 0, 1, 2)
        next_row = 1
        if self.configure_brats_button.isVisible():
            self.action_layout.addWidget(self.configure_brats_button, next_row, 0, 1, 2)
            next_row += 1
        self.action_layout.addWidget(self.stop_button, next_row, 0)
        self.action_layout.addWidget(self.export_button, next_row, 1)
        self.action_layout.setColumnStretch(0, 1)
        self.action_layout.setColumnStretch(1, 1)

    def resizeEvent(self, event: Any) -> None:  # noqa: N802 - Qt API
        super().resizeEvent(event)
        orientation = _VERTICAL if self.width() < 760 else _HORIZONTAL
        if self.main_splitter.orientation() != orientation:
            self.main_splitter.setOrientation(orientation)
        row_orientation = _VERTICAL if self.width() < 700 else _HORIZONTAL
        for row in (self.demo_top_row, self.demo_bottom_row):
            if row.orientation() != row_orientation:
                row.setOrientation(row_orientation)
        self._layout_device_controls()
        self._layout_action_controls()

    def shutdown(self) -> None:
        """Cooperatively cancel background work during application close."""

        self._poll_timer.stop()
        if self._active_task is not None:
            self._active_task.cancel()
            self._active_task = None
        self._runner.shutdown(wait=False, cancel_pending=True)


__all__ = ["BundledExperimentResult", "BundledModelsPanel"]
