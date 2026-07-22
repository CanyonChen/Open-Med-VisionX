"""Qt dataset and evaluation workspace built on immutable core contracts."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from PyQt5.QtCore import QSize, Qt, pyqtSignal
from PyQt5.QtGui import QResizeEvent
from PyQt5.QtWidgets import (
    QAbstractItemView,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from ..errors import ViewerError
from ..evaluation import (
    ArtifactReference,
    BinaryClassificationMetrics,
    DatasetManifest,
    ExperimentRecord,
    compute_binary_classification_metrics,
    export_experiment_record,
    load_dataset_manifest,
)
from .i18n import Language

_COPY: dict[str, tuple[str, str]] = {
    "title": ("Dataset & Evaluation", "数据集与评价"),
    "intro": (
        "Validate group-safe dataset splits, inspect operating points, calibration, and "
        "uncertainty. This workspace records evidence; it does not make clinical claims.",
        "验证按组隔离的数据集划分，并检查工作阈值、校准与不确定性。本工作区用于记录证据，"
        "不作临床结论。",
    ),
    "manifest_group": ("1. Dataset manifest", "1. 数据集清单"),
    "manifest_help": (
        "Open a path-free OpenMedVisionX JSON/YAML manifest. Samples from one group must "
        "never cross train, validation, and test splits.",
        "打开不含路径的 OpenMedVisionX JSON/YAML 清单。同一组的样本不得跨越训练、验证和测试集。",
    ),
    "open_manifest": ("Open manifest…", "打开清单…"),
    "no_manifest": ("No manifest loaded.", "尚未加载清单。"),
    "manifest_ready": (
        "Manifest validated. Group leakage check passed.",
        "清单已验证，组级数据泄漏检查通过。",
    ),
    "split": ("Split", "划分"),
    "samples": ("Samples", "样本数"),
    "groups": ("Groups", "组数"),
    "evaluation_group": ("2. Binary classification evaluation", "2. 二分类评价"),
    "evaluation_help": (
        "Paste comma-separated reference labels (0/1) and probabilities. Choose the "
        "threshold explicitly; AUROC alone does not define a deployable operating point.",
        "粘贴以逗号分隔的真值标签（0/1）和概率。请显式选择阈值；仅有 AUROC "
        "不能定义可部署的工作点。",
    ),
    "truth": ("Reference labels", "真值标签"),
    "scores": ("Predicted probabilities", "预测概率"),
    "threshold": ("Decision threshold", "决策阈值"),
    "run": ("Evaluate", "开始评价"),
    "export": ("Export experiment record…", "导出实验记录…"),
    "results_tab": ("Metrics", "指标"),
    "calibration_tab": ("Calibration", "校准"),
    "metric": ("Metric", "指标"),
    "value": ("Value", "数值"),
    "interval": ("95% interval", "95% 区间"),
    "bin": ("Probability bin", "概率区间"),
    "count": ("Count", "数量"),
    "mean_score": ("Mean score", "平均分数"),
    "observed": ("Observed positive rate", "实际阳性率"),
    "ready": (
        "Next: load a manifest or evaluate the example probabilities.",
        "下一步：加载清单，或评价示例概率。",
    ),
    "evaluated": (
        "Evaluation complete. Review threshold-dependent metrics and calibration.",
        "评价完成。请检查依赖阈值的指标与校准结果。",
    ),
    "privacy": (
        "Only pseudonymous IDs and hashes belong in manifests. Exported experiment records "
        "contain no image pixels and never overwrite an existing file.",
        "清单中只能使用假名化 ID 与哈希。导出的实验记录不含图像像素，且绝不覆盖现有文件。",
    ),
    "load_failed": ("Manifest validation failed", "清单验证失败"),
    "evaluation_failed": ("Evaluation input is invalid", "评价输入无效"),
    "exported": ("Experiment record exported.", "实验记录已导出。"),
}


def _copy(key: str, language: Language) -> str:
    return _COPY[key][1 if language == "zh_CN" else 0]


class EvaluationPage(QWidget):
    """Compact, responsive UI for manifest checks and binary evaluation."""

    statusChanged = pyqtSignal(str)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._language: Language = "en"
        self.manifest: DatasetManifest | None = None
        self.metrics: BinaryClassificationMetrics | None = None
        self.experiment_record: ExperimentRecord | None = None
        self._build_ui()
        self.set_language("en")

    @property
    def current_status_text(self) -> str:
        return self.status.text()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 8, 0, 0)
        root.setSpacing(10)
        self.title = QLabel()
        self.title.setObjectName("pageTitle")
        self.intro = QLabel()
        self.intro.setObjectName("infoBanner")
        self.intro.setWordWrap(True)
        root.addWidget(self.title)
        root.addWidget(self.intro)

        self.workspace = QSplitter(Qt.Horizontal)
        self.workspace.setChildrenCollapsible(False)
        self.workspace.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Expanding)
        self.workspace.addWidget(self._build_manifest_panel())
        self.workspace.addWidget(self._build_evaluation_panel())
        self.workspace.setStretchFactor(0, 2)
        self.workspace.setStretchFactor(1, 3)
        self.workspace.setSizes([390, 610])
        self.workspace_scroll = QScrollArea()
        self.workspace_scroll.setWidgetResizable(True)
        self.workspace_scroll.setFrameShape(QFrame.NoFrame)
        self.workspace_scroll.setWidget(self.workspace)
        root.addWidget(self.workspace_scroll, 1)

        self.privacy_note = QLabel()
        self.privacy_note.setObjectName("warningBanner")
        self.privacy_note.setWordWrap(True)
        self.privacy_note.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.status = QLabel()
        self.status.setObjectName("actionStatus")
        self.status.setWordWrap(True)
        root.addWidget(self.privacy_note)
        root.addWidget(self.status)

        QWidget.setTabOrder(self.open_manifest_button, self.truth_input)
        QWidget.setTabOrder(self.truth_input, self.score_input)
        QWidget.setTabOrder(self.score_input, self.threshold)
        QWidget.setTabOrder(self.threshold, self.evaluate_button)
        QWidget.setTabOrder(self.evaluate_button, self.export_button)

    def _build_manifest_panel(self) -> QWidget:
        panel = QGroupBox()
        self.manifest_group = panel
        layout = QVBoxLayout(panel)
        layout.setSpacing(8)
        self.manifest_help = QLabel()
        self.manifest_help.setWordWrap(True)
        self.open_manifest_button = QPushButton()
        self.open_manifest_button.clicked.connect(self._choose_manifest)
        self.manifest_status = QLabel()
        self.manifest_status.setObjectName("mutedText")
        self.manifest_status.setWordWrap(True)
        self.manifest_table = QTableWidget(3, 3)
        self.manifest_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.manifest_table.setSelectionMode(QAbstractItemView.NoSelection)
        self.manifest_table.verticalHeader().setVisible(False)
        self.manifest_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.manifest_table.setMinimumHeight(150)
        for row, split in enumerate(("train", "validation", "test")):
            self.manifest_table.setItem(row, 0, QTableWidgetItem(split))
            self.manifest_table.setItem(row, 1, QTableWidgetItem("—"))
            self.manifest_table.setItem(row, 2, QTableWidgetItem("—"))
        layout.addWidget(self.manifest_help)
        layout.addWidget(self.open_manifest_button, 0, Qt.AlignLeft)
        layout.addWidget(self.manifest_status)
        layout.addWidget(self.manifest_table)
        layout.addStretch(1)
        return panel

    def _build_evaluation_panel(self) -> QWidget:
        panel = QGroupBox()
        self.evaluation_group = panel
        layout = QVBoxLayout(panel)
        layout.setSpacing(8)
        self.evaluation_help = QLabel()
        self.evaluation_help.setWordWrap(True)
        form = QFormLayout()
        form.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
        self.truth_label = QLabel()
        self.score_label = QLabel()
        self.threshold_label = QLabel()
        self.truth_input = QPlainTextEdit("0, 0, 1, 1, 1, 0, 1, 0")
        self.score_input = QPlainTextEdit("0.05, 0.25, 0.45, 0.72, 0.91, 0.38, 0.83, 0.62")
        for editor in (self.truth_input, self.score_input):
            editor.setMaximumHeight(58)
            editor.setTabChangesFocus(True)
        self.threshold = QDoubleSpinBox()
        self.threshold.setRange(0.0, 1.0)
        self.threshold.setSingleStep(0.05)
        self.threshold.setDecimals(2)
        self.threshold.setValue(0.5)
        form.addRow(self.truth_label, self.truth_input)
        form.addRow(self.score_label, self.score_input)
        form.addRow(self.threshold_label, self.threshold)
        actions = QFrame()
        actions.setObjectName("toolbarSurface")
        action_layout = QHBoxLayout(actions)
        action_layout.setContentsMargins(8, 6, 8, 6)
        self.evaluate_button = QPushButton()
        self.evaluate_button.setObjectName("primary")
        self.evaluate_button.clicked.connect(self.evaluate)
        self.export_button = QPushButton()
        self.export_button.setEnabled(False)
        self.export_button.clicked.connect(self._choose_export)
        action_layout.addWidget(self.evaluate_button)
        action_layout.addWidget(self.export_button)
        action_layout.addStretch(1)

        self.result_tabs = QTabWidget()
        self.metrics_table = QTableWidget(0, 3)
        self.metrics_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.metrics_table.setSelectionMode(QAbstractItemView.NoSelection)
        self.metrics_table.verticalHeader().setVisible(False)
        self.metrics_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.calibration_table = QTableWidget(0, 4)
        self.calibration_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.calibration_table.setSelectionMode(QAbstractItemView.NoSelection)
        self.calibration_table.verticalHeader().setVisible(False)
        self.calibration_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.result_tabs.addTab(self.metrics_table, "")
        self.result_tabs.addTab(self.calibration_table, "")

        layout.addWidget(self.evaluation_help)
        layout.addLayout(form)
        layout.addWidget(actions)
        layout.addWidget(self.result_tabs, 1)
        return panel

    def set_language(self, language: Language) -> None:
        if language not in {"en", "zh_CN"}:
            raise ValueError(f"Unsupported UI language: {language}")
        self._language = language
        self.title.setText(_copy("title", language))
        self.intro.setText(_copy("intro", language))
        self.manifest_group.setTitle(_copy("manifest_group", language))
        self.manifest_help.setText(_copy("manifest_help", language))
        self.open_manifest_button.setText(_copy("open_manifest", language))
        self.open_manifest_button.setAccessibleName(_copy("open_manifest", language))
        self.manifest_status.setText(
            _copy("manifest_ready" if self.manifest is not None else "no_manifest", language)
        )
        self.evaluation_group.setTitle(_copy("evaluation_group", language))
        self.evaluation_help.setText(_copy("evaluation_help", language))
        self.truth_label.setText(_copy("truth", language))
        self.score_label.setText(_copy("scores", language))
        self.threshold_label.setText(_copy("threshold", language))
        self.truth_input.setAccessibleName(_copy("truth", language))
        self.score_input.setAccessibleName(_copy("scores", language))
        self.threshold.setAccessibleName(_copy("threshold", language))
        self.evaluate_button.setText(_copy("run", language))
        self.export_button.setText(_copy("export", language))
        self.result_tabs.setTabText(0, _copy("results_tab", language))
        self.result_tabs.setTabText(1, _copy("calibration_tab", language))
        self.metrics_table.setHorizontalHeaderLabels(
            [_copy("metric", language), _copy("value", language), _copy("interval", language)]
        )
        self.calibration_table.setHorizontalHeaderLabels(
            [
                _copy("bin", language),
                _copy("count", language),
                _copy("mean_score", language),
                _copy("observed", language),
            ]
        )
        self.privacy_note.setText(_copy("privacy", language))
        if self.metrics is None:
            self._set_status(_copy("ready", language), emit=False)
        else:
            self._set_status(_copy("evaluated", language), emit=False)

    def _choose_manifest(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            _copy("open_manifest", self._language),
            "",
            "Dataset manifests (*.json *.yaml *.yml);;All files (*)",
        )
        if path:
            self.load_manifest(path)

    def load_manifest(self, path: str | Path) -> DatasetManifest | None:
        try:
            manifest = load_dataset_manifest(path)
        except ViewerError as exc:
            self._set_status(f"{_copy('load_failed', self._language)}: {exc}")
            return None
        self.manifest = manifest
        report = manifest.split_report
        for row, split in enumerate(("train", "validation", "test")):
            self.manifest_table.item(row, 1).setText(str(report.sample_counts[split]))
            self.manifest_table.item(row, 2).setText(str(report.group_counts[split]))
        self.manifest_status.setText(
            f"{_copy('manifest_ready', self._language)}\n"
            f"{manifest.dataset_id} · {manifest.dataset_version} · {manifest.task}"
        )
        self._set_status(_copy("manifest_ready", self._language))
        return manifest

    @staticmethod
    def _parse_vector(text: str, *, binary: bool) -> np.ndarray:
        tokens = [item.strip() for item in text.replace("\n", ",").split(",") if item.strip()]
        if not tokens:
            raise ValueError("Enter at least one value.")
        values = np.asarray([float(item) for item in tokens], dtype=np.float64)
        if not np.all(np.isfinite(values)):
            raise ValueError("Values must be finite.")
        if binary and not np.all(np.isin(values, (0.0, 1.0))):
            raise ValueError("Reference labels must contain only 0 and 1.")
        if not binary and (np.any(values < 0.0) or np.any(values > 1.0)):
            raise ValueError("Predicted probabilities must remain in [0, 1].")
        return values.astype(np.int8 if binary else np.float64)

    def evaluate(self) -> BinaryClassificationMetrics | None:
        try:
            truth = self._parse_vector(self.truth_input.toPlainText(), binary=True)
            scores = self._parse_vector(self.score_input.toPlainText(), binary=False)
            if truth.shape != scores.shape:
                raise ValueError("Reference labels and probabilities must have equal length.")
            metrics = compute_binary_classification_metrics(
                truth,
                scores,
                threshold=self.threshold.value(),
            )
        except (ValueError, ViewerError) as exc:
            self.metrics = None
            self.experiment_record = None
            self.export_button.setEnabled(False)
            self._set_status(f"{_copy('evaluation_failed', self._language)}: {exc}")
            return None
        self.metrics = metrics
        self._render_metrics(metrics)
        self.experiment_record = self._build_record(truth, scores, metrics)
        self.export_button.setEnabled(True)
        self._set_status(_copy("evaluated", self._language))
        return metrics

    def _render_metrics(self, metrics: BinaryClassificationMetrics) -> None:
        rows: tuple[tuple[str, float | None, str], ...] = (
            ("Accuracy", metrics.accuracy, "accuracy"),
            ("Sensitivity", metrics.sensitivity, "sensitivity"),
            ("Specificity", metrics.specificity, "specificity"),
            ("PPV", metrics.positive_predictive_value, "positive_predictive_value"),
            ("NPV", metrics.negative_predictive_value, "negative_predictive_value"),
            ("F1", metrics.f1, ""),
            ("AUROC", metrics.auroc, "auroc"),
            ("AUPRC", metrics.average_precision, ""),
            ("Brier score", metrics.brier_score, ""),
            ("Expected calibration error", metrics.expected_calibration_error, ""),
        )
        self.metrics_table.setRowCount(len(rows))
        for row, (name, value, interval_key) in enumerate(rows):
            interval = metrics.confidence_intervals.get(interval_key)
            interval_text = (
                "—" if interval is None else f"{interval.lower:.3f}–{interval.upper:.3f}"
            )
            for column, text in enumerate(
                (name, "—" if value is None else f"{value:.4f}", interval_text)
            ):
                self.metrics_table.setItem(row, column, QTableWidgetItem(text))
        self.calibration_table.setRowCount(len(metrics.calibration_bins))
        for row, item in enumerate(metrics.calibration_bins):
            values = (
                f"[{item.lower:.1f}, {item.upper:.1f}{']' if item.upper == 1.0 else ')'}",
                str(item.count),
                "—" if item.mean_score is None else f"{item.mean_score:.3f}",
                "—" if item.positive_fraction is None else f"{item.positive_fraction:.3f}",
            )
            for column, text in enumerate(values):
                self.calibration_table.setItem(row, column, QTableWidgetItem(text))

    def _build_record(
        self,
        truth: np.ndarray,
        scores: np.ndarray,
        metrics: BinaryClassificationMetrics,
    ) -> ExperimentRecord:
        canonical = json.dumps(
            {"truth": truth.tolist(), "scores": scores.tolist()},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        metric_values: dict[str, Any] = {
            "accuracy": metrics.accuracy,
            "sensitivity": metrics.sensitivity,
            "specificity": metrics.specificity,
            "positive_predictive_value": metrics.positive_predictive_value,
            "negative_predictive_value": metrics.negative_predictive_value,
            "f1": metrics.f1,
            "auroc": metrics.auroc,
            "average_precision": metrics.average_precision,
            "brier_score": metrics.brier_score,
            "expected_calibration_error": metrics.expected_calibration_error,
        }
        now = datetime.now(timezone.utc)
        dataset_id = None
        if self.manifest is not None:
            dataset_id = f"{self.manifest.dataset_id}-{self.manifest.dataset_version}"
        return ExperimentRecord(
            record_id=f"evaluation-{now.strftime('%Y%m%dT%H%M%SZ')}",
            created_at=now,
            application_version="0.1.0",
            code_revision="working-tree",
            task="binary-classification-evaluation",
            model_id="manual-probabilities",
            model_version="user-input",
            dataset_manifest_id=dataset_id,
            inputs=(
                ArtifactReference(
                    artifact_id="reference-and-probability-vectors",
                    sha256=hashlib.sha256(canonical).hexdigest(),
                    kind="classification-scores",
                    media_type="application/json",
                    shape=(int(truth.size), 2),
                ),
            ),
            transforms=(),
            outputs=(),
            parameters={"threshold": metrics.threshold},
            metrics=metric_values,
            warnings=(
                "Manually entered teaching data; review its provenance before interpretation.",
            ),
        )

    def _choose_export(self) -> None:
        if self.experiment_record is None:
            return
        path, _ = QFileDialog.getSaveFileName(
            self,
            _copy("export", self._language),
            "openmedvisionx-evaluation.json",
            "JSON (*.json);;YAML (*.yaml *.yml)",
        )
        if not path:
            return
        try:
            export_experiment_record(path, self.experiment_record)
        except ViewerError as exc:
            QMessageBox.warning(self, _copy("export", self._language), str(exc))
            return
        self._set_status(_copy("exported", self._language))

    def _set_status(self, text: str, *, emit: bool = True) -> None:
        self.status.setText(text)
        if emit:
            self.statusChanged.emit(text)

    def resizeEvent(self, event: QResizeEvent) -> None:
        orientation = Qt.Vertical if event.size().width() < 800 else Qt.Horizontal
        if self.workspace.orientation() != orientation:
            self.workspace.setOrientation(orientation)
        self.workspace.setMinimumHeight(720 if orientation == Qt.Vertical else 0)
        super().resizeEvent(event)

    def minimumSizeHint(self) -> QSize:
        """Allow the shell to keep its supported logical minimum and rely on reflow."""

        return QSize(0, 0)
