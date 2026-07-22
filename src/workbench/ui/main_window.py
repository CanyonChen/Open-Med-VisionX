"""Capability-driven OpenMedVisionX desktop interface."""

from __future__ import annotations

import hashlib
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace
from html import escape
from io import BytesIO
from ipaddress import ip_address
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import numpy as np
from PyQt5.QtCore import QRect, QSize, Qt, QTimer, QUrl, pyqtSignal
from PyQt5.QtGui import QCloseEvent, QDesktopServices, QIcon, QKeySequence, QPixmap
from PyQt5.QtWidgets import (
    QAbstractButton,
    QAbstractItemView,
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QInputDialog,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QShortcut,
    QSizePolicy,
    QSlider,
    QSpinBox,
    QSplitter,
    QStackedWidget,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from ..algorithms import (
    ReconstructionRequest,
    ReconstructionSourceKind,
    generate_ct_phantom,
)
from ..domain.display import ColorDisplayMapping, GrayscaleDisplayMapping
from ..domain.images import (
    Capability,
    ImageData,
    ImageSequence2D,
    ImageVolume,
    RasterImage2D,
)
from ..domain.resampling import resample_segmentation_layer
from ..domain.studies import (
    ContourLayer,
    ImageStudy,
    InterpolationMode,
    SegmentationLayer,
    SegmentationValueType,
    SpatialGeometry,
    VolumeLayer,
)
from ..domain.transforms import TransformRecord
from ..errors import OperationCancelled, ViewerError
from ..inference import (
    InferenceResult,
    ModelManifest,
    VisualizationKind,
    inverse_map_spatial_array,
)
from ..io import (
    DicomAnnotationImport,
    NiftiVolumeSelectionRequiredError,
    SeriesSelectionRequiredError,
    SeriesSummary,
    import_dicom_annotation,
    import_label_map,
)
from ..llm import (
    ArtifactReview,
    ArtifactReviewDecision,
    LLMArtifactResponse,
    LLMProvider,
    LLMResponse,
    LLMTaskRequest,
    RenderedPreview,
    TransferPlan,
)
from ..runtime import BackgroundTask, CredentialReference, TaskRunner
from ..services import (
    AnnotationOverlay,
    ImageService,
    LoadedStudy,
    ModelInferenceService,
    ProviderConfiguration,
    ReconstructionService,
    TeachingAssistantService,
    create_experiment_record,
    export_rendered_png,
    load_local_annotations,
    load_local_mask,
    save_experiment_record,
)
from .assistant_artifacts import AssistantArtifactsPanel
from .bundled_models import BundledModelsPanel
from .evaluation_workspace import EvaluationPage
from .i18n import Language, translate
from .image_view import ImageView, array_to_qimage
from .study_layers import StudyLayersPanel
from .theme import APP_STYLE
from .widgets import HistogramDialog, SafeMarkdownBrowser, SubmitPlainTextEdit

APP_NAME = "OpenMedVisionX"
APP_SUBTITLE = "An Open Interactive Platform for Medical Computer Vision Learning and Exploration"
APP_SHORT_SUBTITLE = "Medical vision workspace"
MODEL_SUMMARY_INTRO = (
    "Manifest details and model outputs appear here. Review the runtime, licenses, "
    "preprocessing, and local weight paths before loading.\n\nPython adapters execute "
    "third-party code and require explicit consent."
)


def _brand_asset_path(filename: str) -> Path | None:
    """Locate a bundled or source-tree brand asset without assuming the launch directory."""

    module_path = Path(__file__).resolve()
    candidates = (
        module_path.parents[3] / "figs" / filename,
        Path.cwd() / "figs" / filename,
        Path(sys.prefix) / "share" / "openmedvisionx" / "figs" / filename,
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def _brand_pixmap(filename: str, height: int) -> QPixmap:
    """Load and trim the known near-white padding around the supplied project logos."""

    path = _brand_asset_path(filename)
    if path is None:
        return QPixmap()
    pixmap = QPixmap(str(path))
    if pixmap.isNull():
        return pixmap
    if filename == "logo_pure.png":
        crop = QRect(
            round(pixmap.width() * 0.21),
            round(pixmap.height() * 0.18),
            round(pixmap.width() * 0.58),
            round(pixmap.height() * 0.58),
        )
    else:
        crop = QRect(
            round(pixmap.width() * 0.07),
            round(pixmap.height() * 0.22),
            round(pixmap.width() * 0.86),
            round(pixmap.height() * 0.56),
        )
    return pixmap.copy(crop).scaledToHeight(height, Qt.SmoothTransformation)


def _apply_logo(label: QLabel, filename: str, height: int) -> bool:
    pixmap = _brand_pixmap(filename, height)
    if pixmap.isNull():
        return False
    label.setPixmap(pixmap)
    label.setFixedSize(pixmap.size())
    label.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
    label.setAccessibleName(f"{APP_NAME} logo")
    label.setProperty("_openmedvisionx_i18n_skip_text", True)
    return True


TEACHING_EXPERIMENTS: dict[str, dict[str, str]] = {
    "2-D pixels, bit depth, and interpolation": {
        "Principle": (
            "A raster is a sampled signal. Bit depth controls representable values; color space "
            "defines channel meaning. Display mapping changes only the view, never decoded data."
        ),
        "Formula": (
            "Quantization step Δ = (Imax − Imin)/(2ᵇ − 1); bilinear interpolation is a "
            "weighted sum of four neighbours."
        ),
        "Parameter explanation": (
            "b is bit depth; display bounds select a visible range. Spacing is pixel-only "
            "unless entered by the user."
        ),
        "Steps": (
            "Open generated PNG/TIFF examples, inspect dtype and histogram, adjust the "
            "display range, zoom, and compare interpolation modes."
        ),
        "Expected observation": (
            "Higher bit depth preserves more levels; JPEG may add artifacts; display changes "
            "do not alter stored values."
        ),
        "Common mistakes": (
            "Treating DPI as medical spacing, calling RGB values HU, or assuming a screenshot "
            "retains 16-bit values."
        ),
        "Reflection question": (
            "Which changes affect measurement data, and which affect only visualization?"
        ),
    },
    "DICOM and NIfTI physical geometry": {
        "Principle": (
            "Medical volumes combine voxels with an affine. OpenMedVisionX displays validated "
            "volumes in RAS+ coordinates."
        ),
        "Formula": (
            "x_world = A · [i, j, k, 1]ᵀ; DICOM intensity = stored value × slope + intercept."
        ),
        "Parameter explanation": (
            "Spacing is physical sample distance; origin and direction locate axes; slope and "
            "intercept define quantitative intensity."
        ),
        "Steps": (
            "Open a generated volume, inspect geometry, navigate three orthogonal planes, and "
            "measure in millimetres."
        ),
        "Expected observation": (
            "Anisotropic spacing changes physical aspect ratio; CT reports HU while ordinary "
            "rasters do not."
        ),
        "Common mistakes": (
            "Sorting by filename, ignoring LPS→RAS conversion, or inventing volume geometry "
            "for a TIFF sequence."
        ),
        "Reflection question": (
            "Why can equal-shaped arrays represent different anatomy orientations?"
        ),
    },
    "Radon transform and filtered backprojection": {
        "Principle": (
            "A sinogram contains line integrals. FBP filters detector profiles before "
            "backprojecting them over image space."
        ),
        "Formula": "p(s,θ)=∫f(x,y)δ(s−x cosθ−y sinθ)dxdy; f≈B{F⁻¹[|ω|F(p)]}.",
        "Parameter explanation": (
            "Angles control sampling; circular support defines the object region; filters trade "
            "resolution against noise."
        ),
        "Steps": (
            "Generate a 180° sinogram, inspect projections and spectrum, reconstruct with BP/FBP, "
            "then compare filters and 360° redundancy."
        ),
        "Expected observation": (
            "BP blurs; ramp filtering sharpens; 360° parallel-beam data repeats 180° information "
            "with detector reversal."
        ),
        "Common mistakes": (
            "Mismatching angles, normalizing images independently before metrics, or changing "
            "circle/output geometry between stages."
        ),
        "Reflection question": ("Why does a smoother filter reduce noise while softening edges?"),
    },
    "SART iterative reconstruction": {
        "Principle": (
            "SART repeatedly corrects an image using the mismatch between measured and "
            "predicted projections."
        ),
        "Formula": "xᵏ⁺¹ = xᵏ + λ C Aᵀ R (b − A xᵏ).",
        "Parameter explanation": (
            "Iterations set update count; relaxation λ controls update size; snapshots reveal "
            "convergence and artifacts."
        ),
        "Steps": (
            "Use one sinogram, vary iteration count and relaxation, then compare snapshots and "
            "joint-range metrics."
        ),
        "Expected observation": (
            "Early iterations recover coarse structure; excessive updates can amplify streaks "
            "and noise."
        ),
        "Common mistakes": (
            "Judging a stretched display, using unstable λ, or interpreting convergence as "
            "clinical validity."
        ),
        "Reflection question": "How would noise level change your stopping criterion?",
    },
    "External model inference": {
        "Principle": (
            "A manifest makes preprocessing, tensor semantics, runtime, licenses, and output "
            "coordinates explicit."
        ),
        "Formula": "x_norm = (resize/crop(x)/scale − mean)/std; y_image = T⁻¹(y_model).",
        "Parameter explanation": (
            "Layout, color order, dtype, resize, and normalization must match training; the "
            "TransformRecord maps results back."
        ),
        "Steps": (
            "Validate a manifest, inspect licenses and capabilities, explicitly load its local "
            "model, then run one rendered plane."
        ),
        "Expected observation": (
            "Typed scores, masks, boxes, or maps follow declared output semantics."
        ),
        "Common mistakes": (
            "Loading a bare checkpoint, swapping RGB/BGR, stretching aspect ratio, or trusting "
            "unknown adapters."
        ),
        "Reflection question": ("Which manifest fields reproduce a prediction on another machine?"),
    },
    "Multimodal AI teaching assistant": {
        "Principle": (
            "A provider can discuss a reviewed rendered slice, but it is not a diagnostic "
            "system and may hallucinate."
        ),
        "Formula": (
            "response = Provider(model_id, prompt, optional rendered PNG); raw volumes and "
            "metadata are excluded."
        ),
        "Parameter explanation": (
            "Endpoint and model ID are user supplied; network and image transfer are separate "
            "opt-ins; credentials remain references."
        ),
        "Steps": (
            "Choose a provider and model ID, inspect the preview, opt in to network and optional "
            "image transfer, then ask a concept question."
        ),
        "Expected observation": (
            "The answer records provider/model/time and a disclaimer; image consent applies "
            "once to the exact reviewed transfer plan."
        ),
        "Common mistakes": (
            "Sending burned-in identifiers, treating text as diagnosis, or placing API keys in "
            "project files."
        ),
        "Reflection question": (
            "What evidence is needed before trusting an AI explanation of an imaging artifact?"
        ),
    },
}


_I18N_SOURCE_PROPERTY = "_openmedvisionx_i18n_source"
_I18N_CONTENT_PROPERTY = "_openmedvisionx_i18n_content"
_I18N_SKIP_ITEMS_PROPERTY = "_openmedvisionx_i18n_skip_items"
_I18N_SKIP_TEXT_PROPERTY = "_openmedvisionx_i18n_skip_text"
_I18N_COMBO_SOURCE_ROLE = Qt.UserRole + 117
_I18N_TAB_SOURCE_PROPERTY = "_openmedvisionx_i18n_tab_source"


def _set_dynamic_property(widget: QWidget, name: str, value: object) -> None:
    """Update one styled state without reparsing the application style sheet."""

    if widget.property(name) == value:
        return
    widget.setProperty(name, value)
    widget.style().unpolish(widget)
    widget.style().polish(widget)
    widget.update()


def _add_accessible_form_row(
    form: QFormLayout,
    text: str,
    field: QWidget,
    *,
    description: str = "",
) -> QLabel:
    """Add an explicit label/buddy pair with stable screen-reader semantics."""

    label = QLabel(text)
    label.setBuddy(field)
    field.setAccessibleName(text.rstrip(":："))
    if description:
        field.setAccessibleDescription(description)
        field.setToolTip(description)
    form.addRow(label, field)
    return label


def _translation_source(
    widget: QWidget,
    property_name: str,
    current_text: str,
    previous_language: Language,
) -> str:
    """Return a stable English source while allowing runtime text updates."""

    stored = widget.property(property_name)
    source = str(stored) if stored is not None else translate(current_text, "en")
    if stored is not None and current_text != translate(source, previous_language):
        source = translate(current_text, "en")
    widget.setProperty(property_name, source)
    return source


def _retranslate_widget_tree(
    root: QWidget,
    language: Language,
    previous_language: Language,
) -> None:
    """Translate ordinary Qt text properties without recreating any widgets."""

    for widget in (root, *root.findChildren(QWidget)):
        if isinstance(widget, ImageView):
            widget.set_language(language)
        else:
            for suffix, current, setter in (
                ("accessible_name", widget.accessibleName(), widget.setAccessibleName),
                (
                    "accessible_description",
                    widget.accessibleDescription(),
                    widget.setAccessibleDescription,
                ),
                ("tooltip", widget.toolTip(), widget.setToolTip),
            ):
                if current:
                    source = _translation_source(
                        widget,
                        f"{_I18N_SOURCE_PROPERTY}_{suffix}",
                        current,
                        previous_language,
                    )
                    setter(translate(source, language))

        if isinstance(widget, QGroupBox):
            source = _translation_source(
                widget,
                f"{_I18N_SOURCE_PROPERTY}_title",
                widget.title(),
                previous_language,
            )
            widget.setTitle(translate(source, language))

        if isinstance(widget, QAbstractButton) or (
            isinstance(widget, QLabel) and not widget.property(_I18N_SKIP_TEXT_PROPERTY)
        ):
            source = _translation_source(
                widget,
                f"{_I18N_SOURCE_PROPERTY}_text",
                widget.text(),
                previous_language,
            )
            widget.setText(translate(source, language))

        if isinstance(widget, (QLineEdit, QPlainTextEdit, QTextEdit)):
            placeholder = widget.placeholderText()
            if placeholder:
                source = _translation_source(
                    widget,
                    f"{_I18N_SOURCE_PROPERTY}_placeholder",
                    placeholder,
                    previous_language,
                )
                widget.setPlaceholderText(translate(source, language))

        if isinstance(widget, (QPlainTextEdit, QTextEdit)) and widget.property(
            _I18N_CONTENT_PROPERTY
        ):
            current = widget.toPlainText()
            source = _translation_source(
                widget,
                f"{_I18N_SOURCE_PROPERTY}_content",
                current,
                previous_language,
            )
            widget.setPlainText(translate(source, language))

        if isinstance(widget, QProgressBar):
            source = _translation_source(
                widget,
                f"{_I18N_SOURCE_PROPERTY}_format",
                widget.format(),
                previous_language,
            )
            widget.setFormat(translate(source, language))

        if isinstance(widget, QComboBox) and not widget.property(_I18N_SKIP_ITEMS_PROPERTY):
            for index in range(widget.count()):
                current = widget.itemText(index)
                source = widget.itemData(index, _I18N_COMBO_SOURCE_ROLE)
                if source is None or current != translate(str(source), previous_language):
                    source = translate(current, "en")
                widget.setItemData(index, source, _I18N_COMBO_SOURCE_ROLE)
                widget.setItemText(index, translate(str(source), language))

        if isinstance(widget, QTabWidget):
            for index in range(widget.count()):
                page = widget.widget(index)
                property_name = f"{_I18N_TAB_SOURCE_PROPERTY}_{index}"
                source = _translation_source(
                    page,
                    property_name,
                    widget.tabText(index),
                    previous_language,
                )
                widget.setTabText(index, translate(source, language))


class _BilingualPage:
    """Small mixin that gives each persistent page an in-place language switch."""

    _language: Language = "en"

    @property
    def language(self) -> Language:
        return self._language

    def _tr(self, text: str) -> str:
        return translate(text, self._language)

    def set_language(self, language: Language) -> None:
        previous_language = getattr(self, "_language", "en")
        self._language = language
        _retranslate_widget_tree(self, language, previous_language)  # type: ignore[arg-type]
        self._language_changed()

    def _language_changed(self) -> None:
        """Refresh state-derived text after ordinary widget properties are translated."""


class TaskWatcher(QWidget):
    """Poll background handles on the Qt thread; workers never mutate widgets."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._watches: list[dict[str, Any]] = []
        self._timer = QTimer(self)
        self._timer.setInterval(60)
        self._timer.timeout.connect(self._poll)

    def watch(
        self,
        task: BackgroundTask[Any],
        *,
        success: Callable[[Any], None],
        error: Callable[[BaseException], None],
        progress: Callable[[float, str], None] | None = None,
    ) -> None:
        self._watches.append(
            {"task": task, "success": success, "error": error, "progress": progress, "seen": None}
        )
        if not self._timer.isActive():
            self._timer.start()

    def _poll(self) -> None:
        remaining: list[dict[str, Any]] = []
        for watch in self._watches:
            task: BackgroundTask[Any] = watch["task"]
            snapshot = task.progress()
            marker = (snapshot.fraction, snapshot.message)
            if marker != watch["seen"] and watch["progress"] is not None:
                watch["progress"](snapshot.fraction, snapshot.message)
                watch["seen"] = marker
            if task.done:
                try:
                    result = task.result(timeout=0)
                except BaseException as exc:  # task errors are surfaced in UI
                    watch["error"](exc)
                else:
                    watch["success"](result)
            else:
                remaining.append(watch)
        self._watches = remaining
        if not remaining:
            self._timer.stop()

    def stop(self) -> None:
        """Stop UI polling and release callbacks before the owning page is torn down."""

        self._timer.stop()
        self._watches.clear()


class _DicomSeriesSelectionDialog(QDialog):
    """Keyboard-accessible, PHI-minimized browser for one multi-series source."""

    _HEADERS = (
        "Status",
        "Modality",
        "Description",
        "Series no.",
        "Instances / slices / frames",
        "Dimensions",
        "Pixel spacing",
        "Slice thickness",
        "Geometry",
        "Warnings",
    )

    def __init__(
        self,
        candidates: Sequence[SeriesSummary],
        *,
        translator: Callable[[str], str],
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._translator = translator
        self._selected_selector: str | None = None
        self.setModal(True)
        self.setWindowTitle(translator("Choose a DICOM series"))
        self.setAccessibleName(translator("DICOM series selection"))

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 14)
        layout.setSpacing(10)
        title = QLabel(translator("Choose a DICOM series"))
        title.setObjectName("dialogTitle")
        title.setWordWrap(True)
        layout.addWidget(title)
        explanation = QLabel(
            translator(
                "This source contains multiple DICOM series. OpenMedVisionX will not "
                "guess. Select one available series to continue; unsupported rows remain "
                "disabled."
            )
        )
        explanation.setWordWrap(True)
        explanation.setTextInteractionFlags(Qt.TextSelectableByMouse)
        layout.addWidget(explanation)

        self.table = QTableWidget(len(candidates), len(self._HEADERS))
        self.table.setObjectName("dicomSeriesTable")
        self.table.setHorizontalHeaderLabels([translator(item) for item in self._HEADERS])
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setAlternatingRowColors(True)
        self.table.setWordWrap(True)
        self.table.verticalHeader().setVisible(False)
        self.table.setAccessibleName(translator("Available DICOM series"))
        self.table.setAccessibleDescription(
            translator(
                "Only PHI-minimized series facts are shown. Raw UIDs, paths, and patient "
                "fields are never displayed."
            )
        )
        for row, summary in enumerate(candidates):
            values = self._row_values(summary)
            warning_text = "\n".join(summary.warnings) or translator("No warnings reported.")
            for column, value in enumerate(values):
                item = QTableWidgetItem(value)
                item.setToolTip(warning_text)
                if column == 0:
                    item.setData(Qt.UserRole, summary.selector)
                if not summary.supported_by_stable_loader:
                    item.setFlags(item.flags() & ~Qt.ItemIsEnabled & ~Qt.ItemIsSelectable)
                self.table.setItem(row, column, item)
        header = self.table.horizontalHeader()
        for column in range(len(self._HEADERS)):
            mode = QHeaderView.Stretch if column in {2, 9} else QHeaderView.ResizeToContents
            header.setSectionResizeMode(column, mode)
        self.table.resizeRowsToContents()
        self.table.itemSelectionChanged.connect(self._selection_changed)
        self.table.itemDoubleClicked.connect(self._activate_row)
        layout.addWidget(self.table, 1)

        supported_count = sum(item.supported_by_stable_loader for item in candidates)
        self.availability_note = QLabel()
        self.availability_note.setObjectName("warningBanner")
        self.availability_note.setWordWrap(True)
        self.availability_note.setVisible(supported_count == 0)
        self.availability_note.setText(
            translator(
                "No series in this source is supported by the stable loader. Review each "
                "warning and choose another source."
            )
        )
        layout.addWidget(self.availability_note)

        self.buttons = QDialogButtonBox(QDialogButtonBox.Open | QDialogButtonBox.Cancel)
        self.open_button = self.buttons.button(QDialogButtonBox.Open)
        self.cancel_button = self.buttons.button(QDialogButtonBox.Cancel)
        self.open_button.setText(translator("Open selected series"))
        self.open_button.setObjectName("primary")
        self.open_button.setEnabled(False)
        self.cancel_button.setText(translator("Cancel"))
        self.cancel_button.setDefault(True)
        self.cancel_button.setFocus(Qt.OtherFocusReason)
        self.open_button.clicked.connect(self.accept)
        self.cancel_button.clicked.connect(self.reject)
        layout.addWidget(self.buttons)

        screen = QApplication.primaryScreen()
        if screen is not None:
            available = screen.availableGeometry()
            self.resize(
                min(980, max(640, available.width() - 64)),
                min(600, max(420, available.height() - 64)),
            )
        else:
            self.resize(900, 560)
        self.table.setFocus(Qt.OtherFocusReason)

    @property
    def selected_selector(self) -> str | None:
        return self._selected_selector

    def _row_values(self, summary: SeriesSummary) -> tuple[str, ...]:
        tr = self._translator
        dimensions = (
            f"{summary.rows} × {summary.columns}"
            if summary.rows is not None and summary.columns is not None
            else tr("Unknown")
        )
        pixel_spacing = (
            f"{summary.pixel_spacing_mm[0]:g} × {summary.pixel_spacing_mm[1]:g} mm"
            if summary.pixel_spacing_mm is not None
            else tr("Unknown")
        )
        thickness = (
            f"{summary.slice_thickness_mm:g} mm"
            if summary.slice_thickness_mm is not None
            else tr("Unknown")
        )
        counts = f"{summary.instance_count} / {summary.slice_count} / {summary.frame_count}"
        return (
            tr("Available") if summary.supported_by_stable_loader else tr("Unavailable"),
            summary.modality,
            summary.series_description or tr("No description"),
            str(summary.series_number) if summary.series_number is not None else "—",
            counts,
            dimensions,
            pixel_spacing,
            thickness,
            tr("Consistent") if summary.geometry_consistent else tr("Needs review"),
            "\n".join(summary.warnings) or tr("No warnings reported."),
        )

    def _selection_changed(self) -> None:
        rows = {index.row() for index in self.table.selectionModel().selectedRows()}
        selector: str | None = None
        if len(rows) == 1:
            item = self.table.item(rows.pop(), 0)
            if item is not None and bool(item.flags() & Qt.ItemIsEnabled):
                value = item.data(Qt.UserRole)
                selector = str(value) if value else None
        self._selected_selector = selector
        self.open_button.setEnabled(selector is not None)

    def _activate_row(self, item: QTableWidgetItem) -> None:
        status = self.table.item(item.row(), 0)
        if status is None or not bool(status.flags() & Qt.ItemIsEnabled):
            return
        self.table.selectRow(item.row())
        self._selection_changed()
        if self._selected_selector is not None:
            self.accept()


class ViewerPage(_BilingualPage, QWidget):
    imageChanged = pyqtSignal(object)
    statusChanged = pyqtSignal(str)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.service = ImageService()
        self.local_runner = TaskRunner(max_workers=1, thread_name_prefix="openmedvisionx-local")
        self.watcher = TaskWatcher(self)
        self.study: LoadedStudy | None = None
        self.image: ImageData | None = None
        self._gray_mapping: GrayscaleDisplayMapping | None = None
        self._color_mapping = ColorDisplayMapping()
        self._active_load_task: BackgroundTask[Any] | None = None
        self._active_local_task: BackgroundTask[Any] | None = None
        self._source_path: Path | None = None
        self._pending_source_path: Path | None = None
        self._paired_plane_key: tuple[str, int] | None = None
        self._paired_mask: np.ndarray | None = None
        self._paired_annotations: AnnotationOverlay | None = None
        self._paired_mask_name = ""
        self._paired_annotation_name = ""
        self._active_layer_id: str | None = None
        self._hidden_label_values: dict[str, set[int]] = {}
        self._last_measurement_metrics: dict[str, float] = {}
        self._metric_roi_plane_key: tuple[str, int] | None = None
        self._metric_roi: tuple[int, int, int, int] | None = None
        self._status_source = "Open a local image, DICOM folder/ZIP, or NIfTI volume."
        self._status_translatable = True
        self._measurement_status: tuple[str, object] | None = None
        self._active_view_name = "axial"
        self._hover_view_name: str | None = None
        self._hover_world_ras: tuple[float, float, float] | None = None
        self._build_ui()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 8, 0, 0)
        root.setSpacing(10)
        toolbar_surface = QFrame()
        toolbar_surface.setObjectName("toolbarSurface")
        toolbar = QGridLayout(toolbar_surface)
        toolbar.setContentsMargins(10, 9, 10, 9)
        toolbar.setHorizontalSpacing(8)
        toolbar.setVerticalSpacing(8)
        import_actions = QHBoxLayout()
        import_actions.setSpacing(8)
        self.open_file_button = QPushButton("Open image / DICOM ZIP")
        self.open_file_button.setObjectName("primary")
        self.open_file_button.setShortcut(QKeySequence.Open)
        self.open_file_button.setToolTip("Open an image, DICOM ZIP, or NIfTI file (Ctrl+O)")
        self.open_file_button.setAccessibleName("Open local medical image")
        self.open_file_button.clicked.connect(self._open_file)
        self.open_folder_button = QPushButton("Open DICOM folder")
        self.open_folder_button.setShortcut(QKeySequence("Ctrl+Shift+O"))
        self.open_folder_button.setToolTip("Open a DICOM folder (Ctrl+Shift+O)")
        self.open_folder_button.setAccessibleName("Open a local DICOM folder")
        self.open_folder_button.clicked.connect(self._open_folder)
        self.cancel_button = QPushButton("Cancel loading")
        self.cancel_button.clicked.connect(self._cancel_loading)
        self.cancel_button.setEnabled(False)
        self.cancel_button.setVisible(False)
        for widget in (
            self.open_file_button,
            self.open_folder_button,
            self.cancel_button,
        ):
            import_actions.addWidget(widget)
        import_actions.addStretch(1)
        toolbar.addLayout(import_actions, 0, 0, 1, 2)

        view_actions = QHBoxLayout()
        view_actions.setSpacing(8)
        self.histogram_button = QPushButton("Histogram")
        self.histogram_button.setToolTip("Open an image first to inspect decoded values.")
        self.histogram_button.clicked.connect(self._show_histogram)
        self.histogram_button.setEnabled(False)
        self.fit_button = QPushButton("Fit views")
        self.fit_button.setShortcut(QKeySequence("Ctrl+0"))
        self.fit_button.setToolTip("Fit all image views (Ctrl+0)")
        self.fit_button.clicked.connect(self._fit_views)
        self.fit_button.setEnabled(False)
        view_actions.addWidget(self.histogram_button)
        view_actions.addWidget(self.fit_button)
        view_actions.addStretch(1)
        toolbar.addLayout(view_actions, 2, 0, 1, 2)

        interaction_actions = QHBoxLayout()
        interaction_actions.setSpacing(8)
        self.tool_label = QLabel("Tool:")
        self.measurement_combo = QComboBox()
        self.tool_label.setBuddy(self.measurement_combo)
        self.measurement_combo.setAccessibleName("Image interaction tool")
        self.measurement_combo.setAccessibleDescription(
            "Choose pan and zoom, distance, area or annotation mode."
        )
        self.measurement_combo.addItems(["Pan / zoom", "Distance", "Area / ROI", "Annotation"])
        self.measurement_combo.currentIndexChanged.connect(self._measurement_mode_changed)
        self.measurement_combo.setEnabled(False)
        self.clear_button = QPushButton("Clear marks")
        self.clear_button.clicked.connect(self._clear_marks)
        self.clear_button.setEnabled(False)
        self.clear_button.setToolTip("Open an image before creating or clearing marks.")
        self.spacing_button = QPushButton("Set raster spacing…")
        self.spacing_button.clicked.connect(self._set_raster_spacing)
        self.spacing_button.setVisible(False)
        for widget in (
            self.tool_label,
            self.measurement_combo,
            self.clear_button,
            self.spacing_button,
        ):
            interaction_actions.addWidget(widget)
        interaction_actions.addStretch(1)
        toolbar.addLayout(interaction_actions, 1, 0, 1, 2)
        toolbar.setColumnStretch(0, 1)
        root.addWidget(toolbar_surface)

        self.progress = QProgressBar()
        self.progress.setVisible(False)
        root.addWidget(self.progress)

        self.workspace_splitter = QSplitter(Qt.Horizontal)
        views = QWidget()
        self.view_grid = QGridLayout(views)
        self.view_grid.setContentsMargins(0, 0, 0, 0)
        self.view_grid.setSpacing(8)
        self.view_context = QLabel()
        self.view_context.setObjectName("viewContext")
        self.view_context.setWordWrap(True)
        self.view_context.setAccessibleName("Active image view and coordinate status")
        self.view_grid.addWidget(self.view_context, 0, 0, 1, 2)
        self.axial_view = ImageView("Image / Axial")
        self.coronal_view = ImageView("Coronal (RAS+)")
        self.sagittal_view = ImageView("Sagittal (RAS+)")
        self.aux_view = ImageView("Overlay / comparison")
        self._views_by_name = {
            "axial": self.axial_view,
            "coronal": self.coronal_view,
            "sagittal": self.sagittal_view,
            "projection": self.aux_view,
        }
        empty_messages = {
            "axial": "Open a local image above to inspect pixels and geometry.",
            "coronal": "Coronal view appears when a 3-D volume is open.",
            "sagittal": "Sagittal view appears when a 3-D volume is open.",
            "projection": "A derived comparison view appears when available.",
        }
        for name, view in self._views_by_name.items():
            view.pixelHovered.connect(self._pixel_hovered)
            view.measurementCompleted.connect(self._measurement_completed)
            view.activated.connect(lambda selected=name: self._set_active_view(selected))
            view.navigationRequested.connect(
                lambda x, y, selected=name: self._navigate_from_view(selected, x, y)
            )
            view.set_empty_message(empty_messages[name])
        self._layout_image_views(has_orthogonal_views=False, has_projection=False)
        self._set_active_view("axial")
        self.workspace_splitter.addWidget(views)

        panel = QScrollArea()
        panel.setWidgetResizable(True)
        panel.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        panel.setMinimumWidth(270)
        controls = QWidget()
        control_layout = QVBoxLayout(controls)
        self.navigation_group = QGroupBox("Navigation")
        self.navigation_group.setVisible(False)
        navigation_form = QFormLayout(self.navigation_group)
        navigation_form.setRowWrapPolicy(QFormLayout.WrapAllRows)
        navigation_form.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
        self.frame_slider = QSlider(Qt.Horizontal)
        self.frame_slider.valueChanged.connect(self._plane_changed)
        self.play_button = QPushButton("Play pages")
        self.play_button.clicked.connect(self._toggle_playback)
        self.play_timer = QTimer(self)
        self.play_timer.setInterval(250)
        self.play_timer.timeout.connect(self._next_frame)
        self.coronal_slider = QSlider(Qt.Horizontal)
        self.coronal_slider.valueChanged.connect(self._plane_changed)
        self.sagittal_slider = QSlider(Qt.Horizontal)
        self.sagittal_slider.valueChanged.connect(self._plane_changed)
        self.frame_label = QLabel("Slice / page")
        self.coronal_label = QLabel("Coronal Y")
        self.sagittal_label = QLabel("Sagittal X")
        self.frame_label.setBuddy(self.frame_slider)
        self.coronal_label.setBuddy(self.coronal_slider)
        self.sagittal_label.setBuddy(self.sagittal_slider)
        self.frame_slider.setAccessibleName("Axial slice or page")
        self.frame_slider.setAccessibleDescription(
            "Use arrow keys for one step, Page Up or Page Down for larger steps, and Home or End."
        )
        self.coronal_slider.setAccessibleName("Coronal Y slice")
        self.sagittal_slider.setAccessibleName("Sagittal X slice")
        navigation_form.addRow(self.frame_label, self.frame_slider)
        navigation_form.addRow(self.coronal_label, self.coronal_slider)
        navigation_form.addRow(self.sagittal_label, self.sagittal_slider)
        navigation_form.addRow(self.play_button)
        control_layout.addWidget(self.navigation_group)

        self.display_group = QGroupBox("Display")
        self.display_group.setToolTip("Display mapping changes only the view, not decoded data.")
        self.display_group.setEnabled(False)
        display_form = QFormLayout(self.display_group)
        display_form.setRowWrapPolicy(QFormLayout.WrapAllRows)
        display_form.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
        display_note = QLabel("View only — source values stay unchanged.")
        display_note.setWordWrap(True)
        display_note.setObjectName("mutedText")
        display_form.addRow(display_note)
        self.lower_spin = QDoubleSpinBox()
        self.upper_spin = QDoubleSpinBox()
        for spin in (self.lower_spin, self.upper_spin):
            spin.setRange(-1e9, 1e9)
            spin.setDecimals(3)
            spin.valueChanged.connect(self._display_range_changed)
        self.range_label = QLabel("Intensity range")
        _add_accessible_form_row(
            display_form,
            "Lower",
            self.lower_spin,
            description="Lower bound of the display mapping; decoded values are unchanged.",
        )
        _add_accessible_form_row(
            display_form,
            "Upper",
            self.upper_spin,
            description="Upper bound of the display mapping; decoded values are unchanged.",
        )
        self.brightness_spin = QDoubleSpinBox()
        self.contrast_spin = QDoubleSpinBox()
        self.gamma_spin = QDoubleSpinBox()
        self.brightness_spin.setRange(-1.0, 1.0)
        self.contrast_spin.setRange(0.1, 5.0)
        self.gamma_spin.setRange(0.1, 5.0)
        for spin, value in (
            (self.brightness_spin, 0.0),
            (self.contrast_spin, 1.0),
            (self.gamma_spin, 1.0),
        ):
            spin.setDecimals(2)
            spin.setSingleStep(0.05)
            spin.setValue(value)
            spin.valueChanged.connect(self._color_mapping_changed)
            spin.setEnabled(False)
        _add_accessible_form_row(display_form, "RGB brightness", self.brightness_spin)
        _add_accessible_form_row(display_form, "RGB contrast", self.contrast_spin)
        _add_accessible_form_row(display_form, "RGB gamma", self.gamma_spin)
        auto_button = QPushButton("Auto range")
        auto_button.clicked.connect(self._reset_display_mapping)
        display_form.addRow(self.range_label, auto_button)
        control_layout.addWidget(self.display_group)

        local_group = QGroupBox("Local files & export")
        local_layout = QVBoxLayout(local_group)
        local_note = QLabel(
            "Only selected files are used. Nothing nearby is scanned or uploaded, and exports "
            "never overwrite an existing file."
        )
        local_note.setWordWrap(True)
        local_note.setObjectName("mutedText")
        local_layout.addWidget(local_note)
        local_buttons = QGridLayout()
        self.pair_mask_button = QPushButton("Mask…")
        self.pair_mask_button.clicked.connect(self._select_local_mask)
        self.pair_annotation_button = QPushButton("Annotations…")
        self.pair_annotation_button.clicked.connect(self._select_local_annotations)
        self.import_dicom_annotation_button = QPushButton("DICOM SEG / RTSTRUCT…")
        self.import_dicom_annotation_button.clicked.connect(self._select_dicom_annotation)
        self.import_dicom_annotation_button.setAccessibleName(
            "Import a DICOM SEG or RTSTRUCT for the selected image series"
        )
        self.import_label_map_button = QPushButton("Label map…")
        self.import_label_map_button.clicked.connect(self._select_label_map)
        self.import_label_map_button.setAccessibleName(
            "Import a lossless PNG, TIFF, or NIfTI label map for the selected image series"
        )
        self.export_button = QPushButton("Export PNG…")
        self.export_button.clicked.connect(self._export_rendered_plane)
        self.save_experiment_button = QPushButton("Save record…")
        self.save_experiment_button.clicked.connect(self._save_experiment)
        self.cancel_local_button = QPushButton("Cancel")
        self.cancel_local_button.clicked.connect(self._cancel_local_operation)
        for button in (
            self.pair_mask_button,
            self.pair_annotation_button,
            self.import_dicom_annotation_button,
            self.import_label_map_button,
            self.export_button,
            self.save_experiment_button,
            self.cancel_local_button,
        ):
            button.setEnabled(False)
        local_buttons.addWidget(self.pair_mask_button, 0, 0)
        local_buttons.addWidget(self.pair_annotation_button, 1, 0)
        local_buttons.addWidget(self.import_dicom_annotation_button, 2, 0)
        local_buttons.addWidget(self.import_label_map_button, 3, 0)
        local_buttons.addWidget(self.export_button, 4, 0)
        local_buttons.addWidget(self.save_experiment_button, 5, 0)
        local_buttons.addWidget(self.cancel_local_button, 6, 0)
        local_layout.addLayout(local_buttons)
        self.pairing_status = QLabel("No local mask or annotation is paired.")
        self.pairing_status.setWordWrap(True)
        local_layout.addWidget(self.pairing_status)
        control_layout.addWidget(local_group)

        info_group = QGroupBox("Image details")
        info_layout = QVBoxLayout(info_group)
        self.warning_label = QLabel()
        self.warning_label.setWordWrap(True)
        self.warning_label.setObjectName("warningBanner")
        self.warning_label.setVisible(False)
        self.info = QPlainTextEdit()
        self.info.setReadOnly(True)
        self.info.setMinimumWidth(0)
        info_layout.addWidget(self.warning_label)
        info_layout.addWidget(self.info)
        control_layout.addWidget(info_group, 1)
        panel.setWidget(controls)
        self.sidebar_tabs = QTabWidget()
        self.sidebar_tabs.setObjectName("viewerSidebarTabs")
        self.sidebar_tabs.setDocumentMode(True)
        self.sidebar_tabs.setMinimumWidth(280)
        self.sidebar_tabs.setAccessibleName("Viewer controls and clinical layer browser")
        self.sidebar_tabs.addTab(panel, "Controls")
        self.study_layers = StudyLayersPanel(language=self._language)
        self.sidebar_tabs.addTab(self.study_layers, "Layers")
        self.study_layers.activeLayerChanged.connect(self._active_layer_changed)
        self.study_layers.layerVisibilityChangeRequested.connect(self._layer_visibility_requested)
        self.study_layers.layerLockChangeRequested.connect(self._layer_lock_requested)
        self.study_layers.layerOpacityChangeRequested.connect(self._layer_opacity_requested)
        self.study_layers.labelVisibilityChangeRequested.connect(self._label_visibility_requested)
        self.workspace_splitter.addWidget(self.sidebar_tabs)
        self.workspace_splitter.setStretchFactor(0, 4)
        self.workspace_splitter.setStretchFactor(1, 1)
        self.workspace_splitter.setSizes([960, 320])
        self.workspace_splitter.setChildrenCollapsible(False)
        root.addWidget(self.workspace_splitter, 1)

        self.status = QLabel("Open a local image, DICOM folder/ZIP, or NIfTI volume.")
        self.status.setObjectName("statusStrip")
        self.status.setWordWrap(True)
        root.addWidget(self.status)

        tab_order = (
            self.open_file_button,
            self.open_folder_button,
            self.measurement_combo,
            self.clear_button,
            self.histogram_button,
            self.fit_button,
            self.axial_view,
            self.frame_slider,
            self.coronal_slider,
            self.sagittal_slider,
            self.lower_spin,
            self.upper_spin,
            self.pair_mask_button,
            self.pair_annotation_button,
            self.import_dicom_annotation_button,
            self.import_label_map_button,
            self.export_button,
            self.save_experiment_button,
            self.study_layers.tree,
        )
        for first, second in zip(tab_order, tab_order[1:], strict=False):
            QWidget.setTabOrder(first, second)
        self._update_view_context()

    def _layout_image_views(
        self,
        *,
        has_orthogonal_views: bool,
        has_projection: bool,
    ) -> None:
        """Keep the primary image large while revealing MPR views only when useful."""

        for view in self._views_by_name.values():
            self.view_grid.removeWidget(view)
        if not has_orthogonal_views:
            self.view_grid.addWidget(self.axial_view, 1, 0, 2, 2)
            self.coronal_view.hide()
            self.sagittal_view.hide()
            self.aux_view.hide()
            return
        self.view_grid.addWidget(self.axial_view, 1, 0)
        self.view_grid.addWidget(self.coronal_view, 1, 1)
        if has_projection:
            self.view_grid.addWidget(self.sagittal_view, 2, 0)
            self.view_grid.addWidget(self.aux_view, 2, 1)
        else:
            self.view_grid.addWidget(self.sagittal_view, 2, 0, 1, 2)
        self.coronal_view.show()
        self.sagittal_view.show()
        self.aux_view.setVisible(has_projection)

    def _set_active_view(self, name: str) -> None:
        view = self._views_by_name.get(name)
        if view is None or (self.image is not None and view.isHidden()):
            return
        self._active_view_name = name
        self._hover_view_name = None
        self._hover_world_ras = None
        for view_name, image_view in self._views_by_name.items():
            image_view.set_active(view_name == name)
        if self.image is not None:
            self._render_paired_overlays()
        self._update_view_context()

    def _active_plane_position(self) -> tuple[str, int, int]:
        if isinstance(self.image, ImageSequence2D):
            return ("Page / frame", self.frame_slider.value() + 1, self.image.frame_count)
        if isinstance(self.image, ImageVolume):
            if self._active_view_name == "coronal":
                return ("Coronal Y", self.coronal_slider.value() + 1, self.image.shape[1])
            if self._active_view_name == "sagittal":
                return ("Sagittal X", self.sagittal_slider.value() + 1, self.image.shape[2])
            if self._active_view_name == "projection":
                return ("Axial projection", 1, 1)
            return ("Axial Z", self.frame_slider.value() + 1, self.image.shape[0])
        return ("2-D image", 1, 1)

    def _update_navigation_labels(self) -> None:
        if isinstance(self.image, ImageSequence2D):
            self.frame_label.setText(
                self._tr("Page / frame {current}/{total}").format(
                    current=self.frame_slider.value() + 1,
                    total=self.image.frame_count,
                )
            )
            return
        if isinstance(self.image, ImageVolume):
            self.frame_label.setText(
                self._tr("Axial Z {current}/{total}").format(
                    current=self.frame_slider.value() + 1,
                    total=self.image.shape[0],
                )
            )
            self.coronal_label.setText(
                self._tr("Coronal Y {current}/{total}").format(
                    current=self.coronal_slider.value() + 1,
                    total=self.image.shape[1],
                )
            )
            self.sagittal_label.setText(
                self._tr("Sagittal X {current}/{total}").format(
                    current=self.sagittal_slider.value() + 1,
                    total=self.image.shape[2],
                )
            )

    def _update_view_context(self) -> None:
        label, current, total = self._active_plane_position()
        view_name = {
            "axial": "Axial",
            "coronal": "Coronal",
            "sagittal": "Sagittal",
            "projection": "Axial projection",
        }.get(self._active_view_name, self._active_view_name)
        coordinate = self._tr("RAS+ hover: —")
        if self._hover_world_ras is not None:
            r, a, s = self._hover_world_ras
            hover_view = {
                "axial": "Axial",
                "coronal": "Coronal",
                "sagittal": "Sagittal",
                "projection": "Axial projection",
            }.get(self._hover_view_name or "", self._hover_view_name or "")
            coordinate = (
                f"{self._tr('RAS+ hover')} ({self._tr(hover_view)}): "
                f"R {r:.2f}, A {a:.2f}, S {s:.2f} mm"
            )
        if self.image is None:
            text = self._tr(
                "Next: open a local image. Axial is the default active view for downstream tasks."
            )
        else:
            text = (
                f"{self._tr('Active view')}: {self._tr(view_name)} · "
                f"{self._tr(label)} {current}/{total} · {coordinate} · "
                f"{self._tr('Used by CT Lab, Models, and Assistant')}"
            )
            if isinstance(self.image, ImageVolume):
                text += f" · {self._tr('Double-click a view to link all MPR planes')}"
        self.view_context.setText(text)
        _set_dynamic_property(self.view_context, "active", self.image is not None)

    def _language_changed(self) -> None:
        self.study_layers.set_language(self._language)
        self.display_group.setToolTip(
            self._tr("Display mapping changes only the view, not decoded data.")
        )
        self.play_button.setText(self._tr("Pause" if self.play_timer.isActive() else "Play pages"))
        window_label = (
            "HU window"
            if self.image is not None and Capability.HU_WINDOWING in self.image.capabilities
            else "Intensity range"
        )
        self.range_label.setText(self._tr(window_label))
        if self.image is not None:
            self._update_views()
            self._update_info()
        self._update_navigation_labels()
        self._update_view_context()
        self._refresh_status()

    def _set_status(self, text: str, *, translatable: bool = True) -> None:
        self._status_source = text
        self._status_translatable = translatable
        self._measurement_status = None
        self._refresh_status()

    def _refresh_status(self) -> None:
        if self._measurement_status is not None:
            mode_label, payload = self._measurement_status
            self.status.setText(f"{self._tr(mode_label)}: {payload}")
            return
        text = self._tr(self._status_source) if self._status_translatable else self._status_source
        self.status.setText(text)

    def _open_file(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            self._tr("Open local image"),
            "",
            self._tr(
                "Supported images (*.dcm *.dicom *.zip *.nii *.nii.gz "
                "*.png *.jpg *.jpeg *.tif *.tiff);;All files (*)"
            ),
        )
        if path:
            self._begin_load(path)

    def _open_folder(self) -> None:
        path = QFileDialog.getExistingDirectory(self, self._tr("Open DICOM folder"))
        if path:
            self._begin_load(path)

    def _cancel_loading(self) -> None:
        if self._active_load_task is None:
            return
        self.cancel_button.setEnabled(False)
        self.cancel_button.setText(self._tr("Cancelling…"))
        self._set_status("Cancelling load safely…")
        self.service.cancel_active()

    def _begin_load(
        self,
        path: str,
        *,
        dicom_series_selector: str | None = None,
        nifti_volume_index: int | None = None,
    ) -> None:
        # Keep the current study usable until the replacement has loaded completely.
        # A failed or cancelled import must not destroy downstream reconstruction/model work.
        self._pending_source_path = Path(path)
        self.progress.setVisible(True)
        self.progress.setValue(0)
        self.open_file_button.setEnabled(False)
        self.open_folder_button.setEnabled(False)
        self.cancel_button.setEnabled(True)
        self.cancel_button.setVisible(True)
        self._set_status("Loading in background…")
        task = self.service.begin_load(
            path,
            prepare_axis_aligned_volume=True,
            dicom_series_selector=dicom_series_selector,
            nifti_volume_index=nifti_volume_index,
        )
        self._active_load_task = task
        self.watcher.watch(
            task,
            success=lambda study, handle=task: self._loaded_if_current(handle, study),
            error=lambda error, handle=task: self._load_failed_if_current(handle, error),
            progress=lambda fraction, message, handle=task: self._load_progress_if_current(
                handle, fraction, message
            ),
        )

    def _clear_loaded_state(self, *, emit_change: bool = True) -> None:
        self.play_timer.stop()
        self.play_button.setText(self._tr("Play pages"))
        self.play_button.setEnabled(False)
        self._cancel_local_operation(clear_handle=True)
        self.study = None
        self.image = None
        self._source_path = None
        self._paired_plane_key = None
        self._paired_mask = None
        self._paired_annotations = None
        self._paired_mask_name = ""
        self._paired_annotation_name = ""
        self._active_layer_id = None
        self._hidden_label_values.clear()
        self.study_layers.clear()
        self._last_measurement_metrics.clear()
        self._metric_roi_plane_key = None
        self._metric_roi = None
        for view in (self.axial_view, self.coronal_view, self.sagittal_view, self.aux_view):
            view.set_orientation_labels(None)
            view.clear_image()
        self._layout_image_views(has_orthogonal_views=False, has_projection=False)
        self._set_active_view("axial")
        self.info.clear()
        self.warning_label.clear()
        self.warning_label.setVisible(False)
        self.pairing_status.setText(self._tr("No local mask or annotation is paired."))
        self.histogram_button.setEnabled(False)
        self.fit_button.setEnabled(False)
        self.measurement_combo.setEnabled(False)
        self.clear_button.setEnabled(False)
        self.display_group.setEnabled(False)
        self.navigation_group.setVisible(False)
        for button in (
            self.pair_mask_button,
            self.pair_annotation_button,
            self.import_dicom_annotation_button,
            self.import_label_map_button,
            self.export_button,
            self.save_experiment_button,
        ):
            button.setEnabled(False)
        if emit_change:
            self.imageChanged.emit(None)

    def _loaded_if_current(self, task: BackgroundTask[Any], study: LoadedStudy) -> None:
        if task is not self._active_load_task:
            return
        self._active_load_task = None
        self._loaded(study)

    def _load_failed_if_current(
        self,
        task: BackgroundTask[Any],
        error: BaseException,
    ) -> None:
        if task is not self._active_load_task:
            return
        self._active_load_task = None
        self._load_failed(error)

    def _load_progress_if_current(
        self,
        task: BackgroundTask[Any],
        fraction: float,
        message: str,
    ) -> None:
        if task is self._active_load_task:
            self._task_progress(fraction, message)

    def _task_progress(self, fraction: float, message: str) -> None:
        self.progress.setValue(int(round(fraction * 100)))
        self._set_status(message)

    def _loaded(self, study: LoadedStudy) -> None:
        source_path = self._pending_source_path
        self._pending_source_path = None
        self._clear_loaded_state(emit_change=False)
        self.study = study
        self.image = study.display_image
        self._source_path = source_path
        self.open_file_button.setEnabled(True)
        self.open_folder_button.setEnabled(True)
        self.cancel_button.setEnabled(False)
        self.cancel_button.setText(self._tr("Cancel loading"))
        self.cancel_button.setVisible(False)
        self.progress.setVisible(False)
        self.histogram_button.setEnabled(True)
        self.fit_button.setEnabled(True)
        self.measurement_combo.setEnabled(True)
        self.clear_button.setEnabled(True)
        self.display_group.setEnabled(True)
        can_overlay = Capability.OVERLAY in self.image.capabilities
        self.pair_mask_button.setEnabled(can_overlay)
        self.pair_annotation_button.setEnabled(can_overlay)
        self.import_dicom_annotation_button.setEnabled(
            bool(
                study.reference_series is not None
                and study.reference_series.series_instance_uid is not None
            )
        )
        self.import_label_map_button.setEnabled(study.reference_series is not None)
        self.export_button.setEnabled(Capability.VIEW_2D in self.image.capabilities)
        self.save_experiment_button.setEnabled(True)
        self._configure_for_image()
        self.study_layers.set_study(study.domain_study)
        self._reset_display_mapping()
        self._update_views(fit=True)
        self._update_info()
        self.imageChanged.emit(self.image)
        loaded_status = self._tr("Loaded {kind}: {shape}").format(
            kind=study.source_kind,
            shape=self.image.shape,
        )
        self._set_status(loaded_status)
        self.statusChanged.emit(loaded_status)

    def _load_failed(self, error: BaseException) -> None:
        if isinstance(error, SeriesSelectionRequiredError):
            # Opening a modal dialog directly from TaskWatcher._poll() would
            # start a nested event loop while the completed watch is still
            # being removed. Defer it one turn so a retry cannot be dropped by
            # a re-entrant timer callback.
            candidates = tuple(error.candidates)
            QTimer.singleShot(0, lambda: self._select_dicom_series(candidates))
            return
        if isinstance(error, NiftiVolumeSelectionRequiredError):
            volume_count = error.volume_count
            QTimer.singleShot(0, lambda: self._select_nifti_volume(volume_count))
            return
        self._pending_source_path = None
        self.open_file_button.setEnabled(True)
        self.open_folder_button.setEnabled(True)
        self.cancel_button.setEnabled(False)
        self.cancel_button.setText(self._tr("Cancel loading"))
        self.cancel_button.setVisible(False)
        self.progress.setVisible(False)
        if error.__class__.__name__ == "OperationCancelled":
            self._set_status("Loading cancelled.")
            return
        self._set_status(f"Load failed: {error}")
        QMessageBox.critical(
            self,
            self._tr("OpenMedVisionX — load failed"),
            str(error),
        )

    def _build_series_selection_dialog(
        self,
        candidates: Sequence[SeriesSummary],
    ) -> _DicomSeriesSelectionDialog:
        return _DicomSeriesSelectionDialog(
            candidates,
            translator=self._tr,
            parent=self,
        )

    def _select_dicom_series(self, candidates: Sequence[SeriesSummary]) -> None:
        """Require an explicit PHI-minimized series choice, then retry the same source."""

        pending_path = self._pending_source_path
        self.open_file_button.setEnabled(True)
        self.open_folder_button.setEnabled(True)
        self.cancel_button.setEnabled(False)
        self.cancel_button.setText(self._tr("Cancel loading"))
        self.cancel_button.setVisible(False)
        self.progress.setVisible(False)
        self._set_status(
            self._tr("{count} DICOM series found. Select one to continue.").format(
                count=len(candidates)
            ),
            translatable=False,
        )
        dialog = self._build_series_selection_dialog(candidates)
        accepted = dialog.exec_() == QDialog.Accepted
        selector = dialog.selected_selector
        dialog.deleteLater()
        if accepted and selector is not None and pending_path is not None:
            self._begin_load(
                str(pending_path),
                dicom_series_selector=selector,
            )
            return
        self._pending_source_path = None
        self._set_status(
            self._tr("Series selection cancelled; the current image is unchanged."),
            translatable=False,
        )

    def _select_nifti_volume(self, volume_count: int) -> None:
        """Require a visible, one-based time-point/channel selection for 4-D NIfTI."""

        pending_path = self._pending_source_path
        self.open_file_button.setEnabled(True)
        self.open_folder_button.setEnabled(True)
        self.cancel_button.setEnabled(False)
        self.cancel_button.setText(self._tr("Cancel loading"))
        self.cancel_button.setVisible(False)
        self.progress.setVisible(False)
        self._set_status(
            self._tr("This NIfTI contains {count} volumes. Select one to continue.").format(
                count=volume_count
            ),
            translatable=False,
        )
        selected, accepted = QInputDialog.getInt(
            self,
            self._tr("Choose a NIfTI volume"),
            self._tr("Time point / channel (1 to {count})").format(count=volume_count),
            1,
            1,
            max(1, volume_count),
            1,
        )
        if accepted and pending_path is not None:
            self._begin_load(
                str(pending_path),
                nifti_volume_index=selected - 1,
            )
            return
        self._pending_source_path = None
        self._set_status(
            self._tr("Volume selection cancelled; the current image is unchanged."),
            translatable=False,
        )

    def _configure_for_image(self) -> None:
        assert self.image is not None
        image = self.image
        capabilities = image.capabilities
        has_orthogonal_views = Capability.ORTHOGONAL_VIEWS in capabilities
        has_frame_navigation = Capability.FRAME_NAVIGATION in capabilities
        has_frame_playback = Capability.FRAME_PLAYBACK in capabilities
        has_projection = bool(
            has_orthogonal_views
            and self.study is not None
            and self.study.volume_projection is not None
        )
        for view in self._views_by_name.values():
            view.set_orientation_labels(None)
        if has_orthogonal_views:
            # Axis-aligned display volumes are canonical RAS+.  Markers describe
            # the patient directions visible at each screen edge; they are not
            # shown for ordinary rasters where patient orientation is unknown.
            self.axial_view.set_orientation_labels(("P", "R", "A", "L"))
            self.coronal_view.set_orientation_labels(("I", "R", "S", "L"))
            self.sagittal_view.set_orientation_labels(("I", "A", "S", "P"))
            self.aux_view.set_orientation_labels(("P", "R", "A", "L"))
        self._layout_image_views(
            has_orthogonal_views=has_orthogonal_views,
            has_projection=has_projection,
        )
        if self._views_by_name[self._active_view_name].isHidden():
            self._set_active_view("axial")
        self.navigation_group.setVisible(has_frame_navigation or has_orthogonal_views)
        self.frame_label.setVisible(has_frame_navigation)
        self.frame_slider.setVisible(has_frame_navigation)
        self.play_button.setVisible(has_frame_playback)
        self.play_button.setEnabled(has_frame_playback)
        self.coronal_label.setVisible(has_orthogonal_views)
        self.coronal_slider.setVisible(has_orthogonal_views)
        self.sagittal_label.setVisible(has_orthogonal_views)
        self.sagittal_slider.setVisible(has_orthogonal_views)
        self.frame_slider.blockSignals(True)
        self.frame_slider.setRange(0, image.shape[0] - 1 if has_frame_navigation else 0)
        self.frame_slider.setValue((image.shape[0] - 1) // 2 if has_orthogonal_views else 0)
        self.frame_slider.blockSignals(False)
        if has_orthogonal_views:
            self.coronal_slider.blockSignals(True)
            self.coronal_slider.setRange(0, image.shape[1] - 1)
            self.coronal_slider.setValue((image.shape[1] - 1) // 2)
            self.coronal_slider.blockSignals(False)
            self.sagittal_slider.blockSignals(True)
            self.sagittal_slider.setRange(0, image.shape[2] - 1)
            self.sagittal_slider.setValue((image.shape[2] - 1) // 2)
            self.sagittal_slider.blockSignals(False)
        self.spacing_button.setVisible(
            isinstance(image, RasterImage2D) and Capability.PHYSICAL_MEASUREMENT not in capabilities
        )
        self.range_label.setText(
            self._tr(
                "HU window" if Capability.HU_WINDOWING in image.capabilities else "Intensity range"
            )
        )
        self._update_navigation_labels()
        self._update_view_context()

    def _reset_display_mapping(self) -> None:
        if self.image is None:
            return
        values = self.current_plane()
        if values.ndim == 2:
            self._gray_mapping = GrayscaleDisplayMapping.from_percentiles(
                values,
                invert=bool(self.image.runtime_metadata.get("display_inverted", False)),
            )
            for spin, value in (
                (self.lower_spin, self._gray_mapping.lower),
                (self.upper_spin, self._gray_mapping.upper),
            ):
                spin.blockSignals(True)
                spin.setValue(value)
                spin.blockSignals(False)
            self.lower_spin.setEnabled(True)
            self.upper_spin.setEnabled(True)
        else:
            self._gray_mapping = None
            self.lower_spin.setEnabled(False)
            self.upper_spin.setEnabled(False)
            self._color_mapping = ColorDisplayMapping()
            for spin, value in (
                (self.brightness_spin, self._color_mapping.brightness),
                (self.contrast_spin, self._color_mapping.contrast),
                (self.gamma_spin, self._color_mapping.gamma),
            ):
                spin.blockSignals(True)
                spin.setValue(value)
                spin.blockSignals(False)
        is_color = values.ndim == 3
        for spin in (self.brightness_spin, self.contrast_spin, self.gamma_spin):
            spin.setEnabled(is_color)
        self._update_views()

    def _display_range_changed(self) -> None:
        if self.lower_spin.value() < self.upper_spin.value():
            self._gray_mapping = GrayscaleDisplayMapping(
                self.lower_spin.value(),
                self.upper_spin.value(),
                invert=bool(self._gray_mapping and self._gray_mapping.invert),
            )
            self._update_views()

    def _color_mapping_changed(self) -> None:
        if self.image is None or self.current_plane().ndim != 3:
            return
        self._color_mapping = ColorDisplayMapping(
            brightness=self.brightness_spin.value(),
            contrast=self.contrast_spin.value(),
            gamma=self.gamma_spin.value(),
        )
        self._update_views()

    def _plane_changed(self, value: int) -> None:
        """Change planes without carrying measurements into another coordinate frame."""

        for view in (self.axial_view, self.coronal_view, self.sagittal_view):
            view.clear_measurements()
        self._last_measurement_metrics.clear()
        self._metric_roi_plane_key = None
        self._metric_roi = None
        self._hover_view_name = None
        self._hover_world_ras = None
        self._update_views(value)

    def _navigate_from_view(self, view_name: str, x: int, y: int) -> None:
        """Move all orthogonal planes from one explicit double-click location."""

        if not isinstance(self.image, ImageVolume):
            return
        z_value = self.frame_slider.value()
        y_value = self.coronal_slider.value()
        x_value = self.sagittal_slider.value()
        if view_name in {"axial", "projection"}:
            x_value, y_value = x, y
        elif view_name == "coronal":
            x_value, z_value = x, y
        elif view_name == "sagittal":
            y_value, z_value = x, y
        else:
            return
        for slider, target in (
            (self.frame_slider, z_value),
            (self.coronal_slider, y_value),
            (self.sagittal_slider, x_value),
        ):
            slider.blockSignals(True)
            slider.setValue(max(slider.minimum(), min(slider.maximum(), int(target))))
            slider.blockSignals(False)
        self._hover_view_name = None
        self._hover_world_ras = None
        self._update_views()

    def _update_views(self, _value: int | None = None, *, fit: bool = False) -> None:
        if self.image is None:
            return
        image = self.image
        if isinstance(image, RasterImage2D):
            self.axial_view.set_array(
                image.array,
                grayscale_mapping=self._gray_mapping,
                color_mapping=self._color_mapping,
                pixel_spacing=image.pixel_spacing,
                physical_units=image.pixel_spacing is not None,
                fit=fit,
            )
            self.axial_view.set_title("2-D raster image")
        elif isinstance(image, ImageSequence2D):
            index = self.frame_slider.value()
            self.axial_view.set_array(
                image.array[index],
                grayscale_mapping=self._gray_mapping,
                color_mapping=self._color_mapping,
                fit=fit,
            )
            self.axial_view.set_title(f"Page / frame {index + 1}/{image.frame_count}")
        elif isinstance(image, ImageVolume):
            z = self.frame_slider.value()
            y = self.coronal_slider.value()
            x = self.sagittal_slider.value()
            self.axial_view.set_array(
                image.axial(z),
                grayscale_mapping=self._gray_mapping,
                pixel_spacing=(image.spacing[0], image.spacing[1]),
                physical_units=True,
                fit=fit,
            )
            self.coronal_view.set_array(
                image.coronal(y),
                grayscale_mapping=self._gray_mapping,
                pixel_spacing=(image.spacing[0], image.spacing[2]),
                physical_units=True,
                fit=fit,
            )
            self.sagittal_view.set_array(
                image.sagittal(x),
                grayscale_mapping=self._gray_mapping,
                pixel_spacing=(image.spacing[1], image.spacing[2]),
                physical_units=True,
                fit=fit,
            )
            self.axial_view.set_title(f"Axial RAS+ – z {z + 1}/{image.shape[0]}")
            self.coronal_view.set_title(f"Coronal RAS+ – y {y + 1}/{image.shape[1]}")
            self.sagittal_view.set_title(f"Sagittal RAS+ – x {x + 1}/{image.shape[2]}")
            projection = self.study.volume_projection if self.study is not None else None
            if projection is not None:
                self.aux_view.set_array(
                    projection,
                    grayscale_mapping=self._gray_mapping,
                    pixel_spacing=(image.spacing[0], image.spacing[1]),
                    physical_units=True,
                    fit=fit,
                )
                self.aux_view.set_title("Axial maximum-intensity projection")
        self._render_paired_overlays()
        self._update_navigation_labels()
        self._update_view_context()

    def _current_plane_key(self) -> tuple[str, int]:
        if isinstance(self.image, ImageSequence2D):
            return ("frame", self.frame_slider.value())
        if isinstance(self.image, ImageVolume):
            if self._active_view_name == "coronal":
                return ("coronal", self.coronal_slider.value())
            if self._active_view_name == "sagittal":
                return ("sagittal", self.sagittal_slider.value())
            if self._active_view_name == "projection":
                return ("axial-projection", 0)
            return ("axial", self.frame_slider.value())
        return ("image", 0)

    def _render_paired_overlays(self) -> None:
        for view in self._views_by_name.values():
            view.clear_overlays()
        if isinstance(self.image, ImageVolume):
            z = self.frame_slider.value()
            y = self.coronal_slider.value()
            x = self.sagittal_slider.value()
            self.axial_view.set_crosshair(x, y)
            self.coronal_view.set_crosshair(x, z)
            self.sagittal_view.set_crosshair(y, z)
            if not self.aux_view.isHidden():
                self.aux_view.set_crosshair(x, y)
            self._render_clinical_layers()
        if self.image is None or self._paired_plane_key != self._current_plane_key():
            if self._paired_plane_key is not None:
                self.pairing_status.setText(
                    self._tr(
                        "A local overlay is paired to another slice/page; return to it to view "
                        "the overlay."
                    )
                )
            return
        target_view = self._views_by_name.get(self._active_view_name, self.axial_view)
        if self._paired_mask is not None:
            target_view.set_mask_overlay(self._paired_mask)
        annotations = self._paired_annotations
        if annotations is not None:
            if annotations.boxes:
                target_view.set_box_overlays(
                    np.asarray(annotations.boxes),
                    labels=list(annotations.box_labels),
                )
            if annotations.points:
                target_view.set_point_overlays(
                    np.asarray(annotations.points),
                    labels=list(annotations.point_labels),
                )
        parts = []
        if self._paired_mask is not None:
            parts.append(f"{self._tr('mask')} {self._paired_mask_name!r}")
        if annotations is not None:
            parts.append(f"{self._tr('annotation')} {self._paired_annotation_name!r}")
        if parts:
            self.pairing_status.setText(
                self._tr("Explicit local pairing on the current plane: {items}.").format(
                    items=self._tr(" and ").join(parts)
                )
            )

    @staticmethod
    def _hex_rgb(color: str) -> tuple[int, int, int]:
        normalized = str(color).lstrip("#")
        if len(normalized) != 6:
            return (255, 64, 64)
        try:
            return tuple(int(normalized[index : index + 2], 16) for index in (0, 2, 4))
        except ValueError:
            return (255, 64, 64)

    def _render_clinical_layers(self) -> None:
        """Overlay only validated layers that match the current display geometry."""

        if not isinstance(self.image, ImageVolume) or self.study is None:
            return
        series = self.study.reference_series
        if series is None:
            return
        target_geometry = SpatialGeometry.from_volume(self.image)
        z = self.frame_slider.value()
        y = self.coronal_slider.value()
        x = self.sagittal_slider.value()
        for layer in series.layers:
            if isinstance(layer, VolumeLayer) or not layer.presentation.visible:
                continue
            if isinstance(layer, SegmentationLayer):
                if not layer.geometry.matches(target_geometry):
                    continue
                values = np.asarray(layer.array)
                if values.ndim == 2:
                    values = values[np.newaxis, ...]
                planes = (
                    (self.axial_view, values[z]),
                    (self.coronal_view, values[:, y, :]),
                    (self.sagittal_view, values[:, :, x]),
                )
                hidden = self._hidden_label_values.get(layer.layer_id, set())
                opacity = max(0, min(255, round(layer.presentation.opacity * 180)))
                if layer.value_type is SegmentationValueType.DISCRETE:
                    for label in layer.labels:
                        if label.value in hidden or not label.visible:
                            continue
                        color = self._hex_rgb(label.color)
                        for view, plane in planes:
                            view.set_mask_overlay(
                                np.asarray(plane == label.value),
                                color=color,
                                opacity=opacity,
                            )
                    continue
                label = layer.labels[0]
                if label.value in hidden or not label.visible:
                    continue
                if layer.value_type is SegmentationValueType.BINARY:
                    masks = tuple(np.asarray(plane != 0) for _, plane in planes)
                elif layer.value_type is SegmentationValueType.FRACTIONAL:
                    assert layer.maximum_fractional_value is not None
                    threshold = 0.5 if layer.threshold is None else layer.threshold
                    masks = tuple(
                        np.asarray(plane, dtype=np.float32) / float(layer.maximum_fractional_value)
                        >= threshold
                        for _, plane in planes
                    )
                else:
                    threshold = 0.5 if layer.threshold is None else layer.threshold
                    masks = tuple(
                        np.asarray(plane, dtype=np.float32) >= threshold for _, plane in planes
                    )
                color = self._hex_rgb(label.color)
                for (view, _plane), mask in zip(planes, masks, strict=True):
                    view.set_mask_overlay(mask, color=color, opacity=opacity)
                continue
            if not isinstance(layer, ContourLayer):
                continue
            hidden = self._hidden_label_values.get(layer.layer_id, set())
            inverse_points: list[tuple[ImageView, np.ndarray, str, str]] = []
            for roi in layer.rois:
                if roi.roi_number in hidden or not roi.visible:
                    continue
                for contour in roi.contours:
                    voxel_xyz = self.image.world_ras_to_voxel_xyz(contour.points_ras)
                    if np.max(np.abs(voxel_xyz[:, 2] - z)) <= 0.75:
                        inverse_points.append(
                            (self.axial_view, voxel_xyz[:, (0, 1)], roi.name, roi.color)
                        )
                    if np.max(np.abs(voxel_xyz[:, 1] - y)) <= 0.75:
                        inverse_points.append(
                            (self.coronal_view, voxel_xyz[:, (0, 2)], roi.name, roi.color)
                        )
                    if np.max(np.abs(voxel_xyz[:, 0] - x)) <= 0.75:
                        inverse_points.append(
                            (self.sagittal_view, voxel_xyz[:, (1, 2)], roi.name, roi.color)
                        )
            for view in (self.axial_view, self.coronal_view, self.sagittal_view):
                selected = [item for item in inverse_points if item[0] is view]
                if selected:
                    view.set_polygon_overlays(
                        [item[1] for item in selected],
                        labels=[item[2] for item in selected],
                        colors=[item[3] for item in selected],
                    )

    def current_plane(self) -> np.ndarray:
        if self.image is None:
            raise ViewerError("No image is loaded.")
        if isinstance(self.image, RasterImage2D):
            return self.image.array
        if isinstance(self.image, ImageSequence2D):
            return self.image.array[self.frame_slider.value()]
        if self._active_view_name == "coronal":
            return self.image.coronal(self.coronal_slider.value())
        if self._active_view_name == "sagittal":
            return self.image.sagittal(self.sagittal_slider.value())
        if self._active_view_name == "projection" and self.study is not None:
            projection = self.study.volume_projection
            if projection is not None:
                return projection
        return self.image.axial(self.frame_slider.value())

    def current_model_input(self) -> RasterImage2D | np.ndarray:
        """Return the visible plane with reversible raster coordinates when available."""

        if self.image is None:
            raise ViewerError("No image is loaded.")
        if isinstance(self.image, RasterImage2D):
            return self.image
        if isinstance(self.image, ImageSequence2D):
            index = self.frame_slider.value()
            return RasterImage2D(
                array=self.image.array[index],
                source_type=self.image.source_type,
                intensity_semantics=self.image.intensity_semantics,
                runtime_metadata={
                    "format": self.image.runtime_metadata.get("format", "TIFF"),
                    "frame_index": index,
                },
                bit_depth=self.image.bit_depth,
                color_space=self.image.color_space,
                channel_order=self.image.channel_order,
                alpha_semantics=self.image.alpha_semantics,
                transform_record=self.image.frame_transforms[index],
            )
        return self.current_plane()

    def current_model_input_context(self) -> dict[str, Any]:
        """Describe the visible canonical plane without exposing source metadata."""

        if self.image is None:
            raise ViewerError("No image is loaded.")
        if isinstance(self.image, RasterImage2D):
            preview_only = bool(
                self.image.runtime_metadata.get("preview_only", False)
                or self.image.runtime_metadata.get("access") == "thumbnail"
            )
            return {
                "modality": "generic-image",
                "spacing": self.image.pixel_spacing,
                "intensity_semantics": self.image.intensity_semantics.value,
                "unit": "",
                "preview_only": preview_only,
                "source_shape": tuple(
                    self.image.runtime_metadata.get("source_shape", self.image.shape[:2])
                ),
                "display_shape": tuple(self.image.shape[:2]),
            }
        if isinstance(self.image, ImageSequence2D):
            return {
                "modality": "generic-image",
                "spacing": None,
                "intensity_semantics": self.image.intensity_semantics.value,
                "unit": "",
                "preview_only": bool(
                    self.image.runtime_metadata.get("preview_only", False)
                    or self.image.runtime_metadata.get("access") == "thumbnail_sequence"
                ),
                "source_shape": tuple(
                    self.image.runtime_metadata.get("source_shape", self.image.shape[1:3])
                ),
                "display_shape": tuple(self.image.shape[1:3]),
            }
        modality = {
            "CT": "ct",
            "MR": "mr",
            "MRI": "mr",
            "CR": "xray",
            "DX": "xray",
            "MG": "mammography",
            "US": "ultrasound",
            "PT": "pet",
            "NM": "spect",
        }.get(self.image.modality, "generic-image")
        spacing = self.image.spacing[:2]
        if self._active_view_name == "coronal":
            spacing = (self.image.spacing[0], self.image.spacing[2])
        elif self._active_view_name == "sagittal":
            spacing = (self.image.spacing[1], self.image.spacing[2])
        unit = str(self.image.runtime_metadata.get("units", "")).strip()
        if self.image.intensity_semantics.value == "hounsfield_unit":
            unit = "HU"
        elif self.image.intensity_semantics.value == "suv":
            unit = "SUV"
        return {
            "modality": modality,
            "spacing": spacing,
            "plane": self._current_plane_key()[0],
            "intensity_semantics": self.image.intensity_semantics.value,
            "unit": unit if len(unit) <= 24 else "",
            "preview_only": False,
            "source_shape": tuple(self.current_plane().shape[:2]),
            "display_shape": tuple(self.current_plane().shape[:2]),
        }

    def current_metric_rois(self) -> dict[str, tuple[int, int, int, int]]:
        """Return only the explicit ROI drawn on the currently visible axial plane."""

        if (
            self._metric_roi is None
            or self._metric_roi_plane_key != self._current_plane_key()
            or min(self._metric_roi[2:]) < 3
        ):
            return {}
        return {"viewer_roi": self._metric_roi}

    def rendered_preview_png(self) -> bytes:
        plane = self.current_plane()
        qimage = array_to_qimage(
            plane,
            grayscale_mapping=self._gray_mapping,
            color_mapping=self._color_mapping,
        )
        # Pillow emits a minimal PNG without DICOM/NIfTI/EXIF metadata.
        try:
            from PIL import Image
        except ImportError as exc:
            raise ViewerError("Pillow is required to create a privacy-filtered preview.") from exc
        pointer = qimage.bits()
        pointer.setsize(qimage.sizeInBytes())
        channels = (
            1
            if qimage.format() == qimage.Format_Grayscale8
            else 4
            if qimage.hasAlphaChannel()
            else 3
        )
        array = np.frombuffer(pointer, np.uint8).reshape(qimage.height(), qimage.bytesPerLine())
        array = array[:, : qimage.width() * channels].reshape(
            qimage.height(), qimage.width(), channels
        )
        if channels == 1:
            array = array[:, :, 0]
        output = BytesIO()
        Image.fromarray(np.array(array, copy=True)).save(output, format="PNG")
        return output.getvalue()

    def _fit_views(self) -> None:
        for view in (self.axial_view, self.coronal_view, self.sagittal_view, self.aux_view):
            if view.isVisible():
                view.fit_image()

    def _show_histogram(self) -> None:
        if self.image is not None:
            HistogramDialog(
                self.current_plane(),
                "Decoded-value histogram",
                self,
                language=self._language,
            ).exec_()

    def _measurement_mode_changed(self, index: int) -> None:
        mode = ("none", "distance", "area", "annotation")[index]
        for view in (self.axial_view, self.coronal_view, self.sagittal_view):
            view.set_measurement_mode(mode)  # type: ignore[arg-type]

    def _clear_marks(self) -> None:
        self._paired_plane_key = None
        self._paired_mask = None
        self._paired_annotations = None
        self._paired_mask_name = ""
        self._paired_annotation_name = ""
        self._metric_roi_plane_key = None
        self._metric_roi = None
        for view in (self.axial_view, self.coronal_view, self.sagittal_view, self.aux_view):
            view.clear_measurements()
            view.clear_overlays()
        self.pairing_status.setText(self._tr("No local mask or annotation is paired."))

    def _pixel_hovered(self, x: int, y: int, value: object) -> None:
        semantics = self.image.intensity_semantics.value if self.image is not None else "value"
        sender = self.sender()
        view_name = next(
            (name for name, view in self._views_by_name.items() if view is sender),
            self._active_view_name,
        )
        self._hover_view_name = view_name
        self._hover_world_ras = None
        if isinstance(self.image, ImageVolume):
            voxel: tuple[int, int, int] | None = None
            if view_name == "axial":
                voxel = (x, y, self.frame_slider.value())
            elif view_name == "coronal":
                voxel = (x, self.coronal_slider.value(), y)
            elif view_name == "sagittal":
                voxel = (self.sagittal_slider.value(), x, y)
            if voxel is not None:
                world = self.image.voxel_xyz_to_world_ras(np.asarray(voxel, dtype=float))
                self._hover_world_ras = tuple(float(item) for item in world)
        self._update_view_context()
        coordinate_text = ""
        if self._hover_world_ras is not None:
            r, a, s = self._hover_world_ras
            coordinate_text = f", RAS+ mm=({r:.2f}, {a:.2f}, {s:.2f})"
        self._set_status(
            f"{view_name}: x={x}, y={y}, {semantics}={np.asarray(value).tolist()}{coordinate_text}",
            translatable=False,
        )

    def _measurement_completed(self, mode: str, payload: object) -> None:
        mode_label = {
            "none": "Pan / zoom",
            "distance": "Distance",
            "area": "Area / ROI",
            "roi": "Area / ROI",
            "annotation": "Annotation",
        }.get(mode, mode)
        self._measurement_status = (mode_label, payload)
        self._refresh_status()
        if isinstance(payload, dict):
            self._last_measurement_metrics = {
                str(key): float(value)
                for key, value in payload.items()
                if isinstance(value, (int, float))
                and not isinstance(value, bool)
                and np.isfinite(float(value))
            }
            rectangle = payload.get("rectangle")
            if (
                mode == "area"
                and (
                    self.sender() is None
                    or self.sender() is self._views_by_name.get(self._active_view_name)
                )
                and isinstance(rectangle, (tuple, list))
                and len(rectangle) == 4
            ):
                self._metric_roi_plane_key = self._current_plane_key()
                self._metric_roi = tuple(int(value) for value in rectangle)

    def _toggle_playback(self) -> None:
        if self.play_timer.isActive():
            self.play_timer.stop()
            self.play_button.setText(self._tr("Play pages"))
        else:
            self.play_timer.start()
            self.play_button.setText(self._tr("Pause"))

    def _next_frame(self) -> None:
        maximum = self.frame_slider.maximum()
        self.frame_slider.setValue((self.frame_slider.value() + 1) % (maximum + 1))

    def _set_raster_spacing(self) -> None:
        if not isinstance(self.image, RasterImage2D):
            return
        dialog = QDialog(self)
        dialog.setWindowTitle(self._tr("User-provided pixel spacing"))
        layout = QFormLayout(dialog)
        x_spin = QDoubleSpinBox()
        y_spin = QDoubleSpinBox()
        for spin in (x_spin, y_spin):
            spin.setRange(0.000001, 10000.0)
            spin.setDecimals(6)
            spin.setValue(1.0)
        layout.addRow(self._tr("X spacing (mm)"), x_spin)
        layout.addRow(self._tr("Y spacing (mm)"), y_spin)
        warning = QLabel(
            self._tr(
                "This value is user-provided. OpenMedVisionX never infers medical spacing from DPI."
            )
        )
        warning.setWordWrap(True)
        layout.addRow(warning)
        buttons = QHBoxLayout()
        accept = QPushButton(self._tr("Apply"))
        cancel = QPushButton(self._tr("Cancel"))
        accept.clicked.connect(dialog.accept)
        cancel.clicked.connect(dialog.reject)
        buttons.addWidget(accept)
        buttons.addWidget(cancel)
        layout.addRow(buttons)
        if dialog.exec_() == QDialog.Accepted:
            self.image = self.image.with_user_spacing(x_spin.value(), y_spin.value())
            if self.study is not None:
                self.study = replace(self.study, display_image=self.image)
                self.service.state.replace(self.study)
            self.spacing_button.setVisible(False)
            self._update_views()
            self._update_info()
            self.imageChanged.emit(self.image)

    def _cancel_local_operation(
        self,
        _checked: bool = False,
        *,
        clear_handle: bool = False,
    ) -> None:
        task = self._active_local_task
        if task is not None and not task.done:
            task.cancel()
        if clear_handle:
            self._active_local_task = None
        self.cancel_local_button.setEnabled(False)

    def _set_local_controls_available(self) -> None:
        has_image = self.image is not None
        can_overlay = bool(
            has_image and self.image is not None and Capability.OVERLAY in self.image.capabilities
        )
        self.pair_mask_button.setEnabled(can_overlay)
        self.pair_annotation_button.setEnabled(can_overlay)
        self.import_dicom_annotation_button.setEnabled(
            bool(
                has_image
                and self.study is not None
                and self.study.reference_series is not None
                and self.study.reference_series.series_instance_uid is not None
            )
        )
        self.import_label_map_button.setEnabled(
            bool(has_image and self.study is not None and self.study.reference_series is not None)
        )
        self.export_button.setEnabled(
            bool(
                has_image
                and self.image is not None
                and Capability.VIEW_2D in self.image.capabilities
            )
        )
        self.save_experiment_button.setEnabled(has_image)
        self.cancel_local_button.setEnabled(False)

    def _start_local_operation(
        self,
        operation: Callable[[Any], Any],
        success: Callable[[Any], None],
        failure: Callable[[BaseException], bool] | None = None,
    ) -> None:
        self._cancel_local_operation(clear_handle=True)
        task = self.local_runner.submit(operation)
        self._active_local_task = task
        for button in (
            self.pair_mask_button,
            self.pair_annotation_button,
            self.import_dicom_annotation_button,
            self.import_label_map_button,
            self.export_button,
            self.save_experiment_button,
        ):
            button.setEnabled(False)
        self.cancel_local_button.setEnabled(True)
        self.progress.setVisible(True)
        self.progress.setValue(0)
        self.watcher.watch(
            task,
            success=lambda result, handle=task: self._local_operation_succeeded(
                handle, success, result
            ),
            error=lambda error, handle=task: self._local_operation_failed(handle, error, failure),
            progress=lambda fraction, message, handle=task: self._local_progress(
                handle, fraction, message
            ),
        )

    def _local_operation_succeeded(
        self,
        task: BackgroundTask[Any],
        callback: Callable[[Any], None],
        result: Any,
    ) -> None:
        if task is not self._active_local_task:
            return
        self._active_local_task = None
        self.progress.setVisible(False)
        self._set_local_controls_available()
        callback(result)

    def _local_operation_failed(
        self,
        task: BackgroundTask[Any],
        error: BaseException,
        failure: Callable[[BaseException], bool] | None = None,
    ) -> None:
        if task is not self._active_local_task:
            return
        self._active_local_task = None
        self.progress.setVisible(False)
        self._set_local_controls_available()
        if failure is not None and failure(error):
            return
        if isinstance(error, OperationCancelled):
            self._set_status("Local operation cancelled safely; no pending result was applied.")
            return
        self._set_status(f"Local operation failed safely: {error}")
        QMessageBox.warning(self, self._tr("Local operation failed"), str(error))

    def _local_progress(
        self,
        task: BackgroundTask[Any],
        fraction: float,
        message: str,
    ) -> None:
        if task is self._active_local_task:
            self.progress.setValue(int(round(fraction * 100)))
            self._set_status(message)

    def _select_label_map(self) -> None:
        study = self.study
        reference_series = None if study is None else study.reference_series
        if study is None or reference_series is None:
            QMessageBox.information(
                self,
                self._tr("Reference image required"),
                self._tr(
                    "Open the intended DICOM or NIfTI image series before importing a label "
                    "map. OpenMedVisionX will not guess a reference grid."
                ),
            )
            return
        path, _ = QFileDialog.getOpenFileName(
            self,
            self._tr("Select one clinical label map"),
            "",
            self._tr("Lossless label maps (*.png *.tif *.tiff *.nii *.nii.gz);;All files (*)"),
        )
        if path:
            self._begin_label_map_import(Path(path), study, volume_index=None)

    def _begin_label_map_import(
        self,
        path: Path,
        expected_study: LoadedStudy,
        *,
        volume_index: int | None,
    ) -> None:
        reference_series = expected_study.reference_series
        if self.study is not expected_study or reference_series is None:
            self._set_status("The label-map request was discarded because the study changed.")
            return

        def operation(context: Any) -> SegmentationLayer:
            context.report_progress(0.08, message="Reading the selected clinical label map")
            layer = import_label_map(
                path,
                reference_series=reference_series,
                volume_index=volume_index,
                cancel=lambda: context.cancelled,
            )
            context.raise_if_cancelled()
            context.report_progress(1.0, message="Label-map validation complete")
            return layer

        def failure(error: BaseException) -> bool:
            if not isinstance(error, NiftiVolumeSelectionRequiredError):
                return False
            QTimer.singleShot(
                0,
                lambda: self._select_label_map_volume(
                    path,
                    expected_study,
                    error.volume_count,
                ),
            )
            return True

        self._start_local_operation(
            operation,
            lambda layer: self._label_map_ready(layer, expected_study),
            failure,
        )

    def _select_label_map_volume(
        self,
        path: Path,
        expected_study: LoadedStudy,
        volume_count: int,
    ) -> None:
        if self.study is not expected_study:
            self._set_status("The label-map request was discarded because the study changed.")
            return
        selected, accepted = QInputDialog.getInt(
            self,
            self._tr("Choose a label-map volume"),
            self._tr("Time point / channel (1 to {count})").format(count=volume_count),
            1,
            1,
            max(1, volume_count),
            1,
        )
        if accepted:
            self._begin_label_map_import(
                path,
                expected_study,
                volume_index=selected - 1,
            )
        else:
            self._set_status("Label-map volume selection cancelled; the study is unchanged.")

    def _label_map_ready(
        self,
        layer: SegmentationLayer,
        expected_study: LoadedStudy,
    ) -> None:
        if self.study is not expected_study or not isinstance(self.image, ImageVolume):
            self._set_status("The label-map result was discarded because the study changed.")
            return
        domain_study = expected_study.domain_study
        reference_series = expected_study.reference_series
        if domain_study is None or reference_series is None:
            raise ViewerError("The active study no longer exposes a reference series.")
        report = layer.reference_report(reference_series)
        if report.status.value not in {"matched", "resampled", "requires-resampling"}:
            raise ViewerError("The label map does not match the explicitly selected series.")

        target_geometry = SpatialGeometry.from_volume(self.image)
        if layer.geometry.matches(target_geometry):
            self._commit_label_map_layers(domain_study, (layer,))
            return

        decision = self._confirm_segmentation_resampling(
            layer,
            target_geometry,
            InterpolationMode.NEAREST,
        )
        if decision == "cancel":
            self._set_status("Label-map import cancelled before any layer changed.")
            return
        original = layer.with_presentation(visible=False)
        if decision == "keep":
            self._commit_label_map_layers(domain_study, (original,))
            return

        def operation(context: Any) -> SegmentationLayer:
            context.report_progress(
                0.1,
                message="Resampling a confirmed label-map preview with nearest neighbour",
            )
            derived = resample_segmentation_layer(
                original,
                target_geometry,
                layer_id=f"{original.layer_id}-display-{target_geometry.fingerprint[:8]}",
                interpolation=InterpolationMode.NEAREST,
                user_confirmed=True,
            ).with_presentation(visible=True)
            context.raise_if_cancelled()
            context.report_progress(1.0, message="Confirmed label-map preview is ready")
            return derived

        self._start_local_operation(
            operation,
            lambda derived: self._commit_label_map_layers(
                domain_study,
                (original, derived),
            ),
        )

    def _commit_label_map_layers(
        self,
        domain_study: ImageStudy,
        layers: tuple[SegmentationLayer, ...],
    ) -> None:
        study = self.study
        if study is None or study.domain_study is not domain_study:
            self._set_status("The label-map result was discarded because the study changed.")
            return
        reference_series = study.reference_series
        if reference_series is None:
            raise ViewerError("The active reference series is unavailable.")
        updated = domain_study
        for layer in layers:
            updated = updated.with_layer(reference_series.series_id, layer)
        active_layer = next(
            (item for item in reversed(layers) if item.presentation.visible),
            layers[-1],
        )
        self._install_domain_study(updated, reference_series.series_id, active_layer.layer_id)
        resampled = sum(bool(item.transform_chain) for item in layers)
        self.pairing_status.setText(
            self._tr(
                "Imported {count} label-map layer(s); {resampled} confirmed display "
                "preview(s). The source file stayed unchanged and local."
            ).format(count=len(layers), resampled=resampled)
        )
        self.statusChanged.emit(self.pairing_status.text())

    def _select_dicom_annotation(self) -> None:
        study = self.study
        reference_series = None if study is None else study.reference_series
        if study is None or reference_series is None:
            QMessageBox.information(
                self,
                self._tr("DICOM reference required"),
                self._tr(
                    "Open the referenced DICOM image series before importing DICOM SEG or "
                    "RTSTRUCT. OpenMedVisionX will not guess a reference series."
                ),
            )
            return
        path, _ = QFileDialog.getOpenFileName(
            self,
            self._tr("Select one DICOM SEG or RTSTRUCT object"),
            "",
            self._tr("DICOM annotation (*.dcm *.dicom);;All files (*)"),
        )
        if not path:
            return

        def operation(context: Any) -> DicomAnnotationImport:
            context.report_progress(0.08, message="Reading the selected DICOM annotation")
            result = import_dicom_annotation(path, reference_series=reference_series)
            context.raise_if_cancelled()
            context.report_progress(1.0, message="DICOM annotation validation complete")
            return result

        self._start_local_operation(
            operation,
            lambda result, expected=study: self._dicom_annotation_ready(result, expected),
        )

    def _dicom_annotation_ready(
        self,
        imported: DicomAnnotationImport,
        expected_study: LoadedStudy,
    ) -> None:
        if self.study is not expected_study or self.image is None:
            self._set_status(
                "The annotation result was discarded because the active study changed."
            )
            return
        domain_study = expected_study.domain_study
        reference_series = expected_study.reference_series
        if domain_study is None or reference_series is None:
            raise ViewerError("The active DICOM study no longer exposes a reference series.")
        target_geometry = (
            SpatialGeometry.from_volume(self.image) if isinstance(self.image, ImageVolume) else None
        )
        prepared: list[SegmentationLayer | ContourLayer] = []
        resample_requests: list[tuple[SegmentationLayer, InterpolationMode]] = []
        for layer in imported.layers:
            report = layer.reference_report(reference_series)
            if not report.overlay_allowed and report.status.value not in {
                "requires-resampling",
            }:
                raise ViewerError(
                    "The imported annotation does not match the active reference series."
                )
            if (
                isinstance(layer, SegmentationLayer)
                and target_geometry is not None
                and not layer.geometry.matches(target_geometry)
            ):
                interpolation = (
                    InterpolationMode.NEAREST
                    if layer.value_type
                    in {
                        SegmentationValueType.BINARY,
                        SegmentationValueType.DISCRETE,
                    }
                    else InterpolationMode.LINEAR
                )
                decision = self._confirm_segmentation_resampling(
                    layer,
                    target_geometry,
                    interpolation,
                )
                if decision == "cancel":
                    self._set_status("DICOM annotation import cancelled before any layer changed.")
                    return
                hidden_original = layer.with_presentation(visible=False)
                prepared.append(hidden_original)
                if decision == "resample":
                    resample_requests.append((hidden_original, interpolation))
                continue
            prepared.append(layer)

        if not resample_requests:
            self._commit_dicom_annotation_layers(domain_study, tuple(prepared), imported)
            return

        def resample_operation(context: Any) -> tuple[SegmentationLayer, ...]:
            results: list[SegmentationLayer] = []
            total = len(resample_requests)
            for index, (layer, interpolation) in enumerate(resample_requests):
                context.raise_if_cancelled()
                context.report_progress(
                    index / total,
                    message="Resampling a confirmed annotation preview",
                    current=index,
                    total=total,
                )
                assert target_geometry is not None
                derived = resample_segmentation_layer(
                    layer,
                    target_geometry,
                    layer_id=f"{layer.layer_id}-display-{target_geometry.fingerprint[:8]}",
                    interpolation=interpolation,
                    user_confirmed=True,
                ).with_presentation(visible=True)
                results.append(derived)
            context.report_progress(
                1.0,
                message="Confirmed annotation previews are ready",
                current=total,
                total=total,
            )
            return tuple(results)

        self._start_local_operation(
            resample_operation,
            lambda derived: self._commit_dicom_annotation_layers(
                domain_study,
                (*prepared, *derived),
                imported,
            ),
        )

    def _confirm_segmentation_resampling(
        self,
        layer: SegmentationLayer,
        target: SpatialGeometry,
        interpolation: InterpolationMode,
    ) -> str:
        difference = layer.geometry.compare_to(target)
        dialog = QMessageBox(self)
        dialog.setIcon(QMessageBox.Warning)
        dialog.setWindowTitle(self._tr("Geometry differs from the display grid"))
        dialog.setText(
            self._tr(
                "The imported layer is valid in its original geometry, but it cannot be "
                "overlaid on the current display grid without a recorded resampling step."
            )
        )
        dialog.setInformativeText(
            self._tr(
                "Choose Create preview to keep the original layer and add a derived display "
                "layer. Choose Keep original to import it hidden without resampling."
            )
        )
        dialog.setDetailedText(
            "\n".join(
                (
                    self._tr("Layer: {name}").format(name=layer.name),
                    self._tr("Source shape ZYX: {shape}").format(shape=layer.geometry.shape_zyx),
                    self._tr("Display shape ZYX: {shape}").format(shape=target.shape_zyx),
                    self._tr("Differences: {components}").format(
                        components=", ".join(
                            self._tr(component) for component in difference.mismatched_components
                        )
                    ),
                    self._tr("Proposed interpolation: {mode}").format(
                        mode=self._tr(interpolation.value)
                    ),
                    self._tr("Outside value: {value}").format(value=0),
                    self._tr("The imported source layer remains immutable."),
                )
            )
        )
        preview_button = dialog.addButton(
            self._tr("Create resampled preview"), QMessageBox.AcceptRole
        )
        keep_button = dialog.addButton(self._tr("Keep original hidden"), QMessageBox.ActionRole)
        cancel_button = dialog.addButton(self._tr("Cancel"), QMessageBox.RejectRole)
        cancel_button.setDefault(True)
        dialog.setEscapeButton(cancel_button)
        dialog.exec_()
        clicked = dialog.clickedButton()
        if clicked is preview_button:
            return "resample"
        if clicked is keep_button:
            return "keep"
        return "cancel"

    def _commit_dicom_annotation_layers(
        self,
        domain_study: ImageStudy,
        layers: tuple[SegmentationLayer | ContourLayer, ...],
        imported: DicomAnnotationImport,
    ) -> None:
        study = self.study
        if study is None or study.domain_study is not domain_study:
            self._set_status(
                "The annotation result was discarded because the active study changed."
            )
            return
        reference_series = study.reference_series
        if reference_series is None:
            raise ViewerError("The active reference series is unavailable.")
        updated = domain_study
        for layer in layers:
            updated = updated.with_layer(reference_series.series_id, layer)
        visible = tuple(layer for layer in layers if layer.presentation.visible)
        active_layer_id = (visible or layers)[-1].layer_id
        self._install_domain_study(
            updated,
            reference_series.series_id,
            active_layer_id,
        )
        layer_count = len(layers)
        resampled_count = sum(bool(layer.transform_chain) for layer in layers)
        self.pairing_status.setText(
            self._tr(
                "Imported {count} {kind} layer(s); {resampled} confirmed display "
                "preview(s). No source file was changed or uploaded."
            ).format(
                count=layer_count,
                kind=imported.kind.value,
                resampled=resampled_count,
            )
        )
        self.statusChanged.emit(self.pairing_status.text())

    def _install_domain_study(
        self,
        domain_study: ImageStudy,
        series_id: str,
        active_layer_id: str,
    ) -> None:
        """Atomically install one immutable clinical-layer revision in UI and service state."""

        study = self.study
        if study is None:
            raise ViewerError("No loaded study is available for a layer update.")
        updated_loaded = replace(study, domain_study=domain_study)
        self.study = updated_loaded
        self.service.state.replace(updated_loaded)
        self._active_layer_id = active_layer_id
        self.study_layers.set_study(domain_study)
        self.study_layers.set_active_layer(series_id, active_layer_id)
        self._render_paired_overlays()

    def _active_layer_changed(self, _series_id: str, layer_id: str) -> None:
        self._active_layer_id = layer_id
        self._render_paired_overlays()

    def _apply_layer_presentation(
        self,
        series_id: str,
        layer_id: str,
        **changes: object,
    ) -> None:
        study = self.study
        domain_study = None if study is None else study.domain_study
        if study is None or domain_study is None:
            self.study_layers.set_study(domain_study)
            return
        try:
            series = domain_study.find_series(series_id)
            updated_series = series.with_layer_presentation(layer_id, **changes)  # type: ignore[arg-type]
            updated = domain_study.replace_series(updated_series)
            self._install_domain_study(updated, series_id, layer_id)
        except BaseException as exc:
            self.study_layers.set_study(domain_study)
            self.study_layers.set_active_layer(series_id, layer_id)
            QMessageBox.warning(self, self._tr("Layer update rejected"), str(exc))

    def _layer_visibility_requested(
        self,
        series_id: str,
        layer_id: str,
        visible: bool,
    ) -> None:
        self._apply_layer_presentation(series_id, layer_id, visible=visible)

    def _layer_lock_requested(
        self,
        series_id: str,
        layer_id: str,
        locked: bool,
    ) -> None:
        self._apply_layer_presentation(series_id, layer_id, locked=locked)

    def _layer_opacity_requested(
        self,
        series_id: str,
        layer_id: str,
        opacity: float,
    ) -> None:
        self._apply_layer_presentation(series_id, layer_id, opacity=opacity)

    def _label_visibility_requested(
        self,
        series_id: str,
        layer_id: str,
        label_value: int,
        visible: bool,
    ) -> None:
        study = self.study
        domain_study = None if study is None else study.domain_study
        if study is None or domain_study is None:
            self.study_layers.set_study(domain_study)
            return
        try:
            series = domain_study.find_series(series_id)
            updated_series = series.with_segmentation_label_visibility(
                layer_id,
                label_value,
                visible,
            )
            updated = domain_study.replace_series(updated_series)
            self._hidden_label_values.pop(layer_id, None)
            self._install_domain_study(updated, series_id, layer_id)
        except BaseException as exc:
            self.study_layers.set_study(domain_study)
            self.study_layers.set_active_layer(series_id, layer_id)
            QMessageBox.warning(self, self._tr("Label update rejected"), str(exc))

    def _select_local_mask(self) -> None:
        if self.image is None:
            return
        path, _ = QFileDialog.getOpenFileName(
            self,
            self._tr("Select one local mask for the current plane"),
            "",
            self._tr("Lossless masks (*.png *.tif *.tiff)"),
        )
        if not path:
            return
        plane_key = self._current_plane_key()
        expected_shape = self.current_plane().shape[:2]

        def operation(context: Any) -> np.ndarray:
            return load_local_mask(
                path,
                expected_shape,
                cancel=lambda: context.cancelled,
                progress=lambda fraction, message: context.report_progress(
                    fraction, message=message
                ),
            )

        self._start_local_operation(
            operation,
            lambda mask: self._local_mask_ready(mask, plane_key, Path(path).name),
        )

    def _local_mask_ready(
        self,
        mask: np.ndarray,
        plane_key: tuple[str, int],
        filename: str,
    ) -> None:
        if self._paired_plane_key not in {None, plane_key}:
            self._paired_annotations = None
            self._paired_annotation_name = ""
        self._paired_plane_key = plane_key
        self._paired_mask = mask
        self._paired_mask_name = filename
        self._render_paired_overlays()
        self.statusChanged.emit(
            self._tr("Local mask paired explicitly; no files were scanned or uploaded.")
        )

    def _select_local_annotations(self) -> None:
        if self.image is None:
            return
        path, _ = QFileDialog.getOpenFileName(
            self,
            self._tr("Select one local annotation JSON for the current plane"),
            "",
            self._tr("OpenMedVisionX annotation (*.json)"),
        )
        if not path:
            return
        plane_key = self._current_plane_key()
        expected_shape = self.current_plane().shape[:2]

        def operation(context: Any) -> AnnotationOverlay:
            return load_local_annotations(
                path,
                expected_shape,
                cancel=lambda: context.cancelled,
                progress=lambda fraction, message: context.report_progress(
                    fraction, message=message
                ),
            )

        self._start_local_operation(
            operation,
            lambda overlay: self._local_annotations_ready(overlay, plane_key, Path(path).name),
        )

    def _local_annotations_ready(
        self,
        overlay: AnnotationOverlay,
        plane_key: tuple[str, int],
        filename: str,
    ) -> None:
        if self._paired_plane_key not in {None, plane_key}:
            self._paired_mask = None
            self._paired_mask_name = ""
        self._paired_plane_key = plane_key
        self._paired_annotations = overlay
        self._paired_annotation_name = filename
        self._render_paired_overlays()
        self.statusChanged.emit(
            self._tr("Local annotation paired explicitly; external references are not followed.")
        )

    def _suggested_output(self, filename: str) -> str:
        source = self._source_path
        if source is None:
            return filename
        directory = source if source.is_dir() else source.parent
        return str(directory / filename)

    def _export_rendered_plane(self) -> None:
        if self.image is None:
            return
        path, _ = QFileDialog.getSaveFileName(
            self,
            self._tr("Export rendered plane to a new local file"),
            self._suggested_output("openmedvisionx-rendered.png"),
            self._tr("PNG image (*.png)"),
        )
        if not path:
            return
        plane = self.current_plane()
        gray_mapping = self._gray_mapping
        color_mapping = self._color_mapping
        source_path = self._source_path

        def operation(context: Any) -> Path:
            return export_rendered_png(
                path,
                plane,
                source_path=source_path,
                grayscale_mapping=gray_mapping,
                color_mapping=color_mapping,
                cancel=lambda: context.cancelled,
                progress=lambda fraction, message: context.report_progress(
                    fraction, message=message
                ),
            )

        self._start_local_operation(operation, self._local_export_ready)

    def _local_export_ready(self, path: Path) -> None:
        self.statusChanged.emit(
            self._tr("Rendered PNG saved as new file: {name}").format(name=path.name)
        )

    def _save_experiment(self) -> None:
        if self.image is None:
            return
        path, _ = QFileDialog.getSaveFileName(
            self,
            self._tr("Save a pixel-free experiment record"),
            self._suggested_output("openmedvisionx-experiment.json"),
            self._tr("Experiment JSON (*.json)"),
        )
        if not path:
            return
        parameters: dict[str, Any] = {
            "plane_kind": self._current_plane_key()[0],
            "plane_index": self._current_plane_key()[1],
            "display_lower": (self._gray_mapping.lower if self._gray_mapping is not None else None),
            "display_upper": (self._gray_mapping.upper if self._gray_mapping is not None else None),
            "display_inverted": bool(self._gray_mapping and self._gray_mapping.invert),
            "local_mask_paired": self._paired_mask is not None,
            "local_annotation_paired": self._paired_annotations is not None,
        }
        record = create_experiment_record(
            "Image exploration",
            image=self.image,
            parameters=parameters,
            metrics=self._last_measurement_metrics,
        )

        def operation(context: Any) -> Path:
            return save_experiment_record(
                path,
                record,
                cancel=lambda: context.cancelled,
                progress=lambda fraction, message: context.report_progress(
                    fraction, message=message
                ),
            )

        self._start_local_operation(operation, self._experiment_saved)

    def _experiment_saved(self, path: Path) -> None:
        self.statusChanged.emit(
            self._tr("Pixel-free experiment parameters and numeric metrics saved: {name}").format(
                name=path.name
            )
        )

    def _update_info(self) -> None:
        if self.image is None:
            return
        image = self.image

        def field(label: str, value: object) -> str:
            return f"{self._tr(label)}: {value}"

        lines = [
            field("Type", type(image).__name__),
            field("Source", image.source_type.value),
            field("Shape", image.shape),
            field("Dtype", image.dtype),
            field("Semantics", image.intensity_semantics.value),
            field(
                "Capabilities",
                ", ".join(sorted(item.value for item in image.capabilities)),
            ),
        ]
        if isinstance(image, RasterImage2D):
            lines.extend(
                [
                    field("Color", f"{image.color_space.value} / {image.channel_order}"),
                    field("Bit depth", image.bit_depth),
                    field("Alpha", image.alpha_semantics.value),
                    field(
                        "Spacing",
                        image.pixel_spacing or self._tr("none (pixel units only)"),
                    ),
                    field(
                        "Spacing source",
                        image.spacing_source.value if image.spacing_source else self._tr("none"),
                    ),
                ]
            )
        if isinstance(image, ImageSequence2D):
            lines.extend(
                [
                    field("Pages/frames", image.frame_count),
                    field("Spatial volume semantics", self._tr("disabled")),
                ]
            )
        if isinstance(image, ImageVolume):
            lines.extend(
                [
                    field("Modality", image.modality),
                    field("Spacing XYZ (mm)", image.spacing),
                    field("Origin RAS+ (mm)", image.origin),
                    field("Direction", f"\n{image.direction}"),
                ]
            )
        lines.append(f"\n{self._tr('Runtime metadata (PHI-filtered)')}:")
        lines.extend(f"  {key}: {value}" for key, value in image.runtime_metadata.items())
        self.info.setPlainText("\n".join(lines))
        warnings: list[str] = []
        if image.runtime_metadata.get("access") in {"thumbnail", "thumbnail_sequence"}:
            warnings.append(
                self._tr(
                    "Large flat raster: displaying a bounded thumbnail with reversible source "
                    "coordinates; full-resolution pixels were not retained in the session."
                )
            )
        if image.runtime_metadata.get("lossy_compression"):
            warnings.append(
                self._tr(
                    "Lossy JPEG compression: artifacts are not original signal or model output."
                )
            )
        if isinstance(image, ImageSequence2D):
            warnings.append(
                self._tr(
                    "Page sequence only: 3-D tools and physical volume measurements are disabled."
                )
            )
        if isinstance(image, RasterImage2D) and image.pixel_spacing is not None:
            warnings.append(
                self._tr("Physical spacing is user-provided, not inferred from the file.")
            )
        self.warning_label.setText("\n".join(warnings))
        self.warning_label.setVisible(bool(warnings))

    def close(self) -> None:  # type: ignore[override]
        self.play_timer.stop()
        self.watcher.stop()
        self.service.close()
        self.local_runner.shutdown(wait=False, cancel_pending=True)
        super().close()


class ReconstructionPage(_BilingualPage, QWidget):
    statusChanged = pyqtSignal(str)

    def __init__(
        self,
        source: Callable[[], np.ndarray],
        parent: QWidget | None = None,
        *,
        roi_source: Callable[[], dict[str, tuple[int, int, int, int]]] | None = None,
    ) -> None:
        super().__init__(parent)
        self.source = source
        self.roi_source = roi_source or (lambda: {})
        self.service = ReconstructionService()
        self.output_runner = TaskRunner(
            max_workers=1,
            thread_name_prefix="openmedvisionx-reconstruction-output",
        )
        self.watcher = TaskWatcher(self)
        self.reference: np.ndarray | None = None
        self.sinogram_result: Any = None
        self.reconstruction_result: Any = None
        self._active_task: BackgroundTask[Any] | None = None
        self._active_output_task: BackgroundTask[Any] | None = None
        self._latest_metrics: dict[str, float | None] = {}
        self._reference_rois: dict[str, tuple[int, int, int, int]] = {}
        self._metric_maps: dict[str, np.ndarray] = {}
        self._process_images: dict[str, np.ndarray] = {}
        self._evaluation_range: tuple[float, float] | None = None
        self._evaluation_unit: str | None = None
        self._source_available = True
        self._auto_reconstruct_after_sinogram = False
        self._build_ui()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 8, 0, 0)
        root.setSpacing(10)
        explanation = QLabel(
            "Default: a synthetic, non-negative attenuation phantom (illustrative mm⁻¹) with "
            "zero background. "
            "It teaches image-domain parallel-beam physics; it is not scanner raw data or "
            "a conversion from clinical HU."
        )
        explanation.setWordWrap(True)
        explanation.setObjectName("infoBanner")
        explanation.setToolTip(
            "p(s, θ + 180°) = p(−s, θ). OpenMedVisionX folds and averages the redundant half."
        )
        root.addWidget(explanation)

        self.reconstruction_splitter = QSplitter(Qt.Horizontal)
        setup_scroll = QScrollArea()
        setup_scroll.setWidgetResizable(True)
        setup_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        # Preserve the one-click CTA and scientific provenance at the supported 1024 px width.
        setup_scroll.setMinimumWidth(320)
        setup_scroll.setAccessibleName("CT Lab experiment setup")
        setup_content = QWidget()
        setup_layout = QVBoxLayout(setup_content)
        setup_layout.setContentsMargins(0, 0, 0, 0)
        setup_layout.setSpacing(10)

        self.source_status = QLabel(
            "Ready: run the synthetic phantom experiment without opening patient data."
        )
        self.source_status.setObjectName("infoBanner")
        self.source_status.setWordWrap(True)
        self.source_status.setAccessibleName("CT Lab input status")
        setup_layout.addWidget(self.source_status)

        self.configuration_card = QGroupBox("Reconstruction setup")
        controls = QFormLayout(self.configuration_card)
        controls.setRowWrapPolicy(QFormLayout.WrapAllRows)
        controls.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
        controls.setHorizontalSpacing(10)
        controls.setVerticalSpacing(9)
        self.source_mode = QComboBox()
        self.source_mode.addItem("Synthetic phantom (recommended)", "phantom")
        self.source_mode.addItem("Active image (advanced)", "image")
        self.source_mode.currentIndexChanged.connect(self._source_mode_changed)
        self.phantom_size = QSpinBox()
        self.phantom_size.setRange(64, 384)
        self.phantom_size.setSingleStep(32)
        self.phantom_size.setValue(192)
        self.phantom_size.setSuffix(" px")
        self.phantom_size.valueChanged.connect(self._scan_parameters_changed)
        self.source_provenance = QLabel(
            "Generated at runtime · illustrative μ in mm⁻¹ · fixed 0–0.03 mm⁻¹ evaluation "
            "range · circular support "
            "outside value = 0 · no medical file required."
        )
        self.source_provenance.setObjectName("mutedText")
        self.source_provenance.setWordWrap(True)
        self.angle_range = QComboBox()
        self.angle_range.addItems(["180", "360"])
        self.angle_range.setProperty(_I18N_SKIP_ITEMS_PROPERTY, True)
        self.projections = QSpinBox()
        self.projections.setRange(8, 1440)
        self.projections.setValue(180)
        self.circle = QCheckBox("Circular support")
        self.circle.setChecked(True)
        self.algorithm = QComboBox()
        self.algorithm.addItems(["FBP", "BP", "DFR", "SART"])
        self.algorithm.setProperty(_I18N_SKIP_ITEMS_PROPERTY, True)

        self.interpolation = QComboBox()
        self.interpolation.addItems(["nearest", "linear", "cubic"])
        self.interpolation.setProperty(_I18N_SKIP_ITEMS_PROPERTY, True)
        self.interpolation.setCurrentText("linear")
        self.filter = QComboBox()
        self.filter.addItems(["ramp", "shepp-logan", "cosine", "hamming", "hann"])
        self.filter.setProperty(_I18N_SKIP_ITEMS_PROPERTY, True)
        self.iterations = QSpinBox()
        self.iterations.setRange(1, 100)
        self.iterations.setValue(5)
        self.relaxation = QDoubleSpinBox()
        self.relaxation.setRange(0.01, 1.0)
        self.relaxation.setSingleStep(0.05)
        self.relaxation.setValue(0.15)

        self.algorithm_options = QStackedWidget()
        fbp_options = QWidget()
        fbp_layout = QFormLayout(fbp_options)
        fbp_layout.setContentsMargins(0, 0, 0, 0)
        fbp_layout.addRow("FBP filter", self.filter)
        bp_options = QWidget()
        bp_layout = QVBoxLayout(bp_options)
        bp_layout.setContentsMargins(0, 0, 0, 0)
        self.bp_note = QLabel("No additional parameters.")
        self.bp_note.setObjectName("mutedText")
        bp_layout.addWidget(self.bp_note)
        dfr_options = QWidget()
        dfr_layout = QFormLayout(dfr_options)
        dfr_layout.setContentsMargins(0, 0, 0, 0)
        dfr_layout.addRow("DFR interpolation", self.interpolation)
        sart_options = QWidget()
        sart_layout = QFormLayout(sart_options)
        sart_layout.setContentsMargins(0, 0, 0, 0)
        sart_layout.addRow("SART iterations", self.iterations)
        sart_layout.addRow("Relaxation", self.relaxation)
        for page in (fbp_options, bp_options, dfr_options, sart_options):
            self.algorithm_options.addWidget(page)

        self.range_control_label = QLabel("Range °")
        self.projection_control_label = QLabel("Angles")
        self.algorithm_control_label = QLabel("Algorithm")
        _add_accessible_form_row(controls, "Input", self.source_mode)
        _add_accessible_form_row(
            controls,
            "Phantom size",
            self.phantom_size,
            description="Square runtime phantom width and height in pixels.",
        )
        controls.addRow(self.source_provenance)
        self.range_control_label.setBuddy(self.angle_range)
        self.projection_control_label.setBuddy(self.projections)
        self.algorithm_control_label.setBuddy(self.algorithm)
        self.angle_range.setAccessibleName("Projection angular range in degrees")
        self.projections.setAccessibleName("Projection angle count")
        self.algorithm.setAccessibleName("Reconstruction algorithm")
        controls.addRow(self.range_control_label, self.angle_range)
        controls.addRow(self.projection_control_label, self.projections)
        controls.addRow(self.circle)
        controls.addRow(self.algorithm_control_label, self.algorithm)
        controls.addRow(self.algorithm_options)

        self.generate_button = QPushButton("Run first phantom experiment")
        self.generate_button.setObjectName("primary")
        self.generate_button.clicked.connect(self._run_selected_source)
        self.generate_button.setAccessibleDescription(
            "Generate the selected image-domain projection and reconstruct it."
        )
        self.reconstruct_button = QPushButton("2. Reconstruct")
        self.reconstruct_button.clicked.connect(self._reconstruct)
        self.reconstruct_button.setEnabled(False)
        self.reconstruct_button.setAccessibleDescription("Generate a sinogram first.")
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.clicked.connect(self._cancel_all)
        self.cancel_button.setEnabled(False)
        action_row = QVBoxLayout()
        action_row.setSpacing(8)
        action_row.addWidget(self.reconstruct_button)
        action_row.addWidget(self.cancel_button)
        controls.addRow(action_row)
        self.action_status = QLabel()
        self.action_status.setObjectName("actionStatus")
        self.action_status.setWordWrap(True)
        self.action_status.setAccessibleName("CT Lab next step")
        setup_layout.addWidget(self.configuration_card)
        setup_layout.insertWidget(1, self.generate_button)
        setup_layout.insertWidget(2, self.action_status)

        self.angle_range.currentTextChanged.connect(self._scan_parameters_changed)
        self.projections.valueChanged.connect(self._scan_parameters_changed)
        self.circle.stateChanged.connect(self._scan_parameters_changed)
        self.algorithm.currentTextChanged.connect(self._update_algorithm_options)
        self.algorithm.currentTextChanged.connect(self._reconstruction_parameters_changed)
        self.interpolation.currentTextChanged.connect(self._reconstruction_parameters_changed)
        self.filter.currentTextChanged.connect(self._reconstruction_parameters_changed)
        self.iterations.valueChanged.connect(self._reconstruction_parameters_changed)
        self.relaxation.valueChanged.connect(self._reconstruction_parameters_changed)
        self._update_algorithm_options(self.algorithm.currentText())

        self.progress = QProgressBar()
        self.progress.setVisible(False)
        setup_layout.addWidget(self.progress)
        setup_layout.addStretch(1)
        setup_scroll.setWidget(setup_content)
        self.reconstruction_splitter.addWidget(setup_scroll)

        results = QWidget()
        results_layout = QVBoxLayout(results)
        results_layout.setContentsMargins(0, 0, 0, 0)
        results_layout.setSpacing(10)
        grid = QGridLayout()
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setSpacing(8)
        self.input_view = ImageView("Input")
        self.sinogram_view = ImageView("Sinogram")
        self.result_view = ImageView("Reconstruction")
        self.error_view = ImageView("Absolute error heatmap")
        self.input_view.set_empty_message("Run the synthetic phantom experiment to begin.")
        self.sinogram_view.set_empty_message("The simulated line integrals will appear here.")
        self.result_view.set_empty_message("Reconstruction appears after projection succeeds.")
        self.error_view.set_empty_message(
            "Shared-range error feedback appears after reconstruction."
        )
        grid.addWidget(self.input_view, 0, 0)
        grid.addWidget(self.sinogram_view, 0, 1)
        grid.addWidget(self.result_view, 1, 0)
        grid.addWidget(self.error_view, 1, 1)
        results_layout.addLayout(grid, 1)
        bottom = QGridLayout()
        self.intermediate_combo = QComboBox()
        self.intermediate_combo.setProperty(_I18N_SKIP_ITEMS_PROPERTY, True)
        self.intermediate_combo.setAccessibleName("Intermediate reconstruction process")
        self.intermediate_combo.setMinimumWidth(200)
        self.intermediate_combo.setSizePolicy(QSizePolicy.MinimumExpanding, QSizePolicy.Fixed)
        self.intermediate_combo.currentIndexChanged.connect(self._show_intermediate)
        self.metrics_label = QLabel("Metrics use one shared intensity range.")
        self.metrics_label.setWordWrap(True)
        bottom.addWidget(QLabel("Intermediate process:"), 0, 0)
        bottom.addWidget(self.intermediate_combo, 1, 0)
        bottom.addWidget(self.metrics_label, 0, 1, 2, 1)
        bottom.setColumnStretch(1, 1)
        self.export_result_button = QPushButton("Export PNG…")
        self.export_result_button.clicked.connect(self._export_result)
        self.export_result_button.setEnabled(False)
        self.save_experiment_button = QPushButton("Save record…")
        self.save_experiment_button.clicked.connect(self._save_experiment)
        self.save_experiment_button.setEnabled(False)
        output_actions = QHBoxLayout()
        output_actions.addStretch(1)
        output_actions.addWidget(self.export_result_button)
        output_actions.addWidget(self.save_experiment_button)
        bottom.addLayout(output_actions, 2, 0, 1, 2)
        results_layout.addLayout(bottom)
        self.reconstruction_splitter.addWidget(results)
        self.reconstruction_splitter.setStretchFactor(0, 0)
        self.reconstruction_splitter.setStretchFactor(1, 1)
        self.reconstruction_splitter.setSizes([300, 980])
        self.reconstruction_splitter.setChildrenCollapsible(False)
        root.addWidget(self.reconstruction_splitter, 1)
        tab_order = (
            self.source_mode,
            self.phantom_size,
            self.angle_range,
            self.projections,
            self.circle,
            self.algorithm,
            self.generate_button,
            self.reconstruct_button,
            self.intermediate_combo,
            self.export_result_button,
            self.save_experiment_button,
        )
        for first, second in zip(tab_order, tab_order[1:], strict=False):
            QWidget.setTabOrder(first, second)
        self._source_mode_changed()

    def _source_mode_changed(self, _value: object = None) -> None:
        phantom_mode = self.source_mode.currentData() == "phantom"
        self.phantom_size.setEnabled(phantom_mode)
        if phantom_mode:
            self.source_provenance.setText(
                self._tr(
                    "Generated at runtime · illustrative μ in mm⁻¹ · fixed 0–0.03 mm⁻¹ "
                    "evaluation range · circular support outside value = 0 · no medical file "
                    "required."
                )
            )
            self.generate_button.setText(self._tr("Run first phantom experiment"))
        else:
            self.source_provenance.setText(
                self._tr(
                    "Advanced image-domain simulation: current pixel values are used as a "
                    "mathematical object. They are not scanner projections, calibrated μ, or "
                    "an automatic HU conversion; RGB conversion and cropping are refused."
                )
            )
            self.generate_button.setText(self._tr("Run image-domain simulation"))
        if self._active_task is None:
            self._invalidate_sinogram()
        self._update_action_status()

    def _update_action_status(self, status: str | None = None, *, blocked: bool = False) -> None:
        if status is None:
            if self._active_task is not None:
                status = "Running in the background. You can cancel safely."
            elif self.reconstruction_result is not None:
                status = "Experiment complete. Review the shared-range metrics or save a record."
            elif self.sinogram_result is not None:
                status = "Sinogram ready. Next: choose an algorithm and reconstruct."
            elif self.source_mode.currentData() == "phantom":
                status = (
                    "Ready: one click generates a safe synthetic phantom, simulates its "
                    "sinogram, and reconstructs it."
                )
            elif not self._source_available:
                status = "Blocked: open an image in Images, then return to this advanced mode."
                blocked = True
            else:
                status = (
                    "Ready: the active 2-D plane will be used for an image-domain simulation; "
                    "review the scientific limitation above."
                )
        self.action_status.setText(self._tr(status))
        _set_dynamic_property(self.action_status, "state", "blocked" if blocked else "ready")
        can_start = bool(
            self._active_task is None
            and (self.source_mode.currentData() == "phantom" or self._source_available)
        )
        self.generate_button.setEnabled(can_start)
        reason = "" if can_start else self._tr(status)
        self.generate_button.setToolTip(reason)
        self.generate_button.setAccessibleDescription(self._tr(status))
        if self._active_task is not None:
            reconstruction_reason = self._tr(status)
        elif self.sinogram_result is not None:
            reconstruction_reason = self._tr(
                "Sinogram ready. Next: choose an algorithm and reconstruct."
            )
        else:
            reconstruction_reason = self._tr("Generate a sinogram first.")
        self.reconstruct_button.setToolTip(
            "" if self.reconstruct_button.isEnabled() else reconstruction_reason
        )
        self.reconstruct_button.setAccessibleDescription(reconstruction_reason)
        self.statusChanged.emit(self._tr(status))

    def _run_selected_source(self) -> None:
        self._auto_reconstruct_after_sinogram = self.source_mode.currentData() == "phantom"
        self._generate()

    def start_first_experiment(self) -> None:
        """Run the deterministic offline phantom path from a single user action."""

        if self._active_task is not None:
            return
        index = self.source_mode.findData("phantom")
        self.source_mode.setCurrentIndex(max(index, 0))
        self.angle_range.setCurrentText("180")
        self.circle.setChecked(True)
        self.algorithm.setCurrentText("FBP")
        self._auto_reconstruct_after_sinogram = True
        self._generate()

    def _update_algorithm_options(self, algorithm: str) -> None:
        self.algorithm_options.setCurrentIndex(
            {"FBP": 0, "BP": 1, "DFR": 2, "SART": 3}.get(algorithm, 1)
        )

    def _set_reconstruction_controls_enabled(self, enabled: bool) -> None:
        self.source_mode.setEnabled(enabled)
        self.phantom_size.setEnabled(enabled and self.source_mode.currentData() == "phantom")
        for widget in (
            self.angle_range,
            self.projections,
            self.circle,
            self.algorithm,
            self.algorithm_options,
        ):
            widget.setEnabled(enabled)

    def _scan_parameters_changed(self, _value: object = None) -> None:
        if self._active_task is None and self.sinogram_result is not None:
            self._invalidate_sinogram()
        self._update_action_status()

    def _reconstruction_parameters_changed(self, _value: object = None) -> None:
        if self._active_task is None and self.reconstruction_result is not None:
            self._invalidate_reconstruction()
        self._update_action_status()

    def _invalidate_sinogram(self) -> None:
        self.sinogram_result = None
        self.sinogram_view.clear_image()
        self.reconstruct_button.setEnabled(False)
        self._invalidate_reconstruction()

    def _invalidate_reconstruction(self) -> None:
        self.reconstruction_result = None
        self.result_view.clear_image()
        self.error_view.clear_image()
        self._latest_metrics.clear()
        self._metric_maps.clear()
        self._refresh_process_images()
        self.metrics_label.setText(self._tr("Metrics use one shared intensity range."))
        self.export_result_button.setEnabled(False)
        self.save_experiment_button.setEnabled(False)
        if hasattr(self, "action_status"):
            self._update_action_status()

    def _language_changed(self) -> None:
        self._populate_process_combo(refresh_view=False)
        metric_text = self.metrics_label.text()
        english_prefix = "Evaluation range "
        chinese_prefix = f"{translate('Evaluation range', 'zh_CN')} "
        source_prefix, target_prefix = (
            (english_prefix, chinese_prefix)
            if self._language == "zh_CN"
            else (chinese_prefix, english_prefix)
        )
        if metric_text.startswith(source_prefix):
            self.metrics_label.setText(target_prefix + metric_text[len(source_prefix) :])

    @staticmethod
    def _prepare_source(source: np.ndarray) -> np.ndarray:
        array = np.asarray(source)
        if array.ndim == 3:
            raise ViewerError(
                "The active image has multiple color channels. This lab will not silently "
                "convert RGB to attenuation; choose a grayscale plane explicitly."
            )
        if array.ndim != 2:
            raise ViewerError("CT reconstruction requires one 2-D grayscale plane.")
        if array.shape[0] != array.shape[1]:
            raise ViewerError(
                "The active plane is not square. This lab will not crop it silently; prepare "
                "an explicit square teaching input or use the synthetic phantom."
            )
        values = np.asarray(array, dtype=np.float64)
        if not np.all(np.isfinite(values)):
            raise ViewerError("The active plane contains NaN or infinity.")
        return values

    @staticmethod
    def _prepare_rois(
        source_shape: tuple[int, int],
        rois: dict[str, tuple[int, int, int, int]],
    ) -> dict[str, tuple[int, int, int, int]]:
        """Clip source ROIs to the square center crop used for reconstruction."""

        source_height, source_width = source_shape
        size = min(source_height, source_width)
        crop_top = (source_height - size) // 2
        crop_left = (source_width - size) // 2
        prepared: dict[str, tuple[int, int, int, int]] = {}
        for name, (x, y, width, height) in rois.items():
            left = max(int(x), crop_left)
            top = max(int(y), crop_top)
            right = min(int(x + width), crop_left + size)
            bottom = min(int(y + height), crop_top + size)
            clipped_width = right - left
            clipped_height = bottom - top
            if clipped_width >= 3 and clipped_height >= 3:
                prepared[str(name)] = (
                    left - crop_left,
                    top - crop_top,
                    clipped_width,
                    clipped_height,
                )
        return prepared

    def _generate(self) -> None:
        try:
            if self.source_mode.currentData() == "phantom":
                phantom = generate_ct_phantom(self.phantom_size.value())
                self.reference = phantom.array
                self._evaluation_range = phantom.evaluation_range
                self._evaluation_unit = phantom.unit
                self._reference_rois = {}
            else:
                source = np.asarray(self.source())
                self.reference = self._prepare_source(source)
                if np.any(self.reference < 0.0):
                    raise ViewerError(
                        "The active image contains negative values. This lab will not treat HU "
                        "or arbitrary negative signal as linear attenuation automatically; use "
                        "the phantom or perform a documented HU-to-μ conversion first."
                    )
                low = float(np.min(self.reference))
                high = float(np.max(self.reference))
                if high <= low:
                    raise ViewerError(
                        "The active image is constant and has no finite evaluation range."
                    )
                self._evaluation_range = (low, high)
                self._evaluation_unit = None
                self._reference_rois = self._prepare_rois(source.shape[:2], self.roi_source())
                if self.circle.isChecked():
                    height, width = self.reference.shape
                    yy, xx = np.ogrid[-1.0 : 1.0 : complex(height), -1.0 : 1.0 : complex(width)]
                    outside = (xx * xx + yy * yy) > 1.0
                    scale = max(float(np.max(np.abs(self.reference))), 1.0)
                    if np.any(np.abs(self.reference[outside]) > scale * 1e-8):
                        raise ViewerError(
                            "Circular support is enabled, but the active image has non-zero "
                            "values outside the reconstruction circle. Disable Circular support "
                            "or use the zero-background phantom."
                        )
        except BaseException as exc:
            self._auto_reconstruct_after_sinogram = False
            self._update_action_status(
                f"Blocked: {exc}",
                blocked=True,
            )
            QMessageBox.warning(
                self,
                self._tr("No reconstruction input"),
                self._tr(str(exc)),
            )
            return
        assert self._evaluation_range is not None
        mapping = GrayscaleDisplayMapping(*self._evaluation_range)
        self.input_view.set_array(self.reference, grayscale_mapping=mapping, fit=True)
        self.progress.setValue(0)
        self.reconstruct_button.setEnabled(False)
        task = self.service.begin_sinogram(
            self.reference,
            projection_count=self.projections.value(),
            angle_range=int(self.angle_range.currentText()),
            circle=self.circle.isChecked(),
            source_kind=(
                ReconstructionSourceKind.SINOGRAM
                if self.source_mode.currentData() == "phantom"
                else ReconstructionSourceKind.IMAGE_DERIVED_SIMULATION
            ),
        )
        self._watch_task(task, self._sinogram_ready)

    def _sinogram_ready(self, result: Any) -> None:
        self.sinogram_result = result
        self.sinogram_view.set_array(result.sinogram, fit=True)
        self._refresh_process_images()
        self.reconstruct_button.setEnabled(True)
        self.progress.setValue(100)
        self._update_action_status()
        if self._auto_reconstruct_after_sinogram:
            self._auto_reconstruct_after_sinogram = False
            QTimer.singleShot(0, self._reconstruct)

    def _reconstruct(self) -> None:
        if self.sinogram_result is None or self.reference is None:
            return
        request = ReconstructionRequest.from_sinogram_result(
            self.sinogram_result,
            output_size=self.reference.shape[0],
        )
        task = self.service.begin_reconstruction(
            request,
            algorithm=self.algorithm.currentText(),
            interpolation=self.interpolation.currentText(),
            filter_name=self.filter.currentText(),
            iterations=self.iterations.value(),
            relaxation=self.relaxation.value(),
        )
        self._watch_task(task, self._reconstruction_ready)

    def _watch_task(self, task: BackgroundTask[Any], success: Callable[[Any], None]) -> None:
        self._active_task = task
        self.progress.setVisible(True)
        self.progress.setValue(0)
        self._set_reconstruction_controls_enabled(False)
        self.generate_button.setEnabled(False)
        self.reconstruct_button.setEnabled(False)
        self.cancel_button.setText(self._tr("Cancel"))
        self.cancel_button.setEnabled(True)
        self._update_action_status("Running in the background. You can cancel safely.")
        self.watcher.watch(
            task,
            success=lambda result, handle=task: self._task_succeeded(handle, success, result),
            error=lambda error, handle=task: self._task_failed(handle, error),
            progress=lambda fraction, message, handle=task: self._task_progress_if_current(
                handle, fraction, message
            ),
        )

    def _task_succeeded(
        self,
        task: BackgroundTask[Any],
        callback: Callable[[Any], None],
        result: Any,
    ) -> None:
        if task is not self._active_task:
            return
        self._active_task = None
        callback(result)
        self.progress.setVisible(False)
        self._set_reconstruction_controls_enabled(True)
        self.generate_button.setEnabled(True)
        self.reconstruct_button.setEnabled(self.sinogram_result is not None)
        self.cancel_button.setText(self._tr("Cancel"))
        self.cancel_button.setEnabled(False)
        self._update_action_status()

    def _task_failed(self, task: BackgroundTask[Any], error: BaseException) -> None:
        if task is not self._active_task:
            return
        self._active_task = None
        self._failed(error)
        self.progress.setVisible(False)
        self._set_reconstruction_controls_enabled(True)
        self.generate_button.setEnabled(True)
        self.reconstruct_button.setEnabled(self.sinogram_result is not None)
        self.cancel_button.setText(self._tr("Cancel"))
        self.cancel_button.setEnabled(False)
        self.progress.setVisible(False)
        self._update_action_status(
            f"Operation stopped safely: {error}",
            blocked=error.__class__.__name__ != "OperationCancelled",
        )

    def _task_progress_if_current(
        self,
        task: BackgroundTask[Any],
        fraction: float,
        message: str,
    ) -> None:
        if task is self._active_task:
            self._progress(fraction, message)

    def reset(self, _image: object | None = None) -> None:
        self._cancel_all()
        self._active_task = None
        self._active_output_task = None
        self.reference = None
        self._evaluation_range = None
        self._evaluation_unit = None
        self.sinogram_result = None
        self.reconstruction_result = None
        self._latest_metrics.clear()
        self._reference_rois.clear()
        self._metric_maps.clear()
        self._process_images.clear()
        self.intermediate_combo.clear()
        self.metrics_label.setText(self._tr("Metrics use one shared intensity range."))
        self._set_reconstruction_controls_enabled(True)
        self._source_available = _image is not None
        self.source_status.setVisible(True)
        self.source_status.setText(
            self._tr(
                "Ready: a synthetic phantom is available; the active image mode is also ready."
                if self._source_available
                else "Ready: run the synthetic phantom experiment without opening patient data."
            )
        )
        self.reconstruct_button.setEnabled(False)
        self.cancel_button.setText(self._tr("Cancel"))
        self.cancel_button.setEnabled(False)
        self.export_result_button.setEnabled(False)
        self.save_experiment_button.setEnabled(False)
        for view in (self.input_view, self.sinogram_view, self.result_view, self.error_view):
            view.clear_image()
        self._update_action_status()

    def _reconstruction_ready(self, result: Any) -> None:
        self.reconstruction_result = result
        assert self.reference is not None
        if self._evaluation_range is None:
            low = float(np.min(self.reference))
            high = float(np.max(self.reference))
            if high <= low:
                high = low + 1.0
            self._evaluation_range = (low, high)
        shared_mapping = GrayscaleDisplayMapping(*self._evaluation_range)
        self.input_view.set_array(
            self.reference,
            grayscale_mapping=shared_mapping,
            fit=False,
        )
        self.input_view.set_value_scale(
            *self._evaluation_range,
            unit=self._evaluation_unit,
        )
        self.result_view.set_array(
            result.image,
            grayscale_mapping=shared_mapping,
            fit=True,
        )
        self.result_view.set_value_scale(
            *self._evaluation_range,
            unit=self._evaluation_unit,
        )
        self.intermediate_combo.clear()
        self._metric_maps.clear()
        try:
            metrics = self.service.compute_metrics(
                self.reference,
                result.image,
                rois=self._reference_rois,
                intensity_range=self._evaluation_range,
                unit=self._evaluation_unit,
            )
            self._metric_maps = {
                "Signed normalized difference": metrics.difference,
                "Absolute error heatmap": metrics.error_heatmap,
            }
            self.error_view.set_array(
                metrics.error_heatmap,
                grayscale_mapping=GrayscaleDisplayMapping(0.0, 1.0),
                fit=True,
            )
            self.error_view.set_value_scale(0.0, 1.0, unit="normalized |error|")
            self.error_view.set_title("Absolute error heatmap")
            metric_text = (
                f"{self._tr('Evaluation range')} "
                f"{metrics.intensity_range[0]:.4g}…{metrics.intensity_range[1]:.4g} | "
                f"MSE {metrics.values.mse:.6g} | "
                f"PSNR {metrics.values.psnr:.4g} dB | "
                f"SSIM {metrics.values.ssim:.5f}"
            )
            if metrics.raw_values is not None:
                unit_suffix = f" {metrics.raw_values.unit}" if metrics.raw_values.unit else ""
                metric_text += (
                    f" | Raw MAE {metrics.raw_values.mae:.6g}{unit_suffix}"
                    f" | RMSE {metrics.raw_values.rmse:.6g}{unit_suffix}"
                    f" | bias {metrics.raw_values.bias:.6g}{unit_suffix}"
                )
            self._latest_metrics = {
                "mse": metrics.values.mse,
                "psnr_db": (metrics.values.psnr if np.isfinite(metrics.values.psnr) else None),
                "ssim": metrics.values.ssim,
                "joint_intensity_min": metrics.intensity_range[0],
                "joint_intensity_max": metrics.intensity_range[1],
            }
            if metrics.raw_values is not None:
                self._latest_metrics.update(
                    {
                        "raw_mae": metrics.raw_values.mae,
                        "raw_rmse": metrics.raw_values.rmse,
                        "raw_bias": metrics.raw_values.bias,
                    }
                )
            for name, values in metrics.roi.items():
                metric_text += (
                    f" | ROI {name}: MSE {values.mse:.6g}, "
                    f"PSNR {values.psnr:.4g} dB, SSIM {values.ssim:.5f}"
                )
                key = "".join(character if character.isalnum() else "_" for character in name)
                self._latest_metrics[f"roi_{key}_mse"] = values.mse
                self._latest_metrics[f"roi_{key}_psnr_db"] = (
                    values.psnr if np.isfinite(values.psnr) else None
                )
                self._latest_metrics[f"roi_{key}_ssim"] = values.ssim
            self.metrics_label.setText(metric_text)
        except BaseException as exc:
            self._latest_metrics.clear()
            self.metrics_label.setText(self._tr("Metrics unavailable: {error}").format(error=exc))
        self._refresh_process_images()
        error_index = self.intermediate_combo.findData("Absolute error heatmap")
        if error_index >= 0:
            self.intermediate_combo.setCurrentIndex(error_index)
        self.export_result_button.setEnabled(True)
        self.save_experiment_button.setEnabled(True)
        self.progress.setValue(100)
        self._update_action_status()

    def _show_intermediate(self, index: int) -> None:
        if index < 0:
            return
        key = self.intermediate_combo.currentData()
        if not isinstance(key, str):
            key = self.intermediate_combo.currentText()
        value = self._process_images.get(key)
        if value is None:
            return
        array = np.asarray(value)
        if array.ndim == 3:
            array = array[:, :, -1]
        if np.iscomplexobj(array):
            array = np.log1p(np.abs(array))
        if array.ndim == 2:
            title = key if key == "Absolute error heatmap" else f"Intermediate: {key}"
            self.error_view.set_title(title)
            self.intermediate_combo.setToolTip(self._tr(f"Intermediate: {key}"))
            if key == "Signed normalized difference":
                self.error_view.set_array(self._diverging_difference_rgb(array), fit=True)
                self.error_view.set_value_scale(-1.0, 1.0, unit="normalized", diverging=True)
            elif key == "Absolute error heatmap":
                self.error_view.set_array(
                    array,
                    grayscale_mapping=GrayscaleDisplayMapping(0.0, 1.0),
                    fit=True,
                )
                self.error_view.set_value_scale(0.0, 1.0, unit="normalized |error|")
            else:
                self.error_view.set_array(array, fit=True)

    @staticmethod
    def _diverging_difference_rgb(values: np.ndarray) -> np.ndarray:
        """Map fixed [-1, 1] signed error to blue-white-red without changing metrics."""

        normalized = np.clip(np.asarray(values, dtype=np.float64), -1.0, 1.0)
        blue = np.asarray((37.0, 99.0, 235.0))
        white = np.asarray((249.0, 250.0, 251.0))
        red = np.asarray((220.0, 38.0, 38.0))
        magnitude = np.abs(normalized)[..., None]
        target = np.where((normalized >= 0.0)[..., None], red, blue)
        rgb = white + magnitude * (target - white)
        return np.ascontiguousarray(np.rint(rgb), dtype=np.uint8)

    def _refresh_process_images(self) -> None:
        """Expose every teaching snapshot instead of collapsing sequences."""

        entries: dict[str, np.ndarray] = {}
        if self.sinogram_result is not None:
            projection_total = len(self.sinogram_result.theta_degrees)
            for key, value in self.sinogram_result.intermediate.items():
                suffix = key.removeprefix("projections_")
                label = (
                    f"Radon projection {suffix}/{projection_total}"
                    if key.startswith("projections_") and suffix.isdigit()
                    else self._humanize_process_key(key)
                )
                self._collect_process_images(entries, label, value)
        if self.reconstruction_result is not None:
            algorithm = str(getattr(self.reconstruction_result, "algorithm", "")).upper()
            for key, value in self.reconstruction_result.intermediate.items():
                if key == "backprojection_steps":
                    label = f"{algorithm or 'Backprojection'} progress"
                elif key == "iteration_images":
                    label = "SART iteration"
                else:
                    label = self._humanize_process_key(key)
                self._collect_process_images(entries, label, value)
        entries.update(self._metric_maps)
        self._process_images = entries
        self._populate_process_combo()

    def _populate_process_combo(self, *, refresh_view: bool = True) -> None:
        selected = self.intermediate_combo.currentData()
        self.intermediate_combo.blockSignals(True)
        self.intermediate_combo.clear()
        for key in self._process_images:
            self.intermediate_combo.addItem(self._tr(key), key)
        if isinstance(selected, str):
            selected_index = self.intermediate_combo.findData(selected)
            if selected_index >= 0:
                self.intermediate_combo.setCurrentIndex(selected_index)
        self.intermediate_combo.blockSignals(False)
        if refresh_view and self.intermediate_combo.currentIndex() >= 0:
            self._show_intermediate(self.intermediate_combo.currentIndex())

    @classmethod
    def _collect_process_images(
        cls,
        entries: dict[str, np.ndarray],
        label: str,
        value: Any,
    ) -> None:
        if isinstance(value, Mapping):
            for nested_key, nested_value in value.items():
                cls._collect_process_images(
                    entries,
                    f"{label} / {cls._humanize_process_key(str(nested_key))}",
                    nested_value,
                )
            return
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes, np.ndarray)):
            total = len(value)
            for index, snapshot in enumerate(value, start=1):
                cls._collect_process_images(
                    entries,
                    f"{label} {index}/{total}",
                    snapshot,
                )
            return
        array = np.asarray(value)
        if array.size:
            entries[label] = array

    @staticmethod
    def _humanize_process_key(key: str) -> str:
        return key.replace("_", " ").strip().capitalize()

    def _cancel_all(self) -> None:
        active = self._active_task is not None or (
            self._active_output_task is not None and not self._active_output_task.done
        )
        self.service.cancel_active()
        if self._active_output_task is not None and not self._active_output_task.done:
            self._active_output_task.cancel()
        if active:
            self.cancel_button.setEnabled(False)
            self.cancel_button.setText(self._tr("Cancelling…"))
            self.progress.setFormat(self._tr("Cancelling safely…"))

    def _start_output_operation(
        self,
        operation: Callable[[Any], Path],
        success_message: str,
    ) -> None:
        if self._active_output_task is not None and not self._active_output_task.done:
            self._active_output_task.cancel()
        task = self.output_runner.submit(operation)
        self._active_output_task = task
        self.progress.setVisible(True)
        self.progress.setValue(0)
        self._set_reconstruction_controls_enabled(False)
        self.generate_button.setEnabled(False)
        self.reconstruct_button.setEnabled(False)
        self.export_result_button.setEnabled(False)
        self.save_experiment_button.setEnabled(False)
        self.cancel_button.setText(self._tr("Cancel"))
        self.cancel_button.setEnabled(True)
        self.watcher.watch(
            task,
            success=lambda path, handle=task: self._output_succeeded(handle, path, success_message),
            error=lambda error, handle=task: self._output_failed(handle, error),
            progress=lambda fraction, message, handle=task: self._output_progress(
                handle, fraction, message
            ),
        )

    def _output_succeeded(
        self,
        task: BackgroundTask[Any],
        path: Path,
        message: str,
    ) -> None:
        if task is not self._active_output_task:
            return
        self._active_output_task = None
        self._set_reconstruction_controls_enabled(True)
        self.generate_button.setEnabled(True)
        self.reconstruct_button.setEnabled(self.sinogram_result is not None)
        self.export_result_button.setEnabled(self.reconstruction_result is not None)
        self.save_experiment_button.setEnabled(self.reconstruction_result is not None)
        self.cancel_button.setText(self._tr("Cancel"))
        self.cancel_button.setEnabled(False)
        self.progress.setValue(100)
        self.progress.setFormat(f"{self._tr(message)}: {path.name}")

    def _output_failed(self, task: BackgroundTask[Any], error: BaseException) -> None:
        if task is not self._active_output_task:
            return
        self._active_output_task = None
        self._set_reconstruction_controls_enabled(True)
        self.generate_button.setEnabled(True)
        self.reconstruct_button.setEnabled(self.sinogram_result is not None)
        self.export_result_button.setEnabled(self.reconstruction_result is not None)
        self.save_experiment_button.setEnabled(self.reconstruction_result is not None)
        self.cancel_button.setText(self._tr("Cancel"))
        self.cancel_button.setEnabled(False)
        self.progress.setFormat(self._tr("Local output failed safely: {error}").format(error=error))
        if error.__class__.__name__ != "OperationCancelled":
            QMessageBox.warning(self, self._tr("Local output failed"), str(error))

    def _output_progress(
        self,
        task: BackgroundTask[Any],
        fraction: float,
        message: str,
    ) -> None:
        if task is self._active_output_task:
            self.progress.setValue(int(round(fraction * 100)))
            self.progress.setFormat(f"{self._tr(message)} – %p%")

    def _export_result(self) -> None:
        if self.reconstruction_result is None:
            return
        path, _ = QFileDialog.getSaveFileName(
            self,
            self._tr("Export reconstruction to a new local file"),
            "openmedvisionx-reconstruction.png",
            self._tr("PNG image (*.png)"),
        )
        if not path:
            return
        image = self.reconstruction_result.image

        def operation(context: Any) -> Path:
            return export_rendered_png(
                path,
                image,
                cancel=lambda: context.cancelled,
                progress=lambda fraction, message: context.report_progress(
                    fraction, message=message
                ),
            )

        self._start_output_operation(operation, "Rendered reconstruction saved as new file")

    def _save_experiment(self) -> None:
        if self.reconstruction_result is None or self.reference is None:
            return
        path, _ = QFileDialog.getSaveFileName(
            self,
            self._tr("Save reconstruction parameters and numeric metrics"),
            "openmedvisionx-reconstruction-experiment.json",
            self._tr("Experiment JSON (*.json)"),
        )
        if not path:
            return
        parameters = {
            "source_kind": (
                self.sinogram_result.source_kind.value
                if self.sinogram_result is not None
                else ReconstructionSourceKind.IMAGE_DERIVED_SIMULATION.value
            ),
            "source_mode": str(self.source_mode.currentData()),
            "input_shape": list(self.reference.shape),
            "evaluation_range": list(self._evaluation_range or ()),
            "evaluation_unit": self._evaluation_unit,
            "angle_range_degrees": int(self.angle_range.currentText()),
            "projection_count": self.projections.value(),
            "circular_support": self.circle.isChecked(),
            "algorithm": self.algorithm.currentText(),
            "dfr_interpolation": self.interpolation.currentText(),
            "fbp_filter": self.filter.currentText(),
            "sart_iterations": self.iterations.value(),
            "sart_relaxation": self.relaxation.value(),
        }
        record = create_experiment_record(
            "CT reconstruction",
            image=None,
            parameters=parameters,
            metrics=self._latest_metrics,
        )

        def operation(context: Any) -> Path:
            return save_experiment_record(
                path,
                record,
                cancel=lambda: context.cancelled,
                progress=lambda fraction, message: context.report_progress(
                    fraction, message=message
                ),
            )

        self._start_output_operation(operation, "Pixel-free experiment record saved")

    def _progress(self, fraction: float, message: str) -> None:
        self.progress.setValue(int(round(fraction * 100)))
        self.progress.setFormat(f"{self._tr(message)} – %p%")

    def _failed(self, error: BaseException) -> None:
        self.progress.setFormat(str(error))
        if error.__class__.__name__ != "OperationCancelled":
            QMessageBox.critical(self, self._tr("Reconstruction failed"), str(error))

    def close(self) -> None:  # type: ignore[override]
        self.watcher.stop()
        self._cancel_all()
        self.service.close()
        self.output_runner.shutdown(wait=False, cancel_pending=True)
        super().close()


class ModelPage(_BilingualPage, QWidget):
    statusChanged = pyqtSignal(str)

    def __init__(
        self,
        source: Callable[[], RasterImage2D | np.ndarray],
        input_context: Callable[[], dict[str, Any]] | None = None,
        service: ModelInferenceService | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.source = source
        self.input_context = input_context or (lambda: {})
        self.service = service or ModelInferenceService()
        self.runner = TaskRunner(max_workers=1, thread_name_prefix="openmedvisionx-model")
        self.watcher = TaskWatcher(self)
        self.manifest: ModelManifest | None = None
        self._plugin_ready = False
        self._source_available = True
        self._operation_stage: str | None = None
        self._active_task: BackgroundTask[Any] | None = None
        self._last_input: np.ndarray | None = None
        self._last_input_context: dict[str, Any] = {}
        self._display_transform: TransformRecord | None = None
        self._source_grayscale_mapping: GrayscaleDisplayMapping | None = None
        self._source_scale_unit = ""
        self._summary_segments: list[tuple[str, str]] = [("owned", MODEL_SUMMARY_INTRO)]
        self._rendered_summary_text = MODEL_SUMMARY_INTRO
        root_layout = QVBoxLayout(self)
        root_layout.setContentsMargins(0, 8, 0, 0)
        root_layout.setSpacing(10)
        self.model_workspace_tabs = QTabWidget()
        self.model_workspace_tabs.setObjectName("modelWorkspaceTabs")
        self.model_workspace_tabs.setDocumentMode(True)
        self.model_workspace_tabs.setAccessibleName("Bundled and external model workflows")
        self.bundled_models = BundledModelsPanel()
        self.bundled_models.setMinimumWidth(0)
        self.bundled_models.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Expanding)
        self.bundled_models_scroll = QScrollArea()
        self.bundled_models_scroll.setWidgetResizable(True)
        self.bundled_models_scroll.setFrameShape(QFrame.NoFrame)
        self.bundled_models_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.bundled_models_scroll.setAccessibleName("Scrollable bundled model catalog")
        self.bundled_models_scroll.setWidget(self.bundled_models)
        self.external_models = QWidget()
        layout = QVBoxLayout(self.external_models)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)
        self.external_models_scroll = QScrollArea()
        self.external_models_scroll.setWidgetResizable(True)
        self.external_models_scroll.setFrameShape(QFrame.NoFrame)
        self.external_models_scroll.setAccessibleName("Scrollable external model workflow")
        self.external_models_scroll.setWidget(self.external_models)
        self.model_workspace_tabs.addTab(self.bundled_models_scroll, "Bundled models")
        self.model_workspace_tabs.addTab(self.external_models_scroll, "External manifests")
        root_layout.addWidget(self.model_workspace_tabs, 1)
        notice = QLabel(
            "Models and weights stay local. Choose a manifest, load its referenced model, "
            "then run it on the current image."
        )
        notice.setWordWrap(True)
        notice.setObjectName("infoBanner")
        layout.addWidget(notice)

        self.resource_status = QLabel()
        self.resource_status.setObjectName("resourceBanner")
        self.resource_status.setWordWrap(True)
        self.resource_status.setTextFormat(Qt.RichText)
        self.resource_status.setTextInteractionFlags(Qt.TextBrowserInteraction)
        self.resource_status.setOpenExternalLinks(True)
        self.resource_status.setAccessibleName("Model resource status")
        layout.addWidget(self.resource_status)

        self.capability_status = QLabel()
        self.capability_status.setObjectName("actionStatus")
        self.capability_status.setWordWrap(True)
        self.capability_status.setAccessibleName("Model page capability check and next step")
        layout.addWidget(self.capability_status)
        self.preview_override = QCheckBox("Allow preview-only teaching inference")
        self.preview_override.setVisible(False)
        self.preview_override.setAccessibleName("Allow preview-only teaching inference")
        self.preview_override.setAccessibleDescription(
            "Explicitly allow a reduced-resolution teaching run; a final confirmation is "
            "still required and the override is recorded."
        )
        self.preview_override.stateChanged.connect(self._update_capability_gate)
        layout.addWidget(self.preview_override)

        actions = QFrame()
        actions.setObjectName("toolbarSurface")
        row = QGridLayout(actions)
        row.setContentsMargins(10, 9, 10, 9)
        row.setHorizontalSpacing(8)
        row.setVerticalSpacing(8)
        self.manifest_button = QPushButton("1. Choose manifest…")
        self.manifest_button.setObjectName("primary")
        self.manifest_button.clicked.connect(self._open_manifest)
        self.load_button = QPushButton("2. Load model")
        self.load_button.clicked.connect(self._load_plugin)
        self.load_button.setEnabled(False)
        self.run_button = QPushButton("3. Run inference")
        self.run_button.clicked.connect(self._run_plugin)
        self.run_button.setEnabled(False)
        self.cancel_button = QPushButton("Stop")
        self.cancel_button.clicked.connect(self._cancel)
        self.cancel_button.setEnabled(False)
        self.model_actions_surface = actions
        self.model_actions_layout = row
        self._model_action_buttons = (
            self.manifest_button,
            self.load_button,
            self.run_button,
            self.cancel_button,
        )
        self._model_action_layout_mode = "grid"
        row.addWidget(self.manifest_button, 0, 0)
        row.addWidget(self.load_button, 0, 1)
        row.addWidget(self.run_button, 1, 0)
        row.addWidget(self.cancel_button, 1, 1)
        row.setColumnStretch(0, 1)
        row.setColumnStretch(1, 1)
        layout.addWidget(actions)
        self.progress = QProgressBar()
        self.progress.setVisible(False)
        layout.addWidget(self.progress)
        self.summary = QPlainTextEdit()
        self.summary.setReadOnly(True)
        self.summary.setPlainText(MODEL_SUMMARY_INTRO)
        self.summary.setAccessibleName("Manifest details and inference summary")
        layout.addWidget(self.summary, 1)
        comparison = QSplitter(Qt.Horizontal)
        self.model_input_view = ImageView("Current model input")
        self.model_output_view = ImageView("Typed model visualization")
        self.model_input_view.set_empty_message(
            "Next: open an image, choose a manifest, and load a compatible 2-D model."
        )
        self.model_output_view.set_empty_message(
            "Validated typed model output will appear here after inference."
        )
        self.model_output_tabs = QTabWidget()
        self.model_output_tabs.setObjectName("modelOutputTabs")
        self.model_output_tabs.setAccessibleName("Model output visualizations")
        self.model_output_tabs.addTab(self.model_output_view, "Visualization")
        comparison.addWidget(self.model_input_view)
        comparison.addWidget(self.model_output_tabs)
        comparison.setStretchFactor(0, 1)
        comparison.setStretchFactor(1, 1)
        comparison.setChildrenCollapsible(False)
        layout.addWidget(comparison, 2)
        self._refresh_model_resource_status()
        self._update_capability_gate()
        tab_order = (
            self.manifest_button,
            self.load_button,
            self.run_button,
            self.cancel_button,
            self.preview_override,
            self.summary,
            self.model_input_view,
            self.model_output_tabs,
        )
        for first, second in zip(tab_order, tab_order[1:], strict=False):
            QWidget.setTabOrder(first, second)
        self.bundled_models.statusChanged.connect(self._bundled_status_changed)
        self.model_workspace_tabs.currentChanged.connect(self._workspace_changed)

    def _language_changed(self) -> None:
        """Retranslate app-owned summary prose while preserving plugin-provided values."""

        if self.summary.toPlainText() != self._rendered_summary_text:
            self._summary_segments = [("external", self.summary.toPlainText())]
        self._render_summary()
        self.bundled_models.set_language(self._language)
        self.model_workspace_tabs.setTabText(0, self._tr("Bundled models"))
        self.model_workspace_tabs.setTabText(1, self._tr("External manifests"))
        self.model_workspace_tabs.setAccessibleName(
            self._tr("Bundled and external model workflows")
        )
        self._refresh_model_resource_status()
        self._update_capability_gate()
        self._layout_model_actions()

    def current_status_text(self) -> str:
        if self.model_workspace_tabs.currentWidget() is self.bundled_models_scroll:
            return self.bundled_models.operation_status.text()
        return self.capability_status.text()

    def _bundled_status_changed(self, message: str) -> None:
        if self.model_workspace_tabs.currentWidget() is self.bundled_models_scroll:
            self.statusChanged.emit(message)

    def _workspace_changed(self, _index: int) -> None:
        self.statusChanged.emit(self.current_status_text())

    def resizeEvent(self, event: object) -> None:  # noqa: N802 - Qt API
        super().resizeEvent(event)  # type: ignore[arg-type]
        self._layout_model_actions()

    def _layout_model_actions(self) -> None:
        """Use one task-flow row when labels fit, otherwise a full-width two-row grid."""

        margins = self.model_actions_layout.contentsMargins()
        available = max(
            0,
            self.model_actions_surface.width() - margins.left() - margins.right(),
        )
        spacing = max(0, self.model_actions_layout.horizontalSpacing())
        required = sum(button.sizeHint().width() for button in self._model_action_buttons)
        required += spacing * (len(self._model_action_buttons) - 1)
        mode = "row" if available >= required else "grid"
        if mode == self._model_action_layout_mode:
            return
        for button in self._model_action_buttons:
            self.model_actions_layout.removeWidget(button)
        for column in range(4):
            self.model_actions_layout.setColumnStretch(column, 0)
        if mode == "row":
            for column, button in enumerate(self._model_action_buttons):
                self.model_actions_layout.addWidget(button, 0, column)
                self.model_actions_layout.setColumnStretch(column, 1)
        else:
            for index, button in enumerate(self._model_action_buttons):
                self.model_actions_layout.addWidget(button, index // 2, index % 2)
            self.model_actions_layout.setColumnStretch(0, 1)
            self.model_actions_layout.setColumnStretch(1, 1)
        self._model_action_layout_mode = mode

    def _update_load_button(self) -> None:
        """Preflight required local resources before any runtime code is loaded."""

        if self._active_task is not None:
            enabled = False
            reason = self._tr("Running in the background. You can cancel safely.")
        elif self.manifest is None:
            enabled = False
            reason = self._tr("Choose a reviewed model manifest to begin.")
        else:
            missing = self._missing_weight_specs()
            enabled = not missing
            if missing:
                names = ", ".join(Path(weight.path).name for weight in missing)
                reason = self._tr("Weight not found: {name}").format(name=names)
            else:
                reason = self._tr("All referenced local weight files are available.")
        self.load_button.setEnabled(enabled)
        self.load_button.setToolTip("" if enabled else reason)
        self.load_button.setAccessibleDescription(reason)

    def _capability_gate(self) -> tuple[bool, str, bool]:
        """Describe whether this 2-D single-image teaching surface can run the manifest."""

        if self._active_task is not None:
            return (
                False,
                "Running in the background. You can cancel safely.",
                False,
            )
        if self.manifest is None:
            return (
                False,
                "Next: choose a reviewed manifest. No model code runs during inspection.",
                False,
            )
        if len(self.manifest.inputs) != 1:
            return (
                False,
                "Blocked: this page supplies one active image, but the manifest declares "
                f"{len(self.manifest.inputs)} inputs. Multi-input and prompt workflows require "
                "a compatible dedicated UI or the Python API.",
                True,
            )
        specification = self.manifest.inputs[0]
        if specification.semantic.value != "image":
            return (
                False,
                "Blocked: this page supplies one image, but the manifest requires "
                f"{specification.semantic.value!r} input. No silent substitution is allowed.",
                True,
            )
        if specification.dimensionality.value != "2d":
            return (
                False,
                "Blocked: this page currently supports declared 2-D image inputs only; the "
                f"manifest requires {specification.dimensionality.value}. The active plane will "
                "not be passed as a silent dimensional downgrade.",
                True,
            )
        if not self._source_available:
            return (
                False,
                "Next: open an image in Images. The selected active 2-D view becomes model input.",
                False,
            )
        try:
            context = self.input_context()
        except BaseException as exc:
            return (False, f"Blocked: model input context is unavailable: {exc}", True)
        if bool(context.get("preview_only", False)) and not self.preview_override.isChecked():
            source_shape = tuple(context.get("source_shape", ()))
            display_shape = tuple(context.get("display_shape", ()))
            return (
                False,
                "Blocked: only a reduced-resolution preview is loaded "
                f"({source_shape} → {display_shape}). Full-resolution pixels are not in this "
                "session. Check the explicit teaching override only if this limitation is "
                "acceptable.",
                True,
            )
        modality = str(context.get("modality", "generic-image"))
        accepted_modalities = {item.value for item in specification.modalities}
        if "generic-image" not in accepted_modalities and modality not in accepted_modalities:
            accepted = ", ".join(sorted(accepted_modalities))
            return (
                False,
                f"Blocked: active input modality is {modality!r}; this manifest declares "
                f"{accepted}. Choose a compatible image or manifest.",
                True,
            )
        if specification.spacing.required and context.get("spacing") is None:
            return (
                False,
                "Blocked: the manifest requires physical spacing, but the active input has no "
                "validated spacing. Set explicit raster spacing or open a geometric volume.",
                True,
            )
        if self._missing_weight_specs():
            return (
                False,
                "Next: add the required local weight files shown above, then load the model.",
                False,
            )
        plane = str(context.get("plane", "2-D image"))
        if not self.service.ready:
            preview_note = (
                " Preview-only teaching override will require final confirmation."
                if bool(context.get("preview_only", False))
                else ""
            )
            return (
                False,
                f"Compatible input: {modality}, {plane}. Next: load the model from its declared "
                f"local paths.{preview_note}",
                False,
            )
        preview_note = (
            " Preview-only teaching override is selected and will be recorded."
            if bool(context.get("preview_only", False))
            else ""
        )
        return (
            True,
            f"Ready: compatible {modality} {plane} input; one declared 2-D image will run with "
            f"manifest preprocessing.{preview_note}",
            False,
        )

    def _update_capability_gate(self) -> None:
        preview_only = False
        if self._source_available:
            try:
                preview_only = bool(self.input_context().get("preview_only", False))
            except BaseException:
                preview_only = False
        self.preview_override.setVisible(preview_only)
        if not preview_only and self.preview_override.isChecked():
            self.preview_override.blockSignals(True)
            self.preview_override.setChecked(False)
            self.preview_override.blockSignals(False)
        ready, message, blocked = self._capability_gate()
        self.capability_status.setText(self._tr(message))
        state = (
            "busy"
            if self._active_task is not None
            else "blocked"
            if blocked
            else "ready"
            if ready
            else "next"
        )
        _set_dynamic_property(
            self.capability_status,
            "state",
            state,
        )
        self.run_button.setEnabled(ready and self._active_task is None)
        self.run_button.setToolTip("" if ready else self._tr(message))
        self.run_button.setAccessibleDescription(self._tr(message))
        self._update_load_button()
        self.statusChanged.emit(self._tr(message))

    def _missing_weight_specs(self) -> tuple[Any, ...]:
        manifest_path = self.service.manifest_path
        if self.manifest is None or manifest_path is None:
            return ()
        root = manifest_path.parent
        return tuple(
            weight
            for weight in self.manifest.weights
            if weight.required and not (root / weight.path).is_file()
        )

    def _manifest_github_url(self) -> str | None:
        if self.manifest is None:
            return None
        candidates = (
            self.manifest.license.urls.get("weights"),
            self.manifest.source.repository,
        )
        for candidate in candidates:
            if not candidate:
                continue
            parsed = urlsplit(candidate)
            if parsed.scheme == "https" and parsed.hostname in {"github.com", "www.github.com"}:
                return candidate
        return None

    def _resource_link(self, url: str, label: str) -> str:
        return (
            f'<a href="{escape(url, quote=True)}" '
            f'style="color:#155eef;text-decoration:none;">{escape(label)} ↗</a>'
        )

    def _refresh_model_resource_status(self) -> None:
        self._update_load_button()
        if self.manifest is None:
            choose_message = escape(self._tr("Choose a reviewed model manifest to begin."))
            validation_message = escape(
                self._tr("The page validates declared inputs before any model executes.")
            )
            self.resource_status.setText(
                f"<b>{escape(self._tr('No model configured'))}</b>"
                f"&nbsp;&nbsp;·&nbsp;&nbsp;{choose_message}"
                f'<br><span style="color:#667085;">'
                f"{validation_message}"
                f"</span>"
            )
            return

        missing = self._missing_weight_specs()
        if self._plugin_ready and not missing:
            loaded_detail = escape(
                self._tr("Loaded from local paths; no files were downloaded or copied.")
            )
            self.resource_status.setText(
                f"<b>{escape(self._tr('Model ready'))}</b>&nbsp;&nbsp;·&nbsp;&nbsp;{loaded_detail}"
            )
            return
        if not missing:
            self.resource_status.setText(
                f"<b>{escape(self._tr('Model manifest ready'))}</b>"
                f"&nbsp;&nbsp;·&nbsp;&nbsp;"
                f"{escape(self._tr('All referenced local weight files are available.'))}"
            )
            return

        names = ", ".join(Path(weight.path).name for weight in missing)
        repository_url = self._manifest_github_url()
        detail = self._tr("Download it, then configure its local path as described by the plugin.")
        action = (
            self._resource_link(repository_url, self._tr("Open official guide"))
            if repository_url is not None
            else escape(self._tr("No GitHub source is declared in this manifest."))
        )
        self.resource_status.setText(
            f"<b>{escape(self._tr('Weight not found: {name}').format(name=names))}</b> "
            f"{escape(detail)}&nbsp;&nbsp;{action}"
        )

    def set_source_available(self, image: object | None) -> None:
        self._source_available = image is not None
        self._last_input = None
        self._last_input_context = {}
        self._display_transform = None
        self._source_grayscale_mapping = None
        self._source_scale_unit = ""
        self.preview_override.blockSignals(True)
        self.preview_override.setChecked(False)
        self.preview_override.blockSignals(False)
        self.model_input_view.clear_image()
        self._reset_artifact_views()
        self._update_capability_gate()

    def _translate_owned_summary(self, text: str) -> str:
        chinese_intro = translate(MODEL_SUMMARY_INTRO, "zh_CN")
        if self._language == "zh_CN":
            text = text.replace(MODEL_SUMMARY_INTRO, chinese_intro)
        else:
            text = text.replace(chinese_intro, MODEL_SUMMARY_INTRO)

        field_labels = (
            "Name",
            "Version",
            "Family/source",
            "Repository",
            "Tasks",
            "Task",
            "Runtime",
            "License",
            "Input resolution",
            "Weights",
            "Capabilities",
            "Prediction result",
            "Type",
            "Duration",
            "Visualizations",
        )
        translated_lines: list[str] = []
        for line in text.splitlines():
            indentation = line[: len(line) - len(line.lstrip())]
            content = line[len(indentation) :]
            translated = translate(content, self._language)
            if translated == content:
                for label in field_labels:
                    english_prefix = f"{label}:"
                    chinese_prefix = f"{translate(label, 'zh_CN')}:"
                    source_prefix, target_prefix = (
                        (english_prefix, chinese_prefix)
                        if self._language == "zh_CN"
                        else (chinese_prefix, english_prefix)
                    )
                    if content.startswith(source_prefix):
                        value = content[len(source_prefix) :]
                        leading_space = value[: len(value) - len(value.lstrip())]
                        canonical_value = translate(value.lstrip(), "en")
                        translated_value = (
                            translate(canonical_value, self._language)
                            if canonical_value in {"not reported", "none"}
                            else value.lstrip()
                        )
                        translated = target_prefix + leading_space + translated_value
                        break
            translated_lines.append(indentation + translated)
        return "\n".join(translated_lines)

    def _render_summary(self) -> None:
        rendered = "".join(
            self._translate_owned_summary(text) if kind == "owned" else text
            for kind, text in self._summary_segments
        )
        if rendered != self.summary.toPlainText():
            self.summary.setPlainText(rendered)
        self._rendered_summary_text = rendered

    def _replace_summary(self, segments: list[tuple[str, str]]) -> None:
        self._summary_segments = segments
        self._render_summary()

    def _append_summary(self, kind: str, text: str) -> None:
        self._summary_segments.append((kind, text))
        self._render_summary()

    def _open_manifest(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            self._tr("Open model manifest"),
            "",
            self._tr("YAML (*.yaml *.yml)"),
        )
        if not path:
            return
        try:
            manifest = self.service.inspect_manifest(path)
        except BaseException as exc:
            QMessageBox.critical(self, self._tr("Manifest validation failed"), str(exc))
            return

        def field(label: str, value: object) -> str:
            return f"{label}: {value}"

        source_label = manifest.source.name
        if manifest.source.model_id:
            source_label += f" / {manifest.source.model_id}"
        license_label = (
            f"code {manifest.license.code}; model {manifest.license.model}; "
            f"weights {manifest.license.weights}"
        )
        manifest_root = Path(path).parent
        weight_lines = [
            f"  {Path(weight.path).name}: "
            f"{'available' if (manifest_root / weight.path).is_file() else 'missing'}"
            for weight in manifest.weights
        ]
        lines = [
            field("Name", manifest.name),
            field("Version", manifest.version),
            field("Family/source", f"{manifest.family} / {source_label}"),
            field("Tasks", ", ".join(item.value for item in manifest.tasks)),
            field("Runtime", manifest.runtime.kind.value),
            field("License", license_label),
            f"Inputs: {len(manifest.inputs)}; outputs: {len(manifest.outputs)}",
            "Weights:",
            *weight_lines,
            field("Capabilities", manifest.capabilities),
        ]
        if manifest.source.repository:
            lines.insert(3, field("Repository", manifest.source.repository))
        segments = [("owned", "\n".join(lines))]
        if self.service.requires_python_consent:
            segments.extend(
                [
                    ("owned", "\n\nSECURITY WARNING:\n"),
                    ("external", self.service.python_adapter_warning),
                ]
            )
        self._close_plugin()
        self._last_input = None
        self._display_transform = None
        self._source_grayscale_mapping = None
        self._source_scale_unit = ""
        self.model_input_view.clear_image()
        self._reset_artifact_views()
        self._replace_summary(segments)
        self.manifest = manifest
        self.run_button.setEnabled(False)
        self.cancel_button.setEnabled(False)
        self._refresh_model_resource_status()
        self._update_capability_gate()

    def _load_plugin(self) -> None:
        if self.manifest is None:
            return
        consent = False
        if self.service.requires_python_consent:
            choice = QMessageBox.warning(
                self,
                self._tr("Execute user-supplied Python code?"),
                self.service.python_adapter_warning
                + "\n\n"
                + self._tr(
                    "A subprocess contains crashes and dependencies but is not a security "
                    "sandbox. Continue only after reviewing the adapter source."
                ),
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if choice != QMessageBox.Yes:
                return
            consent = True
        self._close_plugin()
        self._operation_stage = "load"

        def operation(task_context: Any) -> Any:
            task_context.report_progress(0.1, message="Validating external references")
            loaded = self.service.load(
                user_consented_python_code=consent,
                cancellation_check=lambda: task_context.cancelled,
            )
            task_context.raise_if_cancelled()
            task_context.report_progress(1.0, message="External model ready")
            return loaded

        self.progress.setVisible(True)
        self.progress.setValue(0)
        self.manifest_button.setEnabled(False)
        self.load_button.setEnabled(False)
        self.cancel_button.setEnabled(True)
        task = self.runner.submit(operation)
        self._active_task = task
        self._update_capability_gate()
        self.watcher.watch(
            task,
            success=lambda loaded, handle=task: self._plugin_loaded_if_current(handle, loaded),
            error=lambda error, handle=task: self._plugin_failed_if_current(handle, error),
            progress=self._plugin_progress,
        )

    def _plugin_loaded_if_current(self, task: BackgroundTask[Any], plugin: Any) -> None:
        if task is not self._active_task:
            return
        self._active_task = None
        self._plugin_loaded(plugin)

    def _plugin_loaded(self, _manifest: ModelManifest) -> None:
        self._plugin_ready = self.service.ready
        self._operation_stage = None
        self.progress.setVisible(False)
        self.manifest_button.setEnabled(True)
        self.load_button.setEnabled(True)
        self.cancel_button.setEnabled(False)
        self._refresh_model_resource_status()
        self._update_capability_gate()
        self._append_summary(
            "owned",
            "\n\nModel loaded from its existing external path; no files were downloaded or copied.",
        )

    def _confirm_preview_inference(self, context: Mapping[str, Any]) -> bool:
        """Require a final, model-specific confirmation for reduced-resolution input."""

        if not bool(context.get("preview_only", False)):
            return True
        source_shape = tuple(context.get("source_shape", ()))
        display_shape = tuple(context.get("display_shape", ()))
        dialog = QMessageBox(self)
        dialog.setIcon(QMessageBox.Warning)
        dialog.setWindowTitle(self._tr("Run on preview-only pixels?"))
        dialog.setText(
            self._tr(
                "Full-resolution pixels are not retained in this session. A result from the "
                "bounded preview must not be described as full-resolution inference."
            )
        )
        dialog.setInformativeText(
            self._tr(
                "This teaching override is recorded with the model input context. Cancel is "
                "the safe default."
            )
        )
        model_name = "—" if self.manifest is None else self.manifest.name
        dialog.setDetailedText(
            "\n".join(
                (
                    f"Model: {model_name}",
                    f"Original shape: {source_shape}",
                    f"Preview shape: {display_shape}",
                    "Input resolution: PREVIEW ONLY",
                    "Override provenance: explicit-final-confirmation",
                )
            )
        )
        run_button = dialog.addButton(
            self._tr("Run preview-only teaching inference"),
            QMessageBox.AcceptRole,
        )
        cancel_button = dialog.addButton(QMessageBox.Cancel)
        cancel_button.setDefault(True)
        dialog.exec_()
        return dialog.clickedButton() is run_button

    def _run_plugin(self) -> None:
        if not self.service.ready or self.manifest is None:
            return
        ready, message, _blocked = self._capability_gate()
        if not ready:
            QMessageBox.warning(
                self,
                self._tr("Model input is not compatible"),
                self._tr(message),
            )
            return
        try:
            input_context = dict(self.input_context())
            if not self._confirm_preview_inference(input_context):
                return
            if bool(input_context.get("preview_only", False)):
                input_context["preview_override_confirmed"] = True
                input_context["input_resolution"] = "preview-only"
            model_input = self.source()
        except BaseException as exc:
            QMessageBox.warning(self, self._tr("No model input"), str(exc))
            return
        self._last_input_context = input_context
        if isinstance(model_input, RasterImage2D):
            self._last_input = np.asarray(model_input.array)
            self._display_transform = model_input.transform_record
        else:
            self._last_input = np.asarray(model_input)
            self._display_transform = None
        source_view = self._overlay_source(self._last_input)
        self._configure_source_display(source_view, input_context)
        self._set_source_view(self.model_input_view, source_view, fit=True)
        self._set_source_view(self.model_output_view, source_view, fit=True)
        input_name = self.manifest.inputs[0].name

        def operation(task_context: Any) -> Any:
            task_context.report_progress(0.05, message="Preparing declared model input")
            result = self.service.predict(
                {input_name: model_input},
                parameters={"input_context": {input_name: input_context}},
                cancellation_check=lambda: task_context.cancelled,
            )
            task_context.raise_if_cancelled()
            task_context.report_progress(1.0, message="Inference complete")
            return result

        self.progress.setVisible(True)
        self.progress.setValue(0)
        self.manifest_button.setEnabled(False)
        self.load_button.setEnabled(False)
        self.run_button.setEnabled(False)
        self.cancel_button.setEnabled(True)
        self._operation_stage = "predict"
        task = self.runner.submit(operation)
        self._active_task = task
        self._update_capability_gate()
        self.watcher.watch(
            task,
            success=lambda result, handle=task: self._prediction_ready_if_current(handle, result),
            error=lambda error, handle=task: self._plugin_failed_if_current(handle, error),
            progress=self._plugin_progress,
        )

    def _prediction_ready_if_current(self, task: BackgroundTask[Any], result: Any) -> None:
        if task is not self._active_task:
            return
        self._active_task = None
        self._prediction_ready(result)

    def _prediction_ready(self, result: InferenceResult) -> None:
        self.progress.setVisible(False)
        self.manifest_button.setEnabled(True)
        self.load_button.setEnabled(True)
        self.cancel_button.setEnabled(False)
        self._operation_stage = None
        self._update_capability_gate()
        artifacts = self.service.visualize(result)
        self._render_artifacts(artifacts)
        duration = result.provenance.duration_ms
        duration_text = "not reported" if duration is None else f"{duration:.3f} ms"
        resolution_text = (
            "preview-only (explicit override recorded)"
            if bool(self._last_input_context.get("preview_only", False))
            else "full session resolution"
        )
        self._append_summary(
            "owned",
            "\n\nPrediction result:\n"
            f"  Type: {type(result).__name__}\n"
            f"  Task: {result.task.value}\n"
            f"  Runtime: {result.provenance.runtime.value}\n"
            f"  Duration: {duration_text}\n"
            f"  Input resolution: {resolution_text}\n"
            f"  Visualizations: {', '.join(item.kind.value for item in artifacts) or 'none'}",
        )

    @staticmethod
    def _overlay_source(array: np.ndarray | None) -> np.ndarray:
        if array is None:
            raise ViewerError("No source image is available for a source-aligned overlay.")
        values = np.asarray(array)
        if values.ndim == 4 and values.shape[0] == 1:
            values = values[0]
        if (
            values.ndim == 3
            and values.shape[0] in {1, 3, 4}
            and values.shape[-1]
            not in {
                3,
                4,
            }
        ):
            values = np.moveaxis(values, 0, -1)
            if values.shape[2] == 1:
                values = values[:, :, 0]
        if values.ndim not in {2, 3}:
            raise ViewerError(f"Cannot display model source with shape {values.shape}.")
        return values

    def _configure_source_display(
        self,
        source: np.ndarray,
        context: Mapping[str, Any] | None = None,
    ) -> None:
        """Freeze one source-derived display window for all aligned comparisons."""

        values = np.asarray(source)
        self._source_grayscale_mapping = (
            GrayscaleDisplayMapping.from_percentiles(values) if values.ndim == 2 else None
        )
        raw_unit = "" if context is None else str(context.get("unit", "")).strip()
        self._source_scale_unit = raw_unit if len(raw_unit) <= 24 else ""

    def _set_source_view(self, view: ImageView, source: np.ndarray, *, fit: bool) -> None:
        values = np.asarray(source)
        if values.ndim == 2:
            if self._source_grayscale_mapping is None:
                self._configure_source_display(values, self._last_input_context)
            mapping = self._source_grayscale_mapping
            assert mapping is not None
            view.set_array(values, grayscale_mapping=mapping, fit=fit)
            view.set_value_scale(mapping.lower, mapping.upper, unit=self._source_scale_unit)
            return
        view.set_array(values, fit=fit)

    @staticmethod
    def _artifact_display_range(
        artifact: Any,
        values: np.ndarray,
    ) -> tuple[float, float, str]:
        """Resolve a stable numeric range without independently percentile-stretching output."""

        metadata = getattr(artifact, "metadata", {})
        declared = metadata.get("value_range") if isinstance(metadata, Mapping) else None
        if declared is None and isinstance(metadata, Mapping):
            declared = (metadata.get("vmin"), metadata.get("vmax"))
        try:
            low, high = (float(item) for item in declared)
        except (TypeError, ValueError, OverflowError):
            finite = np.asarray(values)[np.isfinite(values)]
            if finite.size == 0:
                low, high = 0.0, 1.0
            else:
                low = float(np.min(finite))
                high = float(np.max(finite))
                if low >= 0.0 and high <= 1.0:
                    low, high = 0.0, 1.0
                elif high <= low:
                    padding = max(abs(low) * 0.01, 0.5)
                    low, high = low - padding, high + padding
        if not np.isfinite(low) or not np.isfinite(high) or high <= low:
            raise ViewerError("Artifact display range must contain two finite increasing values.")
        raw_unit = str(metadata.get("unit", "")).strip() if isinstance(metadata, Mapping) else ""
        unit = raw_unit if len(raw_unit) <= 24 else ""
        return low, high, unit

    @staticmethod
    def _source_aligned_mask(payload: Any, source_shape: tuple[int, int]) -> np.ndarray:
        mask = np.asarray(payload)
        mask = np.squeeze(mask)
        if mask.ndim == 3:
            if mask.shape[1:] == source_shape:
                mask = np.argmax(mask, axis=0)
            elif mask.shape[:2] == source_shape:
                mask = np.argmax(mask, axis=2)
        if mask.shape != source_shape:
            raise ViewerError(
                f"Source-aligned mask shape {mask.shape} does not match {source_shape}."
            )
        return mask

    def _source_array_to_display(self, payload: Any, *, discrete: bool) -> np.ndarray:
        transform = self._display_transform
        values = np.asarray(payload)
        if transform is None or (
            transform.original_shape == transform.output_shape
            and np.allclose(transform.matrix, np.eye(3))
        ):
            return values
        # ``inverse_map_spatial_array`` maps a transform's output grid back to
        # its original grid.  Reversing the source-to-display EXIF transform
        # therefore maps source-aligned model evidence onto the canonical view.
        display_to_source = TransformRecord(
            np.linalg.inv(transform.matrix),
            transform.output_shape,
            transform.original_shape,
        )
        return inverse_map_spatial_array(
            values,
            display_to_source,
            discrete=discrete,
        )

    def _source_points_to_display(self, points: np.ndarray) -> np.ndarray:
        if self._display_transform is None:
            return points
        return self._display_transform.forward(points)

    def _source_boxes_to_display(self, boxes: np.ndarray) -> np.ndarray:
        if self._display_transform is None:
            return boxes
        return self._display_transform.forward_boxes(boxes)

    def _reset_artifact_views(self) -> None:
        while self.model_output_tabs.count() > 1:
            widget = self.model_output_tabs.widget(1)
            self.model_output_tabs.removeTab(1)
            widget.deleteLater()
        self.model_output_tabs.setTabText(0, self._tr("Visualization"))
        self.model_output_view.clear_image()

    def _artifact_view(self, title: str, index: int) -> ImageView:
        if index == 0:
            view = self.model_output_view
            self.model_output_tabs.setTabText(0, self._tr(title))
        else:
            view = ImageView(title)
            view.set_language(self._language)
            self.model_output_tabs.addTab(view, self._tr(title))
        view.set_title(title)
        return view

    def _render_artifacts(self, artifacts: Any) -> None:
        """Render every declarative artifact in source-comparison tabs."""

        self._reset_artifact_views()
        visual_index = 0
        for artifact in artifacts:
            kind = artifact.kind
            if kind is VisualizationKind.MASK_OVERLAY:
                source = self._overlay_source(self._last_input)
                view = self._artifact_view(artifact.title, visual_index)
                visual_index += 1
                self._set_source_view(view, source, fit=True)
                payload = artifact.payload
                if artifact.coordinate_system == "source-pixel":
                    payload = self._source_array_to_display(payload, discrete=True)
                mask = self._source_aligned_mask(payload, source.shape[:2])
                view.set_mask_overlay(mask)
                continue
            if kind is VisualizationKind.BOX_OVERLAY:
                source = self._overlay_source(self._last_input)
                view = self._artifact_view(artifact.title, visual_index)
                visual_index += 1
                self._set_source_view(view, source, fit=True)
                boxes = [
                    item for item in artifact.payload if len(getattr(item, "coordinates", ())) == 4
                ]
                coordinates = np.asarray([item.coordinates for item in boxes], dtype=float)
                if artifact.coordinate_system == "source-pixel" and len(coordinates):
                    coordinates = self._source_boxes_to_display(coordinates)
                view.set_box_overlays(
                    coordinates,
                    labels=[item.label or "" for item in boxes],
                )
                continue
            if kind is VisualizationKind.KEYPOINT_OVERLAY:
                source = self._overlay_source(self._last_input)
                view = self._artifact_view(artifact.title, visual_index)
                visual_index += 1
                self._set_source_view(view, source, fit=True)
                point_items = [
                    item
                    for item in artifact.payload
                    if len(getattr(item, "coordinates", item)) == 2
                ]
                points = np.asarray(
                    [getattr(item, "coordinates", item) for item in point_items],
                    dtype=float,
                )
                if artifact.coordinate_system == "source-pixel" and len(points):
                    points = self._source_points_to_display(points)
                view.set_point_overlays(
                    points,
                    labels=[getattr(item, "label", "") or "" for item in point_items],
                )
                continue
            if kind in {VisualizationKind.IMAGE, VisualizationKind.HEATMAP}:
                values = np.squeeze(np.asarray(artifact.payload))
                if artifact.coordinate_system == "source-pixel" and values.ndim >= 2:
                    values = self._source_array_to_display(values, discrete=False)
                if values.ndim in {2, 3}:
                    view = self._artifact_view(artifact.title, visual_index)
                    visual_index += 1
                    source = self._overlay_source(self._last_input)
                    if (
                        kind is VisualizationKind.IMAGE
                        and values.ndim == 2
                        and source.ndim == 2
                        and values.shape == source.shape
                    ):
                        if self._source_grayscale_mapping is None:
                            self._configure_source_display(source, self._last_input_context)
                        mapping = self._source_grayscale_mapping
                        assert mapping is not None
                        view.set_array(values, grayscale_mapping=mapping, fit=True)
                        view.set_value_scale(
                            mapping.lower,
                            mapping.upper,
                            unit=self._source_scale_unit,
                        )
                    elif values.ndim == 2:
                        low, high, unit = self._artifact_display_range(artifact, values)
                        view.set_array(
                            values,
                            grayscale_mapping=GrayscaleDisplayMapping(low, high),
                            fit=True,
                        )
                        view.set_value_scale(low, high, unit=unit)
                    else:
                        view.set_array(values, fit=True)
                    continue
            if kind is VisualizationKind.TEXT:
                self._append_summary(
                    "external",
                    f"\n\n{artifact.title}:\n{artifact.payload}",
                )
            elif kind in {VisualizationKind.TABLE, VisualizationKind.PLOT}:
                self._append_summary(
                    "external",
                    f"\n\n{artifact.title}: {artifact.payload}",
                )
        if visual_index:
            self.model_output_tabs.setCurrentIndex(0)

    def _plugin_progress(self, fraction: float, message: str) -> None:
        self.progress.setValue(int(round(fraction * 100)))
        self.progress.setFormat(f"{self._tr(message)} – %p%")

    def _plugin_failed(self, error: BaseException) -> None:
        if self._operation_stage == "load":
            self._close_plugin()
        self._operation_stage = None
        self.progress.setVisible(False)
        self.manifest_button.setEnabled(True)
        self.load_button.setEnabled(self.manifest is not None)
        self.cancel_button.setEnabled(False)
        self._plugin_ready = self.service.ready
        self._refresh_model_resource_status()
        self._update_capability_gate()
        if error.__class__.__name__ == "OperationCancelled":
            return
        if self._missing_weight_specs():
            dialog = QMessageBox(self)
            dialog.setIcon(QMessageBox.Warning)
            dialog.setWindowTitle(self._tr("Model weight not found"))
            dialog.setText(self._tr("One or more required local weight files are missing."))
            dialog.setInformativeText(
                self._tr("Review the model resource card and manifest paths, then retry.")
            )
            dialog.setDetailedText(str(error))
            dialog.exec_()
            return
        QMessageBox.critical(
            self,
            self._tr("External model failed safely"),
            str(error),
        )

    def _plugin_failed_if_current(
        self,
        task: BackgroundTask[Any],
        error: BaseException,
    ) -> None:
        if task is not self._active_task:
            return
        self._active_task = None
        self._plugin_failed(error)

    def _cancel(self) -> None:
        if self._active_task is not None:
            self._active_task.cancel()
        self.service.cancel()
        self._plugin_ready = self.service.ready
        self.cancel_button.setEnabled(False)
        self._update_capability_gate()
        if self.service.reload_required:
            self._append_summary(
                "owned",
                "\n\nThe cancelled Python adapter must be explicitly loaded again.",
            )

    def _close_plugin(self) -> None:
        self.service.unload()
        self._plugin_ready = False

    def close(self) -> None:  # type: ignore[override]
        self.watcher.stop()
        self.bundled_models.shutdown()
        self._cancel()
        self.service.close()
        self.runner.shutdown(wait=False, cancel_pending=True)
        super().close()


class _ScaledTransferPreview(QLabel):
    """Keep the exact reviewed pixmap legible without distorting its aspect ratio."""

    def __init__(self, pixmap: QPixmap, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._source_pixmap = pixmap
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(180, 150)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setAccessibleName("Exact outbound PNG preview")
        self._update_scaled_pixmap()

    def resizeEvent(self, event: Any) -> None:  # noqa: N802 - Qt API
        super().resizeEvent(event)
        self._update_scaled_pixmap()

    def _update_scaled_pixmap(self) -> None:
        available = self.contentsRect().size()
        if available.width() <= 0 or available.height() <= 0:
            return
        self.setPixmap(
            self._source_pixmap.scaled(
                available,
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation,
            )
        )


class _ImageTransferConfirmationDialog(QDialog):
    """Scrollable, plain-text final review for one exact outbound PNG request."""

    def __init__(
        self,
        *,
        preview: RenderedPreview,
        preview_pixmap: QPixmap,
        details: str,
        translator: Callable[[str], str],
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._details_text = details
        self._preview_pixmap = preview_pixmap
        self.setModal(True)
        self.setWindowTitle(translator("Confirm one-time image transfer"))
        self.setAccessibleName(translator("One-time cloud image transfer review"))
        self.setMinimumSize(480, 340)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 14)
        layout.setSpacing(12)
        title = QLabel(translator("Review the exact outbound request"))
        title.setObjectName("dialogTitle")
        title.setWordWrap(True)
        layout.addWidget(title)

        review_splitter = QSplitter(Qt.Horizontal)
        review_splitter.setChildrenCollapsible(False)
        preview_label = _ScaledTransferPreview(preview_pixmap)
        preview_label.setAccessibleName(translator("Exact outbound PNG preview"))
        preview_label.setAccessibleDescription(
            translator(
                "Canonical PNG, {width} by {height} pixels, {bytes} bytes, SHA-256 {sha}."
            ).format(
                width=preview.width,
                height=preview.height,
                bytes=len(preview.data),
                sha=preview.sha256,
            )
        )
        review_splitter.addWidget(preview_label)

        details_view = QPlainTextEdit()
        details_view.setObjectName("transferPlanDetails")
        details_view.setReadOnly(True)
        details_view.setLineWrapMode(QPlainTextEdit.WidgetWidth)
        details_view.setPlainText(details)
        details_view.setAccessibleName(translator("Exact outbound transfer plan"))
        review_splitter.addWidget(details_view)
        review_splitter.setStretchFactor(0, 1)
        review_splitter.setStretchFactor(1, 2)
        review_splitter.setSizes([260, 440])
        layout.addWidget(review_splitter, 1)

        self._buttons = QDialogButtonBox(QDialogButtonBox.Yes | QDialogButtonBox.No)
        self._buttons.button(QDialogButtonBox.Yes).setText(translator("Authorize once and send"))
        self._buttons.button(QDialogButtonBox.No).setText(translator("Cancel"))
        self._buttons.button(QDialogButtonBox.No).setDefault(True)
        self._buttons.button(QDialogButtonBox.No).setFocus(Qt.OtherFocusReason)
        self._buttons.button(QDialogButtonBox.Yes).clicked.connect(
            lambda: self.done(QMessageBox.Yes)
        )
        self._buttons.button(QDialogButtonBox.No).clicked.connect(lambda: self.done(QMessageBox.No))
        layout.addWidget(self._buttons)

        screen = QApplication.primaryScreen()
        if screen is not None:
            available = screen.availableGeometry()
            self.resize(
                min(760, max(480, available.width() - 80)),
                min(620, max(340, available.height() - 80)),
            )
        else:
            self.resize(720, 560)

    # Compatibility helpers keep focused tests independent of internal widget lookup.
    def informativeText(self) -> str:  # noqa: N802 - QMessageBox-compatible API
        return self._details_text

    def iconPixmap(self) -> QPixmap:  # noqa: N802 - QMessageBox-compatible API
        return self._preview_pixmap

    def textFormat(self) -> Qt.TextFormat:  # noqa: N802 - QMessageBox-compatible API
        return Qt.PlainText

    def button(self, which: QMessageBox.StandardButton) -> QAbstractButton | None:
        return self._buttons.button(QDialogButtonBox.StandardButton(int(which)))

    def showEvent(self, event: Any) -> None:  # noqa: N802 - Qt API
        super().showEvent(event)
        authorize = self._buttons.button(QDialogButtonBox.Yes)
        cancel = self._buttons.button(QDialogButtonBox.No)
        authorize.setDefault(False)
        authorize.setAutoDefault(False)
        cancel.setAutoDefault(True)
        cancel.setDefault(True)
        cancel.setFocus(Qt.OtherFocusReason)


class AssistantPage(_BilingualPage, QWidget):
    statusChanged = pyqtSignal(str)

    def __init__(
        self,
        preview_source: Callable[[], bytes],
        service: TeachingAssistantService | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.preview_source = preview_source
        self.service = service or TeachingAssistantService()
        self.runner = TaskRunner(max_workers=1, thread_name_prefix="openmedvisionx-llm")
        self.watcher = TaskWatcher(self)
        self._active_task: BackgroundTask[Any] | None = None
        self._provider_model_ids: dict[str, str] = {}
        self._current_provider_name = ""
        self._answer_status_source: str | None = None
        self._last_prompt = ""
        self._source_available = True
        self._settings_visible = True
        self._has_answer = False
        self._request_status_source = "Markdown supported · Responses are for learning only."
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 8, 0, 0)
        layout.setSpacing(10)
        disclaimer = QLabel(
            "For learning only — not medical advice or a clinical decision tool. "
            "Credentials stay in your environment or system keyring; an explicit none "
            "reference is allowed only for a loopback endpoint."
        )
        disclaimer.setWordWrap(True)
        disclaimer.setObjectName("warningBanner")
        layout.addWidget(disclaimer)

        self.workspace_splitter = QSplitter(Qt.Horizontal)
        self.settings_scroll = QScrollArea()
        self.settings_scroll.setWidgetResizable(True)
        self.settings_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.settings_scroll.setMinimumWidth(250)
        self.settings_scroll.setAccessibleName("Assistant provider setup")
        settings = QWidget()
        settings_layout = QVBoxLayout(settings)
        settings_layout.setContentsMargins(0, 0, 8, 0)
        self.config_group = QGroupBox("Provider configuration")
        form = QFormLayout(self.config_group)
        # Stacked labels keep long endpoints and credential references readable in the
        # narrow setup pane; lower options remain reachable through the explicit scroll area.
        form.setRowWrapPolicy(QFormLayout.WrapAllRows)
        form.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
        self.provider = QComboBox()
        self.provider.setAccessibleName("AI provider")
        self.provider.addItems(list(self.service.provider_names))
        self.provider.setProperty(_I18N_SKIP_ITEMS_PROPERTY, True)
        self.provider.currentTextChanged.connect(self._provider_changed)
        self.model = QLineEdit()
        self.model.setAccessibleName("Provider model ID")
        self.model.setPlaceholderText("User-supplied model ID (never hard-coded)")
        self.endpoint = QLineEdit()
        self.endpoint.setAccessibleName("Provider endpoint")
        self.endpoint.textChanged.connect(self._update_cloud_status)
        self.endpoint.textChanged.connect(self._update_provider_field_tooltips)
        self.credential = QLineEdit()
        self.credential.setAccessibleName("Credential environment or keyring reference")
        self.credential.textChanged.connect(self._update_provider_field_tooltips)
        self.network_enabled = QCheckBox("Enable network")
        self.network_enabled.stateChanged.connect(self._update_cloud_status)
        self.vision = QCheckBox("Vision input")
        self.vision.stateChanged.connect(self._vision_changed)
        self.send_preview = QCheckBox("Attach active rendered plane")
        self.send_preview.setChecked(False)
        self.send_preview.stateChanged.connect(self._update_cloud_status)
        self.cloud_status = QLabel("Cloud image transfer: OFF")
        self.cloud_status.setObjectName("cloudStatus")
        self.cloud_status.setWordWrap(True)
        self.cloud_status.setAccessibleName("Cloud image transfer status")
        _add_accessible_form_row(form, "Provider", self.provider)
        _add_accessible_form_row(form, "Model ID", self.model)
        _add_accessible_form_row(
            form,
            "Endpoint",
            self.endpoint,
            description="Exact HTTPS provider endpoint used for the request.",
        )
        _add_accessible_form_row(
            form,
            "Credential reference",
            self.credential,
            description=(
                "Use env:NAME or keyring:service/user; use none only for a localhost or "
                "loopback endpoint. Never enter a secret here."
            ),
        )
        form.addRow(self.network_enabled)
        form.addRow(self.vision)
        form.addRow(self.send_preview)
        form.addRow(self.cloud_status)
        settings_layout.addWidget(self.config_group)
        privacy = QLabel(
            "Image sharing sends exactly the PNG shown in the final review. In the viewer this "
            "is the complete active 2-D plane after display mapping; viewport zoom/pan, overlays, "
            "and annotations are excluded. Inspect pixels for burned-in private text. Consent is "
            "one-time and bound to the exact provider, endpoint, model, prompt, task, and bytes."
        )
        privacy.setWordWrap(True)
        privacy.setObjectName("mutedText")
        settings_layout.addWidget(privacy)
        settings_layout.addStretch(1)
        self.settings_scroll.setWidget(settings)
        self.workspace_splitter.addWidget(self.settings_scroll)

        conversation = QFrame()
        conversation.setObjectName("conversationSurface")
        conversation_layout = QVBoxLayout(conversation)
        conversation_layout.setContentsMargins(16, 16, 16, 16)
        conversation_layout.setSpacing(12)
        conversation_header = QHBoxLayout()
        heading = QVBoxLayout()
        heading.setSpacing(2)
        conversation_title = QLabel("Ask the assistant")
        conversation_title.setObjectName("pageTitle")
        self.request_status = QLabel(self._request_status_source)
        self.request_status.setObjectName("mutedText")
        self.request_status.setWordWrap(True)
        heading.addWidget(conversation_title)
        heading.addWidget(self.request_status)
        conversation_header.addLayout(heading, 1)
        self.settings_toggle = QPushButton("Hide setup")
        self.settings_toggle.setObjectName("linkButton")
        self.settings_toggle.setAccessibleName("Show or hide assistant provider setup")
        self.settings_toggle.clicked.connect(self._toggle_settings)
        conversation_header.addWidget(self.settings_toggle, 0, Qt.AlignTop)
        conversation_layout.addLayout(conversation_header)

        self.question_context = QLabel()
        self.question_context.setObjectName("questionContext")
        self.question_context.setWordWrap(True)
        self.question_context.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.question_context.setVisible(False)
        self.question_context.setProperty(_I18N_SKIP_TEXT_PROPERTY, True)
        conversation_layout.addWidget(self.question_context)

        self.response_stack = QStackedWidget()
        empty_state = QWidget()
        empty_layout = QVBoxLayout(empty_state)
        empty_layout.setContentsMargins(28, 28, 28, 28)
        empty_layout.addStretch(1)
        self.empty_title = QLabel("Ready when you are")
        self.empty_title.setObjectName("emptyStateTitle")
        self.empty_title.setAlignment(Qt.AlignCenter)
        self.empty_body = QLabel(
            "Try: “Explain windowing”, “Why does FBP use a ramp filter?”, or “Help me inspect "
            "this result.” Configure a provider only when you are ready to make a network request."
        )
        self.empty_body.setObjectName("mutedText")
        self.empty_body.setAlignment(Qt.AlignCenter)
        self.empty_body.setWordWrap(True)
        empty_layout.addWidget(self.empty_title)
        empty_layout.addWidget(self.empty_body)
        empty_layout.addStretch(1)
        self.response_stack.addWidget(empty_state)

        answer_page = QWidget()
        answer_layout = QVBoxLayout(answer_page)
        answer_layout.setContentsMargins(0, 0, 0, 0)
        answer_layout.setSpacing(8)
        self.answer_metadata = QLabel()
        self.answer_metadata.setObjectName("answerMetadata")
        self.answer_metadata.setWordWrap(True)
        self.answer_metadata.setVisible(False)
        self.answer_metadata.setProperty(_I18N_SKIP_TEXT_PROPERTY, True)
        answer_layout.addWidget(self.answer_metadata)
        self.answer = SafeMarkdownBrowser()
        self.answer.setObjectName("markdownAnswer")
        self.answer.setAccessibleName("Latest assistant response in Markdown")
        self.answer.setPlaceholderText("Assistant responses appear here.")
        self.answer.anchorClicked.connect(self._open_response_link)
        answer_layout.addWidget(self.answer, 1)
        self.answer_disclaimer = QLabel()
        self.answer_disclaimer.setObjectName("answerDisclaimer")
        self.answer_disclaimer.setWordWrap(True)
        self.answer_disclaimer.setVisible(False)
        self.answer_disclaimer.setProperty(_I18N_SKIP_TEXT_PROPERTY, True)
        answer_layout.addWidget(self.answer_disclaimer)
        self.response_stack.addWidget(answer_page)
        conversation_layout.addWidget(self.response_stack, 1)

        self.composer = QFrame()
        self.composer.setObjectName("composerSurface")
        composer_layout = QVBoxLayout(self.composer)
        composer_layout.setContentsMargins(12, 10, 12, 10)
        composer_layout.setSpacing(7)
        self.prompt = SubmitPlainTextEdit()
        self.prompt.setObjectName("promptEditor")
        self.prompt.setAccessibleName("Question for the AI teaching assistant")
        self.prompt.setAccessibleDescription(
            "Multi-line question. Press Control or Command plus Enter to send."
        )
        self.prompt.setPlaceholderText(
            "Ask about concepts, parameters, reconstruction results, or learning guidance…"
        )
        line_height = max(self.prompt.fontMetrics().lineSpacing(), 16)
        self.prompt.setMinimumHeight(line_height * 3 + 20)
        self.prompt.setMaximumHeight(line_height * 7 + 24)
        composer_layout.addWidget(self.prompt)
        prompt_help = QHBoxLayout()
        self.shortcut_hint = QLabel("Ctrl/Cmd+Enter to send")
        self.shortcut_hint.setObjectName("keyboardHint")
        self.shortcut_hint.setWordWrap(True)
        self.shortcut_hint.setMinimumWidth(0)
        self.shortcut_hint.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        self.character_count = QLabel("0 characters")
        self.character_count.setObjectName("keyboardHint")
        prompt_help.addWidget(self.shortcut_hint)
        prompt_help.addStretch(1)
        prompt_help.addWidget(self.character_count)
        composer_layout.addLayout(prompt_help)
        self.send_button = QPushButton("Send")
        self.send_button.setObjectName("primary")
        self.send_button.clicked.connect(self._send)
        self.send_button.setEnabled(False)
        self.send_button.setToolTip("Send to the configured provider (Ctrl/Cmd+Enter)")
        self.cancel_request_button = QPushButton("Cancel request")
        self.cancel_request_button.clicked.connect(self._cancel_request)
        self.cancel_request_button.setEnabled(False)
        self.progress = QProgressBar()
        self.progress.setRange(0, 0)
        self.progress.setVisible(False)
        request_buttons = QHBoxLayout()
        request_buttons.addWidget(self.send_button)
        request_buttons.addWidget(self.cancel_request_button)
        request_buttons.addStretch(1)
        composer_layout.addLayout(request_buttons)
        self.send_reason = QLabel()
        self.send_reason.setObjectName("nextStep")
        self.send_reason.setWordWrap(True)
        self.send_reason.setAccessibleName("Assistant send availability")
        composer_layout.addWidget(self.send_reason)
        composer_layout.addWidget(self.progress)
        conversation_layout.addWidget(self.composer)

        self.assistant_workspace_tabs = QTabWidget()
        self.assistant_workspace_tabs.setObjectName("assistantWorkspaceTabs")
        self.assistant_workspace_tabs.setDocumentMode(True)
        self.assistant_workspace_tabs.setAccessibleName(
            "Teaching chat and structured artifact workspaces"
        )
        self.assistant_workspace_tabs.addTab(conversation, "Teaching chat")
        self.artifact_workbench = AssistantArtifactsPanel(language=self._language)
        self.artifact_workbench.reviewActionRequested.connect(self._review_structured_artifact)
        self.artifact_scroll = QScrollArea()
        self.artifact_scroll.setWidgetResizable(True)
        self.artifact_scroll.setFrameShape(QFrame.NoFrame)
        self.artifact_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.artifact_scroll.setAccessibleName("Structured AI artifact contract and review")
        self.artifact_scroll.setWidget(self.artifact_workbench)
        self.assistant_workspace_tabs.addTab(
            self.artifact_scroll,
            "Structured artifacts · API preview",
        )
        self.workspace_splitter.addWidget(self.assistant_workspace_tabs)
        self.workspace_splitter.setStretchFactor(0, 0)
        self.workspace_splitter.setStretchFactor(1, 1)
        self.workspace_splitter.setSizes([340, 940])
        self.workspace_splitter.setChildrenCollapsible(False)
        layout.addWidget(self.workspace_splitter, 1)

        self.model.textChanged.connect(self._update_send_button)
        self.endpoint.textChanged.connect(self._update_send_button)
        self.credential.textChanged.connect(self._update_send_button)
        self.prompt.textChanged.connect(self._update_send_button)
        self.prompt.textChanged.connect(self._update_character_count)
        self.prompt.submitRequested.connect(self._send)
        self.network_enabled.stateChanged.connect(self._update_send_button)
        self.cancel_shortcut = QShortcut(QKeySequence(Qt.Key_Escape), self)
        self.cancel_shortcut.setContext(Qt.WidgetWithChildrenShortcut)
        self.cancel_shortcut.activated.connect(self._cancel_request)
        self._provider_changed(self.provider.currentText())
        self._update_character_count()
        tab_order = (
            self.provider,
            self.model,
            self.endpoint,
            self.credential,
            self.network_enabled,
            self.vision,
            self.send_preview,
            self.settings_toggle,
            self.prompt,
            self.send_button,
            self.cancel_request_button,
        )
        for first, second in zip(tab_order, tab_order[1:], strict=False):
            QWidget.setTabOrder(first, second)

    def _language_changed(self) -> None:
        self.assistant_workspace_tabs.setTabText(0, self._tr("Teaching chat"))
        self.assistant_workspace_tabs.setTabText(
            1,
            self._tr("Structured artifacts · API preview"),
        )
        self.assistant_workspace_tabs.setAccessibleName(
            self._tr("Teaching chat and structured artifact workspaces")
        )
        self.artifact_workbench.set_language(self._language)
        self._update_cloud_status()
        self._update_character_count()
        self._update_settings_button()
        self.request_status.setText(self._tr(self._request_status_source))
        if self._last_prompt:
            self.question_context.setText(f"{self._tr('Question')}: {self._last_prompt}")
        if self._answer_status_source is not None:
            self.answer.setPlainText(self._tr(self._answer_status_source))
        self._update_provider_field_tooltips()

    def set_structured_artifact_context(
        self,
        request: LLMTaskRequest | None,
        artifact: LLMArtifactResponse | None,
    ) -> None:
        """Load an already validated typed request/response without starting a network call."""

        self.artifact_workbench.set_context(request, artifact)
        self.assistant_workspace_tabs.setCurrentWidget(self.artifact_scroll)

    def _review_structured_artifact(self, artifact_id: str, decision: str) -> None:
        """Append a local immutable review; never create or overwrite an image layer."""

        artifact = self.artifact_workbench.artifact
        if artifact is None or artifact.artifact_id != artifact_id:
            self.artifact_workbench.set_artifact(artifact)
            return
        normalized = ArtifactReviewDecision(decision)
        note = f"openmedvisionx-local-artifact-review:{normalized.value}".encode()
        review = ArtifactReview(
            decision=normalized,
            reviewer_id="local-reviewer",
            note_sha256=hashlib.sha256(note).hexdigest(),
        )
        self.artifact_workbench.set_artifact(artifact.with_review(review))
        status = (
            "Structured artifact confirmed locally; no image layer was created."
            if normalized is ArtifactReviewDecision.CONFIRMED
            else "Structured artifact rejected locally; source data is unchanged."
        )
        self.statusChanged.emit(self._tr(status))

    def _update_provider_field_tooltips(self, _value: object = None) -> None:
        endpoint = self.endpoint.text().strip() or self._tr("not set")
        credential = self.credential.text().strip() or self._tr("not set")
        self.endpoint.setToolTip(self._tr("Exact endpoint: {value}").format(value=endpoint))
        self.credential.setToolTip(
            self._tr(
                "Credential reference (not the secret): {value}. Use none only for a "
                "localhost or loopback endpoint."
            ).format(value=credential)
        )

    def _provider_changed(self, name: str) -> None:
        if self._current_provider_name:
            self._provider_model_ids[self._current_provider_name] = self.model.text().strip()
        self._current_provider_name = name
        defaults = self.service.provider_defaults(name)
        self.model.setText(self._provider_model_ids.get(name, ""))
        self.endpoint.setText(defaults.endpoint)
        self.credential.setText(defaults.credential_ref)
        deepseek_text_only = name == "DeepSeek"
        if deepseek_text_only:
            self.vision.setChecked(False)
            self.send_preview.setChecked(False)
            self.model.setPlaceholderText("deepseek-v4-flash or deepseek-v4-pro")
        else:
            self.model.setPlaceholderText("User-supplied model ID (never hard-coded)")
        self.vision.setEnabled(not deepseek_text_only)
        self._update_attachment_availability()
        self._update_cloud_status()
        self._update_send_button()

    def _vision_changed(self, _state: int) -> None:
        if not self.vision.isChecked():
            self.send_preview.setChecked(False)
        self._update_attachment_availability()
        self._update_cloud_status()

    def set_source_available(self, image: object | None) -> None:
        """Keep image attachment controls aligned with the actual viewer state."""

        self._source_available = image is not None
        if not self._source_available:
            self.send_preview.setChecked(False)
        self._update_attachment_availability()
        self._update_cloud_status()

    def _update_attachment_availability(self) -> None:
        available = bool(
            self._active_task is None
            and self._source_available
            and self.vision.isChecked()
            and self.provider.currentText() != "DeepSeek"
        )
        self.send_preview.setEnabled(available)
        if not available and (
            not self._source_available or self.provider.currentText() == "DeepSeek"
        ):
            self.send_preview.setChecked(False)
        if available:
            reason = "Attach the active rendered 2-D plane after reviewing its exact PNG."
        elif not self._source_available:
            reason = "Open an image before attaching a rendered preview."
        elif self.provider.currentText() == "DeepSeek":
            reason = "The selected provider is configured as text-only."
        elif not self.vision.isChecked():
            reason = "Enable Vision input before attaching an image."
        else:
            reason = "Image attachment is unavailable while a request is active."
        self.send_preview.setToolTip(self._tr(reason))
        self.send_preview.setAccessibleDescription(self._tr(reason))

    def _toggle_settings(self) -> None:
        self._settings_visible = not self._settings_visible
        self.settings_scroll.setVisible(self._settings_visible)
        if self._settings_visible:
            total = max(self.workspace_splitter.width(), 900)
            self.workspace_splitter.setSizes([340, max(total - 340, 560)])
            self.provider.setFocus(Qt.OtherFocusReason)
        else:
            self.prompt.setFocus(Qt.OtherFocusReason)
        self._update_settings_button()

    def _update_settings_button(self) -> None:
        self.settings_toggle.setText(
            self._tr("Hide setup" if self._settings_visible else "Show setup")
        )

    def _update_character_count(self) -> None:
        count = len(self.prompt.toPlainText())
        self.character_count.setText(self._tr("{count} characters").format(count=count))

    def _update_send_button(self, _value: object = None) -> None:
        required_fields = (self.model, self.endpoint)
        validate = self.network_enabled.isChecked()
        for field in required_fields:
            state = "error" if validate and not field.text().strip() else "normal"
            _set_dynamic_property(field, "validation", state)
        credential_issue = self._credential_validation_issue()
        _set_dynamic_property(
            self.credential,
            "validation",
            "error" if validate and credential_issue is not None else "normal",
        )
        ready = bool(
            self._active_task is None
            and self.network_enabled.isChecked()
            and self.model.text().strip()
            and self.endpoint.text().strip()
            and credential_issue is None
            and self.prompt.toPlainText().strip()
        )
        self.send_button.setEnabled(ready)
        if self._active_task is not None:
            reason = "Request in progress. Cancel it or wait for completion."
        elif not self.network_enabled.isChecked():
            reason = "Next: enable network access for this provider request."
        elif not self.model.text().strip():
            reason = "Next: enter the exact provider model ID."
        elif not self.endpoint.text().strip():
            reason = "Next: enter the exact provider endpoint."
        elif credential_issue is not None:
            reason = credential_issue
        elif not self.prompt.toPlainText().strip():
            reason = "Next: type a learning question. No request is sent until you press Send."
        else:
            reason = "Ready: review the provider details, then send this learning request."
        self.send_reason.setText(self._tr(reason))
        _set_dynamic_property(self.send_reason, "state", "ready" if ready else "blocked")
        self.send_button.setToolTip(self._tr(reason))
        self.send_button.setAccessibleDescription(self._tr(reason))
        self.statusChanged.emit(self._tr(reason))

    def _credential_validation_issue(self) -> str | None:
        reference = self.credential.text().strip()
        if not reference:
            return "Next: enter env:NAME, keyring:service/user, or none for a loopback endpoint."
        try:
            parsed = CredentialReference.parse(reference)
        except Exception:
            return (
                "Next: use env:NAME, keyring:service/user, or none for a loopback endpoint; "
                "never paste a raw secret."
            )
        if parsed.scheme != "none":
            return None
        try:
            host = (urlsplit(self.endpoint.text().strip()).hostname or "").casefold()
        except ValueError:
            return "Next: enter the exact provider endpoint."
        loopback = host == "localhost" or host.endswith(".localhost")
        if not loopback:
            try:
                loopback = ip_address(host).is_loopback
            except ValueError:
                loopback = False
        if not loopback:
            return "Next: credential reference none is allowed only for a loopback endpoint."
        return None

    def _update_cloud_status(self) -> None:
        enabled = self.network_enabled.isChecked() and self.send_preview.isChecked()
        if not enabled:
            status = self._tr("OFF")
        elif self._active_task is not None:
            status = self._tr("ON — reviewed one-time request in progress")
        else:
            status = self._tr("ON — one-time review required for this exact request")
        self.cloud_status.setText(self._tr("Cloud image transfer: {status}").format(status=status))
        state = (
            "active"
            if enabled and self._active_task is not None
            else "pending"
            if enabled
            else "safe"
        )
        _set_dynamic_property(self.cloud_status, "state", state)

    def _make_provider(self) -> LLMProvider:
        defaults = self.service.provider_defaults(self.provider.currentText())
        configuration = ProviderConfiguration(
            provider_name=self.provider.currentText(),
            model_id=self.model.text().strip(),
            endpoint=self.endpoint.text().strip(),
            credential_ref=self.credential.text().strip(),
            supports_vision=self.vision.isChecked(),
            network_enabled=self.network_enabled.isChecked(),
            timeout=defaults.timeout,
        )
        return self.service.create_provider(configuration)

    def _set_request_state(self, *, busy: bool, status: str) -> None:
        self._request_status_source = status
        self.request_status.setText(self._tr(status))
        self.progress.setVisible(busy)
        self.config_group.setEnabled(not busy)
        self.cancel_request_button.setEnabled(busy)
        _set_dynamic_property(self.composer, "busy", busy)
        self._update_attachment_availability()
        self._update_cloud_status()
        self._update_send_button()
        self.statusChanged.emit(self._tr(status))

    def _open_response_link(self, link: QUrl) -> None:
        """Require a deliberate confirmation before leaving the application."""

        if not link.isValid() or link.scheme().lower() not in {"http", "https"}:
            QMessageBox.warning(
                self,
                self._tr("Link blocked"),
                self._tr("Only confirmed HTTP or HTTPS links can be opened."),
            )
            return
        confirmation = QMessageBox.question(
            self,
            self._tr("Open external link"),
            self._tr("Open this external link in your browser?\n\n{url}").format(
                url=link.toDisplayString()
            ),
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if confirmation == QMessageBox.Yes:
            QDesktopServices.openUrl(link)

    def _transfer_confirmation_text(self, plan: TransferPlan) -> str:
        """Describe every field bound to the one-shot image-transfer authorization."""

        lines = [
            f"{self._tr('Provider')}: {plan.provider_name}",
            f"{self._tr('Destination host')}: {plan.endpoint_host}",
            f"{self._tr('Endpoint')}: {plan.endpoint}",
            f"{self._tr('Model ID')}: {plan.model_id}",
            f"{self._tr('Task')}: {plan.task}",
            f"{self._tr('Total outbound bytes')}: {plan.total_bytes}",
            f"{self._tr('Prompt fingerprint')}: {plan.prompt_sha256}",
        ]
        for index, item in enumerate(plan.items, start=1):
            dimensions = (
                f"{item.width} × {item.height} px"
                if item.width is not None and item.height is not None
                else self._tr("not available")
            )
            lines.extend(
                (
                    "",
                    self._tr("Outbound item {index}").format(index=index),
                    f"{self._tr('Name')}: {item.name}",
                    f"{self._tr('MIME type')}: {item.mime_type}",
                    f"{self._tr('Dimensions')}: {dimensions}",
                    f"{self._tr('Bytes')}: {item.size_bytes}",
                    f"SHA-256: {item.sha256}",
                    f"{self._tr('Input transform')}: {self._tr(item.transform)}",
                    self._tr("De-identification and payload minimization"),
                )
            )
            lines.extend(f"• {self._tr(action)}" for action in item.deidentification_actions)
            lines.append(
                f"{self._tr('Burned-in text review')}: {self._tr(item.burned_in_text_review)}"
            )
        lines.extend(("", self._tr("Residual risks")))
        lines.extend(f"• {self._tr(risk)}" for risk in plan.residual_risks)
        lines.append(
            "• "
            + self._tr(
                "The full prompt text is also sent; remove names, identifiers, and other "
                "private information before authorizing."
            )
        )
        lines.extend(
            (
                "",
                self._tr(
                    "This consent is valid once and only for the exact details above. Any "
                    "change requires a new review."
                ),
            )
        )
        return "\n".join(lines)

    def _build_image_transfer_confirmation(
        self,
        plan: TransferPlan,
        preview: RenderedPreview,
    ) -> _ImageTransferConfirmationDialog:
        """Build the final review dialog with canonical pixels and exact plan fields."""

        if not plan.matches_preview(preview):
            raise ViewerError(
                self._tr(
                    "The displayed preview does not match the reviewed transfer plan; "
                    "create and review a new plan."
                )
            )

        pixmap = QPixmap()
        if not pixmap.loadFromData(preview.data, "PNG") or pixmap.isNull():
            raise ViewerError(
                self._tr("The sanitized rendered preview could not be displayed for review.")
            )
        return _ImageTransferConfirmationDialog(
            preview=preview,
            preview_pixmap=pixmap,
            details=self._transfer_confirmation_text(plan),
            translator=self._tr,
            parent=self,
        )

    def _confirm_image_transfer(
        self,
        plan: TransferPlan,
        preview: RenderedPreview,
    ) -> bool:
        """Request one-shot consent at the final cloud-transfer boundary."""

        dialog = self._build_image_transfer_confirmation(plan, preview)
        return dialog.exec_() == QMessageBox.Yes

    def _send(self) -> None:
        if self._active_task is not None:
            return
        if not self.network_enabled.isChecked():
            QMessageBox.warning(
                self,
                self._tr("Network disabled"),
                self._tr("Explicitly enable provider network requests first."),
            )
            return
        prompt = self.prompt.toPlainText().strip()
        if not prompt:
            return
        try:
            provider = self._make_provider()
            preview: RenderedPreview | None = None
            transfer_plan: TransferPlan | None = None
            if self.send_preview.isChecked():
                if not self.vision.isChecked():
                    raise ViewerError(
                        "Declare vision capability before authorizing an image preview."
                    )
                if not self._source_available:
                    raise ViewerError("Open an image before attaching the active rendered plane.")
                preview = RenderedPreview.from_png(self.preview_source())
                transfer_plan = self.service.plan_image_transfer(
                    provider,
                    prompt,
                    preview,
                    task="teaching-explanation",
                )
                if not self._confirm_image_transfer(transfer_plan, preview):
                    return
                self.service.authorize_image_transfer(provider, transfer_plan)
        except BaseException as exc:
            QMessageBox.critical(
                self,
                self._tr("Provider configuration error"),
                str(exc),
            )
            return

        def operation(context: Any) -> Any:
            context.raise_if_cancelled()
            response = self.service.chat(
                provider,
                prompt,
                preview=preview,
                transfer_plan=transfer_plan,
                cancellation_token=context.token,
            )
            context.raise_if_cancelled()
            return response

        task = self.runner.submit(operation)
        self._active_task = task
        self._last_prompt = prompt
        self.question_context.setText(f"{self._tr('Question')}: {prompt}")
        self.question_context.setVisible(True)
        self.prompt.clear()
        self._answer_status_source = None
        self._set_request_state(busy=True, status="Generating a new response…")
        self.watcher.watch(
            task,
            success=lambda response, handle=task: self._answered_if_current(handle, response),
            error=lambda error, handle=task: self._failed_if_current(handle, error),
        )

    def _answered_if_current(self, task: BackgroundTask[Any], response: Any) -> None:
        if task is not self._active_task:
            return
        self._active_task = None
        self._answered(response)

    def _failed_if_current(self, task: BackgroundTask[Any], error: BaseException) -> None:
        if task is not self._active_task:
            return
        self._active_task = None
        self._failed(error)

    def _answered(self, response: LLMResponse) -> None:
        self._answer_status_source = None
        answer_text = getattr(response, "text", None)
        if not isinstance(answer_text, str):
            answer_text = str(getattr(response, "content", response))
        self.answer.set_markdown_text(answer_text)
        provider = getattr(response, "provider", "")
        model = getattr(response, "model", "")
        timestamp = getattr(response, "timestamp", None)
        metadata = " · ".join(
            item for item in (str(provider), str(model), str(timestamp or "")) if item
        )
        self.answer_metadata.setText(metadata)
        self.answer_metadata.setVisible(bool(metadata))
        disclaimer = getattr(response, "disclaimer", "")
        self.answer_disclaimer.setText(str(disclaimer))
        self.answer_disclaimer.setVisible(bool(disclaimer))
        self.response_stack.setCurrentIndex(1)
        self._has_answer = True
        self._set_request_state(busy=False, status="Response ready · Markdown rendered safely.")

    def _failed(self, error: BaseException) -> None:
        if isinstance(error, OperationCancelled):
            self._answer_status_source = "Request cancelled."
        else:
            self._answer_status_source = f"Request failed safely: {error}"
        if not self._has_answer:
            self.answer.setPlainText(self._tr(self._answer_status_source))
            self.response_stack.setCurrentIndex(1)
        if not self.prompt.toPlainText().strip() and self._last_prompt:
            self.prompt.setPlainText(self._last_prompt)
        self._set_request_state(busy=False, status=self._answer_status_source)

    def _cancel_request(self) -> None:
        if self._active_task is not None and self._active_task.cancel():
            self.cancel_request_button.setEnabled(False)
            self._answer_status_source = "Cancelling request safely…"
            self._request_status_source = self._answer_status_source
            self.request_status.setText(self._tr(self._answer_status_source))
            if not self._has_answer:
                self.answer.setPlainText(self._tr(self._answer_status_source))
                self.response_stack.setCurrentIndex(1)

    def close(self) -> None:  # type: ignore[override]
        self.watcher.stop()
        if self._active_task is not None:
            self._active_task.cancel()
        self.runner.shutdown(wait=False, cancel_pending=True)
        super().close()


class TeachingExperimentPage(_BilingualPage, QWidget):
    """A structured, data-free learning path generated entirely at runtime."""

    startFirstExperiment = pyqtSignal()
    statusChanged = pyqtSignal(str)

    SECTION_ORDER = (
        "Principle",
        "Formula",
        "Parameter explanation",
        "Steps",
        "Expected observation",
        "Common mistakes",
        "Reflection question",
    )

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 8, 0, 0)
        root.setSpacing(10)
        brand_row = QHBoxLayout()
        brand_row.setContentsMargins(0, 0, 0, 0)
        brand_row.setSpacing(10)
        self.brand_logo = QLabel()
        if not _apply_logo(self.brand_logo, "logo_pure.png", 30):
            self.brand_logo.setText(APP_NAME)
            self.brand_logo.setObjectName("appTitle")
        brand_row.addWidget(self.brand_logo)
        brand = QLabel("Learning Experiments")
        brand.setObjectName("pageTitle")
        brand_row.addWidget(brand)
        brand_row.addStretch(1)
        root.addLayout(brand_row)
        self.introduction = QLabel(
            "Choose an experiment to review its principle, parameters, expected result, "
            "and reflection prompt. One public LoDoPaB benchmark teaching case is bundled "
            "for the model exercise; it contains no DICOM metadata, patient identifiers, "
            "or clinical case narrative."
        )
        self.introduction.setWordWrap(True)
        self.introduction.setTextInteractionFlags(Qt.TextSelectableByMouse)
        root.addWidget(self.introduction)
        first_step = QFrame()
        first_step.setObjectName("toolbarSurface")
        first_step_layout = QVBoxLayout(first_step)
        first_step_layout.setContentsMargins(12, 10, 12, 10)
        first_step_copy = QVBoxLayout()
        first_step_title = QLabel("Recommended first step: CT phantom reconstruction")
        first_step_title.setObjectName("emptyStateTitle")
        first_step_title.setWordWrap(True)
        first_step_detail = QLabel(
            "About 1 minute · offline · synthetic attenuation data · no model or medical file."
        )
        first_step_detail.setObjectName("mutedText")
        first_step_detail.setWordWrap(True)
        first_step_copy.addWidget(first_step_title)
        first_step_copy.addWidget(first_step_detail)
        first_step_layout.addLayout(first_step_copy, 1)
        self.first_experiment_button = QPushButton("Start first experiment")
        self.first_experiment_button.setObjectName("primary")
        self.first_experiment_button.setAccessibleDescription(
            "Open CT Lab and run the deterministic synthetic phantom experiment."
        )
        self.first_experiment_button.clicked.connect(self.startFirstExperiment.emit)
        first_step_layout.addWidget(self.first_experiment_button, 0, Qt.AlignLeft)
        root.addWidget(first_step)
        selector_row = QHBoxLayout()
        selector_row.addWidget(QLabel("Experiment:"))
        self.experiment_selector = QComboBox()
        self.experiment_selector.setSizeAdjustPolicy(
            QComboBox.AdjustToMinimumContentsLengthWithIcon
        )
        self.experiment_selector.setMinimumContentsLength(24)
        self.experiment_selector.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        for experiment_title in TEACHING_EXPERIMENTS:
            self.experiment_selector.addItem(experiment_title, experiment_title)
        self.experiment_selector.currentIndexChanged.connect(self._show_experiment)
        selector_row.addWidget(self.experiment_selector, 1)
        root.addLayout(selector_row)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        content = QWidget()
        content_layout = QVBoxLayout(content)
        self.section_labels: dict[str, QLabel] = {}
        for section in self.SECTION_ORDER:
            group = QGroupBox(section)
            group_layout = QVBoxLayout(group)
            label = QLabel()
            label.setWordWrap(True)
            label.setTextInteractionFlags(Qt.TextSelectableByMouse)
            group_layout.addWidget(label)
            content_layout.addWidget(group)
            self.section_labels[section] = label
        content_layout.addStretch(1)
        scroll.setWidget(content)
        root.addWidget(scroll, 1)
        self.local_only_note = QLabel(
            "OpenMedVisionX is local-first and for learning/research only—not for clinical use. "
            "Cloud image transfer remains off unless separately authorized in the assistant."
        )
        self.local_only_note.setWordWrap(True)
        self.local_only_note.setObjectName("warningBanner")
        root.addWidget(self.local_only_note)
        self._show_experiment(self.experiment_selector.currentIndex())
        QWidget.setTabOrder(self.first_experiment_button, self.experiment_selector)

    def _show_experiment(self, selection: int | str) -> None:
        title = (
            self.experiment_selector.itemData(selection)
            if isinstance(selection, int)
            else translate(selection, "en")
        )
        if not isinstance(title, str):
            return
        self.experiment_selector.setToolTip(self.experiment_selector.currentText())
        lesson = TEACHING_EXPERIMENTS.get(title)
        if lesson is None:
            return
        for section in self.SECTION_ORDER:
            self.section_labels[section].setText(self._tr(lesson[section]))
        self.statusChanged.emit(self._tr("Learning path: {title}").format(title=self._tr(title)))

    def _language_changed(self) -> None:
        self._show_experiment(self.experiment_selector.currentIndex())


class OpenMedVisionXWindow(QMainWindow):
    TAB_TITLES = (
        "Images",
        "CT Lab",
        "Models",
        "Learn",
        "Evaluate",
        "AI Assistant",
    )

    def __init__(self) -> None:
        super().__init__()
        self._language: Language = "en"
        self._status_source = (
            "Local-first. No data is uploaded automatically. Not for clinical use."
        )
        self.setWindowTitle(f"{APP_NAME} — {APP_SUBTITLE}")
        self.resize(1440, 900)
        # Qt sizes are logical pixels. The responsive workspaces fit at 900x620;
        # platform display scaling is applied by Qt afterwards.
        self.setMinimumSize(900, 620)
        self.setStyleSheet(APP_STYLE)
        icon_pixmap = _brand_pixmap("logo_pure.png", 96)
        if not icon_pixmap.isNull():
            self.setWindowIcon(QIcon(icon_pixmap))
        central = QWidget()
        central.setObjectName("appShell")
        root = QVBoxLayout(central)
        root.setContentsMargins(14, 12, 14, 10)
        root.setSpacing(10)
        header_frame = QFrame()
        header_frame.setObjectName("appHeader")
        header = QHBoxLayout(header_frame)
        header.setContentsMargins(14, 10, 12, 10)
        header.setSpacing(11)
        self.header_logo = QLabel()
        if not _apply_logo(self.header_logo, "logo_pure.png", 38):
            self.header_logo.setText("OMVX")
            self.header_logo.setObjectName("appTitle")
        header.addWidget(self.header_logo, 0, Qt.AlignVCenter)
        heading = QVBoxLayout()
        heading.setSpacing(1)
        self.title_label = QLabel(APP_NAME)
        self.title_label.setObjectName("appTitle")
        self.subtitle_label = QLabel(APP_SHORT_SUBTITLE)
        self.subtitle_label.setObjectName("appSubtitle")
        self.subtitle_label.setWordWrap(True)
        heading.addWidget(self.title_label)
        heading.addWidget(self.subtitle_label)
        header.addLayout(heading, 1)
        self.privacy_label = QLabel("Local · Research")
        self.privacy_label.setObjectName("privacyChip")
        self.privacy_label.setToolTip("Local first · Research only")
        self.privacy_label.setAlignment(Qt.AlignCenter)
        self.privacy_label.setMinimumWidth(110)
        self.privacy_label.setMaximumWidth(200)
        self.privacy_label.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        header.addWidget(self.privacy_label, 0, Qt.AlignVCenter)
        self.language_button = QPushButton("中文")
        self.language_button.setObjectName("languageSwitchButton")
        self.language_button.setAccessibleName("Switch language")
        self.language_button.clicked.connect(self.toggle_language)
        header.addWidget(self.language_button, 0, Qt.AlignVCenter)
        root.addWidget(header_frame)
        self.tabs = QTabWidget()
        self.tabs.setObjectName("workspaceTabs")
        self.tabs.setDocumentMode(True)
        self.tabs.setUsesScrollButtons(True)
        self.tabs.tabBar().setExpanding(True)
        self.tabs.setAccessibleName("OpenMedVisionX workspaces")
        self.viewer = ViewerPage()
        self.reconstruction = ReconstructionPage(
            self.viewer.current_plane,
            roi_source=self.viewer.current_metric_rois,
        )
        self.models = ModelPage(
            self.viewer.current_model_input,
            self.viewer.current_model_input_context,
        )
        self.evaluation = EvaluationPage()
        self.learning = TeachingExperimentPage()
        self.assistant = AssistantPage(self.viewer.rendered_preview_png)
        for page, tab_title in zip(
            (
                self.viewer,
                self.reconstruction,
                self.models,
                self.learning,
                self.evaluation,
                self.assistant,
            ),
            self.TAB_TITLES,
            strict=True,
        ):
            self.tabs.addTab(page, tab_title)
        root.addWidget(self.tabs, 1)
        self.setCentralWidget(central)
        for page in (
            self.viewer,
            self.reconstruction,
            self.models,
            self.learning,
            self.evaluation,
            self.assistant,
        ):
            page.statusChanged.connect(
                lambda message, source_page=page: self._show_page_status(
                    source_page,
                    message,
                )
            )
        self.tabs.currentChanged.connect(self._sync_current_page_status)
        self.learning.startFirstExperiment.connect(self._start_first_experiment)
        self.viewer.imageChanged.connect(self.reconstruction.reset)
        self.viewer.imageChanged.connect(self.models.set_source_available)
        self.viewer.imageChanged.connect(self.assistant.set_source_available)
        self.reconstruction.reset(None)
        self.models.set_source_available(None)
        self.assistant.set_source_available(None)
        self._workspace_shortcuts: list[QShortcut] = []
        for index in range(self.tabs.count()):
            shortcut = QShortcut(QKeySequence(f"Alt+{index + 1}"), self)
            shortcut.setContext(Qt.WindowShortcut)
            shortcut.activated.connect(lambda selected=index: self.tabs.setCurrentIndex(selected))
            self._workspace_shortcuts.append(shortcut)
        self.set_language("en")

    def minimumSizeHint(self) -> QSize:  # noqa: N802 - Qt API
        """Advertise the responsive shell floor instead of a platform font-dependent hint."""

        return QSize(self.minimumWidth(), self.minimumHeight())

    def _start_first_experiment(self) -> None:
        self.tabs.setCurrentIndex(1)
        QTimer.singleShot(0, self.reconstruction.start_first_experiment)

    def _show_page_status(self, page: QWidget, message: str) -> None:
        if self.tabs.currentWidget() is page:
            self._show_status_message(message)

    def _sync_current_page_status(self, _index: int = -1) -> None:
        page = self.tabs.currentWidget()
        if page is self.viewer:
            message = self.viewer.status.text()
        elif page is self.reconstruction:
            message = self.reconstruction.action_status.text()
        elif page is self.models:
            message = self.models.current_status_text()
        elif page is self.evaluation:
            message = self.evaluation.current_status_text
        elif page is self.learning:
            message = self._tr_learning_status()
        else:
            message = self.assistant.request_status.text()
        self._show_status_message(message)

    def _tr_learning_status(self) -> str:
        return translate("Learning path: {title}", self._language).format(
            title=self.learning.experiment_selector.currentText()
        )

    @property
    def language(self) -> Language:
        return self._language

    def toggle_language(self) -> None:
        self.set_language("zh_CN" if self._language == "en" else "en")

    def set_language(self, language: Language) -> None:
        """Switch all persistent pages in place, preserving their runtime state."""

        if language not in {"en", "zh_CN"}:
            raise ValueError(f"Unsupported UI language: {language}")
        self._language = language
        window_subtitle = translate(APP_SUBTITLE, language)
        self.setWindowTitle(f"{APP_NAME} — {window_subtitle}")
        self.title_label.setText(APP_NAME)
        self.subtitle_label.setText(translate(APP_SHORT_SUBTITLE, language))
        self.privacy_label.setText(translate("Local · Research", language))
        self.privacy_label.setToolTip(translate("Local first · Research only", language))
        self.language_button.setAccessibleName(translate("Switch language", language))
        self.tabs.setAccessibleName(translate("OpenMedVisionX workspaces", language))
        for index, title in enumerate(self.TAB_TITLES):
            self.tabs.setTabText(index, translate(title, language))
        for page in (
            self.viewer,
            self.reconstruction,
            self.models,
            self.learning,
            self.evaluation,
            self.assistant,
        ):
            page.set_language(language)
        if language == "en":
            self.language_button.setText("中文")
            self.language_button.setToolTip("切换到中文")
        else:
            self.language_button.setText("English")
            self.language_button.setToolTip("Switch to English")
        self._sync_current_page_status()

    def _show_status_message(self, message: str) -> None:
        self._status_source = translate(message, "en")
        self.statusBar().showMessage(translate(self._status_source, self._language))

    def closeEvent(self, event: QCloseEvent) -> None:
        self.viewer.close()
        self.reconstruction.close()
        self.assistant.close()
        self.models.close()
        event.accept()
