"""Capability-driven OpenMedVisionX desktop interface."""

from __future__ import annotations

import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace
from html import escape
from io import BytesIO
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import numpy as np
from PyQt5.QtCore import QRect, Qt, QTimer, QUrl, pyqtSignal
from PyQt5.QtGui import QCloseEvent, QDesktopServices, QIcon, QKeySequence, QPixmap
from PyQt5.QtWidgets import (
    QAbstractButton,
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
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
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from ..algorithms import ReconstructionRequest
from ..domain.display import ColorDisplayMapping, GrayscaleDisplayMapping
from ..domain.images import (
    Capability,
    ImageData,
    ImageSequence2D,
    ImageVolume,
    RasterImage2D,
)
from ..domain.transforms import TransformRecord
from ..errors import OperationCancelled, ViewerError
from ..inference import (
    InferenceResult,
    ModelManifest,
    VisualizationKind,
    inverse_map_spatial_array,
)
from ..llm import LLMProvider, LLMResponse, RenderedPreview
from ..runtime import BackgroundTask, TaskRunner
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
from .i18n import Language, translate
from .image_view import ImageView, array_to_qimage
from .theme import APP_STYLE
from .widgets import HistogramDialog, SafeMarkdownBrowser, SubmitPlainTextEdit

APP_NAME = "OpenMedVisionX"
APP_SUBTITLE = "An Open Interactive Platform for Medical Computer Vision Learning and Exploration"
APP_SHORT_SUBTITLE = "Medical vision workspace"
SAM_REPOSITORY_URL = (
    "https://github.com/facebookresearch/segment-anything#model-checkpoints"
)
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
            "The answer records provider/model/time and a disclaimer; revoking consent stops "
            "image transfer."
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
        self._last_measurement_metrics: dict[str, float] = {}
        self._metric_roi_plane_key: tuple[str, int] | None = None
        self._metric_roi: tuple[int, int, int, int] | None = None
        self._status_source = "Open a local image, DICOM folder/ZIP, or NIfTI volume."
        self._status_translatable = True
        self._measurement_status: tuple[str, object] | None = None
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
        self.histogram_button.clicked.connect(self._show_histogram)
        self.histogram_button.setEnabled(False)
        self.fit_button = QPushButton("Fit views")
        self.fit_button.setShortcut(QKeySequence("Ctrl+0"))
        self.fit_button.setToolTip("Fit all image views (Ctrl+0)")
        self.fit_button.clicked.connect(self._fit_views)
        self.fit_button.setEnabled(False)
        view_actions.addWidget(self.histogram_button)
        view_actions.addWidget(self.fit_button)
        toolbar.addLayout(view_actions, 0, 1, Qt.AlignRight)

        interaction_actions = QHBoxLayout()
        interaction_actions.setSpacing(8)
        self.tool_label = QLabel("Tool:")
        self.measurement_combo = QComboBox()
        self.measurement_combo.addItems(["Pan / zoom", "Distance", "Area / ROI", "Annotation"])
        self.measurement_combo.currentIndexChanged.connect(self._measurement_mode_changed)
        self.measurement_combo.setEnabled(False)
        self.clear_button = QPushButton("Clear marks")
        self.clear_button.clicked.connect(self._clear_marks)
        self.clear_button.setEnabled(False)
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
        grid = QGridLayout(views)
        self.axial_view = ImageView("Image / Axial")
        self.coronal_view = ImageView("Coronal (RAS+)")
        self.sagittal_view = ImageView("Sagittal (RAS+)")
        self.aux_view = ImageView("Overlay / comparison")
        for view in (self.axial_view, self.coronal_view, self.sagittal_view, self.aux_view):
            view.pixelHovered.connect(self._pixel_hovered)
            view.measurementCompleted.connect(self._measurement_completed)
        grid.addWidget(self.axial_view, 0, 0)
        grid.addWidget(self.coronal_view, 0, 1)
        grid.addWidget(self.sagittal_view, 1, 0)
        grid.addWidget(self.aux_view, 1, 1)
        self.workspace_splitter.addWidget(views)

        panel = QScrollArea()
        panel.setWidgetResizable(True)
        panel.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        panel.setMinimumWidth(300)
        controls = QWidget()
        control_layout = QVBoxLayout(controls)
        self.navigation_group = QGroupBox("Navigation")
        self.navigation_group.setVisible(False)
        navigation_form = QFormLayout(self.navigation_group)
        navigation_form.setRowWrapPolicy(QFormLayout.WrapLongRows)
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
        navigation_form.addRow(self.frame_label, self.frame_slider)
        navigation_form.addRow(self.coronal_label, self.coronal_slider)
        navigation_form.addRow(self.sagittal_label, self.sagittal_slider)
        navigation_form.addRow(self.play_button)
        control_layout.addWidget(self.navigation_group)

        self.display_group = QGroupBox("Display")
        self.display_group.setToolTip("Display mapping changes only the view, not decoded data.")
        self.display_group.setEnabled(False)
        display_form = QFormLayout(self.display_group)
        display_form.setRowWrapPolicy(QFormLayout.WrapLongRows)
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
        display_form.addRow("Lower", self.lower_spin)
        display_form.addRow("Upper", self.upper_spin)
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
        display_form.addRow("RGB brightness", self.brightness_spin)
        display_form.addRow("RGB contrast", self.contrast_spin)
        display_form.addRow("RGB gamma", self.gamma_spin)
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
        self.export_button = QPushButton("Export PNG…")
        self.export_button.clicked.connect(self._export_rendered_plane)
        self.save_experiment_button = QPushButton("Save record…")
        self.save_experiment_button.clicked.connect(self._save_experiment)
        self.cancel_local_button = QPushButton("Cancel")
        self.cancel_local_button.clicked.connect(self._cancel_local_operation)
        for button in (
            self.pair_mask_button,
            self.pair_annotation_button,
            self.export_button,
            self.save_experiment_button,
            self.cancel_local_button,
        ):
            button.setEnabled(False)
        local_buttons.addWidget(self.pair_mask_button, 0, 0)
        local_buttons.addWidget(self.pair_annotation_button, 1, 0)
        local_buttons.addWidget(self.export_button, 2, 0)
        local_buttons.addWidget(self.save_experiment_button, 3, 0)
        local_buttons.addWidget(self.cancel_local_button, 4, 0)
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
        self.info.setMinimumWidth(270)
        info_layout.addWidget(self.warning_label)
        info_layout.addWidget(self.info)
        control_layout.addWidget(info_group, 1)
        panel.setWidget(controls)
        self.workspace_splitter.addWidget(panel)
        self.workspace_splitter.setStretchFactor(0, 4)
        self.workspace_splitter.setStretchFactor(1, 1)
        self.workspace_splitter.setSizes([960, 320])
        self.workspace_splitter.setChildrenCollapsible(False)
        root.addWidget(self.workspace_splitter, 1)

        self.status = QLabel("Open a local image, DICOM folder/ZIP, or NIfTI volume.")
        self.status.setObjectName("statusStrip")
        self.status.setWordWrap(True)
        root.addWidget(self.status)

    def _language_changed(self) -> None:
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

    def _begin_load(self, path: str) -> None:
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
        task = self.service.begin_load(path, prepare_axis_aligned_volume=True)
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
        self._last_measurement_metrics.clear()
        self._metric_roi_plane_key = None
        self._metric_roi = None
        for view in (self.axial_view, self.coronal_view, self.sagittal_view, self.aux_view):
            view.clear_image()
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
        self.export_button.setEnabled(Capability.VIEW_2D in self.image.capabilities)
        self.save_experiment_button.setEnabled(True)
        self._configure_for_image()
        self._reset_display_mapping()
        self._update_views(fit=True)
        self._update_info()
        self.imageChanged.emit(self.image)
        self.statusChanged.emit(
            self._tr("Loaded {kind}: {shape}").format(
                kind=study.source_kind,
                shape=self.image.shape,
            )
        )

    def _load_failed(self, error: BaseException) -> None:
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
            self._tr("OpenMedVisionX – load failed"),
            str(error),
        )

    def _configure_for_image(self) -> None:
        assert self.image is not None
        image = self.image
        capabilities = image.capabilities
        has_orthogonal_views = Capability.ORTHOGONAL_VIEWS in capabilities
        has_frame_navigation = Capability.FRAME_NAVIGATION in capabilities
        has_frame_playback = Capability.FRAME_PLAYBACK in capabilities
        self.coronal_view.setVisible(has_orthogonal_views)
        self.sagittal_view.setVisible(has_orthogonal_views)
        self.aux_view.setVisible(
            has_orthogonal_views
            and self.study is not None
            and self.study.volume_projection is not None
        )
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
                "HU window"
                if Capability.HU_WINDOWING in image.capabilities
                else "Intensity range"
            )
        )

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
        self._update_views(value)

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

    def _current_plane_key(self) -> tuple[str, int]:
        if isinstance(self.image, ImageSequence2D):
            return ("frame", self.frame_slider.value())
        if isinstance(self.image, ImageVolume):
            return ("axial", self.frame_slider.value())
        return ("image", 0)

    def _render_paired_overlays(self) -> None:
        self.axial_view.clear_overlays()
        if self.image is None or self._paired_plane_key != self._current_plane_key():
            if self._paired_plane_key is not None:
                self.pairing_status.setText(
                    self._tr(
                        "A local overlay is paired to another slice/page; return to it to view "
                        "the overlay."
                    )
                )
            return
        if self._paired_mask is not None:
            self.axial_view.set_mask_overlay(self._paired_mask)
        annotations = self._paired_annotations
        if annotations is not None:
            if annotations.boxes:
                self.axial_view.set_box_overlays(
                    np.asarray(annotations.boxes),
                    labels=list(annotations.box_labels),
                )
            if annotations.points:
                self.axial_view.set_point_overlays(
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

    def current_plane(self) -> np.ndarray:
        if self.image is None:
            raise ViewerError("No image is loaded.")
        if isinstance(self.image, RasterImage2D):
            return self.image.array
        if isinstance(self.image, ImageSequence2D):
            return self.image.array[self.frame_slider.value()]
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
            return {
                "modality": "generic-image",
                "spacing": self.image.pixel_spacing,
            }
        if isinstance(self.image, ImageSequence2D):
            return {"modality": "generic-image", "spacing": None}
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
        return {"modality": modality, "spacing": self.image.spacing[:2]}

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
        self._set_status(
            f"x={x}, y={y}, {semantics}={np.asarray(value).tolist()}",
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
                and (self.sender() is None or self.sender() is self.axial_view)
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
                "This value is user-provided. OpenMedVisionX never infers medical spacing "
                "from DPI."
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
    ) -> None:
        self._cancel_local_operation(clear_handle=True)
        task = self.local_runner.submit(operation)
        self._active_local_task = task
        for button in (
            self.pair_mask_button,
            self.pair_annotation_button,
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
            error=lambda error, handle=task: self._local_operation_failed(handle, error),
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
    ) -> None:
        if task is not self._active_local_task:
            return
        self._active_local_task = None
        self.progress.setVisible(False)
        self._set_local_controls_available()
        self._set_status(f"Local operation failed safely: {error}")
        if error.__class__.__name__ != "OperationCancelled":
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
            self._tr(
                "Local annotation paired explicitly; external references are not followed."
            )
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
            self._tr(
                "Pixel-free experiment parameters and numeric metrics saved: {name}"
            ).format(name=path.name)
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
                    "Page sequence only: 3-D tools and physical volume measurements are "
                    "disabled."
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
        self.service.close()
        self.local_runner.shutdown(wait=False, cancel_pending=True)
        super().close()


class ReconstructionPage(_BilingualPage, QWidget):
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
        self._build_ui()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 8, 0, 0)
        root.setSpacing(10)
        explanation = QLabel(
            "For parallel-beam CT, 180° contains the complete projection set; "
            "360° data is folded and averaged."
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
        setup_scroll.setMinimumWidth(280)
        setup_content = QWidget()
        setup_layout = QVBoxLayout(setup_content)
        setup_layout.setContentsMargins(0, 0, 8, 0)
        setup_layout.setSpacing(10)

        self.source_status = QLabel(
            "Open an image in Images to enable reconstruction."
        )
        self.source_status.setObjectName("infoBanner")
        self.source_status.setWordWrap(True)
        setup_layout.addWidget(self.source_status)

        self.configuration_card = QGroupBox("Reconstruction setup")
        controls = QFormLayout(self.configuration_card)
        controls.setRowWrapPolicy(QFormLayout.WrapLongRows)
        controls.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
        controls.setHorizontalSpacing(10)
        controls.setVerticalSpacing(9)
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
        controls.addRow(self.range_control_label, self.angle_range)
        controls.addRow(self.projection_control_label, self.projections)
        controls.addRow(self.circle)
        controls.addRow(self.algorithm_control_label, self.algorithm)
        controls.addRow(self.algorithm_options)

        self.generate_button = QPushButton("1. Generate sinogram")
        self.generate_button.setObjectName("primary")
        self.generate_button.clicked.connect(self._generate)
        self.reconstruct_button = QPushButton("2. Reconstruct")
        self.reconstruct_button.clicked.connect(self._reconstruct)
        self.reconstruct_button.setEnabled(False)
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.clicked.connect(self._cancel_all)
        self.cancel_button.setEnabled(False)
        action_row = QVBoxLayout()
        action_row.setSpacing(8)
        action_row.addWidget(self.generate_button)
        action_row.addWidget(self.reconstruct_button)
        action_row.addWidget(self.cancel_button)
        controls.addRow(action_row)
        setup_layout.addWidget(self.configuration_card)

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
        grid.addWidget(self.input_view, 0, 0)
        grid.addWidget(self.sinogram_view, 0, 1)
        grid.addWidget(self.result_view, 1, 0)
        grid.addWidget(self.error_view, 1, 1)
        results_layout.addLayout(grid, 1)
        bottom = QGridLayout()
        self.intermediate_combo = QComboBox()
        self.intermediate_combo.setProperty(_I18N_SKIP_ITEMS_PROPERTY, True)
        self.intermediate_combo.currentIndexChanged.connect(self._show_intermediate)
        self.metrics_label = QLabel(
            "Metrics use one shared intensity range."
        )
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

    def _update_algorithm_options(self, algorithm: str) -> None:
        self.algorithm_options.setCurrentIndex(
            {"FBP": 0, "BP": 1, "DFR": 2, "SART": 3}.get(algorithm, 1)
        )

    def _set_reconstruction_controls_enabled(self, enabled: bool) -> None:
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

    def _reconstruction_parameters_changed(self, _value: object = None) -> None:
        if self._active_task is None and self.reconstruction_result is not None:
            self._invalidate_reconstruction()

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
        self.metrics_label.setText(
            self._tr("Metrics use one shared intensity range.")
        )
        self.export_result_button.setEnabled(False)
        self.save_experiment_button.setEnabled(False)

    def _language_changed(self) -> None:
        self._populate_process_combo(refresh_view=False)
        metric_text = self.metrics_label.text()
        english_prefix = "Joint range "
        chinese_prefix = f"{translate('Joint range', 'zh_CN')} "
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
            rgb = array[:, :, :3].astype(np.float64)
            array = 0.2126 * rgb[:, :, 0] + 0.7152 * rgb[:, :, 1] + 0.0722 * rgb[:, :, 2]
        if array.ndim != 2:
            raise ViewerError("CT reconstruction requires one 2-D grayscale plane.")
        size = min(array.shape)
        top = (array.shape[0] - size) // 2
        left = (array.shape[1] - size) // 2
        return np.asarray(array[top : top + size, left : left + size], dtype=np.float64)

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
            source = np.asarray(self.source())
            self.reference = self._prepare_source(source)
            self._reference_rois = self._prepare_rois(source.shape[:2], self.roi_source())
        except BaseException as exc:
            QMessageBox.warning(self, self._tr("No reconstruction input"), str(exc))
            return
        mapping = GrayscaleDisplayMapping.from_percentiles(self.reference)
        self.input_view.set_array(self.reference, grayscale_mapping=mapping, fit=True)
        self.progress.setValue(0)
        self.reconstruct_button.setEnabled(False)
        task = self.service.begin_sinogram(
            self.reference,
            projection_count=self.projections.value(),
            angle_range=int(self.angle_range.currentText()),
            circle=self.circle.isChecked(),
        )
        self._watch_task(task, self._sinogram_ready)

    def _sinogram_ready(self, result: Any) -> None:
        self.sinogram_result = result
        self.sinogram_view.set_array(result.sinogram, fit=True)
        self._refresh_process_images()
        self.reconstruct_button.setEnabled(True)
        self.progress.setValue(100)

    def _reconstruct(self) -> None:
        if self.sinogram_result is None or self.reference is None:
            return
        request = ReconstructionRequest(
            self.sinogram_result.sinogram,
            self.sinogram_result.theta_degrees,
            output_size=self.reference.shape[0],
            circle=self.sinogram_result.circle,
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
        self.sinogram_result = None
        self.reconstruction_result = None
        self._latest_metrics.clear()
        self._reference_rois.clear()
        self._metric_maps.clear()
        self._process_images.clear()
        self.intermediate_combo.clear()
        self.metrics_label.setText(
            self._tr("Metrics use one shared intensity range.")
        )
        self._set_reconstruction_controls_enabled(True)
        self.source_status.setVisible(_image is None)
        self.generate_button.setEnabled(_image is not None)
        self.reconstruct_button.setEnabled(False)
        self.cancel_button.setText(self._tr("Cancel"))
        self.cancel_button.setEnabled(False)
        self.export_result_button.setEnabled(False)
        self.save_experiment_button.setEnabled(False)
        for view in (self.input_view, self.sinogram_view, self.result_view, self.error_view):
            view.clear_image()

    def _reconstruction_ready(self, result: Any) -> None:
        self.reconstruction_result = result
        self.result_view.set_array(result.image, fit=True)
        self.intermediate_combo.clear()
        self._metric_maps.clear()
        try:
            assert self.reference is not None
            metrics = self.service.compute_metrics(
                self.reference,
                result.image,
                rois=self._reference_rois,
            )
            self._metric_maps = {
                "Signed normalized difference": metrics.difference,
                "Absolute error heatmap": metrics.error_heatmap,
            }
            self.error_view.set_array(metrics.error_heatmap, fit=True)
            self.error_view.set_title("Absolute error heatmap")
            metric_text = (
                f"{self._tr('Joint range')} "
                f"{metrics.intensity_range[0]:.4g}…{metrics.intensity_range[1]:.4g} | "
                f"MSE {metrics.values.mse:.6g} | "
                f"PSNR {metrics.values.psnr:.4g} dB | "
                f"SSIM {metrics.values.ssim:.5f}"
            )
            self._latest_metrics = {
                "mse": metrics.values.mse,
                "psnr_db": (metrics.values.psnr if np.isfinite(metrics.values.psnr) else None),
                "ssim": metrics.values.ssim,
                "joint_intensity_min": metrics.intensity_range[0],
                "joint_intensity_max": metrics.intensity_range[1],
            }
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
            self.metrics_label.setText(
                self._tr("Metrics unavailable: {error}").format(error=exc)
            )
        self._refresh_process_images()
        self.export_result_button.setEnabled(True)
        self.save_experiment_button.setEnabled(True)
        self.progress.setValue(100)

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
            self.error_view.set_title(f"Intermediate: {key}")
            self.error_view.set_array(array, fit=True)

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
        self.progress.setFormat(
            self._tr("Local output failed safely: {error}").format(error=error)
        )
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
            "input_shape": list(self.reference.shape),
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
        self.service.close()
        self.output_runner.shutdown(wait=False, cancel_pending=True)
        super().close()


class ModelPage(_BilingualPage, QWidget):
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
        self._display_transform: TransformRecord | None = None
        self._summary_segments: list[tuple[str, str]] = [("owned", MODEL_SUMMARY_INTRO)]
        self._rendered_summary_text = MODEL_SUMMARY_INTRO
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 8, 0, 0)
        layout.setSpacing(10)
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
        layout.addWidget(self.resource_status)

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
        row.addWidget(self.manifest_button, 0, 0)
        row.addWidget(self.load_button, 0, 1)
        row.addWidget(self.run_button, 1, 0)
        row.addWidget(self.cancel_button, 1, 1)
        row.setColumnStretch(0, 1)
        row.setColumnStretch(1, 1)
        row.setColumnStretch(2, 2)
        layout.addWidget(actions)
        self.progress = QProgressBar()
        self.progress.setVisible(False)
        layout.addWidget(self.progress)
        self.summary = QPlainTextEdit()
        self.summary.setReadOnly(True)
        self.summary.setPlainText(MODEL_SUMMARY_INTRO)
        layout.addWidget(self.summary, 1)
        comparison = QSplitter(Qt.Horizontal)
        self.model_input_view = ImageView("Current model input")
        self.model_output_view = ImageView("Typed model visualization")
        self.model_output_tabs = QTabWidget()
        self.model_output_tabs.setObjectName("modelOutputTabs")
        self.model_output_tabs.addTab(self.model_output_view, "Visualization")
        comparison.addWidget(self.model_input_view)
        comparison.addWidget(self.model_output_tabs)
        comparison.setStretchFactor(0, 1)
        comparison.setStretchFactor(1, 1)
        comparison.setChildrenCollapsible(False)
        layout.addWidget(comparison, 2)
        self._refresh_model_resource_status()

    def _language_changed(self) -> None:
        """Retranslate app-owned summary prose while preserving plugin-provided values."""

        if self.summary.toPlainText() != self._rendered_summary_text:
            self._summary_segments = [("external", self.summary.toPlainText())]
        self._render_summary()
        self._refresh_model_resource_status()

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

    def _manifest_is_sam(self) -> bool:
        if self.manifest is None:
            return False
        identity = " ".join(
            part
            for part in (
                self.manifest.name,
                self.manifest.family,
                self.manifest.source.name,
                self.manifest.source.model_id or "",
            )
            if part
        ).lower()
        normalized = "".join(character if character.isalnum() else " " for character in identity)
        words = set(normalized.split())
        return (
            "sam" in words
            or "sam2" in words
            or "medsam" in words
            or "segment anything" in identity
        )

    def _resource_link(self, url: str, label: str) -> str:
        return (
            f'<a href="{escape(url, quote=True)}" '
            f'style="color:#155eef;text-decoration:none;">{escape(label)} ↗</a>'
        )

    def _refresh_model_resource_status(self) -> None:
        if self.manifest is None:
            link = self._resource_link(
                SAM_REPOSITORY_URL,
                self._tr("SAM download & setup"),
            )
            self.resource_status.setText(
                f"<b>{escape(self._tr('No model configured'))}</b>"
                f"&nbsp;&nbsp;·&nbsp;&nbsp;{escape(self._tr('Choose a model manifest to begin.'))} "
                f"{link}<br><span style=\"color:#667085;\">"
                f"{escape(self._tr('A compatible plugin manifest is required.'))}</span>"
            )
            return

        missing = self._missing_weight_specs()
        if self._plugin_ready and not missing:
            loaded_detail = escape(
                self._tr("Loaded from local paths; no files were downloaded or copied.")
            )
            self.resource_status.setText(
                f"<b>{escape(self._tr('Model ready'))}</b>"
                f"&nbsp;&nbsp;·&nbsp;&nbsp;"
                f"{loaded_detail}"
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
        if repository_url is None and self._manifest_is_sam():
            repository_url = SAM_REPOSITORY_URL
        detail = self._tr(
            "Download it, then configure its local path as described by the plugin."
        )
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
        self._display_transform = None
        self.model_input_view.clear_image()
        self._reset_artifact_views()
        self.run_button.setEnabled(self.service.ready and self._source_available)

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
        self.model_input_view.clear_image()
        self._reset_artifact_views()
        self._replace_summary(segments)
        self.manifest = manifest
        self.load_button.setEnabled(True)
        self.run_button.setEnabled(False)
        self.cancel_button.setEnabled(False)
        self._refresh_model_resource_status()

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
        self.run_button.setEnabled(self._source_available)
        self.cancel_button.setEnabled(False)
        self._refresh_model_resource_status()
        self._append_summary(
            "owned",
            "\n\nModel loaded from its existing external path; no files were downloaded or "
            "copied.",
        )

    def _run_plugin(self) -> None:
        if not self.service.ready or self.manifest is None:
            return
        if len(self.manifest.inputs) != 1:
            QMessageBox.warning(
                self,
                self._tr("Multi-input plugin"),
                self._tr(
                    "This teaching page can supply one current image. "
                    "Use the Python API for multi-input/prompts."
                ),
            )
            return
        try:
            model_input = self.source()
        except BaseException as exc:
            QMessageBox.warning(self, self._tr("No model input"), str(exc))
            return
        if isinstance(model_input, RasterImage2D):
            self._last_input = np.asarray(model_input.array)
            self._display_transform = model_input.transform_record
        else:
            self._last_input = np.asarray(model_input)
            self._display_transform = None
        source_view = self._overlay_source(self._last_input)
        self.model_input_view.set_array(source_view, fit=True)
        self.model_output_view.set_array(source_view, fit=True)
        input_name = self.manifest.inputs[0].name
        input_context = self.input_context()

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
        self.run_button.setEnabled(self._source_available)
        self.cancel_button.setEnabled(False)
        self._operation_stage = None
        artifacts = self.service.visualize(result)
        self._render_artifacts(artifacts)
        duration = result.provenance.duration_ms
        duration_text = "not reported" if duration is None else f"{duration:.3f} ms"
        self._append_summary(
            "owned",
            "\n\nPrediction result:\n"
            f"  Type: {type(result).__name__}\n"
            f"  Task: {result.task.value}\n"
            f"  Runtime: {result.provenance.runtime.value}\n"
            f"  Duration: {duration_text}\n"
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
                view.set_array(source, fit=True)
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
                view.set_array(source, fit=True)
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
                view.set_array(source, fit=True)
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
        self.run_button.setEnabled(self._plugin_ready and self._source_available)
        self._refresh_model_resource_status()
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
        self.run_button.setEnabled(self._plugin_ready and self._source_available)
        self.cancel_button.setEnabled(False)
        if self.service.reload_required:
            self._append_summary(
                "owned",
                "\n\nThe cancelled Python adapter must be explicitly loaded again.",
            )

    def _close_plugin(self) -> None:
        self.service.unload()
        self._plugin_ready = False

    def close(self) -> None:  # type: ignore[override]
        self._cancel()
        self.service.close()
        self.runner.shutdown(wait=False, cancel_pending=True)
        super().close()


class AssistantPage(_BilingualPage, QWidget):
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
        self._authorized_image_destinations: set[tuple[str, str]] = set()
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
            "Credentials stay in your environment or system keyring."
        )
        disclaimer.setWordWrap(True)
        disclaimer.setObjectName("warningBanner")
        layout.addWidget(disclaimer)

        self.workspace_splitter = QSplitter(Qt.Horizontal)
        self.settings_scroll = QScrollArea()
        self.settings_scroll.setWidgetResizable(True)
        self.settings_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.settings_scroll.setMinimumWidth(280)
        self.settings_scroll.setAccessibleName("Assistant provider setup")
        settings = QWidget()
        settings_layout = QVBoxLayout(settings)
        settings_layout.setContentsMargins(0, 0, 8, 0)
        self.config_group = QGroupBox("Provider configuration")
        form = QFormLayout(self.config_group)
        form.setRowWrapPolicy(QFormLayout.WrapLongRows)
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
        self.credential = QLineEdit()
        self.credential.setAccessibleName("Credential environment or keyring reference")
        self.network_enabled = QCheckBox("Enable network")
        self.network_enabled.stateChanged.connect(self._update_cloud_status)
        self.vision = QCheckBox("Vision input")
        self.vision.stateChanged.connect(self._vision_changed)
        self.send_preview = QCheckBox("Attach visible slice")
        self.send_preview.setChecked(False)
        self.send_preview.stateChanged.connect(self._update_cloud_status)
        self.revoke_button = QPushButton("Revoke permission")
        self.revoke_button.clicked.connect(self._revoke_image_authorization)
        self.cloud_status = QLabel("Cloud image transfer: OFF")
        self.cloud_status.setObjectName("cloudStatus")
        self.cloud_status.setWordWrap(True)
        self.cloud_status.setAccessibleName("Cloud image transfer status")
        form.addRow("Provider", self.provider)
        form.addRow("Model ID", self.model)
        form.addRow("Endpoint", self.endpoint)
        form.addRow("Credential reference", self.credential)
        form.addRow(self.network_enabled)
        form.addRow(self.vision)
        form.addRow(self.send_preview)
        form.addRow(self.revoke_button)
        form.addRow(self.cloud_status)
        settings_layout.addWidget(self.config_group)
        privacy = QLabel(
            "Image sharing sends only the visible rendered PNG. Review it for burned-in private "
            "text before authorizing a destination."
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
            "Ask about an imaging concept, a reconstruction parameter, or the visible result. "
            "The latest answer will appear here with Markdown formatting."
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
        self.shortcut_hint = QLabel("Ctrl/⌘+Enter to send")
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
        self.send_button.setToolTip("Send to the configured provider (Ctrl/⌘+Enter)")
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
        composer_layout.addWidget(self.progress)
        conversation_layout.addWidget(self.composer)
        self.workspace_splitter.addWidget(conversation)
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

    def _language_changed(self) -> None:
        self._update_cloud_status()
        self._update_character_count()
        self._update_settings_button()
        self.request_status.setText(self._tr(self._request_status_source))
        if self._last_prompt:
            self.question_context.setText(
                f"{self._tr('Question')}: {self._last_prompt}"
            )
        if self._answer_status_source is not None:
            self.answer.setPlainText(self._tr(self._answer_status_source))

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
        self.character_count.setText(
            self._tr("{count} characters").format(count=count)
        )

    def _update_send_button(self, _value: object = None) -> None:
        required_fields = (self.model, self.endpoint, self.credential)
        validate = self.network_enabled.isChecked()
        for field in required_fields:
            state = "error" if validate and not field.text().strip() else "normal"
            _set_dynamic_property(field, "validation", state)
        ready = bool(
            self._active_task is None
            and self.network_enabled.isChecked()
            and self.model.text().strip()
            and self.endpoint.text().strip()
            and self.credential.text().strip()
            and self.prompt.toPlainText().strip()
        )
        self.send_button.setEnabled(ready)

    def _authorization_key(self) -> tuple[str, str]:
        return (self.provider.currentText(), self.endpoint.text().strip())

    def _update_cloud_status(self) -> None:
        enabled = self.network_enabled.isChecked() and self.send_preview.isChecked()
        authorized = self._authorization_key() in self._authorized_image_destinations
        if not enabled and authorized:
            status = self._tr("OFF — destination authorization retained")
        elif not enabled:
            status = self._tr("OFF")
        elif authorized:
            status = self._tr("ON — destination authorized")
        else:
            status = self._tr("ON — confirmation required before transfer")
        self.cloud_status.setText(
            self._tr("Cloud image transfer: {status}").format(status=status)
        )
        if not enabled:
            state = "safe"
        elif authorized:
            state = "active"
        else:
            state = "pending"
        _set_dynamic_property(self.cloud_status, "state", state)
        self.revoke_button.setEnabled(authorized and self._active_task is None)

    def _revoke_image_authorization(self) -> None:
        self._authorized_image_destinations.discard(self._authorization_key())
        self._update_cloud_status()

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
            preview = None
            if self.send_preview.isChecked():
                if not self.vision.isChecked():
                    raise ViewerError(
                        "Declare vision capability before authorizing an image preview."
                    )
                if not self._source_available:
                    raise ViewerError("Open an image before attaching the visible slice.")
                destination = self._authorization_key()
                if destination not in self._authorized_image_destinations:
                    confirmation = QMessageBox.warning(
                        self,
                        self._tr("Authorize cloud image transfer"),
                        self._tr(
                            "Only the currently rendered PNG preview will be sent to this "
                            "provider endpoint. Burned-in text may contain private information. "
                            "Keep this destination authorized until explicitly revoked?"
                        ),
                        QMessageBox.Yes | QMessageBox.No,
                        QMessageBox.No,
                    )
                    if confirmation != QMessageBox.Yes:
                        return
                    self._authorized_image_destinations.add(destination)
                    self._update_cloud_status()
                preview = RenderedPreview.from_png(self.preview_source())
                self.service.authorize_image_transfer(provider)
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
        if self._active_task is not None:
            self._active_task.cancel()
        self.runner.shutdown(wait=False, cancel_pending=True)
        super().close()


class TeachingExperimentPage(_BilingualPage, QWidget):
    """A structured, data-free learning path generated entirely at runtime."""

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
        introduction = QLabel(
            "Choose an experiment to review its principle, parameters, expected result, "
            "and reflection prompt. No medical dataset is bundled."
        )
        introduction.setWordWrap(True)
        root.addWidget(introduction)
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

    def _language_changed(self) -> None:
        self._show_experiment(self.experiment_selector.currentIndex())


class OpenMedVisionXWindow(QMainWindow):
    TAB_TITLES = (
        "Images",
        "CT Lab",
        "Models",
        "Learn",
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
        self.setMinimumSize(1024, 680)
        self.setStyleSheet(APP_STYLE)
        icon_pixmap = _brand_pixmap("logo_pure.png", 96)
        if not icon_pixmap.isNull():
            self.setWindowIcon(QIcon(icon_pixmap))
        central = QWidget()
        central.setObjectName("appShell")
        root = QVBoxLayout(central)
        root.setContentsMargins(18, 16, 18, 12)
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
        self.privacy_label.setMaximumWidth(200)
        self.privacy_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
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
        self.tabs.tabBar().setExpanding(False)
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
        self.learning = TeachingExperimentPage()
        self.assistant = AssistantPage(self.viewer.rendered_preview_png)
        for page, tab_title in zip(
            (self.viewer, self.reconstruction, self.models, self.learning, self.assistant),
            self.TAB_TITLES,
            strict=True,
        ):
            self.tabs.addTab(page, tab_title)
        root.addWidget(self.tabs, 1)
        self.setCentralWidget(central)
        self.viewer.statusChanged.connect(self._show_status_message)
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
            shortcut.activated.connect(
                lambda selected=index: self.tabs.setCurrentIndex(selected)
            )
            self._workspace_shortcuts.append(shortcut)
        self.set_language("en")

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
        self.privacy_label.setText(
            translate("Local · Research", language)
        )
        self.privacy_label.setToolTip(
            translate("Local first · Research only", language)
        )
        self.language_button.setAccessibleName(translate("Switch language", language))
        self.tabs.setAccessibleName(translate("OpenMedVisionX workspaces", language))
        for index, title in enumerate(self.TAB_TITLES):
            self.tabs.setTabText(index, translate(title, language))
        for page in (
            self.viewer,
            self.reconstruction,
            self.models,
            self.learning,
            self.assistant,
        ):
            page.set_language(language)
        if language == "en":
            self.language_button.setText("中文")
            self.language_button.setToolTip("切换到中文")
        else:
            self.language_button.setText("English")
            self.language_button.setToolTip("Switch to English")
        self.statusBar().showMessage(translate(self._status_source, language))

    def _show_status_message(self, message: str) -> None:
        self._status_source = translate(message, "en")
        self.statusBar().showMessage(translate(self._status_source, self._language))

    def closeEvent(self, event: QCloseEvent) -> None:
        self.viewer.close()
        self.reconstruction.close()
        self.assistant.close()
        self.models.close()
        event.accept()
