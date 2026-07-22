"""Privacy-minimized Qt browser for the immutable study/layer domain model.

The widget is deliberately presentation-only.  It never replaces a layer or
changes an :class:`ImageStudy`; interaction is expressed through request
signals so the owning page can create a new immutable study revision and then
call :meth:`StudyLayersPanel.set_study`.
"""

from __future__ import annotations

import re
from typing import Final

from PyQt5.QtCore import QSize, Qt, pyqtSignal
from PyQt5.QtGui import QColor, QIcon, QPixmap, QResizeEvent
from PyQt5.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QSplitter,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ..domain.studies import (
    ContourLayer,
    GeometryMatchStatus,
    ImageSeries,
    ImageStudy,
    Layer,
    LayerCreator,
    LayerValidationState,
    SegmentationLayer,
    SourceFormat,
    VolumeLayer,
)
from .i18n import Language

_COPY: Final[dict[str, tuple[str, str]]] = {
    "title": ("Study layers", "检查与图层"),
    "intro": (
        "Choose a layer, then adjust its presentation. Source paths, DICOM identifiers, "
        "hashes, and patient fields are never shown.",
        "选择图层后调整其显示方式。这里绝不显示源路径、DICOM 标识符、哈希或患者字段。",
    ),
    "contents": ("Study contents", "检查内容"),
    "state": ("State", "状态"),
    "current_study": ("Current study · {count} series", "当前检查 · {count} 个序列"),
    "series": ("Series {index} · {modality}", "序列 {index} · {modality}"),
    "series_summary": ("{shape} · {count} layers", "{shape} · {count} 个图层"),
    "no_study": ("No study loaded", "尚未加载检查"),
    "no_study_help": (
        "Open a supported image study to inspect its series and layers.",
        "打开受支持的影像检查，即可查看序列与图层。",
    ),
    "no_series": ("This study contains no series", "此检查中没有序列"),
    "no_layers": ("No layers", "没有图层"),
    "select_layer": (
        "Select a layer to inspect and control it.",
        "请选择一个图层以查看并控制其显示。",
    ),
    "details": ("Layer details", "图层详情"),
    "presentation": ("Presentation", "显示设置"),
    "visible": ("Visible", "可见"),
    "locked": ("Locked", "锁定"),
    "opacity": ("Opacity", "不透明度"),
    "labels": ("Segmentation labels", "分割标签"),
    "type": ("Type", "类型"),
    "created_by": ("Created by", "创建方式"),
    "validation": ("Validation", "验证状态"),
    "format": ("Source format", "源格式"),
    "generated_format": ("Generated", "生成数据"),
    "plugin_format": ("Plugin", "插件"),
    "geometry": ("Geometry", "几何信息"),
    "revision": ("Revision", "修订版本"),
    "volume": ("Image volume", "影像体数据"),
    "segmentation": ("Segmentation", "分割"),
    "contour": ("Contours", "轮廓"),
    "import": ("Imported", "导入"),
    "user": ("User-created", "用户创建"),
    "algorithm": ("Algorithm", "算法"),
    "model": ("Model", "模型"),
    "llm": ("AI assistant", "AI 助手"),
    "pending": ("Pending review", "等待检查"),
    "validated": ("Validated", "已验证"),
    "rejected": ("Rejected", "已拒绝"),
    "shown": ("Visible", "可见"),
    "hidden": ("Hidden", "隐藏"),
    "shown_locked": ("Visible · Locked", "可见 · 已锁定"),
    "hidden_locked": ("Hidden · Locked", "隐藏 · 已锁定"),
    "matched": ("Matched", "已匹配"),
    "resampled": ("Resampled", "已重采样"),
    "requires_resampling": ("Requires resampling", "需要重采样"),
    "reference_mismatch": ("Reference mismatch", "引用不匹配"),
    "unresolved_reference": ("Reference unresolved", "引用未解析"),
    "series_grid": ("Series grid", "序列网格"),
    "world_contours": ("World-space contours", "世界坐标轮廓"),
    "roi_count": ("{count} regions", "{count} 个区域"),
    "label_fallback": ("Label {value}", "标签 {value}"),
    "layer_fallback": ("Layer {index}", "图层 {index}"),
    "label_tooltip": (
        "{name} · color {color} · toggle visibility",
        "{name} · 颜色 {color} · 切换可见性",
    ),
    "tree_accessible": ("Study, series, and layer browser", "检查、序列与图层浏览器"),
    "tree_description": (
        "Use arrow keys to navigate. Identifiers and patient fields are hidden.",
        "使用方向键导航。标识符与患者字段均已隐藏。",
    ),
    "visible_description": (
        "Request a visibility change for the active layer.",
        "请求更改当前图层的可见性。",
    ),
    "locked_description": (
        "Request a lock-state change for the active layer.",
        "请求更改当前图层的锁定状态。",
    ),
    "opacity_description": (
        "Request active-layer opacity from 0 to 100 percent.",
        "请求将当前图层不透明度设为 0% 至 100%。",
    ),
    "labels_accessible": (
        "Segmentation label visibility and color list",
        "分割标签可见性与颜色列表",
    ),
    "labels_description": (
        "Use Space to request visibility for one label. The color is also written as text.",
        "按空格可请求切换单个标签的可见性；颜色同时以文本说明。",
    ),
}

_KIND_ROLE = Qt.UserRole
_SERIES_ID_ROLE = Qt.UserRole + 1
_LAYER_ID_ROLE = Qt.UserRole + 2
_LABEL_VALUE_ROLE = Qt.UserRole + 3
_LAYER_ITEM = "layer"
_UNSAFE_UID = re.compile(r"(?<![\d.])\d+(?:\.\d+){2,}(?![\d.])")
_UNSAFE_PHI_FIELD = re.compile(
    r"\b(?:patient(?:\s*name|\s*id)?|birth\s*date|accession(?:\s*number)?|mrn)\s*[:=]",
    re.IGNORECASE,
)


def _copy(key: str, language: Language) -> str:
    return _COPY[key][1 if language == "zh_CN" else 0]


def _safe_display_name(value: str, fallback: str) -> str:
    """Return a short label without rendering path/UID/obvious PHI payloads."""

    normalized = " ".join(str(value).split()).strip()
    is_path_like = (
        "/" in normalized
        or "\\" in normalized
        or normalized.lower().startswith(("file:", "http:", "https:"))
        or bool(re.match(r"^[A-Za-z]:", normalized))
    )
    if (
        not normalized
        or is_path_like
        or _UNSAFE_UID.search(normalized)
        or _UNSAFE_PHI_FIELD.search(normalized)
    ):
        return fallback
    if len(normalized) > 100:
        return f"{normalized[:99]}…"
    return normalized


def _format_source(source_format: SourceFormat, language: Language) -> str:
    return {
        SourceFormat.PNG: "PNG",
        SourceFormat.JPEG: "JPEG",
        SourceFormat.TIFF: "TIFF",
        SourceFormat.DICOM: "DICOM",
        SourceFormat.DICOM_SEG: "DICOM SEG",
        SourceFormat.RTSTRUCT: "DICOM RTSTRUCT",
        SourceFormat.NIFTI: "NIfTI",
        SourceFormat.GENERATED: _copy("generated_format", language),
        SourceFormat.PLUGIN: _copy("plugin_format", language),
    }[source_format]


class StudyLayersPanel(QWidget):
    """Compact browser and presentation-control surface for an ``ImageStudy``.

    Signal arguments use local opaque series/layer IDs for routing.  No domain
    object is changed by this widget.
    """

    activeLayerChanged = pyqtSignal(str, str)
    layerVisibilityChangeRequested = pyqtSignal(str, str, bool)
    layerLockChangeRequested = pyqtSignal(str, str, bool)
    layerOpacityChangeRequested = pyqtSignal(str, str, float)
    labelVisibilityChangeRequested = pyqtSignal(str, str, int, bool)

    _NARROW_WIDTH = 680

    def __init__(
        self,
        parent: QWidget | None = None,
        *,
        language: Language = "en",
    ) -> None:
        super().__init__(parent)
        self._language: Language = "en"
        self._study: ImageStudy | None = None
        self._active_series_id: str | None = None
        self._active_layer_id: str | None = None
        self._layer_items: dict[tuple[str, str], QTreeWidgetItem] = {}
        self._updating = False
        self.setObjectName("studyLayersPanel")
        self._build_ui()
        self.set_language(language)

    @property
    def study(self) -> ImageStudy | None:
        return self._study

    @property
    def active_layer_key(self) -> tuple[str, str] | None:
        if self._active_series_id is None or self._active_layer_id is None:
            return None
        return self._active_series_id, self._active_layer_id

    @property
    def active_layer(self) -> Layer | None:
        found = self._find_active()
        return None if found is None else found[1]

    def minimumSizeHint(self) -> QSize:  # noqa: N802 - Qt API
        return QSize(0, 0)

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(10)

        self.title_label = QLabel()
        self.title_label.setObjectName("pageTitle")
        self.intro_label = QLabel()
        self.intro_label.setObjectName("infoBanner")
        self.intro_label.setWordWrap(True)
        root.addWidget(self.title_label)
        root.addWidget(self.intro_label)

        self.splitter = QSplitter(Qt.Horizontal)
        self.splitter.setObjectName("studyLayerSplitter")
        self.splitter.setChildrenCollapsible(False)
        self.splitter.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Expanding)
        self.splitter.addWidget(self._build_tree_panel())
        self.splitter.addWidget(self._build_detail_scroll())
        self.splitter.setStretchFactor(0, 2)
        self.splitter.setStretchFactor(1, 3)
        self.splitter.setSizes([300, 450])
        root.addWidget(self.splitter, 1)

        QWidget.setTabOrder(self.tree, self.visible_checkbox)
        QWidget.setTabOrder(self.visible_checkbox, self.locked_checkbox)
        QWidget.setTabOrder(self.locked_checkbox, self.opacity_slider)
        QWidget.setTabOrder(self.opacity_slider, self.label_list)

    def _build_tree_panel(self) -> QWidget:
        panel = QFrame()
        panel.setObjectName("studyLayerTreeSurface")
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        self.tree = QTreeWidget()
        self.tree.setObjectName("studyLayerTree")
        self.tree.setColumnCount(2)
        self.tree.setAlternatingRowColors(False)
        self.tree.setAnimated(False)
        self.tree.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.tree.setSelectionMode(QAbstractItemView.SingleSelection)
        self.tree.setUniformRowHeights(True)
        self.tree.header().setStretchLastSection(False)
        self.tree.header().setSectionResizeMode(0, self.tree.header().Stretch)
        self.tree.header().setSectionResizeMode(1, self.tree.header().ResizeToContents)
        self.tree.currentItemChanged.connect(self._on_current_item_changed)
        layout.addWidget(self.tree)
        return panel

    def _build_detail_scroll(self) -> QScrollArea:
        self.detail_scroll = QScrollArea()
        self.detail_scroll.setObjectName("studyLayerDetailScroll")
        self.detail_scroll.setWidgetResizable(True)
        self.detail_scroll.setFrameShape(QFrame.NoFrame)

        content = QWidget()
        content.setObjectName("studyLayerDetailSurface")
        content.setMinimumSize(0, 0)
        layout = QVBoxLayout(content)
        layout.setContentsMargins(2, 0, 2, 0)
        layout.setSpacing(10)

        self.detail_group = QGroupBox()
        detail_layout = QVBoxLayout(self.detail_group)
        detail_layout.setSpacing(8)
        self.layer_name_label = QLabel()
        self.layer_name_label.setObjectName("sectionTitle")
        self.layer_name_label.setWordWrap(True)
        self.empty_detail_label = QLabel()
        self.empty_detail_label.setObjectName("mutedText")
        self.empty_detail_label.setWordWrap(True)
        self.detail_form = QFormLayout()
        self.detail_form.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
        self._detail_field_labels: dict[str, QLabel] = {}
        self._detail_value_labels: dict[str, QLabel] = {}
        for key in ("type", "created_by", "validation", "format", "geometry", "revision"):
            field_label = QLabel()
            value_label = QLabel()
            value_label.setWordWrap(True)
            value_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
            self.detail_form.addRow(field_label, value_label)
            self._detail_field_labels[key] = field_label
            self._detail_value_labels[key] = value_label
        detail_layout.addWidget(self.layer_name_label)
        detail_layout.addWidget(self.empty_detail_label)
        detail_layout.addLayout(self.detail_form)

        self.presentation_group = QGroupBox()
        presentation_layout = QVBoxLayout(self.presentation_group)
        presentation_layout.setSpacing(7)
        toggles = QHBoxLayout()
        toggles.setContentsMargins(0, 0, 0, 0)
        self.visible_checkbox = QCheckBox()
        self.locked_checkbox = QCheckBox()
        self.visible_checkbox.clicked.connect(self._request_visibility)
        self.locked_checkbox.clicked.connect(self._request_lock)
        toggles.addWidget(self.visible_checkbox)
        toggles.addWidget(self.locked_checkbox)
        toggles.addStretch(1)
        presentation_layout.addLayout(toggles)
        opacity_row = QHBoxLayout()
        opacity_row.setContentsMargins(0, 0, 0, 0)
        self.opacity_label = QLabel()
        self.opacity_slider = QSlider(Qt.Horizontal)
        self.opacity_slider.setRange(0, 100)
        self.opacity_slider.setSingleStep(5)
        self.opacity_slider.setPageStep(10)
        self.opacity_slider.valueChanged.connect(self._request_opacity)
        self.opacity_value_label = QLabel("—")
        self.opacity_value_label.setMinimumWidth(42)
        self.opacity_value_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.opacity_label.setBuddy(self.opacity_slider)
        opacity_row.addWidget(self.opacity_label)
        opacity_row.addWidget(self.opacity_slider, 1)
        opacity_row.addWidget(self.opacity_value_label)
        presentation_layout.addLayout(opacity_row)

        self.labels_group = QGroupBox()
        labels_layout = QVBoxLayout(self.labels_group)
        labels_layout.setContentsMargins(8, 8, 8, 8)
        self.label_list = QListWidget()
        self.label_list.setObjectName("segmentationLabelList")
        self.label_list.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.label_list.setSelectionMode(QAbstractItemView.SingleSelection)
        self.label_list.setIconSize(QSize(14, 14))
        self.label_list.setUniformItemSizes(True)
        self.label_list.itemChanged.connect(self._request_label_visibility)
        labels_layout.addWidget(self.label_list)

        layout.addWidget(self.detail_group)
        layout.addWidget(self.presentation_group)
        layout.addWidget(self.labels_group)
        layout.addStretch(1)
        self.detail_scroll.setWidget(content)
        return self.detail_scroll

    def set_language(self, language: Language) -> None:
        if language not in {"en", "zh_CN"}:
            raise ValueError(f"Unsupported UI language: {language}")
        self._language = language
        self.title_label.setText(_copy("title", language))
        self.intro_label.setText(_copy("intro", language))
        self.tree.setHeaderLabels([_copy("contents", language), _copy("state", language)])
        self.tree.setAccessibleName(_copy("tree_accessible", language))
        self.tree.setAccessibleDescription(_copy("tree_description", language))
        self.detail_group.setTitle(_copy("details", language))
        self.presentation_group.setTitle(_copy("presentation", language))
        self.labels_group.setTitle(_copy("labels", language))
        self.visible_checkbox.setText(_copy("visible", language))
        self.visible_checkbox.setAccessibleName(_copy("visible", language))
        self.visible_checkbox.setAccessibleDescription(_copy("visible_description", language))
        self.locked_checkbox.setText(_copy("locked", language))
        self.locked_checkbox.setAccessibleName(_copy("locked", language))
        self.locked_checkbox.setAccessibleDescription(_copy("locked_description", language))
        self.opacity_label.setText(_copy("opacity", language))
        self.opacity_slider.setAccessibleName(_copy("opacity", language))
        self.opacity_slider.setAccessibleDescription(_copy("opacity_description", language))
        self.label_list.setAccessibleName(_copy("labels_accessible", language))
        self.label_list.setAccessibleDescription(_copy("labels_description", language))
        for key, field_label in self._detail_field_labels.items():
            field_label.setText(_copy(key, language))
        self._rebuild_tree()

    def set_study(self, study: ImageStudy | None) -> None:
        if study is not None and not isinstance(study, ImageStudy):
            raise TypeError("study must be an ImageStudy or None")
        self._study = study
        self._rebuild_tree()

    def clear(self) -> None:
        self.set_study(None)

    def set_active_layer(
        self,
        series_id: str,
        layer_id: str,
        *,
        emit_signal: bool = False,
    ) -> bool:
        item = self._layer_items.get((series_id, layer_id))
        if item is None:
            return False
        previous = self._updating
        self._updating = not emit_signal
        try:
            self.tree.setCurrentItem(item)
            self.tree.scrollToItem(item, QAbstractItemView.EnsureVisible)
            if not emit_signal:
                self._active_series_id = series_id
                self._active_layer_id = layer_id
                found = self._find_active()
                self._show_layer(
                    None if found is None else found[0], None if found is None else found[1]
                )
        finally:
            self._updating = previous
        return True

    def _rebuild_tree(self) -> None:
        previous_key = self.active_layer_key
        previous_updating = self._updating
        self._updating = True
        self.tree.blockSignals(True)
        try:
            self.tree.clear()
            self._layer_items.clear()
            first_key: tuple[str, str] | None = None
            study = self._study
            if study is None:
                empty = QTreeWidgetItem([_copy("no_study", self._language), ""])
                empty.setFlags(Qt.NoItemFlags)
                self.tree.addTopLevelItem(empty)
                self._active_series_id = None
                self._active_layer_id = None
                self._show_layer(None, None)
                return

            root = QTreeWidgetItem(
                [
                    _copy("current_study", self._language).format(count=len(study.series)),
                    "",
                ]
            )
            root.setFlags(Qt.ItemIsEnabled)
            self.tree.addTopLevelItem(root)
            if not study.series:
                empty = QTreeWidgetItem([_copy("no_series", self._language), ""])
                empty.setFlags(Qt.NoItemFlags)
                root.addChild(empty)

            for series_index, series in enumerate(study.series, start=1):
                series_item = QTreeWidgetItem(
                    [
                        _copy("series", self._language).format(
                            index=series_index,
                            modality=series.modality,
                        ),
                        _copy("series_summary", self._language).format(
                            shape=self._shape_text(series.shape),
                            count=len(series.layers),
                        ),
                    ]
                )
                series_item.setFlags(Qt.ItemIsEnabled)
                root.addChild(series_item)
                if not series.layers:
                    empty = QTreeWidgetItem([_copy("no_layers", self._language), ""])
                    empty.setFlags(Qt.NoItemFlags)
                    series_item.addChild(empty)
                for layer_index, layer in enumerate(series.layers, start=1):
                    fallback = _copy("layer_fallback", self._language).format(index=layer_index)
                    item = QTreeWidgetItem(
                        [
                            _safe_display_name(layer.name, fallback),
                            self._layer_state_text(layer),
                        ]
                    )
                    item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
                    item.setData(0, _KIND_ROLE, _LAYER_ITEM)
                    item.setData(0, _SERIES_ID_ROLE, series.series_id)
                    item.setData(0, _LAYER_ID_ROLE, layer.layer_id)
                    item.setToolTip(0, self._layer_accessible_description(layer))
                    item.setData(
                        0, Qt.AccessibleDescriptionRole, self._layer_accessible_description(layer)
                    )
                    series_item.addChild(item)
                    key = (series.series_id, layer.layer_id)
                    self._layer_items[key] = item
                    if first_key is None:
                        first_key = key

            root.setExpanded(True)
            for index in range(root.childCount()):
                root.child(index).setExpanded(True)
            selected_key = previous_key if previous_key in self._layer_items else first_key
            if selected_key is None:
                self._active_series_id = None
                self._active_layer_id = None
                self._show_layer(None, None)
            else:
                self._active_series_id, self._active_layer_id = selected_key
                self.tree.setCurrentItem(self._layer_items[selected_key])
                found = self._find_active()
                self._show_layer(
                    None if found is None else found[0], None if found is None else found[1]
                )
        finally:
            self.tree.blockSignals(False)
            self._updating = previous_updating

    def _on_current_item_changed(
        self,
        current: QTreeWidgetItem | None,
        _previous: QTreeWidgetItem | None,
    ) -> None:
        if current is None or current.data(0, _KIND_ROLE) != _LAYER_ITEM:
            return
        series_id = current.data(0, _SERIES_ID_ROLE)
        layer_id = current.data(0, _LAYER_ID_ROLE)
        if not isinstance(series_id, str) or not isinstance(layer_id, str):
            return
        self._active_series_id = series_id
        self._active_layer_id = layer_id
        found = self._find_active()
        self._show_layer(None if found is None else found[0], None if found is None else found[1])
        if not self._updating:
            self.activeLayerChanged.emit(series_id, layer_id)

    def _find_active(self) -> tuple[ImageSeries, Layer] | None:
        if self._study is None or self._active_series_id is None or self._active_layer_id is None:
            return None
        for series in self._study.series:
            if series.series_id != self._active_series_id:
                continue
            for layer in series.layers:
                if layer.layer_id == self._active_layer_id:
                    return series, layer
        return None

    def _show_layer(self, series: ImageSeries | None, layer: Layer | None) -> None:
        self._updating = True
        try:
            if series is None or layer is None:
                self.layer_name_label.clear()
                self.empty_detail_label.setText(
                    _copy("no_study_help", self._language)
                    if self._study is None
                    else _copy("select_layer", self._language)
                )
                for value_label in self._detail_value_labels.values():
                    value_label.clear()
                self._set_controls_enabled(False)
                self.labels_group.setVisible(False)
                self.label_list.clear()
                self.opacity_value_label.setText("—")
                return

            fallback = _copy("layer_fallback", self._language).format(index=1)
            safe_name = _safe_display_name(layer.name, fallback)
            self.layer_name_label.setText(safe_name)
            self.layer_name_label.setAccessibleName(safe_name)
            self.layer_name_label.setAccessibleDescription(
                self._layer_accessible_description(layer)
            )
            self.empty_detail_label.clear()
            values = self._detail_values(series, layer)
            for key, value_label in self._detail_value_labels.items():
                value_label.setText(values[key])
            presentation_editable = not (isinstance(layer, VolumeLayer) and layer.is_base_image)
            self._set_controls_enabled(presentation_editable)
            self.visible_checkbox.setChecked(layer.presentation.visible)
            self.locked_checkbox.setChecked(layer.presentation.locked)
            opacity = round(layer.presentation.opacity * 100)
            self.opacity_slider.setValue(opacity)
            self.opacity_value_label.setText(f"{opacity}%")
            self.opacity_slider.setEnabled(presentation_editable and not layer.presentation.locked)
            self._populate_labels(layer)
        finally:
            self._updating = False

    def _set_controls_enabled(self, enabled: bool) -> None:
        self.visible_checkbox.setEnabled(enabled)
        self.locked_checkbox.setEnabled(enabled)
        self.opacity_slider.setEnabled(enabled)

    def _populate_labels(self, layer: Layer) -> None:
        self.label_list.clear()
        is_segmentation = isinstance(layer, SegmentationLayer)
        self.labels_group.setVisible(is_segmentation)
        if not is_segmentation:
            return
        for label in layer.labels:
            fallback = _copy("label_fallback", self._language).format(value=label.value)
            name = _safe_display_name(label.name, fallback)
            item = QListWidgetItem(self._swatch_icon(label.color), f"{label.value} · {name}")
            item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Checked if label.visible else Qt.Unchecked)
            item.setData(_SERIES_ID_ROLE, layer.series_id)
            item.setData(_LAYER_ID_ROLE, layer.layer_id)
            item.setData(_LABEL_VALUE_ROLE, label.value)
            tooltip = _copy("label_tooltip", self._language).format(
                name=name,
                color=label.color,
            )
            item.setToolTip(tooltip)
            item.setData(Qt.AccessibleDescriptionRole, tooltip)
            self.label_list.addItem(item)
        self.label_list.setEnabled(not layer.presentation.locked)

    @staticmethod
    def _swatch_icon(color: str) -> QIcon:
        pixmap = QPixmap(14, 14)
        pixmap.fill(QColor(color))
        return QIcon(pixmap)

    def _detail_values(self, series: ImageSeries, layer: Layer) -> dict[str, str]:
        if isinstance(layer, VolumeLayer):
            layer_type = _copy("volume", self._language)
            volume_geometry = layer.current_geometry
            geometry = (
                f"{self._geometry_text(volume_geometry.shape_zyx, volume_geometry.spacing_xyz)}"
                f" · {_copy('series_grid', self._language)}"
            )
        elif isinstance(layer, SegmentationLayer):
            layer_type = _copy("segmentation", self._language)
            geometry = (
                f"{self._geometry_text(layer.geometry.shape_zyx, layer.geometry.spacing_xyz)}"
                f" · {self._geometry_status_text(layer.geometry_match_status)}"
            )
        elif isinstance(layer, ContourLayer):
            layer_type = _copy("contour", self._language)
            if layer.current_geometry is None:
                geometry = _copy("world_contours", self._language)
            else:
                geometry = self._geometry_text(
                    layer.current_geometry.shape_zyx,
                    layer.current_geometry.spacing_xyz,
                )
            geometry = (
                f"{geometry} · {_copy('roi_count', self._language).format(count=len(layer.rois))}"
            )
        else:  # pragma: no cover - the public Layer union is exhaustive
            layer_type = "—"
            geometry = "—"
        return {
            "type": layer_type,
            "created_by": self._creator_text(layer.created_by),
            "validation": self._validation_text(layer.validation_state),
            "format": _format_source(layer.source.source_format, self._language),
            "geometry": geometry,
            "revision": str(layer.revision),
        }

    def _layer_state_text(self, layer: Layer) -> str:
        if layer.presentation.locked:
            key = "shown_locked" if layer.presentation.visible else "hidden_locked"
        else:
            key = "shown" if layer.presentation.visible else "hidden"
        return _copy(key, self._language)

    def _layer_accessible_description(self, layer: Layer) -> str:
        layer_type = (
            _copy("volume", self._language)
            if isinstance(layer, VolumeLayer)
            else _copy("segmentation", self._language)
            if isinstance(layer, SegmentationLayer)
            else _copy("contour", self._language)
        )
        return f"{layer_type} · {self._layer_state_text(layer)}"

    def _creator_text(self, creator: LayerCreator) -> str:
        return _copy(creator.value, self._language)

    def _validation_text(self, state: LayerValidationState) -> str:
        return _copy(state.value, self._language)

    def _geometry_status_text(self, status: GeometryMatchStatus) -> str:
        key = status.value.replace("-", "_")
        return _copy(key, self._language)

    @staticmethod
    def _shape_text(shape: tuple[int, ...]) -> str:
        return "×".join(str(value) for value in shape)

    @classmethod
    def _geometry_text(
        cls,
        shape: tuple[int, int, int],
        spacing_xyz: tuple[float, float, float],
    ) -> str:
        spacing = "×".join(f"{value:.3g}" for value in spacing_xyz)
        return f"{cls._shape_text(shape)} · {spacing} mm"

    def _request_visibility(self, checked: bool) -> None:
        key = self.active_layer_key
        if self._updating or key is None:
            return
        self.layerVisibilityChangeRequested.emit(key[0], key[1], bool(checked))

    def _request_lock(self, checked: bool) -> None:
        key = self.active_layer_key
        if self._updating or key is None:
            return
        self.layerLockChangeRequested.emit(key[0], key[1], bool(checked))

    def _request_opacity(self, value: int) -> None:
        self.opacity_value_label.setText(f"{value}%")
        key = self.active_layer_key
        if self._updating or key is None:
            return
        self.layerOpacityChangeRequested.emit(key[0], key[1], value / 100.0)

    def _request_label_visibility(self, item: QListWidgetItem) -> None:
        if self._updating:
            return
        series_id = item.data(_SERIES_ID_ROLE)
        layer_id = item.data(_LAYER_ID_ROLE)
        label_value = item.data(_LABEL_VALUE_ROLE)
        if (
            not isinstance(series_id, str)
            or not isinstance(layer_id, str)
            or not isinstance(label_value, int)
        ):
            return
        self.labelVisibilityChangeRequested.emit(
            series_id,
            layer_id,
            label_value,
            item.checkState() == Qt.Checked,
        )

    def resizeEvent(self, event: QResizeEvent) -> None:  # noqa: N802 - Qt API
        orientation = Qt.Vertical if event.size().width() < self._NARROW_WIDTH else Qt.Horizontal
        if self.splitter.orientation() != orientation:
            self.splitter.setOrientation(orientation)
            if orientation == Qt.Horizontal:
                self.splitter.setSizes([300, max(380, event.size().width() - 320)])
            else:
                self.splitter.setSizes([220, 380])
        super().resizeEvent(event)


# Friendly aliases for callers that prefer singular or generic widget naming.
StudyLayerPanel = StudyLayersPanel
StudyLayersWidget = StudyLayersPanel


__all__ = ["StudyLayerPanel", "StudyLayersPanel", "StudyLayersWidget"]
