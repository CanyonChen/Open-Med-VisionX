"""Local review surface for provider-neutral structured AI artifacts.

The widget deliberately contains no provider adapter, network action, credential
field, prompt body, image preview, or filesystem picker.  It presents the
contracts from :mod:`workbench.llm.artifacts` and emits review requests so an
owning page can append an immutable :class:`ArtifactReview` itself.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Final

from PyQt5.QtCore import QSize, Qt, pyqtSignal
from PyQt5.QtGui import QResizeEvent
from PyQt5.QtWidgets import (
    QComboBox,
    QFormLayout,
    QFrame,
    QGridLayout,
    QGroupBox,
    QLabel,
    QPushButton,
    QSizePolicy,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from ..errors import ViewerError
from ..llm.artifacts import (
    ArtifactReviewDecision,
    ArtifactValidationStatus,
    LLMArtifactResponse,
    LLMArtifactType,
    LLMTaskKind,
    LLMTaskRequest,
)
from .i18n import Language


@dataclass(frozen=True, slots=True)
class AssistantTaskTemplate:
    """Safe UI metadata for one exact structured artifact contract."""

    template_id: str
    artifact_type: LLMArtifactType
    compatible_tasks: tuple[LLMTaskKind, ...]
    title: tuple[str, str]
    description: tuple[str, str]

    def localized_title(self, language: Language) -> str:
        return self.title[1 if language == "zh_CN" else 0]

    def localized_description(self, language: Language) -> str:
        return self.description[1 if language == "zh_CN" else 0]


_ALL_TASKS = tuple(LLMTaskKind)

ARTIFACT_TASK_TEMPLATES: Final[tuple[AssistantTaskTemplate, ...]] = (
    AssistantTaskTemplate(
        "text_explanation",
        LLMArtifactType.TEXT,
        _ALL_TASKS,
        ("Explanation · text", "概念解释 · 文本"),
        (
            "UTF-8 text with a language tag and opaque citation IDs. It never becomes an "
            "image layer.",
            "带语言标签与不透明引用 ID 的 UTF-8 文本；它不能创建影像图层。",
        ),
    ),
    AssistantTaskTemplate(
        "class_scores",
        LLMArtifactType.CLASS_SCORES,
        (LLMTaskKind.CLASSIFY,),
        ("Classification · class scores", "分类 · 类别分数"),
        (
            "Typed class scores with declared score semantics, thresholds, and calibration.",
            "类型明确的类别分数，并声明分数语义、阈值与校准方式。",
        ),
    ),
    AssistantTaskTemplate(
        "semantic_labels",
        LLMArtifactType.LABELS,
        (LLMTaskKind.LABELS,),
        ("Semantic labels", "语义标签"),
        (
            "Named labels with optional confidence and opaque region references; no pixel mask.",
            "命名标签可含置信度与不透明区域引用，但不包含像素掩膜。",
        ),
    ),
    AssistantTaskTemplate(
        "mask_2d",
        LLMArtifactType.MASK_2D,
        (LLMTaskKind.SEGMENT,),
        ("Segmentation · 2-D mask", "分割 · 二维掩膜"),
        (
            "A typed mask bound to one exact source slice, geometry, and label schema.",
            "绑定到精确源切片、几何信息与标签架构的类型化掩膜。",
        ),
    ),
    AssistantTaskTemplate(
        "mask_3d",
        LLMArtifactType.MASK_3D,
        (LLMTaskKind.SEGMENT,),
        ("Segmentation · 3-D mask", "分割 · 三维掩膜"),
        (
            "A full-volume mask whose shape and affine must match the referenced volume.",
            "完整体掩膜；其形状与仿射必须匹配引用体数据。",
        ),
    ),
    AssistantTaskTemplate(
        "reconstructed_image",
        LLMArtifactType.RECONSTRUCTED_IMAGE,
        (
            LLMTaskKind.RECONSTRUCT,
            LLMTaskKind.RESTORE,
            LLMTaskKind.ENHANCE,
            LLMTaskKind.GENERATE,
        ),
        ("Image result · reconstructed image", "影像结果 · 重建图像"),
        (
            "A 2-D result with exact source geometry and declared intensity semantics.",
            "带精确源几何信息与已声明强度语义的二维结果。",
        ),
    ),
    AssistantTaskTemplate(
        "reconstructed_volume",
        LLMArtifactType.RECONSTRUCTED_VOLUME,
        (
            LLMTaskKind.RECONSTRUCT,
            LLMTaskKind.RESTORE,
            LLMTaskKind.ENHANCE,
            LLMTaskKind.GENERATE,
        ),
        ("Volume result · reconstructed volume", "体数据结果 · 重建体数据"),
        (
            "A full-volume result with exact source geometry and declared intensity semantics.",
            "带精确源几何信息与已声明强度语义的完整体结果。",
        ),
    ),
)

assert tuple(item.artifact_type for item in ARTIFACT_TASK_TEMPLATES) == tuple(LLMArtifactType)


_COPY: Final[dict[str, tuple[str, str]]] = {
    "title": ("Structured artifact contracts", "结构化产物契约"),
    "boundary": (
        "Integration API preview: the configured chat provider and desktop file picker do not "
        "populate this panel in this release. A trusted host workflow may supply an already "
        "typed request and response through the Python API. This panel sends nothing, creates "
        "no layer, and does not establish clinical validity.",
        "集成 API 预览：当前版本中，已配置的对话服务和桌面文件选择器都不会填充本面板。"
        "受信任的上层工作流可通过 Python API 提供已经类型化的请求与响应。本面板不会发送"
        "数据、不会创建图层，也不代表产物具备临床有效性。",
    ),
    "template_group": ("Task template", "任务模板"),
    "template_label": ("Expected artifact", "预期产物"),
    "template_accessible": (
        "Structured AI artifact task template",
        "结构化 AI 产物任务模板",
    ),
    "template_description": (
        "Choose a contract to inspect. A trusted host integration prepares the exact request.",
        "选择要查看的产物契约；受信任的上层集成负责准备精确请求。",
    ),
    "request_group": ("Request validation", "请求校验"),
    "artifact_group": ("Artifact validation", "产物校验"),
    "task": ("Task", "任务"),
    "inputs": ("Inputs", "输入"),
    "outputs": ("Requested output", "请求输出"),
    "binding": ("Exact binding", "精确绑定"),
    "artifact_type": ("Artifact type", "产物类型"),
    "payload": ("Payload", "载荷"),
    "integrity": ("Provenance & integrity", "来源与完整性"),
    "request_match": ("Request match", "请求匹配"),
    "review": ("Review status", "审查状态"),
    "no_request": ("No typed request loaded", "尚未加载类型化请求"),
    "no_artifact": ("No structured artifact loaded", "尚未加载结构化产物"),
    "not_available": ("Not available", "不可用"),
    "deidentified": ("{count} deidentified · {kinds}", "{count} 项已去标识 · {kinds}"),
    "hash_bound": (
        "Transfer plan + prompt digests bound{evidence}",
        "已绑定传输计划与提示词摘要{evidence}",
    ),
    "evidence_present": (" · physics evidence present", " · 含物理证据"),
    "requested": ("{count} contract(s) · {types}", "{count} 项契约 · {types}"),
    "authenticated": (
        "Authenticated response · SHA-256 verified",
        "响应已认证 · SHA-256 已验证",
    ),
    "warning_count": ("{count} warning(s)", "{count} 条提示"),
    "match_valid": ("Matches the exact typed request", "匹配精确类型化请求"),
    "match_missing": ("Load its typed request before review", "审查前请加载对应类型化请求"),
    "match_invalid": ("Does not match the loaded request", "与已加载请求不匹配"),
    "unverified": ("Unverified · explicit review required", "未验证 · 需要明确审查"),
    "user_confirmed": ("User confirmed", "用户已确认"),
    "rejected": ("Rejected", "已拒绝"),
    "confirm": ("Confirm artifact", "确认产物"),
    "reject": ("Reject artifact", "拒绝产物"),
    "confirm_description": (
        "Request confirmation of the exact loaded artifact; this does not create a layer.",
        "请求确认当前精确产物；此操作不会创建图层。",
    ),
    "reject_description": (
        "Request rejection of the exact loaded artifact; source data remains unchanged.",
        "请求拒绝当前精确产物；源数据保持不变。",
    ),
    "ready_empty": (
        "Choose a contract to inspect. Typed data can only arrive through the integration API.",
        "选择要查看的契约。类型化数据只能通过集成 API 提供。",
    ),
    "ready_review": (
        "The artifact matches its request. Review it explicitly before downstream use.",
        "产物与请求匹配；在后续使用前请明确审查。",
    ),
    "reviewed": (
        "Review is complete. The immutable response remains unchanged here.",
        "审查已完成；本面板不会更改不可变响应。",
    ),
    "blocked_request": (
        "Review blocked until the matching typed request is loaded.",
        "加载匹配的类型化请求后才能审查。",
    ),
    "blocked_mismatch": (
        "Review blocked because the artifact does not match the loaded request.",
        "产物与已加载请求不匹配，审查已阻止。",
    ),
    "review_pending": (
        "Review action requested. Waiting for the owning page to return an updated artifact.",
        "已请求审查操作；正在等待上层页面返回更新后的产物。",
    ),
}


_ARTIFACT_NAMES: Final[dict[LLMArtifactType, tuple[str, str]]] = {
    LLMArtifactType.TEXT: ("Text", "文本"),
    LLMArtifactType.CLASS_SCORES: ("Class scores", "类别分数"),
    LLMArtifactType.LABELS: ("Semantic labels", "语义标签"),
    LLMArtifactType.MASK_2D: ("2-D mask", "二维掩膜"),
    LLMArtifactType.MASK_3D: ("3-D mask", "三维掩膜"),
    LLMArtifactType.RECONSTRUCTED_IMAGE: ("Reconstructed image", "重建图像"),
    LLMArtifactType.RECONSTRUCTED_VOLUME: ("Reconstructed volume", "重建体数据"),
}

_TASK_NAMES: Final[dict[LLMTaskKind, tuple[str, str]]] = {
    LLMTaskKind.RECONSTRUCT: ("Physics reconstruction", "物理重建"),
    LLMTaskKind.RESTORE: ("Restoration", "复原"),
    LLMTaskKind.ENHANCE: ("Enhancement", "增强"),
    LLMTaskKind.GENERATE: ("Generation", "生成"),
    LLMTaskKind.SEGMENT: ("Segmentation", "分割"),
    LLMTaskKind.CLASSIFY: ("Classification", "分类"),
    LLMTaskKind.LABELS: ("Semantic labeling", "语义标注"),
}

_INPUT_NAMES: Final[dict[str, tuple[str, str]]] = {
    "rendered_slice": ("rendered slice", "渲染切片"),
    "image_2d": ("2-D image", "二维图像"),
    "volume_3d": ("3-D volume", "三维体数据"),
    "volume_4d": ("4-D volume", "四维体数据"),
    "kspace": ("k-space", "k-space"),
    "sinogram": ("sinogram", "正弦图"),
}


def _copy(key: str, language: Language) -> str:
    return _COPY[key][1 if language == "zh_CN" else 0]


def _localized(pair: tuple[str, str], language: Language) -> str:
    return pair[1 if language == "zh_CN" else 0]


def _repolish(widget: QWidget) -> None:
    style = widget.style()
    if style is None:
        return
    style.unpolish(widget)
    style.polish(widget)
    widget.update()


class AssistantArtifactsPanel(QWidget):
    """Responsive, presentation-only workbench for structured LLM contracts.

    ``reviewActionRequested`` carries only the response's opaque artifact ID and
    an :class:`ArtifactReviewDecision` value.  The owner remains responsible for
    appending an immutable review and calling :meth:`set_artifact` again.
    """

    taskTemplateChanged = pyqtSignal(str, str)
    reviewActionRequested = pyqtSignal(str, str)

    _NARROW_WIDTH = 680

    def __init__(
        self,
        parent: QWidget | None = None,
        *,
        language: Language = "en",
    ) -> None:
        super().__init__(parent)
        self._language: Language = "en"
        self._request: LLMTaskRequest | None = None
        self._artifact: LLMArtifactResponse | None = None
        self._binding_valid = False
        self._review_action_pending = False
        self.setObjectName("assistantArtifactsPanel")
        self._build_ui()
        self.set_language(language)

    @property
    def request(self) -> LLMTaskRequest | None:
        return self._request

    @property
    def artifact(self) -> LLMArtifactResponse | None:
        return self._artifact

    @property
    def binding_valid(self) -> bool:
        return self._binding_valid

    @property
    def selected_template(self) -> AssistantTaskTemplate:
        index = max(0, self.template_combo.currentIndex())
        return ARTIFACT_TASK_TEMPLATES[index]

    def minimumSizeHint(self) -> QSize:  # noqa: N802 - Qt API
        return QSize(0, 0)

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(10)

        self.title_label = QLabel()
        self.title_label.setObjectName("pageTitle")
        self.boundary_label = QLabel()
        self.boundary_label.setObjectName("warningBanner")
        self.boundary_label.setWordWrap(True)
        self.boundary_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        root.addWidget(self.title_label)
        root.addWidget(self.boundary_label)

        self.template_group = QGroupBox()
        template_layout = QVBoxLayout(self.template_group)
        template_layout.setSpacing(7)
        self.template_field_label = QLabel()
        self.template_combo = QComboBox()
        self.template_combo.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
        self.template_field_label.setBuddy(self.template_combo)
        self.template_description = QLabel()
        self.template_description.setObjectName("mutedText")
        self.template_description.setWordWrap(True)
        template_layout.addWidget(self.template_field_label)
        template_layout.addWidget(self.template_combo)
        template_layout.addWidget(self.template_description)
        root.addWidget(self.template_group)

        self.summary_splitter = QSplitter(Qt.Horizontal)
        self.summary_splitter.setObjectName("assistantArtifactSummarySplitter")
        self.summary_splitter.setChildrenCollapsible(False)
        self.summary_splitter.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Expanding)
        self.summary_splitter.addWidget(self._build_request_group())
        self.summary_splitter.addWidget(self._build_artifact_group())
        self.summary_splitter.setStretchFactor(0, 1)
        self.summary_splitter.setStretchFactor(1, 1)
        self.summary_splitter.setSizes([360, 360])
        root.addWidget(self.summary_splitter, 1)

        self.action_surface = QFrame()
        self.action_surface.setObjectName("toolbarSurface")
        self.action_layout = QGridLayout(self.action_surface)
        self.action_layout.setContentsMargins(8, 6, 8, 6)
        self.action_layout.setHorizontalSpacing(8)
        self.action_layout.setVerticalSpacing(6)
        self.confirm_button = QPushButton()
        self.confirm_button.setObjectName("primary")
        self.reject_button = QPushButton()
        self.confirm_button.clicked.connect(
            lambda: self._request_review(ArtifactReviewDecision.CONFIRMED)
        )
        self.reject_button.clicked.connect(
            lambda: self._request_review(ArtifactReviewDecision.REJECTED)
        )
        self._layout_actions(narrow=False)
        root.addWidget(self.action_surface)

        self.status_label = QLabel()
        self.status_label.setObjectName("actionStatus")
        self.status_label.setWordWrap(True)
        root.addWidget(self.status_label)

        self.template_combo.currentIndexChanged.connect(self._template_changed)
        QWidget.setTabOrder(self.template_combo, self.confirm_button)
        QWidget.setTabOrder(self.confirm_button, self.reject_button)

    def _build_request_group(self) -> QGroupBox:
        self.request_group = QGroupBox()
        layout = QFormLayout(self.request_group)
        layout.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
        layout.setRowWrapPolicy(QFormLayout.WrapLongRows)
        self._request_field_labels: dict[str, QLabel] = {}
        self._request_value_labels: dict[str, QLabel] = {}
        for key in ("task", "inputs", "outputs", "binding"):
            field, value = self._summary_row()
            self._request_field_labels[key] = field
            self._request_value_labels[key] = value
            layout.addRow(field, value)
        return self.request_group

    def _build_artifact_group(self) -> QGroupBox:
        self.artifact_group = QGroupBox()
        layout = QFormLayout(self.artifact_group)
        layout.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
        layout.setRowWrapPolicy(QFormLayout.WrapLongRows)
        self._artifact_field_labels: dict[str, QLabel] = {}
        self._artifact_value_labels: dict[str, QLabel] = {}
        for key in (
            "artifact_type",
            "payload",
            "integrity",
            "request_match",
            "review",
        ):
            field, value = self._summary_row()
            self._artifact_field_labels[key] = field
            self._artifact_value_labels[key] = value
            layout.addRow(field, value)
        return self.artifact_group

    @staticmethod
    def _summary_row() -> tuple[QLabel, QLabel]:
        field = QLabel()
        value = QLabel()
        value.setObjectName("mutedText")
        value.setWordWrap(True)
        value.setTextInteractionFlags(Qt.TextSelectableByMouse)
        return field, value

    def set_language(self, language: Language) -> None:
        if language not in {"en", "zh_CN"}:
            raise ValueError(f"Unsupported UI language: {language}")
        selected_type = self.selected_template.artifact_type
        self._language = language
        self.title_label.setText(_copy("title", language))
        self.boundary_label.setText(_copy("boundary", language))
        self.boundary_label.setAccessibleName(_copy("boundary", language))
        self.template_group.setTitle(_copy("template_group", language))
        self.template_field_label.setText(_copy("template_label", language))
        self.template_combo.setAccessibleName(_copy("template_accessible", language))
        self.template_combo.setAccessibleDescription(_copy("template_description", language))
        self.template_combo.setToolTip(_copy("template_description", language))
        self.request_group.setTitle(_copy("request_group", language))
        self.request_group.setAccessibleName(_copy("request_group", language))
        self.artifact_group.setTitle(_copy("artifact_group", language))
        self.artifact_group.setAccessibleName(_copy("artifact_group", language))
        for key, label in self._request_field_labels.items():
            label.setText(_copy(key, language))
        for key, label in self._artifact_field_labels.items():
            label.setText(_copy(key, language))
        self.confirm_button.setText(_copy("confirm", language))
        self.confirm_button.setAccessibleName(_copy("confirm", language))
        self.confirm_button.setAccessibleDescription(_copy("confirm_description", language))
        self.confirm_button.setToolTip(_copy("confirm_description", language))
        self.reject_button.setText(_copy("reject", language))
        self.reject_button.setAccessibleName(_copy("reject", language))
        self.reject_button.setAccessibleDescription(_copy("reject_description", language))
        self.reject_button.setToolTip(_copy("reject_description", language))
        self._populate_templates(selected_type)
        self._refresh()

    def set_selected_artifact_type(self, artifact_type: LLMArtifactType | str) -> None:
        normalized = LLMArtifactType(artifact_type)
        for index, template in enumerate(ARTIFACT_TASK_TEMPLATES):
            if template.artifact_type is normalized:
                self.template_combo.setCurrentIndex(index)
                return
        raise ValueError(f"Unsupported artifact type: {normalized.value}")  # pragma: no cover

    def set_request(self, request: LLMTaskRequest | None) -> None:
        if request is not None and not isinstance(request, LLMTaskRequest):
            raise TypeError("request must be an LLMTaskRequest or None")
        self._request = request
        self._review_action_pending = False
        if request is not None and self._artifact is None:
            self._select_artifact_type(request.requested_artifact_types[0])
        self._refresh()

    def set_artifact(self, artifact: LLMArtifactResponse | None) -> None:
        if artifact is not None and not isinstance(artifact, LLMArtifactResponse):
            raise TypeError("artifact must be an LLMArtifactResponse or None")
        self._artifact = artifact
        self._review_action_pending = False
        if artifact is not None:
            self._select_artifact_type(artifact.artifact_type)
        self._refresh()

    def set_context(
        self,
        request: LLMTaskRequest | None,
        artifact: LLMArtifactResponse | None,
    ) -> None:
        if request is not None and not isinstance(request, LLMTaskRequest):
            raise TypeError("request must be an LLMTaskRequest or None")
        if artifact is not None and not isinstance(artifact, LLMArtifactResponse):
            raise TypeError("artifact must be an LLMArtifactResponse or None")
        self._request = request
        self._artifact = artifact
        self._review_action_pending = False
        if artifact is not None:
            self._select_artifact_type(artifact.artifact_type)
        elif request is not None:
            self._select_artifact_type(request.requested_artifact_types[0])
        self._refresh()

    def clear(self) -> None:
        self.set_context(None, None)

    def _populate_templates(self, selected_type: LLMArtifactType) -> None:
        self.template_combo.blockSignals(True)
        try:
            self.template_combo.clear()
            selected_index = 0
            for index, template in enumerate(ARTIFACT_TASK_TEMPLATES):
                self.template_combo.addItem(
                    template.localized_title(self._language),
                    template.artifact_type.value,
                )
                self.template_combo.setItemData(
                    index,
                    template.localized_description(self._language),
                    Qt.ToolTipRole,
                )
                if template.artifact_type is selected_type:
                    selected_index = index
            self.template_combo.setCurrentIndex(selected_index)
        finally:
            self.template_combo.blockSignals(False)
        self._refresh_template_description()

    def _select_artifact_type(self, artifact_type: LLMArtifactType) -> None:
        for index, template in enumerate(ARTIFACT_TASK_TEMPLATES):
            if template.artifact_type is artifact_type:
                self.template_combo.blockSignals(True)
                try:
                    self.template_combo.setCurrentIndex(index)
                finally:
                    self.template_combo.blockSignals(False)
                self._refresh_template_description()
                return

    def _template_changed(self, _index: int) -> None:
        self._refresh_template_description()
        template = self.selected_template
        self.taskTemplateChanged.emit(template.template_id, template.artifact_type.value)

    def _refresh_template_description(self) -> None:
        template = self.selected_template
        task_names = ", ".join(
            _localized(_TASK_NAMES[item], self._language) for item in template.compatible_tasks
        )
        description = f"{template.localized_description(self._language)}  ·  {task_names}"
        self.template_description.setText(description)
        self.template_description.setAccessibleName(description)

    def _refresh(self) -> None:
        self._binding_valid = self._validate_binding()
        self._refresh_request_summary()
        self._refresh_artifact_summary()
        self._refresh_actions_and_status()

    def _validate_binding(self) -> bool:
        if self._request is None or self._artifact is None:
            return False
        try:
            self._artifact.validate_against(self._request)
        except ViewerError:
            return False
        return True

    def _refresh_request_summary(self) -> None:
        request = self._request
        values = self._request_value_labels
        if request is None:
            for value in values.values():
                value.setText(_copy("not_available", self._language))
            values["task"].setText(_copy("no_request", self._language))
        else:
            values["task"].setText(_localized(_TASK_NAMES[request.task], self._language))
            input_names = tuple(
                dict.fromkeys(
                    _localized(_INPUT_NAMES[item.kind.value], self._language)
                    for item in request.inputs
                )
            )
            values["inputs"].setText(
                _copy("deidentified", self._language).format(
                    count=len(request.inputs), kinds=", ".join(input_names)
                )
            )
            output_names = [
                _localized(_ARTIFACT_NAMES[item], self._language)
                for item in request.requested_artifact_types
            ]
            values["outputs"].setText(
                _copy("requested", self._language).format(
                    count=len(output_names), types=", ".join(output_names)
                )
            )
            evidence = (
                _copy("evidence_present", self._language)
                if request.reconstruction_evidence is not None
                else ""
            )
            values["binding"].setText(_copy("hash_bound", self._language).format(evidence=evidence))
        self._update_summary_accessibility(self._request_field_labels, values)

    def _refresh_artifact_summary(self) -> None:
        artifact = self._artifact
        values = self._artifact_value_labels
        if artifact is None:
            for value in values.values():
                value.setText(_copy("not_available", self._language))
            values["artifact_type"].setText(_copy("no_artifact", self._language))
        else:
            values["artifact_type"].setText(
                _localized(_ARTIFACT_NAMES[artifact.artifact_type], self._language)
            )
            values["payload"].setText(self._payload_summary(artifact))
            values["integrity"].setText(_copy("authenticated", self._language))
            if self._request is None:
                match_key = "match_missing"
            elif self._binding_valid:
                match_key = "match_valid"
            else:
                match_key = "match_invalid"
            values["request_match"].setText(_copy(match_key, self._language))
            values["review"].setText(_copy(artifact.validation_status.value, self._language))
        self._update_summary_accessibility(self._artifact_field_labels, values)

    def _payload_summary(self, artifact: LLMArtifactResponse) -> str:
        payload: Any = artifact.payload
        details: list[str] = []
        array = getattr(payload, "array", None)
        if array is not None:
            details.append("×".join(str(int(item)) for item in array.shape))
        elif artifact.artifact_type is LLMArtifactType.CLASS_SCORES:
            count = len(payload.scores)
            details.append(f"{count} " + ("项分数" if self._language == "zh_CN" else "score(s)"))
        elif artifact.artifact_type is LLMArtifactType.LABELS:
            count = len(payload.labels)
            details.append(f"{count} " + ("个标签" if self._language == "zh_CN" else "label(s)"))
        else:
            details.append("UTF-8")
        details.extend((artifact.mime_type, self._format_bytes(len(artifact.encoded_bytes))))
        details.append(_copy("warning_count", self._language).format(count=len(artifact.warnings)))
        return " · ".join(details)

    def _refresh_actions_and_status(self) -> None:
        artifact = self._artifact
        can_review = (
            artifact is not None
            and self._binding_valid
            and artifact.validation_status is ArtifactValidationStatus.UNVERIFIED
            and not self._review_action_pending
        )
        self.confirm_button.setEnabled(can_review)
        self.reject_button.setEnabled(can_review)
        if self._review_action_pending:
            key, state = "review_pending", "busy"
        elif artifact is None:
            key, state = "ready_empty", "blocked"
        elif self._request is None:
            key, state = "blocked_request", "blocked"
        elif not self._binding_valid:
            key, state = "blocked_mismatch", "blocked"
        elif artifact.validation_status is ArtifactValidationStatus.UNVERIFIED:
            key, state = "ready_review", "ready"
        else:
            key, state = "reviewed", "ready"
        text = _copy(key, self._language)
        self.status_label.setText(text)
        self.status_label.setAccessibleName(text)
        self.status_label.setProperty("state", state)
        _repolish(self.status_label)

    def _request_review(self, decision: ArtifactReviewDecision) -> None:
        artifact = self._artifact
        if (
            artifact is None
            or not self._binding_valid
            or artifact.validation_status is not ArtifactValidationStatus.UNVERIFIED
            or self._review_action_pending
        ):
            return
        self._review_action_pending = True
        self._refresh_actions_and_status()
        self.reviewActionRequested.emit(artifact.artifact_id, decision.value)

    @staticmethod
    def _format_bytes(byte_count: int) -> str:
        if byte_count < 1024:
            return f"{byte_count} B"
        if byte_count < 1024 * 1024:
            return f"{byte_count / 1024:.1f} KiB"
        return f"{byte_count / (1024 * 1024):.1f} MiB"

    @staticmethod
    def _update_summary_accessibility(fields: dict[str, QLabel], values: dict[str, QLabel]) -> None:
        for key, value in values.items():
            value.setAccessibleName(f"{fields[key].text()}: {value.text()}")

    def _layout_actions(self, *, narrow: bool) -> None:
        for button in (self.confirm_button, self.reject_button):
            self.action_layout.removeWidget(button)
        if narrow:
            self.action_layout.addWidget(self.confirm_button, 0, 0)
            self.action_layout.addWidget(self.reject_button, 1, 0)
        else:
            self.action_layout.addWidget(self.confirm_button, 0, 0)
            self.action_layout.addWidget(self.reject_button, 0, 1)
        self.action_layout.setColumnStretch(0, 1)
        self.action_layout.setColumnStretch(1, 0 if narrow else 1)

    def resizeEvent(self, event: QResizeEvent) -> None:  # noqa: N802 - Qt API
        narrow = event.size().width() < self._NARROW_WIDTH
        orientation = Qt.Vertical if narrow else Qt.Horizontal
        if self.summary_splitter.orientation() != orientation:
            self.summary_splitter.setOrientation(orientation)
            self.summary_splitter.setSizes([250, 300] if narrow else [360, 360])
        self._layout_actions(narrow=narrow)
        super().resizeEvent(event)


AssistantArtifactWorkbench = AssistantArtifactsPanel

__all__ = [
    "ARTIFACT_TASK_TEMPLATES",
    "AssistantArtifactWorkbench",
    "AssistantArtifactsPanel",
    "AssistantTaskTemplate",
]
