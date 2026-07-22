"""Local-only BraTS 2021 case validation and manifest installation dialog.

The dialog intentionally provides no dataset downloader.  It opens only the
official acquisition page after confirmation, validates user-managed files in
the background, and writes a path-free manifest after an explicit terms check.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from PyQt5.QtCore import QDir, Qt, QTimer, QUrl, pyqtSignal
from PyQt5.QtGui import QCloseEvent, QDesktopServices, QKeySequence
from PyQt5.QtWidgets import (
    QAbstractItemView,
    QBoxLayout,
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFrame,
    QGroupBox,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ..cases import (
    BRATS_2021_MODALITIES,
    BRATS_2021_OFFICIAL_ACQUISITION_URL,
    BraTS2021Issue,
    BraTS2021ValidationReport,
    validate_brats2021_case,
    write_brats2021_manifest,
)
from ..errors import OperationCancelled
from ..runtime import BackgroundTask, TaskContext, TaskRunner
from .i18n import Language

Validator = Callable[[str | Path], BraTS2021ValidationReport]
ManifestWriter = Callable[..., Path]
DirectoryDialog = Callable[[QWidget, str], str]
SaveDialog = Callable[[QWidget, str, str, str], tuple[str, str]]
LinkConfirmer = Callable[[QWidget, QUrl], bool]
LinkOpener = Callable[[QUrl], bool]
OverwriteConfirmer = Callable[[QWidget, Path], bool]

_HORIZONTAL = QBoxLayout.LeftToRight
_VERTICAL = QBoxLayout.TopToBottom
_ITEM_FLAGS = Qt.ItemIsEnabled | Qt.ItemIsSelectable  # type: ignore[attr-defined]

_COPY: dict[str, tuple[str, str]] = {
    "window_title": ("Install a local BraTS 2021 case", "安装本地 BraTS 2021 案例"),
    "title": ("BraTS 2021 local case", "BraTS 2021 本地案例"),
    "intro": (
        "Connect a case you obtained through the official process. OpenMedVisionX "
        "validates files in place; it never downloads, copies, or uploads the volumes.",
        "连接您通过官方流程获取的案例。OpenMedVisionX 仅在原位置校验文件，"
        "绝不会下载、复制或上传体数据。",
    ),
    "research_note": (
        "Learning and research only · not for diagnosis · source files remain user-managed",
        "仅用于学习与研究 · 不用于诊断 · 源文件始终由用户自行管理",
    ),
    "step_access": ("1 · Official access", "1 · 官方获取"),
    "access_help": (
        "Review eligibility, citation, ethics, and data terms on the official BraTS 2021 "
        "page. Opening the page does not download data or accept any terms.",
        "请在 BraTS 2021 官方页面查看资格、引用、伦理及数据条款。"
        "打开页面不会下载数据，也不会代您接受任何条款。",
    ),
    "open_official": ("&Open official BraTS 2021 page…", "打开 BraTS 2021 官方页面(&O)…"),
    "open_official_accessible": (
        "Open the official BraTS 2021 acquisition page after confirmation",
        "确认后打开 BraTS 2021 官方获取页面",
    ),
    "step_select": ("2 · Select and validate", "2 · 选择并校验"),
    "select_help": (
        "Choose one local case directory containing exactly T1, T1ce, T2, FLAIR, and SEG "
        "NIfTI volumes. Validation starts automatically in the background.",
        "选择一个本地案例目录，其中应恰好包含 T1、T1ce、T2、FLAIR 和 SEG "
        "NIfTI 体数据。选择后会自动在后台校验。",
    ),
    "directory_label": ("Local case directory", "本地案例目录"),
    "directory_placeholder": ("No directory selected", "尚未选择目录"),
    "choose_directory": ("Choose &folder…", "选择文件夹(&F)…"),
    "choose_directory_accessible": (
        "Choose a user-managed BraTS 2021 case directory and validate it",
        "选择用户自行管理的 BraTS 2021 案例目录并进行校验",
    ),
    "revalidate": ("&Validate again", "重新校验(&V)"),
    "revalidate_accessible": (
        "Validate the selected local directory again",
        "重新校验所选本地目录",
    ),
    "cancel_validation": ("&Cancel validation", "取消校验(&C)"),
    "cancel_validation_accessible": (
        "Safely cancel the current background validation",
        "安全取消当前后台校验",
    ),
    "choose_title": ("Choose a BraTS 2021 case directory", "选择 BraTS 2021 案例目录"),
    "step_review": ("3 · Review validation", "3 · 查看校验结果"),
    "input_header": ("Input", "输入"),
    "status_header": ("Status", "状态"),
    "details_header": ("Geometry / details", "几何 / 详情"),
    "not_checked": ("Not checked", "尚未校验"),
    "checking": ("Checking…", "正在校验…"),
    "ready": ("Ready", "就绪"),
    "ready_warning": ("Ready with warning", "就绪（有提示）"),
    "needs_attention": ("Needs attention", "需要处理"),
    "not_found": ("Not found", "未找到"),
    "no_details": ("Select a directory to inspect this input.", "选择目录后可检查此输入。"),
    "checking_details": ("Reading metadata and content hash…", "正在读取元数据并计算内容哈希…"),
    "missing_details": ("No validated file record is available.", "没有可用的已校验文件记录。"),
    "geometry": (
        "{shape} · spacing {spacing} mm · orientation {orientation}",
        "{shape} · 间距 {spacing} mm · 方向 {orientation}",
    ),
    "seg_counts": ("Label voxels", "标签体素数"),
    "issues_label": ("Validation findings", "校验发现"),
    "no_findings": ("No validation findings.", "没有校验问题。"),
    "error_label": ("Error", "错误"),
    "warning_label": ("Warning", "提示"),
    "global_label": ("Case", "案例"),
    "step_save": ("4 · Save anonymous manifest", "4 · 保存匿名清单"),
    "save_help": (
        "The JSON manifest contains checksums and geometry summaries only—no volume "
        "pixels, source filenames, absolute paths, or person identifiers.",
        "JSON 清单只包含校验和与几何摘要，不包含体素、源文件名、绝对路径或个人标识。",
    ),
    "terms": (
        "Official terms reviewed",
        "已阅读官方条款",
    ),
    "terms_accessible": (
        "Confirm that you personally reviewed the official BraTS data terms",
        "确认您已亲自阅读 BraTS 官方数据条款",
    ),
    "save_manifest": ("&Save manifest…", "保存清单(&S)…"),
    "save_manifest_accessible": (
        "Save an anonymous local manifest for this validated case",
        "为已通过校验的案例保存匿名本地清单",
    ),
    "save_title": ("Save anonymous BraTS manifest", "保存匿名 BraTS 清单"),
    "json_filter": ("JSON files (*.json)", "JSON 文件 (*.json)"),
    "close": ("Close", "关闭"),
    "next_step": ("Next: choose a local case directory.", "下一步：选择本地案例目录。"),
    "next_busy": (
        "Next: wait for validation or cancel it safely.",
        "下一步：等待校验完成，或安全取消。",
    ),
    "next_invalid": (
        "Next: resolve the listed errors, then validate the directory again.",
        "下一步：处理列出的错误，然后重新校验目录。",
    ),
    "next_terms": (
        "Next: review the official terms, then select the confirmation checkbox.",
        "下一步：阅读官方条款，然后勾选确认框。",
    ),
    "next_save": (
        "Next: save the anonymous manifest to a location you control.",
        "下一步：将匿名清单保存到您管理的位置。",
    ),
    "next_saved": (
        "Manifest saved. Keep the validated source volumes in place for later workflows.",
        "清单已保存。请保留已校验源体数据的位置，以便后续工作流使用。",
    ),
    "status_ready": (
        "Ready · no data has been accessed or transferred.",
        "已就绪 · 尚未访问或传输任何数据。",
    ),
    "status_validating": (
        "Validating locally in the background… no data is uploaded.",
        "正在本地后台校验…不会上传数据。",
    ),
    "status_valid": (
        "Validation passed · all five inputs and their geometry are compatible.",
        "校验通过 · 五项输入及其几何信息兼容。",
    ),
    "status_invalid": (
        "Validation blocked · {count} error(s) must be resolved.",
        "校验未通过 · 需要处理 {count} 个错误。",
    ),
    "status_cancelling": ("Cancelling safely…", "正在安全取消…"),
    "status_cancelled": (
        "Validation cancelled · no manifest was written.",
        "校验已取消 · 未写入任何清单。",
    ),
    "status_failed": (
        "Validation failed safely: {reason}",
        "校验已安全失败：{reason}",
    ),
    "status_saved": ("Anonymous manifest saved: {path}", "匿名清单已保存：{path}"),
    "status_save_failed": ("Manifest was not saved: {reason}", "未保存清单：{reason}"),
    "status_exists": (
        "Existing file kept unchanged. Choose another destination or explicitly approve "
        "replacement.",
        "现有文件保持不变。请选择其他位置，或明确批准替换。",
    ),
    "open_confirm_title": ("Open official website?", "打开官方网站？"),
    "open_confirm_text": (
        "Open this external page in your browser?\n\n{url}\n\nThis only opens the "
        "official page. It does not download data or accept terms.",
        "是否在浏览器中打开此外部页面？\n\n{url}\n\n此操作只会打开官方网站，不会下载数据或接受条款。",
    ),
    "open_failed": ("The official page could not be opened.", "无法打开官方网站。"),
    "overwrite_title": ("Replace existing file?", "替换现有文件？"),
    "overwrite_text": (
        "A file already exists at this destination. Replace only this file?\n\n{path}",
        "此位置已存在文件。是否只替换这个文件？\n\n{path}",
    ),
}

_ISSUES_ZH: dict[str, str] = {
    "directory_unavailable": "所选 BraTS 案例目录不可用。",
    "candidate_outside_directory": "一个 NIfTI 候选项解析到了所选目录之外。",
    "unrecognized_nifti_name": "一个 NIfTI 文件名未标明所需 BraTS 模态。",
    "ambiguous_modality_name": "一个 NIfTI 文件名包含多个模态标记。",
    "missing_modality": "缺少必需的 {modality} 模态。",
    "duplicate_modality": "{modality} 模态存在多个候选项。",
    "nibabel_unavailable": "BraTS 校验需要安装 NIfTI 依赖组中的 nibabel。",
    "candidate_unreadable": "必需的 NIfTI 候选项无法读取。",
    "candidate_empty": "必需的 NIfTI 候选项为空。",
    "candidate_too_large": "必需的 NIfTI 候选项超过 2 GiB 安全限制。",
    "qform_sform_conflict": "{modality} 的 qform 与 sform 描述了不同的几何。",
    "qform_missing": "{modality} 没有编码 qform；仍会检查 affine 与 sform。",
    "sform_missing": "{modality} 没有编码 sform；仍会检查 affine 与 qform。",
    "voxel_limit_exceeded": "必需的 NIfTI 候选项超过体素安全限制。",
    "nifti_decode_failed": "必需的候选项不是受支持的有效三维 NIfTI 体数据。",
    "file_changed_during_validation": "必需的 NIfTI 候选项在校验期间发生了变化。",
    "seg_decode_failed": "无法解码 SEG 体素数据。",
    "seg_nonfinite": "SEG 包含 NaN 或无穷值。",
    "seg_not_integer": "SEG 必须使用整数 NIfTI 数据类型和未缩放的整数体素。",
    "seg_invalid_labels": "SEG 包含允许集合 {{0, 1, 2, 4}} 之外的标签。",
    "shape_mismatch": "{modality} 的形状与 T1 参考几何不一致。",
    "spacing_mismatch": "{modality} 的体素间距与 T1 参考几何不一致。",
    "affine_mismatch": "{modality} 的 affine 与 T1 参考几何不一致。",
    "orientation_mismatch": "{modality} 的方向与 T1 参考几何不一致。",
    "world_coverage_mismatch": "{modality} 的世界坐标覆盖范围与 T1 参考几何不一致。",
    "qform_code_mismatch": "{modality} 的 qform code 与 T1 参考几何不一致。",
    "qform_mismatch": "{modality} 的 qform 与 T1 参考几何不一致。",
    "sform_code_mismatch": "{modality} 的 sform code 与 T1 参考几何不一致。",
    "sform_mismatch": "{modality} 的 sform 与 T1 参考几何不一致。",
}


def _repolish(widget: QWidget) -> None:
    style = widget.style()
    if style is not None:
        style.unpolish(widget)
        style.polish(widget)
    widget.update()


class BraTS2021InstallerDialog(QDialog):
    """Guide users through official access, local validation, and manifest export."""

    validationFinished = pyqtSignal(object)
    manifestSaved = pyqtSignal(str)
    statusChanged = pyqtSignal(str)

    def __init__(
        self,
        parent: QWidget | None = None,
        *,
        language: Language = "en",
        validator: Validator = validate_brats2021_case,
        writer: ManifestWriter = write_brats2021_manifest,
        directory_dialog: DirectoryDialog | None = None,
        save_dialog: SaveDialog | None = None,
        link_confirmer: LinkConfirmer | None = None,
        link_opener: LinkOpener | None = None,
        overwrite_confirmer: OverwriteConfirmer | None = None,
    ) -> None:
        super().__init__(parent)
        self._language: Language = "en"
        self._validator = validator
        self._writer = writer
        self._directory_dialog = directory_dialog or QFileDialog.getExistingDirectory
        self._save_dialog = save_dialog or QFileDialog.getSaveFileName
        self._link_confirmer = link_confirmer or self._confirm_external_link
        self._link_opener = link_opener or QDesktopServices.openUrl
        self._overwrite_confirmer = overwrite_confirmer or self._confirm_overwrite
        self._runner = TaskRunner(max_workers=1, thread_name_prefix="openmedvisionx-brats")
        self._active_task: BackgroundTask[BraTS2021ValidationReport] | None = None
        self._selected_directory: Path | None = None
        self._report: BraTS2021ValidationReport | None = None
        self._saved_destination: Path | None = None
        self._closed = False
        self._status_key = "status_ready"
        self._status_values: dict[str, object] = {}

        self._poll_timer = QTimer(self)
        self._poll_timer.setInterval(50)
        self._poll_timer.timeout.connect(self._poll_validation)

        self.setModal(True)
        self.setMinimumSize(560, 520)
        self.resize(780, 760)

        self._build_ui()
        self.set_language(language)
        self._render_empty_results()
        self._update_actions()

    @property
    def active_task(self) -> BackgroundTask[BraTS2021ValidationReport] | None:
        """Return the current validation task, if any."""

        return self._active_task

    @property
    def validation_report(self) -> BraTS2021ValidationReport | None:
        """Return the last completed report."""

        return self._report

    @property
    def selected_directory(self) -> Path | None:
        """Return the user-selected source directory."""

        return self._selected_directory

    def _build_ui(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)

        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setFrameShape(QFrame.NoFrame)
        self.scroll_area.setFocusPolicy(Qt.StrongFocus)  # type: ignore[attr-defined]
        outer.addWidget(self.scroll_area, 1)

        content = QWidget()
        self.content_layout = QVBoxLayout(content)
        self.content_layout.setContentsMargins(22, 20, 22, 20)
        self.content_layout.setSpacing(16)
        self.scroll_area.setWidget(content)

        self.title_label = QLabel()
        title_font = self.title_label.font()
        title_font.setPointSize(max(title_font.pointSize() + 5, 16))
        title_font.setWeight(600)
        self.title_label.setFont(title_font)
        self.content_layout.addWidget(self.title_label)

        self.intro_label = QLabel()
        self.intro_label.setWordWrap(True)
        self.intro_label.setTextInteractionFlags(Qt.TextSelectableByMouse)  # type: ignore[attr-defined]
        self.content_layout.addWidget(self.intro_label)

        self.research_note = QLabel()
        self.research_note.setObjectName("privacyNotice")
        self.research_note.setWordWrap(True)
        self.content_layout.addWidget(self.research_note)

        self.access_group = QGroupBox()
        access_layout = QVBoxLayout(self.access_group)
        access_layout.setSpacing(10)
        self.access_help = QLabel()
        self.access_help.setWordWrap(True)
        self.access_help.setTextInteractionFlags(Qt.TextSelectableByMouse)  # type: ignore[attr-defined]
        access_layout.addWidget(self.access_help)
        self.official_row = QBoxLayout(_HORIZONTAL)
        self.official_row.setSpacing(10)
        self.official_url = QLabel(BRATS_2021_OFFICIAL_ACQUISITION_URL)
        self.official_url.setTextInteractionFlags(Qt.TextSelectableByMouse)  # type: ignore[attr-defined]
        self.official_url.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        self.official_url.setWordWrap(True)
        self.open_official_button = QPushButton()
        self.open_official_button.setAutoDefault(False)
        self.open_official_button.clicked.connect(self._open_official_page)
        self.official_row.addWidget(self.official_url, 1)
        self.official_row.addWidget(self.open_official_button)
        access_layout.addLayout(self.official_row)
        self.content_layout.addWidget(self.access_group)

        self.select_group = QGroupBox()
        select_layout = QVBoxLayout(self.select_group)
        select_layout.setSpacing(10)
        self.select_help = QLabel()
        self.select_help.setWordWrap(True)
        self.select_help.setTextInteractionFlags(Qt.TextSelectableByMouse)  # type: ignore[attr-defined]
        select_layout.addWidget(self.select_help)
        self.directory_label = QLabel()
        select_layout.addWidget(self.directory_label)
        self.directory_row = QBoxLayout(_HORIZONTAL)
        self.directory_row.setSpacing(8)
        self.directory_edit = QLineEdit()
        self.directory_edit.setReadOnly(True)
        self.directory_edit.setClearButtonEnabled(False)
        self.directory_row.addWidget(self.directory_edit, 1)
        self.choose_button = QPushButton()
        self.choose_button.setAutoDefault(False)
        self.choose_button.clicked.connect(self._choose_directory)
        self.directory_row.addWidget(self.choose_button)
        select_layout.addLayout(self.directory_row)
        self.validation_actions = QBoxLayout(_HORIZONTAL)
        self.validation_actions.setSpacing(8)
        self.revalidate_button = QPushButton()
        self.revalidate_button.setShortcut(QKeySequence("Ctrl+R"))
        self.revalidate_button.setAutoDefault(False)
        self.revalidate_button.clicked.connect(self._start_validation)
        self.cancel_button = QPushButton()
        self.cancel_button.setAutoDefault(False)
        self.cancel_button.clicked.connect(self.cancel_validation)
        self.validation_actions.addWidget(self.revalidate_button)
        self.validation_actions.addWidget(self.cancel_button)
        self.validation_actions.addStretch(1)
        select_layout.addLayout(self.validation_actions)
        self.progress = QProgressBar()
        self.progress.setRange(0, 0)
        self.progress.setTextVisible(False)
        self.progress.setVisible(False)
        select_layout.addWidget(self.progress)
        self.content_layout.addWidget(self.select_group)

        self.review_group = QGroupBox()
        review_layout = QVBoxLayout(self.review_group)
        review_layout.setSpacing(10)
        self.modality_table = QTableWidget(len(BRATS_2021_MODALITIES), 3)
        self.modality_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.modality_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.modality_table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.modality_table.verticalHeader().setVisible(False)
        self.modality_table.horizontalHeader().setStretchLastSection(True)
        self.modality_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.modality_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.modality_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.Stretch)
        self.modality_table.setWordWrap(True)
        self.modality_table.setMinimumHeight(205)
        for row, modality in enumerate(BRATS_2021_MODALITIES):
            item = QTableWidgetItem(modality if modality != "T1CE" else "T1ce")
            item.setFlags(_ITEM_FLAGS)
            self.modality_table.setItem(row, 0, item)
        review_layout.addWidget(self.modality_table)
        self.counts_label = QLabel()
        self.counts_label.setWordWrap(True)
        self.counts_label.setTextInteractionFlags(Qt.TextSelectableByMouse)  # type: ignore[attr-defined]
        self.counts_label.setVisible(False)
        review_layout.addWidget(self.counts_label)
        self.issues_label = QLabel()
        review_layout.addWidget(self.issues_label)
        self.issues_text = QPlainTextEdit()
        self.issues_text.setReadOnly(True)
        self.issues_text.setMaximumHeight(150)
        self.issues_text.setTabChangesFocus(True)
        review_layout.addWidget(self.issues_text)
        self.content_layout.addWidget(self.review_group)

        self.save_group = QGroupBox()
        save_layout = QVBoxLayout(self.save_group)
        save_layout.setSpacing(10)
        self.save_help = QLabel()
        self.save_help.setWordWrap(True)
        self.save_help.setTextInteractionFlags(Qt.TextSelectableByMouse)  # type: ignore[attr-defined]
        save_layout.addWidget(self.save_help)
        self.terms_checkbox = QCheckBox()
        self.terms_checkbox.setChecked(False)
        self.terms_checkbox.stateChanged.connect(self._update_actions)
        save_layout.addWidget(self.terms_checkbox)
        self.save_button = QPushButton()
        self.save_button.setShortcut(QKeySequence("Ctrl+S"))
        self.save_button.setDefault(True)
        self.save_button.clicked.connect(self._save_manifest)
        save_layout.addWidget(self.save_button, 0, Qt.AlignLeft)  # type: ignore[attr-defined]
        self.content_layout.addWidget(self.save_group)

        self.status_label = QLabel()
        self.status_label.setObjectName("actionStatus")
        self.status_label.setWordWrap(True)
        self.status_label.setTextInteractionFlags(Qt.TextSelectableByMouse)  # type: ignore[attr-defined]
        self.content_layout.addWidget(self.status_label)
        self.next_step_label = QLabel()
        self.next_step_label.setObjectName("nextStep")
        self.next_step_label.setWordWrap(True)
        self.next_step_label.setTextInteractionFlags(Qt.TextSelectableByMouse)  # type: ignore[attr-defined]
        self.content_layout.addWidget(self.next_step_label)

        self.button_box = QDialogButtonBox(QDialogButtonBox.Close)
        self.button_box.rejected.connect(self.reject)
        footer = QWidget()
        footer_layout = QVBoxLayout(footer)
        footer_layout.setContentsMargins(22, 0, 22, 16)
        footer_layout.addWidget(self.button_box)
        outer.addWidget(footer)

        self.setTabOrder(self.open_official_button, self.directory_edit)
        self.setTabOrder(self.directory_edit, self.choose_button)
        self.setTabOrder(self.choose_button, self.revalidate_button)
        self.setTabOrder(self.revalidate_button, self.cancel_button)
        self.setTabOrder(self.cancel_button, self.modality_table)
        self.setTabOrder(self.modality_table, self.issues_text)
        self.setTabOrder(self.issues_text, self.terms_checkbox)
        self.setTabOrder(self.terms_checkbox, self.save_button)

    def _text(self, key: str, **values: object) -> str:
        pair = _COPY[key]
        text = pair[1 if self._language == "zh_CN" else 0]
        return text.format(**values)

    def set_language(self, language: Language) -> None:
        """Retranslate the dialog without losing validation state."""

        if language not in {"en", "zh_CN"}:
            raise ValueError("language must be 'en' or 'zh_CN'")
        self._language = language
        self.setWindowTitle(self._text("window_title"))
        self.title_label.setText(self._text("title"))
        self.intro_label.setText(self._text("intro"))
        self.research_note.setText(self._text("research_note"))
        self.access_group.setTitle(self._text("step_access"))
        self.access_help.setText(self._text("access_help"))
        self.open_official_button.setText(self._text("open_official"))
        self.open_official_button.setAccessibleName(self._text("open_official_accessible"))
        self.open_official_button.setAccessibleDescription(self._text("access_help"))
        self.official_url.setAccessibleName(self._text("open_official_accessible"))
        self.select_group.setTitle(self._text("step_select"))
        self.select_help.setText(self._text("select_help"))
        self.directory_label.setText(self._text("directory_label"))
        self.directory_edit.setPlaceholderText(self._text("directory_placeholder"))
        self.directory_edit.setAccessibleName(self._text("directory_label"))
        self.directory_edit.setAccessibleDescription(self._text("select_help"))
        self.choose_button.setText(self._text("choose_directory"))
        self.choose_button.setAccessibleName(self._text("choose_directory_accessible"))
        self.choose_button.setAccessibleDescription(self._text("select_help"))
        self.revalidate_button.setText(self._text("revalidate"))
        self.revalidate_button.setAccessibleName(self._text("revalidate_accessible"))
        self.revalidate_button.setAccessibleDescription(self._text("select_help"))
        self.cancel_button.setText(self._text("cancel_validation"))
        self.cancel_button.setAccessibleName(self._text("cancel_validation_accessible"))
        self.cancel_button.setAccessibleDescription(self._text("cancel_validation_accessible"))
        self.progress.setAccessibleName(self._text("status_validating"))
        self.review_group.setTitle(self._text("step_review"))
        self.modality_table.setHorizontalHeaderLabels(
            [self._text("input_header"), self._text("status_header"), self._text("details_header")]
        )
        self.modality_table.setAccessibleName(self._text("step_review"))
        self.modality_table.setAccessibleDescription(self._text("select_help"))
        self.issues_label.setText(self._text("issues_label"))
        self.issues_text.setAccessibleName(self._text("issues_label"))
        self.issues_text.setAccessibleDescription(self._text("step_review"))
        self.save_group.setTitle(self._text("step_save"))
        self.save_help.setText(self._text("save_help"))
        self.terms_checkbox.setText(self._text("terms"))
        self.terms_checkbox.setAccessibleName(self._text("terms_accessible"))
        self.terms_checkbox.setAccessibleDescription(self._text("access_help"))
        self.save_button.setText(self._text("save_manifest"))
        self.save_button.setAccessibleName(self._text("save_manifest_accessible"))
        self.save_button.setAccessibleDescription(self._text("save_help"))
        close_button = self.button_box.button(QDialogButtonBox.Close)
        if close_button is not None:
            close_button.setText(self._text("close"))
            close_button.setAccessibleName(self._text("close"))
        self.scroll_area.setAccessibleName(self._text("window_title"))
        self._render_status()
        if self._active_task is not None:
            self._render_checking_results()
        elif self._report is not None:
            self._render_report(self._report)
        else:
            self._render_empty_results()
        self._update_actions()

    def _choose_directory(self) -> None:
        if self._active_task is not None or self._closed:
            return
        selected = self._directory_dialog(self, self._text("choose_title"))
        if not selected:
            return
        self._selected_directory = Path(selected)
        self.directory_edit.setText(QDir.toNativeSeparators(str(self._selected_directory)))
        self.terms_checkbox.setChecked(False)
        self._report = None
        self._saved_destination = None
        self._start_validation()

    @staticmethod
    def _validation_operation(
        context: TaskContext,
        validator: Validator,
        directory: Path,
    ) -> BraTS2021ValidationReport:
        context.report_progress(0.05, message="validating-local-case")
        context.raise_if_cancelled()
        report = validator(directory)
        context.raise_if_cancelled()
        if not isinstance(report, BraTS2021ValidationReport):
            raise TypeError("BraTS validator returned an unsupported report type")
        context.report_progress(1.0, message="validation-complete")
        return report

    def _start_validation(self) -> None:
        if self._closed or self._active_task is not None or self._selected_directory is None:
            return
        self._report = None
        self._saved_destination = None
        self.terms_checkbox.setChecked(False)
        self._active_task = self._runner.submit(
            self._validation_operation,
            self._validator,
            self._selected_directory,
        )
        self._set_status("status_validating", "busy")
        self._render_checking_results()
        self._update_actions()
        self._poll_timer.start()

    def _poll_validation(self) -> None:
        task = self._active_task
        if task is None:
            self._poll_timer.stop()
            return
        if not task.done:
            return
        self._poll_timer.stop()
        self._active_task = None
        try:
            report = task.result()
        except OperationCancelled:
            self._set_status("status_cancelled", "blocked")
            self._render_empty_results()
        except BaseException as error:  # noqa: BLE001 - surface worker failures safely
            self._set_status("status_failed", "blocked", reason=str(error))
            self._render_empty_results()
        else:
            self._report = report
            self._render_report(report)
            if report.is_valid:
                self._set_status("status_valid", "ready")
            else:
                errors = sum(issue.severity == "error" for issue in report.issues)
                self._set_status("status_invalid", "blocked", count=errors)
            self.validationFinished.emit(report)
        self._update_actions()

    def cancel_validation(self) -> None:
        """Request cooperative cancellation without terminating a worker thread."""

        task = self._active_task
        if task is None:
            return
        task.cancel()
        self._set_status("status_cancelling", "busy")
        self.cancel_button.setEnabled(False)

    def _render_empty_results(self) -> None:
        for row in range(len(BRATS_2021_MODALITIES)):
            self._set_table_item(row, 1, self._text("not_checked"))
            self._set_table_item(row, 2, self._text("no_details"))
        self.counts_label.clear()
        self.counts_label.setVisible(False)
        self.issues_text.setPlainText(self._text("no_findings"))
        self.modality_table.resizeRowsToContents()

    def _render_checking_results(self) -> None:
        for row in range(len(BRATS_2021_MODALITIES)):
            self._set_table_item(row, 1, self._text("checking"))
            self._set_table_item(row, 2, self._text("checking_details"))
        self.counts_label.clear()
        self.counts_label.setVisible(False)
        self.issues_text.setPlainText(self._text("checking"))
        self.modality_table.resizeRowsToContents()

    def _render_report(self, report: BraTS2021ValidationReport) -> None:
        for row, modality in enumerate(BRATS_2021_MODALITIES):
            record = report.file_for(modality)
            modality_issues = tuple(issue for issue in report.issues if issue.modality == modality)
            if record is None:
                status = self._text("not_found")
                details = self._text("missing_details")
            else:
                has_error = any(issue.severity == "error" for issue in modality_issues)
                has_warning = any(issue.severity == "warning" for issue in modality_issues)
                status = self._text(
                    "needs_attention" if has_error else "ready_warning" if has_warning else "ready"
                )
                geometry = record.geometry
                details = self._text(
                    "geometry",
                    shape=" × ".join(str(size) for size in geometry.shape),
                    spacing=" × ".join(f"{value:g}" for value in geometry.spacing_mm),
                    orientation="".join(geometry.orientation),
                )
            self._set_table_item(row, 1, status)
            self._set_table_item(row, 2, details)

        if report.segmentation_counts:
            counts = " · ".join(
                f"{label}: {count:,}" for label, count in report.segmentation_counts
            )
            self.counts_label.setText(f"{self._text('seg_counts')}: {counts}")
            self.counts_label.setVisible(True)
        else:
            self.counts_label.clear()
            self.counts_label.setVisible(False)

        findings = [self._format_issue(issue) for issue in report.issues]
        self.issues_text.setPlainText("\n".join(findings) or self._text("no_findings"))
        self.modality_table.resizeRowsToContents()

    def _format_issue(self, issue: BraTS2021Issue) -> str:
        severity = self._text("error_label" if issue.severity == "error" else "warning_label")
        scope = issue.modality or self._text("global_label")
        if self._language == "zh_CN":
            template = _ISSUES_ZH.get(issue.code, issue.message)
            message = template.format(modality=issue.modality or self._text("global_label"))
        else:
            message = issue.message
        return f"{severity} · {scope} · {issue.code}: {message}"

    def _set_table_item(self, row: int, column: int, text: str) -> None:
        item = self.modality_table.item(row, column)
        if item is None:
            item = QTableWidgetItem()
            item.setFlags(_ITEM_FLAGS)
            self.modality_table.setItem(row, column, item)
        item.setText(text)
        item.setToolTip(text)
        item.setData(Qt.AccessibleTextRole, text)  # type: ignore[attr-defined]

    def _update_actions(self) -> None:
        busy = self._active_task is not None
        valid = self._report is not None and self._report.is_valid
        confirmed = self.terms_checkbox.isChecked()
        self.choose_button.setEnabled(not busy and not self._closed)
        self.revalidate_button.setEnabled(
            not busy and not self._closed and self._selected_directory is not None
        )
        self.cancel_button.setEnabled(busy and not self._closed)
        self.terms_checkbox.setEnabled(valid and not busy and not self._closed)
        self.save_button.setEnabled(valid and confirmed and not busy and not self._closed)
        self.progress.setVisible(busy)

        if self._saved_destination is not None:
            next_key = "next_saved"
            next_state = "ready"
        elif busy:
            next_key = "next_busy"
            next_state = "busy"
        elif self._report is not None and not self._report.is_valid:
            next_key = "next_invalid"
            next_state = "blocked"
        elif valid and not confirmed:
            next_key = "next_terms"
            next_state = "blocked"
        elif valid:
            next_key = "next_save"
            next_state = "ready"
        else:
            next_key = "next_step"
            next_state = ""
        self.next_step_label.setText(self._text(next_key))
        self.next_step_label.setAccessibleName(self._text(next_key))
        self.next_step_label.setProperty("state", next_state)
        _repolish(self.next_step_label)

    def _save_manifest(self) -> None:
        report = self._report
        if (
            self._closed
            or self._active_task is not None
            or report is None
            or not report.is_valid
            or not self.terms_checkbox.isChecked()
        ):
            return
        suggested = f"{report.case_alias}.manifest.json"
        selected = self._save_dialog(
            self,
            self._text("save_title"),
            suggested,
            self._text("json_filter"),
        )
        destination_text = selected[0] if isinstance(selected, tuple) else selected
        if not destination_text:
            return
        destination = Path(destination_text)
        if destination.exists() and not self._overwrite_confirmer(self, destination):
            self._set_status("status_exists", "blocked")
            return
        try:
            written = Path(
                self._writer(
                    report,
                    destination,
                    terms_confirmed_by_user=True,
                )
            )
        except (OSError, ValueError, PermissionError, TypeError) as error:
            self._set_status("status_save_failed", "blocked", reason=str(error))
            return
        self._saved_destination = written
        rendered_path = QDir.toNativeSeparators(str(written))
        self._set_status("status_saved", "ready", path=rendered_path)
        self._update_actions()
        self.manifestSaved.emit(str(written))

    def _open_official_page(self) -> None:
        url = QUrl(BRATS_2021_OFFICIAL_ACQUISITION_URL)
        if not self._link_confirmer(self, url):
            return
        if not bool(self._link_opener(url)):
            QMessageBox.warning(self, self._text("open_confirm_title"), self._text("open_failed"))

    def _confirm_external_link(self, parent: QWidget, url: QUrl) -> bool:
        answer = QMessageBox.question(
            parent,
            self._text("open_confirm_title"),
            self._text("open_confirm_text", url=url.toDisplayString()),
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        return answer == QMessageBox.Yes

    def _confirm_overwrite(self, parent: QWidget, path: Path) -> bool:
        answer = QMessageBox.question(
            parent,
            self._text("overwrite_title"),
            self._text("overwrite_text", path=QDir.toNativeSeparators(str(path))),
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        return answer == QMessageBox.Yes

    def _set_status(self, key: str, state: str, **values: object) -> None:
        self._status_key = key
        self._status_values = values
        self.status_label.setProperty("state", state)
        self._render_status()
        _repolish(self.status_label)

    def _render_status(self) -> None:
        text = self._text(self._status_key, **self._status_values)
        self.status_label.setText(text)
        self.status_label.setAccessibleName(text)
        self.statusChanged.emit(text)

    def resizeEvent(self, event: Any) -> None:  # noqa: N802 - Qt API
        super().resizeEvent(event)
        direction = _VERTICAL if self.width() < 900 else _HORIZONTAL
        for layout in (self.official_row, self.directory_row, self.validation_actions):
            if layout.direction() != direction:
                layout.setDirection(direction)
        if self.width() < 650:
            header = self.modality_table.horizontalHeader()
            header.setSectionResizeMode(0, QHeaderView.ResizeToContents)
            header.setSectionResizeMode(1, QHeaderView.ResizeToContents)
            header.setSectionResizeMode(2, QHeaderView.Stretch)
        self.modality_table.resizeRowsToContents()

    def shutdown(self) -> None:
        """Cancel current work and detach the dialog from its worker safely."""

        if self._closed:
            return
        self._closed = True
        self._poll_timer.stop()
        if self._active_task is not None:
            self._active_task.cancel()
            self._active_task = None
        self._runner.shutdown(wait=False, cancel_pending=True)
        self._update_actions()

    def reject(self) -> None:
        self.shutdown()
        super().reject()

    def closeEvent(self, event: QCloseEvent) -> None:  # noqa: N802 - Qt API
        self.shutdown()
        super().closeEvent(event)


__all__ = ["BraTS2021InstallerDialog"]
