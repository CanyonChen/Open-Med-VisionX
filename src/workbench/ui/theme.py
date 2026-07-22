"""Application typography and the shared Qt widget theme."""

from __future__ import annotations

import sys

from PyQt5.QtGui import QFont, QFontDatabase
from PyQt5.QtWidgets import QApplication

_MINIMUM_POINT_SIZE = 9.5

_PLATFORM_FONT_FALLBACKS = {
    "darwin": ("SF Pro Text", "PingFang SC"),
    "win32": (
        "Segoe UI Variable Text",
        "Segoe UI",
        "Microsoft YaHei UI",
    ),
    "linux": ("Noto Sans", "Noto Sans CJK SC", "DejaVu Sans"),
}


def _platform_font_fallbacks() -> tuple[str, ...]:
    if sys.platform == "darwin":
        return _PLATFORM_FONT_FALLBACKS["darwin"]
    if sys.platform.startswith("win"):
        return _PLATFORM_FONT_FALLBACKS["win32"]
    return _PLATFORM_FONT_FALLBACKS["linux"]


def application_font() -> QFont:
    """Return a system-sized UI font with installed Latin and Chinese fallbacks."""

    font = QFontDatabase.systemFont(QFontDatabase.GeneralFont)
    installed = {family.casefold(): family for family in QFontDatabase().families()}

    families: list[str] = []
    seen: set[str] = set()

    def append_family(family: str) -> None:
        key = family.casefold()
        if family and key not in seen:
            families.append(family)
            seen.add(key)

    for candidate in _platform_font_fallbacks():
        installed_family = installed.get(candidate.casefold())
        if installed_family is not None:
            append_family(installed_family)
    # Keep the platform's general font as the final safety net while preferring
    # the installed UI/CJK pair above for predictable bilingual metrics.
    append_family(font.family())

    # Qt 5.13+ supports an ordered fallback list. The fallback keeps this module
    # usable with older PyQt5 builds without replacing the system-selected face.
    if hasattr(font, "setFamilies"):
        font.setFamilies(families)
    elif families:
        font.setFamily(families[0])

    point_sizes = [font.pointSizeF(), _MINIMUM_POINT_SIZE]
    application = QApplication.instance()
    if application is not None:
        point_sizes.append(application.font().pointSizeF())
    font.setPointSizeF(max(size for size in point_sizes if size > 0))
    return font


APP_STYLE = """
QMainWindow, QWidget#appShell {
    background: #f4f7fb;
}

QWidget {
    color: #101828;
}

QLabel {
    background: transparent;
}

QFrame#appHeader,
QFrame#toolbarSurface,
QFrame#conversationSurface,
QFrame#composerSurface {
    background: #ffffff;
    border: 1px solid #e4e9f2;
    border-radius: 12px;
}

QFrame#composerSurface {
    background: #fbfcfe;
    border-color: #d8dee9;
}

QFrame#composerSurface[busy="true"] {
    background: #f7f9ff;
    border-color: #8daeff;
}

QLabel#appTitle {
    color: #07163d;
    font-size: 15pt;
    font-weight: 700;
}

QLabel#pageTitle {
    color: #07163d;
    font-size: 13pt;
    font-weight: 700;
}

QLabel#appSubtitle,
QLabel#mutedText {
    color: #667085;
}

QLabel#emptyStateTitle {
    color: #344054;
    font-size: 12pt;
    font-weight: 700;
}

QLabel#keyboardHint,
QLabel#answerMetadata {
    color: #667085;
    font-size: 9pt;
}

QLabel#questionContext {
    background: #edf3ff;
    color: #174ea6;
    border: 1px solid #d6e3ff;
    border-radius: 9px;
    padding: 8px 10px;
}

QLabel#answerDisclaimer {
    background: #fff8e8;
    color: #854d0e;
    border: 1px solid #f5dfad;
    border-radius: 9px;
    padding: 8px 10px;
}

QLabel#privacyChip {
    background: #edf3ff;
    color: #174ea6;
    border: 1px solid #d6e3ff;
    border-radius: 11px;
    padding: 5px 10px;
    font-weight: 600;
}

QLabel#infoBanner,
QLabel#resourceBanner,
QLabel#warningBanner,
QLabel#dangerBanner {
    border-radius: 10px;
    padding: 10px 12px;
}

QLabel#infoBanner,
QLabel#resourceBanner {
    background: #eef4ff;
    color: #24426f;
    border: 1px solid #d9e5ff;
}

QLabel#warningBanner {
    background: #fff8e8;
    color: #854d0e;
    border: 1px solid #f5dfad;
}

QLabel#dangerBanner {
    background: #fff1f1;
    color: #9b1c1c;
    border: 1px solid #ffd5d5;
}

QLabel#statusStrip {
    background: #ffffff;
    color: #475467;
    border: 1px solid #e4e9f2;
    border-radius: 8px;
    padding: 7px 10px;
}

QLabel#viewContext,
QLabel#actionStatus,
QLabel#nextStep {
    background: #f8faff;
    color: #344054;
    border: 1px solid #d9e5ff;
    border-radius: 8px;
    padding: 7px 10px;
}

QLabel#viewContext[active="true"],
QLabel#actionStatus[state="ready"] {
    background: #edf3ff;
    color: #174ea6;
    border-color: #a9bfff;
}

QLabel#actionStatus[state="busy"] {
    background: #eef4ff;
    color: #24426f;
    border-color: #8daeff;
}

QLabel#actionStatus[state="blocked"],
QLabel#nextStep[state="blocked"] {
    background: #fff8e8;
    color: #854d0e;
    border-color: #f5dfad;
}

QLabel#cloudStatus {
    background: #f2f4f7;
    color: #475467;
    border: 1px solid #d0d5dd;
    border-radius: 9px;
    padding: 7px 10px;
    font-weight: 600;
}

QLabel#cloudStatus[state="off"],
QLabel#cloudStatus[state="safe"] {
    background: #ecfdf3;
    color: #166534;
    border-color: #abefc6;
}

QLabel#cloudStatus[state="on"],
QLabel#cloudStatus[state="active"],
QLabel#cloudStatus[state="warning"] {
    background: #fff1f1;
    color: #b42318;
    border-color: #fecdca;
}

QLabel#cloudStatus[state="pending"] {
    background: #fff8e8;
    color: #854d0e;
    border-color: #f5dfad;
}

QGroupBox {
    background: #ffffff;
    border: 1px solid #e4e9f2;
    border-radius: 11px;
    margin-top: 14px;
    padding: 16px 12px 12px 12px;
    font-weight: 600;
}

QGroupBox::title {
    subcontrol-origin: margin;
    left: 12px;
    padding: 0 5px;
    color: #344054;
}

QPushButton {
    background: #ffffff;
    color: #344054;
    border: 1px solid #d0d5dd;
    border-radius: 8px;
    padding: 7px 12px;
    min-height: 20px;
    font-weight: 600;
}

QPushButton:hover {
    background: #f5f8ff;
    color: #174ea6;
    border-color: #8daeff;
}

QPushButton:pressed {
    background: #e4ecff;
    color: #103f91;
    border-color: #155eef;
}

QPushButton:focus {
    background: #f7f9ff;
    border: 2px solid #2f6bff;
    padding: 6px 11px;
}

QPushButton:disabled {
    background: #f2f4f7;
    color: #98a2b3;
    border-color: #e4e7ec;
}

QPushButton#primary {
    background: #155eef;
    color: #ffffff;
    border-color: #155eef;
}

QPushButton#primary:hover {
    background: #004eeb;
    color: #ffffff;
    border-color: #004eeb;
}

QPushButton#primary:pressed {
    background: #00359e;
    color: #ffffff;
    border-color: #00359e;
}

QPushButton#primary:focus {
    background: #155eef;
    color: #ffffff;
    border: 2px solid #07163d;
    padding: 6px 11px;
}

QPushButton#primary:disabled {
    background: #e4e7ec;
    color: #667085;
    border-color: #cfd4dc;
}

QPushButton#languageSwitchButton {
    min-width: 72px;
}

QPushButton#linkButton {
    background: transparent;
    color: #155eef;
    border-color: transparent;
    padding-left: 7px;
    padding-right: 7px;
}

QPushButton#linkButton:hover {
    background: #eef4ff;
    border-color: #d9e5ff;
}

QLineEdit,
QPlainTextEdit,
QTextEdit,
QTextBrowser,
QComboBox,
QSpinBox,
QDoubleSpinBox {
    background: #ffffff;
    color: #101828;
    border: 1px solid #d0d5dd;
    border-radius: 8px;
    padding: 6px 8px;
    selection-background-color: #cfe0ff;
    selection-color: #07163d;
}

QLineEdit,
QComboBox,
QSpinBox,
QDoubleSpinBox {
    min-height: 20px;
}

QLineEdit:focus,
QPlainTextEdit:focus,
QTextEdit:focus,
QTextBrowser:focus,
QComboBox:focus,
QSpinBox:focus,
QDoubleSpinBox:focus {
    border: 2px solid #2f6bff;
    padding: 5px 7px;
}

QLineEdit:disabled,
QPlainTextEdit:disabled,
QTextEdit:disabled,
QTextBrowser:disabled,
QComboBox:disabled,
QSpinBox:disabled,
QDoubleSpinBox:disabled {
    background: #f2f4f7;
    color: #98a2b3;
    border-color: #e4e7ec;
}

QLineEdit[validation="error"] {
    background: #fffafa;
    border-color: #d92d20;
}

QTextBrowser#markdownAnswer,
QTextEdit#markdownAnswer {
    background: #fbfcfe;
    border: none;
    border-radius: 9px;
    padding: 12px;
}

QTextBrowser#markdownAnswer:focus,
QTextEdit#markdownAnswer:focus {
    background: #ffffff;
    border: 2px solid #8daeff;
    padding: 10px;
}

QFrame#composerSurface QPlainTextEdit,
QPlainTextEdit#assistantPrompt {
    background: transparent;
    border: none;
    padding: 8px;
}

QFrame#composerSurface QPlainTextEdit:focus,
QPlainTextEdit#assistantPrompt:focus {
    background: #ffffff;
    border: 2px solid #8daeff;
    padding: 6px;
}

QComboBox::drop-down {
    border: none;
    width: 26px;
}

QComboBox QAbstractItemView {
    background: #ffffff;
    color: #101828;
    border: 1px solid #d0d5dd;
    border-radius: 8px;
    padding: 4px;
    selection-background-color: #e8efff;
    selection-color: #0b4dd8;
}

QAbstractItemView::item {
    min-height: 24px;
    padding: 4px 8px;
    border-radius: 6px;
}

QAbstractItemView:focus {
    border: 2px solid #2f6bff;
}

QCheckBox {
    spacing: 8px;
    border: 1px solid transparent;
    border-radius: 7px;
    padding: 4px 6px;
}

QCheckBox:hover {
    background: #f5f8ff;
}

QCheckBox:focus {
    background: #eef4ff;
    color: #0b4dd8;
    border-color: #2f6bff;
}

QCheckBox:disabled {
    color: #98a2b3;
}

QTabWidget#workspaceTabs::pane {
    background: transparent;
    border: none;
    top: -1px;
}

QTabWidget#modelWorkspaceTabs::pane {
    background: transparent;
    border: none;
    top: -1px;
}

QTabWidget#modelWorkspaceTabs > QTabBar::tab {
    background: #f2f4f7;
    color: #667085;
    border: 1px solid #d0d5dd;
    border-radius: 8px;
    padding: 7px 12px;
    margin-right: 4px;
    font-weight: 600;
}

QTabWidget#modelWorkspaceTabs > QTabBar::tab:selected {
    background: #ffffff;
    color: #0b4dd8;
    border-color: #8daeff;
}

QListWidget#modelCatalog {
    background: #ffffff;
    border: 1px solid #d0d5dd;
    border-radius: 10px;
    padding: 5px;
    outline: none;
}

QListWidget#modelCatalog::item {
    border-radius: 8px;
    padding: 9px 10px;
    margin: 2px;
}

QListWidget#modelCatalog::item:hover {
    background: #f5f8ff;
    color: #174ea6;
}

QListWidget#modelCatalog::item:selected {
    background: #e8efff;
    color: #0b4dd8;
    border: 1px solid #8daeff;
}

QPlainTextEdit#bundleContract,
QTextBrowser#modelCard {
    background: #fbfcfe;
}

QTabWidget#workspaceTabs QTabBar::tab {
    background: #eaecf0;
    color: #667085;
    border: 1px solid #d0d5dd;
    border-radius: 8px;
    padding: 8px 12px;
    margin-right: 4px;
    font-weight: 600;
}

QTabWidget#workspaceTabs QTabBar::tab:hover {
    background: #edf3ff;
    color: #174ea6;
    border-color: #a9bfff;
}

QTabWidget#workspaceTabs QTabBar::tab:selected {
    background: #ffffff;
    color: #0b4dd8;
    border-color: #8daeff;
}

QTabWidget#workspaceTabs QTabBar::tab:selected:focus {
    border: 2px solid #2f6bff;
    padding: 7px 11px;
}

QTabWidget#workspaceTabs QTabBar::tab:disabled {
    background: #f2f4f7;
    color: #98a2b3;
    border-color: #e4e7ec;
}

QTabWidget#modelOutputTabs::pane {
    background: #ffffff;
    border: 1px solid #e4e9f2;
    border-radius: 9px;
}

QTabWidget#modelOutputTabs QTabBar::tab {
    background: #f2f4f7;
    color: #667085;
    border: 1px solid #e4e7ec;
    border-radius: 7px;
    padding: 7px 11px;
    min-width: 112px;
    margin-right: 3px;
}

QTabWidget#modelOutputTabs QTabBar::tab:hover {
    background: #edf3ff;
    color: #174ea6;
}

QTabWidget#modelOutputTabs QTabBar::tab:selected {
    background: #ffffff;
    color: #155eef;
    border-color: #8daeff;
}

QTabWidget#modelOutputTabs QTabBar::tab:selected:focus {
    border: 2px solid #2f6bff;
    padding: 6px 10px;
}

QTabBar QToolButton {
    background: #ffffff;
    border: 1px solid #d0d5dd;
    border-radius: 7px;
    padding: 3px;
}

QScrollArea {
    background: transparent;
    border: none;
}

QScrollArea > QWidget > QWidget {
    background: transparent;
}

QScrollArea:focus {
    border: 1px solid #8daeff;
    border-radius: 8px;
}

QScrollBar:vertical {
    background: #f2f4f7;
    border: none;
    border-radius: 6px;
    width: 12px;
    margin: 2px;
}

QScrollBar::handle:vertical {
    background: #b8c1cf;
    border-radius: 4px;
    min-height: 28px;
}

QScrollBar::handle:vertical:hover,
QScrollBar::handle:vertical:pressed {
    background: #7d8da5;
}

QScrollBar::add-line:vertical,
QScrollBar::sub-line:vertical {
    height: 0;
    border: none;
}

QScrollBar::add-page:vertical,
QScrollBar::sub-page:vertical {
    background: transparent;
}

QScrollBar:horizontal {
    background: #f2f4f7;
    border: none;
    border-radius: 6px;
    height: 12px;
    margin: 2px;
}

QScrollBar::handle:horizontal {
    background: #b8c1cf;
    border-radius: 4px;
    min-width: 28px;
}

QScrollBar::handle:horizontal:hover,
QScrollBar::handle:horizontal:pressed {
    background: #7d8da5;
}

QScrollBar::add-line:horizontal,
QScrollBar::sub-line:horizontal {
    width: 0;
    border: none;
}

QScrollBar::add-page:horizontal,
QScrollBar::sub-page:horizontal {
    background: transparent;
}

QSplitter::handle {
    background: transparent;
    border-radius: 2px;
}

QSplitter::handle:horizontal {
    width: 3px;
    margin: 0 4px;
}

QSplitter::handle:vertical {
    height: 3px;
    margin: 4px 0;
}

QSplitter::handle:hover,
QSplitter::handle:pressed,
QSplitter::handle:focus {
    background: #8daeff;
}

QProgressBar {
    background: #e9eef6;
    color: #344054;
    border: none;
    border-radius: 5px;
    min-height: 10px;
    text-align: center;
}

QProgressBar::chunk {
    background: #155eef;
    border-radius: 5px;
}

QSlider {
    border: 1px solid transparent;
    border-radius: 7px;
    padding: 2px;
}

QSlider:focus {
    background: #eef4ff;
    border-color: #2f6bff;
}

QSlider::groove:horizontal {
    background: #d9e0ea;
    height: 4px;
    border-radius: 2px;
}

QSlider::sub-page:horizontal {
    background: #6c95f8;
    border-radius: 2px;
}

QSlider::handle:horizontal {
    background: #ffffff;
    border: 2px solid #155eef;
    border-radius: 8px;
    width: 14px;
    margin: -6px 0;
}

QSlider::handle:horizontal:hover,
QSlider::handle:horizontal:pressed {
    background: #edf3ff;
    border-color: #00359e;
}

QGraphicsView {
    background: #07152f;
    border: 1px solid #273552;
    border-radius: 9px;
}

QGraphicsView:focus {
    border: 2px solid #6c95f8;
}

QGraphicsView[active="true"] {
    border: 2px solid #5b8def;
}

QGraphicsView[active="true"]:focus {
    border: 2px solid #2f6bff;
}

QStatusBar {
    background: #ffffff;
    color: #667085;
    border-top: 1px solid #e4e9f2;
}

QStatusBar::item {
    border: none;
}

QToolBar {
    background: #ffffff;
    border: none;
    spacing: 6px;
    padding: 6px;
}

QToolBar::separator {
    background: #e4e9f2;
    width: 1px;
    margin: 5px;
}

QMenu {
    background: #ffffff;
    color: #101828;
    border: 1px solid #d0d5dd;
    border-radius: 9px;
    padding: 5px;
}

QMenu::item {
    border-radius: 6px;
    padding: 6px 24px 6px 10px;
}

QMenu::item:selected {
    background: #e8efff;
    color: #0b4dd8;
}

QMenu::item:disabled {
    color: #98a2b3;
}

QMenu::separator {
    background: #e4e9f2;
    height: 1px;
    margin: 5px 8px;
}

QToolTip {
    background: #07163d;
    color: #ffffff;
    border: 1px solid #243760;
    border-radius: 7px;
    padding: 6px 8px;
}
"""
