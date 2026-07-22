"""Small reusable teaching widgets."""

from __future__ import annotations

import numpy as np
from PyQt5.QtCore import Qt, QUrl, QVariant, pyqtSignal
from PyQt5.QtGui import QColor, QKeyEvent, QPainter, QPen, QTextDocument
from PyQt5.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QLabel,
    QPlainTextEdit,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)

from .i18n import Language, translate


class _SafeMarkdownDocument(QTextDocument):
    """A Markdown document that never resolves model-supplied image resources."""

    def loadResource(self, resource_type: int, name: QUrl):  # noqa: N802 - Qt API
        if resource_type == QTextDocument.ImageResource:
            return QVariant()
        return super().loadResource(resource_type, name)


class SafeMarkdownBrowser(QTextBrowser):
    """Read-only native Markdown rendering without HTML or external resource loading."""

    _MARKDOWN_FEATURES = QTextDocument.MarkdownFeatures(
        QTextDocument.MarkdownDialectGitHub | QTextDocument.MarkdownNoHTML
    )

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setDocument(_SafeMarkdownDocument(self))
        self.setOpenLinks(False)
        self.setOpenExternalLinks(False)
        self.setReadOnly(True)
        self.setUndoRedoEnabled(False)
        self.setFocusPolicy(Qt.StrongFocus)
        self.setTextInteractionFlags(
            Qt.TextSelectableByMouse
            | Qt.TextSelectableByKeyboard
            | Qt.LinksAccessibleByMouse
            | Qt.LinksAccessibleByKeyboard
        )

    def set_markdown_text(self, markdown: str) -> None:
        """Render provider text as GitHub-style Markdown with raw HTML disabled."""

        if not isinstance(markdown, str):
            raise TypeError("markdown must be text")
        document = self.document()
        document.setDefaultFont(self.font())
        document.setMarkdown(markdown, self._MARKDOWN_FEATURES)


class SubmitPlainTextEdit(QPlainTextEdit):
    """Multi-line editor that submits on Ctrl/Command+Enter."""

    submitRequested = pyqtSignal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setTabChangesFocus(True)

    def keyPressEvent(self, event: QKeyEvent) -> None:  # noqa: N802 - Qt API
        submit_modifier = event.modifiers() & (Qt.ControlModifier | Qt.MetaModifier)
        if submit_modifier and event.key() in (Qt.Key_Return, Qt.Key_Enter):
            event.accept()
            self.submitRequested.emit()
            return
        super().keyPressEvent(event)


class HistogramWidget(QWidget):
    def __init__(self, array: np.ndarray, parent=None) -> None:
        super().__init__(parent)
        values = np.asarray(array)
        if values.ndim == 3:
            values = 0.2126 * values[:, :, 0] + 0.7152 * values[:, :, 1] + 0.0722 * values[:, :, 2]
        finite = values[np.isfinite(values)].astype(np.float64)
        self.minimum = float(finite.min()) if finite.size else 0.0
        self.maximum = float(finite.max()) if finite.size else 1.0
        if self.maximum <= self.minimum:
            self.maximum = self.minimum + 1.0
        self.counts, self.edges = np.histogram(finite, bins=256, range=(self.minimum, self.maximum))
        self.setMinimumSize(640, 320)
        self.setAccessibleName("Decoded-value histogram")
        self.setAccessibleDescription(
            f"Histogram from {self.minimum:.4g} to {self.maximum:.4g} across 256 bins"
        )

    def paintEvent(self, event) -> None:  # noqa: N802 - Qt API
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("#07152f"))
        margin = 34
        plot = self.rect().adjusted(margin, margin, -margin, -margin)
        painter.setPen(QPen(QColor("#64748b"), 1))
        painter.drawRect(plot)
        maximum = max(int(self.counts.max()), 1)
        painter.setPen(QPen(QColor("#3b82f6"), 2))
        points = []
        for index, value in enumerate(self.counts):
            x = plot.left() + index * plot.width() / max(len(self.counts) - 1, 1)
            y = plot.bottom() - int(value) * plot.height() / maximum
            points.append((int(x), int(y)))
        for first, second in zip(points, points[1:], strict=False):
            painter.drawLine(first[0], first[1], second[0], second[1])
        painter.setPen(QColor("#e2e8f0"))
        painter.drawText(plot.left(), plot.bottom() + 22, f"{self.minimum:.4g}")
        painter.drawText(plot.right() - 80, plot.bottom() + 22, f"{self.maximum:.4g}")


class HistogramDialog(QDialog):
    def __init__(
        self,
        array: np.ndarray,
        title: str,
        parent=None,
        *,
        language: Language = "en",
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle(translate(title, language))
        layout = QVBoxLayout(self)
        layout.addWidget(
            QLabel(
                translate(
                    "Decoded-value histogram (display mapping does not alter these values).",
                    language,
                )
            )
        )
        layout.addWidget(HistogramWidget(array, self))
        buttons = QDialogButtonBox(QDialogButtonBox.Close)
        close_button = buttons.button(QDialogButtonBox.Close)
        if close_button is not None:
            close_button.setText(translate("Close", language))
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
