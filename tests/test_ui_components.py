from __future__ import annotations

import pytest

pytest.importorskip("PyQt5")
from PyQt5.QtCore import Qt, QUrl, QVariant
from PyQt5.QtGui import QTextCursor, QTextDocument
from PyQt5.QtWidgets import QLineEdit, QVBoxLayout, QWidget

from workbench.ui.widgets import SafeMarkdownBrowser, SubmitPlainTextEdit


def test_safe_markdown_browser_renders_structured_bilingual_content(qtbot) -> None:
    browser = SafeMarkdownBrowser()
    qtbot.addWidget(browser)

    browser.set_markdown_text(
        "# 标题 Title\n\n"
        "- 中文项目\n"
        "- English item\n\n"
        "| 参数 | Value |\n"
        "| --- | --- |\n"
        "| 窗宽 | 400 |\n\n"
        "```python\n"
        "print('你好, world')\n"
        "```"
    )

    plain_text = browser.toPlainText()
    rendered_markdown = browser.document().toMarkdown(QTextDocument.MarkdownDialectGitHub)
    assert "标题 Title" in plain_text
    assert "中文项目" in plain_text
    assert "窗宽" in plain_text
    assert "print('你好, world')" in plain_text
    assert rendered_markdown.startswith("# 标题 Title")
    assert "- 中文项目" in rendered_markdown
    assert "|参数|Value|" in rendered_markdown
    assert "```python" in rendered_markdown


def test_safe_markdown_browser_treats_raw_html_as_text(qtbot) -> None:
    browser = SafeMarkdownBrowser()
    qtbot.addWidget(browser)

    browser.set_markdown_text("Before <b>unsafe</b> <script>alert('x')</script> after")

    assert "<b>unsafe</b>" in browser.toPlainText()
    assert "<script>alert('x')</script>" in browser.toPlainText()
    rendered_html = browser.document().toHtml()
    assert "&lt;b&gt;unsafe&lt;/b&gt;" in rendered_html
    assert "&lt;script&gt;alert('x')&lt;/script&gt;" in rendered_html


@pytest.mark.parametrize(
    "url",
    (
        "file:///tmp/private-image.png",
        "https://example.invalid/tracking.png",
        "data:image/png;base64,iVBORw0KGgo=",
    ),
)
def test_safe_markdown_browser_blocks_image_resources(qtbot, url: str) -> None:
    browser = SafeMarkdownBrowser()
    qtbot.addWidget(browser)
    browser.set_markdown_text(f"![private image]({url})")

    resource = browser.document().resource(QTextDocument.ImageResource, QUrl(url))
    assert resource is None or isinstance(resource, QVariant) and not resource.isValid()
    assert not browser.openLinks()
    assert not browser.openExternalLinks()


@pytest.mark.parametrize(
    ("modifier", "key"),
    (
        (Qt.ControlModifier, Qt.Key_Return),
        (Qt.ControlModifier, Qt.Key_Enter),
        (Qt.MetaModifier, Qt.Key_Return),
        (Qt.MetaModifier, Qt.Key_Enter),
    ),
)
def test_submit_plain_text_edit_emits_once_for_submit_shortcut(
    qtbot,
    modifier: Qt.KeyboardModifier,
    key: int,
) -> None:
    editor = SubmitPlainTextEdit()
    qtbot.addWidget(editor)
    editor.setPlainText("draft")
    editor.moveCursor(QTextCursor.End)
    received: list[None] = []
    editor.submitRequested.connect(lambda: received.append(None))

    qtbot.keyClick(editor, key, modifier)

    assert received == [None]
    assert editor.toPlainText() == "draft"


def test_submit_plain_text_edit_keeps_plain_enter_as_newline(qtbot) -> None:
    editor = SubmitPlainTextEdit()
    qtbot.addWidget(editor)
    editor.setPlainText("第一行")
    editor.moveCursor(QTextCursor.End)
    received: list[None] = []
    editor.submitRequested.connect(lambda: received.append(None))

    qtbot.keyClick(editor, Qt.Key_Return)
    qtbot.keyClicks(editor, "second")

    assert received == []
    assert editor.toPlainText() == "第一行\nsecond"


def test_submit_plain_text_edit_uses_tab_for_focus_navigation(qtbot) -> None:
    window = QWidget()
    layout = QVBoxLayout(window)
    editor = SubmitPlainTextEdit()
    next_control = QLineEdit()
    layout.addWidget(editor)
    layout.addWidget(next_control)
    QWidget.setTabOrder(editor, next_control)
    qtbot.addWidget(window)
    window.show()
    editor.setFocus()
    qtbot.waitUntil(editor.hasFocus)

    qtbot.keyClick(editor, Qt.Key_Tab)

    assert next_control.hasFocus()
    assert "\t" not in editor.toPlainText()
