from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pytest

pytest.importorskip("PyQt5")
from PyQt5.QtCore import QUrl
from PyQt5.QtGui import QDesktopServices, QTextDocument
from PyQt5.QtWidgets import QMessageBox

from workbench.domain import IntensitySemantics, RasterImage2D, SourceType
from workbench.errors import OperationCancelled
from workbench.llm import LLMResponse
from workbench.services import LoadedStudy
from workbench.ui.main_window import AssistantPage, ViewerPage


def _study(value: int) -> LoadedStudy:
    image = RasterImage2D(
        np.full((8, 8), value, dtype=np.uint16),
        SourceType.RASTER,
        IntensitySemantics.GRAYSCALE,
        bit_depth=16,
        runtime_metadata={"format": "PNG", "lossy_compression": False},
    )
    return LoadedStudy(
        image=image,
        display_image=image,
        imported_at=datetime.now(timezone.utc),
        source_kind="PNG",
    )


def test_viewer_replaces_the_current_study_only_after_a_new_load_succeeds(qtbot) -> None:
    page = ViewerPage()
    qtbot.addWidget(page)
    original = _study(1)
    page._loaded(original)
    task = Mock()

    with (
        patch.object(page.service, "begin_load", return_value=task),
        patch.object(page.watcher, "watch") as watch,
    ):
        page._begin_load("replacement.png")

    assert page.study is original
    assert page.image is original.display_image
    np.testing.assert_array_equal(page.axial_view.array, original.display_image.array)
    assert page._pending_source_path == Path("replacement.png")
    watch.assert_called_once()

    page._active_load_task = None
    page._load_failed(OperationCancelled("cancelled"))
    assert page.study is original
    assert page.image is original.display_image
    page.close()


def test_assistant_renders_only_provider_text_as_markdown_and_keeps_provenance_plain(
    qtbot,
) -> None:
    page = AssistantPage(lambda: b"unused")
    qtbot.addWidget(page)
    response = LLMResponse(
        text="# 结果 Result\n\n- **重点**\n- `code`",
        provider="Research gateway",
        model="teaching-model",
        timestamp=datetime(2026, 7, 21, 8, 30, tzinfo=timezone.utc),
        disclaimer="Trusted educational disclaimer.",
    )

    page._answered(response)

    rendered = page.answer.document().toMarkdown(QTextDocument.MarkdownDialectGitHub)
    assert rendered.startswith("# 结果 Result")
    assert "重点" in page.answer.toPlainText()
    assert "Research gateway" not in page.answer.toPlainText()
    assert "Research gateway" in page.answer_metadata.text()
    assert "teaching-model" in page.answer_metadata.text()
    assert page.answer_disclaimer.text() == "Trusted educational disclaimer."
    assert page.response_stack.currentIndex() == 1

    page.set_language("zh_CN")
    assert "结果 Result" in page.answer.toPlainText()
    assert page.answer.accessibleName() == "采用 Markdown 的最新助手回复"
    assert "Research gateway" in page.answer_metadata.text()
    page.close()


def test_assistant_busy_state_freezes_provider_configuration_but_keeps_drafting(qtbot) -> None:
    page = AssistantPage(lambda: b"unused")
    qtbot.addWidget(page)
    page._active_task = Mock()

    page._set_request_state(busy=True, status="Generating a new response…")

    assert not page.config_group.isEnabled()
    assert page.prompt.isEnabled()
    assert page.cancel_request_button.isEnabled()
    assert page.composer.property("busy") is True

    page._active_task = None
    page._set_request_state(busy=False, status="Request cancelled.")
    assert page.config_group.isEnabled()
    assert not page.cancel_request_button.isEnabled()
    page.close()


def test_assistant_attachment_tracks_image_and_vision_availability(qtbot) -> None:
    page = AssistantPage(lambda: b"unused")
    qtbot.addWidget(page)
    page.set_source_available(None)
    page.vision.setChecked(True)
    assert not page.send_preview.isEnabled()

    page.set_source_available(object())
    assert page.send_preview.isEnabled()
    page.vision.setChecked(False)
    assert not page.send_preview.isEnabled()
    assert not page.send_preview.isChecked()
    page.close()


def test_assistant_never_opens_untrusted_or_unconfirmed_links(qtbot) -> None:
    page = AssistantPage(lambda: b"unused")
    qtbot.addWidget(page)
    with (
        patch.object(QMessageBox, "warning") as warning,
        patch.object(QDesktopServices, "openUrl") as open_url,
    ):
        page._open_response_link(QUrl("file:///private/report.txt"))
    warning.assert_called_once()
    open_url.assert_not_called()

    with (
        patch.object(QMessageBox, "question", return_value=QMessageBox.No) as question,
        patch.object(QDesktopServices, "openUrl") as open_url,
    ):
        page._open_response_link(QUrl("https://example.org/reference"))
    question.assert_called_once()
    open_url.assert_not_called()
    page.close()
