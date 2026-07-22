from __future__ import annotations

from pathlib import Path
from unittest.mock import Mock

import pytest

pytest.importorskip("PyQt5")
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QDialog

from workbench.io import SeriesSelectionRequiredError, SeriesSummary
from workbench.ui.i18n import translate
from workbench.ui.main_window import ViewerPage, _DicomSeriesSelectionDialog


def _summary(*, supported: bool, selector: str) -> SeriesSummary:
    return SeriesSummary(
        selector=selector,
        study_identifier="sha256:study-not-for-display",
        series_identifier="sha256:series-not-for-display",
        modality="CT",
        series_description="CHEST AXIAL",
        series_number=3,
        instance_count=12,
        slice_count=12,
        frame_count=12,
        rows=512,
        columns=512,
        pixel_spacing_mm=(0.7, 0.7),
        slice_thickness_mm=1.0,
        estimated_slice_spacing_mm=1.0,
        geometry_consistent=supported,
        supported_by_stable_loader=supported,
        warnings=() if supported else ("Unsupported geometry",),
    )


def test_series_dialog_is_phi_minimized_bilingual_and_gates_unsupported_rows(qtbot) -> None:
    supported = _summary(supported=True, selector="sha256:supported")
    unsupported = _summary(supported=False, selector="sha256:unsupported")
    dialog = _DicomSeriesSelectionDialog(
        (supported, unsupported),
        translator=lambda text: translate(text, "zh_CN"),
    )
    qtbot.addWidget(dialog)

    assert dialog.windowTitle() == "选择 DICOM 序列"
    assert dialog.table.horizontalHeaderItem(0).text() == "状态"
    visible_text = "\n".join(
        dialog.table.item(row, column).text()
        for row in range(dialog.table.rowCount())
        for column in range(dialog.table.columnCount())
    )
    assert "study-not-for-display" not in visible_text
    assert "series-not-for-display" not in visible_text

    unavailable = dialog.table.item(1, 0)
    assert not bool(unavailable.flags() & Qt.ItemIsEnabled)
    assert not bool(unavailable.flags() & Qt.ItemIsSelectable)
    assert not dialog.open_button.isEnabled()

    dialog.table.selectRow(0)
    assert dialog.open_button.isEnabled()
    assert dialog.selected_selector == supported.selector
    dialog.close()


def test_series_selection_retries_the_same_source_with_explicit_selector(qtbot) -> None:
    page = ViewerPage()
    qtbot.addWidget(page)
    page._pending_source_path = Path("selected-folder")
    selector = "sha256:chosen"
    fake_dialog = Mock()
    fake_dialog.exec_.return_value = QDialog.Accepted
    fake_dialog.selected_selector = selector
    page._build_series_selection_dialog = Mock(return_value=fake_dialog)  # type: ignore[method-assign]
    page._begin_load = Mock()  # type: ignore[method-assign]

    page._select_dicom_series((_summary(supported=True, selector=selector),))

    page._begin_load.assert_called_once_with(
        "selected-folder",
        dicom_series_selector=selector,
    )
    fake_dialog.deleteLater.assert_called_once_with()
    page.close()


def test_series_dialog_is_deferred_out_of_task_watcher_callback(qtbot) -> None:
    page = ViewerPage()
    qtbot.addWidget(page)
    candidate = _summary(supported=True, selector="sha256:chosen")
    page._select_dicom_series = Mock()  # type: ignore[method-assign]

    page._load_failed(SeriesSelectionRequiredError((candidate,)))

    page._select_dicom_series.assert_not_called()
    qtbot.waitUntil(lambda: page._select_dicom_series.call_count == 1)
    page._select_dicom_series.assert_called_once_with((candidate,))
    page.close()
