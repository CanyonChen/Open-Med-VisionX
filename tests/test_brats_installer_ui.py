from __future__ import annotations

import threading
from pathlib import Path

import pytest

pytest.importorskip("PyQt5")
from PyQt5.QtCore import QUrl
from PyQt5.QtWidgets import QBoxLayout

from workbench.cases import (
    BRATS_2021_MODALITIES,
    BRATS_2021_OFFICIAL_ACQUISITION_URL,
    BraTS2021FileRecord,
    BraTS2021Geometry,
    BraTS2021Issue,
    BraTS2021ValidationReport,
)
from workbench.ui.brats_installer import BraTS2021InstallerDialog


def _geometry(*, shape: tuple[int, int, int] = (240, 240, 155)) -> BraTS2021Geometry:
    identity = (
        (1.0, 0.0, 0.0, 0.0),
        (0.0, 1.0, 0.0, 0.0),
        (0.0, 0.0, 1.0, 0.0),
        (0.0, 0.0, 0.0, 1.0),
    )
    return BraTS2021Geometry(
        shape=shape,
        spacing_mm=(1.0, 1.0, 1.0),
        affine=identity,
        orientation=("R", "A", "S"),
        world_bounds_mm=((0.0, 239.0), (0.0, 239.0), (0.0, 154.0)),
        qform_code=1,
        qform=identity,
        sform_code=1,
        sform=identity,
    )


def _record(modality: str) -> BraTS2021FileRecord:
    return BraTS2021FileRecord(
        modality=modality,
        relative_id=f"modalities/{modality.casefold()}",
        sha256=(str(BRATS_2021_MODALITIES.index(modality) + 1) * 64)[:64],
        size_bytes=1024,
        geometry=_geometry(),
    )


def _valid_report() -> BraTS2021ValidationReport:
    return BraTS2021ValidationReport(
        case_alias="brats-2021-anonymous1234",
        files=tuple(_record(modality) for modality in BRATS_2021_MODALITIES),
        issues=(),
        segmentation_counts=((0, 8_000_000), (1, 12_345), (2, 6_789), (4, 321)),
    )


def _invalid_report() -> BraTS2021ValidationReport:
    return BraTS2021ValidationReport(
        case_alias="brats-2021-unresolved",
        files=tuple(_record(modality) for modality in BRATS_2021_MODALITIES if modality != "FLAIR"),
        issues=(
            BraTS2021Issue(
                "missing_modality",
                "error",
                "Required modality FLAIR is missing.",
                modality="FLAIR",
            ),
            BraTS2021Issue(
                "affine_mismatch",
                "error",
                "T2 affine does not match the T1 reference geometry.",
                modality="T2",
            ),
            BraTS2021Issue(
                "seg_invalid_labels",
                "error",
                "SEG contains labels outside {0, 1, 2, 4}: [3].",
                modality="SEG",
            ),
        ),
        segmentation_counts=((0, 20), (1, 4), (2, 3), (4, 2)),
    )


def _dialog(qtbot, **kwargs) -> BraTS2021InstallerDialog:
    dialog = BraTS2021InstallerDialog(**kwargs)
    qtbot.addWidget(dialog)
    dialog.show()
    return dialog


def test_dialog_has_four_step_local_only_flow_accessibility_and_reflow(qtbot) -> None:
    dialog = _dialog(qtbot)

    assert dialog.access_group.title().startswith("1")
    assert dialog.select_group.title().startswith("2")
    assert dialog.review_group.title().startswith("3")
    assert dialog.save_group.title().startswith("4")
    assert dialog.modality_table.rowCount() == 5
    assert [dialog.modality_table.item(row, 0).text() for row in range(5)] == [
        "T1",
        "T1ce",
        "T2",
        "FLAIR",
        "SEG",
    ]
    assert "never downloads, copies, or uploads" in dialog.intro_label.text()
    assert BRATS_2021_OFFICIAL_ACQUISITION_URL in dialog.official_url.text()
    assert dialog.open_official_button.accessibleName()
    assert dialog.choose_button.accessibleName()
    assert dialog.revalidate_button.accessibleName()
    assert dialog.cancel_button.accessibleName()
    assert dialog.terms_checkbox.accessibleName()
    assert dialog.save_button.accessibleName()
    assert not dialog.terms_checkbox.isEnabled()
    assert not dialog.save_button.isEnabled()

    dialog.resize(620, 740)
    qtbot.waitUntil(lambda: dialog.directory_row.direction() == QBoxLayout.TopToBottom)
    assert dialog.official_row.direction() == QBoxLayout.TopToBottom
    assert dialog.validation_actions.direction() == QBoxLayout.TopToBottom

    dialog.resize(960, 740)
    qtbot.waitUntil(lambda: dialog.directory_row.direction() == QBoxLayout.LeftToRight)


def test_official_page_is_never_opened_before_explicit_confirmation(qtbot) -> None:
    confirmations: list[QUrl] = []
    opened: list[QUrl] = []

    def confirm(_parent, url: QUrl) -> bool:
        confirmations.append(url)
        return False

    dialog = _dialog(
        qtbot,
        link_confirmer=confirm,
        link_opener=lambda url: opened.append(url) is None,
    )
    dialog.open_official_button.click()

    assert [url.toString() for url in confirmations] == [BRATS_2021_OFFICIAL_ACQUISITION_URL]
    assert opened == []

    dialog._link_confirmer = lambda _parent, _url: True
    dialog.open_official_button.click()
    assert [url.toString() for url in opened] == [BRATS_2021_OFFICIAL_ACQUISITION_URL]


def test_selecting_directory_validates_off_gui_thread_and_gates_manifest(qtbot, tmp_path) -> None:
    worker_threads: list[int] = []
    gui_thread = threading.get_ident()
    source = tmp_path / "user-managed-case"
    source.mkdir()

    def validator(path: str | Path) -> BraTS2021ValidationReport:
        worker_threads.append(threading.get_ident())
        assert Path(path) == source
        return _valid_report()

    dialog = _dialog(
        qtbot,
        validator=validator,
        directory_dialog=lambda _parent, _title: str(source),
    )
    reports: list[BraTS2021ValidationReport] = []
    dialog.validationFinished.connect(reports.append)

    dialog.choose_button.click()

    assert dialog.active_task is not None
    assert dialog.progress.isVisible()
    assert not dialog.choose_button.isEnabled()
    qtbot.waitUntil(lambda: dialog.active_task is None, timeout=3_000)

    assert worker_threads and worker_threads[-1] != gui_thread
    assert reports == [_valid_report()]
    assert dialog.validation_report is reports[0]
    assert dialog.selected_directory == source
    assert dialog.terms_checkbox.isEnabled()
    assert not dialog.save_button.isEnabled()
    assert "all five inputs" in dialog.status_label.text()
    assert "240 × 240 × 155" in dialog.modality_table.item(0, 2).text()
    assert "8,000,000" in dialog.counts_label.text()
    assert "No validation findings" in dialog.issues_text.toPlainText()

    dialog.terms_checkbox.setChecked(True)
    assert dialog.save_button.isEnabled()
    assert "save the anonymous manifest" in dialog.next_step_label.text()


def test_invalid_case_shows_missing_geometry_and_label_findings_in_both_languages(
    qtbot, tmp_path
) -> None:
    source = tmp_path / "invalid-case"
    source.mkdir()
    dialog = _dialog(
        qtbot,
        validator=lambda _path: _invalid_report(),
        directory_dialog=lambda _parent, _title: str(source),
    )

    dialog.choose_button.click()
    qtbot.waitUntil(lambda: dialog.active_task is None, timeout=3_000)

    assert not dialog.validation_report.is_valid
    assert "3 error(s)" in dialog.status_label.text()
    assert dialog.modality_table.item(3, 1).text() == "Not found"
    assert dialog.modality_table.item(2, 1).text() == "Needs attention"
    assert "affine_mismatch" in dialog.issues_text.toPlainText()
    assert "seg_invalid_labels" in dialog.issues_text.toPlainText()
    assert not dialog.terms_checkbox.isEnabled()
    assert not dialog.save_button.isEnabled()

    dialog.set_language("zh_CN")
    assert dialog.windowTitle() == "安装本地 BraTS 2021 案例"
    assert dialog.modality_table.item(3, 1).text() == "未找到"
    assert "与 T1 参考几何不一致" in dialog.issues_text.toPlainText()
    assert "允许集合" in dialog.issues_text.toPlainText()
    assert "需要处理 3 个错误" in dialog.status_label.text()
    assert "处理列出的错误" in dialog.next_step_label.text()


def test_manifest_writer_receives_only_valid_confirmed_request(qtbot, tmp_path) -> None:
    source = tmp_path / "case"
    source.mkdir()
    destination = tmp_path / "anonymous-manifest.json"
    calls: list[tuple[BraTS2021ValidationReport, Path, bool]] = []

    def writer(
        report: BraTS2021ValidationReport,
        path: str | Path,
        *,
        terms_confirmed_by_user: bool,
    ) -> Path:
        calls.append((report, Path(path), terms_confirmed_by_user))
        return Path(path)

    dialog = _dialog(
        qtbot,
        validator=lambda _path: _valid_report(),
        writer=writer,
        directory_dialog=lambda _parent, _title: str(source),
        save_dialog=lambda _parent, _title, _suggested, _filter: (str(destination), _filter),
    )
    saved: list[str] = []
    dialog.manifestSaved.connect(saved.append)
    dialog.choose_button.click()
    qtbot.waitUntil(lambda: dialog.active_task is None, timeout=3_000)

    dialog.save_button.click()
    assert calls == []

    dialog.terms_checkbox.setChecked(True)
    dialog.save_button.click()

    assert calls == [(_valid_report(), destination, True)]
    assert saved == [str(destination)]
    assert "Anonymous manifest saved" in dialog.status_label.text()
    assert "Keep the validated source volumes in place" in dialog.next_step_label.text()


def test_existing_manifest_is_kept_by_default(qtbot, tmp_path) -> None:
    source = tmp_path / "case"
    source.mkdir()
    destination = tmp_path / "existing.json"
    destination.write_text("keep-me", encoding="utf-8")
    calls: list[Path] = []
    confirmations: list[Path] = []

    dialog = _dialog(
        qtbot,
        validator=lambda _path: _valid_report(),
        writer=lambda _report, path, **_kwargs: calls.append(Path(path)) or Path(path),
        directory_dialog=lambda _parent, _title: str(source),
        save_dialog=lambda _parent, _title, _suggested, _filter: (str(destination), _filter),
        overwrite_confirmer=lambda _parent, path: confirmations.append(path) is None and False,
    )
    dialog.choose_button.click()
    qtbot.waitUntil(lambda: dialog.active_task is None, timeout=3_000)
    dialog.terms_checkbox.setChecked(True)
    dialog.save_button.click()

    assert confirmations == [destination]
    assert calls == []
    assert destination.read_text(encoding="utf-8") == "keep-me"
    assert "kept unchanged" in dialog.status_label.text()


def test_cancel_and_close_discard_slow_background_result_safely(qtbot, tmp_path) -> None:
    source = tmp_path / "case"
    source.mkdir()
    started = threading.Event()
    release = threading.Event()
    finished = threading.Event()

    def validator(_path: str | Path) -> BraTS2021ValidationReport:
        started.set()
        release.wait(timeout=3)
        finished.set()
        return _valid_report()

    dialog = _dialog(
        qtbot,
        validator=validator,
        directory_dialog=lambda _parent, _title: str(source),
    )
    reports: list[object] = []
    dialog.validationFinished.connect(reports.append)
    dialog.choose_button.click()
    assert started.wait(timeout=1)
    task = dialog.active_task
    assert task is not None

    dialog.cancel_button.click()
    assert "Cancelling safely" in dialog.status_label.text()
    dialog.close()
    release.set()

    assert finished.wait(timeout=1)
    qtbot.waitUntil(lambda: task.done, timeout=3_000)
    assert task.cancelled
    assert dialog.active_task is None
    assert reports == []


def test_dialog_source_contains_no_forbidden_role_term() -> None:
    source = Path("src/workbench/ui/brats_installer.py").read_text(encoding="utf-8")
    assert "\u5b66\u751f" not in source
