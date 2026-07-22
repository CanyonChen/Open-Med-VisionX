from __future__ import annotations

import json
from pathlib import Path

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QFileDialog

from workbench.evaluation import DatasetManifest, DatasetSample
from workbench.ui.evaluation_workspace import EvaluationPage


def _manifest() -> DatasetManifest:
    return DatasetManifest(
        dataset_id="demo-dataset",
        dataset_version="v1",
        task="classification",
        license_id="CC-BY-4.0",
        samples=(
            DatasetSample(
                sample_id="sample-1",
                group_id="group-a",
                split="train",
                artifact_sha256="a" * 64,
                modality="MR",
            ),
            DatasetSample(
                sample_id="sample-2",
                group_id="group-b",
                split="test",
                artifact_sha256="b" * 64,
                modality="MR",
            ),
        ),
    )


def test_evaluation_page_runs_default_example_and_builds_pixel_free_record(qtbot) -> None:
    page = EvaluationPage()
    qtbot.addWidget(page)

    metrics = page.evaluate()

    assert metrics is not None
    assert page.metrics_table.rowCount() == 10
    assert page.calibration_table.rowCount() == 10
    assert page.export_button.isEnabled()
    assert page.experiment_record is not None
    payload = page.experiment_record.to_dict()
    assert payload["contains_phi"] is False
    assert payload["inputs"][0]["shape"] == [8, 2]
    assert "threshold" in payload["parameters"]


def test_evaluation_page_reports_invalid_vectors_inline(qtbot) -> None:
    page = EvaluationPage()
    qtbot.addWidget(page)
    page.truth_input.setPlainText("0, 1, 2")

    assert page.evaluate() is None
    assert "invalid" in page.status.text().lower()
    assert not page.export_button.isEnabled()


def test_evaluation_page_loads_group_safe_manifest_and_switches_language(
    qtbot,
    tmp_path: Path,
) -> None:
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(_manifest().to_dict()), encoding="utf-8")
    page = EvaluationPage()
    qtbot.addWidget(page)

    loaded = page.load_manifest(path)
    page.set_language("zh_CN")

    assert loaded is not None
    assert page.manifest_table.item(0, 1).text() == "1"
    assert page.manifest_table.item(2, 2).text() == "1"
    assert page.title.text() == "数据集与评价"
    assert "组级数据泄漏" in page.manifest_status.text()


def test_evaluation_workspace_reflows_at_narrow_width(qtbot) -> None:
    page = EvaluationPage()
    qtbot.addWidget(page)
    page.resize(760, 720)
    page.show()
    qtbot.wait(10)
    assert page.workspace.orientation() == Qt.Vertical

    page.resize(1200, 720)
    qtbot.wait(10)
    assert page.workspace.orientation() == Qt.Horizontal


def test_evaluation_page_exports_the_current_record(qtbot, tmp_path: Path, monkeypatch) -> None:
    destination = tmp_path / "evaluation.json"
    monkeypatch.setattr(
        QFileDialog,
        "getSaveFileName",
        lambda *_args, **_kwargs: (str(destination), "JSON (*.json)"),
    )
    page = EvaluationPage()
    qtbot.addWidget(page)
    assert page.evaluate() is not None

    page._choose_export()

    payload = json.loads(destination.read_text(encoding="utf-8"))
    assert payload["record_id"].startswith("evaluation-")
    assert payload["contains_phi"] is False
    assert "export" in page.status.text().lower()
