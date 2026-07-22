from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pytest

from workbench.models import (
    BUNDLED_MODEL_IDS,
    MODEL_BUDGET_BYTES,
    ModelBundleRecord,
    list_bundled_models,
    load_bundled_model,
    run_deepinv_mri_modl,
    run_dival_fbp_unet,
    run_monai_brats_patch,
    validate_bundle_golden,
    verify_all_bundles,
)


def test_bundled_model_release_gate() -> None:
    records = verify_all_bundles()

    assert tuple(record.bundle_id for record in records) == BUNDLED_MODEL_IDS
    assert all(isinstance(record, ModelBundleRecord) for record in records)
    assert all(record.verified for record in records)
    assert all(record.artifact_path.is_file() for record in records)
    assert all(record.golden_path.is_file() for record in records)
    assert all(record.licenses["redistribution_reviewed"] is True for record in records)
    assert (
        sum(record.artifact_size_bytes + record.golden_size_bytes for record in records)
        <= MODEL_BUDGET_BYTES
    )
    assert {record.bundle_id: record.artifact_format for record in records} == {
        "deepinv-mri-modl": "numpy-npz-state-dict",
        "dival-lodopab-fbpunet": "numpy-npz-state-dict",
        "monai-brats-segmentation": "torchscript",
    }


def test_catalog_exposes_bilingual_ui_metadata() -> None:
    for record in list_bundled_models(verify=False):
        assert record.id in BUNDLED_MODEL_IDS
        assert record.display_name
        assert record.display_name_zh
        assert record.description_en
        assert record.description_zh
        assert record.task
        assert record.modalities
        assert record.dimensionality in {"2d", "3d"}
        assert record.runtime
        assert record.license
        assert record.source_url.startswith("https://")
        assert record.limitations
        assert record.manifest_path.name == "bundle.yaml"
        assert record.weights_paths == (record.artifact_path,)
        assert not record.verified


@pytest.mark.parametrize("bundle_id", BUNDLED_MODEL_IDS)
def test_packaged_cpu_golden_contract(bundle_id: str) -> None:
    pytest.importorskip("torch")
    result = validate_bundle_golden(bundle_id, device="cpu")

    assert result.device == "cpu"
    assert result.maximum_absolute_error == 0.0
    assert result.maximum_relative_error == 0.0


def test_npz_model_payloads_are_numeric_and_pickle_free() -> None:
    records = list_bundled_models(verify=True)
    for record in records:
        if record.artifact_format != "numpy-npz-state-dict":
            continue
        with np.load(record.artifact_path, allow_pickle=False) as payload:
            assert payload.files
            assert all(not payload[name].dtype.hasobject for name in payload.files)
        assert record.artifact_path.name == "weights.npz"
        assert not (record.artifact_path.parent / "model.ts").exists()


def test_runtime_source_never_deserializes_pickle() -> None:
    model_source = Path(__file__).resolve().parents[1] / "src" / "workbench" / "models"
    for path in model_source.glob("*.py"):
        assert "torch.load(" not in path.read_text(encoding="utf-8")


def test_auto_device_prefers_cuda_when_available() -> None:
    torch = pytest.importorskip("torch")
    runtime = load_bundled_model("deepinv-mri-modl", device="auto")
    assert runtime.device == ("cuda" if torch.cuda.is_available() else "cpu")


def test_read_only_teaching_input_is_copied_without_warning_or_mutation() -> None:
    pytest.importorskip("torch")
    fbp = np.zeros((362, 362), dtype=np.float32)
    fbp.flags.writeable = False

    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        result = run_dival_fbp_unet(fbp, device="cpu")

    assert not fbp.flags.writeable
    assert np.count_nonzero(fbp) == 0
    assert result.outputs["reconstruction"].shape == fbp.shape


def test_task_adapters_expose_scientific_semantics() -> None:
    pytest.importorskip("torch")

    ct = run_dival_fbp_unet(np.zeros((362, 362), dtype=np.float32), device="cpu")
    assert ct.outputs["reconstruction"].shape == (362, 362)
    assert not ct.outputs["reconstruction"].flags.writeable
    assert "not HU" in ct.warnings[0]

    kspace = np.zeros((32, 32), dtype=np.complex64)
    mask = np.zeros((32, 32), dtype=bool)
    mask[::4] = True
    mri = run_deepinv_mri_modl(
        kspace,
        mask,
        source_kind="synthetic-phantom",
        device="cpu",
    )
    assert mri.outputs["complex_channels"].shape == (2, 32, 32)
    assert mri.outputs["magnitude"].shape == (32, 32)
    assert "not scanner raw data" in mri.warnings[0]

    coordinates = np.indices((32, 32, 32), dtype=np.float32)
    volume = np.stack(
        [1.0 + coordinates[index % 3] + index for index in range(4)],
        axis=0,
    )
    brats = run_monai_brats_patch(volume, device="cpu")
    assert brats.output_channels == ("WT", "TC", "ET")
    assert brats.outputs["probabilities"].shape == (3, 32, 32, 32)
    assert brats.outputs["masks"].shape == (3, 32, 32, 32)
    assert not brats.outputs["masks"].flags.writeable
