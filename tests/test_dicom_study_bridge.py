from __future__ import annotations

from dataclasses import FrozenInstanceError
from inspect import signature
from pathlib import Path

import numpy as np
import pytest

try:
    import pydicom
    from pydicom.dataset import FileDataset, FileMetaDataset
    from pydicom.uid import CTImageStorage, ExplicitVRLittleEndian, generate_uid
except ImportError:
    pytest.skip("pydicom is not installed", allow_module_level=True)

from workbench.domain import ImageStudy, VolumeLayer
from workbench.errors import DecodeError
from workbench.io import DicomLoader, ImageLoaderRegistry, discover_dicom_series
from workbench.services import ImageService


def _write_slice(
    path: Path,
    *,
    study_uid: str | None,
    series_uid: str,
    frame_uid: str,
    position: float,
) -> Path:
    meta = FileMetaDataset()
    meta.MediaStorageSOPClassUID = CTImageStorage
    meta.MediaStorageSOPInstanceUID = generate_uid()
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    meta.ImplementationClassUID = generate_uid()
    dataset = FileDataset(str(path), {}, file_meta=meta, preamble=b"\0" * 128)
    dataset.SOPClassUID = CTImageStorage
    dataset.SOPInstanceUID = meta.MediaStorageSOPInstanceUID
    if study_uid is not None:
        dataset.StudyInstanceUID = study_uid
    dataset.SeriesInstanceUID = series_uid
    dataset.FrameOfReferenceUID = frame_uid
    dataset.Modality = "CT"
    dataset.Rows = 2
    dataset.Columns = 3
    dataset.SamplesPerPixel = 1
    dataset.PhotometricInterpretation = "MONOCHROME2"
    dataset.BitsAllocated = 16
    dataset.BitsStored = 16
    dataset.HighBit = 15
    dataset.PixelRepresentation = 1
    dataset.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]
    dataset.ImagePositionPatient = [0.0, 0.0, position]
    dataset.PixelSpacing = [0.75, 0.5]
    dataset.SliceThickness = 2.5
    dataset.SpacingBetweenSlices = 2.5
    dataset.RescaleSlope = 1
    dataset.RescaleIntercept = -1024
    dataset.InstanceNumber = int(position / 2.5) + 1
    dataset.PixelData = np.full((2, 3), int(position) + 1, dtype=np.int16).tobytes()
    save_options = (
        {"enforce_file_format": True}
        if "enforce_file_format" in signature(dataset.save_as).parameters
        else {"write_like_original": False}
    )
    dataset.save_as(path, **save_options)
    return path


def test_load_with_reference_reuses_selected_complete_dataset_reads(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    study_uid = generate_uid()
    series_uid = generate_uid()
    frame_uid = generate_uid()
    for index in range(2):
        _write_slice(
            tmp_path / f"selected-{index}.dcm",
            study_uid=study_uid,
            series_uid=series_uid,
            frame_uid=frame_uid,
            position=index * 2.5,
        )
    _write_slice(
        tmp_path / "other.dcm",
        study_uid=study_uid,
        series_uid=generate_uid(),
        frame_uid=generate_uid(),
        position=0.0,
    )
    selector = next(
        item.selector for item in discover_dicom_series(tmp_path).series if item.instance_count == 2
    )

    real_dcmread = pydicom.dcmread
    read_modes: list[bool] = []

    def tracked_dcmread(*args: object, **kwargs: object) -> object:
        read_modes.append(bool(kwargs.get("stop_before_pixels", False)))
        return real_dcmread(*args, **kwargs)

    monkeypatch.setattr(pydicom, "dcmread", tracked_dcmread)
    result = DicomLoader().load_with_reference(tmp_path, series_selector=selector)

    assert result.volume.shape == (2, 2, 3)
    assert result.reference.selector == selector
    assert result.reference.study_instance_uid == study_uid
    assert result.reference.series_instance_uid == series_uid
    assert result.reference.frame_of_reference_uid == frame_uid
    assert read_modes.count(False) == 2
    assert study_uid not in repr(result.reference)
    assert series_uid not in repr(result.reference)
    assert frame_uid not in repr(result.reference)
    runtime_text = repr(dict(result.volume.runtime_metadata))
    assert study_uid not in runtime_text
    assert series_uid not in runtime_text
    assert frame_uid not in runtime_text
    with pytest.raises(FrozenInstanceError):
        result.reference.selector = "sha256:000000000000"  # type: ignore[misc]


def test_reference_bridge_requires_exact_study_identity_without_breaking_plain_load(
    tmp_path: Path,
) -> None:
    path = _write_slice(
        tmp_path / "missing-study.dcm",
        study_uid=None,
        series_uid=generate_uid(),
        frame_uid=generate_uid(),
        position=0.0,
    )

    assert DicomLoader().load(path).shape == (1, 2, 3)
    with pytest.raises(DecodeError, match="StudyInstanceUID"):
        DicomLoader().load_with_reference(path)


def test_image_service_builds_one_safe_base_study_for_dicom(tmp_path: Path) -> None:
    study_uid = generate_uid()
    series_uid = generate_uid()
    frame_uid = generate_uid()
    for index in range(2):
        _write_slice(
            tmp_path / f"slice-{index}.dcm",
            study_uid=study_uid,
            series_uid=series_uid,
            frame_uid=frame_uid,
            position=index * 2.5,
        )

    service = ImageService(registry=ImageLoaderRegistry((DicomLoader(),)))
    try:
        loaded = service.begin_load(tmp_path).result(timeout=10)
    finally:
        service.close()

    assert isinstance(loaded.domain_study, ImageStudy)
    reference_series = loaded.reference_series
    assert reference_series is not None
    assert reference_series.series_id.startswith("sha256:")
    assert reference_series.source.source_id == reference_series.series_id
    assert reference_series.study_instance_uid == study_uid
    assert reference_series.series_instance_uid == series_uid
    assert reference_series.frame_of_reference_uid == frame_uid
    assert len(reference_series.layers) == 1
    base_layer = reference_series.layers[0]
    assert isinstance(base_layer, VolumeLayer)
    assert base_layer.is_base_image
    assert base_layer.volume is loaded.image
    assert base_layer.series_id == reference_series.series_id
    assert service.committed_study is loaded
    loaded_repr = repr(loaded)
    assert study_uid not in loaded_repr
    assert series_uid not in loaded_repr
    assert frame_uid not in loaded_repr
    assert not any(
        value in repr(dict(loaded.image.runtime_metadata))
        for value in (study_uid, series_uid, frame_uid)
    )


def test_non_dicom_loaded_study_keeps_domain_bridge_optional(tmp_path: Path) -> None:
    pillow = pytest.importorskip("PIL.Image")
    path = tmp_path / "image.png"
    pillow.fromarray(np.zeros((4, 5), dtype=np.uint8)).save(path)

    service = ImageService()
    try:
        loaded = service.begin_load(path).result(timeout=10)
    finally:
        service.close()

    assert loaded.domain_study is None
    assert loaded.reference_series is None
