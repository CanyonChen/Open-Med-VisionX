from __future__ import annotations

import zipfile
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

from workbench.errors import DecodeError, OperationCancelled, ResourceLimitError
from workbench.io import (
    DicomLoader,
    SeriesSelectionRequiredError,
    discover_dicom_series,
)
from workbench.services import ImageService


def _write_slice(
    path: Path,
    *,
    study_uid: str,
    series_uid: str,
    frame_uid: str,
    position: float = 0.0,
    description: str = "AX T1",
    series_number: int = 1,
    orientation: tuple[float, ...] = (1, 0, 0, 0, 1, 0),
    number_of_frames: int | None = None,
) -> Path:
    meta = FileMetaDataset()
    meta.MediaStorageSOPClassUID = CTImageStorage
    meta.MediaStorageSOPInstanceUID = generate_uid()
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    meta.ImplementationClassUID = generate_uid()
    dataset = FileDataset(str(path), {}, file_meta=meta, preamble=b"\0" * 128)
    dataset.SOPClassUID = CTImageStorage
    dataset.SOPInstanceUID = meta.MediaStorageSOPInstanceUID
    dataset.StudyInstanceUID = study_uid
    dataset.SeriesInstanceUID = series_uid
    dataset.FrameOfReferenceUID = frame_uid
    dataset.SeriesDescription = description
    dataset.SeriesNumber = series_number
    dataset.Modality = "CT"
    dataset.PatientName = "SECRET^PERSON"
    dataset.PatientID = "SECRET-123"
    dataset.Rows = 2
    dataset.Columns = 3
    dataset.SamplesPerPixel = 1
    dataset.PhotometricInterpretation = "MONOCHROME2"
    dataset.BitsAllocated = 16
    dataset.BitsStored = 16
    dataset.HighBit = 15
    dataset.PixelRepresentation = 1
    dataset.ImageOrientationPatient = list(orientation)
    dataset.ImagePositionPatient = [0.0, 0.0, position]
    dataset.PixelSpacing = [0.75, 0.5]
    dataset.SliceThickness = 2.5
    dataset.SpacingBetweenSlices = 2.5
    dataset.RescaleSlope = 1
    dataset.RescaleIntercept = -1024
    dataset.InstanceNumber = int(position / 2.5) + 1
    if number_of_frames is not None:
        dataset.NumberOfFrames = number_of_frames
    dataset.PixelData = np.ones((2, 3), dtype=np.int16).tobytes()
    save_options = (
        {"enforce_file_format": True}
        if "enforce_file_format" in signature(dataset.save_as).parameters
        else {"write_like_original": False}
    )
    dataset.save_as(path, **save_options)
    return path


def test_discovery_groups_by_study_and_series_and_returns_phi_minimized_summaries(
    tmp_path: Path,
) -> None:
    shared_series_uid = generate_uid()
    study_a = generate_uid()
    study_b = generate_uid()
    frame_a = generate_uid()
    _write_slice(
        tmp_path / "a-1.dcm",
        study_uid=study_a,
        series_uid=shared_series_uid,
        frame_uid=frame_a,
        position=0.0,
    )
    _write_slice(
        tmp_path / "a-2.dcm",
        study_uid=study_a,
        series_uid=shared_series_uid,
        frame_uid=frame_a,
        position=2.5,
    )
    _write_slice(
        tmp_path / "b.dcm",
        study_uid=study_b,
        series_uid=shared_series_uid,
        frame_uid=generate_uid(),
        description="Patient JOHN",
        series_number=2,
    )
    (tmp_path / "notes.txt").write_text("not dicom", encoding="utf-8")

    discovery = discover_dicom_series(tmp_path)

    assert discovery.source_kind == "directory"
    assert discovery.inspected_member_count == 4
    assert discovery.dicom_object_count == 3
    assert discovery.skipped_member_count == 1
    assert len(discovery.series) == 2
    first, second = discovery.series
    assert first.instance_count == 2
    assert first.slice_count == 2
    assert first.frame_count == 2
    assert first.geometry_consistent
    assert first.supported_by_stable_loader
    assert first.pixel_spacing_mm == (0.75, 0.5)
    assert first.estimated_slice_spacing_mm == pytest.approx(2.5)
    assert second.series_description == "Withheld (untrusted DICOM free text)"
    assert any("PHI" in warning for warning in second.warnings)
    serialized = repr(discovery)
    assert study_a not in serialized
    assert study_b not in serialized
    assert shared_series_uid not in serialized
    assert "SECRET" not in serialized
    assert all(item.selector.startswith("sha256:") for item in discovery.series)
    with pytest.raises(FrozenInstanceError):
        first.modality = "MR"  # type: ignore[misc]


def test_multi_series_loader_refuses_to_guess_and_attaches_safe_candidates(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    study_uid = generate_uid()
    frame_uid = generate_uid()
    large_series_uid = generate_uid()
    for index in range(3):
        _write_slice(
            tmp_path / f"large-{index}.dcm",
            study_uid=study_uid,
            series_uid=large_series_uid,
            frame_uid=frame_uid,
            position=index * 2.5,
        )
    _write_slice(
        tmp_path / "small.dcm",
        study_uid=study_uid,
        series_uid=generate_uid(),
        frame_uid=generate_uid(),
        series_number=2,
    )

    real_dcmread = pydicom.dcmread
    read_modes: list[bool] = []

    def tracked_dcmread(*args: object, **kwargs: object) -> object:
        read_modes.append(bool(kwargs.get("stop_before_pixels", False)))
        return real_dcmread(*args, **kwargs)

    monkeypatch.setattr(pydicom, "dcmread", tracked_dcmread)

    with pytest.raises(SeriesSelectionRequiredError) as raised:
        DicomLoader().load(tmp_path)

    assert read_modes and all(read_modes), "initial multi-series scan must remain header-only"
    assert [candidate.instance_count for candidate in raised.value.candidates] == [3, 1]
    assert large_series_uid not in repr(raised.value.candidates)
    assert "will not guess" in str(raised.value)

    selector = raised.value.candidates[0].selector
    read_modes.clear()
    selected = DicomLoader().load(
        tmp_path,
        series_selector=selector,
    )
    assert selected.shape == (3, 2, 3)
    assert selected.runtime_metadata["slice_count"] == 3
    assert selected.runtime_metadata["series_selection"] == "explicit-user-selection"
    assert selected.runtime_metadata["series_selector_fingerprint"] == selector
    assert read_modes.count(False) == 3, "only the three selected instances may be read fully"

    with pytest.raises(DecodeError, match="missing or ambiguous"):
        DicomLoader().load(tmp_path, series_selector="sha256:" + "0" * 12)


def test_discovery_reads_headers_without_decoding_invalid_pixel_payload(tmp_path: Path) -> None:
    path = _write_slice(
        tmp_path / "header-only.dcm",
        study_uid=generate_uid(),
        series_uid=generate_uid(),
        frame_uid=generate_uid(),
    )
    dataset = pydicom.dcmread(path)
    dataset.PixelData = b"\x00\x01"
    save_options = (
        {"enforce_file_format": True}
        if "enforce_file_format" in signature(dataset.save_as).parameters
        else {"write_like_original": False}
    )
    dataset.save_as(path, **save_options)

    discovery = DicomLoader().discover_series(path)

    assert len(discovery.series) == 1
    assert discovery.series[0].rows == 2
    with pytest.raises(DecodeError, match="pixel data could not be decoded"):
        DicomLoader().load(path)


def test_discovery_reports_inconsistent_geometry_and_multi_frame_as_unsupported(
    tmp_path: Path,
) -> None:
    study_uid = generate_uid()
    series_uid = generate_uid()
    frame_uid = generate_uid()
    _write_slice(
        tmp_path / "first.dcm",
        study_uid=study_uid,
        series_uid=series_uid,
        frame_uid=frame_uid,
        number_of_frames=4,
    )
    _write_slice(
        tmp_path / "second.dcm",
        study_uid=study_uid,
        series_uid=series_uid,
        frame_uid=frame_uid,
        position=4.0,
        orientation=(0, 1, 0, 0, 0, 1),
    )

    summary = discover_dicom_series(tmp_path).series[0]

    assert summary.frame_count == 5
    assert not summary.geometry_consistent
    assert not summary.supported_by_stable_loader
    assert any("multi-frame" in warning for warning in summary.warnings)
    assert any("ImageOrientationPatient" in warning for warning in summary.warnings)


def test_discovery_is_cancellable_and_enforces_member_limits(tmp_path: Path) -> None:
    _write_slice(
        tmp_path / "one.dcm",
        study_uid=generate_uid(),
        series_uid=generate_uid(),
        frame_uid=generate_uid(),
    )

    with pytest.raises(OperationCancelled):
        discover_dicom_series(tmp_path, cancel=lambda: True)

    from workbench.io import LoadLimits

    with pytest.raises(ResourceLimitError, match="more files"):
        discover_dicom_series(tmp_path, limits=LoadLimits(max_zip_members=0))


def test_discovery_rejects_unsafe_zip_and_reports_all_invalid_members(tmp_path: Path) -> None:
    unsafe_zip = tmp_path / "unsafe.zip"
    with zipfile.ZipFile(unsafe_zip, "w") as archive:
        archive.writestr("../escape.dcm", b"not dicom")
    with pytest.raises(ResourceLimitError, match="unsafe path"):
        discover_dicom_series(unsafe_zip)

    invalid_zip = tmp_path / "invalid.zip"
    with zipfile.ZipFile(invalid_zip, "w") as archive:
        archive.writestr("readme.txt", b"not dicom")
        archive.writestr("other.bin", b"also not dicom")
    with pytest.raises(DecodeError, match=r"2 source member\(s\); 2 member\(s\)"):
        discover_dicom_series(invalid_zip)


def test_discovery_scans_safe_zip_without_extracting_members(tmp_path: Path) -> None:
    source = _write_slice(
        tmp_path / "source.dcm",
        study_uid=generate_uid(),
        series_uid=generate_uid(),
        frame_uid=generate_uid(),
    )
    archive_path = tmp_path / "study.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.write(source, "nested/image.dcm")
        archive.writestr("nested/readme.txt", b"not dicom")
    source.unlink()

    discovery = discover_dicom_series(archive_path)

    assert discovery.source_kind == "zip"
    assert discovery.inspected_member_count == 2
    assert discovery.dicom_object_count == 1
    assert discovery.skipped_member_count == 1
    assert len(discovery.series) == 1
    assert not (tmp_path / "nested").exists()


def test_zip_selection_reads_pixels_only_for_the_explicit_series(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    study_uid = generate_uid()
    selected_uid = generate_uid()
    selected_frame_uid = generate_uid()
    selected_paths = [
        _write_slice(
            tmp_path / f"selected-{index}.dcm",
            study_uid=study_uid,
            series_uid=selected_uid,
            frame_uid=selected_frame_uid,
            position=index * 2.5,
        )
        for index in range(2)
    ]
    other_path = _write_slice(
        tmp_path / "other.dcm",
        study_uid=study_uid,
        series_uid=generate_uid(),
        frame_uid=generate_uid(),
        series_number=2,
    )
    archive_path = tmp_path / "study.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        for path in (*selected_paths, other_path):
            archive.write(path, path.name)
    for path in (*selected_paths, other_path):
        path.unlink()

    real_dcmread = pydicom.dcmread
    read_modes: list[bool] = []

    def tracked_dcmread(*args: object, **kwargs: object) -> object:
        read_modes.append(bool(kwargs.get("stop_before_pixels", False)))
        return real_dcmread(*args, **kwargs)

    monkeypatch.setattr(pydicom, "dcmread", tracked_dcmread)
    with pytest.raises(SeriesSelectionRequiredError) as raised:
        DicomLoader().load(archive_path)
    assert read_modes and all(read_modes)

    selector = next(item.selector for item in raised.value.candidates if item.instance_count == 2)
    read_modes.clear()
    volume = DicomLoader().load(archive_path, series_selector=selector)

    assert volume.shape == (2, 2, 3)
    assert read_modes.count(False) == 2


def test_image_service_discovery_is_read_only(tmp_path: Path) -> None:
    _write_slice(
        tmp_path / "one.dcm",
        study_uid=generate_uid(),
        series_uid=generate_uid(),
        frame_uid=generate_uid(),
    )
    service = ImageService()
    generation_before = service.state.generation

    try:
        discovery = service.discover_dicom_series(tmp_path)
    finally:
        service.close()

    assert len(discovery.series) == 1
    assert service.state.generation == generation_before
    assert service.committed_study is None
