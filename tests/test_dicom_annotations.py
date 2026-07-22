from __future__ import annotations

from dataclasses import dataclass
from inspect import signature
from pathlib import Path

import numpy as np
import pytest
from pydicom.dataset import Dataset, FileDataset, FileMetaDataset
from pydicom.pixels import pack_bits
from pydicom.sequence import Sequence
from pydicom.uid import (
    CTImageStorage,
    ExplicitVRLittleEndian,
    RTStructureSetStorage,
    SegmentationStorage,
    generate_uid,
)

from workbench.domain.images import IntensitySemantics, SourceType
from workbench.domain.studies import (
    CompressionKind,
    ContourLayer,
    FractionalType,
    GeometryMatchStatus,
    ImageSeries,
    LayerValidationState,
    SegmentationLayer,
    SegmentationValueType,
    SourceFormat,
    SourceReference,
    SpatialGeometry,
)
from workbench.errors import DecodeError, ResourceLimitError, UnsupportedFormatError
from workbench.io.dicom_annotations import (
    DicomAnnotationKind,
    DicomAnnotationLimits,
    import_dicom_annotation,
)


@dataclass(frozen=True)
class ReferenceFixture:
    series: ImageSeries
    study_uid: str
    series_uid: str
    frame_uid: str
    sop_uids: tuple[str, ...]


def _reference_fixture(
    *,
    orientation: tuple[float, ...] = (1, 0, 0, 0, 1, 0),
    positions: tuple[tuple[float, float, float], ...] = ((0, 0, 0), (0, 0, 2)),
) -> ReferenceFixture:
    study_uid = generate_uid()
    series_uid = generate_uid()
    frame_uid = generate_uid()
    sop_uids = tuple(generate_uid() for _ in positions)
    first_axis_lps = np.asarray(orientation[:3], dtype=np.float64)
    second_axis_lps = np.asarray(orientation[3:], dtype=np.float64)
    normal_lps = np.cross(first_axis_lps, second_axis_lps)
    projections = np.asarray([np.dot(position, normal_lps) for position in positions])
    order = np.argsort(projections)
    sorted_positions = [positions[int(index)] for index in order]
    spacing_z = float(np.diff(np.sort(projections))[0]) if len(positions) > 1 else 2.0
    affine_lps = np.eye(4, dtype=np.float64)
    affine_lps[:3, 0] = first_axis_lps * 0.5
    affine_lps[:3, 1] = second_axis_lps * 0.75
    affine_lps[:3, 2] = normal_lps * spacing_z
    affine_lps[:3, 3] = sorted_positions[0]
    affine_ras = np.diag((-1.0, -1.0, 1.0, 1.0)) @ affine_lps
    geometry = SpatialGeometry((len(positions), 2, 3), affine_ras)
    source = SourceReference(
        source_id="reference-source",
        source_type=SourceType.DICOM,
        source_format=SourceFormat.DICOM,
        compression=CompressionKind.NONE,
    )
    return ReferenceFixture(
        series=ImageSeries(
            series_id="reference-series",
            modality="CT",
            source=source,
            geometry=geometry,
            intensity_semantics=IntensitySemantics.HOUNSFIELD_UNIT,
            study_instance_uid=study_uid,
            series_instance_uid=series_uid,
            frame_of_reference_uid=frame_uid,
        ),
        study_uid=study_uid,
        series_uid=series_uid,
        frame_uid=frame_uid,
        sop_uids=sop_uids,
    )


def _file_dataset(path: Path, sop_class_uid: str) -> FileDataset:
    meta = FileMetaDataset()
    meta.MediaStorageSOPClassUID = sop_class_uid
    meta.MediaStorageSOPInstanceUID = generate_uid()
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    meta.ImplementationClassUID = generate_uid()
    dataset = FileDataset(str(path), {}, file_meta=meta, preamble=b"\0" * 128)
    dataset.SOPClassUID = sop_class_uid
    dataset.SOPInstanceUID = meta.MediaStorageSOPInstanceUID
    return dataset


def _save_dataset(dataset: FileDataset, path: Path) -> Path:
    options = (
        {"enforce_file_format": True}
        if "enforce_file_format" in signature(dataset.save_as).parameters
        else {"write_like_original": False}
    )
    dataset.save_as(path, **options)
    return path


def _write_seg(
    path: Path,
    reference: ReferenceFixture,
    *,
    frames: np.ndarray,
    segment_numbers: tuple[int, ...],
    positions: tuple[tuple[float, float, float], ...],
    source_indexes: tuple[int, ...],
    orientation: tuple[float, ...] = (1, 0, 0, 0, 1, 0),
    segmentation_type: str = "BINARY",
    referenced_series_uid: str | None = None,
    frame_of_reference_uid: str | None = None,
    include_orientation: bool = True,
) -> Path:
    dataset = _file_dataset(path, str(SegmentationStorage))
    dataset.StudyInstanceUID = reference.study_uid
    dataset.SeriesInstanceUID = generate_uid()
    dataset.FrameOfReferenceUID = frame_of_reference_uid or reference.frame_uid
    dataset.Modality = "SEG"
    dataset.Rows = int(frames.shape[1])
    dataset.Columns = int(frames.shape[2])
    dataset.NumberOfFrames = int(frames.shape[0])
    dataset.SamplesPerPixel = 1
    dataset.PhotometricInterpretation = "MONOCHROME2"
    dataset.PixelRepresentation = 0
    dataset.SegmentationType = segmentation_type
    if segmentation_type == "BINARY":
        dataset.BitsAllocated = 1
        dataset.BitsStored = 1
        dataset.HighBit = 0
        dataset.PixelData = pack_bits(frames.astype(np.uint8))
    else:
        dataset.BitsAllocated = 8
        dataset.BitsStored = 8
        dataset.HighBit = 7
        dataset.MaximumFractionalValue = 100
        dataset.SegmentationFractionalType = "PROBABILITY"
        dataset.PixelData = frames.astype(np.uint8).tobytes()

    definitions = []
    for number in sorted(set(segment_numbers)):
        definition = Dataset()
        definition.SegmentNumber = number
        definition.SegmentLabel = f"Region {number}"
        definitions.append(definition)
    dataset.SegmentSequence = Sequence(definitions)

    referenced_series = Dataset()
    referenced_series.SeriesInstanceUID = referenced_series_uid or reference.series_uid
    referenced_instances = []
    for sop_uid in reference.sop_uids:
        instance = Dataset()
        instance.ReferencedSOPClassUID = CTImageStorage
        instance.ReferencedSOPInstanceUID = sop_uid
        referenced_instances.append(instance)
    referenced_series.ReferencedInstanceSequence = Sequence(referenced_instances)
    dataset.ReferencedSeriesSequence = Sequence([referenced_series])

    shared = Dataset()
    if include_orientation:
        orientation_item = Dataset()
        orientation_item.ImageOrientationPatient = list(orientation)
        shared.PlaneOrientationSequence = Sequence([orientation_item])
    measures = Dataset()
    measures.PixelSpacing = [0.75, 0.5]
    measures.SpacingBetweenSlices = 2.0
    measures.SliceThickness = 2.0
    shared.PixelMeasuresSequence = Sequence([measures])
    dataset.SharedFunctionalGroupsSequence = Sequence([shared])

    per_frame = []
    for segment_number, position, source_index in zip(
        segment_numbers, positions, source_indexes, strict=True
    ):
        group = Dataset()
        identification = Dataset()
        identification.ReferencedSegmentNumber = segment_number
        group.SegmentIdentificationSequence = Sequence([identification])
        plane_position = Dataset()
        plane_position.ImagePositionPatient = list(position)
        group.PlanePositionSequence = Sequence([plane_position])
        derivation = Dataset()
        source = Dataset()
        source.ReferencedSOPClassUID = CTImageStorage
        source.ReferencedSOPInstanceUID = reference.sop_uids[source_index]
        derivation.SourceImageSequence = Sequence([source])
        group.DerivationImageSequence = Sequence([derivation])
        per_frame.append(group)
    dataset.PerFrameFunctionalGroupsSequence = Sequence(per_frame)
    return _save_dataset(dataset, path)


def _write_rtstruct(
    path: Path,
    reference: ReferenceFixture,
    *,
    referenced_series_uid: str | None = None,
    frame_of_reference_uid: str | None = None,
) -> Path:
    dataset = _file_dataset(path, str(RTStructureSetStorage))
    dataset.StudyInstanceUID = reference.study_uid
    dataset.SeriesInstanceUID = generate_uid()
    dataset.Modality = "RTSTRUCT"

    selected_frame_uid = frame_of_reference_uid or reference.frame_uid
    frame_reference = Dataset()
    frame_reference.FrameOfReferenceUID = selected_frame_uid
    study_reference = Dataset()
    study_reference.ReferencedSOPClassUID = "1.2.840.10008.3.1.2.3.1"
    study_reference.ReferencedSOPInstanceUID = reference.study_uid
    series_reference = Dataset()
    series_reference.SeriesInstanceUID = referenced_series_uid or reference.series_uid
    referenced_images = []
    for sop_uid in reference.sop_uids:
        image = Dataset()
        image.ReferencedSOPClassUID = CTImageStorage
        image.ReferencedSOPInstanceUID = sop_uid
        referenced_images.append(image)
    series_reference.ContourImageSequence = Sequence(referenced_images)
    study_reference.RTReferencedSeriesSequence = Sequence([series_reference])
    frame_reference.RTReferencedStudySequence = Sequence([study_reference])
    dataset.ReferencedFrameOfReferenceSequence = Sequence([frame_reference])

    structure = Dataset()
    structure.ROINumber = 1
    structure.ReferencedFrameOfReferenceUID = selected_frame_uid
    structure.ROIName = "Liver\tROI"
    dataset.StructureSetROISequence = Sequence([structure])

    roi_contour = Dataset()
    roi_contour.ReferencedROINumber = 1
    roi_contour.ROIDisplayColor = [12, 34, 56]
    contour = Dataset()
    contour.ContourGeometricType = "CLOSED_PLANAR"
    contour.NumberOfContourPoints = 4
    contour.ContourData = [
        1.0,
        2.0,
        0.0,
        3.0,
        2.0,
        0.0,
        3.0,
        5.0,
        0.0,
        1.0,
        5.0,
        0.0,
    ]
    contour_image = Dataset()
    contour_image.ReferencedSOPClassUID = CTImageStorage
    contour_image.ReferencedSOPInstanceUID = reference.sop_uids[0]
    contour.ContourImageSequence = Sequence([contour_image])
    roi_contour.ContourSequence = Sequence([contour])
    dataset.ROIContourSequence = Sequence([roi_contour])
    return _save_dataset(dataset, path)


def test_binary_seg_creates_one_native_layer_per_segment(tmp_path: Path) -> None:
    reference = _reference_fixture()
    frames = np.asarray(
        [
            [[1, 1, 0], [0, 0, 0]],  # Segment 1, z=2
            [[0, 1, 0], [0, 1, 0]],  # Segment 2, z=0
            [[1, 0, 0], [0, 0, 1]],  # Segment 1, z=0
            [[1, 0, 1], [1, 0, 1]],  # Segment 2, z=2
        ],
        dtype=np.uint8,
    )
    path = _write_seg(
        tmp_path / "binary-seg.dcm",
        reference,
        frames=frames,
        segment_numbers=(1, 2, 1, 2),
        positions=((0, 0, 2), (0, 0, 0), (0, 0, 0), (0, 0, 2)),
        source_indexes=(1, 0, 0, 1),
    )

    result = import_dicom_annotation(path, reference_series=reference.series)

    assert result.kind is DicomAnnotationKind.SEGMENTATION
    assert len(result.layers) == 2
    assert all(isinstance(layer, SegmentationLayer) for layer in result.layers)
    first, second = result.layers
    assert first.value_type is SegmentationValueType.BINARY
    assert first.labels[0].value == 1
    assert first.presentation.color == "#0A84FF"
    np.testing.assert_array_equal(first.array, frames[[2, 0]])
    np.testing.assert_array_equal(second.array, frames[[1, 3]])
    assert first.array.dtype == frames.dtype
    assert not first.array.flags.writeable
    assert first.validation_state is LayerValidationState.VALIDATED
    assert first.reference.dicom_series_uid == reference.series_uid
    assert first.reference.frame_of_reference_uid == reference.frame_uid
    assert set(first.reference.referenced_sop_instance_uids) == set(reference.sop_uids)
    assert first.reference_report(reference.series).status is GeometryMatchStatus.MATCHED
    assert first.source.content_sha256 is not None
    assert str(tmp_path) not in repr(first.source.provenance)
    assert str(tmp_path) not in repr(first.provenance)


def test_fractional_seg_preserves_stored_scale_and_fractional_type(tmp_path: Path) -> None:
    reference = _reference_fixture()
    frames = np.asarray(
        [
            [[0, 25, 100], [50, 75, 100]],
            [[100, 75, 50], [25, 0, 10]],
        ],
        dtype=np.uint8,
    )
    path = _write_seg(
        tmp_path / "fractional-seg.dcm",
        reference,
        frames=frames,
        segment_numbers=(1, 1),
        positions=((0, 0, 0), (0, 0, 2)),
        source_indexes=(0, 1),
        segmentation_type="FRACTIONAL",
    )

    result = import_dicom_annotation(path, reference_series=reference.series)
    layer = result.layers[0]

    assert isinstance(layer, SegmentationLayer)
    assert layer.value_type is SegmentationValueType.FRACTIONAL
    assert layer.maximum_fractional_value == 100
    assert layer.fractional_type is FractionalType.PROBABILITY
    assert layer.provenance["native_values_preserved"] is True
    np.testing.assert_array_equal(layer.array, frames)
    assert layer.array.dtype == np.uint8
    assert layer.statistics[0].maximum == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("wrong_series", "wrong_frame", "expected"),
    [
        (True, False, "Series Instance UID"),
        (False, True, "Frame of Reference UID"),
    ],
)
def test_seg_rejects_reference_identity_mismatch(
    tmp_path: Path,
    wrong_series: bool,
    wrong_frame: bool,
    expected: str,
) -> None:
    reference = _reference_fixture()
    frames = np.zeros((2, 2, 3), dtype=np.uint8)
    path = _write_seg(
        tmp_path / "mismatch.dcm",
        reference,
        frames=frames,
        segment_numbers=(1, 1),
        positions=((0, 0, 0), (0, 0, 2)),
        source_indexes=(0, 1),
        referenced_series_uid=generate_uid() if wrong_series else reference.series_uid,
        frame_of_reference_uid=generate_uid() if wrong_frame else reference.frame_uid,
    )

    with pytest.raises(DecodeError, match=expected):
        import_dicom_annotation(path, reference_series=reference.series)


def test_rtstruct_imports_safe_rois_and_converts_lps_points_to_ras(tmp_path: Path) -> None:
    reference = _reference_fixture()
    path = _write_rtstruct(tmp_path / "contours.dcm", reference)

    result = import_dicom_annotation(path, reference_series=reference.series)
    layer = result.layers[0]

    assert result.kind is DicomAnnotationKind.RTSTRUCT
    assert isinstance(layer, ContourLayer)
    assert layer.validation_state is LayerValidationState.VALIDATED
    assert layer.reference.dicom_series_uid == reference.series_uid
    assert layer.reference.frame_of_reference_uid == reference.frame_uid
    assert layer.reference_report(reference.series).status is GeometryMatchStatus.MATCHED
    roi = layer.rois[0]
    assert roi.name == "Liver ROI"
    assert roi.color == "#0C2238"
    np.testing.assert_allclose(
        roi.contours[0].points_ras,
        [
            [-1.0, -2.0, 0.0],
            [-3.0, -2.0, 0.0],
            [-3.0, -5.0, 0.0],
            [-1.0, -5.0, 0.0],
        ],
    )
    assert roi.contours[0].referenced_sop_instance_uid == reference.sop_uids[0]
    assert not roi.contours[0].points_ras.flags.writeable
    assert str(tmp_path) not in repr(layer.source.provenance)


@pytest.mark.parametrize(
    ("wrong_series", "wrong_frame", "expected"),
    [
        (True, False, "Series Instance UID"),
        (False, True, "Frame of Reference UID"),
    ],
)
def test_rtstruct_rejects_reference_identity_mismatch(
    tmp_path: Path,
    wrong_series: bool,
    wrong_frame: bool,
    expected: str,
) -> None:
    reference = _reference_fixture()
    path = _write_rtstruct(
        tmp_path / "rt-mismatch.dcm",
        reference,
        referenced_series_uid=generate_uid() if wrong_series else reference.series_uid,
        frame_of_reference_uid=generate_uid() if wrong_frame else reference.frame_uid,
    )

    with pytest.raises(DecodeError, match=expected):
        import_dicom_annotation(path, reference_series=reference.series)


def test_seg_geometry_handles_sagittal_direction_and_ras_conversion(tmp_path: Path) -> None:
    orientation = (0, 1, 0, 0, 0, 1)
    positions = ((0, 0, 0), (2, 0, 0))
    reference = _reference_fixture(orientation=orientation, positions=positions)
    frames = np.asarray(
        [
            [[1, 0, 0], [0, 0, 0]],
            [[0, 0, 0], [0, 0, 1]],
        ],
        dtype=np.uint8,
    )
    path = _write_seg(
        tmp_path / "sagittal-seg.dcm",
        reference,
        frames=frames,
        segment_numbers=(1, 1),
        positions=positions,
        source_indexes=(0, 1),
        orientation=orientation,
    )

    layer = import_dicom_annotation(path, reference_series=reference.series).layers[0]

    np.testing.assert_allclose(
        layer.geometry.affine_ras,
        [
            [0.0, 0.0, -2.0, 0.0],
            [-0.5, 0.0, 0.0, 0.0],
            [0.0, 0.75, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
    )
    assert layer.reference_report(reference.series).status is GeometryMatchStatus.MATCHED


def test_seg_rejects_missing_and_nonuniform_functional_group_geometry(
    tmp_path: Path,
) -> None:
    reference = _reference_fixture(positions=((0, 0, 0), (0, 0, 2), (0, 0, 4)))
    frames = np.zeros((3, 2, 3), dtype=np.uint8)
    missing = _write_seg(
        tmp_path / "missing-orientation.dcm",
        reference,
        frames=frames,
        segment_numbers=(1, 1, 1),
        positions=((0, 0, 0), (0, 0, 2), (0, 0, 4)),
        source_indexes=(0, 1, 2),
        include_orientation=False,
    )
    with pytest.raises(DecodeError, match="missing ImageOrientationPatient"):
        import_dicom_annotation(missing, reference_series=reference.series)

    nonuniform = _write_seg(
        tmp_path / "nonuniform.dcm",
        reference,
        frames=frames,
        segment_numbers=(1, 1, 1),
        positions=((0, 0, 0), (0, 0, 2), (0, 0, 5)),
        source_indexes=(0, 1, 2),
    )
    with pytest.raises(DecodeError, match="non-uniform"):
        import_dicom_annotation(nonuniform, reference_series=reference.series)


def test_annotation_dispatch_rejects_ordinary_dicom_before_pixel_decode(
    tmp_path: Path,
) -> None:
    reference = _reference_fixture()
    path = tmp_path / "ordinary-ct.dcm"
    dataset = _file_dataset(path, str(CTImageStorage))
    dataset.StudyInstanceUID = reference.study_uid
    dataset.SeriesInstanceUID = reference.series_uid
    dataset.Modality = "CT"
    _save_dataset(dataset, path)

    with pytest.raises(UnsupportedFormatError, match="ordinary image-storage"):
        import_dicom_annotation(path, reference_series=reference.series)


def test_encoded_input_limit_is_enforced_before_dicom_decode(tmp_path: Path) -> None:
    reference = _reference_fixture()
    path = tmp_path / "too-large.dcm"
    path.write_bytes(b"0" * 32)

    with pytest.raises(ResourceLimitError, match="encoded-size"):
        import_dicom_annotation(
            path,
            reference_series=reference.series,
            limits=DicomAnnotationLimits(max_file_bytes=16),
        )
