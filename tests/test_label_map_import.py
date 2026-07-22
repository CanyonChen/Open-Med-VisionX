from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from workbench.domain.images import IntensitySemantics, SourceType
from workbench.domain.studies import (
    CompressionKind,
    GeometryMatchStatus,
    ImageSeries,
    LabelDefinition,
    LayerValidationState,
    SegmentationValueType,
    SourceFormat,
    SourceReference,
    SpatialGeometry,
)
from workbench.errors import ResourceLimitError, UnsupportedFormatError, ValidationError
from workbench.io import (
    LabelMapLimits,
    NiftiVolumeSelectionRequiredError,
    import_label_map,
)


def _reference_series(
    shape_zyx: tuple[int, int, int],
    affine: np.ndarray | None = None,
) -> ImageSeries:
    geometry = SpatialGeometry(
        shape_zyx,
        np.eye(4, dtype=np.float64) if affine is None else affine,
    )
    return ImageSeries(
        series_id="selected-reference",
        modality="MR",
        source=SourceReference(
            source_id="selected-reference-source",
            source_type=SourceType.GENERATED,
            source_format=SourceFormat.GENERATED,
            compression=CompressionKind.NONE,
        ),
        geometry=geometry,
        intensity_semantics=IntensitySemantics.ARBITRARY_SIGNAL,
    )


def test_png_import_is_discrete_immutable_and_matched_to_explicit_reference(
    tmp_path: Path,
) -> None:
    patient_named_directory = tmp_path / "private-subject-folder"
    patient_named_directory.mkdir()
    path = patient_named_directory / "private-label-name.png"
    stored = np.asarray([[0, 2, 2], [0, 0, 5]], dtype=np.uint8)
    Image.fromarray(stored).save(path)
    reference = _reference_series((1, 2, 3))

    layer = import_label_map(path, reference_series=reference)

    assert layer.value_type is SegmentationValueType.DISCRETE
    assert layer.validation_state is LayerValidationState.VALIDATED
    assert layer.presentation.visible is True
    assert layer.reference_report(reference).status is GeometryMatchStatus.MATCHED
    assert layer.reference_report(reference).overlay_allowed is True
    np.testing.assert_array_equal(layer.array, stored)
    assert not layer.array.flags.writeable
    assert tuple(item.value for item in layer.labels) == (2, 5)
    assert tuple(item.name for item in layer.labels) == ("Label 2", "Label 5")
    assert layer.source.source_format is SourceFormat.PNG
    assert layer.source.compression is CompressionKind.LOSSLESS
    assert layer.source.content_sha256 is not None
    assert str(patient_named_directory) not in repr(layer.source.provenance)
    assert path.name not in repr(layer.provenance)


def test_caller_label_schema_is_preserved_and_must_cover_stored_values(
    tmp_path: Path,
) -> None:
    path = tmp_path / "labels.png"
    Image.fromarray(np.asarray([[0, 7], [7, 0]], dtype=np.uint8)).save(path)
    reference = _reference_series((1, 2, 2))
    schema = (LabelDefinition(value=7, name="Reviewed region", color="#30D158"),)

    layer = import_label_map(
        path,
        reference_series=reference,
        label_schema=schema,
    )

    assert layer.labels == schema
    assert layer.provenance["label_schema_origin"] == "caller-supplied"
    with pytest.raises(ValidationError, match="missing stored values"):
        import_label_map(
            path,
            reference_series=reference,
            label_schema=(LabelDefinition(3, "Wrong", "#0A84FF"),),
        )


def test_jpeg_is_always_rejected_as_a_label_map(tmp_path: Path) -> None:
    path = tmp_path / "lossy-label.jpg"
    Image.fromarray(np.zeros((4, 4), dtype=np.uint8)).save(path, quality=100)

    with pytest.raises(UnsupportedFormatError, match="always rejected.*lossy"):
        import_label_map(path, reference_series=_reference_series((1, 4, 4)))


def test_lossless_multipage_tiff_preserves_integer_pages(tmp_path: Path) -> None:
    path = tmp_path / "labels.tiff"
    stored = np.asarray(
        [
            [[0, 1, 1], [0, 0, 0]],
            [[0, 0, 0], [2, 2, 0]],
        ],
        dtype=np.uint8,
    )
    pages = [Image.fromarray(frame) for frame in stored]
    pages[0].save(
        path,
        save_all=True,
        append_images=pages[1:],
        compression="tiff_lzw",
    )

    layer = import_label_map(path, reference_series=_reference_series(stored.shape))

    np.testing.assert_array_equal(layer.array, stored)
    assert layer.source.source_format is SourceFormat.TIFF
    assert layer.source.compression is CompressionKind.LOSSLESS
    assert layer.source.provenance["lossless_verified"] is True
    assert layer.source.provenance["compression_codes"] == (5, 5)


def test_color_tiff_is_not_treated_as_a_label_map(tmp_path: Path) -> None:
    path = tmp_path / "color-label.tif"
    Image.fromarray(np.zeros((3, 4, 3), dtype=np.uint8), mode="RGB").save(
        path,
        compression="tiff_lzw",
    )

    with pytest.raises(ValidationError, match="single-channel integer"):
        import_label_map(path, reference_series=_reference_series((1, 3, 4)))


def test_unverifiable_or_lossy_tiff_compression_is_rejected() -> None:
    from workbench.io.label_maps import _verified_tiff_compression

    class _Tags:
        def __init__(self, value: object) -> None:
            self.value = value

        def get(self, _tag: int) -> object:
            return self.value

    class _Image:
        def __init__(self, value: object) -> None:
            self.tag_v2 = _Tags(value)

    with pytest.raises(ValidationError, match="cannot be verified as lossless"):
        _verified_tiff_compression(_Image(7))
    with pytest.raises(ValidationError, match="explicitly identifiable"):
        _verified_tiff_compression(_Image(None))


def test_nifti_keeps_canonical_ras_affine_and_requires_explicit_4d_selection(
    tmp_path: Path,
) -> None:
    nib = pytest.importorskip("nibabel")
    path = tmp_path / "dynamic-labels.nii.gz"
    xyzt = np.stack(
        [
            np.zeros((3, 2, 2), dtype=np.int16),
            np.full((3, 2, 2), 4, dtype=np.int16),
        ],
        axis=3,
    )
    source_affine = np.diag((-0.7, -0.8, 2.5, 1.0))
    nib.save(nib.Nifti1Image(xyzt, source_affine), path)
    canonical = nib.as_closest_canonical(nib.load(path))
    reference = _reference_series((2, 2, 3), np.asarray(canonical.affine))

    with pytest.raises(NiftiVolumeSelectionRequiredError):
        import_label_map(path, reference_series=reference)

    layer = import_label_map(path, reference_series=reference, volume_index=1)

    np.testing.assert_array_equal(layer.array, 4)
    np.testing.assert_allclose(layer.geometry.affine_ras, canonical.affine)
    assert layer.geometry.convention.value == "RAS+"
    assert layer.source.source_format is SourceFormat.NIFTI
    assert layer.source.compression is CompressionKind.LOSSLESS
    assert layer.source.provenance["selected_volume_index"] == 1
    assert layer.reference_report(reference).status is GeometryMatchStatus.MATCHED


def test_nifti_geometry_mismatch_is_returned_hidden_without_resampling(
    tmp_path: Path,
) -> None:
    nib = pytest.importorskip("nibabel")
    path = tmp_path / "offset-labels.nii"
    affine = np.eye(4, dtype=np.float64)
    affine[:3, 3] = (12.0, -4.0, 7.0)
    nib.save(nib.Nifti1Image(np.ones((3, 2, 2), dtype=np.uint8), affine), path)
    reference = _reference_series((2, 2, 3), np.eye(4))

    layer = import_label_map(path, reference_series=reference)
    report = layer.reference_report(reference)

    assert report.status is GeometryMatchStatus.REQUIRES_RESAMPLING
    assert report.overlay_allowed is False
    assert layer.presentation.visible is False
    assert layer.validation_state is LayerValidationState.PENDING
    assert layer.transform_chain == ()
    np.testing.assert_allclose(layer.geometry.affine_ras, affine)
    assert layer.provenance["automatic_resampling"] is False


def test_nifti_float_storage_is_rejected_even_when_values_look_integral(
    tmp_path: Path,
) -> None:
    nib = pytest.importorskip("nibabel")
    path = tmp_path / "floating-labels.nii"
    nib.save(nib.Nifti1Image(np.zeros((3, 2, 2), dtype=np.float32), np.eye(4)), path)

    with pytest.raises(ValidationError, match="integer dtype"):
        import_label_map(path, reference_series=_reference_series((2, 2, 3)))


def test_limits_and_format_specific_options_are_enforced(tmp_path: Path) -> None:
    path = tmp_path / "labels.png"
    Image.fromarray(np.zeros((8, 8), dtype=np.uint8)).save(path)
    reference = _reference_series((1, 8, 8))

    with pytest.raises(ResourceLimitError, match="encoded-size"):
        import_label_map(
            path,
            reference_series=reference,
            limits=LabelMapLimits(max_file_bytes=8),
        )
    with pytest.raises(ResourceLimitError, match="voxel limit"):
        import_label_map(
            path,
            reference_series=reference,
            limits=LabelMapLimits(max_voxels=16),
        )
    with pytest.raises(ValidationError, match="only be used with a 4-D NIfTI"):
        import_label_map(path, reference_series=reference, volume_index=0)


def test_reference_series_is_mandatory_and_type_checked(tmp_path: Path) -> None:
    path = tmp_path / "labels.png"
    Image.fromarray(np.zeros((2, 2), dtype=np.uint8)).save(path)

    with pytest.raises(TypeError):
        import_label_map(path)  # type: ignore[call-arg]
    with pytest.raises(ValidationError, match="reference_series"):
        import_label_map(path, reference_series=object())  # type: ignore[arg-type]
