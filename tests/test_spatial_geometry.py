from __future__ import annotations

from inspect import signature
from pathlib import Path

import numpy as np
import pytest

from dicom_viewer.domain import (
    ImageVolume,
    IntensitySemantics,
    SourceType,
    resample_to_ras_grid,
)
from dicom_viewer.io import DicomLoader, NiftiLoader

_CYCLIC_DIRECTION_RAS = np.asarray(
    [
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ],
    dtype=np.float64,
)
_ORIGIN_RAS = np.asarray((5.0, -7.0, 11.0), dtype=np.float64)


def _source_voxels() -> np.ndarray:
    z, y, x = np.indices((3, 4, 5))
    return (100 * z + 10 * y + x).astype(np.int16)


def _source_affine() -> np.ndarray:
    affine = np.eye(4, dtype=np.float64)
    affine[:3, :3] = _CYCLIC_DIRECTION_RAS
    affine[:3, 3] = _ORIGIN_RAS
    return affine


def _expected_axis_aligned_voxels() -> np.ndarray:
    # The source x/y/z axes point along RAS y/z/x respectively.  After
    # resampling to an axis-aligned RAS grid, output (z, y, x) therefore maps
    # to source (y, x, z).
    output_z, output_y, output_x = np.indices((4, 5, 3))
    return (100 * output_x + 10 * output_z + output_y).astype(np.int16)


def test_direction_matrix_controls_coronal_and_sagittal_planes() -> None:
    volume = ImageVolume(
        array=_source_voxels(),
        source_type=SourceType.NIFTI,
        intensity_semantics=IntensitySemantics.QUANTITATIVE,
        affine=_source_affine(),
        spacing=(1.0, 1.0, 1.0),
        origin=tuple(_ORIGIN_RAS),
        direction=_CYCLIC_DIRECTION_RAS,
        modality="SYNTHETIC",
    )

    ras_volume = resample_to_ras_grid(
        volume,
        target_spacing=(1.0, 1.0, 1.0),
        interpolation_order=0,
    )
    expected = _expected_axis_aligned_voxels()

    np.testing.assert_array_equal(ras_volume.array, expected)
    np.testing.assert_array_equal(ras_volume.direction, np.eye(3))
    np.testing.assert_allclose(ras_volume.origin, _ORIGIN_RAS)
    # These explicit plane checks guard against applying the direction matrix
    # in xyz order directly to the zyx storage array.
    np.testing.assert_array_equal(ras_volume.coronal(2), expected[:, 2, :])
    np.testing.assert_array_equal(ras_volume.sagittal(1), expected[:, :, 1])


def _write_dicom_series(directory: Path, voxels_zyx: np.ndarray) -> None:
    pytest.importorskip("pydicom")
    from pydicom.dataset import FileDataset, FileMetaDataset
    from pydicom.uid import CTImageStorage, ExplicitVRLittleEndian, generate_uid

    series_uid = generate_uid()
    study_uid = generate_uid()
    frame_uid = generate_uid()
    # Convert the RAS origin and source x/y directions to DICOM LPS.  The
    # source x/y/z axes become LPS -A/+S/-L, respectively.
    origin_lps = (-_ORIGIN_RAS[0], -_ORIGIN_RAS[1], _ORIGIN_RAS[2])
    direction_x_lps = np.asarray((0.0, -1.0, 0.0))
    direction_y_lps = np.asarray((0.0, 0.0, 1.0))
    direction_z_lps = np.cross(direction_x_lps, direction_y_lps)

    for z_index, pixels in enumerate(voxels_zyx):
        path = directory / f"slice-{z_index}.dcm"
        meta = FileMetaDataset()
        meta.MediaStorageSOPClassUID = CTImageStorage
        meta.MediaStorageSOPInstanceUID = generate_uid()
        meta.TransferSyntaxUID = ExplicitVRLittleEndian
        meta.ImplementationClassUID = generate_uid()
        dataset = FileDataset(str(path), {}, file_meta=meta, preamble=b"\0" * 128)
        dataset.SOPClassUID = CTImageStorage
        dataset.SOPInstanceUID = meta.MediaStorageSOPInstanceUID
        dataset.SeriesInstanceUID = series_uid
        dataset.StudyInstanceUID = study_uid
        dataset.FrameOfReferenceUID = frame_uid
        dataset.Modality = "CT"
        dataset.Rows, dataset.Columns = pixels.shape
        dataset.SamplesPerPixel = 1
        dataset.PhotometricInterpretation = "MONOCHROME2"
        dataset.BitsAllocated = 16
        dataset.BitsStored = 16
        dataset.HighBit = 15
        dataset.PixelRepresentation = 1
        dataset.ImageOrientationPatient = [*direction_x_lps, *direction_y_lps]
        dataset.ImagePositionPatient = (np.asarray(origin_lps) + z_index * direction_z_lps).tolist()
        dataset.PixelSpacing = [1.0, 1.0]
        dataset.SliceThickness = 1.0
        dataset.SpacingBetweenSlices = 1.0
        dataset.RescaleSlope = 1.0
        dataset.RescaleIntercept = 0.0
        dataset.InstanceNumber = z_index + 1
        dataset.PixelData = np.ascontiguousarray(pixels).tobytes()
        save_options = (
            {"enforce_file_format": True}
            if "enforce_file_format" in signature(dataset.save_as).parameters
            else {"write_like_original": False}
        )
        dataset.save_as(path, **save_options)


def _write_nifti(path: Path, voxels_zyx: np.ndarray) -> None:
    nibabel = pytest.importorskip("nibabel")
    voxels_xyz = np.transpose(voxels_zyx, (2, 1, 0))
    nibabel.save(nibabel.Nifti1Image(voxels_xyz, _source_affine()), path)


def test_runtime_dicom_and_nifti_have_identical_ras_triplanar_views(tmp_path: Path) -> None:
    voxels = _source_voxels()
    dicom_directory = tmp_path / "dicom"
    dicom_directory.mkdir()
    nifti_path = tmp_path / "synthetic.nii.gz"
    _write_dicom_series(dicom_directory, voxels)
    _write_nifti(nifti_path, voxels)

    dicom_source = DicomLoader().load(dicom_directory)
    nifti_source = NiftiLoader().load(nifti_path)
    np.testing.assert_allclose(dicom_source.direction, _CYCLIC_DIRECTION_RAS)

    dicom_ras = resample_to_ras_grid(
        dicom_source,
        target_spacing=(1.0, 1.0, 1.0),
        interpolation_order=0,
    )
    nifti_ras = resample_to_ras_grid(
        nifti_source,
        target_spacing=(1.0, 1.0, 1.0),
        interpolation_order=0,
    )

    np.testing.assert_allclose(dicom_ras.affine, nifti_ras.affine)
    np.testing.assert_array_equal(dicom_ras.array, nifti_ras.array)
    for index in range(dicom_ras.shape[0]):
        np.testing.assert_array_equal(dicom_ras.axial(index), nifti_ras.axial(index))
    for index in range(dicom_ras.shape[1]):
        np.testing.assert_array_equal(dicom_ras.coronal(index), nifti_ras.coronal(index))
    for index in range(dicom_ras.shape[2]):
        np.testing.assert_array_equal(dicom_ras.sagittal(index), nifti_ras.sagittal(index))

    np.testing.assert_array_equal(dicom_ras.array, _expected_axis_aligned_voxels())
