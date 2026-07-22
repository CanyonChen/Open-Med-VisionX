from __future__ import annotations

import unittest
import zipfile
from inspect import signature
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np

try:
    import pydicom
    from pydicom.dataset import FileDataset, FileMetaDataset
    from pydicom.uid import CTImageStorage, ExplicitVRLittleEndian, generate_uid
except ImportError:  # pragma: no cover - skip in minimal environment
    pydicom = None

from workbench.domain import IntensitySemantics
from workbench.errors import DecodeError, ResourceLimitError
from workbench.io import DicomLoader, LoadLimits


@unittest.skipIf(pydicom is None, "pydicom is not installed")
class DicomLoaderTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = TemporaryDirectory()
        self.directory = Path(self.temporary_directory.name)
        self.series_uid = generate_uid()
        self.frame_of_reference_uid = generate_uid()

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def _write_slice(
        self,
        name: str,
        position_x: float,
        stored_value: int,
        *,
        orientation: tuple[float, ...] = (0, 1, 0, 0, 0, 1),
        frame_of_reference_uid: str | None = None,
        photometric: str = "MONOCHROME2",
        modality: str = "CT",
        series_uid: str | None = None,
        intensity_tags: dict[str, object] | None = None,
        include_rescale: bool = True,
    ) -> Path:
        meta = FileMetaDataset()
        meta.MediaStorageSOPClassUID = CTImageStorage
        meta.MediaStorageSOPInstanceUID = generate_uid()
        meta.TransferSyntaxUID = ExplicitVRLittleEndian
        meta.ImplementationClassUID = generate_uid()
        path = self.directory / name
        dataset = FileDataset(str(path), {}, file_meta=meta, preamble=b"\0" * 128)
        dataset.SOPClassUID = CTImageStorage
        dataset.SOPInstanceUID = meta.MediaStorageSOPInstanceUID
        dataset.SeriesInstanceUID = series_uid or self.series_uid
        dataset.FrameOfReferenceUID = frame_of_reference_uid or self.frame_of_reference_uid
        dataset.Modality = modality
        dataset.PatientName = "runtime-only"
        dataset.PatientID = "runtime-only"
        dataset.Rows = 2
        dataset.Columns = 3
        dataset.SamplesPerPixel = 1
        dataset.PhotometricInterpretation = photometric
        dataset.BitsAllocated = 16
        dataset.BitsStored = 16
        dataset.HighBit = 15
        dataset.PixelRepresentation = 1
        # Oblique test: first IOP vector advances columns along +A, second
        # advances rows along +S, so the slice normal is +L.
        dataset.ImageOrientationPatient = list(orientation)
        dataset.ImagePositionPatient = [position_x, 10, 20]
        dataset.PixelSpacing = [0.75, 0.5]
        dataset.SliceThickness = 2.5
        if include_rescale:
            dataset.RescaleSlope = 2
            dataset.RescaleIntercept = -1024
        for keyword, value in (intensity_tags or {}).items():
            setattr(dataset, keyword, value)
        dataset.InstanceNumber = int(position_x / 2.5) + 1
        dataset.PixelData = np.full((2, 3), stored_value, dtype=np.int16).tobytes()
        save_options = (
            {"enforce_file_format": True}
            if "enforce_file_format" in signature(dataset.save_as).parameters
            else {"write_like_original": False}
        )
        dataset.save_as(path, **save_options)
        return path

    def test_hu_sorting_spacing_and_lps_to_ras(self) -> None:
        self._write_slice("slice-c.dcm", 5.0, 3)
        self._write_slice("slice-a.dcm", 0.0, 1)
        self._write_slice("slice-b.dcm", 2.5, 2)
        volume = DicomLoader().load(self.directory)
        self.assertEqual(volume.intensity_semantics, IntensitySemantics.HOUNSFIELD_UNIT)
        np.testing.assert_allclose(volume.array[:, 0, 0], [-1022, -1020, -1018])
        np.testing.assert_allclose(volume.spacing, (0.5, 0.75, 2.5))
        # LPS (0, 10, 20) becomes RAS (0, -10, 20).
        np.testing.assert_allclose(volume.origin, (0.0, -10.0, 20.0))
        self.assertNotIn("PatientName", volume.runtime_metadata)
        self.assertFalse(volume.runtime_metadata["display_inverted"])

    def test_non_ct_dicom_is_not_silently_called_quantitative(self) -> None:
        self._write_slice("mr.dcm", 0.0, 20, modality="MR")

        volume = DicomLoader().load(self.directory)

        self.assertEqual(volume.modality, "MR")
        self.assertEqual(volume.intensity_semantics, IntensitySemantics.ARBITRARY_SIGNAL)
        self.assertFalse(volume.has_explicit_quantitative_semantics)

    def test_ct_without_modality_lut_is_not_assumed_to_be_hu(self) -> None:
        self._write_slice("ct-unknown.dcm", 0.0, 20, include_rescale=False)

        volume = DicomLoader().load(self.directory)

        self.assertEqual(volume.intensity_semantics, IntensitySemantics.UNKNOWN)
        self.assertEqual(
            volume.runtime_metadata["intensity_semantics_source"],
            "dicom_ct_modality_lut_missing",
        )

    def test_ct_with_explicit_non_hu_rescale_type_is_unknown(self) -> None:
        self._write_slice(
            "ct-unspecified.dcm",
            0.0,
            20,
            intensity_tags={"RescaleType": "US"},
        )

        volume = DicomLoader().load(self.directory)

        self.assertEqual(volume.intensity_semantics, IntensitySemantics.UNKNOWN)
        self.assertEqual(
            volume.runtime_metadata["intensity_semantics_source"],
            "dicom_ct_non_hu_rescale_type",
        )

    def test_pet_without_complete_unit_and_correction_evidence_is_unknown(self) -> None:
        self._write_slice(
            "pet-unknown.dcm",
            0.0,
            20,
            modality="PT",
            intensity_tags={"Units": "GML", "RescaleSlope": 1, "RescaleIntercept": 0},
        )
        incomplete = DicomLoader().load(self.directory)
        self.assertEqual(incomplete.intensity_semantics, IntensitySemantics.UNKNOWN)

    def test_pet_with_complete_unit_and_correction_evidence_is_suv(self) -> None:
        self._write_slice(
            "pet-suv.dcm",
            0.0,
            20,
            modality="PT",
            intensity_tags={
                "Units": "GML",
                "SUVType": "BW",
                "DecayCorrection": "START",
                "CorrectedImage": ["ATTN", "DECY"],
                "RescaleSlope": 1,
                "RescaleIntercept": 0,
            },
        )
        complete = DicomLoader().load(self.directory)
        self.assertEqual(complete.intensity_semantics, IntensitySemantics.SUV)

    def test_pet_gml_without_suv_type_uses_the_dicom_bw_default(self) -> None:
        self._write_slice(
            "pet-suv-default-bw.dcm",
            0.0,
            20,
            modality="PT",
            intensity_tags={
                "Units": "GML",
                "DecayCorrection": "START",
                "CorrectedImage": ["ATTN", "DECY"],
                "RescaleSlope": 1,
                "RescaleIntercept": 0,
            },
        )

        volume = DicomLoader().load(self.directory)

        self.assertEqual(volume.intensity_semantics, IntensitySemantics.SUV)

    def test_pet_invalid_suv_type_or_missing_modality_lut_stays_unknown(self) -> None:
        self._write_slice(
            "pet-invalid-type.dcm",
            0.0,
            20,
            modality="PT",
            intensity_tags={
                "Units": "GML",
                "SUVType": "INVALID",
                "DecayCorrection": "START",
                "CorrectedImage": ["ATTN", "DECY"],
                "RescaleSlope": 1,
                "RescaleIntercept": 0,
            },
        )
        invalid_type = DicomLoader().load(self.directory)
        self.assertEqual(invalid_type.intensity_semantics, IntensitySemantics.UNKNOWN)

        (self.directory / "pet-invalid-type.dcm").unlink()
        self._write_slice(
            "pet-no-lut.dcm",
            0.0,
            20,
            modality="PT",
            include_rescale=False,
            intensity_tags={
                "Units": "GML",
                "DecayCorrection": "START",
                "CorrectedImage": ["ATTN", "DECY"],
            },
        )
        missing_lut = DicomLoader().load(self.directory)
        self.assertEqual(missing_lut.intensity_semantics, IntensitySemantics.UNKNOWN)

    def test_rescale_overflow_is_rejected_instead_of_creating_nonfinite_volume(self) -> None:
        self._write_slice(
            "overflow.dcm",
            0.0,
            20,
            intensity_tags={"RescaleSlope": 1e308, "RescaleIntercept": 0},
        )

        with self.assertRaisesRegex(DecodeError, "non-finite"):
            DicomLoader().load(self.directory)

    def test_single_file_byte_limit_is_enforced_before_decode(self) -> None:
        path = self._write_slice("large.dcm", 0.0, 1)
        limits = LoadLimits(max_zip_member_bytes=path.stat().st_size - 1)
        with self.assertRaises(ResourceLimitError):
            DicomLoader().load(path, limits=limits)

    def test_directory_byte_limit_stops_enumeration_without_materializing_paths(self) -> None:
        first = self._write_slice("first.dcm", 0.0, 1)
        file_size = first.stat().st_size

        def guarded_rglob(path: Path, pattern: str):
            self.assertEqual(path, self.directory)
            self.assertEqual(pattern, "*")
            yield first
            raise AssertionError("directory enumeration advanced past the byte limit")

        limits = LoadLimits(
            max_zip_member_bytes=file_size + 1,
            max_zip_total_bytes=file_size - 1,
        )
        with (
            patch.object(Path, "rglob", guarded_rglob),
            self.assertRaisesRegex(ResourceLimitError, "byte limits"),
        ):
            DicomLoader()._read_directory(self.directory, pydicom, limits, None)

    def test_directory_member_limit_stops_enumeration_at_first_excess_file(self) -> None:
        first = self._write_slice("first.dcm", 0.0, 1)
        second = self._write_slice("second.dcm", 2.5, 2)

        def guarded_rglob(path: Path, pattern: str):
            self.assertEqual(path, self.directory)
            self.assertEqual(pattern, "*")
            yield first
            yield second
            raise AssertionError("directory enumeration advanced past the member limit")

        limits = LoadLimits(max_zip_members=1)
        with (
            patch.object(Path, "rglob", guarded_rglob),
            self.assertRaisesRegex(ResourceLimitError, "more files"),
        ):
            DicomLoader()._read_directory(self.directory, pydicom, limits, None)

    def test_inconsistent_slice_orientation_is_rejected(self) -> None:
        self._write_slice("first.dcm", 0.0, 1)
        self._write_slice(
            "second.dcm",
            2.5,
            2,
            orientation=(1, 0, 0, 0, 0, 1),
        )
        with self.assertRaisesRegex(DecodeError, "inconsistent ImageOrientationPatient"):
            DicomLoader().load(self.directory)

    def test_multiple_series_are_rejected_instead_of_silently_guessing(self) -> None:
        self._write_slice("series-a.dcm", 0.0, 1)
        self._write_slice(
            "series-b.dcm",
            2.5,
            2,
            series_uid=generate_uid(),
        )

        with self.assertRaisesRegex(
            DecodeError,
            "no series was selected.*will not guess",
        ):
            DicomLoader().load(self.directory)

    def test_inconsistent_frame_of_reference_is_rejected(self) -> None:
        self._write_slice("first.dcm", 0.0, 1)
        self._write_slice(
            "second.dcm",
            2.5,
            2,
            frame_of_reference_uid=generate_uid(),
        )
        with self.assertRaisesRegex(DecodeError, "FrameOfReferenceUID"):
            DicomLoader().load(self.directory)

    def test_monochrome1_sets_display_inversion_and_mixed_modes_are_rejected(self) -> None:
        self._write_slice("mono1.dcm", 0.0, 1, photometric="MONOCHROME1")
        volume = DicomLoader().load(self.directory)
        self.assertTrue(volume.runtime_metadata["display_inverted"])
        self.assertEqual(
            volume.runtime_metadata["photometric_interpretations"],
            ("MONOCHROME1",),
        )

        self._write_slice("mono2.dcm", 2.5, 2, photometric="MONOCHROME2")
        with self.assertRaisesRegex(DecodeError, "photometric interpretations"):
            DicomLoader().load(self.directory)

    def test_zip_path_traversal_is_rejected_without_extraction(self) -> None:
        valid = self._write_slice("valid.dcm", 0.0, 1)
        archive_path = self.directory / "unsafe.zip"
        with zipfile.ZipFile(archive_path, "w") as archive:
            archive.write(valid, "../escape.dcm")
        with self.assertRaises(ResourceLimitError):
            DicomLoader().load(archive_path)
        self.assertFalse((self.directory.parent / "escape.dcm").exists())


if __name__ == "__main__":
    unittest.main()
