from __future__ import annotations

import unittest
from inspect import signature
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

try:
    import pydicom
    from pydicom.dataset import Dataset, FileDataset, FileMetaDataset
    from pydicom.sequence import Sequence
    from pydicom.uid import (
        EnhancedCTImageStorage,
        EnhancedMRImageStorage,
        ExplicitVRLittleEndian,
        generate_uid,
    )
except ImportError:  # pragma: no cover - skip in minimal environment
    pydicom = None

from workbench.domain import IntensitySemantics
from workbench.errors import DecodeError
from workbench.io import DicomLoader


@unittest.skipIf(pydicom is None, "pydicom is not installed")
class EnhancedDicomLoaderTests(unittest.TestCase):
    def _write_multiframe(
        self,
        path: Path,
        *,
        modality: str = "CT",
        positions: tuple[float, ...] = (5.0, 0.0, 2.5),
    ) -> Path:
        storage_uid = EnhancedCTImageStorage if modality == "CT" else EnhancedMRImageStorage
        meta = FileMetaDataset()
        meta.MediaStorageSOPClassUID = storage_uid
        meta.MediaStorageSOPInstanceUID = generate_uid()
        meta.TransferSyntaxUID = ExplicitVRLittleEndian
        meta.ImplementationClassUID = generate_uid()
        dataset = FileDataset(str(path), {}, file_meta=meta, preamble=b"\0" * 128)
        dataset.SOPClassUID = storage_uid
        dataset.SOPInstanceUID = meta.MediaStorageSOPInstanceUID
        dataset.StudyInstanceUID = generate_uid()
        dataset.SeriesInstanceUID = generate_uid()
        dataset.FrameOfReferenceUID = generate_uid()
        dataset.Modality = modality
        dataset.SeriesNumber = 1
        dataset.SeriesDescription = f"ENHANCED {modality}"
        dataset.NumberOfFrames = len(positions)
        dataset.Rows = 2
        dataset.Columns = 3
        dataset.SamplesPerPixel = 1
        dataset.PhotometricInterpretation = "MONOCHROME2"
        dataset.BitsAllocated = 16
        dataset.BitsStored = 16
        dataset.HighBit = 15
        dataset.PixelRepresentation = 1

        shared = Dataset()
        orientation = Dataset()
        orientation.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]
        shared.PlaneOrientationSequence = Sequence([orientation])
        measures = Dataset()
        measures.PixelSpacing = [0.75, 0.5]
        measures.SliceThickness = 2.5
        measures.SpacingBetweenSlices = 2.5
        shared.PixelMeasuresSequence = Sequence([measures])
        transform = Dataset()
        transform.RescaleSlope = 2
        transform.RescaleIntercept = -1024
        transform.RescaleType = "HU" if modality == "CT" else "US"
        shared.PixelValueTransformationSequence = Sequence([transform])
        dataset.SharedFunctionalGroupsSequence = Sequence([shared])

        per_frame = []
        for z_position in positions:
            group = Dataset()
            plane_position = Dataset()
            plane_position.ImagePositionPatient = [0, 0, z_position]
            group.PlanePositionSequence = Sequence([plane_position])
            per_frame.append(group)
        dataset.PerFrameFunctionalGroupsSequence = Sequence(per_frame)

        stored_by_position = {
            position: index + 1 for index, position in enumerate(sorted(positions))
        }
        dataset.PixelData = np.stack(
            [
                np.full((2, 3), stored_by_position[position], dtype=np.int16)
                for position in positions
            ],
            axis=0,
        ).tobytes()
        save_options = (
            {"enforce_file_format": True}
            if "enforce_file_format" in signature(dataset.save_as).parameters
            else {"write_like_original": False}
        )
        dataset.save_as(path, **save_options)
        return path

    def test_enhanced_ct_discovery_and_load_use_functional_group_geometry(self) -> None:
        with TemporaryDirectory() as directory:
            path = self._write_multiframe(Path(directory) / "enhanced-ct.dcm")

            summary = DicomLoader().discover_series(path).series[0]
            volume = DicomLoader().load(path)

            self.assertTrue(summary.supported_by_stable_loader)
            self.assertEqual(summary.frame_count, 3)
            self.assertEqual(summary.slice_count, 3)
            np.testing.assert_allclose(volume.array[:, 0, 0], [-1022, -1020, -1018])
            np.testing.assert_allclose(volume.spacing, (0.5, 0.75, 2.5))
            np.testing.assert_allclose(volume.origin, (0.0, 0.0, 0.0))
            self.assertEqual(volume.intensity_semantics, IntensitySemantics.HOUNSFIELD_UNIT)
            self.assertTrue(volume.runtime_metadata["enhanced_multiframe"])
            self.assertEqual(volume.runtime_metadata["format"], "DICOM_ENHANCED_MULTIFRAME")

    def test_enhanced_mr_preserves_arbitrary_signal_semantics(self) -> None:
        with TemporaryDirectory() as directory:
            path = self._write_multiframe(
                Path(directory) / "enhanced-mr.dcm",
                modality="MR",
            )

            volume = DicomLoader().load(path)

            self.assertEqual(volume.modality, "MR")
            self.assertEqual(volume.intensity_semantics, IntensitySemantics.ARBITRARY_SIGNAL)

    def test_nonuniform_enhanced_frame_positions_are_rejected(self) -> None:
        with TemporaryDirectory() as directory:
            path = self._write_multiframe(
                Path(directory) / "nonuniform.dcm",
                positions=(0.0, 2.5, 6.0),
            )

            with self.assertRaisesRegex(DecodeError, "non-uniform"):
                DicomLoader().load(path)


if __name__ == "__main__":
    unittest.main()
