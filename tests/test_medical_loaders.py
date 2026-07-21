from __future__ import annotations

import importlib.util
import unittest
import zipfile
from inspect import signature
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from dicom_viewer.domain import IntensitySemantics
from dicom_viewer.errors import ResourceLimitError
from dicom_viewer.io import DicomLoader, NiftiLoader

HAS_PYDICOM = importlib.util.find_spec("pydicom") is not None
HAS_NIBABEL = importlib.util.find_spec("nibabel") is not None


@unittest.skipUnless(HAS_PYDICOM, "pydicom is an optional test dependency")
class DicomLoaderTests(unittest.TestCase):
    @staticmethod
    def write_slice(path: Path, *, instance: int, position: tuple[float, float, float]) -> None:
        from pydicom.dataset import FileDataset, FileMetaDataset
        from pydicom.uid import CTImageStorage, ExplicitVRLittleEndian, generate_uid

        file_meta = FileMetaDataset()
        file_meta.MediaStorageSOPClassUID = CTImageStorage
        file_meta.MediaStorageSOPInstanceUID = generate_uid()
        file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
        dataset = FileDataset(str(path), {}, file_meta=file_meta, preamble=b"\0" * 128)
        dataset.SOPClassUID = CTImageStorage
        dataset.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
        dataset.SeriesInstanceUID = "1.2.826.0.1.3680043.10.543.1"
        dataset.StudyInstanceUID = "1.2.826.0.1.3680043.10.543.2"
        dataset.Modality = "CT"
        dataset.PatientName = "Runtime^Only"
        dataset.PatientID = "not-exported"
        dataset.Rows = 3
        dataset.Columns = 4
        dataset.SamplesPerPixel = 1
        dataset.PhotometricInterpretation = "MONOCHROME2"
        dataset.BitsAllocated = 16
        dataset.BitsStored = 16
        dataset.HighBit = 15
        dataset.PixelRepresentation = 1
        dataset.ImageOrientationPatient = [0, 1, 0, 0, 0, 1]
        dataset.ImagePositionPatient = list(position)
        dataset.PixelSpacing = [0.75, 0.5]
        dataset.SliceThickness = 2.5
        dataset.InstanceNumber = instance
        dataset.RescaleSlope = 2
        dataset.RescaleIntercept = -1024
        pixels = np.full((3, 4), instance, dtype=np.int16)
        dataset.PixelData = pixels.tobytes()
        save_options = (
            {"enforce_file_format": True}
            if "enforce_file_format" in signature(dataset.save_as).parameters
            else {"write_like_original": False}
        )
        dataset.save_as(path, **save_options)

    def test_folder_load_applies_hu_and_oblique_normal_sorting(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            self.write_slice(root / "third.bin", instance=3, position=(5.0, 0.0, 0.0))
            self.write_slice(root / "first.bin", instance=1, position=(0.0, 0.0, 0.0))
            self.write_slice(root / "second.bin", instance=2, position=(2.5, 0.0, 0.0))
            volume = DicomLoader().load(root)
        self.assertEqual(volume.shape, (3, 3, 4))
        self.assertEqual(volume.intensity_semantics, IntensitySemantics.HOUNSFIELD_UNIT)
        np.testing.assert_array_equal(volume.array[:, 0, 0], [-1022, -1020, -1018])
        np.testing.assert_allclose(volume.spacing, (0.5, 0.75, 2.5))
        self.assertNotIn("PatientName", volume.runtime_metadata)

    def test_unsafe_zip_path_is_refused_without_extraction(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "unsafe.zip"
            with zipfile.ZipFile(path, "w") as archive:
                archive.writestr("../escape.dcm", b"not dicom")
            with self.assertRaises(ResourceLimitError):
                DicomLoader().load(path)
            self.assertFalse((Path(directory).parent / "escape.dcm").exists())


@unittest.skipUnless(HAS_NIBABEL, "nibabel is an optional test dependency")
class NiftiLoaderTests(unittest.TestCase):
    def test_nifti_is_reoriented_to_ras_and_stored_zyx(self) -> None:
        import nibabel as nib

        with TemporaryDirectory() as directory:
            path = Path(directory) / "runtime.nii.gz"
            xyz = np.arange(2 * 3 * 4, dtype=np.int16).reshape(2, 3, 4)
            affine = np.diag([-2.0, 3.0, 4.0, 1.0])
            nib.save(nib.Nifti1Image(xyz, affine), path)
            volume = NiftiLoader().load(path)
        self.assertEqual(volume.shape, (4, 3, 2))
        self.assertEqual(volume.runtime_metadata["canonical_orientation"], "RAS+")
        self.assertTrue(np.all(np.diag(volume.affine)[:3] > 0))


if __name__ == "__main__":
    unittest.main()
