from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

try:
    import nibabel as nib
except ImportError:  # pragma: no cover - skip in minimal environment
    nib = None

from dicom_viewer.errors import ResourceLimitError
from dicom_viewer.io import LoadLimits, NiftiLoader


@unittest.skipIf(nib is None, "nibabel is not installed")
class NiftiLoaderTests(unittest.TestCase):
    def test_runtime_generated_nifti_is_canonical_ras(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "volume.nii.gz"
            xyz = np.arange(24, dtype=np.int16).reshape(2, 3, 4)
            lps_affine = np.diag([-0.5, -0.75, 2.0, 1.0])
            nib.save(nib.Nifti1Image(xyz, lps_affine), path)
            volume = NiftiLoader().load(path)
            self.assertEqual(volume.shape, (4, 3, 2))
            self.assertGreater(volume.affine[0, 0], 0)
            self.assertGreater(volume.affine[1, 1], 0)
            np.testing.assert_allclose(volume.spacing, (0.5, 0.75, 2.0))

    def test_uncompressed_nifti_with_gzip_suffix_loads_by_signature(self) -> None:
        with TemporaryDirectory() as directory:
            uncompressed_path = Path(directory) / "volume.nii"
            mislabeled_path = Path(directory) / "volume.nii.gz"
            xyz = np.arange(24, dtype=np.int16).reshape(2, 3, 4)
            nib.save(nib.Nifti1Image(xyz, np.eye(4)), uncompressed_path)
            uncompressed_path.rename(mislabeled_path)

            volume = NiftiLoader().load(mislabeled_path)

            self.assertEqual(volume.shape, (4, 3, 2))
            self.assertEqual(volume.runtime_metadata["storage_compression"], "none")
            np.testing.assert_array_equal(volume.array, np.transpose(xyz, (2, 1, 0)))

    def test_scaled_data_uses_actual_decoded_byte_limit(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "scaled.nii"
            image = nib.Nifti1Image(
                np.arange(24, dtype=np.int16).reshape(2, 3, 4),
                np.eye(4),
            )
            image.header.set_slope_inter(2.0, 10.0)
            nib.save(image, path)
            with self.assertRaisesRegex(ResourceLimitError, "decoded bytes"):
                NiftiLoader().load(path, limits=LoadLimits(max_decoded_bytes=100))


if __name__ == "__main__":
    unittest.main()
