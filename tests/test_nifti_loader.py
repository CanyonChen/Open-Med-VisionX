from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

try:
    import nibabel as nib
except ImportError:  # pragma: no cover - skip in minimal environment
    nib = None

from workbench.domain import IntensitySemantics
from workbench.errors import OperationCancelled, ResourceLimitError
from workbench.io import LoadLimits, NiftiLoader, NiftiVolumeSelectionRequiredError


@unittest.skipIf(nib is None, "nibabel is not installed")
class NiftiLoaderTests(unittest.TestCase):
    def test_4d_nifti_requires_explicit_volume_selection(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "dynamic.nii.gz"
            xyzt = np.stack(
                [
                    np.full((2, 3, 4), 11, dtype=np.int16),
                    np.full((2, 3, 4), 29, dtype=np.int16),
                ],
                axis=3,
            )
            nib.save(nib.Nifti1Image(xyzt, np.eye(4)), path)

            with self.assertRaises(NiftiVolumeSelectionRequiredError) as context:
                NiftiLoader().load(path)

            self.assertEqual(context.exception.shape, (2, 3, 4, 2))
            self.assertEqual(context.exception.volume_count, 2)

    def test_4d_nifti_loads_only_the_explicit_selection(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "dynamic.nii"
            xyzt = np.stack(
                [
                    np.full((2, 3, 4), 11, dtype=np.int16),
                    np.full((2, 3, 4), 29, dtype=np.int16),
                ],
                axis=3,
            )
            nib.save(nib.Nifti1Image(xyzt, np.eye(4)), path)

            volume = NiftiLoader().load(path, volume_index=1)

            self.assertEqual(volume.shape, (4, 3, 2))
            np.testing.assert_array_equal(volume.array, 29)
            self.assertEqual(volume.runtime_metadata["source_shape"], (2, 3, 4, 2))
            self.assertEqual(volume.runtime_metadata["selected_volume_index"], 1)
            self.assertEqual(
                volume.runtime_metadata["volume_selection"],
                "explicit-user-selection",
            )

    def test_volume_index_is_rejected_for_3d_nifti(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "volume.nii"
            nib.save(nib.Nifti1Image(np.zeros((2, 3, 4)), np.eye(4)), path)

            with self.assertRaisesRegex(ValueError, "only with a 4-D"):
                NiftiLoader().load(path, volume_index=0)

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
            self.assertEqual(volume.modality, "UNKNOWN")
            self.assertEqual(volume.intensity_semantics, IntensitySemantics.UNKNOWN)
            self.assertFalse(volume.has_explicit_quantitative_semantics)

    def test_explicit_nifti_semantics_are_recorded_without_container_inference(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "probability.nii"
            nib.save(nib.Nifti1Image(np.zeros((3, 4, 5)), np.eye(4)), path)

            volume = NiftiLoader().load(
                path,
                modality="MR",
                intensity_semantics=IntensitySemantics.PROBABILITY,
            )

            self.assertEqual(volume.modality, "MR")
            self.assertEqual(volume.intensity_semantics, IntensitySemantics.PROBABILITY)
            self.assertEqual(
                volume.runtime_metadata["intensity_semantics_source"],
                "user_declared",
            )

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

    def test_cancellation_during_proxy_preflight_is_not_wrapped_as_decode_failure(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "volume.nii"
            nib.save(nib.Nifti1Image(np.zeros((3, 4, 5)), np.eye(4)), path)
            calls = 0

            def cancel_after_preflight() -> bool:
                nonlocal calls
                calls += 1
                return calls >= 2

            with self.assertRaises(OperationCancelled):
                NiftiLoader().load(path, cancel=cancel_after_preflight)


if __name__ == "__main__":
    unittest.main()
